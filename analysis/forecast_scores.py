"""Forecast accuracy and calibration of the applied models (Figure S2 C-D; Results, Dispersion and forecast error).

Per-forecast scores (absolute error, squared error, bias, Gaussian NLL and CRPS, interval coverage and width at several
levels, quantile calibration, and for SBP the probability that the next-hour mean falls below 90 mmHg) are averaged with
every forecast weighted equally; 95% CIs are from 300 patient-level bootstrap resamples.

Usage: python -m analysis.forecast_scores --forecasts <forecasts.parquet> --label A_application --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

IDS = ['site_group', 'source_system', 'hd_session_id', 'patient_token_v1', 'hour_index']
LEVELS = (.5, .8, .9, .95)
QUANTILES = (.025, .05, .1, .25, .5, .75, .9, .95, .975)
REPS = 300
SEED = 20260912
RELIABILITY_EDGES = np.array([0, .005, .01, .025, .05, .1, .2, .4, .6, 1.000000001])


def crps(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-.5 * z * z) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))


def verify_formula():
    """Closed-form Gaussian CRPS checked against numerical integration."""
    cases = []
    for y, mu, sig in [(0., 0., 1.), (0., .1, .4), (90., 105., 15.), (120., 90., 7.)]:
        integral = quad(lambda x: ndtr((x - mu) / sig) ** 2, -np.inf, y)[0]
        integral += quad(lambda x: (1 - ndtr((x - mu) / sig)) ** 2, y, np.inf)[0]
        analytic = float(crps(y, mu, sig))
        assert abs(integral - analytic) < 1e-7, (integral, analytic)
        cases.append(dict(y=y, mu=mu, sigma=sig, analytic=analytic, integral=integral))
    return cases


def scores(frame, bp):
    y, mu, sig = (frame[f'{x}_{bp}'].to_numpy(dtype=float) for x in ('y', 'mu', 'sigma'))
    assert np.isfinite(y).all() and np.isfinite(mu).all() and np.isfinite(sig).all() and (sig > 0).all()
    err = mu - y
    z = (y - mu) / sig
    data = {'mae': abs(err), 'rmse': err ** 2, 'bias': err,
            'gaussian_nll': .5 * np.log(2 * np.pi) + np.log(sig) + .5 * z * z,
            'gaussian_crps': crps(y, mu, sig), 'mean_sigma': sig}
    nominals = {}
    for level in LEVELS:
        label = str(round(level * 100))
        critical = 1.96 if level == .95 else ndtri((1 + level) / 2)
        lo, hi = mu - critical * sig, mu + critical * sig
        data[f'coverage_{label}'] = ((y >= lo) & (y <= hi)).astype(float)
        data[f'width_{label}'] = 2 * critical * sig
        data[f'lower_violation_{label}'] = (y < lo).astype(float)
        data[f'upper_violation_{label}'] = (y > hi).astype(float)
        nominals[f'coverage_{label}'] = level
        nominals[f'lower_violation_{label}'] = (1 - level) / 2
        nominals[f'upper_violation_{label}'] = (1 - level) / 2
    for p in QUANTILES:
        name = f'quantile_cdf_{p:g}'
        data[name] = (y <= mu + ndtri(p) * sig).astype(float)
        nominals[name] = p
    if bp == 'sbp':
        prob = ndtr((90 - mu) / sig)
        data['brier_sbp_lt90'] = (prob - (y < 90)) ** 2
        data['mean_probability_sbp_lt90'] = prob
        data['observed_sbp_lt90'] = (y < 90).astype(float)
        data['error_gt20'] = (abs(err) > 20).astype(float)
    return list(data), data, nominals


def reliability(frame, source, label):
    p = ndtr((90 - frame.mu_sbp.to_numpy(float)) / frame.sigma_sbp.to_numpy(float))
    y = frame.y_sbp.to_numpy(float) < 90
    bins = np.searchsorted(RELIABILITY_EDGES, p, side='right') - 1
    rows = []
    for j in range(len(RELIABILITY_EDGES) - 1):
        m = bins == j
        if m.any():
            rows.append(dict(label=label, analysis_group=source, bin_lower=RELIABILITY_EDGES[j], bin_upper=min(1., RELIABILITY_EDGES[j + 1]),
                             n=int(m.sum()), mean_predicted_probability=float(p[m].mean()), observed_fraction=float(y[m].mean())))
    return rows


def evaluate(path, label, out, reps=REPS, seed=SEED):
    use = IDS + [f'{x}_{bp}' for bp in ('sbp', 'dbp') for x in ('y', 'mu', 'sigma')]
    frame = pd.read_parquet(path)
    frame = frame[[c for c in use if c in frame]]
    for c in IDS[:-1]:
        frame[c] = frame[c].astype(str)
    assert not frame.duplicated(IDS).any()
    rows, reliability_rows = [], []
    for si, source in enumerate(sorted(frame.site_group.unique())):
        part = frame.loc[frame.site_group == source]
        key = part.site_group + '__' + part.patient_token_v1
        codes, patients = pd.factorize(key, sort=True)
        n_patients = len(patients)
        counts = np.bincount(codes, minlength=n_patients).astype(float)
        rng = np.random.default_rng(seed + si)
        weights = np.vstack([np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients) for _ in range(reps)]).astype(float)
        denoms = weights @ counts
        metadata = dict(label=label, analysis_group=source, n_rows=len(part), n_patients=n_patients)
        for bp in ('sbp', 'dbp'):
            if f'y_{bp}' not in part:
                continue
            names, data, nominal = scores(part, bp)
            sums = np.column_stack([np.bincount(codes, weights=data[n], minlength=n_patients) for n in names])
            values = sums.sum(axis=0) / counts.sum()
            boot = (weights @ sums) / denoms[:, None]
            idx = names.index('rmse')
            values[idx] = np.sqrt(values[idx])
            boot[:, idx] = np.sqrt(boot[:, idx])
            for j, name in enumerate(names):
                lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
                rows.append(dict(**metadata, bp=bp, metric=name, value=values[j], ci_low=lo, ci_high=hi, nominal=nominal.get(name, np.nan), reps=reps))
        reliability_rows += reliability(part, source, label)
        print(f'{label} {source}: {len(part):,} rows / {n_patients} patients', flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f'{label}_metrics.csv', index=False)
    pd.DataFrame(reliability_rows).to_csv(out / f'{label}_sbp90_reliability.csv', index=False)
    (out / f'{label}_settings.json').write_text(json.dumps(dict(levels=LEVELS, quantiles=QUANTILES, reps=reps, seed=seed, crps_checks=verify_formula()), indent=2), encoding='utf-8')
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forecasts', required=True)
    p.add_argument('--label', required=True, help='e.g. A_application')
    p.add_argument('--out', required=True)
    p.add_argument('--reps', type=int, default=REPS)
    args = p.parse_args()
    evaluate(args.forecasts, args.label, args.out, args.reps)


if __name__ == '__main__':
    main()
