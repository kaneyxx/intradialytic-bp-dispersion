"""Forecast error by dispersion: selective retention of forecasts by sigma (Figure S2 A-D, F).

Forecasts are ranked by sigma (and, where the preceding-hour SBP change is observed, by its absolute value) and the
error of the retained forecasts is computed at retention fractions from 50% to 100%; at 80% retention the retained and
flagged portions are compared (mean absolute error, 95% interval coverage, errors above 20 mmHg, next-hour SBP below
90 mmHg). Intervals are 95% patient-cluster bootstrap intervals with the selection thresholds recomputed in each
replicate (retention_stats.analyze).

Usage: python -m analysis.retention --forecasts <forecasts.parquet> --features <feature_table.parquet> --source Chiayi --seed 20260927 --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .landmarks import KEY, P, bucket
from .retention_stats import analyze

METRICS = ['mae', 'coverage95', 'lower_tail', 'upper_tail', 'width95', 'error_gt20', 'next_mean_lt90', 'mean_sigma']
QUOTAS = [i / 100 for i in range(50, 101, 5)]
REPS = 1000


def row_metrics(d):
    e = d.y_sbp.to_numpy(float) - d.mu_sbp.to_numpy(float)
    s = d.sigma_sbp.to_numpy(float)
    return np.column_stack([abs(e), abs(e) <= 1.96 * s, e < -1.96 * s, e > 1.96 * s, 3.92 * s, abs(e) > 20, d.y_sbp.to_numpy(float) < 90, s])


def weights_at(score, quota):
    cut = np.sort(score)[min(len(score) - 1, max(0, int(np.ceil(quota * len(score))) - 1))]
    lo, eq = score < cut, score == cut
    weight = lo.astype(float)
    weight[eq] = (quota * len(score) - lo.sum()) / eq.sum()
    return weight


def protected_rates(d, scores):
    """Fail closed rather than report a rate supported by fewer than ten patients."""
    y = d.y_sbp.to_numpy(float)
    e = y - d.mu_sbp.to_numpy(float)
    s = d.sigma_sbp.to_numpy(float)
    codes = pd.factorize(d[P], sort=True)[0]
    outcomes = [abs(e) > 1.96 * s, e < -1.96 * s, e > 1.96 * s, abs(e) > 20, y < 90]
    for score in scores.values():
        for q in QUOTAS:
            w = weights_at(score, q)
            for take in (w > 0, w < 1):
                if not take.any():
                    continue
                for yy in outcomes:
                    for condition in (yy, ~yy):
                        n = len(np.unique(codes[take & condition]))
                        assert n == 0 or n >= 10, 'Rate privacy check failed; additional suppression required'


def run(forecasts, features, source, seed, out, reps=REPS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cols = KEY + [P, 'hour_index', 'dialysis_date', 'y_sbp', 'mu_sbp', 'sigma_sbp']
    d = pd.read_parquet(forecasts, columns=cols)
    for c in KEY + [P]:
        d[c] = d[c].astype(str)
    assert not d.duplicated(KEY + ['hour_index']).any(), 'Duplicate forecast keys'
    d['dialysis_date'] = pd.to_datetime(d.dialysis_date)
    assert np.isfinite(d[['y_sbp', 'mu_sbp', 'sigma_sbp']].to_numpy(float)).all() and d.sigma_sbp.gt(0).all()
    f = pd.read_parquet(features, columns=KEY + ['hour_index', 'delta_sbp_from_previous_bin_mmhg'])
    for c in KEY:
        f[c] = f[c].astype(str)
    n1 = len(d)
    d = d.merge(f, on=KEY + ['hour_index'], how='left', validate='one_to_one')
    assert len(d) == n1
    d['abs_delta'] = d.delta_sbp_from_previous_bin_mmhg.abs()
    known = np.isfinite(d.abs_delta.to_numpy(float))
    all_metrics, all_contrasts, flows, thresholds = [], [], [], []

    def flow(stage, x):
        flows.append(dict(source=source, stage=stage, forecast_rows=bucket(len(x)), patients=bucket(x[P].nunique()),
                          sessions=bucket(len(x[KEY].drop_duplicates()))))

    flow('all_forecasts', d)
    flow('delta_observed', d.loc[known])
    flow('delta_missing', d.loc[~known])
    for population, frame in [('all_forecasts', d), ('delta_observed', d.loc[known].copy())]:
        print(f'{source}: {population} paired patient bootstrap.', flush=True)
        codes = pd.factorize(frame[P], sort=True)[0]
        values = row_metrics(frame)
        scores = {'sigma': frame.sigma_sbp.to_numpy(float)}
        if population == 'delta_observed':
            scores['abs_delta'] = frame.abs_delta.to_numpy(float)
        protected_rates(frame, scores)
        results = analyze(scores, values, codes, QUOTAS, reps=reps, seed=seed)
        entries = results['selectors']
        for selector, obj in entries.items():
            point, boots = obj['point'], obj['bootstrap']
            assert np.allclose(point[-1, 0], values.mean(axis=0))
            for qi, q in enumerate(QUOTAS):
                if selector != 'random':
                    thresholds.append(dict(source=source, population=population, selector=selector, retention=q,
                                           score_threshold=float(obj['thresholds'][qi])))
                for gi, portion in enumerate(['retained', 'flagged']):
                    if q == 1 and portion == 'flagged':
                        continue
                    interval = np.quantile(boots[:, qi, gi], [.025, .975], axis=0)
                    for mi, metric in enumerate(METRICS):
                        all_metrics.append(dict(source=source, population=population, selector=selector, retention=q,
                                                portion=portion, metric=metric, estimate=float(point[qi, gi, mi]),
                                                ci_low=float(interval[0, mi]), ci_high=float(interval[1, mi])))
        for comparison in ['random'] + (['abs_delta'] if 'abs_delta' in entries else []):
            a, b = entries['sigma'], entries[comparison]
            for qi, q in enumerate(QUOTAS):
                delta = a['bootstrap'][:, qi, 0] - b['bootstrap'][:, qi, 0]
                interval = np.quantile(delta, [.025, .975], axis=0)
                for mi, metric in enumerate(METRICS):
                    all_contrasts.append(dict(source=source, population=population, retention=q, contrast='sigma_minus_' + comparison,
                                              metric=metric, estimate=float(a['point'][qi, 0, mi] - b['point'][qi, 0, mi]),
                                              ci_low=float(interval[0, mi]), ci_high=float(interval[1, mi])))
    pd.DataFrame(all_metrics).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(all_contrasts).to_csv(out / 'contrasts.csv', index=False)
    pd.DataFrame(flows).to_csv(out / 'cohort_flow.csv', index=False)
    pd.DataFrame(thresholds).to_csv(out / 'score_thresholds.csv', index=False)
    (out / 'settings.json').write_text(json.dumps(dict(source=source, seed=seed, bootstrap_replicates=reps, quotas=QUOTAS), indent=2), encoding='utf-8')
    print(f'{source}: aggregate outputs written to {out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forecasts', required=True)
    p.add_argument('--features', required=True, help='hourly feature table (for the preceding-hour SBP change)')
    p.add_argument('--source', required=True)
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--reps', type=int, default=REPS)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run(args.forecasts, args.features, args.source, args.seed, args.out, args.reps)


if __name__ == '__main__':
    main()
