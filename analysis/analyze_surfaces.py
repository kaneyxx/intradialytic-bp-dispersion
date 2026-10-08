"""Reproduce fixed M2 models and export aggregate conditional-OR surfaces only."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

from . import component_statistics as core

OUT = Path('.')          # set by analyze(frames, out_dir)
MU, SIGMA, NONLINEAR = 'mu_mean90', 'sigma_mean90', 'mu_mean90_rcs_nonlinear'
TERMS = [MU, NONLINEAR, SIGMA]
REPS, SEED = 1000, 20260927
RADIUS = np.array([5.0, 0.5])
MIN_PATIENTS = 30


def dump(name, obj):
    core.dump_json(obj, OUT / name)


def supported_at(frame, points):
    """Outcome-blind distinct-patient exposure support; no local counts emitted."""
    observed = frame[[MU, SIGMA]].to_numpy(float)
    codes = pd.factorize(frame[core.PATIENT], sort=False)[0]
    answer = []
    for point in np.asarray(points, float).reshape(-1, 2):
        near = np.all(np.abs(observed - point) <= RADIUS, axis=1)
        answer.append(np.unique(codes[near]).size >= MIN_PATIENTS)
    return np.asarray(answer, bool)


def configure(frames):
    """Uses exposures only, before any outcome regression."""
    quantiles = {s: {c: np.quantile(f[c], [.01, .1, .5, .9, .99]).tolist()
                     for c in [MU, SIGMA]} for s, f in frames.items()}
    domains = []
    for c, step in [(MU, 1.), (SIGMA, .1)]:
        low = max(quantiles[s][c][0] for s in frames)
        high = min(quantiles[s][c][-1] for s in frames)
        bounds = [float(np.ceil(low / step) * step), float(np.floor(high / step) * step)]
        assert bounds[0] < bounds[1], 'No_common_display_domain'
        domains.append(bounds)
    ref = np.array([round(np.mean([quantiles[s][MU][2] for s in frames])),
                    round(np.mean([quantiles[s][SIGMA][2] for s in frames]), 1)], float)
    fallback = False
    axes = [np.linspace(*d, 81) for d in domains]
    if (not all(d[0] <= v <= d[1] for d, v in zip(domains, ref))
            or not all(supported_at(f, [ref])[0] for f in frames.values())):
        mesh = np.meshgrid(*axes)
        points = np.column_stack([a.ravel() for a in mesh])
        good = np.logical_and.reduce([supported_at(f, points) for f in frames.values()])
        assert good.any(), 'No_common_supported_reference'
        candidates = points[good]
        span = np.array([d[1] - d[0] for d in domains])
        ref = candidates[np.argmin(np.sum(((candidates - ref) / span) ** 2, axis=1))]
        fallback = True
    axes = [np.unique(np.r_[a, r]) for a, r in zip(axes, ref)]
    grid = np.column_stack([a.ravel() for a in np.meshgrid(*axes)])
    curves = {}
    for j, variable in enumerate(['mu', 'sigma']):
        values = np.unique(np.r_[np.linspace(*domains[j], 161), ref[j]])
        points = np.tile(ref, (len(values), 1))
        points[:, j] = values
        curves[variable] = (values, points)
    config = {'created_utc_before_regression': datetime.now(timezone.utc).isoformat(),
              'reference': dict(zip(['mu', 'sigma'], ref.tolist())),
              'reference_fallback_used': fallback, 'reference_support_verified_both_sources': True,
              'mu_domain': domains[0], 'sigma_domain': domains[1],
              'bootstrap_replicates': REPS, 'seed_by_source': {'CY': SEED, 'TT': SEED + 1},
              'ci_type': 'pointwise percentile patient-cluster bootstrap',
              'model': 'M2: mean RCS3 + linear sigma + fixed covariates',
              'mu_knots': {s: quantiles[s][MU][1:4] for s in frames},
              'exposure_quantiles_p01_p10_p50_p90_p99': quantiles,
              'support': {'halfwidth_mu_mmhg': 5, 'halfwidth_sigma_mmhg': .5,
                          'minimum_distinct_patients': MIN_PATIENTS, 'outcome_independent': True},
              'absolute_risk_or_interaction_estimated': False}
    dump('plot_config.json', config)
    return config, ref, grid, curves


def difference_matrix(points, reference, knots):
    p = np.asarray(points, float).reshape(-1, 2)
    return np.column_stack([p[:, 0] - reference[0],
        core.rcs3(p[:, 0], knots) - core.rcs3(np.asarray([reference[0]]), knots)[0],
        p[:, 1] - reference[1]])


def contrast_estimates(delta, beta, draws):
    log_or = delta @ beta
    log_draws = draws @ delta.T
    limits = np.quantile(log_draws, [.025, .975], axis=0)
    assert np.isfinite(log_or).all() and np.isfinite(limits).all(), 'Nonfinite_contrast'
    assert np.max(np.abs(np.r_[log_or, limits.ravel()])) < 700, 'Contrast_overflow'
    return np.exp(log_or), np.exp(limits[0]), np.exp(limits[1])


def cluster_bootstrap(raw, y, codes, names, source, seed):
    rng = np.random.default_rng(seed)
    by_patient = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    draws, failures = [], Counter()
    for rep in range(REPS):
        chosen = rng.integers(0, len(by_patient), len(by_patient))
        ix = np.concatenate([by_patient[j] for j in chosen])
        try:
            fit = core.fit_array(raw[ix], y[ix], names, TERMS, MU)
            assert len(fit['names']) == 12, 'Unexpected_covariate_drop'
            indices = [fit['names'].index(t) for t in TERMS]
            draws.append(fit['beta'][indices] / fit['scales'][indices])
        except core.Nonestimable as exc:
            failures[str(exc)] += 1
        except Exception as exc:
            failures[type(exc).__name__] += 1
        if (rep + 1) % 100 == 0:
            print(f'{source}: patient bootstrap {rep + 1}/{REPS}.', flush=True)
    dump(f'bootstrap_{source}.json', {'requested': REPS, 'valid': len(draws), 'failure_reasons': dict(failures)})
    assert len(draws) >= 0.95 * REPS, 'Insufficient_successful_bootstrap_replicates'
    return np.asarray(draws)


def cluster_wald(fit, source):
    rows = []
    for label, terms in [('mu_overall', [MU, NONLINEAR]), ('mu_nonlinearity', [NONLINEAR]), ('sigma', [SIGMA])]:
        ix = [fit['names'].index(t) for t in terms]
        beta = fit['beta'][ix]
        cov = fit['cov'][np.ix_(ix, ix)]
        statistic = float(beta @ np.linalg.solve(cov, beta))
        rows.append({'source': source, 'test': label, 'df': len(ix), 'wald_chi2': statistic,
                     'p_nominal': float(chi2.sf(statistic, len(ix))),
                     'inference': 'exploratory patient-cluster Wald; not multiplicity adjusted'})
    return rows


def analyze(frames, out_dir):
    """frames: {source: landmark frame}; writes the aggregate tables (joint_grid, curves, overall_tests,
    prespecified_contrasts, model_coefficients, cohort_flow, covariate_preprocessing) to out_dir."""
    global OUT
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    config, reference, grid, curves = configure(frames)
    all_grid, all_curves, tests, contrasts, coefficients, counts, checks, preprocessing = [], [], [], [], [], [], [], []
    with threadpool_limits(limits=1):
        for source_index, (source, frame) in enumerate(frames.items()):
            assert frame.source.eq(source).all(), 'Source_frame_mismatch'
            assert not frame.duplicated([core.PATIENT, 'landmark_date']).any(), 'Duplicate_landmark'
            y = frame.death90.to_numpy(int)
            assert set(np.unique(y)) == {0, 1}, 'Invalid_outcome'
            codes = pd.factorize(frame[core.PATIENT], sort=True)[0]
            baseline = core.make_source_configs()[source]['baseline_columns']
            knots = config['mu_knots'][source]
            raw, names = core.make_design(frame, [SIGMA], baseline, MU, knots)
            fit = core.full_fit(raw, y, codes, names, TERMS, MU)
            assert len(fit['names']) == 12 and not fit['dropped'], 'Unexpected_M2_design'
            assert fit['verification']['passed'], 'Independent_GLM_verification_failed'
            indices = [fit['names'].index(t) for t in TERMS]
            beta = fit['beta'][indices] / fit['scales'][indices]
            actual_sigma_or = float(np.exp(fit['beta'][fit['names'].index(SIGMA)]))
            check = {'source': source, **fit['verification'], 'sigma_or_per_sd': actual_sigma_or,
                     'design_parameters': len(fit['names']), 'condition_number': fit['condition'],
                     'events_per_parameter_below_10': bool(y.sum() / len(fit['names']) < 10)}
            print(f'{source}: full original M2 and independent GLM verified; bootstrapping.', flush=True)
            draws = cluster_bootstrap(raw, y, codes, names, source, SEED + source_index)
            delta = difference_matrix(grid, reference, knots)
            mask = supported_at(frame, grid)
            values = contrast_estimates(delta, beta, draws)
            g = pd.DataFrame({'source': source, 'mu': grid[:, 0], 'sigma': grid[:, 1],
                              'or': values[0], 'ci_low': values[1], 'ci_high': values[2], 'supported': mask})
            # Do not export fitted OR/CI values in unsupported regions.
            g.loc[~g.supported, ['or', 'ci_low', 'ci_high']] = np.nan
            all_grid.append(g)
            for variable, (axis, points) in curves.items():
                values = contrast_estimates(difference_matrix(points, reference, knots), beta, draws)
                supported = supported_at(frame, points)
                c = pd.DataFrame({'source': source, 'exposure': variable, 'value': axis,
                                  'or': values[0], 'ci_low': values[1], 'ci_high': values[2], 'supported': supported})
                c.loc[~c.supported, ['or', 'ci_low', 'ci_high']] = np.nan
                all_curves.append(c)
            at_reference = difference_matrix([reference], reference, knots)
            assert np.array_equal(at_reference, np.zeros((1, 3))), 'Nonzero_reference_contrast'
            assert all(float(v[0]) == 1 for v in contrast_estimates(at_reference, beta, draws)), 'Reference_OR_CI_not_one'
            # Additivity is an exact algebra check, not a finding of no interaction.
            delta_m = difference_matrix(np.column_stack([grid[:, 0], np.full(len(grid), reference[1])]), reference, knots)
            delta_s = difference_matrix(np.column_stack([np.full(len(grid), reference[0]), grid[:, 1]]), reference, knots)
            assert np.allclose(delta, delta_m + delta_s, rtol=0, atol=1e-12), 'Contrast_additivity_failed'
            check.update(reference_or_ci_exactly_one=True, contrast_additivity_verified=True,
                         bootstrap_valid=len(draws), support_mask_outcome_independent=True)
            checks.append(check)
            tests.extend(cluster_wald(fit, source))
            for a, b in [(120., 140.), (140., 160.)]:
                pts = np.array([[a, reference[1]], [b, reference[1]]])
                supported = bool(supported_at(frame, pts).all()
                    and all(config['mu_domain'][0] <= v <= config['mu_domain'][1] for v in (a, b)))
                d = difference_matrix(pts[:1], pts[1], knots)
                v = contrast_estimates(d, beta, draws)
                contrasts.append({'source': source, 'contrast': f'mu_{int(a)}_vs_{int(b)}',
                    'unit': 'mmHg', 'or': v[0][0] if supported else np.nan,
                    'ci_low': v[1][0] if supported else np.nan, 'ci_high': v[2][0] if supported else np.nan,
                    'supported': supported, 'increment': a - b, 'bootstrap_valid': len(draws)})
            for label, increment in [('sigma_per_1_mmhg', 1.), ('sigma_per_source_sd', float(frame[SIGMA].std(ddof=0)))]:
                v = contrast_estimates(np.array([[0., 0., increment]]), beta, draws)
                contrasts.append({'source': source, 'contrast': label, 'unit': 'mmHg', 'or': v[0][0],
                    'ci_low': v[1][0], 'ci_high': v[2][0], 'supported': True,
                    'increment': increment, 'bootstrap_valid': len(draws)})
            for name, b, se, scale in zip(fit['names'], fit['beta'], fit['se'], fit['scales']):
                if name != 'intercept':
                    coefficients.append({'source': source, 'term': name, 'beta_raw': float(b / scale),
                                         'cluster_se_raw': float(se / scale)})
            for col in baseline:
                missing = ~np.isfinite(frame[col].to_numpy(float))
                nmiss = int(frame.loc[missing, core.PATIENT].nunique())
                preprocessing.append({'source': source, 'variable': col,
                    'missing_windows_range': core.bucket_counts({'n': missing.sum()})['n'],
                    'patients_with_missing_range': core.bucket_counts({'n': nmiss})['n'],
                    'method': 'source median refit in each patient bootstrap'})
            counts.append({'source': source, **core.bucket_counts({'windows': len(frame),
                'patients': frame[core.PATIENT].nunique(), 'deaths': y.sum()}),
                'scope': 'original full mortality M2 cohort; no older-history restriction'})
    pd.concat(all_grid, ignore_index=True).to_csv(OUT / 'joint_grid.csv', index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(OUT / 'curves.csv', index=False)
    pd.DataFrame(tests).to_csv(OUT / 'overall_tests.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / 'prespecified_contrasts.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT / 'model_coefficients.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'cohort_flow.csv', index=False)
    pd.DataFrame(preprocessing).to_csv(OUT / 'covariate_preprocessing.csv', index=False)
    dump('numerical_verification.json', {'all_passed': all(x['passed'] for x in checks), 'sources': checks})
    return checks


def self_test():
    knots = [100., 140., 180.]
    reference = np.array([140., 15.])
    points = np.array([[120., 12.], [140., 15.], [160., 18.]])
    delta = difference_matrix(points, reference, knots)
    assert np.array_equal(delta[1], np.zeros(3))
    beta = np.array([-.02, .01, .2])
    bdraws = np.vstack([beta + [-.001, .001, -.01], beta, beta + [.001, -.001, .01]])
    values = contrast_estimates(delta, beta, bdraws)
    assert all(v[1] == 1 for v in values)
    shifted = difference_matrix(points, [130., 14.], knots)
    ref_to_shift = difference_matrix([reference], [130., 14.], knots)
    assert np.allclose(shifted - ref_to_shift, delta)
    # The support rule must count people, not repeated windows.
    fake = pd.DataFrame({MU: [140.] * 60, SIGMA: [15.] * 60, core.PATIENT: np.repeat(np.arange(20), 3)})
    assert not supported_at(fake, [reference])[0]
    fake[core.PATIENT] = np.repeat(np.arange(30), 2)
    assert supported_at(fake, [reference])[0]
    assert not supported_at(fake, [[146., 15.]])[0]
    return {'passed': True, 'checks': ['zero_reference_CI', 'reference_invariant_contrasts',
        'distinct_patient_not_window_support', 'fixed_joint_neighbourhood']}
