"""Per-forecast contributions of each input to log sigma, averaged over the exposure window and summed within four
groups (Table S5, Figure S4).

Each input's fitted contribution to log sigma (numeric term plus missing-indicator term, averaged over the five members
and centred on the member's fitting-partition mean) is computed for every forecast, then averaged over the forecasts of
each patient-quarter's exposure window and summed within the groups: blood pressure and heart rate; treatment settings
and session hour; demographics; laboratory values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .landmarks import P, ROWKEY, bucket

FEATURES = ['bin_index', 'sbp_mean_all_mmhg', 'dbp_mean_all_mmhg', 'hr_mean_all_bpm', 'uf_rate_mean_ml_hr',
            'age_at_session_years', 'sex_female', 'hb_value', 'ca_value', 'actual_qb_mean_ml_min',
            'dialysate_temperature_last_c', 'actual_qd_last_ml_min', 'delta_sbp_from_previous_bin_mmhg']
FEATURE_GROUPS = {
    'group_bp': ['sbp_mean_all_mmhg', 'dbp_mean_all_mmhg', 'delta_sbp_from_previous_bin_mmhg', 'hr_mean_all_bpm'],
    'group_treatment': ['bin_index', 'uf_rate_mean_ml_hr', 'actual_qb_mean_ml_min', 'actual_qd_last_ml_min', 'dialysate_temperature_last_c'],
    'group_demographic': ['age_at_session_years', 'sex_female'],
    'group_labs': ['hb_value', 'ca_value'],
}
CONTRIBUTIONS = ['ls_c_' + f for f in FEATURES]


def row_contributions(bundle, pre, raw, centers):
    """Grouped (numeric + indicator) member-average contributions to log sigma for each row of `raw`, centred with the
    per-member means `centers` (model.contributions.member_centers for head 'log_sigma' on the fitting partition)."""
    import torch
    torch.set_num_threads(2)
    n, nf = len(raw), len(FEATURES)
    grouped = np.zeros((n, nf), dtype=np.float64)
    rawlog = np.zeros(n, dtype=np.float64)
    cfg = bundle['config']
    members = bundle['models']
    for member in members:
        member.cpu().eval()
    for start in range(0, n, 8192):
        stop = min(n, start + 8192)
        x = pre.transform(raw.iloc[start:stop][FEATURES]).to_numpy(dtype=np.float32)
        assert np.isfinite(x).all(), 'nonfinite preprocessed inputs'
        with torch.inference_mode():
            tensor = torch.as_tensor(x)
            for m, member in enumerate(members):
                cc = member.log_sigma_sbp.contributions(tensor).cpu().numpy().astype(np.float64)
                values = float(member.log_sigma_sbp.intercept.detach()) + cc.sum(axis=1)
                grouped[start:stop] += (cc[:, :nf] + cc[:, nf:] - centers[m, :nf] - centers[m, nf:]) / len(members)
                rawlog[start:stop] += values / len(members)
    out = pd.DataFrame(grouped, columns=CONTRIBUTIONS, index=raw.index)
    out['ensemble_mean_raw_log_sigma'] = rawlog
    for f in FEATURES:
        out['raw_' + f] = raw[f].to_numpy(float)
    return out


def exposure_membership(frame, rows):
    """Select the union of forecast rows that fall in the exposure windows of `frame`."""
    rows = rows.sort_values([P, 'dialysis_date', 'hour_index']).reset_index(drop=True)
    assert not rows.duplicated(ROWKEY).any(), 'Nonunique forecast row key'
    used = np.zeros(len(rows), dtype=bool)
    by_patient = {p: g for p, g in frame.groupby(P, sort=False)}
    mismatches = 0
    max_mu_error = max_sigma_error = 0.
    seen = 0
    for token, g in rows.groupby(P, sort=False):
        windows = by_patient.get(token)
        if windows is None:
            continue
        dates = g.dialysis_date.to_numpy(dtype='datetime64[ns]')
        lm = windows.landmark_date.to_numpy(dtype='datetime64[ns]')
        left = np.searchsorted(dates, lm - np.timedelta64(90, 'D'), side='left')
        right = np.searchsorted(dates, lm, side='left')
        n = right - left
        mismatches += np.count_nonzero(n != windows.prior90_n_rows.to_numpy(int))
        assert np.all(n > 0), 'An existing landmark has no forecast rows'
        for name, expected in [('mu_sbp', 'mu_mean90'), ('sigma_sbp', 'sigma_mean90')]:
            cs = np.r_[0., np.cumsum(g[name].to_numpy(float))]
            actual = (cs[right] - cs[left]) / n
            diff = float(np.max(np.abs(actual - windows[expected].to_numpy(float))))
            if name == 'mu_sbp':
                max_mu_error = max(max_mu_error, diff)
            else:
                max_sigma_error = max(max_sigma_error, diff)
        idx = g.index.to_numpy()
        for lo, hi in zip(left, right):
            used[idx[lo:hi]] = True
        seen += len(windows)
    assert seen == len(frame) and mismatches == 0, 'Exposure-row count not reproduced'
    assert max_mu_error < 1e-7 and max_sigma_error < 1e-7, 'Exposure means not reproduced'
    return rows.loc[used].reset_index(drop=True), {'unique_exposure_rows_range': bucket(used.sum()), 'window_count_range': bucket(len(frame))}


def aggregate_windows(source, frame, rows):
    """Average row contributions over exactly the same [landmark-90d, landmark) windows."""
    rows = rows.sort_values([P, 'dialysis_date', 'hour_index']).reset_index(drop=True)
    rows['row_log_sigma'] = np.log(rows.sigma_sbp.to_numpy(float))
    rows['row_pooling_residual'] = rows.row_log_sigma - rows.ensemble_mean_raw_log_sigma
    numerical_cols = CONTRIBUTIONS + ['ensemble_mean_raw_log_sigma', 'row_log_sigma', 'row_pooling_residual']
    output_names = CONTRIBUTIONS + ['mean_raw_log_sigma90', 'mean_log_sigma90', 'mean_log_sigma_pooling_residual90']
    raw_cols = ['raw_' + f for f in FEATURES]
    rows['raw_abs_delta_sbp'] = np.abs(rows['raw_delta_sbp_from_previous_bin_mmhg'])
    raw_cols += ['raw_abs_delta_sbp']
    raw_names = ['raw_mean_' + f for f in FEATURES] + ['raw_mean_abs_delta_sbp']
    missing_names = ['raw_missing_fraction_' + f for f in FEATURES] + ['raw_missing_fraction_abs_delta_sbp']
    outputs = np.full((len(frame), len(numerical_cols) + len(raw_cols) * 2), np.nan)
    names = output_names + raw_names + missing_names
    by_patient = {p: g for p, g in rows.groupby(P, sort=False)}
    count_mismatch = 0
    direct_max = 0.
    for token, windows in frame.groupby(P, sort=False):
        g = by_patient[token]
        dates = g.dialysis_date.to_numpy(dtype='datetime64[ns]')
        landmarks = windows.landmark_date.to_numpy(dtype='datetime64[ns]')
        left = np.searchsorted(dates, landmarks - np.timedelta64(90, 'D'), side='left')
        right = np.searchsorted(dates, landmarks, side='left')
        n = right - left
        count_mismatch += np.count_nonzero(n != windows.prior90_n_rows.to_numpy(int))
        a = g[numerical_cols].to_numpy(float)
        cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        means = (cs[right] - cs[left]) / n[:, None]
        raw = g[raw_cols].to_numpy(float)
        finite = np.isfinite(raw)
        raw_cs = np.vstack([np.zeros((1, raw.shape[1])), np.cumsum(np.where(finite, raw, 0.), axis=0)])
        n_cs = np.vstack([np.zeros((1, raw.shape[1])), np.cumsum(finite, axis=0)])
        denominator = n_cs[right] - n_cs[left]
        raw_means = np.divide(raw_cs[right] - raw_cs[left], denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
        missing = 1. - denominator / n[:, None]
        outputs[windows.index.to_numpy()] = np.column_stack([means, raw_means, missing])
        for k, (lo, hi) in enumerate(zip(left, right)):
            direct_max = max(direct_max, float(np.max(np.abs(a[lo:hi].mean(axis=0) - means[k]))))
    assert count_mismatch == 0 and direct_max < 1e-9, 'Contribution-window aggregation mismatch'
    result = pd.concat([frame, pd.DataFrame(outputs, columns=names, index=frame.index)], axis=1)
    for group, features in FEATURE_GROUPS.items():
        result[group] = result[['ls_c_' + f for f in features]].sum(axis=1)
    assert np.isfinite(result[CONTRIBUTIONS + list(FEATURE_GROUPS)].to_numpy(float)).all()
    total = result[list(FEATURE_GROUPS)].sum(axis=1)
    assert np.max(np.abs(total - result[CONTRIBUTIONS].sum(axis=1))) < 1e-12
    return result
