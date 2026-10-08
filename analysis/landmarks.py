"""Landmark cohorts (patient-quarters) from the forecast table, death dates and per-session observed SBP summaries.

Supplementary Methods S2. At each landmark (the first day of a calendar quarter) the preceding 90 days form the exposure
window and the following 90 days the outcome window. A patient contributes a patient-quarter when at least one eligible
forecast falls in the exposure window. The outcome is death within 90 days; a patient-quarter counts as survival when the
patient has a dialysis session with recorded blood pressure on or after day 90 or a death recorded after day 90, and is
excluded (unknown vital status) otherwise. Exposures: mean sigma and mean mu over the forecasts of the window, the SD and
10th percentile of mu, the fraction of forecasts with mu below 90 mmHg, and from the observed sessions of the window the
fraction with a nadir SBP below 90 mmHg (hypotension history), the between-session SD of session-mean SBP and the mean
within-session SD. The change analysis adds the earlier window (days -180 to -91). The timing analysis splits the
90-day outcome into days 1-30 and 31-90.

Inputs (see docs/data_dictionary.md): forecasts (one row per forecast), observed_sessions (one row per session with
recorded SBP), persons (birth year and sex), deaths (one row per patient with a vital-status record, survivors
included: the death date, NaT when no death is recorded; the date of the last session with recorded blood pressure; and
optionally a flag that the death date was the start of the current status rather than a dated change). Patients with
forecasts but no row in deaths are excluded from the cohort. All identifiers are pseudonymised; how these tables are
built from hospital records is not part of this repository.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

P = 'patient_token_v1'
KEY = ['source_system', 'hd_session_id']
ROWKEY = KEY + [P, 'hour_index']
WINDOWKEY = [P, 'landmark_date']
MIN_OLD_SESSIONS = 4


def bucket(value):
    """Bound every detailed count, avoiding complementary small-cell disclosure."""
    n = int(value)
    return '0' if n == 0 else '<10' if n < 10 else f'{n // 10 * 10}-{n // 10 * 10 + 9}'


def safe_counts(counts):
    return {k: bucket(v) for k, v in counts.items()}


def keys(d):
    for c in KEY + [P]:
        if c in d:
            d[c] = d[c].astype(str)
    return d


def mortality_label(landmark, death, last_bp):
    end = landmark + pd.Timedelta(days=90)
    if pd.notna(death) and death <= landmark:
        return 'post_death', None
    if pd.notna(death) and death <= end:
        return 'death', 1
    if (pd.notna(death) and death > end) or (pd.notna(last_bp) and last_bp >= end):
        return 'alive_evidence', 0
    return 'unknown', None


def boundary_checks():
    l = pd.Timestamp('2020-01-01')
    for death, last, want in [
        (l, pd.NaT, 'post_death'),
        (l + pd.Timedelta(days=90), pd.NaT, 'death'),
        (l + pd.Timedelta(days=91), pd.NaT, 'alive_evidence'),
        (pd.NaT, l + pd.Timedelta(days=90), 'alive_evidence'),
        (pd.NaT, l + pd.Timedelta(days=89), 'unknown'),
    ]:
        assert mortality_label(l, death, last)[0] == want, 'Synthetic endpoint boundary failure'
    dates = np.array([l - pd.Timedelta(days=90), l - pd.Timedelta(days=1), l], dtype='datetime64[ns]')
    assert int(((dates >= np.datetime64(l - pd.Timedelta(days=90))) & (dates < np.datetime64(l))).sum()) == 2
    assert mortality_label(l, pd.Timestamp('2014-01-01'), l + pd.Timedelta(days=100))[0] == 'post_death'


def session_summaries(hourly):
    """Per-session mean SBP and within-session SD from hourly mean SBP (not from every cuff reading)."""
    hourly = hourly.copy()
    hourly['sq'] = hourly.sbp_mean_all_mmhg ** 2
    hs = hourly.groupby(KEY, sort=False).agg(session_mean=('sbp_mean_all_mmhg', 'mean'), sq=('sq', 'mean'),
                                              n=('sbp_mean_all_mmhg', 'count')).reset_index()
    hs['session_within_sd'] = np.sqrt(np.maximum(0, hs.sq - hs.session_mean ** 2)).where(hs.n.ge(2))
    return hs[KEY + ['session_mean', 'session_within_sd']]


def build_landmark_cohort(fc, obs, person, deaths, first_landmark, end, source):
    """One row per analysable patient-quarter.

    fc: forecasts with KEY, P, dialysis_date, hour_index, mu_sbp, sigma_sbp, point_nadir90.
    obs: observed sessions with KEY, P, dialysis_date, observed_session_sbp_min, session_mean, session_within_sd.
    person: index P, columns birth_year, sex_female.
    deaths: index P, columns death_date (NaT if none), pure_current_state_start (bool), last_bp (last session date with
            recorded blood pressure).
    """
    boundary_checks()
    fc = keys(fc.copy()).sort_values([P, 'dialysis_date', 'hour_index'])
    obs = keys(obs.copy()).sort_values([P, 'dialysis_date'])
    status = deaths.to_dict('index')
    last = deaths['last_bp'].to_dict()
    obs_groups = {p: g for p, g in obs.groupby(P, sort=False)}
    landmarks = pd.date_range(first_landmark, end - pd.Timedelta(days=90), freq='QS')
    windows = []
    flow = Counter()
    for token, g in fc.groupby(P, sort=False):
        if token not in status:
            flow['patients_without_status_record'] += 1
            continue
        s = status[token]
        dates = g.dialysis_date.to_numpy(dtype='datetime64[ns]')
        mu = g.mu_sbp.to_numpy(float)
        sigma = g.sigma_sbp.to_numpy(float)
        pt = g.point_nadir90.to_numpy(float)
        session_codes = pd.factorize(pd.MultiIndex.from_frame(g[KEY]))[0]
        o = obs_groups.get(token)
        od = o.dialysis_date.to_numpy(dtype='datetime64[ns]') if o is not None else np.array([], dtype='datetime64[ns]')
        age_birth, sex = person.loc[token, ['birth_year', 'sex_female']] if token in person.index else (np.nan, np.nan)
        sex = float(sex) if pd.notna(sex) else np.nan
        for l in landmarks:
            a, b = np.searchsorted(dates, [np.datetime64(l - pd.Timedelta(days=90)), np.datetime64(l)])
            if b <= a:
                flow['no_prior_forecast_windows'] += 1
                continue
            flow['prior_forecast_candidate_windows'] += 1
            endpoint, y = mortality_label(l, s['death_date'], last.get(token, pd.NaT))
            flow[endpoint + '_windows'] += 1
            if y is None:
                continue
            oa, ob = np.searchsorted(od, [np.datetime64(l - pd.Timedelta(days=90)), np.datetime64(l)])
            hist = o.iloc[oa:ob] if o is not None else None
            nobs = ob - oa
            hx = float(hist.observed_session_sbp_min.lt(90).mean()) if nobs else np.nan
            means = hist.session_mean.dropna().to_numpy(float) if nobs else np.array([])
            within = hist.session_within_sd.dropna().to_numpy(float) if nobs else np.array([])
            age = float(l.year - age_birth) if pd.notna(age_birth) else np.nan
            if pd.notna(age) and not 0 <= age <= 120:
                age = np.nan
            nsession = len(np.unique(session_codes[a:b]))
            fa, fb = np.searchsorted(od, [np.datetime64(l), np.datetime64(l + pd.Timedelta(days=90))], side='right')
            windows.append({P: token, 'source': source, 'landmark_date': l, 'death90': int(y),
                            'sigma_mean90': float(sigma[a:b].mean()), 'mu_mean90': float(mu[a:b].mean()),
                            'mu_sd': float(mu[a:b].std(ddof=0)), 'mu_p10': float(np.quantile(mu[a:b], .1)),
                            'age_at_index_years': age, 'sex_female': sex, 'log1p_prior90_sessions': float(np.log1p(nsession)),
                            'index_year': l.year, 'prior90_observed_nadir90_burden': hx,
                            'prior90_point_nadir90_burden': float(pt[a:b].mean()),
                            'observed_between_sd': float(means.std(ddof=0)) if len(means) else np.nan,
                            'observed_within_sd': float(within.mean()) if len(within) else np.nan,
                            'observed_mean': float(means.mean()) if len(means) else np.nan,
                            'observed_nadir_p10': float(np.quantile(hist.observed_session_sbp_min.to_numpy(float), .1)) if nobs else np.nan,
                            'pure_current_state_start': bool(s.get('pure_current_state_start', False)),
                            'prior90_n_rows': int(b - a), 'prior90_n_sessions': nsession,
                            'prior90_observed_sessions': int(nobs), 'future90_observed_sessions': int(fb - fa)})
    frame = pd.DataFrame(windows)
    assert len(frame) > 0 and not frame.duplicated(WINDOWKEY).any(), 'Empty or duplicate mortality cohort'
    assert frame.death90.isin([0, 1]).all() and frame.death90.nunique() == 2, 'Outcome not estimable'
    assert frame.prior90_n_rows.gt(0).all(), 'Empty exposure window'
    assert frame.loc[frame.death90.eq(1)].groupby(P).size().le(1).all(), 'Death counted in multiple quarterly horizons'
    flow['analysable_windows'] = len(frame)
    return frame, {'window_flow_count_ranges': safe_counts(flow), 'endpoint': 'death within 90 days with documented vital status'}


def add_prior_history(frame, rows, start, end):
    """Change analysis: derive the earlier window (days -180 to -91) from the forecasts; keep windows with at least four
    earlier sessions. Exact recent-window means are reproduced from the same rows as a check."""
    boundary_checks()
    rows = rows.sort_values([P, 'dialysis_date', 'hour_index']).reset_index(drop=True)
    assert not rows.duplicated(ROWKEY).any(), 'Nonunique forecast row key'
    assert np.isfinite(rows[['mu_sbp', 'sigma_sbp']].to_numpy(float)).all(), 'Nonfinite prediction'
    assert rows.sigma_sbp.gt(0).all(), 'Invalid sigma'
    assert frame.landmark_date.sub(pd.Timedelta(days=180)).ge(start).all(), 'Old window crosses the data start'
    assert frame.landmark_date.add(pd.Timedelta(days=90)).le(end).all(), 'Outcome crosses the data end'
    columns = ['mu_old90', 'sigma_old90', 'old90_n_rows', 'old90_n_sessions',
               'old90_observed_span_days', 'recent90_n_sessions', 'recent90_observed_span_days']
    values = np.full((len(frame), len(columns)), np.nan)
    by_patient = {p: g for p, g in rows.groupby(P, sort=False)}
    max_mu_error = max_sigma_error = max_direct_error = max_session_error = 0.
    count_mismatches = 0
    for token, windows in frame.groupby(P, sort=False):
        g = by_patient[token]
        dates = g.dialysis_date.to_numpy(dtype='datetime64[ns]')
        landmarks = windows.landmark_date.to_numpy(dtype='datetime64[ns]')
        a = np.searchsorted(dates, landmarks - np.timedelta64(180, 'D'), side='left')
        b = np.searchsorted(dates, landmarks - np.timedelta64(90, 'D'), side='left')
        c = np.searchsorted(dates, landmarks, side='left')
        nold, nrecent = b - a, c - b
        count_mismatches += int(np.count_nonzero(nrecent != windows.prior90_n_rows.to_numpy(int)))
        assert np.all(nrecent > 0), 'An existing landmark has no recent forecasts'
        predictions = g[['mu_sbp', 'sigma_sbp']].to_numpy(float)
        cs = np.vstack([np.zeros((1, 2)), np.cumsum(predictions, axis=0)])
        old_means = np.divide(cs[b] - cs[a], nold[:, None], out=np.full((len(windows), 2), np.nan), where=nold[:, None] > 0)
        recent_means = (cs[c] - cs[b]) / nrecent[:, None]
        max_mu_error = max(max_mu_error, float(np.max(np.abs(recent_means[:, 0] - windows.mu_mean90))))
        max_sigma_error = max(max_sigma_error, float(np.max(np.abs(recent_means[:, 1] - windows.sigma_mean90))))
        sessions = pd.factorize(pd.MultiIndex.from_frame(g[KEY]))[0]
        old_ns, recent_ns, old_span, recent_span = [], [], [], []
        for k, (lo, mid, hi) in enumerate(zip(a, b, c)):
            old_ns.append(len(np.unique(sessions[lo:mid])))
            recent_ns.append(len(np.unique(sessions[mid:hi])))
            old_span.append(float((dates[mid - 1] - dates[lo]) / np.timedelta64(1, 'D')) if mid > lo else np.nan)
            recent_span.append(float((dates[hi - 1] - dates[mid]) / np.timedelta64(1, 'D')))
            if mid > lo:
                max_direct_error = max(max_direct_error, float(np.max(np.abs(predictions[lo:mid].mean(axis=0) - old_means[k]))))
        expected_sessions = np.expm1(windows.log1p_prior90_sessions.to_numpy(float))
        max_session_error = max(max_session_error, float(np.max(np.abs(expected_sessions - recent_ns))))
        values[windows.index.to_numpy()] = np.column_stack([old_means, nold, old_ns, old_span, recent_ns, recent_span])
    assert count_mismatches == 0, 'Recent forecast counts not reproduced'
    assert max_mu_error < 1e-7 and max_sigma_error < 1e-7, 'Recent means not reproduced'
    assert max_direct_error < 1e-7, 'Old-window direct means disagree with cumulative means'
    assert max_session_error < 1e-7, 'Recent session counts not reproduced'
    augmented = pd.concat([frame, pd.DataFrame(values, columns=columns, index=frame.index)], axis=1)
    for column in ['old90_n_rows', 'old90_n_sessions', 'recent90_n_sessions']:
        augmented[column] = augmented[column].astype(int)
    included = augmented.old90_n_sessions.ge(MIN_OLD_SESSIONS)
    result = augmented.loc[included].copy().reset_index(drop=True)
    result['delta_sigma'] = result.sigma_mean90 - result.sigma_old90
    result['delta_mu'] = result.mu_mean90 - result.mu_old90
    result['log1p_old90_sessions'] = np.log1p(result.old90_n_sessions)
    assert np.isfinite(result[['sigma_old90', 'mu_old90', 'delta_sigma', 'delta_mu']].to_numpy(float)).all(), 'Nonfinite dynamic exposures'
    assert not result.duplicated(WINDOWKEY).any(), 'Duplicate dynamic landmark'
    assert result.loc[result.death90.eq(1)].groupby(P).size().le(1).all(), 'Death repeated in dynamic risk set'
    audit = {'window_definition': {'old': '[L-180 days, L-90 days)', 'recent': '[L-90 days, L)', 'outcome': '(L, L+90 days]'},
             'minimum_old_distinct_forecast_sessions': MIN_OLD_SESSIONS,
             'flow': {'original': safe_counts({'windows': len(frame), 'patients': frame[P].nunique(), 'deaths': frame.death90.sum()}),
                      'excluded_no_old_forecasts': bucket(augmented.old90_n_rows.eq(0).sum()),
                      'excluded_one_to_three_old_sessions': bucket(augmented.old90_n_sessions.between(1, MIN_OLD_SESSIONS - 1).sum()),
                      'included': safe_counts({'windows': len(result), 'patients': result[P].nunique(), 'deaths': result.death90.sum()})}}
    return result, audit


def label_intervals(landmarks, deaths):
    """Right-closed outcomes; death exactly at L+30 belongs to the early interval only."""
    end30 = landmarks + pd.Timedelta(days=30)
    end90 = landmarks + pd.Timedelta(days=90)
    early = deaths.gt(landmarks) & deaths.le(end30)
    late = deaths.gt(end30) & deaths.le(end90)
    any90 = deaths.gt(landmarks) & deaths.le(end90)
    eligible_late = ~early
    return early, late, any90, eligible_late


def attach_intervals(frame, death_dates):
    """Timing analysis: deaths during days 1-30, and days 31-90 among patient-quarters alive at day 30.
    death_dates: Series aligned with frame (NaT when no death)."""
    deaths = death_dates
    assert len(deaths) == len(frame) and deaths.index.equals(frame.index), 'Death dates lost frame alignment'
    assert (deaths.isna() | deaths.gt(frame.landmark_date)).all(), 'Death on or before a retained landmark'
    early, late, any90, eligible_late = label_intervals(frame.landmark_date, deaths)
    assert np.array_equal(any90.astype(int), frame.death90.to_numpy(int)), 'death90 not reproduced from the death date'
    assert np.array_equal(any90, early | late) and not (early & late).any(), 'Outcome partition failed'
    result = frame.copy()
    result['death_early30'] = early.astype(int)
    result['eligible_late'] = eligible_late.astype(bool)
    result['death_late31_90'] = late.astype(float).where(eligible_late)
    assert result.death_late31_90.isna().eq(result.death_early30.eq(1)).all()
    assert result.loc[result.eligible_late, 'death_late31_90'].isin([0., 1.]).all()
    for outcome in ['death90', 'death_early30', 'death_late31_90']:
        assert result.loc[result[outcome].eq(1)].groupby(P).size().le(1).all(), 'Repeated death in a timing endpoint'
    return result


def window_sigma(lm, rows):
    """Mean sigma, mean mu and row count per landmark over [landmark-90d, landmark) (cross-validated discrimination)."""
    sig = np.full(len(lm), np.nan)
    mu = np.full(len(lm), np.nan)
    n = np.zeros(len(lm), int)
    rows = rows.sort_values([P, 'dialysis_date'])
    groups = {p: g for p, g in rows.groupby(P, sort=False)}
    for i, one in enumerate(lm.itertuples(index=False)):
        g = groups.get(getattr(one, P))
        if g is None:
            continue
        d = g.dialysis_date.to_numpy()
        start = np.datetime64(one.landmark_date - pd.Timedelta(days=90))
        end = np.datetime64(one.landmark_date)
        m = (d >= start) & (d < end)
        n[i] = int(m.sum())
        if n[i]:
            sig[i] = float(g.sigma_sbp.to_numpy(float)[m].mean())
            mu[i] = float(g.mu_sbp.to_numpy(float)[m].mean())
    return sig, mu, n
