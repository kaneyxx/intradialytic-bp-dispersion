"""Forecast next-hour SBP and DBP with a trained ensemble: predicted mean (mu) and dispersion (sigma) per row.

Supplementary Methods S1, Application: mu is the mean of the five member means (plus the current-hour mean, because the
members predict a change); sigma is the square root of the mean of the member variances; the disagreement among member
means is reported separately (epi_var) and is not part of sigma. No calibration adjustment is applied.

Usage: python -m model.predict --bundle <checkpoints dir> --model A --table <feature_table.parquet> --out <forecasts.parquet>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr

from .train_members import IDS, TARGETS, keys, offsets
from .training import predict_ensemble


def load_bundle(checkpoints, label):
    checkpoints = Path(checkpoints)
    return joblib.load(checkpoints / f'model_{label}_bundle.joblib'), joblib.load(checkpoints / f'model_{label}_preprocessor.joblib')


def forecasts(frame, bundle, pre, device='cpu'):
    """One row per eligible hourly pair. Output columns: ids, y_*, mu_*, sigma_*, ale_var_*, epi_var_*, predictive_var_*,
    and the fixed-threshold flags used in the paper (point forecast below 90 mmHg; 90 mmHg inside the 95% interval)."""
    x = pre.transform(frame).to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite model input after preprocessing')
    cfg = bundle['config']
    pred = predict_ensemble(bundle['models'], x, cfg, device=device)
    pred['mu'] += offsets(frame, bundle['extra']['offset_fill'])
    out = frame[[c for c in IDS + ['dialysis_date'] if c in frame]].copy()
    for j, target in enumerate(['sbp', 'dbp']):
        if TARGETS[j] in frame:
            out[f'y_{target}'] = frame[TARGETS[j]].to_numpy()
        out[f'mu_{target}'] = pred['mu'][:, j]
        out[f'sigma_{target}'] = pred['sigma'][:, j]
        for key, name in [('aleatoric_var', 'ale_var'), ('epistemic_var', 'epi_var'), ('predictive_var', 'predictive_var')]:
            out[f'{name}_{target}'] = pred[key][:, j]
    out['point_nadir90'] = out.mu_sbp.lt(90)
    out['fixed196_interval_nadir90'] = (out.mu_sbp - 1.96 * out.sigma_sbp).lt(90)
    out['ordinal_p_sbp_lt90'] = ndtr((90 - out.mu_sbp) / out.sigma_sbp)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, help='checkpoints directory written by train_members')
    p.add_argument('--model', required=True, choices=['A', 'B'])
    p.add_argument('--table', required=True, help='hourly feature table (parquet)')
    p.add_argument('--sources', nargs='*', default=None, help='site_group values to forecast (default: all rows)')
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    bundle, pre = load_bundle(args.bundle, args.model)
    frame = keys(pd.read_parquet(args.table))
    if args.sources:
        frame = frame.loc[frame.site_group.isin(args.sources)]
    if 'forecast_eligible' in frame:
        frame = frame.loc[frame.forecast_eligible.fillna(False)]
    frame = frame.reset_index(drop=True)
    forecasts(frame, bundle, pre, args.device).to_parquet(args.out, index=False)
    print(f'{len(frame)} forecasts written to {args.out}')


if __name__ == '__main__':
    main()
