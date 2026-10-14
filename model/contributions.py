"""Fitted contribution of each input to the forecast (Figure 2B, Figure S3).

Because both heads of the model are additive, the contribution of an input is its learned function evaluated on a grid
of values while every other input is held at its fitting-partition median. Contributions are centred on each member's
mean contribution over a reference set of rows (all forecast-eligible rows of --reference-sources, that is the fitting
and development partitions of the model's fitting sources), so that the curves of one model are comparable
with each other; the numeric term and the missing-indicator term of an input are added together. Curves are summarised
as the mean across the five members and the member range.

Usage: python -m model.contributions --bundle <checkpoints dir> --model A --table <feature_table.parquet> \
           --reference-sources Chiayi --curve-sources Taichung --out curves.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .predict import load_bundle
from .train_members import ELIGIBLE, keys

HEADS = {'mu': 'mu_sbp', 'log_sigma': 'log_sigma_sbp'}
BATCH = 8192


def member_centers(models, head, xref, device):
    """Per member, the mean contribution of every encoded column over the reference rows."""
    nf = xref.shape[1] // 2
    centers = []
    for model in models:
        model.to(device).eval()
        total = np.zeros(2 * nf)
        with torch.inference_mode():
            for s in range(0, len(xref), BATCH):
                xx = torch.as_tensor(xref[s:s + BATCH], device=device)
                total += getattr(model, HEADS[head]).contributions(xx).cpu().numpy().sum(axis=0, dtype=np.float64)
        centers.append(total / len(xref))
    return np.stack(centers)


def grid_for(feature, frame):
    observed = frame[feature].dropna().to_numpy(float)
    if len(observed):
        unique = np.unique(observed)
        grid = unique if len(unique) <= 101 else np.linspace(np.quantile(observed, .01), np.quantile(observed, .99), 101)
    else:
        grid = np.array([])
    return np.r_[grid, np.nan]                                    # the last point is the missing state


def grouped_curve(models, head, pre, features, feature, j, values, centers, device):
    nf = len(features)
    raw_grid = pd.DataFrame({f: np.repeat(pre.medians_[f], len(values)) for f in features})
    raw_grid[feature] = values
    xx = torch.as_tensor(pre.transform(raw_grid).to_numpy(np.float32), device=device)
    member = []
    with torch.inference_mode():
        for m, model in enumerate(models):
            cc = getattr(model, HEADS[head]).contributions(xx).cpu().numpy()
            member.append(cc[:, j] + cc[:, j + nf] - centers[m, j] - centers[m, j + nf])
    return np.stack(member)


def curve_rows(models, head, pre, features, feature, j, grid, centers, device, model_name):
    values = grouped_curve(models, head, pre, features, feature, j, grid, centers, device)
    return [dict(model=model_name, head=head, feature=feature, raw_value=v, state='missing' if np.isnan(v) else 'observed',
                 ensemble_mean_centered=float(values[:, k].mean()), member_min=float(values[:, k].min()),
                 member_max=float(values[:, k].max())) for k, v in enumerate(grid)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--model', required=True, choices=['A', 'B'])
    p.add_argument('--table', required=True)
    p.add_argument('--reference-sources', nargs='+', required=True, help='site_group values of the fitting data (centring)')
    p.add_argument('--curve-sources', nargs='+', required=True, help='site_group values whose observed range sets the grid')
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    bundle, pre = load_bundle(args.bundle, args.model)
    features = bundle['extra']['feature_columns']
    frame = keys(pd.read_parquet(args.table))
    if ELIGIBLE in frame:
        frame = frame.loc[frame[ELIGIBLE].fillna(False)]
    ref = frame.loc[frame.site_group.isin(args.reference_sources)]
    cur = frame.loc[frame.site_group.isin(args.curve_sources)]
    xref = pre.transform(ref).to_numpy(np.float32)
    rows = []
    for head in HEADS:
        centers = member_centers(bundle['models'], head, xref, args.device)
        for j, feature in enumerate(features):
            rows += curve_rows(bundle['models'], head, pre, features, feature, j, grid_for(feature, cur), centers, args.device, args.model)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'{len(rows)} curve points written to {args.out}')


if __name__ == '__main__':
    main()
