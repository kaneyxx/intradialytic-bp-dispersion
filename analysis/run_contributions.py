"""Groups of contributions to forecast dispersion and 90-day mortality (Table S5) and the clinical profile of wider
forecasts (Figure S4).

For each system: the forecasts of the exposure windows are scored with the model that produced them (per-input
contributions to log sigma, centred on the forecast-eligible rows of the model's fitting sources), averaged per
patient-quarter and summed within
four groups (contribution_windows); then the exploratory group models (component_statistics.analyze_components) and the
outcome-free profiles (component_profiles.run_profiles) are run.

Usage: python -m analysis.run_contributions --source CY --model B --bundle <checkpoints dir of Model B> \
           --features <feature_table.parquet> --reference-sources Taichung Tainan --frame <CY_frame.parquet> \
           --forecasts <CY_forecasts.parquet> --out <dir>   (then the same for TT with Model A; finally --combine)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model.contributions import member_centers
from model.predict import load_bundle
from model.train_members import ELIGIBLE, keys as model_keys
from . import component_profiles, component_statistics
from .contribution_windows import FEATURES, aggregate_windows, exposure_membership, row_contributions
from .landmarks import P, ROWKEY


def windows_for_source(args):
    bundle, pre = load_bundle(args.bundle, args.model)
    features = model_keys(pd.read_parquet(args.features))
    if ELIGIBLE in features:
        features = features.loc[features[ELIGIBLE].fillna(False)]
    ref = features.loc[features.site_group.isin(args.reference_sources)]
    centers = member_centers(bundle['models'], 'log_sigma', pre.transform(ref).to_numpy(np.float32), 'cpu')
    frame = pd.read_parquet(args.frame)
    frame['landmark_date'] = pd.to_datetime(frame.landmark_date)
    frame['source'] = args.source
    if args.source == 'CY' and 'male' not in frame:
        frame['male'] = 1 - frame.sex_female
    rows = pd.read_parquet(args.forecasts)
    rows['dialysis_date'] = pd.to_datetime(rows.dialysis_date)
    for c in ('source_system', 'hd_session_id', P):
        rows[c] = rows[c].astype(str)
    frame[P] = frame[P].astype(str)
    rows, _ = exposure_membership(frame.reset_index(drop=True), rows)
    raw = rows.merge(features[ROWKEY + FEATURES], on=ROWKEY, how='left', validate='one_to_one')
    assert len(raw) == len(rows)
    contrib = row_contributions(bundle, pre, raw, centers)
    rows = pd.concat([rows, contrib], axis=1)
    return aggregate_windows(args.source, frame.reset_index(drop=True), rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['CY', 'TT'])
    p.add_argument('--model', choices=['A', 'B'])
    p.add_argument('--bundle')
    p.add_argument('--features')
    p.add_argument('--reference-sources', nargs='+')
    p.add_argument('--frame')
    p.add_argument('--forecasts')
    p.add_argument('--out', required=True)
    p.add_argument('--combine', action='store_true', help='run the statistics on the window frames already written to --out')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.combine:
        result = windows_for_source(args)
        result.to_parquet(out / f'{args.source}_contribution_windows.parquet', index=False)
        print(args.source, 'contribution windows written', flush=True)
        return
    frames = {s: pd.read_parquet(out / f'{s}_contribution_windows.parquet') for s in ('CY', 'TT')}
    component_profiles.run_profiles(frames, out)
    frame = pd.concat(frames.values(), ignore_index=True)
    component_statistics.analyze_components(frame, out, component_statistics.make_source_configs())
    print('done', out)


if __name__ == '__main__':
    main()
