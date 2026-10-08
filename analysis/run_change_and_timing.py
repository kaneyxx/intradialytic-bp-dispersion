"""Change in dispersion and 90-day mortality (Table 2, change rows; Table S4 timing of death).

Steps: the earlier window (days -180 to -91) is derived from the forecasts (landmarks.add_prior_history; at least four
earlier sessions), the change model is fitted (dynamic_statistics), and deaths are split into days 1-30 and 31-90
(landmarks.attach_intervals, timing_statistics) using the change model's knots and scales.

Usage: python -m analysis.run_change_and_timing --cy-frame ... --cy-forecasts ... --cy-deaths ... --tt-frame ... \
           --tt-forecasts ... --tt-deaths ... --start 2015-01-01 --end 2025-12-27 --out <dir>
deaths: parquet with patient_token_v1 and death_date (NaT when none).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import dynamic_statistics, timing_statistics
from .landmarks import P, add_prior_history, attach_intervals


def prepare(frame_path, forecasts_path, deaths_path, source, start, end):
    frame = pd.read_parquet(frame_path)
    frame['landmark_date'] = pd.to_datetime(frame.landmark_date)
    frame['source'] = source
    if source == 'CY' and 'male' not in frame:
        frame['male'] = 1 - frame.sex_female
    rows = pd.read_parquet(forecasts_path)
    rows['dialysis_date'] = pd.to_datetime(rows.dialysis_date)
    for c in ('source_system', 'hd_session_id', P):
        rows[c] = rows[c].astype(str)
    frame[P] = frame[P].astype(str)
    dynamic, audit = add_prior_history(frame.reset_index(drop=True), rows, pd.Timestamp(start), pd.Timestamp(end))
    deaths = pd.read_parquet(deaths_path)
    deaths['death_date'] = pd.to_datetime(deaths.death_date)
    lookup = deaths.set_index(deaths[P].astype(str)).death_date
    dates = dynamic[P].map(lookup)
    timing = attach_intervals(dynamic, dates)
    return dynamic, timing, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for s in ('cy', 'tt'):
        p.add_argument(f'--{s}-frame', required=True)
        p.add_argument(f'--{s}-forecasts', required=True)
        p.add_argument(f'--{s}-deaths', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    dynamic_frames, timing_frames = [], []
    for source in ('CY', 'TT'):
        a = vars(args)
        d, t, audit = prepare(a[f'{source.lower()}_frame'], a[f'{source.lower()}_forecasts'], a[f'{source.lower()}_deaths'], source, args.start, args.end)
        dynamic_frames.append(d)
        timing_frames.append(t)
        print(source, audit['flow'], flush=True)
    dynamic = pd.concat(dynamic_frames, ignore_index=True)
    timing = pd.concat(timing_frames, ignore_index=True)
    dynamic_statistics.analyze_dynamics(dynamic, out / 'change_model')
    reference = timing_statistics.load_reference(out / 'change_model')
    timing_statistics.analyze_timing(timing, out / 'timing', reference=reference)
    print('done', out)


if __name__ == '__main__':
    main()
