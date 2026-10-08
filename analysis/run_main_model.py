"""Main model: mean sigma and 90-day mortality, mean mu as a spline (Table 2, Table S3, Figure 1, Table S2 set 4).

Input: one landmark frame per system (parquet) from analysis.landmarks. The source labels must be "CY" (Chiayi) and
"TT" (Taichung): component_statistics.make_source_configs() holds the locked covariate sets of the two systems (the
Chiayi set codes sex as `male`, the Taichung set as `sex_female`).

Usage: python -m analysis.run_main_model --cy <CY_frame.parquet> --tt <TT_frame.parquet> --out <dir>
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import analyze_surfaces


def load_frame(path, source):
    frame = pd.read_parquet(path)
    frame['landmark_date'] = pd.to_datetime(frame.landmark_date)
    frame['source'] = source
    if source == 'CY' and 'male' not in frame:
        frame['male'] = 1 - frame.sex_female
    return frame.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cy', required=True)
    p.add_argument('--tt', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    frames = {'CY': load_frame(args.cy, 'CY'), 'TT': load_frame(args.tt, 'TT')}
    checks = analyze_surfaces.analyze(frames, args.out)
    print('all checks passed:', all(c['passed'] for c in checks))


if __name__ == '__main__':
    main()
