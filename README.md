# Intradialytic blood pressure forecast dispersion and 90-day mortality

Code for the forecasting models and the mortality analyses of *When intradialytic blood
pressure becomes hard to predict: forecast dispersion and 90-day mortality*.

A neural additive model (NAMLSS) forecasts next-hour systolic and diastolic blood pressure
during hemodialysis. Each forecast has a predicted mean (μ) and a predicted SD, the forecast
dispersion (σ). The analyses average σ over the 90 days before each quarterly landmark and
relate it to death in the following 90 days.

This repository contains **no** patient data, trained model weights or result files. The
trained model files can be requested from the corresponding author.

## Installation

```bash
python -m pip install -r requirements.txt
```

The analyses were run with Python 3.14.2. `requirements.txt` pins numpy, pandas,
scikit-learn, statsmodels and torch. PyTorch was the CUDA 13.0 build (`2.9.1+cu130`):

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

`model.train_members` uses the GPU by default; pass `--device cpu` to train without one.

## Quick check

The statistical modules have self-tests on synthetic data, so they can be checked without
any patient data:

```bash
python -m analysis.component_statistics --self-test
python -m analysis.dynamic_statistics --self-test
python -m analysis.timing_statistics --self-test
python -m analysis.tt_adjustment_sets --self-test --out analysis/synthetic_statistics
python -m analysis.retention_stats
python -m analysis.component_profiles
```

The first four write their synthetic outputs to `analysis/synthetic_statistics/`, which git
ignores; the last two print their checks.

## Input data

Everything runs on derived tables, described column by column in
[`docs/data_dictionary.md`](docs/data_dictionary.md):

- the **hourly feature table** (`feature_table.parquet`): one row per hour of a dialysis
  session, with the 13 model inputs and the next-hour targets;
- the **forecast table**, written by `model.predict`;
- per-patient tables for the cohorts: observed sessions, persons and vital status.

Building these tables from hospital records is site-specific and is not part of this
repository. The data come from three hemodialysis systems: Chiayi (`CY`), Taichung (`TT`)
and Tainan, which was used only to fit Model B.

## Method in brief

- **Forecasts.** For each SBP and DBP target, one additive head gives the mean and one the
  log SD; every input passes through a smooth function of 24 radial basis functions. σ is
  bounded between 2.0 and 60 mmHg for SBP and between 1.5 and 60 mmHg for DBP. Each model
  is an ensemble of five members trained on patient-level bootstrap samples; the ensemble
  σ is the square root of the mean member variance.
- **Missing inputs.** Median imputation plus a missing indicator, fitted on the fitting
  data. An indicator that is constant there (the input was never, or always, missing) is
  held at that value when the model is applied elsewhere.
- **Patient-quarters.** Landmarks are the first day of each calendar quarter. The 90 days
  before the landmark are the exposure window; the outcome is death within the following
  90 days. A patient-quarter counts as survival only when the patient has a session with
  recorded blood pressure on or after day 90, or a death recorded after day 90, and is
  excluded otherwise.
- **Statistics.** Logistic regression with mean μ as a three-knot restricted cubic
  spline, cluster-robust (patient) Wald tests and patient-level bootstrap intervals.

## Usage

### 1. Forecasting models

Model A is fitted on Chiayi and applied to Taichung; Model B is fitted on Taichung and
Tainan and applied to Chiayi. Inputs, seeds and training settings are in
`model/spec_model_A.json` and `model/spec_model_B.json`.

```bash
# train the five ensemble members
python -m model.train_members --spec model/spec_model_A.json --table feature_table.parquet --out runs/A --device cuda

# forecasts (mu, sigma) for the application system
python -m model.predict --bundle runs/A/checkpoints --model A --table feature_table.parquet --sources Taichung --out TT_forecasts.parquet

# fitted contribution curve of each input
python -m model.contributions --bundle runs/A/checkpoints --model A --table feature_table.parquet --reference-sources Chiayi --curve-sources Taichung --out A_curves.csv
```

Model B uses the same commands with `spec_model_B.json`, applied to Chiayi, with Taichung
and Tainan as the fitting sources.

### 2. Landmark cohorts

```python
import pandas as pd
from analysis.landmarks import build_landmark_cohort

frame, audit = build_landmark_cohort(
    forecasts, observed_sessions, persons, deaths,
    first_landmark=pd.Timestamp("2018-07-01"),  # Taichung: 2018-04-01
    end=pd.Timestamp("2022-12-31"),             # Taichung: 2025-12-27
    source="CY",
)
frame.to_parquet("CY_frame.parquet", index=False)
```

The change and timing analyses also take a deaths file per system: a parquet with
`patient_token_v1` and `death_date` (NaT when no death is recorded).

### 3. Mortality analyses

```bash
# mean sigma and 90-day mortality
python -m analysis.run_main_model --cy CY_frame.parquet --tt TT_frame.parquet --out results/main

# change in sigma, and deaths during days 1-30 and 31-90
# (--start and --end are data bounds, used only for range checks of the windows)
python -m analysis.run_change_and_timing \
  --cy-frame CY_frame.parquet --cy-forecasts CY_forecasts.parquet --cy-deaths CY_deaths.parquet \
  --tt-frame TT_frame.parquet --tt-forecasts TT_forecasts.parquet --tt-deaths TT_deaths.parquet \
  --start 2015-01-01 --end 2025-12-27 --out results/change

# alternative adjustment sets (Taichung)
python -m analysis.tt_adjustment_sets --frame TT_frame.parquet --out results/tt_sets

# cross-validated discrimination with and without sigma
python -m analysis.discrimination --frame CY_frame.parquet --forecasts CY_forecasts.parquet --source CY --out results/discrimination/CY

# forecast accuracy and calibration
python -m analysis.forecast_scores --forecasts TT_forecasts.parquet --label A_application --out results/scores

# forecast error when the widest forecasts are set aside
python -m analysis.retention --forecasts CY_forecasts.parquet --features feature_table.parquet --source CY --seed 20260927 --out results/retention/CY

# which inputs make forecasts wider
python -m analysis.run_contributions --source CY --model B --bundle runs/B/checkpoints --features feature_table.parquet --reference-sources Taichung Tainan --frame CY_frame.parquet --forecasts CY_forecasts.parquet --out results/contributions
python -m analysis.run_contributions --source TT --model A --bundle runs/A/checkpoints --features feature_table.parquet --reference-sources Chiayi --frame TT_frame.parquet --forecasts TT_forecasts.parquet --out results/contributions
python -m analysis.run_contributions --combine --out results/contributions
```

### Notes on the outputs

- **17 versus 13 mmHg.** Odds ratios for a mean σ of 17 versus 13 mmHg are the
  per-1-mmHg OR and both of its CI limits raised to the fourth power. The per-1-mmHg values
  are `or`, `ci_low` and `ci_high` of the `sigma_per_1_mmhg` row in
  `prespecified_contrasts.csv` (`analysis.run_main_model`), and `or_per_unit`,
  `bootstrap_raw_ci_low` and `bootstrap_raw_ci_high` in `sigma_associations.csv`
  (`analysis.tt_adjustment_sets`). Compute from the unrounded values.
- **Input profiles.** The difference in an input for a mean σ of 17 versus 13 mmHg at the
  same predicted mean is `difference_per_sigma_residual_sd × 4 / sigma_residual_sd_mmhg`,
  both from the same row of `component_profiles.csv`.
- **Small counts.** Counts in the audit outputs (cohort flow, bootstrap failures) are
  printed as ranges of ten, and counts below ten as `<10`. The result tables are not
  affected.

## Repository map

```text
model/
├── model.py                 NAMLSS: additive mean and log-SD heads for next-hour SBP and DBP
├── losses.py                Gaussian NLL with bounded SD; mean-only warm-up loss
├── preprocessing.py         median imputation, missing indicators, standardisation
├── training.py              bootstrap samples, ridge initialisation, ensemble prediction
├── train_members.py         trains the five members of a model
├── predict.py               forecasts (mu, sigma) with a trained ensemble
├── contributions.py         fitted contribution curve of each input
├── config.py                training configuration
├── spec_model_A.json        Model A: inputs, sources, seed, training settings
└── spec_model_B.json        Model B: inputs, sources, seed, training settings
analysis/
├── landmarks.py             patient-quarters, exposure and outcome windows, covariates
├── component_statistics.py  spline, cluster-robust logistic regression, patient bootstrap
├── run_main_model.py        mean sigma and 90-day mortality
├── analyze_surfaces.py      main model fits and OR surfaces
├── run_change_and_timing.py change in sigma; early and late deaths
├── dynamic_statistics.py    change models
├── timing_statistics.py     early and late death models
├── tt_adjustment_sets.py    alternative adjustment sets (Taichung)
├── run_contributions.py     contributions to sigma and 90-day mortality
├── contribution_windows.py  per-forecast contributions averaged over the exposure window
├── component_profiles.py    clinical profile of wider forecasts
├── discrimination.py        cross-validated discrimination with and without sigma
├── forecast_scores.py       forecast accuracy and calibration
├── retention.py             selective retention of forecasts by sigma
└── retention_stats.py       retention summaries and bootstrap
docs/
└── data_dictionary.md       input tables
```

## License

MIT (see [`LICENSE`](LICENSE)).
