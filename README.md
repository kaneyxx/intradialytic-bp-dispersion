# Forecast dispersion and 90-day mortality in hemodialysis — code

Code for the forecasting models and the mortality analyses of *When intradialytic blood pressure becomes hard to
predict: forecast dispersion and 90-day mortality* (Nephrology Dialysis Transplantation, submitted). Everything here runs
on two derived tables, the hourly feature table and the forecast table (see `docs/data_dictionary.md`); the construction
of those tables from each hospital's records is described in Supplementary Methods S1 and S2 and is not part of this
repository. No patient data, model weights or result files are included; the trained model files can be requested from the
corresponding author.

## Layout

| Path | What it does | Paper |
|---|---|---|
| `model/model.py` | NAMLSS: two additive heads per outcome (mean and log SD of next-hour SBP and DBP), one smooth function of 24 radial basis functions per input; `contributions()` returns the per-input terms | Methods S1, Model structure |
| `model/losses.py` | Gaussian negative log-likelihood with the SD bounded between 2.0 and 60 mmHg; mean-only warm-up loss | Methods S1 |
| `model/preprocessing.py` | Median imputation, missing indicators, standardisation (fitted on the fitting partition); support guard for indicators that are constant in the fitting data | Methods S1, Missing data |
| `model/training.py` | Patient-level bootstrap samples, ridge initialisation, ensemble prediction (sigma = square root of the mean member variance) | Methods S1 |
| `model/train_members.py` | Trains the five members of Model A or Model B: patient split (about 20% development), warm-up, AdamW on the Gaussian NLL, early stopping on the development NLL (patience 100 epochs, cap 1,500), bundle | Methods S1, Training; Figure S5 |
| `model/spec_model_A.json`, `spec_model_B.json` | The 13 inputs, fitting sources, seeds, optimiser settings and stopping rule of the two models | Methods S1, Tables S7–S9 |
| `model/predict.py` | Forecasts (mu, sigma) for every eligible hourly pair with a trained bundle | Methods S1, Application |
| `model/contributions.py` | Fitted contribution curves of each input (centred, member mean and range) | Figure 2B, Figure S3 |
| `analysis/landmarks.py` | Patient-quarters: quarterly landmarks, 90-day exposure and outcome windows, exposures and covariates, vital-status rules; earlier window for the change analysis; death intervals for the timing analysis | Methods S2, Table S1 |
| `analysis/component_statistics.py` | Shared statistics: three-knot restricted cubic spline, logistic regression with cluster-robust Wald tests, patient-level bootstrap, group models | Methods S3 |
| `analysis/analyze_surfaces.py`, `run_main_model.py` | Main model (mean sigma linear, mean mu spline, locked covariates); contrasts for 17 versus 13 mmHg, per 1 mmHg and per SD; mu contrasts; OR surfaces | Table 2, Table S3, Figure 1, Table S2 set 4 |
| `analysis/dynamic_statistics.py`, `timing_statistics.py`, `run_change_and_timing.py` | Change model (Δσ with the earlier-window σ and change in μ) and deaths during days 1–30 and 31–90 | Table 2, Table S4 |
| `analysis/tt_adjustment_sets.py` | Taichung models of the dated analysis plan (sets 1, 5 and 6) | Table S2 |
| `analysis/contribution_windows.py`, `component_profiles.py`, `run_contributions.py` | Per-forecast contributions to log sigma averaged over the exposure window and summed within four groups; group models; outcome-free profiles | Table S5, Figure S4 |
| `analysis/discrimination.py` | Five-fold patient-level cross-validation of the basic and extended models with and without sigma (ridge logistic regression, C = 1) | Table S6 |
| `analysis/forecast_scores.py` | Forecast accuracy and calibration with patient-level bootstrap CIs | Figure S2 C–D, Results |
| `analysis/retention.py`, `retention_stats.py` | Selective retention of forecasts by sigma and by the preceding-hour SBP change; retained versus flagged portions | Figure S2 A–D, F |

## Running

```
pip install -r requirements.txt          # Python 3.14.2 was used; exact package versions are in requirements.txt
# models
python -m model.train_members --spec model/spec_model_A.json --table feature_table.parquet --out runs/A --device cuda
python -m model.predict --bundle runs/A/checkpoints --model A --table feature_table.parquet --sources Taichung --out TT_forecasts.parquet
python -m model.contributions --bundle runs/A/checkpoints --model A --table feature_table.parquet --reference-sources Chiayi --curve-sources Taichung --out A_curves.csv
# cohorts (Python: analysis.landmarks.build_landmark_cohort on the forecasts, observed sessions, persons and deaths)
#   first landmark and outcome end: Chiayi 2018-07-01 and 2022-12-31; Taichung 2018-04-01 and 2025-12-27
# analyses
python -m analysis.run_main_model --cy CY_frame.parquet --tt TT_frame.parquet --out results/main
# --start and --end of the change analysis are data bounds used only for range checks of the windows
python -m analysis.run_change_and_timing --cy-frame CY_frame.parquet --cy-forecasts CY_forecasts.parquet --cy-deaths CY_deaths.parquet --tt-frame TT_frame.parquet --tt-forecasts TT_forecasts.parquet --tt-deaths TT_deaths.parquet --start 2015-01-01 --end 2025-12-27 --out results/change
python -m analysis.tt_adjustment_sets --frame TT_frame.parquet --out results/tt_sets
python -m analysis.discrimination --frame CY_frame.parquet --forecasts CY_forecasts.parquet --source CY --out results/discrimination/CY
python -m analysis.forecast_scores --forecasts TT_forecasts.parquet --label A_application --out results/scores
python -m analysis.retention --forecasts CY_forecasts.parquet --features feature_table.parquet --source CY --seed 20260927 --out results/retention/CY
python -m analysis.run_contributions --source CY --model B --bundle runs/B/checkpoints --features feature_table.parquet --reference-sources Taichung Tainan --frame CY_frame.parquet --forecasts CY_forecasts.parquet --out results/contributions
python -m analysis.run_contributions --source TT --model A --bundle runs/A/checkpoints --features feature_table.parquet --reference-sources Chiayi --frame TT_frame.parquet --forecasts TT_forecasts.parquet --out results/contributions
python -m analysis.run_contributions --combine --out results/contributions
```

Sources are labelled `CY` (Chiayi) and `TT` (Taichung) inside the analysis modules, which hold the locked covariate sets
of the two systems; `Tainan` contributed only to the development of Model B. Figures of the paper were drawn from the CSV
outputs of these modules. Counts in the audit outputs (cohort flow, bootstrap failures) are printed as ranges of ten, and
counts below ten as "<10"; the result tables are unaffected.

## Licence

MIT (see `LICENSE`).
