# Data dictionary: input tables of this repository

The repository starts from two derived tables that each hemodialysis system builds from its own records. How the tables
are built from hospital records (extraction, plausibility limits, hourly summaries, laboratory linkage, linkage of death
or treatment-status records) is described in Supplementary Methods S1 and S2 of the paper and is not part of this
repository. All identifiers are pseudonymised codes with no meaning outside the system that issued them. Two keys must
not collide across systems: the session key (`source_system`, `hd_session_id`), and `patient_token_v1` among the systems
pooled to fit one model (Taichung and Tainan for Model B); the joins and partition checks that use them stop with an
error otherwise.

## 1. Hourly feature table (`feature_table.parquet`)

One row per hour of a dialysis session (the "hourly pair": the inputs of one hour and the observed outcome of the next).

| Column | Type | Meaning |
|---|---|---|
| `site_group` | string | System label as used in the model specs (`Chiayi`, `Taichung`, `Tainan`) |
| `source_system` | string | Record source within the system (any label; part of the session key) |
| `hd_session_id` | string | Pseudonymised session identifier (unique within `source_system`) |
| `patient_token_v1` | string | Pseudonymised patient identifier |
| `hour_index` | int | Hour of the session, 0 = first hour (0 to <60 min), 1 = second hour, … |
| `dialysis_date` | date | Session date |
| `bin_index` | int | Model input: the hour of the session in which the forecast is made (= `hour_index`) |
| `sbp_mean_all_mmhg`, `dbp_mean_all_mmhg` | float | Mean systolic / diastolic pressure of the current hour, mmHg |
| `hr_mean_all_bpm` | float | Mean heart rate of the current hour, beats/min |
| `uf_rate_mean_ml_hr` | float | Mean ultrafiltration rate of the current hour, mL/h |
| `age_at_session_years` | float | Calendar year of the session minus year of birth |
| `sex_female` | 0/1 | Sex |
| `hb_value` | float | Hemoglobin, g/dL (latest value available at the time of the forecast) |
| `ca_value` | float | Total serum calcium, mg/dL (latest value available) |
| `actual_qb_mean_ml_min` | float | Mean blood flow of the current hour, mL/min |
| `dialysate_temperature_last_c` | float | Latest dialysate temperature of the current hour, °C |
| `actual_qd_last_ml_min` | float | Latest dialysate flow of the current hour, mL/min |
| `delta_sbp_from_previous_bin_mmhg` | float | Last valid SBP of the current hour minus last valid SBP of the preceding hour of the same session |
| `target_sbp_next_hour_mean_mmhg`, `target_dbp_next_hour_mean_mmhg` | float | Forecasting targets: mean SBP / DBP of the next hour |
| `forecast_eligible` | bool | Row satisfies the forecast eligibility rules (Supplementary Methods S1) |

Missing inputs are allowed (NaN); the preprocessor imputes the fitting-partition median and adds a missing indicator.

## 2. Forecast table (`forecasts.parquet`) — written by `model.predict`

One row per eligible hourly pair: the identifier columns above plus `y_sbp`, `y_dbp` (observed next-hour means),
`mu_sbp`, `mu_dbp` (predicted means), `sigma_sbp`, `sigma_dbp` (forecast dispersion), `ale_var_*`, `epi_var_*`,
`predictive_var_*` (within-member variance, between-member variance and their sum), `point_nadir90` (mu_sbp < 90),
`fixed196_interval_nadir90` (mu_sbp − 1.96 sigma_sbp < 90) and `ordinal_p_sbp_lt90` (probability that next-hour SBP is
below 90 mmHg under the forecast distribution). `dialysis_date` is carried from the feature table.

## 3. Tables for the mortality analyses (`analysis.landmarks.build_landmark_cohort`)

| Table | Columns |
|---|---|
| `observed_sessions` | `source_system`, `hd_session_id`, `patient_token_v1`, `dialysis_date`, `observed_session_sbp_min` (lowest SBP reading of the session, readings outside 40–250 mmHg disregarded), `session_mean`, `session_within_sd` (from `landmarks.session_summaries`: mean and SD of the hourly mean SBP of the session) |
| `persons` | index `patient_token_v1`; `birth_year`, `sex_female` |
| `deaths` | index `patient_token_v1`; `death_date` (NaT when no death is recorded), `last_bp` (date of the last session with recorded blood pressure), `pure_current_state_start` (bool; Taichung only: the death date is the start of the current status rather than a dated change; False elsewhere) |

The landmark frame produced from these tables has one row per analysable patient-quarter with the exposures and
covariates named in Supplementary Methods S2 and S3 (`sigma_mean90`, `mu_mean90`, `mu_sd`, `mu_p10`,
`age_at_index_years`, `sex_female`, `log1p_prior90_sessions`, `index_year`, `prior90_observed_nadir90_burden`,
`prior90_point_nadir90_burden`, `observed_between_sd`, `observed_within_sd`, `observed_mean`, `observed_nadir_p10`,
`death90`, `landmark_date`, `prior90_n_rows`).
