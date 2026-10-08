"""Taichung adjustment sets of the dated analysis plan (Table S2); patient data stay in memory.

Usage: python -m analysis.tt_adjustment_sets --frame <TT_frame.parquet> --out <dir>
M1 is Table S2 set 1, M2 the main model (set 4) and M3 set 5; M1 refitted after excluding patients whose death date
came only from a status-start record is set 6. Only aggregate tables and JSON summaries are saved.
``--self-test`` exercises estimation and cluster bootstrap with synthetic data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

for thread_option in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_option] = "1"

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from threadpoolctl import threadpool_limits

OUT = Path('.')          # set from the command line
SEED = 20260926
REPS = 300
TARGET = "sigma_mean90"
PATIENT = "patient_token_v1"
BASE = ["age_at_index_years", "sex_female", "log1p_prior90_sessions", "index_year",
        "prior90_observed_nadir90_burden", "prior90_point_nadir90_burden"]
BPV = ["observed_between_sd", "observed_within_sd"]
EXTRA = ["mu_sd", "mu_p10"]
SUPPRESSED = "suppressed"


class Nonestimable(ValueError):
    """Safe, aggregate-only explanation of an estimation problem."""


def dump(value, path):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (np.ndarray, tuple)):
            return list(obj)
        if isinstance(obj, (datetime, Path)):
            return str(obj)
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     default=convert, allow_nan=False), encoding="utf-8")


def rcs3(x, knots):
    t1, t2, t3 = np.asarray(knots, float)
    if not t1 < t2 < t3:
        raise Nonestimable("coincident_mean_spline_knots")
    cube = lambda v: np.maximum(v, 0.0) ** 3
    return (cube(x - t1) - cube(x - t2) * (t3 - t1) / (t3 - t2)
            + cube(x - t3) * (t2 - t1) / (t3 - t2)) / (t3 - t1) ** 2


def make_design(frame, model, knots):
    cols = [TARGET, "mu_mean90"] + BASE
    if model in ("M2", "M3"):
        cols += BPV
    if model == "M3":
        cols += EXTRA
    raw = frame[cols].to_numpy(float)
    if model in ("M2", "M3"):
        raw = np.column_stack([raw, rcs3(frame.mu_mean90.to_numpy(float), knots)])
        cols += ["mu_mean90_rcs_nonlinear"]
    return raw, cols


def fit_array(raw, y, names):
    raw = np.asarray(raw, float)
    y = np.asarray(y, int)
    finite = np.isfinite(raw)
    if np.any(~finite.any(axis=0)):
        raise Nonestimable("all_missing_covariate:" + ",".join(np.asarray(names)[~finite.any(axis=0)]))
    for name in (TARGET, "mu_mean90"):
        if name in names and not finite[:, names.index(name)].all():
            raise Nonestimable("missing_primary_exposure")
    med = np.nanmedian(np.where(finite, raw, np.nan), axis=0)
    filled = np.where(finite, raw, med)
    means, scales = filled.mean(axis=0), filled.std(axis=0, ddof=0)
    keep = scales > 1e-14
    if not keep[names.index(TARGET)]:
        raise Nonestimable("constant_sigma")
    x = (filled[:, keep] - means[keep]) / scales[keep]
    z = np.column_stack([np.ones(len(x)), x])
    if np.unique(y).size != 2:
        raise Nonestimable("single_outcome_class")
    if len(y) <= z.shape[1] or np.linalg.matrix_rank(z) != z.shape[1]:
        raise Nonestimable("rank_deficient_design")
    if np.linalg.cond(z) > 1e12:
        raise Nonestimable("numerically_degenerate_design")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000, tol=1e-10)
        model.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise Nonestimable("logistic_nonconvergence")
    beta = np.r_[model.intercept_, model.coef_[0]]
    if not np.isfinite(beta).all():
        raise Nonestimable("nonfinite_coefficient")
    return {"beta": beta, "z": z, "prob": model.predict_proba(x)[:, 1],
            "names": ["intercept"] + list(np.asarray(names)[keep]),
            "scales": np.r_[1.0, scales[keep]], "medians": med,
            "dropped": list(np.asarray(names)[~keep]),
            "condition": float(np.linalg.cond(z)), "iterations": int(model.n_iter_[0])}


def full_fit(raw, y, codes, names):
    result = fit_array(raw, y, names)
    z, p = result["z"], result["prob"]
    n, k = z.shape
    groups = int(codes.max()) + 1
    if groups < 2:
        raise Nonestimable("insufficient_patient_clusters")
    bread = np.linalg.inv((z.T * (p * (1 - p))) @ z + np.diag([0.] + [1e-6] * (k - 1)))
    score = np.zeros((groups, k))
    np.add.at(score, codes, z * (y - p)[:, None])
    correction = groups / (groups - 1) * (n - 1) / (n - k)
    cov = bread @ (score.T @ score) @ bread * correction
    se = np.sqrt(np.diag(cov))
    if not np.isfinite(se).all() or np.any(se <= 0):
        raise Nonestimable("invalid_cluster_covariance")
    result.update(cov=cov, se=se, correction=float(correction))
    with warnings.catch_warnings(record=True):
        verified = sm.GLM(y, z, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": codes, "use_correction": True},
            tol=1e-12, maxiter=300)
    i = result["names"].index(TARGET)
    delta_beta = float(np.max(np.abs(verified.params - result["beta"])))
    delta_se = float(abs(verified.bse[i] - se[i]))
    check = {"max_coefficient_difference": delta_beta,
             "sigma_cluster_se_difference": delta_se,
             "statsmodels_converged": bool(verified.converged),
             "passed": bool(verified.converged and delta_beta < 1e-4 and delta_se < 1e-5)}
    result["verification"] = check
    return result


def sample_patient_indices(codes, rng):
    """Sample G patients G times; repeated selection repeats ALL their windows."""
    g = int(codes.max()) + 1
    by_patient = [np.flatnonzero(codes == group) for group in range(g)]
    chosen = rng.integers(0, g, size=g)
    return np.concatenate([by_patient[group] for group in chosen]), chosen


def bootstrap(raws, y, codes, model_names, reps=REPS):
    rng = np.random.default_rng(SEED)
    by_patient = [np.flatnonzero(codes == group) for group in range(int(codes.max()) + 1)]
    samples = {key: [] for key in raws}
    failures = {key: Counter() for key in raws}
    for rep in range(reps):
        chosen = rng.integers(0, len(by_patient), size=len(by_patient))
        ix = np.concatenate([by_patient[group] for group in chosen])
        for key, raw in raws.items():
            try:
                fit = fit_array(raw[ix], y[ix], model_names[key])
                samples[key].append({name: (float(b), float(b / s))
                                     for name, b, s in zip(fit["names"], fit["beta"], fit["scales"])})
            except Nonestimable as exc:
                failures[key][str(exc)] += 1
            except Exception as exc:
                failures[key][type(exc).__name__] += 1
        if reps >= 100 and (rep + 1) % 50 == 0:
            print(f"Cluster bootstrap: {rep + 1}/{reps} completed.", flush=True)
    return samples, failures


def protected_counts(counts):
    """Complementary suppression within one additive family."""
    if any(0 < int(value) < 10 for value in counts.values()):
        return {key: SUPPRESSED for key in counts}
    return {key: int(value) for key, value in counts.items()}


def descriptive_table(frame):
    records = []
    for col in [TARGET, "mu_mean90"] + BASE + BPV + EXTRA:
        values = frame[col].to_numpy(float)
        valid = values[np.isfinite(values)]
        counts = protected_counts({"windows": len(values), "available": len(valid),
                                   "missing": len(values) - len(valid)})
        rec = {"variable": col, **counts}
        if len(valid) >= 10:
            rec.update(mean=float(valid.mean()), sd=float(valid.std(ddof=0)),
                       p10=float(np.quantile(valid, .10)), median=float(np.median(valid)),
                       p90=float(np.quantile(valid, .90)))
        else:
            rec.update({k: SUPPRESSED for k in ["mean", "sd", "p10", "median", "p90"]})
        if col == "sex_female":
            # No imputed sex is treated as observed. Suppress the entire binary family if needed.
            sex_counts = protected_counts({"observed_female": int((valid == 1).sum()),
                                           "observed_male": int((valid == 0).sum()),
                                           "observed_missing": len(values) - len(valid),
                                           "observed_total": len(values)})
            rec.update(sex_counts)
            if SUPPRESSED in sex_counts.values():
                rec.update({k: SUPPRESSED for k in ["mean", "sd", "p10", "median", "p90"]})
        records.append(rec)
    return pd.DataFrame(records)


def analyze_subset(frame, label, knots, reps=REPS, models=("M1", "M2", "M3")):
    y = frame.death90.to_numpy(int)
    codes, _ = pd.factorize(frame[PATIENT], sort=True)
    rows, all_terms, checks = [], [], []
    raws, names, fits = {}, {}, {}
    for key in models:
        try:
            if min(int(y.sum()), len(y) - int(y.sum())) < 10:
                raise Nonestimable("fewer_than_10_events_or_nonevents")
            raw, cols = make_design(frame, key, knots)
            fit = full_fit(raw, y, codes, cols)
            raws[key], names[key], fits[key] = raw, cols, fit
            checks.append({"analysis": label, "model": key, **fit["verification"]})
        except Nonestimable as exc:
            rows.append({"analysis": label, "model": key, "status": str(exc)})
        except Exception as exc:
            rows.append({"analysis": label, "model": key, "status": type(exc).__name__})
    samples, failures = bootstrap(raws, y, codes, names, reps=reps) if fits else ({}, {})
    for key, fit in fits.items():
        for j, term in enumerate(fit["names"]):
            if term == "intercept":
                continue
            beta, se, scale = float(fit["beta"][j]), float(fit["se"][j]), float(fit["scales"][j])
            draws = np.asarray([draw[term] for draw in samples[key] if term in draw], float)
            rec = {"analysis": label, "model": key, "term": term,
                   "status": "estimated" if fit["verification"]["passed"] else "verification_mismatch",
                   "beta_standardized": beta, "or_per_sd": float(np.exp(beta)),
                   "exposure_sd": scale, "beta_raw": beta / scale, "or_per_unit": float(np.exp(beta / scale)),
                   "cluster_se": se, "cluster_se_raw": se / scale,
                   "cluster_wald_p": float(2 * norm.sf(abs(beta / se))),
                   "cluster_ci_low": float(np.exp(beta - 1.96 * se)),
                   "cluster_ci_high": float(np.exp(beta + 1.96 * se)),
                   "bootstrap_valid": len(draws), "bootstrap_requested": reps,
                   "bootstrap_failure_reasons": json.dumps(dict(failures[key]), sort_keys=True),
                   "dropped_constant_covariates": ";".join(fit["dropped"]),
                   "design_condition": fit["condition"], "design_parameters": len(fit["names"]),
                   "events_per_parameter": float(y.sum() / len(fit["names"])),
                   "finite_sample_correction": fit["correction"]}
            if len(draws) >= 2:
                lo, hi = np.quantile(draws, [.025, .975], axis=0)
                rec.update(bootstrap_ci_low=float(np.exp(lo[0])), bootstrap_ci_high=float(np.exp(hi[0])),
                           bootstrap_raw_ci_low=float(np.exp(lo[1])), bootstrap_raw_ci_high=float(np.exp(hi[1])))
            if len(draws) < .95 * reps:
                rec["status"] += ";insufficient_valid_bootstrap"
            if rec["events_per_parameter"] < 10:
                rec["sparse_event_caution"] = True
            else:
                rec["sparse_event_caution"] = False
            all_terms.append(rec)
            if term == TARGET:
                rows.append(rec.copy())
    counts = {"windows": len(frame), "patients": int(frame[PATIENT].nunique()),
              "event_windows": int(y.sum()), "nonevent_windows": int(len(y) - y.sum()),
              "event_patients": int(frame.loc[frame.death90.eq(1), PATIENT].nunique())}
    return rows, all_terms, checks, counts


def fit_and_report(frame, audit):
    required = {PATIENT, "landmark_date", "death90", TARGET, "mu_mean90", "pure_current_state_start", *BASE, *BPV, *EXTRA}
    if not required.issubset(frame.columns):
        raise Nonestimable("required_analysis_columns_missing")
    if frame.empty:
        raise Nonestimable("empty_analysis_cohort")
    if frame[PATIENT].isna().any() or frame.landmark_date.isna().any():
        raise Nonestimable("missing_patient_or_landmark")
    if frame.duplicated([PATIENT, "landmark_date"]).any():
        raise Nonestimable("duplicate_patient_landmark")
    if not frame.death90.isin([0, 1]).all():
        raise Nonestimable("invalid_or_unknown_binary_outcome")
    if not np.isfinite(frame[[TARGET, "mu_mean90"]].to_numpy(float)).all():
        raise Nonestimable("missing_primary_exposure")
    if frame.pure_current_state_start.isna().any():
        raise Nonestimable("unknown_current_state_start_sensitivity_flag")
    knots = np.quantile(frame.mu_mean90.to_numpy(float), [.10, .50, .90]).tolist()
    lock = {"created_utc_before_fitting": datetime.now(timezone.utc).isoformat(), "seed": SEED,
            "bootstrap_replicates": REPS, "mean_rcs_knots_10_50_90": knots,
            "knots_distinct": bool(np.all(np.diff(knots) > 0)),
            "software": {"numpy": np.__version__, "pandas": pd.__version__},
            "endpoint_status": "death dates and status-start flags as supplied in the deaths table (Supplementary Methods S2)",
            "primary_test": "M1 sigma_mean90", "no_patient_level_outputs": True}
    dump(lock, OUT / "analysis_lock.json")
    dump(audit, OUT / "cohort_audit.json")
    descriptive_table(frame).to_csv(OUT / "variable_summary.csv", index=False)
    all_rows, all_terms, all_checks, count_rows = [], [], [], []
    subsets = [("main", frame)]
    if frame.pure_current_state_start.astype(bool).any():
        excluded_patients = frame.loc[frame.pure_current_state_start.astype(bool), PATIENT].unique()
        sensitivity = frame.loc[~frame[PATIENT].isin(excluded_patients)]
        if not sensitivity.empty:
            subsets.append(("exclude_pure_current_state_start", sensitivity))
        else:
            all_rows.append({"analysis": "exclude_pure_current_state_start", "model": "M1",
                             "status": "empty_sensitivity_cohort"})
    else:
        all_rows.append({"analysis": "exclude_pure_current_state_start", "model": "M1",
                         "status": "not_applicable_no_exposed_patients"})
    with threadpool_limits(limits=1):
        for label, data in subsets:
            models = ("M1", "M2", "M3") if label == "main" else ("M1",)
            print(f"Fitting {label}: fixed {'/'.join(models)}; aggregate outputs only.", flush=True)
            rows, terms, checks, counts = analyze_subset(data, label, knots, models=models)
            all_rows += rows
            all_terms += terms
            all_checks += checks
            count_rows.append({"analysis": label, **counts})
    count_keys = [key for key in count_rows[0] if key != "analysis"]
    # Shared counts are suppressed across main and sensitivity when a difference is small.
    unsafe = any(0 < row[key] < 10 for row in count_rows for key in count_keys)
    if len(count_rows) > 1:
        unsafe |= any(0 < count_rows[0][key] - count_rows[1][key] < 10 for key in count_keys)
    if unsafe:
        for row in count_rows:
            row.update({key: SUPPRESSED for key in count_keys})
        # EPP is an algebraic disclosure of the event count; suppress with event totals.
        for row in all_rows + all_terms:
            row.pop("events_per_parameter", None)
            row.pop("finite_sample_correction", None)
    result = pd.DataFrame(all_rows).sort_values(["analysis", "model"])
    result.to_csv(OUT / "sigma_associations.csv", index=False)
    pd.DataFrame(all_terms).to_csv(OUT / "all_coefficients.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUT / "analysis_counts.csv", index=False)
    dump(all_checks, OUT / "statsmodels_verification.json")
    artifacts = ["sigma_associations.csv", "all_coefficients.csv", "analysis_counts.csv",
                 "variable_summary.csv", "statsmodels_verification.json", "analysis_lock.json", "cohort_audit.json"]
    dump({"all_result_files_written": True, "finished_utc": datetime.now(timezone.utc).isoformat(),
          "statsmodels_all_passed": bool(all_checks and all(c["passed"] for c in all_checks)),
          "artifact_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in artifacts}},
         OUT / "ANALYSIS_COMPLETION.json")
    print("Analysis completed; aggregate results and verification written.", flush=True)
    return result


def self_test():
    rng = np.random.default_rng(20260926)
    patients = 200
    codes = np.repeat(np.arange(patients), 4)
    n = len(codes)
    frame = pd.DataFrame({PATIENT: np.asarray([f"synthetic-{i}" for i in codes]),
                          "mu_mean90": rng.normal(130, 12, n), TARGET: rng.normal(15, 3, n)})
    for col in BASE + BPV + EXTRA:
        frame[col] = rng.normal(size=n)
    frame["sex_female"] = rng.binomial(1, .5, n)
    frame.loc[::17, "age_at_index_years"] = np.nan
    eta = -1.3 + .25 * (frame[TARGET].to_numpy() - 15) / 3 + .01 * (frame.mu_mean90 - 130)
    frame["death90"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    knots = np.quantile(frame.mu_mean90, [.1, .5, .9])
    with threadpool_limits(limits=1):
        rows, terms, checks, counts = analyze_subset(frame, "synthetic", knots, reps=12)
    assert len(rows) == 3 and len(checks) == 3
    assert all(check["passed"] for check in checks)
    assert all(row["bootstrap_valid"] == 12 for row in rows)
    ix, chosen = sample_patient_indices(codes, np.random.default_rng(31))
    assert np.array_equal(np.bincount(codes[ix], minlength=patients), 4 * np.bincount(chosen, minlength=patients))
    assert all(value == SUPPRESSED for value in protected_counts({"total": 100, "yes": 7, "no": 93}).values())
    assert not np.array_equal(knots, np.quantile(frame.mu_mean90.iloc[ix], [.1, .5, .9]))
    try:
        rcs3(np.ones(3), [1, 1, 2])
        raise AssertionError("Repeated knots accepted")
    except Nonestimable:
        pass
    raw, cols = make_design(frame, "M1", knots)
    first_ix, _ = sample_patient_indices(codes, np.random.default_rng(SEED))
    first_fit = fit_array(raw[first_ix], frame.death90.to_numpy()[first_ix], cols)
    draws, _ = bootstrap({"M1": raw}, frame.death90.to_numpy(), codes, {"M1": cols}, reps=2)
    target_j = first_fit["names"].index(TARGET)
    assert draws["M1"][0][TARGET][1] == first_fit["beta"][target_j] / first_fit["scales"][target_j]
    raw[:, cols.index("age_at_index_years")] = np.nan
    try:
        fit_array(raw, frame.death90.to_numpy(), cols)
        raise AssertionError("All-missing covariate accepted")
    except Nonestimable:
        pass
    dump({"synthetic_only": True, "assertions_passed": True,
          "checks": ["M1_M2_M3", "independent_coefficients_and_cluster_SE", "patient_multiplicity",
                     "per_replicate_raw_unit_conversion", "small_cell_complementary_suppression",
                     "coincident_knots_and_all_missing_covariate_rejection"],
          "independent_verification": checks},
         OUT / "synthetic_verification.json")
    print("Synthetic checks passed: M1–M3, independent cluster SE, patient multiplicity, raw-unit CI, suppression, and nonestimability.", flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description="Taichung adjustment sets (Table S2 sets 1, 5 and 6; models M1-M3) on a landmark frame")
    parser.add_argument("--frame", help="landmark frame of the Taichung system (parquet) from analysis.landmarks")
    parser.add_argument("--out", default=".")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test()
        return
    frame = pd.read_parquet(args.frame)
    frame["landmark_date"] = pd.to_datetime(frame.landmark_date)
    fit_and_report(frame, {"source": "landmark frame supplied on the command line"})


if __name__ == "__main__":
    main()
