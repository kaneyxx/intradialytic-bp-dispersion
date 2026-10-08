"""Exploratory mortality associations of fixed groups of log-sigma contributions.

Patient-level frames are supplied in memory and are never serialized or printed.
No model selection is performed. Each group's baseline-adjusted association is
primary exploratory; all four groups together are a secondary redundancy check.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import warnings

for _thread_option in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_option] = "1"

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from threadpoolctl import threadpool_limits

PATIENT = "patient_token_v1"
GROUPS = ("group_bp", "group_treatment", "group_demographic", "group_labs")
SOURCES = ("CY", "TT")
SEED = 20260926
REPS = 300
SUPPRESSED = "suppressed"
TT_BASELINE = ("age_at_index_years", "sex_female", "log1p_prior90_sessions", "index_year",
               "prior90_observed_nadir90_burden", "prior90_point_nadir90_burden",
               "observed_between_sd", "observed_within_sd")
CY_BASELINE = ("prior90_observed_nadir90_burden", "prior90_point_nadir90_burden",
               "age_at_index_years", "male", "log1p_prior90_sessions", "index_year",
               "observed_between_sd", "observed_within_sd")


def make_source_configs():
    """Main-model covariate sets (Table 2 footnote) of the two systems; Chiayi codes sex as `male`."""
    return {source: {"baseline_columns": list(baseline), "mean_column": "mu_mean90",
                     "mean_form": "rcs3"}
            for source, baseline in (("CY", CY_BASELINE), ("TT", TT_BASELINE))}


class Nonestimable(ValueError):
    """Exceptions contain only hard-coded aggregate diagnostic reasons."""


def dump_json(value, path):
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


def exp_or(log_or):
    """Retain the log coefficient, but do not output infinities as finite ORs."""
    if not np.isfinite(log_or) or abs(log_or) > 700:
        return np.nan
    return float(np.exp(log_or))


def rcs3(x, knots):
    t1, t2, t3 = np.asarray(knots, float)
    if not t1 < t2 < t3:
        raise Nonestimable("coincident_mean_spline_knots")
    cube = lambda v: np.maximum(v, 0.0) ** 3
    return (cube(x - t1) - cube(x - t2) * (t3 - t1) / (t3 - t2)
            + cube(x - t3) * (t2 - t1) / (t3 - t2)) / (t3 - t1) ** 2


def _configuration(source, source_configs):
    if source not in source_configs:
        raise Nonestimable("missing_source_baseline_configuration")
    config = source_configs[source]
    baseline = list(config["baseline_columns"])
    mean_col = config.get("mean_column", "mu_mean90")
    if not baseline or len(baseline) != len(set(baseline)):
        raise Nonestimable("invalid_baseline_configuration")
    if mean_col in baseline or set(baseline).intersection(GROUPS):
        raise Nonestimable("duplicate_or_exposure_in_baseline")
    if config.get("mean_form", "rcs3") != "rcs3":
        raise Nonestimable("unsupported_mean_form")
    return baseline, mean_col


def make_design(frame, exposure_names, baseline, mean_col, knots):
    names = list(exposure_names) + [mean_col] + list(baseline)
    raw = frame[names].to_numpy(dtype=float, na_value=np.nan)
    raw = np.column_stack([raw, rcs3(frame[mean_col].to_numpy(float), knots)])
    return raw, names + [mean_col + "_rcs_nonlinear"]


def prepare_design(raw, names, exposure_names, mean_col):
    raw = np.asarray(raw, float)
    finite = np.isfinite(raw)
    if np.any(~finite.any(axis=0)):
        raise Nonestimable("all_missing_covariate")
    for name in list(exposure_names) + [mean_col]:
        if not finite[:, names.index(name)].all():
            raise Nonestimable("missing_component_or_mean_exposure")
    medians = np.nanmedian(np.where(finite, raw, np.nan), axis=0)
    filled = np.where(finite, raw, medians)
    means, scales = filled.mean(axis=0), filled.std(axis=0, ddof=0)
    keep = scales > 1e-14
    if any(not keep[names.index(name)] for name in exposure_names):
        raise Nonestimable("constant_component_exposure")
    x = (filled[:, keep] - means[keep]) / scales[keep]
    return {"x": x, "z": np.column_stack([np.ones(len(x)), x]),
            "names": ["intercept"] + list(np.asarray(names)[keep]),
            "scales": np.r_[1.0, scales[keep]], "medians": medians,
            "dropped": list(np.asarray(names)[~keep])}


def fit_array(raw, y, names, exposure_names, mean_col):
    y = np.asarray(y, int)
    result = prepare_design(raw, names, exposure_names, mean_col)
    z, x = result["z"], result["x"]
    if np.unique(y).size != 2:
        raise Nonestimable("single_outcome_class")
    if len(y) <= z.shape[1] or np.linalg.matrix_rank(z) != z.shape[1]:
        raise Nonestimable("rank_deficient_design")
    condition = float(np.linalg.cond(z))
    if not np.isfinite(condition) or condition > 1e12:
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
    result.update(beta=beta, prob=model.predict_proba(x)[:, 1],
                  condition=condition, iterations=int(model.n_iter_[0]))
    return result


def full_fit(raw, y, codes, names, exposure_names, mean_col):
    result = fit_array(raw, y, names, exposure_names, mean_col)
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
    diagonal = np.diag(cov)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise Nonestimable("invalid_cluster_covariance")
    se = np.sqrt(diagonal)
    result.update(cov=cov, se=se)
    with warnings.catch_warnings(record=True):
        verified = sm.GLM(y, z, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": codes, "use_correction": True},
            tol=1e-12, maxiter=300)
    indices = [result["names"].index(name) for name in exposure_names]
    delta_beta = float(np.max(np.abs(verified.params - result["beta"])))
    delta_se = float(np.max(np.abs(verified.bse[indices] - se[indices])))
    result["verification"] = {
        "max_coefficient_difference": delta_beta,
        "max_component_cluster_se_difference": delta_se,
        "statsmodels_converged": bool(verified.converged),
        "passed": bool(verified.converged and delta_beta < 1e-4 and delta_se < 1e-5)}
    return result


def sample_patient_indices(codes, rng):
    by_patient = [np.flatnonzero(codes == group) for group in range(int(codes.max()) + 1)]
    chosen = rng.integers(0, len(by_patient), size=len(by_patient))
    return np.concatenate([by_patient[group] for group in chosen]), chosen


def cluster_bootstrap(models, y, codes, mean_col, reps=REPS, seed=SEED, progress_label=""):
    rng = np.random.default_rng(seed)
    by_patient = [np.flatnonzero(codes == group) for group in range(int(codes.max()) + 1)]
    samples = {key: [] for key in models}
    failures = {key: Counter() for key in models}
    for rep in range(reps):
        chosen = rng.integers(0, len(by_patient), size=len(by_patient))
        ix = np.concatenate([by_patient[group] for group in chosen])
        for key, (raw, names, exposures) in models.items():
            try:
                fit = fit_array(raw[ix], y[ix], names, exposures, mean_col)
                samples[key].append({name: (float(b), float(b / s))
                                     for name, b, s in zip(fit["names"], fit["beta"], fit["scales"])
                                     if name in exposures})
            except Nonestimable as exc:
                failures[key][str(exc)] += 1
            except Exception as exc:
                failures[key][type(exc).__name__] += 1
        if reps >= 100 and (rep + 1) % 50 == 0:
            print(f"{progress_label} component bootstrap: {rep + 1}/{reps} completed.", flush=True)
    return samples, failures


def protected_counts(counts):
    if any(0 < int(value) < 10 for value in counts.values()):
        return {key: SUPPRESSED for key in counts}
    return {key: int(value) for key, value in counts.items()}


def bucket_counts(counts):
    """Avoid exact cohort totals that could complement older suppressed analyses."""
    def bucket(value):
        value = int(value)
        if value == 0:
            return "0"
        if value < 10:
            return "<10"
        low = value // 10 * 10
        return f"{low}-{low + 9}"
    return {key: bucket(value) for key, value in counts.items()}


def bh_fixed_family(values):
    """NA estimates use p=1 internally and retain NA in displayed q values."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise Nonestimable("invalid_p_value")
    work = np.where(finite, values, 1.0)
    order = np.argsort(work, kind="stable")
    adjusted = work[order] * len(work) / np.arange(1, len(work) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(len(work))
    result[order] = np.minimum(adjusted, 1.0)
    result[~finite] = np.nan
    return result


def component_diagnostics(frame, source, baseline, mean_col, knots):
    correlations, residuals, summaries = [], [], []
    group_values = frame[list(GROUPS)].to_numpy(float)
    for group in GROUPS:
        values = frame[group].to_numpy(float)
        summaries.append({"source": source, "group": group,
                          "mean_log_contribution": float(values.mean()),
                          "sd_log_contribution": float(values.std(ddof=0)),
                          "p10": float(np.quantile(values, .1)),
                          "median": float(np.median(values)), "p90": float(np.quantile(values, .9))})
    for i, group_a in enumerate(GROUPS):
        for j, group_b in enumerate(GROUPS):
            if j <= i:
                continue
            if group_values[:, i].std() <= 1e-14 or group_values[:, j].std() <= 1e-14:
                correlation = np.nan
            else:
                correlation = float(np.corrcoef(group_values[:, i], group_values[:, j])[0, 1])
            correlations.append({"source": source, "group_a": group_a, "group_b": group_b,
                                 "pearson_r": correlation})
    for group in GROUPS:
        try:
            raw, names = make_design(frame, [group], baseline, mean_col, knots)
            design = prepare_design(raw, names, [group], mean_col)
            group_index = design["names"].index(group)
            target = design["z"][:, group_index]
            adjust = np.delete(design["z"], group_index, axis=1)
            residual = target - adjust @ np.linalg.lstsq(adjust, target, rcond=None)[0]
            fraction = float(np.mean(residual ** 2))
            residuals.append({"source": source, "group": group, "status": "estimated",
                              "baseline_residual_variance_fraction": fraction,
                              "baseline_vif": 1 / fraction if fraction > 1e-15 else np.nan})
        except Nonestimable as exc:
            residuals.append({"source": source, "group": group, "status": str(exc)})
        except Exception as exc:
            residuals.append({"source": source, "group": group, "status": type(exc).__name__})
    return correlations, residuals, summaries


def analyze_source(frame, source, baseline, mean_col, knots, reps=REPS, seed=SEED):
    y = frame.death90.to_numpy(int)
    codes, _ = pd.factorize(frame[PATIENT], sort=True)
    definitions = {"single_" + group: [group] for group in GROUPS}
    definitions["joint"] = list(GROUPS)
    models, fits, rows, checks = {}, {}, [], []
    for key, exposures in definitions.items():
        analysis = "joint_secondary" if key == "joint" else "single_group_primary_exploratory"
        try:
            if min(frame.loc[frame.death90.eq(1), PATIENT].nunique(),
                   frame.loc[frame.death90.eq(0), PATIENT].nunique()) < 10:
                raise Nonestimable("fewer_than_10_event_or_nonevent_patients")
            raw, names = make_design(frame, exposures, baseline, mean_col, knots)
            fit = full_fit(raw, y, codes, names, exposures, mean_col)
            models[key], fits[key] = (raw, names, exposures), fit
            checks.append({"source": source, "model": key, **fit["verification"]})
        except Nonestimable as exc:
            rows.extend({"source": source, "analysis": analysis, "model": key,
                         "group": group, "status": str(exc)} for group in exposures)
        except Exception as exc:
            rows.extend({"source": source, "analysis": analysis, "model": key,
                         "group": group, "status": type(exc).__name__} for group in exposures)
    samples, failures = cluster_bootstrap(models, y, codes, mean_col, reps, seed, source) if models else ({}, {})
    for key, fit in fits.items():
        analysis = "joint_secondary" if key == "joint" else "single_group_primary_exploratory"
        for group in definitions[key]:
            j = fit["names"].index(group)
            beta, se, scale = float(fit["beta"][j]), float(fit["se"][j]), float(fit["scales"][j])
            draws = np.asarray([draw[group] for draw in samples[key] if group in draw], float)
            rec = {"source": source, "analysis": analysis, "model": key, "group": group,
                   "status": "estimated" if fit["verification"]["passed"] else "verification_mismatch",
                   "beta_standardized": beta, "or_per_sd": exp_or(beta),
                   "contribution_sd": scale, "beta_raw": beta / scale,
                   "or_per_log_contribution_unit": exp_or(beta / scale),
                   "raw_unit_or_numerically_representable": bool(abs(beta / scale) <= 700),
                   "cluster_se": se, "cluster_se_raw": se / scale,
                   "cluster_wald_p": float(2 * norm.sf(abs(beta / se))),
                   "cluster_ci_low": exp_or(beta - 1.96 * se),
                   "cluster_ci_high": exp_or(beta + 1.96 * se),
                   "bootstrap_valid": len(draws), "bootstrap_requested": reps,
                   "bootstrap_failure_reasons": json.dumps(dict(failures[key]), sort_keys=True),
                   "dropped_constant_covariates": ";".join(fit["dropped"]),
                   "design_condition": fit["condition"], "design_parameters": len(fit["names"]),
                   "sparse_event_caution": bool(y.sum() / len(fit["names"]) < 10)}
            if len(draws) >= 2:
                lo, hi = np.quantile(draws, [.025, .975], axis=0)
                rec.update(bootstrap_ci_low=exp_or(lo[0]), bootstrap_ci_high=exp_or(hi[0]),
                           bootstrap_raw_ci_low=exp_or(lo[1]), bootstrap_raw_ci_high=exp_or(hi[1]))
            if len(draws) < .95 * reps:
                rec["status"] += ";insufficient_valid_bootstrap"
            rows.append(rec)
    counts = bucket_counts({"windows": len(frame), "patients": int(frame[PATIENT].nunique()),
                            "event_windows": int(y.sum()), "nonevent_windows": int(len(y) - y.sum()),
                            "event_patients": int(frame.loc[frame.death90.eq(1), PATIENT].nunique())})
    return rows, checks, {"source": source, **counts}


def analyze_components(frame, out_dir, source_configs, reps=REPS, seed=SEED):
    """Fit both sources, writing only aggregate results; baseline configs are required.

    source_configs[source] = {"baseline_columns": [...], "mean_column": "mu_mean90",
                               "mean_form": "rcs3"}
    The baselines are the main-model covariate sets of make_source_configs().
    """
    required = {"source", PATIENT, "landmark_date", "death90", *GROUPS}
    if not required.issubset(frame.columns):
        raise Nonestimable("required_analysis_columns_missing")
    if frame.empty or set(frame.source.unique()) != set(SOURCES):
        raise Nonestimable("expected_CY_and_TT_sources")
    if frame[PATIENT].isna().any() or frame.landmark_date.isna().any():
        raise Nonestimable("missing_patient_or_landmark")
    if frame.duplicated(["source", PATIENT, "landmark_date"]).any():
        raise Nonestimable("duplicate_patient_landmark")
    if not frame.death90.isin([0, 1]).all():
        raise Nonestimable("invalid_or_unknown_binary_outcome")
    if not np.isfinite(frame[list(GROUPS)].to_numpy(float)).all():
        raise Nonestimable("missing_component_exposure")
    if reps < 2:
        raise Nonestimable("insufficient_bootstrap_replicates")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    locked = {}
    for source in SOURCES:
        baseline, mean_col = _configuration(source, source_configs)
        if not set([mean_col] + baseline).issubset(frame.columns):
            raise Nonestimable("required_baseline_columns_missing")
        sub = frame.loc[frame.source.eq(source)]
        if len(sub) < 10 or sub[PATIENT].nunique() < 10:
            raise Nonestimable("insufficient_source_size_for_aggregate_description")
        mean = sub[mean_col].to_numpy(float)
        if not np.isfinite(mean).all():
            raise Nonestimable("missing_mean_exposure")
        knots = np.quantile(mean, [.1, .5, .9]).tolist()
        if not np.all(np.diff(knots) > 0):
            raise Nonestimable("coincident_mean_spline_knots")
        locked[source] = {"baseline_columns": baseline, "mean_column": mean_col,
                          "mean_form": "rcs3", "mean_rcs_knots_10_50_90": knots}
    dump_json({"created_utc_before_fitting": datetime.now(timezone.utc).isoformat(),
               "source_configuration": locked, "seed": seed, "bootstrap_replicates": reps,
               "group_columns": list(GROUPS), "primary_exploratory_family_size": 8,
               "BH_nonestimable_handling": "p=1 internally in fixed family; reported p and q remain missing",
               "bootstrap_resamples": "patients with all windows and multiplicity; imputation/scaling refit",
               "mean_knots": "fixed within source across models and bootstrap",
               "no_patient_level_outputs": True}, out_dir / "component_analysis_lock.json")
    rows, checks, counts, correlations, residuals, summaries = [], [], [], [], [], []
    with threadpool_limits(limits=1):
        for source in SOURCES:
            sub = frame.loc[frame.source.eq(source)]
            config = locked[source]
            baseline, mean_col = config["baseline_columns"], config["mean_column"]
            knots = config["mean_rcs_knots_10_50_90"]
            print(f"Fitting fixed {source} component models; aggregate outputs only.", flush=True)
            source_rows, source_checks, source_counts = analyze_source(sub, source, baseline, mean_col, knots, reps, seed)
            rows.extend(source_rows)
            checks.extend(source_checks)
            counts.append(source_counts)
            corr, resid, summary = component_diagnostics(sub, source, baseline, mean_col, knots)
            correlations.extend(corr)
            residuals.extend(resid)
            summaries.extend(summary)
    result = pd.DataFrame(rows).sort_values(["source", "analysis", "group"])
    if "cluster_wald_p" not in result:
        result["cluster_wald_p"] = np.nan
    primary = result.analysis.eq("single_group_primary_exploratory")
    if primary.sum() != 8:
        raise Nonestimable("unexpected_primary_family_size")
    p = result.loc[primary, "cluster_wald_p"].to_numpy(float, copy=True)
    valid = result.loc[primary, "status"].eq("estimated").to_numpy()
    p[~valid] = np.nan
    result.loc[primary, "bh_fdr_q_8_primary_tests"] = bh_fixed_family(p)
    result.to_csv(out_dir / "component_associations.csv", index=False)
    pd.DataFrame(counts).to_csv(out_dir / "component_analysis_counts.csv", index=False)
    pd.DataFrame(correlations).to_csv(out_dir / "component_correlations.csv", index=False)
    pd.DataFrame(residuals).to_csv(out_dir / "component_baseline_redundancy.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "component_distribution_summary.csv", index=False)
    dump_json(checks, out_dir / "component_statsmodels_verification.json")
    return {"associations": result, "verification": checks, "counts": pd.DataFrame(counts),
            "correlations": pd.DataFrame(correlations), "baseline_redundancy": pd.DataFrame(residuals)}


def self_test(out_dir):
    rng = np.random.default_rng(SEED)
    frames = []
    patients, repeats = 240, 5
    codes = np.repeat(np.arange(patients), repeats)
    n = len(codes)
    for source in SOURCES:
        f = pd.DataFrame({"source": source, PATIENT: [f"synthetic-{i}" for i in codes],
                          "landmark_date": np.tile(pd.date_range("2020-01-01", periods=repeats, freq="QS"), patients),
                          "mu_mean90": rng.normal(130, 12, n)})
        for col in TT_BASELINE:
            f[col] = rng.normal(size=n)
        f["sex_female"] = rng.binomial(1, .5, n)
        f["group_bp"] = rng.normal(0, .2, n)
        f["group_treatment"] = rng.normal(0, .12, n)
        f["group_demographic"] = .03 * f.age_at_index_years ** 2 + .01 * f.sex_female
        f["group_labs"] = rng.normal(0, .09, n)
        f.loc[::17, "prior90_observed_nadir90_burden"] = np.nan
        eta = -1.2 + 1.8 * f.group_bp - 1.2 * f.group_labs + .01 * (f.mu_mean90 - 130)
        f["death90"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
        frames.append(f)
    configs = {source: {"baseline_columns": list(TT_BASELINE)} for source in SOURCES}
    result = analyze_components(pd.concat(frames, ignore_index=True), out_dir, configs, reps=8)
    assert len(result["associations"]) == 16
    assert len(result["verification"]) == 10
    assert all(check["passed"] for check in result["verification"])
    assert result["associations"].status.eq("estimated").all()
    assert result["associations"].bootstrap_valid.eq(8).all()
    ix, chosen = sample_patient_indices(codes, np.random.default_rng(31))
    assert np.array_equal(np.bincount(codes[ix], minlength=patients), repeats * np.bincount(chosen, minlength=patients))
    assert all(value == SUPPRESSED for value in protected_counts({"total": 100, "event": 7, "nonevent": 93}).values())
    assert bucket_counts({"total": 123, "rare": 5, "zero": 0}) == {"total": "120-129", "rare": "<10", "zero": "0"}
    assert np.allclose(bh_fixed_family([.01, .03, np.nan, .4])[[0, 1, 3]], [.04, .06, .5333333333333333])
    assert np.isnan(bh_fixed_family([.01, .03, np.nan, .4])[2])
    assert np.isnan(exp_or(710)) and exp_or(0) == 1
    f = frames[0].copy()
    f["group_demographic"] = f.age_at_index_years
    knots = np.quantile(f.mu_mean90, [.1, .5, .9])
    raw, names = make_design(f, ["group_demographic"], TT_BASELINE, "mu_mean90", knots)
    try:
        fit_array(raw, f.death90.to_numpy(), names, ["group_demographic"], "mu_mean90")
        raise AssertionError("Rank-deficient demographic group accepted")
    except Nonestimable as exc:
        assert str(exc) == "rank_deficient_design"
    raw, names = make_design(f, ["group_bp"], TT_BASELINE, "mu_mean90", knots)
    first_ix, _ = sample_patient_indices(codes, np.random.default_rng(SEED))
    fit = fit_array(raw[first_ix], f.death90.to_numpy()[first_ix], names, ["group_bp"], "mu_mean90")
    samples, _ = cluster_bootstrap({"single": (raw, names, ["group_bp"])}, f.death90.to_numpy(), codes, "mu_mean90", reps=2)
    j = fit["names"].index("group_bp")
    assert samples["single"][0]["group_bp"][1] == fit["beta"][j] / fit["scales"][j]
    raw[0, names.index("group_bp")] = np.nan
    try:
        fit_array(raw, f.death90.to_numpy(), names, ["group_bp"], "mu_mean90")
        raise AssertionError("Missing group contribution was imputed")
    except Nonestimable as exc:
        assert str(exc) == "missing_component_or_mean_exposure"
    dump_json({"synthetic_only": True, "assertions_passed": True,
               "checks": ["eight_primary_and_eight_joint_component_estimates", "independent_GLM_coefficients_and_cluster_SE",
                          "patient_bootstrap_multiplicity", "raw_unit_bootstrap_scaling", "small_cell_complementary_suppression",
                          "fixed_BH_family_with_nonestimable_slot", "rank_deficient_group_rejected", "missing_component_rejected"],
               "max_independent_coefficient_difference": max(c["max_coefficient_difference"] for c in result["verification"]),
               "max_independent_component_SE_difference": max(c["max_component_cluster_se_difference"] for c in result["verification"])},
              Path(out_dir) / "synthetic_verification.json")
    print("Synthetic component statistics checks passed; no clinical data accessed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("This module takes an in-memory frame; use --self-test for synthetic checks.")
    self_test(Path(__file__).resolve().parent / "synthetic_statistics")
