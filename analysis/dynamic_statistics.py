"""Fixed exploratory landmark models for changes in predicted SBP dispersion.

All clinical data stay in RAM. Only aggregate coefficients, diagnostics, and
protected counts are written. analyze_dynamics is called by
analysis.run_change_and_timing; --self-test accesses synthetic data only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from . import component_statistics as core

import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

PATIENT = core.PATIENT
SOURCES = core.SOURCES
REPS = 300
SEED = 20260926
MODEL_EXPOSURES = {"dynamic_primary": ["delta_sigma", "sigma_old90"],
                   "recent_sigma_restricted_secondary": ["sigma_mean90"]}
CORE_CONTINUOUS = ["sigma_mean90", "sigma_old90", "mu_mean90", "mu_old90", "old90_n_sessions", "log1p_prior90_sessions"]


def make_source_configs():
    return core.make_source_configs()


def prepare_frame(frame, source_configs):
    """Check exposure availability and exact differences without imputing them."""
    required = {"source", PATIENT, "landmark_date", "death90", *CORE_CONTINUOUS}
    if not required.issubset(frame.columns):
        raise core.Nonestimable("required_dynamic_analysis_columns_missing")
    if frame.empty or set(frame.source.unique()) != set(SOURCES):
        raise core.Nonestimable("expected_CY_and_TT_sources")
    if frame[PATIENT].isna().any() or frame.landmark_date.isna().any():
        raise core.Nonestimable("missing_patient_or_landmark")
    if frame.duplicated(["source", PATIENT, "landmark_date"]).any():
        raise core.Nonestimable("duplicate_patient_landmark")
    if not frame.death90.isin([0, 1]).all():
        raise core.Nonestimable("invalid_or_unknown_binary_outcome")
    if not np.isfinite(frame[CORE_CONTINUOUS].to_numpy(float)).all():
        raise core.Nonestimable("missing_dynamic_exposure_or_old_session_count")
    if not frame[["sigma_mean90", "sigma_old90"]].gt(0).all().all():
        raise core.Nonestimable("nonpositive_sigma")
    if not (frame.old90_n_sessions.ge(4) & frame.old90_n_sessions.mod(1).eq(0)).all():
        raise core.Nonestimable("insufficient_or_noninteger_old_session_count")
    if not frame.log1p_prior90_sessions.gt(0).all():
        raise core.Nonestimable("nonpositive_recent_session_count")
    result = frame.copy()
    for name, current, previous in (("delta_sigma", "sigma_mean90", "sigma_old90"),
                                    ("delta_mu", "mu_mean90", "mu_old90")):
        values = result[current].to_numpy(float) - result[previous].to_numpy(float)
        if name in result and not np.allclose(result[name].to_numpy(float), values, atol=1e-12, rtol=0):
            raise core.Nonestimable("precomputed_dynamic_exposure_mismatch")
        result[name] = values
    if not np.isfinite(result[["delta_sigma", "delta_mu"]].to_numpy(float)).all():
        raise core.Nonestimable("nonfinite_computed_difference")
    result["log1p_old90_sessions"] = np.log1p(result.old90_n_sessions.to_numpy(float))
    for source in SOURCES:
        config = source_configs[source]
        if config.get("mean_column", "mu_mean90") != "mu_mean90" or config.get("mean_form", "rcs3") != "rcs3":
            raise core.Nonestimable("unsupported_dynamic_mean_configuration")
        baseline = list(config["baseline_columns"])
        if len(baseline) != len(set(baseline)) or len(baseline) != 8:
            raise core.Nonestimable("unexpected_source_M2_baseline")
        if set(baseline).intersection({"mu_mean90", "delta_sigma", "sigma_old90", "sigma_mean90", "delta_mu", "log1p_old90_sessions"}):
            raise core.Nonestimable("duplicated_dynamic_design_column")
        if not set(baseline).issubset(result.columns):
            raise core.Nonestimable("required_source_M2_baseline_missing")
    return result


def model_baseline(key, baseline):
    return list(baseline) + (["delta_mu", "log1p_old90_sessions"] if key == "dynamic_primary" else [])


def model_design(frame, key, baseline, knots):
    return core.make_design(frame, MODEL_EXPOSURES[key], model_baseline(key, baseline), "mu_mean90", knots)


def term_analysis(key, term):
    if key == "recent_sigma_restricted_secondary":
        return "restricted_cohort_recent_sigma_context"
    return "change_primary_exploratory" if term == "delta_sigma" else "prior_level_explanatory"


def summarize_coefficient(source, key, term, fit, draws, failures, reps, events):
    """All bootstrap contrasts use the fixed full-analysis exposure SD."""
    j = fit["names"].index(term)
    beta, se, scale = (float(fit[name][j]) for name in ("beta", "se", "scales"))
    raw_draws = np.asarray([draw[term][1] for draw in draws if term in draw], float)
    rec = {"source": source, "model": key, "term": term, "analysis": term_analysis(key, term),
           "status": "estimated" if fit["verification"]["passed"] else "verification_mismatch",
           "beta_standardized": beta, "or_per_sd": core.exp_or(beta),
           "exposure_sd_mmhg": scale, "beta_per_mmhg": beta / scale,
           "or_per_mmhg": core.exp_or(beta / scale),
           "cluster_se": se, "cluster_se_per_mmhg": se / scale,
           "cluster_wald_p": float(2 * norm.sf(abs(beta / se))),
           "cluster_ci_low": core.exp_or(beta - 1.96 * se),
           "cluster_ci_high": core.exp_or(beta + 1.96 * se),
           "bootstrap_valid": int(len(raw_draws)), "bootstrap_requested": reps,
           "bootstrap_failure_reasons": json.dumps(dict(failures), sort_keys=True),
           "dropped_constant_covariates": ";".join(fit["dropped"]),
           "design_condition": fit["condition"], "design_parameters": len(fit["names"]),
           "sparse_event_caution": bool(events / len(fit["names"]) < 10),
           "bootstrap_standardization": "raw-coefficient draws times full-analysis exposure SD"}
    if len(raw_draws) >= 2:
        lo, hi = np.quantile(raw_draws, [.025, .975])
        rec.update(bootstrap_ci_low=core.exp_or(lo * scale), bootstrap_ci_high=core.exp_or(hi * scale),
                   bootstrap_per_mmhg_ci_low=core.exp_or(lo), bootstrap_per_mmhg_ci_high=core.exp_or(hi))
    if len(raw_draws) < .95 * reps:
        rec["status"] += ";insufficient_valid_bootstrap"
    return rec


def cluster_bootstrap(models, y, codes, reps, seed, source):
    """Reuse numerical fit core, but reject constant covariates in any resample."""
    rng = np.random.default_rng(seed)
    by_patient = [np.flatnonzero(codes == group) for group in range(int(codes.max()) + 1)]
    samples, failures = {key: [] for key in models}, {key: Counter() for key in models}
    for rep in range(reps):
        chosen = rng.integers(0, len(by_patient), size=len(by_patient))
        ix = np.concatenate([by_patient[group] for group in chosen])
        for key, (raw, names, exposures) in models.items():
            try:
                fit = core.fit_array(raw[ix], y[ix], names, exposures, "mu_mean90")
                if fit["dropped"]:
                    raise core.Nonestimable("constant_planned_covariate_no_automatic_deletion")
                samples[key].append({name: (float(b), float(b / s))
                                     for name, b, s in zip(fit["names"], fit["beta"], fit["scales"])
                                     if name in exposures})
            except core.Nonestimable as exc:
                failures[key][str(exc)] += 1
            except Exception as exc:
                failures[key][type(exc).__name__] += 1
        if reps >= 100 and (rep + 1) % 50 == 0:
            print(f"{source} dynamic bootstrap: {rep + 1}/{reps} completed.", flush=True)
    return samples, failures


def exposure_diagnostics(frame, source, fit):
    rows = []
    for term in ("sigma_mean90", "sigma_old90", "delta_sigma", "mu_mean90", "mu_old90", "delta_mu"):
        x = frame[term].to_numpy(float)
        rows.append({"source": source, "term": term, "mean": float(x.mean()),
                     "sd": float(x.std(ddof=0)), "p10": float(np.quantile(x, .1)),
                     "median": float(np.median(x)), "p90": float(np.quantile(x, .9))})
    corr = [{"source": source, "term_a": a, "term_b": b,
             "pearson_r": float(np.corrcoef(frame[a], frame[b])[0, 1])}
            for a, b in (("sigma_old90", "sigma_mean90"), ("sigma_old90", "delta_sigma"),
                         ("mu_mean90", "delta_mu"), ("delta_sigma", "delta_mu"),
                         ("log1p_prior90_sessions", "log1p_old90_sessions"))]
    redundancy = []
    if fit is not None:
        for term in MODEL_EXPOSURES["dynamic_primary"]:
            j = fit["names"].index(term)
            x, adjust = fit["z"][:, j], np.delete(fit["z"], j, axis=1)
            residual = x - adjust @ np.linalg.lstsq(adjust, x, rcond=None)[0]
            fraction = float(np.mean(residual ** 2))
            redundancy.append({"source": source, "term": term,
                               "conditional_residual_variance_fraction": fraction,
                               "vif": 1 / fraction if fraction > 1e-15 else np.nan})
    return rows, corr, redundancy


def analyze_source(frame, source, baseline, knots, reps, seed):
    y = frame.death90.to_numpy(int)
    codes, _ = pd.factorize(frame[PATIENT], sort=True)
    models, fits, rows, checks, preprocessing = {}, {}, [], [], []
    enough = min(frame.loc[frame.death90.eq(1), PATIENT].nunique(),
                 frame.loc[frame.death90.eq(0), PATIENT].nunique()) >= 10
    for key, exposures in MODEL_EXPOSURES.items():
        try:
            if not enough:
                raise core.Nonestimable("fewer_than_10_event_or_nonevent_patients")
            raw, names = model_design(frame, key, baseline, knots)
            fit = core.full_fit(raw, y, codes, names, exposures, "mu_mean90")
            if fit["dropped"]:
                raise core.Nonestimable("constant_planned_covariate_no_automatic_deletion")
            fits[key], models[key] = fit, (raw, names, exposures)
            checks.append({"source": source, "model": key, **fit["verification"]})
            for j, name in enumerate(names):
                missing = int(np.count_nonzero(~np.isfinite(raw[:, j])))
                preprocessing.append({"source": source, "model": key, "column": name,
                                      "full_sample_imputation_median": float(fit["medians"][j]),
                                      "missing_count_bucket": core.bucket_counts({"count": missing})["count"]})
        except core.Nonestimable as exc:
            rows.extend({"source": source, "model": key, "term": term,
                         "analysis": term_analysis(key, term), "status": str(exc)} for term in exposures)
        except Exception as exc:
            rows.extend({"source": source, "model": key, "term": term,
                         "analysis": term_analysis(key, term), "status": type(exc).__name__} for term in exposures)
    samples, failures = cluster_bootstrap(models, y, codes, reps, seed, source) if models else ({}, {})
    for key, fit in fits.items():
        for term in MODEL_EXPOSURES[key]:
            rows.append(summarize_coefficient(source, key, term, fit, samples[key], failures[key], reps, int(y.sum())))
    counts = {"source": source, **core.bucket_counts({"windows": len(frame), "patients": frame[PATIENT].nunique(),
                   "event_windows": int(y.sum()), "nonevent_windows": len(y) - int(y.sum()),
                   "event_patients": frame.loc[frame.death90.eq(1), PATIENT].nunique()})}
    summaries, correlations, redundancy = exposure_diagnostics(frame, source, fits.get("dynamic_primary"))
    return rows, checks, counts, summaries, correlations, redundancy, preprocessing


def analyze_dynamics(frame, out_dir, source_configs=None, reps=REPS, seed=SEED):
    """Run locked models on an in-memory, two-window eligible CY/TT cohort.

    This estimates a change association conditional on old sigma and listed
    covariates. It is not patient fixed-effects, causal, or a pure separation of
    between-person and within-person effects. The old-sigma coefficient holds
    change fixed: both old and recent sigma move together.
    """
    if source_configs is None:
        source_configs = make_source_configs()
    if reps < 2:
        raise core.Nonestimable("insufficient_bootstrap_replicates")
    frame = prepare_frame(frame, source_configs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    locked = {}
    for source in SOURCES:
        sub = frame.loc[frame.source.eq(source)]
        if len(sub) < 10 or sub[PATIENT].nunique() < 10:
            raise core.Nonestimable("insufficient_source_size_for_aggregate_description")
        knots = np.quantile(sub.mu_mean90.to_numpy(float), [.1, .5, .9])
        if not np.all(np.diff(knots) > 0):
            raise core.Nonestimable("coincident_mean_spline_knots")
        locked[source] = {"baseline_columns": list(source_configs[source]["baseline_columns"]),
                          "mean_column": "mu_mean90", "mean_form": "rcs3",
                          "mean_rcs_knots_10_50_90": knots.tolist()}
    core.dump_json({"created_utc_before_fitting": datetime.now(timezone.utc).isoformat(),
                    "source_configuration": locked, "seed": seed, "bootstrap_replicates": reps,
                    "models": MODEL_EXPOSURES, "dynamic_additional_covariates": ["delta_mu", "log1p_old90_sessions"],
                    "mean_knots": "fixed full restricted-source 10/50/90 quantiles across models/bootstrap",
                    "bootstrap": "patient multiplicity; imputation/scaling refit; raw beta times full-analysis exposure SD",
                    "primary_family": "two delta_sigma coefficients; BH fixed family n=2",
                    "nonestimable_primary": "p=1 internally for BH; displayed p/q missing",
                    "no_patient_level_outputs": True}, out_dir / "dynamic_analysis_lock.json")
    rows, checks, counts, summaries, correlations, redundancy, preprocessing = [], [], [], [], [], [], []
    with threadpool_limits(limits=1):
        for source in SOURCES:
            print(f"Fitting fixed {source} dynamic and contextual models; aggregate outputs only.", flush=True)
            sub, config = frame.loc[frame.source.eq(source)], locked[source]
            values = analyze_source(sub, source, config["baseline_columns"], config["mean_rcs_knots_10_50_90"], reps, seed)
            for dest, value in zip((rows, checks, counts, summaries, correlations, redundancy, preprocessing), values):
                dest.extend(value if isinstance(value, list) else [value])
    result = pd.DataFrame(rows).sort_values(["source", "model", "term"])
    if "cluster_wald_p" not in result:
        result["cluster_wald_p"] = np.nan
    primary = result.analysis.eq("change_primary_exploratory")
    if int(primary.sum()) != 2:
        raise core.Nonestimable("unexpected_dynamic_primary_family_size")
    p = result.loc[primary, "cluster_wald_p"].to_numpy(float, copy=True)
    p[~result.loc[primary, "status"].eq("estimated").to_numpy()] = np.nan
    result.loc[primary, "bh_fdr_q_2_primary_tests"] = core.bh_fixed_family(p)
    result.to_csv(out_dir / "dynamic_associations.csv", index=False)
    for values, name in ((counts, "dynamic_analysis_counts"), (summaries, "dynamic_exposure_summary"),
                          (correlations, "dynamic_exposure_correlations"), (redundancy, "dynamic_design_redundancy"),
                          (preprocessing, "dynamic_covariate_preprocessing")):
        pd.DataFrame(values).to_csv(out_dir / (name + ".csv"), index=False)
    core.dump_json(checks, out_dir / "dynamic_statsmodels_verification.json")
    return {"associations": result, "verification": checks, "counts": pd.DataFrame(counts),
            "all_passed": bool(len(checks) == 4 and all(x["passed"] for x in checks) and
                               len(result) == 6 and result.status.eq("estimated").all()),
            "correlations": pd.DataFrame(correlations), "redundancy": pd.DataFrame(redundancy)}


def self_test(out_dir):
    rng = np.random.default_rng(SEED)
    frames, patients, repeats = [], 240, 5
    codes = np.repeat(np.arange(patients), repeats)
    n = len(codes)
    for source in SOURCES:
        old = rng.normal(10, 1.8, n)
        delta = -.15 * (old - 10) + rng.normal(0, .8, n)
        old_mu = rng.normal(130, 12, n)
        f = pd.DataFrame({"source": source, PATIENT: [f"synthetic-{i}" for i in codes],
                          "landmark_date": np.tile(pd.date_range("2020-01-01", periods=repeats, freq="QS"), patients),
                          "sigma_old90": old, "sigma_mean90": old + delta,
                          "mu_old90": old_mu, "mu_mean90": old_mu + rng.normal(0, 5, n),
                          "old90_n_sessions": rng.integers(4, 40, n)})
        for col in core.TT_BASELINE:
            f[col] = rng.normal(size=n)
        f["sex_female"] = rng.binomial(1, .5, n)
        f["male"] = 1 - f.sex_female
        f["log1p_prior90_sessions"] = np.log1p(rng.integers(4, 40, n))
        f.loc[::17, "prior90_observed_nadir90_burden"] = np.nan
        eta = -1.2 + .55 * delta + .18 * (old - 10) + .01 * (f.mu_mean90 - 130)
        f["death90"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    result = analyze_dynamics(frame, out_dir, reps=8)
    assert len(result["associations"]) == 6
    assert len(result["verification"]) == 4
    assert all(check["passed"] for check in result["verification"])
    assert result["associations"].status.eq("estimated").all()
    assert result["associations"].bootstrap_valid.eq(8).all()
    assert result["associations"].loc[result["associations"].model.eq("dynamic_primary"), "design_parameters"].eq(15).all()
    assert result["associations"].loc[result["associations"].model.eq("recent_sigma_restricted_secondary"), "design_parameters"].eq(12).all()
    f = prepare_frame(frame, make_source_configs()).loc[lambda x: x.source.eq("CY")]
    baseline = list(core.CY_BASELINE)
    knots = np.quantile(f.mu_mean90, [.1, .5, .9])
    raw, names = model_design(f, "dynamic_primary", baseline, knots)
    fit = core.full_fit(raw, f.death90.to_numpy(), codes, names, MODEL_EXPOSURES["dynamic_primary"], "mu_mean90")
    # The same unpenalized parameterization: recent+old equals delta+old.
    alt = raw.copy()
    delta_j, old_j = names.index("delta_sigma"), names.index("sigma_old90")
    alt[:, delta_j] = raw[:, delta_j] + raw[:, old_j]
    altfit = core.fit_array(alt, f.death90.to_numpy(), names, MODEL_EXPOSURES["dynamic_primary"], "mu_mean90")
    assert np.max(np.abs(fit["prob"] - altfit["prob"])) < 1e-6
    b_delta = fit["beta"][1] / fit["scales"][1]
    b_old = fit["beta"][2] / fit["scales"][2]
    assert abs(b_delta - altfit["beta"][1] / altfit["scales"][1]) < 1e-5
    assert abs(b_old - b_delta - altfit["beta"][2] / altfit["scales"][2]) < 1e-5
    samples, failures = cluster_bootstrap({"dynamic_primary": (raw, names, MODEL_EXPOSURES["dynamic_primary"])},
                                         f.death90.to_numpy(), codes, reps=8, seed=SEED, source="synthetic")
    rec = summarize_coefficient("CY", "dynamic_primary", "delta_sigma", fit, samples["dynamic_primary"], failures["dynamic_primary"], 8, int(f.death90.sum()))
    draws = np.array([x["delta_sigma"][1] for x in samples["dynamic_primary"]])
    lo, hi = np.quantile(draws, [.025, .975])
    assert np.isclose(rec["bootstrap_ci_low"], np.exp(lo * rec["exposure_sd_mmhg"]))
    assert np.isclose(rec["bootstrap_ci_high"], np.exp(hi * rec["exposure_sd_mmhg"]))
    malformed = frame.copy()
    malformed["delta_sigma"] = 0.
    try:
        prepare_frame(malformed, make_source_configs())
        raise AssertionError("Inconsistent differences accepted")
    except core.Nonestimable as exc:
        assert str(exc) == "precomputed_dynamic_exposure_mismatch"
    malformed = frame.copy()
    malformed.loc[0, "sigma_old90"] = np.nan
    try:
        prepare_frame(malformed, make_source_configs())
        raise AssertionError("Missing old exposure imputed")
    except core.Nonestimable as exc:
        assert str(exc) == "missing_dynamic_exposure_or_old_session_count"
    core.dump_json({"synthetic_only": True, "assertions_passed": True,
                    "checks": ["two_primary_exploratory_delta_estimates", "six_reported_coefficients_four_models",
                               "independent_GLM_coefficients_and_cluster_SE", "expected_15_and_12_parameters",
                               "algebraic_recent_old_parameterization_equivalence", "fixed_full_sample_SD_bootstrap_CI",
                               "difference_identity_guard", "missing_old_exposure_rejected"],
                    "max_independent_coefficient_difference": max(c["max_coefficient_difference"] for c in result["verification"]),
                    "max_independent_exposure_SE_difference": max(c["max_component_cluster_se_difference"] for c in result["verification"])},
                   Path(out_dir) / "synthetic_verification.json")
    print("Synthetic dynamic statistics checks passed; no clinical data accessed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("This module takes an in-memory frame; use --self-test for synthetic checks.")
    self_test(Path(__file__).resolve().parent / "synthetic_statistics")
