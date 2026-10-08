"""Fixed exploratory early/late mortality associations of change in sigma.

Clinical frames are passed in RAM; only aggregate estimates are written.
analyze_timing is called by analysis.run_change_and_timing. --self-test uses synthetic data only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from . import dynamic_statistics as dynamic

import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

core = dynamic.core
PATIENT, SOURCES = core.PATIENT, core.SOURCES
EXPOSURES = ("delta_sigma", "sigma_old90")
INTERVALS = ("days_1_30", "days_31_90")
OUTCOMES = {"days_1_30": "death_early30", "days_31_90": "death_late31_90"}
REPS, SEED = 300, 20260926


def load_reference(dynamic_dir):
    """Reuse the change model's mean knots and full-cohort exposure SDs (from its output directory), not refit them."""
    dynamic_dir = Path(dynamic_dir)
    lock = json.loads((dynamic_dir / "dynamic_analysis_lock.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(dynamic_dir / "dynamic_exposure_summary.csv")
    configs = lock["source_configuration"]
    result = {}
    for source in SOURCES:
        scales = {}
        for term in EXPOSURES:
            selected = summary.loc[summary.source.eq(source) & summary.term.eq(term), "sd"]
            if len(selected) != 1:
                raise core.Nonestimable("previous_dynamic_scale_not_unique")
            scales[term] = float(selected.item())
        result[source] = {**configs[source], "fixed_exposure_sd_mmhg": scales}
    return result


def prepare_frame(frame, reference):
    result = dynamic.prepare_frame(frame, reference)
    required = {*OUTCOMES.values(), "eligible_late"}
    if not required.issubset(result.columns):
        raise core.Nonestimable("required_timing_columns_missing")
    if not result.death_early30.isin([0, 1]).all():
        raise core.Nonestimable("unknown_or_invalid_early_binary_outcome")
    if not result.eligible_late.isin([True, False]).all():
        raise core.Nonestimable("invalid_late_eligibility")
    eligible = result.eligible_late.to_numpy(bool)
    early = result.death_early30.to_numpy(int)
    if not np.array_equal(eligible, early == 0):
        raise core.Nonestimable("late_risk_set_must_exclude_early_deaths_only_in_known90_cohort")
    if not result.loc[eligible, "death_late31_90"].isin([0, 1]).all():
        raise core.Nonestimable("unknown_or_invalid_late_binary_outcome")
    if result.loc[~eligible, "death_late31_90"].notna().any():
        raise core.Nonestimable("early_death_in_late_risk_set")
    if not np.array_equal(early + result.death_late31_90.fillna(0).to_numpy(int), result.death90.to_numpy(int)):
        raise core.Nonestimable("timing_partition_not_equal_original_death90")
    for source in SOURCES:
        sub, config = result.loc[result.source.eq(source)], reference[source]
        if not np.allclose(np.quantile(sub.mu_mean90.to_numpy(float), [.1, .5, .9]),
                           config["mean_rcs_knots_10_50_90"], rtol=0, atol=1e-9):
            raise core.Nonestimable("original_dynamic_cohort_mean_knots_not_reproduced")
        for term in EXPOSURES:
            scale = config["fixed_exposure_sd_mmhg"][term]
            if not np.isfinite(scale) or scale <= 1e-14:
                raise core.Nonestimable("invalid_fixed_original_exposure_scale")
            if not np.isclose(sub[term].to_numpy(float).std(ddof=0), scale, atol=1e-12, rtol=0):
                raise core.Nonestimable("original_dynamic_cohort_exposure_scale_not_reproduced")
    return result


def interval_masks(frame):
    return {"days_1_30": np.ones(len(frame), dtype=bool),
            "days_31_90": frame.eligible_late.to_numpy(bool)}


def paired_cluster_bootstrap(raw, names, outcomes, masks, codes, reps, seed, source):
    """Keep replicate index; both fits use identical original-patient multiplicities.

    Sampling is from all original source patients. The late fit then retains only
    their eligible late-risk rows, including multiplicities. No independently
    sampled risk-set bootstrap and no patient-level output are used.
    """
    rng = np.random.default_rng(seed)
    by_patient = [np.flatnonzero(codes == group) for group in range(int(codes.max()) + 1)]
    samples = {key: np.full((reps, len(EXPOSURES)), np.nan) for key in INTERVALS}
    failures = {key: Counter() for key in INTERVALS}
    for rep in range(reps):
        chosen = rng.integers(0, len(by_patient), size=len(by_patient))
        ix = np.concatenate([by_patient[group] for group in chosen])
        for key in INTERVALS:
            use = ix[masks[key][ix]]
            try:
                fit = core.fit_array(raw[use], outcomes[key][use], names, EXPOSURES, "mu_mean90")
                if fit["dropped"]:
                    raise core.Nonestimable("constant_planned_covariate_no_automatic_deletion")
                for j, term in enumerate(EXPOSURES):
                    k = fit["names"].index(term)
                    samples[key][rep, j] = float(fit["beta"][k] / fit["scales"][k])
            except core.Nonestimable as exc:
                failures[key][str(exc)] += 1
            except Exception as exc:
                failures[key][type(exc).__name__] += 1
        if reps >= 100 and (rep + 1) % 50 == 0:
            print(f"{source} paired timing bootstrap: {rep + 1}/{reps} completed.", flush=True)
    return samples, failures


def summarize_coefficient(source, interval, term, fit, draws, failures, fixed_scale, reps, events):
    j = fit["names"].index(term)
    beta = float(fit["beta"][j] / fit["scales"][j])
    se = float(fit["se"][j] / fit["scales"][j])
    valid = draws[np.isfinite(draws)]
    rec = {"source": source, "interval": interval, "model": "dynamic_primary",
           "term": term, "analysis": "timing_change_exploratory" if term == "delta_sigma" else "prior_level_explanatory",
           "status": "estimated" if fit["verification"]["passed"] else "verification_mismatch",
           "beta_per_mmhg": beta, "or_per_mmhg": core.exp_or(beta),
           "fixed_original_cohort_sd_mmhg": fixed_scale,
           "beta_per_fixed_sd": beta * fixed_scale, "or_per_fixed_sd": core.exp_or(beta * fixed_scale),
           "cluster_se_per_mmhg": se, "cluster_se_per_fixed_sd": se * fixed_scale,
           "cluster_wald_p": float(2 * norm.sf(abs(beta / se))),
           "cluster_per_mmhg_ci_low": core.exp_or(beta - 1.96 * se),
           "cluster_per_mmhg_ci_high": core.exp_or(beta + 1.96 * se),
           "bootstrap_valid": int(len(valid)), "bootstrap_requested": reps,
           "bootstrap_failure_reasons": json.dumps(dict(failures), sort_keys=True),
           "dropped_constant_covariates": ";".join(fit["dropped"]),
           "design_parameters": len(fit["names"]), "design_condition": fit["condition"],
           "sparse_event_caution": bool(events / len(fit["names"]) < 10),
           "bootstrap_standardization": "raw-coefficient draws times fixed prior full-cohort exposure SD"}
    if len(valid) >= 2:
        lo, hi = np.quantile(valid, [.025, .975])
        rec.update(bootstrap_per_mmhg_ci_low=core.exp_or(lo), bootstrap_per_mmhg_ci_high=core.exp_or(hi),
                   bootstrap_fixed_sd_ci_low=core.exp_or(lo * fixed_scale),
                   bootstrap_fixed_sd_ci_high=core.exp_or(hi * fixed_scale))
    if len(valid) < .95 * reps:
        rec["status"] += ";insufficient_valid_bootstrap"
    return rec


def summarize_contrast(source, fits, samples, fixed_scale, reps):
    rec = {"source": source, "term": "delta_sigma", "contrast": "late_minus_early",
           "analysis": "paired_interval_contrast_secondary", "bootstrap_requested": reps,
           "fixed_original_cohort_sd_mmhg": fixed_scale,
           "interpretation": "difference in conditional log odds ratios across unequal intervals; not a hazard-ratio interaction"}
    if any(key not in fits for key in INTERVALS):
        rec.update(status="missing_interval_point_estimate", paired_bootstrap_valid=0)
        return rec
    point = {}
    for key in INTERVALS:
        fit = fits[key]
        j = fit["names"].index("delta_sigma")
        point[key] = float(fit["beta"][j] / fit["scales"][j])
    difference = point[INTERVALS[1]] - point[INTERVALS[0]]
    valid = np.isfinite(samples[INTERVALS[0]][:, 0]) & np.isfinite(samples[INTERVALS[1]][:, 0])
    draws = samples[INTERVALS[1]][valid, 0] - samples[INTERVALS[0]][valid, 0]
    rec.update(status="estimated" if all(fits[key]["verification"]["passed"] for key in INTERVALS) else "verification_mismatch",
               log_or_difference_per_mmhg=difference, ratio_of_ors_per_mmhg=core.exp_or(difference),
               log_or_difference_per_fixed_sd=difference * fixed_scale,
               ratio_of_ors_per_fixed_sd=core.exp_or(difference * fixed_scale),
               paired_bootstrap_valid=int(valid.sum()))
    if len(draws) >= 2:
        lo, hi = np.quantile(draws, [.025, .975])
        rec.update(bootstrap_log_difference_ci_low=float(lo), bootstrap_log_difference_ci_high=float(hi),
                   bootstrap_ratio_of_ors_ci_low=core.exp_or(lo), bootstrap_ratio_of_ors_ci_high=core.exp_or(hi),
                   bootstrap_fixed_sd_log_difference_ci_low=float(lo * fixed_scale),
                   bootstrap_fixed_sd_log_difference_ci_high=float(hi * fixed_scale),
                   bootstrap_fixed_sd_ratio_of_ors_ci_low=core.exp_or(lo * fixed_scale),
                   bootstrap_fixed_sd_ratio_of_ors_ci_high=core.exp_or(hi * fixed_scale),
                   paired_bootstrap_se=float(draws.std(ddof=1)))
    if len(draws) < .95 * reps:
        rec["status"] += ";insufficient_valid_paired_bootstrap"
    return rec


def analyze_source(frame, source, config, reps, seed):
    raw, names = dynamic.model_design(frame, "dynamic_primary", config["baseline_columns"], config["mean_rcs_knots_10_50_90"])
    if len(names) + 1 != 15:
        raise core.Nonestimable("unexpected_planned_parameter_count")
    masks = interval_masks(frame)
    outcomes = {key: frame[col].to_numpy(float) for key, col in OUTCOMES.items()}
    codes, _ = pd.factorize(frame[PATIENT], sort=True)
    fits, rows, checks, counts, preprocessing = {}, [], [], [], []
    for interval in INTERVALS:
        use = masks[interval]
        sub = frame.loc[use]
        y = outcomes[interval][use].astype(int)
        local_codes, _ = pd.factorize(sub[PATIENT], sort=True)
        counts.append({"source": source, "interval": interval,
                       **core.bucket_counts({"windows": len(sub), "patients": sub[PATIENT].nunique(),
                           "event_windows": int(y.sum()), "nonevent_windows": len(y) - int(y.sum()),
                           "event_patients": sub.loc[y == 1, PATIENT].nunique(),
                           "nonevent_patients": sub.loc[y == 0, PATIENT].nunique()})})
        try:
            if min(sub.loc[y == 1, PATIENT].nunique(), sub.loc[y == 0, PATIENT].nunique()) < 10:
                raise core.Nonestimable("fewer_than_10_event_or_nonevent_patients")
            fit = core.full_fit(raw[use], y, local_codes, names, EXPOSURES, "mu_mean90")
            if fit["dropped"]:
                raise core.Nonestimable("constant_planned_covariate_no_automatic_deletion")
            fits[interval] = fit
            checks.append({"source": source, "interval": interval, **fit["verification"]})
            for j, name in enumerate(names):
                missing = int(np.count_nonzero(~np.isfinite(raw[use, j])))
                preprocessing.append({"source": source, "interval": interval, "column": name,
                    "interval_imputation_median": float(fit["medians"][j]),
                    "missing_count_bucket": core.bucket_counts({"count": missing})["count"]})
        except core.Nonestimable as exc:
            rows.extend({"source": source, "interval": interval, "model": "dynamic_primary", "term": term,
                         "analysis": "timing_change_exploratory" if term == "delta_sigma" else "prior_level_explanatory",
                         "status": str(exc)} for term in EXPOSURES)
        except Exception as exc:
            rows.extend({"source": source, "interval": interval, "model": "dynamic_primary", "term": term,
                         "analysis": "timing_change_exploratory" if term == "delta_sigma" else "prior_level_explanatory",
                         "status": type(exc).__name__} for term in EXPOSURES)
    samples, failures = paired_cluster_bootstrap(raw, names, outcomes, masks, codes, reps, seed, source)
    for interval, fit in fits.items():
        for j, term in enumerate(EXPOSURES):
            rows.append(summarize_coefficient(source, interval, term, fit, samples[interval][:, j], failures[interval],
                config["fixed_exposure_sd_mmhg"][term], reps, int(outcomes[interval][masks[interval]].sum())))
    contrast = summarize_contrast(source, fits, samples, config["fixed_exposure_sd_mmhg"]["delta_sigma"], reps)
    return rows, contrast, checks, counts, preprocessing


def analyze_timing(frame, out_dir, reference=None, reps=REPS, seed=SEED):
    """Analyze original dynamic cohort with conditional early and late risk sets.

    The two full interval models retain all 15 original planned parameters. No
    model selection, automatic covariate deletion, or redefinition of exposure
    windows is performed. Mean knots and reporting scales come from the original
    full dynamic source cohort. There are four exploratory delta-sigma estimates
    (BH family n=4) and two secondary paired bootstrap interval contrasts.
    """
    if reference is None:
        raise core.Nonestimable("reference_required: pass load_reference(<change-model output dir>)")
    if reps < 2:
        raise core.Nonestimable("insufficient_bootstrap_replicates")
    frame = prepare_frame(frame, reference)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    core.dump_json({"created_utc_before_fitting": datetime.now(timezone.utc).isoformat(),
                    "source_configuration": reference, "seed": seed, "bootstrap_replicates": reps,
                    "models": {key: list(EXPOSURES) for key in INTERVALS}, "parameters_per_model": 15,
                    "dynamic_additional_covariates": ["delta_mu", "log1p_old90_sessions"],
                    "bootstrap": "original source patients sampled identically across intervals; all eligible rows retained with multiplicity; imputation/scaling refit",
                    "mean_knots_and_reporting_sd": "fixed from prior original full dynamic cohort",
                    "primary_exploratory_family": "four delta_sigma coefficients; BH fixed n=4; CY primary scientific focus, TT operational exploratory support",
                    "secondary_contrasts": "two late-minus-early raw log odds ratio percentile CIs from paired patient bootstrap; unequal interval lengths and conditional populations",
                    "nonestimable_primary": "p=1 internally for BH; displayed p/q missing",
                    "no_patient_level_outputs": True}, out_dir / "timing_analysis_lock.json")
    rows, contrasts, checks, counts, preprocessing = [], [], [], [], []
    with threadpool_limits(limits=1):
        for source in SOURCES:
            print(f"Fitting fixed {source} early and late dynamic models; aggregate outputs only.", flush=True)
            values = analyze_source(frame.loc[frame.source.eq(source)], source, reference[source], reps, seed)
            for dest, value in zip((rows, contrasts, checks, counts, preprocessing), values):
                dest.extend(value if isinstance(value, list) else [value])
    result = pd.DataFrame(rows).sort_values(["source", "interval", "term"])
    contrast = pd.DataFrame(contrasts).sort_values("source")
    if "cluster_wald_p" not in result:
        result["cluster_wald_p"] = np.nan
    primary = result.analysis.eq("timing_change_exploratory")
    if int(primary.sum()) != 4:
        raise core.Nonestimable("unexpected_timing_primary_family_size")
    p = result.loc[primary, "cluster_wald_p"].to_numpy(float, copy=True)
    p[~result.loc[primary, "status"].eq("estimated").to_numpy()] = np.nan
    result.loc[primary, "bh_fdr_q_4_timing_tests"] = core.bh_fixed_family(p)
    primary_result = result.loc[primary].copy()
    prior_level_result = result.loc[~primary].copy()
    primary_result.to_csv(out_dir / "timing_associations.csv", index=False)
    prior_level_result.to_csv(out_dir / "timing_prior_level_coefficients.csv", index=False)
    contrast.to_csv(out_dir / "timing_contrasts.csv", index=False)
    pd.DataFrame(counts).to_csv(out_dir / "timing_analysis_counts.csv", index=False)
    pd.DataFrame(preprocessing).to_csv(out_dir / "timing_covariate_preprocessing.csv", index=False)
    core.dump_json(checks, out_dir / "timing_statsmodels_verification.json")
    return {"associations": primary_result, "prior_level_coefficients": prior_level_result,
            "contrasts": contrast, "verification": checks, "counts": pd.DataFrame(counts),
            "all_passed": bool(len(checks) == 4 and all(x["passed"] for x in checks) and len(result) == 8
                               and result.status.eq("estimated").all() and contrast.status.eq("estimated").all())}


def self_test(out_dir):
    rng = np.random.default_rng(SEED)
    frames, reference = [], {}
    patients, repeats = 260, 5
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
        eta = -1.8 + .55 * delta + .18 * (old - 10) + .01 * (f.mu_mean90 - 130)
        early = rng.binomial(1, 1 / (1 + np.exp(-eta)))
        late = rng.binomial(1, 1 / (1 + np.exp(-(eta + .4 - .25 * delta))))
        f["death_early30"] = early
        f["eligible_late"] = early == 0
        f["death_late31_90"] = np.where(early == 0, late, np.nan)
        f["death90"] = early + (1 - early) * late
        config = dynamic.make_source_configs()[source]
        reference[source] = {**config, "mean_rcs_knots_10_50_90": np.quantile(f.mu_mean90, [.1, .5, .9]).tolist(),
                             "fixed_exposure_sd_mmhg": {"delta_sigma": float(delta.std()), "sigma_old90": float(old.std())}}
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    result = analyze_timing(frame, out_dir, reference=reference, reps=8)
    assert result["all_passed"]
    assert len(result["associations"]) == 4
    assert len(result["prior_level_coefficients"]) == 4
    assert result["associations"].bootstrap_valid.eq(8).all()
    assert result["associations"].design_parameters.eq(15).all()
    assert result["contrasts"].paired_bootstrap_valid.eq(8).all()
    for source in SOURCES:
        selected = result["associations"].loc[lambda x: x.source.eq(source) & x.term.eq("delta_sigma")].set_index("interval")
        c = result["contrasts"].loc[lambda x: x.source.eq(source)]
        expected = selected.loc["days_31_90", "beta_per_mmhg"] - selected.loc["days_1_30", "beta_per_mmhg"]
        assert np.isclose(float(c.log_or_difference_per_mmhg.item()), expected)
        assert selected.fixed_original_cohort_sd_mmhg.nunique() == 1
        assert np.allclose(selected.or_per_fixed_sd, np.exp(selected.beta_per_mmhg * reference[source]["fixed_exposure_sd_mmhg"]["delta_sigma"]))
    # Independently reproduce the first paired replicate and both raw coefficients.
    f = prepare_frame(frame, reference).loc[lambda x: x.source.eq("CY")]
    raw, names = dynamic.model_design(f, "dynamic_primary", reference["CY"]["baseline_columns"], reference["CY"]["mean_rcs_knots_10_50_90"])
    masks = interval_masks(f)
    outcomes = {key: f[col].to_numpy(float) for key, col in OUTCOMES.items()}
    samples, failures = paired_cluster_bootstrap(raw, names, outcomes, masks, codes, 2, SEED, "synthetic")
    ix, chosen = core.sample_patient_indices(codes, np.random.default_rng(SEED))
    assert np.array_equal(np.bincount(codes[ix], minlength=patients), repeats * np.bincount(chosen, minlength=patients))
    for interval in INTERVALS:
        use = ix[masks[interval][ix]]
        fit = core.fit_array(raw[use], outcomes[interval][use], names, EXPOSURES, "mu_mean90")
        j = fit["names"].index("delta_sigma")
        assert np.isclose(samples[interval][0, 0], fit["beta"][j] / fit["scales"][j])
        assert not failures[interval]
    malformed = frame.copy()
    malformed.loc[malformed.death_early30.eq(1), "eligible_late"] = True
    try:
        prepare_frame(malformed, reference)
        raise AssertionError("Early deaths accepted in late risk set")
    except core.Nonestimable as exc:
        assert str(exc) == "late_risk_set_must_exclude_early_deaths_only_in_known90_cohort"
    malformed = frame.copy()
    malformed["death90"] = 1 - malformed.death90
    try:
        prepare_frame(malformed, reference)
        raise AssertionError("Inconsistent timing partition accepted")
    except core.Nonestimable as exc:
        assert str(exc) == "timing_partition_not_equal_original_death90"
    core.dump_json({"synthetic_only": True, "assertions_passed": True,
                    "checks": ["four_fixed_15_parameter_models_eight_coefficients", "independent_GLM_and_patient_cluster_SE",
                               "same_patient_multiplicities_across_intervals", "paired_replicate_index_retained",
                               "late_risk_set_excludes_early_deaths", "timing_partition_matches_death90",
                               "fixed_original_SD_for_both_intervals", "raw_log_OR_contrast_identity"],
                    "max_independent_coefficient_difference": max(c["max_coefficient_difference"] for c in result["verification"]),
                    "max_independent_exposure_SE_difference": max(c["max_component_cluster_se_difference"] for c in result["verification"])},
                   Path(out_dir) / "synthetic_verification.json")
    print("Synthetic timing statistics checks passed; no clinical data accessed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("This module takes an in-memory frame; use --self-test for synthetic checks.")
    self_test(Path(__file__).resolve().parent / "synthetic_statistics")
