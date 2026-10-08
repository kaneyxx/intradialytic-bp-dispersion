"""Aggregate, outcome-free descriptions of the fixed sigma contributions.

No patient rows are written. Slopes/correlations are descriptive and have no p values.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

PATIENT = "patient_token_v1"
GROUPS = ["group_bp", "group_treatment", "group_demographic", "group_labs"]


def spline_design(mu, knots):
    lo, mid, hi = np.asarray(knots, float)
    assert lo < mid < hi, "coincident_mu_knots"
    cube = lambda x: np.maximum(x, 0.0) ** 3
    nonlin = (cube(mu-lo) - cube(mu-mid)*(hi-lo)/(hi-mid)
              + cube(mu-hi)*(mid-lo)/(hi-mid)) / (hi-lo)**2
    raw = np.column_stack([mu, nonlin])
    scales = raw.std(axis=0)
    assert (scales > 1e-12).all(), "constant_mean_design"
    return np.column_stack([np.ones(len(mu)), (raw-raw.mean(axis=0))/scales])


def residual_pair(x, sigma, design):
    """Frisch-Waugh-Lovell description on the same finite observations."""
    valid = np.isfinite(x) & np.isfinite(sigma)
    xx, ss, zz = x[valid], sigma[valid], design[valid]
    if len(xx) <= zz.shape[1] or np.linalg.matrix_rank(zz) != zz.shape[1]:
        return {"profile_status": "insufficient_or_degenerate_mean_support"}
    resid = np.column_stack([xx, ss]) - zz @ np.linalg.lstsq(zz, np.column_stack([xx, ss]), rcond=None)[0]
    rx, rs = resid[:, 0], resid[:, 1]
    sx, st = rx.std(ddof=0), rs.std(ddof=0)
    if sx <= max(1e-12, xx.std(ddof=0)*1e-10) or st <= max(1e-12, ss.std(ddof=0)*1e-10):
        return {"profile_status": "constant_or_near_constant_residual"}
    cov = float(np.mean((rx-rx.mean())*(rs-rs.mean())))
    return {"profile_status": "descriptive", "mean_adjusted_correlation": cov/(sx*st),
            "difference_per_sigma_residual_sd": cov/st,
            "sigma_residual_sd_mmhg": float(st)}


def bucket(n):
    n = int(n)
    return "0" if n == 0 else "<10" if n < 10 else f"{n//10*10}-{n//10*10+9}"


def feature_kind(name):
    if name.startswith("ls_c_"):
        return "feature_contribution"
    if name.startswith("group_"):
        return "group_contribution"
    if name.startswith("raw_missing_fraction_"):
        return "input_missing_fraction"
    if name.startswith("raw_mean_"):
        return "clinical_input_window_summary"
    return "sigma_summary"


def describe_source(source, frame):
    assert frame[PATIENT].notna().all()
    mu, sigma = frame.mu_mean90.to_numpy(float), frame.sigma_mean90.to_numpy(float)
    assert np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()
    knots = np.quantile(mu, [.1, .5, .9])
    design = spline_design(mu, knots)
    data = frame.copy()
    data["pooling_gap"] = np.log(sigma) - data.mean_raw_log_sigma90.to_numpy(float)
    variables = ["sigma_mean90", "mean_raw_log_sigma90", "pooling_gap"] + GROUPS
    variables += sorted(c for c in data if c.startswith(("ls_c_", "raw_mean_", "raw_missing_fraction_")))
    assert len([c for c in variables if c.startswith("ls_c_")]) == 13
    records = []
    for variable in variables:
        values = data[variable].to_numpy(float)
        valid = np.isfinite(values)
        usable = values[valid]
        available_patients = data.loc[valid, PATIENT].nunique()
        missing_patients = data.loc[~valid, PATIENT].nunique()
        record = {"source": source, "variable": variable, "kind": feature_kind(variable),
                  "available_windows_bucket": bucket(valid.sum()),
                  "available_patients_bucket": bucket(available_patients),
                  "missing_patients_bucket": bucket(missing_patients)}
        # An exact zero or one missingness/sex fraction can encode a rare person.
        protected = available_patients < 10
        if variable.startswith("raw_missing_fraction_") or variable == "raw_mean_sex_female":
            for selection in [(values > 0) & valid, (values < 1) & valid]:
                n = data.loc[selection, PATIENT].nunique()
                protected |= 0 < n < 10
        if protected:
            record["profile_status"] = "privacy_suppressed_or_no_data"
            records.append(record)
            continue
        record.update(p10=float(np.quantile(usable, .1)), median=float(np.median(usable)),
                      p90=float(np.quantile(usable, .9)))
        if variable != "raw_mean_age_at_session_years":
            record.update(mean=float(usable.mean()), window_sd=float(usable.std(ddof=0)))
        if record["kind"] in ["feature_contribution", "group_contribution"]:
            record["mean_absolute_reference_centered_contribution"] = float(np.abs(usable).mean())
        record.update(residual_pair(values, sigma, design))
        if variable == "pooling_gap" and usable.std(ddof=0) > 1e-12:
            record["unadjusted_correlation_with_sigma"] = float(np.corrcoef(values, sigma)[0, 1])
        records.append(record)
    table = pd.DataFrame(records)
    mask = table.kind.eq("feature_contribution")
    table.loc[mask, "within_cohort_sd_rank"] = table.loc[mask, "window_sd"].rank(ascending=False, method="min")
    total = data[[c for c in data if c.startswith("ls_c_")]].sum(axis=1).to_numpy(float)
    reference = data.mean_raw_log_sigma90.to_numpy(float)-total
    group_error = float(np.max(np.abs(data[GROUPS].sum(axis=1).to_numpy(float)-total)))
    reference_range = float(np.ptp(reference))
    assert group_error < 1e-10 and reference_range < 1e-5, "contribution_partition_or_reference_mismatch"
    check = {"source": source, "mu_knots": knots.tolist(), "group_partition_max_error": group_error,
             "raw_log_sd_reference": float(np.median(reference)), "raw_log_sd_reference_range": reference_range,
             "same_mean_design_for_feature_and_sigma": True, "outcome_used_in_profile": False}
    return table, check


def run_profiles(frames, output_dir):
    tables, checks = [], []
    for source, frame in frames.items():
        table, check = describe_source(source, frame)
        tables.append(table)
        checks.append(check)
    result = pd.concat(tables, ignore_index=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(output_dir)/"component_profiles.csv", index=False)
    (Path(output_dir)/"profile_verification.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    return result, checks


def self_test():
    rng = np.random.default_rng(20260926)
    mu = rng.normal(130, 15, 1000)
    z = spline_design(mu, np.quantile(mu, [.1, .5, .9]))
    residual = rng.normal(size=1000)
    residual -= z @ np.linalg.lstsq(z, residual, rcond=None)[0]
    sigma = 15+.02*mu+residual
    x = 2.5*residual + z @ np.array([3., 2., -1.])
    out = residual_pair(x, sigma, z)
    assert abs(out["mean_adjusted_correlation"]-1) < 1e-12
    assert abs(out["difference_per_sigma_residual_sd"]-2.5*residual.std()) < 1e-10
    x[::7] = np.nan
    out = residual_pair(x, sigma, z)
    assert abs(out["mean_adjusted_correlation"]-1) < 1e-12
    assert residual_pair(np.ones(1000), sigma, z)["profile_status"] == "constant_or_near_constant_residual"
    return {"passed": True, "checks": ["known_conditional_slope", "same_subset_residualization", "constant_residual"]}


if __name__ == "__main__":
    print(json.dumps(self_test()))
