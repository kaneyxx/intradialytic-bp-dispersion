"""Exact fixed-model retention summaries with paired whole-patient bootstrap.

No file I/O, model fitting, or individual-data output is performed here. Ties at
a retention boundary receive one common fractional retention weight: this is
the expectation over random ordering within a score tie, with an exact quota.
"""

from __future__ import annotations

import numpy as np


def _validate(scores, metrics, codes, quotas, reps):
    metrics = np.asarray(metrics, dtype=np.float64)
    codes = np.asarray(codes)
    quotas = np.asarray(quotas, dtype=np.float64)
    if metrics.ndim != 2 or metrics.shape[0] == 0:
        raise ValueError("metrics must be a nonempty row-by-metric matrix")
    if not np.isfinite(metrics).all():
        raise ValueError("metrics must all be finite on the common analysis set")
    if codes.ndim != 1 or len(codes) != len(metrics):
        raise ValueError("codes must have one entry per metric row")
    if not np.issubdtype(codes.dtype, np.integer) or codes.min() != 0:
        raise ValueError("codes must be contiguous nonnegative patient integers")
    codes = codes.astype(np.int64, copy=False)
    patients = int(codes.max()) + 1
    if len(np.unique(codes)) != patients:
        raise ValueError("codes must be contiguous patient integers")
    if quotas.ndim != 1 or not len(quotas):
        raise ValueError("quotas must be a nonempty one-dimensional array")
    if not np.isfinite(quotas).all() or np.any((quotas <= 0) | (quotas > 1)):
        raise ValueError("quotas must be finite fractions in (0, 1]")
    if np.any(np.diff(quotas) <= 0):
        raise ValueError("quotas must be strictly increasing")
    if not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError("reps must be a positive integer")
    if not scores or "random" in scores:
        raise ValueError("provide at least one score; random is a reserved name")
    clean_scores = {}
    for name, score in scores.items():
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (len(metrics),) or not np.isfinite(score).all():
            raise ValueError(f"score {name!r} must be finite and match all rows")
        clean_scores[name] = score
    return clean_scores, metrics, codes, quotas, patients


def _score_partition(score, metrics, codes, quotas, patient_weights):
    """Sort once; use exact block totals and resolve the boundary tie locally."""
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_codes = codes[order]
    sorted_metrics = metrics[order]
    n, metric_count = metrics.shape
    draw_count, patient_count = patient_weights.shape
    patient_maximum = np.full(patient_count, -np.inf)
    np.maximum.at(patient_maximum, sorted_codes, sorted_score)
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    group_ends = np.r_[group_starts[1:], n]

    # Blocks never split ties. The patient-by-block sufficient summaries are
    # merely a computational accelerator; no score or threshold is coarsened.
    max_blocks = max(1, min(256, 4_000_000 // (patient_count * (metric_count + 1))))
    desired_ends = np.ceil(np.linspace(0, n, max_blocks + 1)[1:]).astype(np.int64)
    end_groups = np.unique(np.searchsorted(group_ends, desired_ends, side="left"))
    block_ends = group_ends[end_groups]
    block_starts = np.r_[0, block_ends[:-1]]
    block_count = len(block_ends)
    block_ids = np.repeat(np.arange(block_count), block_ends - block_starts)
    patient_block_ids = sorted_codes * block_count + block_ids
    patient_blocks = np.empty((patient_count, block_count, metric_count + 1))
    patient_blocks[:, :, 0] = np.bincount(
        patient_block_ids, minlength=patient_count * block_count
    ).reshape(patient_count, block_count)
    for metric in range(metric_count):
        patient_blocks[:, :, metric + 1] = np.bincount(
            patient_block_ids,
            weights=sorted_metrics[:, metric],
            minlength=patient_count * block_count,
        ).reshape(patient_count, block_count)
    del patient_block_ids, block_ids
    cumulative_blocks = (
        patient_weights @ patient_blocks.reshape(patient_count, -1)
    ).reshape(draw_count, block_count, metric_count + 1)
    np.cumsum(cumulative_blocks, axis=1, out=cumulative_blocks)
    del patient_blocks

    out = np.full((draw_count, len(quotas), 2, metric_count), np.nan)
    thresholds = np.empty((draw_count, len(quotas)))
    achieved = np.empty((draw_count, len(quotas)))
    zero = np.zeros(metric_count + 1)
    for draw in range(draw_count):
        total_count = cumulative_blocks[draw, -1, 0]
        total_metrics = cumulative_blocks[draw, -1, 1:]
        cache = {}
        for qi, quota in enumerate(quotas):
            target = quota * total_count
            if quota == 1.0:
                out[draw, qi, 0] = total_metrics / total_count
                # Score maximum among rows actually present in this draw.
                thresholds[draw, qi] = patient_maximum[patient_weights[draw] > 0].max()
                achieved[draw, qi] = 1.0
                continue
            block = min(
                int(np.searchsorted(cumulative_blocks[draw, :, 0], target, side="left")),
                block_count - 1,
            )
            before = cumulative_blocks[draw, block - 1] if block else zero
            start, end = block_starts[block], block_ends[block]
            if block not in cache:
                local_weights = patient_weights[draw, sorted_codes[start:end]]
                first_group = np.searchsorted(group_starts, start)
                last_group = np.searchsorted(group_starts, end)
                local_starts = group_starts[first_group:last_group] - start
                local_group_weights = np.add.reduceat(local_weights, local_starts)
                local_cumulative = np.cumsum(local_group_weights)
                cache[block] = (local_weights, local_starts, local_cumulative)
            local_weights, local_starts, local_cumulative = cache[block]
            local_target = target - before[0]
            group = int(np.searchsorted(local_cumulative, local_target, side="left"))
            group = min(group, len(local_starts) - 1)
            tie_start = local_starts[group]
            tie_end = local_starts[group + 1] if group + 1 < len(local_starts) else end - start
            count_before_tie = local_cumulative[group - 1] if group else 0.0
            tie_count = local_cumulative[group] - count_before_tie
            if tie_count <= 0:
                raise ArithmeticError("quota boundary unexpectedly has no bootstrap weight")
            fraction = (local_target - count_before_tie) / tie_count
            if not -1e-10 <= fraction <= 1 + 1e-10:
                raise ArithmeticError("invalid fractional retention at a score tie")
            fraction = np.clip(fraction, 0.0, 1.0)
            local_metrics = sorted_metrics[start:end]
            retained_sum = before[1:] + np.einsum(
                "i,ij->j", local_weights[:tie_start], local_metrics[:tie_start]
            ) + fraction * np.einsum(
                "i,ij->j", local_weights[tie_start:tie_end], local_metrics[tie_start:tie_end]
            )
            retained_count = before[0] + count_before_tie + fraction * tie_count
            out[draw, qi, 0] = retained_sum / retained_count
            out[draw, qi, 1] = (total_metrics - retained_sum) / (total_count - retained_count)
            thresholds[draw, qi] = sorted_score[start + tie_start]
            achieved[draw, qi] = retained_count / total_count
    if not np.allclose(achieved, quotas[None, :], rtol=0, atol=2e-13):
        raise ArithmeticError("retention quota was not achieved exactly")
    return {
        "point": out[0],
        "bootstrap": out[1:],
        "thresholds": thresholds[0],
        "bootstrap_thresholds": thresholds[1:],
        "achieved_fraction": achieved[0],
        "bootstrap_achieved_fraction": achieved[1:],
        "score_tie_groups": int(len(group_starts)),
        "computation_blocks": int(block_count),
    }


def analyze(scores, metrics, codes, quotas, reps=1000, seed=20260927):
    """Return paired retention metrics for each score and analytic random selection.

    Parameters
    ----------
    scores : mapping of string to row-length finite arrays
        Smaller values are retained first. Conventionally sigma and abs_delta.
    metrics : array, shape (rows, metrics)
        Each column is averaged. Expected caller order is absolute error,
        coverage95, below95, above95, width95, abs_error_gt20,
        target_sbp_lt90, sigma; the calculation is column-agnostic.
    codes : contiguous integer array, shape (rows,)
        Local patient grouping only; not returned or written anywhere.
    quotas : increasing fractions in (0, 1]
        Proportion of forecast rows retained, distinct from interval coverage.
    reps, seed : int
        Multinomial resampling draws as many patients as were observed.

    Returns
    -------
    dict
        selectors[name]['point'] has shape (quotas, 2, metrics), bootstrap
        has shape (reps, quotas, 2, metrics). Group 0 is retained and 1 flagged.
        At quota=1, the empty flagged group is NaN. Every selector uses the
        same patient draws. Random gives the analytic expected full-cohort
        mean in each nonempty group rather than a noisy random ordering.
        Thresholds are recomputed per draw, including the fraction of ties.

    Notes
    -----
    Forecast rows, not patients, receive equal metric weight. Resampling is
    by whole patient, including all their rows and any duplicate appearances.
    The model is frozen, so intervals capture test-cohort sampling variation,
    not training variation or uncertainty from changing the selection model.
    """
    scores, metrics, codes, quotas, patient_count = _validate(scores, metrics, codes, quotas, reps)
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(patient_count, np.full(patient_count, 1.0 / patient_count), size=reps)
    patient_weights = np.vstack([np.ones(patient_count), sampled]).astype(np.float64)
    del sampled
    selectors = {}
    for name, score in scores.items():
        selectors[name] = _score_partition(score, metrics, codes, quotas, patient_weights)

    per_patient_count = np.bincount(codes, minlength=patient_count)
    per_patient_metrics = np.column_stack([
        np.bincount(codes, weights=metrics[:, j], minlength=patient_count)
        for j in range(metrics.shape[1])
    ])
    totals = patient_weights @ per_patient_metrics
    counts = patient_weights @ per_patient_count
    full_mean = totals / counts[:, None]
    random_out = np.broadcast_to(
        full_mean[:, None, None, :], (reps + 1, len(quotas), 2, metrics.shape[1])
    ).copy()
    random_out[:, quotas == 1.0, 1, :] = np.nan
    selectors["random"] = {
        "point": random_out[0],
        "bootstrap": random_out[1:],
        "thresholds": np.full(len(quotas), np.nan),
        "bootstrap_thresholds": np.full((reps, len(quotas)), np.nan),
        "achieved_fraction": quotas.copy(),
        "bootstrap_achieved_fraction": np.broadcast_to(quotas, (reps, len(quotas))).copy(),
    }
    # For every bootstrap draw, retained and flagged groups must reconstruct
    # the full cohort at their exact specified retention/flagging proportions.
    active = quotas < 1
    for name, result in selectors.items():
        combined = np.concatenate([result["point"][None], result["bootstrap"]])
        reconstructed = (
            quotas[None, active, None] * combined[:, active, 0, :]
            + (1 - quotas[None, active, None]) * combined[:, active, 1, :]
        )
        if not np.allclose(reconstructed, full_mean[:, None, :], rtol=2e-11, atol=2e-11):
            raise ArithmeticError(f"retained/flagged complement identity failed: {name}")
    return {
        "selectors": selectors,
        "quotas": quotas.copy(),
        "metric_count": int(metrics.shape[1]),
        "reps": int(reps),
        "seed": int(seed),
        "group_order": ["retained", "flagged"],
        "bootstrap_method": "paired whole-patient multinomial; score thresholds recomputed per draw",
        "tie_method": "exact quota with uniform fractional retention within each boundary score tie",
        "random_method": "analytic expectation conditional on each bootstrap sample",
        "checks": {"exact_retention_quotas": True, "retained_flagged_complement_identity": True},
    }


def _reference(score, metrics, codes, quotas, patient_weights):
    """Independent tiny-data reference, grouping unique scores directly."""
    row_weights = patient_weights[codes]
    total_n = row_weights.sum()
    total_sum = (row_weights[:, None] * metrics).sum(axis=0)
    results = np.full((len(quotas), 2, metrics.shape[1]), np.nan)
    for qi, quota in enumerate(quotas):
        target = quota * total_n
        retained_n = 0.0
        retained_sum = np.zeros(metrics.shape[1])
        for value in np.unique(score):
            tie = score == value
            tie_n = row_weights[tie].sum()
            if not tie_n:
                continue
            fraction = min(1.0, max(0.0, (target - retained_n) / tie_n))
            retained_sum += fraction * (row_weights[tie, None] * metrics[tie]).sum(axis=0)
            retained_n += fraction * tie_n
        results[qi, 0] = retained_sum / target
        if quota < 1:
            results[qi, 1] = (total_sum - retained_sum) / (total_n - target)
    return results


def synthetic_verification():
    """Meaningful tie, quota, complement and cluster-duplication checks only."""
    score = np.array([1, 1, 2, 3, 3, 3], dtype=float)
    delta = np.array([0, 1, 1, 2, 3, 3], dtype=float)
    codes = np.array([0, 0, 1, 1, 2, 2])
    metrics = np.column_stack([np.arange(1, 7), np.array([0, 1, 0, 1, 1, 0])]).astype(float)
    quotas = np.array([0.5, 0.8, 1.0])
    reps, seed = 31, 17
    result = analyze({"sigma": score, "abs_delta": delta}, metrics, codes, quotas, reps=reps, seed=seed)
    np.testing.assert_allclose(result["selectors"]["sigma"]["point"][1, 0, 0], 3.125)
    draws = np.random.default_rng(seed).multinomial(3, np.full(3, 1.0 / 3), size=reps)
    for name, values in {"sigma": score, "abs_delta": delta}.items():
        actual = result["selectors"][name]
        np.testing.assert_allclose(actual["point"], _reference(values, metrics, codes, quotas, np.ones(3)), equal_nan=True)
        for draw, weights in enumerate(draws):
            expected = _reference(values, metrics, codes, quotas, weights)
            np.testing.assert_allclose(actual["bootstrap"][draw], expected, equal_nan=True)
            # Literally duplicate every row of a sampled patient and compare
            # with the weighted-bootstrap result (not row-wise resampling).
            duplicated = np.repeat(np.arange(len(codes)), weights[codes])
            duplicate_reference = _reference(
                values[duplicated], metrics[duplicated], np.zeros(len(duplicated), dtype=int),
                quotas, np.ones(1),
            )
            np.testing.assert_allclose(actual["bootstrap"][draw], duplicate_reference, equal_nan=True)
        np.testing.assert_allclose(actual["bootstrap_achieved_fraction"], np.broadcast_to(quotas, (reps, len(quotas))))
        if not np.isnan(actual["point"][-1, 1]).all():
            raise AssertionError("full-retention flagged group must be undefined")
    random = result["selectors"]["random"]
    for draw, weights in enumerate(draws):
        expected = np.average(metrics, weights=weights[codes], axis=0)
        np.testing.assert_allclose(random["bootstrap"][draw, :, 0], np.broadcast_to(expected, (len(quotas), len(expected))))
    # Also exercise many computational blocks with continuous scores and
    # unequal numbers of rows per cluster; no clinical data are used.
    rng = np.random.default_rng(55)
    larger_codes = np.repeat(np.arange(19), np.arange(2, 21))
    larger_scores = {"sigma": rng.normal(size=len(larger_codes)), "abs_delta": rng.integers(0, 8, size=len(larger_codes)).astype(float)}
    larger_metrics = rng.uniform(0, 20, size=(len(larger_codes), 3))
    larger = analyze(larger_scores, larger_metrics, larger_codes, quotas, reps=23, seed=39)
    larger_draws = np.random.default_rng(39).multinomial(19, np.full(19, 1 / 19), size=23)
    for name, values in larger_scores.items():
        for draw, weights in enumerate(larger_draws):
            np.testing.assert_allclose(larger["selectors"][name]["bootstrap"][draw], _reference(values, larger_metrics, larger_codes, quotas, weights), atol=1e-11, equal_nan=True)
    return {
        "synthetic_only": True,
        "boundary_ties_fractional_and_exact": True,
        "all_quotas_exact": True,
        "complement_identity": True,
        "whole_patient_duplication_matches_weighted_reference": True,
        "random_matches_analytic_full_cohort_mean": True,
        "unequal_cluster_sizes_and_multiple_blocks": True,
        "full_retention_empty_flagged_group_is_nan": True,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(synthetic_verification(), indent=2))
