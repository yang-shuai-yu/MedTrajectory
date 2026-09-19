from __future__ import annotations

import bisect
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


DAYS_PER_YEAR = 365.25
MASK_TIME = -10000.0


def compute_midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        midranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    out = np.empty(len(values), dtype=np.float64)
    out[order] = midranks + 1.0
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions_sorted_transposed, dtype=np.float64)
    positive_count = int(label_1_count)
    negative_count = predictions.shape[1] - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("DeLong variance requires at least two cases and two controls")

    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    classifier_count = predictions.shape[0]
    positive_ranks = np.empty((classifier_count, positive_count), dtype=np.float64)
    negative_ranks = np.empty((classifier_count, negative_count), dtype=np.float64)
    all_ranks = np.empty((classifier_count, positive_count + negative_count), dtype=np.float64)
    for classifier_idx in range(classifier_count):
        positive_ranks[classifier_idx] = compute_midrank(positive[classifier_idx])
        negative_ranks[classifier_idx] = compute_midrank(negative[classifier_idx])
        all_ranks[classifier_idx] = compute_midrank(predictions[classifier_idx])

    aucs = (
        all_ranks[:, :positive_count].sum(axis=1) / positive_count / negative_count
        - (positive_count + 1.0) / 2.0 / negative_count
    )
    v01 = (all_ranks[:, :positive_count] - positive_ranks) / negative_count
    v10 = 1.0 - (all_ranks[:, positive_count:] - negative_ranks) / positive_count
    covariance = np.atleast_2d(np.cov(v01)) / positive_count + np.atleast_2d(np.cov(v10)) / negative_count
    return aucs, covariance


def binary_auc(control_scores: np.ndarray, case_scores: np.ndarray) -> float:
    controls = np.asarray(control_scores, dtype=np.float64)
    cases = np.asarray(case_scores, dtype=np.float64)
    if len(controls) == 0 or len(cases) == 0:
        return float("nan")
    combined = np.concatenate([cases, controls])
    ranks = compute_midrank(combined)
    return float((ranks[: len(cases)].sum() - len(cases) * (len(cases) + 1) / 2.0) / len(cases) / len(controls))


def get_auc_delong_var(control_scores: np.ndarray, case_scores: np.ndarray) -> tuple[float, float]:
    controls = np.asarray(control_scores, dtype=np.float64)
    cases = np.asarray(case_scores, dtype=np.float64)
    auc = binary_auc(controls, cases)
    if len(controls) < 2 or len(cases) < 2:
        return auc, float("nan")
    ground_truth = np.concatenate([np.ones(len(cases), dtype=np.int8), np.zeros(len(controls), dtype=np.int8)])
    predictions = np.concatenate([cases, controls])
    order = np.argsort(-ground_truth, kind="mergesort")
    aucs, covariance = fast_delong(predictions[np.newaxis, order], len(cases))
    return float(aucs[0]), float(covariance[0, 0])


def validate_age_groups(age_groups: Sequence[float]) -> tuple[np.ndarray, float]:
    groups = np.asarray(age_groups, dtype=np.float64)
    if groups.ndim != 1 or len(groups) < 2:
        raise ValueError("age_groups must contain at least two values")
    steps = np.diff(groups)
    if not np.all(steps > 0) or not np.allclose(steps, steps[0]):
        raise ValueError("age_groups must be strictly increasing and equally spaced")
    return groups, float(steps[0])


def precompute_official_prediction_indices(
    input_ages: np.ndarray,
    target_ages: np.ndarray,
    offset_days: float,
) -> np.ndarray:
    input_ages = np.asarray(input_ages)
    target_ages = np.asarray(target_ages)
    return (input_ages[:, :, np.newaxis] < target_ages[:, np.newaxis, :] - float(offset_days)).sum(axis=1) - 1


def extract_official_lm_records(
    targets: np.ndarray,
    input_ages: np.ndarray,
    target_ages: np.ndarray,
    token_scores: np.ndarray,
    token_id: int,
    patient_ids: np.ndarray,
    offset_days: float = 0.1,
    prediction_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    targets = np.asarray(targets)
    input_ages = np.asarray(input_ages)
    target_ages = np.asarray(target_ages)
    token_scores = np.asarray(token_scores)
    patient_ids = np.asarray(patient_ids)
    if targets.shape != input_ages.shape or targets.shape != target_ages.shape or targets.shape != token_scores.shape:
        raise ValueError("targets, ages, and token_scores must have the same [patient, position] shape")
    if len(patient_ids) != targets.shape[0]:
        raise ValueError("patient_ids must align with the first tensor dimension")

    case_rows, case_positions = np.where(targets == int(token_id))
    if len(case_rows) < 2:
        return {key: np.asarray([], dtype=dtype) for key, dtype in (
            ("scores", np.float64),
            ("labels", np.int8),
            ("prediction_age_days", np.float64),
            ("patient_ids", patient_ids.dtype),
        )}

    control_patient_mask = ~(targets == int(token_id)).any(axis=1)
    control_rows, control_positions = np.where(control_patient_mask[:, np.newaxis] & (targets != int(token_id)))
    rows = np.concatenate([case_rows, control_rows])
    positions = np.concatenate([case_positions, control_positions])
    labels = np.concatenate([np.ones(len(case_rows), dtype=np.int8), np.zeros(len(control_rows), dtype=np.int8)])

    if prediction_indices is None:
        prediction_indices = precompute_official_prediction_indices(input_ages, target_ages, offset_days)
    pred_idx = np.asarray(prediction_indices)[rows, positions]
    valid = pred_idx != -1
    rows = rows[valid]
    pred_idx = pred_idx[valid]
    labels = labels[valid]
    return {
        "scores": token_scores[rows, pred_idx].astype(np.float64, copy=False),
        "labels": labels,
        "prediction_age_days": input_ages[rows, pred_idx].astype(np.float64, copy=False),
        "patient_ids": patient_ids[rows],
    }


def age_stratified_delong_rows(
    scores: np.ndarray,
    labels: np.ndarray,
    prediction_age_days: np.ndarray,
    patient_ids: np.ndarray,
    age_groups: Sequence[float],
    rng: np.random.Generator,
    metadata: Mapping[str, object] | None = None,
) -> list[dict]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    ages = np.asarray(prediction_age_days, dtype=np.float64) / DAYS_PER_YEAR
    patient_ids = np.asarray(patient_ids)
    if not (len(scores) == len(labels) == len(ages) == len(patient_ids)):
        raise ValueError("scores, labels, ages, and patient_ids must have equal lengths")
    groups, age_step = validate_age_groups(age_groups)
    base = dict(metadata or {})
    rows: list[dict] = []

    for age_start in groups:
        in_bin = (ages >= age_start) & (ages < age_start + age_step)
        selected_idx = np.flatnonzero(in_bin)
        if len(selected_idx) == 0:
            continue
        selected_idx = rng.permutation(selected_idx)
        _, unique_positions = np.unique(patient_ids[selected_idx], return_index=True)
        selected_idx = selected_idx[unique_positions]
        selected_labels = labels[selected_idx]
        control = scores[selected_idx][selected_labels == 0]
        case = scores[selected_idx][selected_labels == 1]
        if len(control) == 0 or len(case) == 0:
            continue
        auc, variance = get_auc_delong_var(control, case)
        rows.append(
            {
                **base,
                "age_start_years": float(age_start),
                "age_end_years": float(age_start + age_step),
                "auc": auc,
                "auc_delong": auc,
                "auc_variance_delong": variance,
                "n_controls": int(len(control)),
                "n_cases": int(len(case)),
            }
        )
    return rows


def select_age_landmarks(
    input_tokens: np.ndarray,
    input_ages: np.ndarray,
    patient_ids: np.ndarray,
    age_groups: Sequence[float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    input_tokens = np.asarray(input_tokens)
    input_ages = np.asarray(input_ages)
    patient_ids = np.asarray(patient_ids)
    if input_tokens.shape != input_ages.shape or len(patient_ids) != input_tokens.shape[0]:
        raise ValueError("input tensors and patient_ids are not aligned")
    groups, age_step = validate_age_groups(age_groups)
    selected_rows: list[int] = []
    selected_positions: list[int] = []
    selected_bins: list[float] = []

    age_years = input_ages / DAYS_PER_YEAR
    valid = (input_tokens > 0) & (input_ages > MASK_TIME / 2)
    for row_idx in range(input_tokens.shape[0]):
        for age_start in groups:
            candidates = np.flatnonzero(valid[row_idx] & (age_years[row_idx] >= age_start) & (age_years[row_idx] < age_start + age_step))
            if len(candidates) == 0:
                continue
            selected_rows.append(row_idx)
            selected_positions.append(int(rng.choice(candidates)))
            selected_bins.append(float(age_start))
    rows = np.asarray(selected_rows, dtype=np.int64)
    positions = np.asarray(selected_positions, dtype=np.int64)
    return {
        "row_indices": rows,
        "positions": positions,
        "patient_ids": patient_ids[rows],
        "prediction_age_days": input_ages[rows, positions].astype(np.float64, copy=False),
        "age_start_years": np.asarray(selected_bins, dtype=np.float64),
    }


def select_shared_age_landmarks(
    input_tokens: np.ndarray,
    input_ages: np.ndarray,
    patient_ids: np.ndarray,
    shared_landmarks: Mapping[int, Sequence[Mapping[str, float]]],
) -> dict[str, np.ndarray]:
    """Resolve frozen target ages against a profile-specific causal history.

    ``shared_landmarks`` is generated once for the locked test split. Each
    profile may have different event positions, so the selected model position
    is the last valid position at or before the frozen target age. Labels are
    evaluated at the frozen target age, making rows pairable across profiles.
    """
    input_tokens = np.asarray(input_tokens)
    input_ages = np.asarray(input_ages)
    patient_ids = np.asarray(patient_ids)
    if input_tokens.shape != input_ages.shape or len(patient_ids) != input_tokens.shape[0]:
        raise ValueError("input tensors and patient_ids are not aligned")
    valid = (input_tokens > 0) & (input_ages > MASK_TIME / 2)
    selected_rows: list[int] = []
    selected_positions: list[int] = []
    selected_bins: list[float] = []
    target_ages: list[float] = []
    position_ages: list[float] = []
    row_lookup = {int(patient_id): row_idx for row_idx, patient_id in enumerate(patient_ids.tolist())}
    for patient_id, entries in shared_landmarks.items():
        row_idx = row_lookup.get(int(patient_id))
        if row_idx is None:
            continue
        for entry in entries:
            age_start = float(entry["age_start_years"])
            target_age = float(entry["target_age_days"])
            lower_age = age_start * DAYS_PER_YEAR
            frozen_position = entry.get("position")
            if frozen_position is not None:
                position = int(frozen_position)
                if (
                    0 <= position < input_tokens.shape[1]
                    and valid[row_idx, position]
                    and np.isclose(input_ages[row_idx, position], target_age, atol=0.25)
                ):
                    selected_rows.append(row_idx)
                    selected_positions.append(position)
                    selected_bins.append(age_start)
                    target_ages.append(target_age)
                    position_ages.append(float(input_ages[row_idx, position]))
                    continue
            candidates = np.flatnonzero(valid[row_idx] & (input_ages[row_idx] >= lower_age) & (input_ages[row_idx] <= target_age + 0.25))
            if len(candidates) == 0:
                continue
            exact_no_event = candidates[(input_tokens[row_idx, candidates] == 1) & np.isclose(input_ages[row_idx, candidates], target_age, atol=0.25)]
            if len(exact_no_event):
                same_age = exact_no_event
                max_age = target_age
            else:
                max_age = float(input_ages[row_idx, candidates].max())
                same_age = candidates[np.isclose(input_ages[row_idx, candidates], max_age, atol=0.25)]
            selected_rows.append(row_idx)
            selected_positions.append(int(same_age[-1]))
            selected_bins.append(age_start)
            target_ages.append(target_age)
            position_ages.append(max_age)
    rows = np.asarray(selected_rows, dtype=np.int64)
    positions = np.asarray(selected_positions, dtype=np.int64)
    return {
        "row_indices": rows,
        "positions": positions,
        "patient_ids": patient_ids[rows],
        "prediction_age_days": np.asarray(target_ages, dtype=np.float64),
        "position_age_days": np.asarray(position_ages, dtype=np.float64),
        "age_start_years": np.asarray(selected_bins, dtype=np.float64),
    }


def build_horizon_case_control(
    patient_ids: np.ndarray,
    prediction_age_days: np.ndarray,
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    patient_last_ages: np.ndarray,
    horizon_years: float,
    disease_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    patient_ids = np.asarray(patient_ids, dtype=np.int64)
    prediction_ages = np.asarray(prediction_age_days, dtype=np.float64)
    horizon_days = float(horizon_years) * DAYS_PER_YEAR
    labels = np.zeros(len(patient_ids), dtype=np.int8)
    eligible = np.zeros(len(patient_ids), dtype=bool)

    for idx, (patient_id, current_age) in enumerate(zip(patient_ids, prediction_ages)):
        label, is_eligible = horizon_case_control_at_age(
            current_age,
            patient_disease_ages[int(patient_id)][int(disease_idx)],
            float(patient_last_ages[int(patient_id)]),
            horizon_years,
        )
        labels[idx] = label
        eligible[idx] = is_eligible
    return labels, eligible


def horizon_case_control_at_age(
    current_age_days: float,
    disease_ages: Sequence[float],
    followup_end_age_days: float,
    horizon_years: float,
) -> tuple[int, bool]:
    """Return a censor-aware fixed-horizon label and eligibility flag."""
    current_age = float(current_age_days)
    followup_end = float(followup_end_age_days)
    horizon_days = float(horizon_years) * DAYS_PER_YEAR
    next_idx = bisect.bisect_right(disease_ages, current_age)
    if next_idx < len(disease_ages):
        event_age = float(disease_ages[next_idx])
        if event_age <= followup_end and event_age - current_age <= horizon_days:
            return 1, True
    return 0, followup_end - current_age >= horizon_days


def aggregate_age_brackets_delong(rows: Sequence[Mapping[str, object]], group_keys: Sequence[str]) -> list[dict]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    output: list[dict] = []
    for key_values, group in grouped.items():
        aucs = np.asarray([float(row["auc_delong"]) for row in group], dtype=np.float64)
        variances = np.asarray([float(row["auc_variance_delong"]) for row in group], dtype=np.float64)
        finite_variances = variances[np.isfinite(variances)]
        item = {key: value for key, value in zip(group_keys, key_values)}
        item.update(
            {
                "auc": float(np.mean(aucs)),
                "auc_variance_delong": (
                    float(finite_variances.sum() / (len(group) ** 2)) if len(finite_variances) == len(group) else float("nan")
                ),
                "n_age_sex_cells": int(len(group)),
                "n_cases": int(sum(int(row["n_cases"]) for row in group)),
                "n_controls": int(sum(int(row["n_controls"]) for row in group)),
            }
        )
        output.append(item)
    return output
