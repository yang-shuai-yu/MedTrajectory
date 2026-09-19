"""Metrics and paired bootstrap helpers for fixed-horizon control tasks."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    positive = labels == 1
    negative = labels == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    positive = int(labels.sum())
    if positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hit = labels[order].astype(bool)
    ranks = np.flatnonzero(hit) + 1
    return float((np.arange(1, positive + 1) / ranks).mean())


def _expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _logistic_calibration(logit_scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit observed label ~ intercept + slope * model logit by IRLS."""
    if labels.min() == labels.max():
        return float("nan"), float("nan")
    design = np.column_stack([np.ones(len(logit_scores)), logit_scores])
    beta = np.zeros(2, dtype=np.float64)
    ridge = np.diag([1e-8, 1e-8])
    for _ in range(50):
        probability = _expit(design @ beta)
        weights = np.maximum(probability * (1.0 - probability), 1e-8)
        working = design @ beta + (labels - probability) / weights
        lhs = design.T @ (weights[:, None] * design) + ridge
        rhs = design.T @ (weights * working)
        updated = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    return float(beta[0]), float(beta[1])


def calibration_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
    probability_scores: bool = True,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if labels.size == 0:
        return {"brier": float("nan"), "ece": float("nan"), "calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    if not probability_scores:
        return {"brier": float("nan"), "ece": float("nan"), "calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    scores = np.clip(scores, 0.0, 1.0)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for idx in range(bins):
        mask = (scores >= edges[idx]) & (scores <= edges[idx + 1] if idx == bins - 1 else scores < edges[idx + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    logit = np.log(np.clip(scores, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - scores, 1e-6, 1.0))
    intercept, slope = _logistic_calibration(logit, labels.astype(np.float64))
    return {
        "brier": float(np.mean((scores - labels) ** 2)),
        "ece": float(ece),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def summarize_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
    probability_scores: bool = True,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    out = {"auc": binary_auc(scores, labels), "auprc": average_precision(scores, labels)}
    out.update(calibration_metrics(scores, labels, bins=bins, probability_scores=probability_scores))
    out["n"] = int(labels.size)
    out["cases"] = int(labels.sum())
    return out


def paired_patient_bootstrap(
    rows: Sequence[Mapping[str, object]],
    model_scores: Mapping[str, str],
    model_a: str,
    model_b: str,
    patient_key: str = "patient_index",
    label_key: str = "label",
    bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap AUC difference on the same patient rows for two models."""
    patients = np.asarray(sorted({int(row[patient_key]) for row in rows}), dtype=np.int64)
    groups = {int(pid): [row for row in rows if int(row[patient_key]) == int(pid)] for pid in patients}
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(int(bootstrap)):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        a_scores: list[float] = []
        b_scores: list[float] = []
        labels: list[int] = []
        for pid in sampled:
            for row in groups[int(pid)]:
                labels.append(int(row[label_key]))
                a_scores.append(float(row[model_scores[model_a]]))
                b_scores.append(float(row[model_scores[model_b]]))
        a = binary_auc(np.asarray(a_scores), np.asarray(labels))
        b = binary_auc(np.asarray(b_scores), np.asarray(labels))
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(float(a - b))
    if not deltas:
        return {"delta_auc": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "bootstrap_used": 0}
    full_labels = np.asarray([int(row[label_key]) for row in rows], dtype=np.int8)
    full_a = np.asarray([float(row[model_scores[model_a]]) for row in rows], dtype=np.float64)
    full_b = np.asarray([float(row[model_scores[model_b]]) for row in rows], dtype=np.float64)
    return {
        "delta_auc": float(binary_auc(full_a, full_labels) - binary_auc(full_b, full_labels)),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_used": len(deltas),
        "patient_count": int(len(patients)),
    }
