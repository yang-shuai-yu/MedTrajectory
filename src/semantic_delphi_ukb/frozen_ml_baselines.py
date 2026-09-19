from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in [ROOT, ROOT / "src", ROOT / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    parse_selected_diseases,
    token_ids_for_disease,
)
from utils import get_p2i  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:  # pragma: no cover
    LogisticRegression = None
    HistGradientBoostingClassifier = None


TYPE_NAMES = ["diagnosis", "procedure", "cancer", "death"]
TYPE_TO_INDEX = {name: idx for idx, name in enumerate(TYPE_NAMES)}


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_horizons(spec: str) -> list[float]:
    values = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    return [value for value in values if value > 0]


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static_matrix = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if len(p2i) != len(static_matrix):
        raise RuntimeError(f"{split}: p2i rows={len(p2i)} but static rows={len(static_matrix)}")
    return data, p2i, static_matrix


def load_static_norm(data_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    payload = json.loads((data_dir / "static_norm.json").read_text(encoding="utf-8"))
    return (
        [str(item) for item in payload["feature_order"]],
        np.asarray(payload["train_mean"], dtype=np.float32),
        np.asarray(payload["train_std"], dtype=np.float32),
    )


def restore_age_recruit_days(
    static_matrix: np.ndarray,
    feature_order: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    age_idx = feature_order.index("age_recruit")
    missing_idx = feature_order.index("age_recruit_missing")
    age_years = static_matrix[:, age_idx] * std[age_idx] + mean[age_idx]
    valid = static_matrix[:, missing_idx] < 0.5
    return age_years.astype(np.float64) * 365.25, valid


def load_token_codes(data_dir: Path) -> dict[int, str]:
    vocab_csv = data_dir / "semantic_token_alignment.csv"
    if not vocab_csv.exists():
        manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
        vocab_csv = Path(str(manifest["vocab_csv"]))
    token_codes: dict[int, str] = {}
    with vocab_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type", "").strip() == "diagnosis" and row.get("code_norm", "").strip():
                token_codes[int(row["token_id"])] = row["code_norm"].strip().upper()
    return token_codes


def load_event_type_lookup(data_dir: Path) -> np.ndarray:
    rows = []
    with (data_dir / "semantic_token_alignment.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["token_id"]), TYPE_TO_INDEX.get((row.get("event_type") or "").strip(), -1)))
    lookup = np.full(max(token_id for token_id, _ in rows) + 1, -1, dtype=np.int16)
    for token_id, event_type in rows:
        lookup[token_id] = event_type
    return lookup


def build_token_group_lookup(token_groups: Sequence[Sequence[int]], vocab_size: int) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(vocab_size)]
    for disease_idx, tokens in enumerate(token_groups):
        for token in tokens:
            if 0 <= int(token) < vocab_size:
                out[int(token)].append(disease_idx)
    return out


def build_landmark_index(data: np.ndarray, p2i: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_ids = []
    row_ids = []
    ages = []
    for patient_idx, (start, length) in enumerate(p2i):
        start = int(start)
        length = int(length)
        rows = data[start : start + length]
        if length < 2:
            continue
        lower = start if block_size <= 0 or length <= block_size + 1 else start + length - block_size - 1
        last_age = float(rows[-1, 1])
        for global_pos in range(lower, start + length):
            current_age = float(data[global_pos, 1])
            if current_age < last_age:
                patient_ids.append(patient_idx)
                row_ids.append(global_pos)
                ages.append(current_age)
    return (
        np.asarray(patient_ids, dtype=np.int32),
        np.asarray(row_ids, dtype=np.int64),
        np.asarray(ages, dtype=np.float64),
    )


def sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False).astype(np.int64))


def feature_names(static_names: Sequence[str], disease_ids: Sequence[str]) -> list[str]:
    names = list(static_names)
    names.extend(["current_age_years", "years_since_recruit", "events_so_far"])
    names.extend([f"{name}_count" for name in TYPE_NAMES])
    names.extend([f"{name}_recency_years" for name in TYPE_NAMES])
    names.extend([f"{disease_id}_count" for disease_id in disease_ids])
    names.extend([f"{disease_id}_recency_years" for disease_id in disease_ids])
    names.extend([f"{disease_id}_ever" for disease_id in disease_ids])
    return names


def build_features(
    data: np.ndarray,
    p2i: np.ndarray,
    static_matrix: np.ndarray,
    selected_landmark_rows: np.ndarray,
    event_type_lookup: np.ndarray,
    token_group_lookup: Sequence[Sequence[int]],
    recruit_days: np.ndarray,
    static_names: Sequence[str],
    disease_ids: Sequence[str],
) -> np.ndarray:
    row_to_out = {int(row_id): idx for idx, row_id in enumerate(selected_landmark_rows.tolist())}
    x = np.zeros((len(selected_landmark_rows), len(feature_names(static_names, disease_ids))), dtype=np.float32)
    static_dim = len(static_names)
    num_diseases = len(disease_ids)
    default_recency = 100.0
    for patient_idx, (start, length) in enumerate(p2i):
        start = int(start)
        length = int(length)
        type_counts = np.zeros(len(TYPE_NAMES), dtype=np.float32)
        type_last = np.full(len(TYPE_NAMES), np.nan, dtype=np.float64)
        disease_counts = np.zeros(num_diseases, dtype=np.float32)
        disease_last = np.full(num_diseases, np.nan, dtype=np.float64)
        for offset in range(length):
            global_pos = start + offset
            token_id = int(data[global_pos, 2]) + 1
            current_age = float(data[global_pos, 1])
            if 0 <= token_id < len(event_type_lookup):
                event_type = int(event_type_lookup[token_id])
                if event_type >= 0:
                    type_counts[event_type] += 1.0
                    type_last[event_type] = current_age
            if 0 <= token_id < len(token_group_lookup):
                for disease_idx in token_group_lookup[token_id]:
                    disease_counts[disease_idx] += 1.0
                    disease_last[disease_idx] = current_age
            out_idx = row_to_out.get(global_pos)
            if out_idx is None:
                continue
            cursor = 0
            x[out_idx, cursor : cursor + static_dim] = static_matrix[patient_idx]
            cursor += static_dim
            x[out_idx, cursor] = current_age / 365.25
            cursor += 1
            x[out_idx, cursor] = (current_age - float(recruit_days[patient_idx])) / 365.25
            cursor += 1
            x[out_idx, cursor] = float(offset + 1)
            cursor += 1
            x[out_idx, cursor : cursor + len(TYPE_NAMES)] = type_counts
            cursor += len(TYPE_NAMES)
            type_recency = np.where(np.isnan(type_last), default_recency, (current_age - type_last) / 365.25)
            x[out_idx, cursor : cursor + len(TYPE_NAMES)] = type_recency.astype(np.float32)
            cursor += len(TYPE_NAMES)
            x[out_idx, cursor : cursor + num_diseases] = disease_counts
            cursor += num_diseases
            disease_recency = np.where(np.isnan(disease_last), default_recency, (current_age - disease_last) / 365.25)
            x[out_idx, cursor : cursor + num_diseases] = disease_recency.astype(np.float32)
            cursor += num_diseases
            x[out_idx, cursor : cursor + num_diseases] = (disease_counts > 0).astype(np.float32)
    return x


def build_labels(
    landmark_patient_ids: np.ndarray,
    landmark_ages: np.ndarray,
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    disease_idx: int,
    horizon_years: float,
) -> np.ndarray:
    import bisect

    labels = np.zeros(len(landmark_ages), dtype=np.int8)
    horizon_days = float(horizon_years) * 365.25
    for idx, (patient_idx, current_age) in enumerate(zip(landmark_patient_ids, landmark_ages)):
        disease_ages = patient_disease_ages[int(patient_idx)][disease_idx]
        next_idx = bisect.bisect_right(disease_ages, float(current_age))
        if next_idx < len(disease_ages):
            delta = float(disease_ages[next_idx]) - float(current_age)
            if 0.0 < delta <= horizon_days:
                labels[idx] = 1
    return labels


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        avg_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = avg_rank
        start = stop
    labels_bool = labels.astype(bool)
    n_pos = int(labels_bool.sum())
    n_neg = int(len(labels_bool) - n_pos)
    return float((ranks[labels_bool].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def top_decile_stats(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    if len(scores) == 0 or labels.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    top_n = max(1, int(math.ceil(len(scores) * 0.10)))
    top_labels = labels[order[:top_n]]
    top_rate = float(top_labels.mean())
    baseline = float(labels.mean())
    return float(top_labels.sum() / labels.sum()), top_rate, top_rate / baseline if baseline else float("nan")


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(bool)
    hit_ranks = np.flatnonzero(ranked) + 1
    return float((np.arange(1, positives + 1, dtype=np.float64) / hit_ranks).mean())


def standardize_from_train(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_eval - mean) / std


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def ridge_linear_scores(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    x_train_i = np.column_stack([np.ones(len(x_train)), x_train])
    x_eval_i = np.column_stack([np.ones(len(x_eval)), x_eval])
    y = y_train.astype(np.float64)
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(len(y) - y.sum()))
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    weighted_x = x_train_i * sample_weight[:, None]
    penalty = np.eye(x_train_i.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(weighted_x.T @ x_train_i + 1.0 * penalty, weighted_x.T @ y)
    return sigmoid(x_eval_i @ coef)


def fit_predict_scores(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, str]:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        prior = positives / len(y_train) if len(y_train) else 0.0
        return np.full(len(x_eval), prior, dtype=np.float64), "constant_prior"
    x_train_scaled, x_eval_scaled = standardize_from_train(x_train.astype(np.float64), x_eval.astype(np.float64))
    if LogisticRegression is not None:
        model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        model.fit(x_train_scaled, y_train)
        return model.predict_proba(x_eval_scaled)[:, 1].astype(np.float64), "sklearn_logistic_regression"
    return ridge_linear_scores(x_train_scaled, y_train, x_eval_scaled), "numpy_weighted_ridge_linear"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen split Static/Dynamic/GBDT baselines at locked-test moments.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "ukb_semantic_multitype_explicit_split")
    parser.add_argument("--diseases-yaml", type=Path, default=ROOT / "docs" / "selected_diseases.yaml")
    parser.add_argument(
        "--reference-raw",
        type=Path,
        default=ROOT / "results" / "locked_test_horizon_risk" / "medtrajectory_test" / "raw_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "locked_test_horizon_risk" / "frozen_ml_baselines")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument("--max-train-landmarks", type=int, default=120_000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser


def read_reference_rows(
    path: Path,
    disease_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[tuple[float, int], np.ndarray], list[dict]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    moment_by_patient: dict[int, float] = {}
    for row in rows:
        patient = int(row["patient_index"])
        age_days = float(row["prediction_age_years"]) * 365.25
        previous = moment_by_patient.get(patient)
        if previous is not None and abs(previous - age_days) > 1.0:
            # Keep the first deterministic locked-test moment if a pathological duplicate appears.
            continue
        moment_by_patient.setdefault(patient, age_days)
    patients = np.asarray(sorted(moment_by_patient), dtype=np.int32)
    ages = np.asarray([moment_by_patient[int(patient)] for patient in patients], dtype=np.float64)
    patient_to_pos = {int(patient): idx for idx, patient in enumerate(patients)}
    labels: dict[tuple[float, int], np.ndarray] = {}
    disease_to_idx = {disease: idx for idx, disease in enumerate(disease_ids)}
    for row in rows:
        patient = int(row["patient_index"])
        if patient not in patient_to_pos:
            continue
        disease_idx = disease_to_idx.get(row["disease_id"])
        if disease_idx is None:
            continue
        key = (float(row["horizon_years"]), disease_idx)
        labels.setdefault(key, np.zeros(len(patients), dtype=np.int8))[patient_to_pos[patient]] = int(float(row["label"]))
    return patients, ages, labels, rows


def build_features_at_moments(
    data: np.ndarray,
    p2i: np.ndarray,
    static_matrix: np.ndarray,
    patient_ids: np.ndarray,
    current_ages: np.ndarray,
    event_type_lookup: np.ndarray,
    token_group_lookup: Sequence[Sequence[int]],
    recruit_days: np.ndarray,
    static_names: Sequence[str],
    disease_ids: Sequence[str],
) -> np.ndarray:
    names = feature_names(static_names, disease_ids)
    x = np.zeros((len(patient_ids), len(names)), dtype=np.float32)
    static_dim = len(static_names)
    type_names = ["diagnosis", "procedure", "cancer", "death"]
    default_recency = 100.0
    num_diseases = len(disease_ids)
    for out_idx, (patient_idx, current_age) in enumerate(zip(patient_ids, current_ages)):
        start, length = p2i[int(patient_idx)]
        rows = data[int(start) : int(start) + int(length)]
        visible = rows[rows[:, 1].astype(np.float64) <= float(current_age)]
        type_counts = np.zeros(len(type_names), dtype=np.float32)
        type_last = np.full(len(type_names), np.nan, dtype=np.float64)
        disease_counts = np.zeros(num_diseases, dtype=np.float32)
        disease_last = np.full(num_diseases, np.nan, dtype=np.float64)
        for _, age_days, raw_token in visible:
            token_id = int(raw_token) + 1
            age_value = float(age_days)
            if 0 <= token_id < len(event_type_lookup):
                event_type = int(event_type_lookup[token_id])
                if event_type >= 0:
                    type_counts[event_type] += 1.0
                    type_last[event_type] = age_value
            if 0 <= token_id < len(token_group_lookup):
                for disease_idx in token_group_lookup[token_id]:
                    disease_counts[disease_idx] += 1.0
                    disease_last[disease_idx] = age_value
        cursor = 0
        x[out_idx, cursor : cursor + static_dim] = static_matrix[int(patient_idx)]
        cursor += static_dim
        x[out_idx, cursor] = float(current_age) / 365.25
        cursor += 1
        x[out_idx, cursor] = (float(current_age) - float(recruit_days[int(patient_idx)])) / 365.25
        cursor += 1
        x[out_idx, cursor] = float(len(visible))
        cursor += 1
        x[out_idx, cursor : cursor + len(type_names)] = type_counts
        cursor += len(type_names)
        type_recency = np.where(np.isnan(type_last), default_recency, (float(current_age) - type_last) / 365.25)
        x[out_idx, cursor : cursor + len(type_names)] = type_recency.astype(np.float32)
        cursor += len(type_names)
        x[out_idx, cursor : cursor + num_diseases] = disease_counts
        cursor += num_diseases
        disease_recency = np.where(np.isnan(disease_last), default_recency, (float(current_age) - disease_last) / 365.25)
        x[out_idx, cursor : cursor + num_diseases] = disease_recency.astype(np.float32)
        cursor += num_diseases
        x[out_idx, cursor : cursor + num_diseases] = (disease_counts > 0).astype(np.float32)
    return x


def static_age_features(static_matrix: np.ndarray, patient_ids: np.ndarray, current_ages: np.ndarray) -> np.ndarray:
    age_years = (current_ages / 365.25).astype(np.float32).reshape(-1, 1)
    return np.hstack([static_matrix[patient_ids.astype(np.int64)], age_years])


def weighted_sample_weights(labels: np.ndarray) -> np.ndarray:
    positives = max(1.0, float(labels.sum()))
    negatives = max(1.0, float(labels.size - labels.sum()))
    return np.where(labels > 0, labels.size / (2.0 * positives), labels.size / (2.0 * negatives))


def fit_predict_gbdt(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        prior = positives / len(y_train) if len(y_train) else 0.0
        return np.full(len(x_eval), prior, dtype=np.float64), "constant_prior"
    if HistGradientBoostingClassifier is None:
        return fit_predict_scores(x_train, y_train, x_eval)
    x_train_scaled, x_eval_scaled = standardize_from_train(x_train.astype(np.float64), x_eval.astype(np.float64))
    model = HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed,
    )
    model.fit(x_train_scaled, y_train, sample_weight=weighted_sample_weights(y_train))
    return model.predict_proba(x_eval_scaled)[:, 1].astype(np.float64), "sklearn_hist_gradient_boosting"


def ece_score(scores: np.ndarray, labels: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (scores >= edges[idx]) & (scores <= edges[idx + 1])
        else:
            mask = (scores >= edges[idx]) & (scores < edges[idx + 1])
        if bool(mask.any()):
            ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return float(ece)


def summarize_scores(scores: np.ndarray, labels: np.ndarray, bins: int) -> dict:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    auc = binary_auc(scores, labels) if positives > 0 and negatives > 0 else float("nan")
    cap, top_rate, lift = top_decile_stats(scores, labels)
    return {
        "prediction_moments": int(labels.size),
        "positives": positives,
        "negatives": negatives,
        "auc": auc,
        "average_precision": average_precision(scores, labels),
        "brier": float(np.mean((scores - labels) ** 2)),
        "ece": ece_score(scores, labels, bins),
        "top_decile_capture": cap,
        "top_decile_event_rate": top_rate,
        "top_decile_lift": lift,
        "event_rate": float(labels.mean()) if labels.size else float("nan"),
    }


def aggregate(rows: Sequence[dict]) -> list[dict]:
    out = []
    for horizon in [5.0, 10.0]:
        rs = [row for row in rows if float(row["horizon_years"]) == horizon and math.isfinite(float(row["auc"]))]
        out.append(aggregate_row(rs, "horizon", horizon))
    out.append(aggregate_row([row for row in rows if math.isfinite(float(row["auc"]))], "overall", float("nan")))
    return out


def aggregate_row(rows: Sequence[dict], scope: str, horizon: float) -> dict:
    def avg(key: str) -> float:
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else float("nan")

    return {
        "aggregate": scope,
        "horizon_years": horizon,
        "diseases_with_auc": len(rows),
        "auc_mean": avg("auc"),
        "average_precision_mean": avg("average_precision"),
        "brier_mean": avg("brier"),
        "ece_mean": avg("ece"),
        "top_decile_capture_mean": avg("top_decile_capture"),
        "event_rate_mean": avg("event_rate"),
    }


def write_status(path: Path, aggregate_rows: Sequence[dict]) -> None:
    lines = [
        "# Frozen ML Baselines",
        "",
        "Evaluation uses the exact locked-test MedTrajectory prediction moments and labels, so patient split, disease set, horizon, and prediction age are aligned with the frozen deep-model comparison.",
        "",
        "| Model | 5y AUC | 10y AUC | Overall AUC | Brier | ECE | Capture |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    models = sorted({row["model"] for row in aggregate_rows})
    for model in models:
        rs = [row for row in aggregate_rows if row["model"] == model]
        by_scope = {row["aggregate"] if row["aggregate"] == "overall" else f"{float(row['horizon_years']):g}": row for row in rs}
        r5 = by_scope.get("5", {})
        r10 = by_scope.get("10", {})
        overall = by_scope.get("overall", {})
        lines.append(
            "| {model} | {auc5:.4f} | {auc10:.4f} | {auc:.4f} | {brier:.4f} | {ece:.4f} | {cap:.4f} |".format(
                model=model,
                auc5=float(r5.get("auc_mean", float("nan"))),
                auc10=float(r10.get("auc_mean", float("nan"))),
                auc=float(overall.get("auc_mean", float("nan"))),
                brier=float(overall.get("brier_mean", float("nan"))),
                ece=float(overall.get("ece_mean", float("nan"))),
                cap=float(overall.get("top_decile_capture_mean", float("nan"))),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)
    diseases = parse_selected_diseases(args.diseases_yaml)
    disease_ids = [disease.disease_id for disease in diseases]
    token_codes = load_token_codes(args.data_dir)
    token_groups = [token_ids_for_disease(disease, token_codes) for disease in diseases]
    feature_order, mean, std = load_static_norm(args.data_dir)

    train_data, train_p2i, train_static = load_split(args.data_dir, args.train_split)
    eval_data, eval_p2i, eval_static = load_split(args.data_dir, args.eval_split)
    train_recruit_days, _ = restore_age_recruit_days(train_static, feature_order, mean, std)
    eval_recruit_days, _ = restore_age_recruit_days(eval_static, feature_order, mean, std)
    event_type_lookup = load_event_type_lookup(args.data_dir)
    vocab_size = max(int(max(train_data[:, 2].max(), eval_data[:, 2].max())) + 2, len(event_type_lookup))
    token_group_lookup = build_token_group_lookup(token_groups, vocab_size)

    train_patient_ids, train_rows, train_ages = build_landmark_index(train_data, train_p2i, args.block_size)
    keep = sample_indices(len(train_rows), args.max_train_landmarks, args.seed)
    train_patient_sample = train_patient_ids[keep]
    train_rows_sample = train_rows[keep]
    train_age_sample = train_ages[keep]
    x_train_static = static_age_features(train_static, train_patient_sample, train_age_sample)
    x_train_dynamic = build_features(
        train_data,
        train_p2i,
        train_static,
        train_rows_sample,
        event_type_lookup,
        token_group_lookup,
        train_recruit_days,
        feature_order,
        disease_ids,
    )
    eval_patient_ids, eval_ages, eval_labels, reference_rows = read_reference_rows(args.reference_raw, disease_ids)
    x_eval_static = static_age_features(eval_static, eval_patient_ids, eval_ages)
    x_eval_dynamic = build_features_at_moments(
        eval_data,
        eval_p2i,
        eval_static,
        eval_patient_ids,
        eval_ages,
        event_type_lookup,
        token_group_lookup,
        eval_recruit_days,
        feature_order,
        disease_ids,
    )
    train_patient_disease_ages, _ = build_patient_disease_ages(train_data, train_p2i, token_groups, vocab_size)

    all_rows = []
    raw_rows = []
    for horizon in horizons:
        for disease_idx, disease in enumerate(diseases):
            y_train = build_labels(train_patient_sample, train_age_sample, train_patient_disease_ages, disease_idx, horizon)
            y_eval = eval_labels[(float(horizon), disease_idx)]
            models = [
                ("Static+age logistic", *fit_predict_scores(x_train_static, y_train, x_eval_static)),
                ("Dynamic logistic", *fit_predict_scores(x_train_dynamic, y_train, x_eval_dynamic)),
                ("Dynamic GBDT", *fit_predict_gbdt(x_train_dynamic, y_train, x_eval_dynamic, args.seed + disease_idx + int(horizon * 10))),
            ]
            for model_name, scores, fitted_model in models:
                row = {
                    "model": model_name,
                    "horizon_years": float(horizon),
                    "disease_id": disease.disease_id,
                    "name": disease.name,
                    "name_cn": disease.name_cn,
                    "fitted_model": fitted_model,
                    "train_landmarks": int(len(y_train)),
                    "train_positives": int(y_train.sum()),
                    **summarize_scores(scores, y_eval, args.calibration_bins),
                }
                all_rows.append(row)
                for patient_idx, pred_age, label, score in zip(eval_patient_ids, eval_ages, y_eval, scores):
                    raw_rows.append(
                        {
                            "model": model_name,
                            "split": args.eval_split,
                            "patient_index": int(patient_idx),
                            "prediction_age_years": float(pred_age) / 365.25,
                            "horizon_years": float(horizon),
                            "disease_id": disease.disease_id,
                            "name": disease.name,
                            "name_cn": disease.name_cn,
                            "label": int(label),
                            "score": float(score),
                        }
                    )

    aggregate_rows = []
    for model in sorted({row["model"] for row in all_rows}):
        for row in aggregate([row for row in all_rows if row["model"] == model]):
            row["model"] = model
            aggregate_rows.append(row)
    write_csv(args.output_dir / "frozen_ml_metrics.csv", all_rows)
    write_csv(args.output_dir / "frozen_ml_aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir / "frozen_ml_raw_predictions.csv", raw_rows)
    write_status(args.output_dir / "STATUS.md", aggregate_rows)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "data_dir": str(args.data_dir),
                "reference_raw": str(args.reference_raw),
                "train_landmarks_total": int(len(train_rows)),
                "train_landmarks_used": int(len(train_rows_sample)),
                "eval_patients": int(len(eval_patient_ids)),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
