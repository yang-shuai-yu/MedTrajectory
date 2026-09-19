from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from semantic_delphi_ukb.selected_disease_demo import (
    ModelSpec,
    binary_auc,
    load_token_codes,
    parse_selected_diseases,
    token_ids_for_disease,
    top_decile_stats,
    write_csv,
)
from semantic_delphi_ukb.static_ml_baseline import (
    average_precision,
    fit_predict_scores,
    load_split,
    load_static_norm,
    parse_horizons,
    restore_age_recruit_days,
)
from semantic_delphi_ukb.tte_targets import build_patient_disease_ages


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_DIR / "data" / "ukb_semantic_multitype_explicit_split_static_baseline"
DEFAULT_OUTPUT_DIR = REPO_DIR / "tests" / "output" / "dynamic_ml_landmark_baseline"
DEFAULT_STATIC_METRICS = REPO_DIR / "tests" / "output" / "static_ml_baseline" / "static_ml_metrics.csv"
DEFAULT_TTE_METRICS = REPO_DIR / "tests" / "output" / "selected_disease_tte_head_direct" / "tte_head_metrics.csv"

TYPE_NAMES = ["diagnosis", "procedure", "cancer", "death"]
TYPE_TO_INDEX = {name: idx for idx, name in enumerate(TYPE_NAMES)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traditional ML landmark baseline using engineered trajectory features.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "selected_diseases.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument("--max-train-landmarks", type=int, default=500_000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--static-metrics", type=Path, default=DEFAULT_STATIC_METRICS)
    parser.add_argument("--tte-metrics", type=Path, default=DEFAULT_TTE_METRICS)
    return parser


def model_spec_for_data_dir(data_dir: Path) -> ModelSpec:
    vocab_csv = data_dir / "semantic_token_alignment.csv"
    return ModelSpec(
        model_id="dynamic_ml_landmark",
        display_name="Dynamic ML landmark baseline",
        model_type="multitype",
        ckpt_path=Path(""),
        data_dir=data_dir,
        token_vocab_csv=vocab_csv,
    )


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


def summarize_scores(scores: np.ndarray, labels: np.ndarray) -> dict:
    positives = int(labels.sum())
    total = int(labels.size)
    negatives = total - positives
    auc = binary_auc(scores, labels) if positives > 0 and negatives > 0 else float("nan")
    top_capture, top_rate, lift = top_decile_stats(scores, labels)
    return {
        "prediction_moments": total,
        "positives": positives,
        "negatives": negatives,
        "baseline_event_rate": positives / total if total else float("nan"),
        "auc": auc,
        "average_precision": average_precision(scores, labels),
        "top_decile_capture": top_capture,
        "top_decile_event_rate": top_rate,
        "top_decile_lift": lift,
        "mean_score": float(scores.mean()) if len(scores) else float("nan"),
    }


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_lookup(path: Path, auc_name: str, capture_name: str) -> dict[tuple[str, float], dict]:
    out = {}
    for row in read_rows(path):
        out[(row["disease_id"], float(row["horizon_years"]))] = {
            auc_name: float(row["auc"]),
            capture_name: float(row["top_decile_capture"]),
        }
    return out


def comparison_rows(rows: Sequence[dict], static_path: Path, tte_path: Path) -> list[dict]:
    static = metric_lookup(static_path, "static_auc", "static_top_decile_capture")
    tte = metric_lookup(tte_path, "tte_auc", "tte_top_decile_capture")
    out = []
    for row in rows:
        key = (row["disease_id"], float(row["horizon_years"]))
        merged = {
            "horizon_years": row["horizon_years"],
            "disease_id": row["disease_id"],
            "name": row["name"],
            "dynamic_ml_auc": row["auc"],
            "dynamic_ml_top_decile_capture": row["top_decile_capture"],
            "dynamic_ml_positives": row["positives"],
            "dynamic_ml_prediction_moments": row["prediction_moments"],
        }
        merged.update(static.get(key, {}))
        merged.update(tte.get(key, {}))
        if "static_auc" in merged:
            merged["delta_dynamic_minus_static_auc"] = float(row["auc"]) - float(merged["static_auc"])
        if "tte_auc" in merged:
            merged["delta_tte_minus_dynamic_auc"] = float(merged["tte_auc"]) - float(row["auc"])
        out.append(merged)
    return out


def write_summary(path: Path, rows: Sequence[dict], comparisons: Sequence[dict], feature_count: int) -> None:
    mean_5 = np.mean([float(row["auc"]) for row in rows if float(row["horizon_years"]) == 5.0])
    mean_10 = np.mean([float(row["auc"]) for row in rows if float(row["horizon_years"]) == 10.0])
    lines = [
        "# Dynamic ML Landmark Baseline",
        "",
        "Traditional ML baseline using engineered trajectory-history features at landmark prediction moments.",
        "",
        f"Feature count: `{feature_count}`. Training landmarks are sampled deterministically when capped.",
        "",
        "| Horizon | Disease | Eval moments | Positives | AUC | AP | Top-decile capture |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {h:g} | {name} | {n} | {pos} | {auc:.4f} | {ap:.4f} | {cap:.4f} |".format(
                h=float(row["horizon_years"]),
                name=row["name"],
                n=int(row["prediction_moments"]),
                pos=int(row["positives"]),
                auc=float(row["auc"]),
                ap=float(row["average_precision"]),
                cap=float(row["top_decile_capture"]),
            )
        )
    lines.extend(
        [
            "",
            f"Mean AUC: 5y `{mean_5:.4f}`, 10y `{mean_10:.4f}`.",
            "",
            "## Comparison",
            "",
            "| Horizon | Disease | Static AUC | Dynamic ML AUC | TTE AUC |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {h:g} | {name} | {static:.4f} | {dyn:.4f} | {tte:.4f} |".format(
                h=float(row["horizon_years"]),
                name=row["name"],
                static=float(row.get("static_auc", float("nan"))),
                dyn=float(row["dynamic_ml_auc"]),
                tte=float(row.get("tte_auc", float("nan"))),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)
    diseases = parse_selected_diseases(args.diseases_yaml)
    disease_ids = [disease.disease_id for disease in diseases]
    token_codes, _ = load_token_codes(model_spec_for_data_dir(args.data_dir))
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
    eval_patient_ids, eval_rows, eval_ages = build_landmark_index(eval_data, eval_p2i, args.block_size)
    train_keep = sample_indices(len(train_rows), args.max_train_landmarks, args.seed)
    train_rows_sample = train_rows[train_keep]
    train_patient_sample = train_patient_ids[train_keep]
    train_age_sample = train_ages[train_keep]

    names = feature_names(feature_order, disease_ids)
    x_train = build_features(
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
    x_eval = build_features(
        eval_data,
        eval_p2i,
        eval_static,
        eval_rows,
        event_type_lookup,
        token_group_lookup,
        eval_recruit_days,
        feature_order,
        disease_ids,
    )

    train_patient_disease_ages, _ = build_patient_disease_ages(train_data, train_p2i, token_groups, vocab_size)
    eval_patient_disease_ages, _ = build_patient_disease_ages(eval_data, eval_p2i, token_groups, vocab_size)

    rows = []
    for horizon in horizons:
        for disease_idx, disease in enumerate(diseases):
            y_train = build_labels(train_patient_sample, train_age_sample, train_patient_disease_ages, disease_idx, horizon)
            y_eval = build_labels(eval_patient_ids, eval_ages, eval_patient_disease_ages, disease_idx, horizon)
            scores, fitted_model = fit_predict_scores(x_train, y_train, x_eval)
            rows.append(
                {
                    "horizon_years": float(horizon),
                    "disease_id": disease.disease_id,
                    "name": disease.name,
                    "name_cn": disease.name_cn,
                    "category": disease.category,
                    "icd10": ";".join(disease.ranges),
                    "model_id": "dynamic_ml_landmark",
                    "model": "Dynamic ML landmark baseline",
                    "fitted_model": fitted_model,
                    "matched_tokens": len(token_groups[disease_idx]),
                    "train_landmarks": int(len(y_train)),
                    "train_positives": int(y_train.sum()),
                    **summarize_scores(scores, y_eval),
                }
            )

    comparisons = comparison_rows(rows, args.static_metrics, args.tte_metrics)
    write_csv(args.output_dir / "dynamic_ml_landmark_metrics.csv", rows)
    write_csv(args.output_dir / "dynamic_ml_landmark_comparison.csv", comparisons)
    write_summary(args.output_dir / "dynamic_ml_landmark_summary.md", rows, comparisons, len(names))
    (args.output_dir / "dynamic_ml_landmark_details.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "train_split": args.train_split,
                "eval_split": args.eval_split,
                "horizons": horizons,
                "max_train_landmarks": args.max_train_landmarks,
                "train_landmarks_total": int(len(train_rows)),
                "train_landmarks_used": int(len(train_rows_sample)),
                "eval_landmarks": int(len(eval_rows)),
                "feature_names": names,
                "metrics": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
