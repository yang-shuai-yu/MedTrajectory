from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

from semantic_delphi_ukb.selected_disease_demo import (
    ModelSpec,
    binary_auc,
    load_token_codes,
    parse_selected_diseases,
    top_decile_stats,
    token_ids_for_disease,
    write_csv,
)
from semantic_delphi_ukb.tte_targets import build_patient_disease_ages
from utils import get_p2i


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_DIR / "data" / "ukb_semantic_multitype_explicit_split"
DEFAULT_OUTPUT_DIR = REPO_DIR / "tests" / "output" / "static_ml_baseline"
DEFAULT_TTE_METRICS = REPO_DIR / "tests" / "output" / "selected_disease_tte_head_direct" / "tte_head_metrics.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Static-feature ML baseline for selected disease horizon risk.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "selected_diseases.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument("--max-train-patients", type=int, default=0)
    parser.add_argument("--max-eval-patients", type=int, default=0)
    parser.add_argument("--tte-metrics", type=Path, default=DEFAULT_TTE_METRICS)
    return parser


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


def model_spec_for_data_dir(data_dir: Path) -> ModelSpec:
    vocab_csv = data_dir / "semantic_token_alignment.csv"
    return ModelSpec(
        model_id="static_ml",
        display_name="Static Logistic Regression",
        model_type="multitype",
        ckpt_path=Path(""),
        data_dir=data_dir,
        token_vocab_csv=vocab_csv if vocab_csv.exists() else None,
    )


def load_static_norm(data_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    path = data_dir / "static_norm.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        [str(item) for item in payload["feature_order"]],
        np.asarray(payload["train_mean"], dtype=np.float32),
        np.asarray(payload["train_std"], dtype=np.float32),
    )


def restore_age_recruit_days(static_matrix: np.ndarray, feature_order: Sequence[str], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    age_idx = feature_order.index("age_recruit")
    missing_idx = feature_order.index("age_recruit_missing")
    age_years = static_matrix[:, age_idx] * std[age_idx] + mean[age_idx]
    valid = static_matrix[:, missing_idx] < 0.5
    return age_years.astype(np.float64) * 365.25, valid


def build_horizon_labels(
    data: np.ndarray,
    p2i: np.ndarray,
    static_matrix: np.ndarray,
    token_groups: Sequence[Sequence[int]],
    horizons: Sequence[float],
    feature_order: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    max_patients: int,
) -> tuple[dict[tuple[float, int], tuple[np.ndarray, np.ndarray]], np.ndarray]:
    vocab_size = max(int(data[:, 2].max()) + 2, max(max(tokens, default=0) for tokens in token_groups) + 1)
    patient_disease_ages, patient_last_ages = build_patient_disease_ages(data, p2i, token_groups, vocab_size)
    baseline_days, valid_baseline = restore_age_recruit_days(static_matrix, feature_order, mean, std)
    valid_baseline &= baseline_days < patient_last_ages
    n = len(p2i) if max_patients <= 0 else min(len(p2i), max_patients)
    valid_baseline[n:] = False

    labels: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    for horizon in horizons:
        horizon_days = float(horizon) * 365.25
        for disease_idx in range(len(token_groups)):
            y = np.zeros(len(p2i), dtype=np.int8)
            for patient_idx in np.flatnonzero(valid_baseline):
                current_age = float(baseline_days[patient_idx])
                disease_ages = patient_disease_ages[int(patient_idx)][disease_idx]
                next_idx = bisect.bisect_right(disease_ages, current_age)
                if next_idx < len(disease_ages) and 0.0 < float(disease_ages[next_idx]) - current_age <= horizon_days:
                    y[patient_idx] = 1
            labels[(float(horizon), disease_idx)] = (y, valid_baseline.copy())
    return labels, valid_baseline


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


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(bool)
    hit_ranks = np.flatnonzero(ranked) + 1
    precision_at_hits = np.arange(1, positives + 1, dtype=np.float64) / hit_ranks
    return float(precision_at_hits.mean())


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


def summarize_scores(scores: np.ndarray, labels: np.ndarray) -> dict:
    positives = int(labels.sum())
    total = int(labels.size)
    negatives = total - positives
    auc = binary_auc(scores, labels) if positives > 0 and negatives > 0 else float("nan")
    ap = average_precision(scores, labels)
    top_capture, top_rate, lift = top_decile_stats(scores, labels)
    return {
        "prediction_moments": total,
        "positives": positives,
        "negatives": negatives,
        "baseline_event_rate": positives / total if total else float("nan"),
        "auc": auc,
        "average_precision": ap,
        "top_decile_capture": top_capture,
        "top_decile_event_rate": top_rate,
        "top_decile_lift": lift,
        "mean_score": float(scores.mean()) if scores.size else float("nan"),
    }


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_comparison_rows(static_rows: Sequence[dict], tte_metrics_path: Path) -> list[dict]:
    tte_rows = read_csv_rows(tte_metrics_path)
    tte_by_key = {(row.get("disease_id"), float(row.get("horizon_years", "nan"))): row for row in tte_rows}
    out = []
    for row in static_rows:
        key = (row["disease_id"], float(row["horizon_years"]))
        tte = tte_by_key.get(key)
        static_auc = float(row["auc"])
        tte_auc = float(tte["auc"]) if tte and tte.get("auc") not in ("", "nan") else float("nan")
        static_capture = float(row["top_decile_capture"])
        tte_capture = float(tte["top_decile_capture"]) if tte and tte.get("top_decile_capture") not in ("", "nan") else float("nan")
        out.append(
            {
                "horizon_years": row["horizon_years"],
                "disease_id": row["disease_id"],
                "name": row["name"],
                "static_auc": static_auc,
                "tte_head_auc": tte_auc,
                "delta_auc_tte_minus_static": tte_auc - static_auc if math.isfinite(static_auc) and math.isfinite(tte_auc) else float("nan"),
                "static_top_decile_capture": static_capture,
                "tte_head_top_decile_capture": tte_capture,
                "delta_capture_tte_minus_static": tte_capture - static_capture if math.isfinite(static_capture) and math.isfinite(tte_capture) else float("nan"),
                "static_positives": row["positives"],
                "tte_prediction_moments": tte.get("prediction_moments", "") if tte else "",
                "comparison_note": "static is one recruitment-time prediction per patient; TTE is evaluated over dynamic trajectory moments",
            }
        )
    return out


def write_summary(path: Path, rows: Sequence[dict], comparison_rows: Sequence[dict], feature_order: Sequence[str]) -> None:
    lines = [
        "# Static ML Baseline",
        "",
        "Model: sklearn `LogisticRegression(class_weight=\"balanced\")` when available; otherwise a weighted ridge linear-probability fallback. One model is trained separately for each selected disease and horizon.",
        "",
        "Static features: " + ", ".join(f"`{name}`" for name in feature_order) + ".",
        "",
        "Comparison caveat: static ML makes one recruitment-time prediction per patient; TTE direct head is evaluated over multiple dynamic trajectory moments. Very low-positive rows should be treated as unstable.",
        "",
        "| Horizon | Disease | Patients | Positives | AUC | AP | Top-decile capture |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {horizon:g} | {name} | {n} | {pos} | {auc:.4f} | {ap:.4f} | {cap:.4f} |".format(
                horizon=float(row["horizon_years"]),
                name=row["name"],
                n=int(row["prediction_moments"]),
                pos=int(row["positives"]),
                auc=float(row["auc"]),
                ap=float(row["average_precision"]),
                cap=float(row["top_decile_capture"]),
            )
        )
    if comparison_rows:
        lines.extend(
            [
                "",
                "## TTE Direct Head Comparison",
                "",
                "| Horizon | Disease | Static AUC | TTE head AUC | Delta |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in comparison_rows:
            lines.append(
                "| {horizon:g} | {name} | {static_auc:.4f} | {tte_auc:.4f} | {delta:.4f} |".format(
                    horizon=float(row["horizon_years"]),
                    name=row["name"],
                    static_auc=float(row["static_auc"]),
                    tte_auc=float(row["tte_head_auc"]),
                    delta=float(row["delta_auc_tte_minus_static"]),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    horizons = parse_horizons(args.horizons)
    diseases = parse_selected_diseases(args.diseases_yaml)
    token_codes, _ = load_token_codes(model_spec_for_data_dir(args.data_dir))
    token_groups = [token_ids_for_disease(disease, token_codes) for disease in diseases]
    feature_order, mean, std = load_static_norm(args.data_dir)

    train_data, train_p2i, train_static = load_split(args.data_dir, args.train_split)
    eval_data, eval_p2i, eval_static = load_split(args.data_dir, args.eval_split)
    train_labels, train_valid = build_horizon_labels(
        train_data,
        train_p2i,
        train_static,
        token_groups,
        horizons,
        feature_order,
        mean,
        std,
        args.max_train_patients,
    )
    eval_labels, eval_valid = build_horizon_labels(
        eval_data,
        eval_p2i,
        eval_static,
        token_groups,
        horizons,
        feature_order,
        mean,
        std,
        args.max_eval_patients,
    )

    rows = []
    for horizon in horizons:
        for disease_idx, disease in enumerate(diseases):
            y_train_all, train_mask = train_labels[(float(horizon), disease_idx)]
            y_eval_all, eval_mask = eval_labels[(float(horizon), disease_idx)]
            x_train = train_static[train_mask]
            y_train = y_train_all[train_mask]
            x_eval = eval_static[eval_mask]
            y_eval = y_eval_all[eval_mask]
            scores, fitted_model = fit_predict_scores(x_train, y_train, x_eval)
            row = {
                "horizon_years": float(horizon),
                "disease_id": disease.disease_id,
                "name": disease.name,
                "name_cn": disease.name_cn,
                "category": disease.category,
                "icd10": ";".join(disease.ranges),
                "model_id": "static_ml",
                "model": "Static ML baseline",
                "fitted_model": fitted_model,
                "matched_tokens": len(token_groups[disease_idx]),
                "train_patients": int(train_mask.sum()),
                "train_positives": int(y_train.sum()),
                **summarize_scores(scores, y_eval),
            }
            rows.append(row)

    comparison_rows = build_comparison_rows(rows, args.tte_metrics)
    write_csv(args.output_dir / "static_ml_metrics.csv", rows)
    write_csv(args.output_dir / "static_ml_comparison.csv", comparison_rows)
    write_summary(args.output_dir / "static_ml_summary.md", rows, comparison_rows, feature_order)
    (args.output_dir / "static_ml_details.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "train_split": args.train_split,
                "eval_split": args.eval_split,
                "horizons": horizons,
                "valid_train_patients": int(train_valid.sum()),
                "valid_eval_patients": int(eval_valid.sum()),
                "tte_metrics": str(args.tte_metrics),
                "metrics": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
