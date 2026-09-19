from __future__ import annotations

import csv
import json
import math
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]


RUNS = [
    ("main_reference", "current_main", "results/locked_test_horizon_risk/medtrajectory_test"),
    ("bert_reference", "baseline", "results/locked_test_horizon_risk/bert_test"),
    ("mamba_reference", "baseline", "results/locked_test_horizon_risk/mamba_test"),
    ("survival_reference_smoke", "survival", "results/locked_test_horizon_risk/survival_test"),
    ("tte_trunk_aux02", "horizon_ablation", "results/locked_test_horizon_risk/high_priority_tte_trunk_aux02"),
    ("modern_trunk_full3000", "horizon_ablation", "results/locked_test_horizon_risk/high_priority_modern_trunk_full3000"),
    ("no_rope_trunk_full3000", "horizon_ablation", "results/locked_test_horizon_risk/high_priority_no_rope_trunk_full3000"),
    ("block48_trunk_full3000", "horizon_ablation", "results/locked_test_horizon_risk/high_priority_block48_trunk_full3000"),
    ("age_only_no_rope", "time_encoding_ablation", "results/locked_test_horizon_risk/high_priority_age_only_no_rope"),
    ("rope_only_no_age", "time_encoding_ablation", "results/locked_test_horizon_risk/high_priority_rope_only_no_age"),
    ("no_age_no_rope", "time_encoding_ablation", "results/locked_test_horizon_risk/high_priority_no_age_no_rope"),
    ("tte_trunk_no_next_aux", "auxiliary_ablation", "results/locked_test_horizon_risk/high_priority_tte_trunk_no_next_aux"),
    ("tte_trunk_next_aux05", "auxiliary_ablation", "results/locked_test_horizon_risk/high_priority_tte_trunk_next_aux05"),
    (
        "surv_bin1y_pos10_aux02_full3000",
        "survival_candidate",
        "results/locked_test_horizon_risk/high_priority_survival_surv_bin1y_pos10_aux02_full3000",
    ),
    (
        "surv_bin2y_pos10_aux02_full3000",
        "survival_candidate",
        "results/locked_test_horizon_risk/high_priority_survival_surv_bin2y_pos10_aux02_full3000",
    ),
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.4f}"


def pick(rows: list[dict], aggregate: str, horizon: float | None = None) -> dict | None:
    for row in rows:
        if row.get("aggregate") != aggregate:
            continue
        if horizon is None:
            return row
        if abs(to_float(row.get("horizon_years")) - horizon) < 1e-6:
            return row
    return None


def main() -> int:
    out_dir = REPO_DIR / "results" / "high_priority_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, family, rel_dir in RUNS:
        result_dir = REPO_DIR / rel_dir
        metrics = read_csv(result_dir / "aggregate_metrics.csv")
        config = read_config(result_dir / "run_config.json")
        overall = pick(metrics, "overall") or {}
        h5 = pick(metrics, "horizon", 5.0) or {}
        h10 = pick(metrics, "horizon", 10.0) or {}
        rows.append(
            {
                "run_id": run_id,
                "family": family,
                "status": "done" if metrics else "missing",
                "checkpoint": config.get("checkpoint", ""),
                "checkpoint_iter": config.get("checkpoint_iter", ""),
                "val_best_auc": config.get("checkpoint_best_val_auc_mean", ""),
                "auc_5y": fmt(to_float(h5.get("auc_mean"))),
                "auc_10y": fmt(to_float(h10.get("auc_mean"))),
                "auc_overall": fmt(to_float(overall.get("auc_mean"))),
                "brier_overall": fmt(to_float(overall.get("brier_mean"))),
                "ece_overall": fmt(to_float(overall.get("ece_mean"))),
                "capture_overall": fmt(to_float(overall.get("top_decile_capture_mean"))),
                "result_dir": rel_dir,
            }
        )

    fieldnames = list(rows[0].keys())
    csv_path = out_dir / "high_priority_ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# High-Priority Ablation Summary",
        "",
        "| Run | Family | Status | 5y AUC | 10y AUC | Overall AUC | Brier | ECE | Capture |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {family} | {status} | {auc_5y} | {auc_10y} | {auc_overall} | {brier_overall} | {ece_overall} | {capture_overall} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Main-Model Decision Rule",
            "",
            "- Keep `MedTrajectory horizon risk` as the main model unless a survival-horizon candidate matches or exceeds it on locked-test overall AUC and does not regress calibration/top-decile capture.",
            "- Treat survival-horizon as the model-innovation branch when it improves architectural coherence but remains below the explicit BCE horizon head.",
        ]
    )
    (out_dir / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
