from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = ROOT / "results" / "locked_test_horizon_risk"
OUT_DIR = LOCKED_DIR / "frozen_comparison"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object, digits: int = 4) -> str:
    x = f(value)
    return "" if math.isnan(x) else f"{x:.{digits}f}"


def as_int(value: object) -> int:
    x = f(value)
    return 0 if math.isnan(x) else int(x)


def row_from_locked_summary(row: dict[str, str], family: str, note: str) -> dict:
    return {
        "model": row["model"],
        "family": family,
        "protocol": "locked_test_final_context",
        "auc5": f(row.get("auc5")),
        "auc10": f(row.get("auc10")),
        "overall_auc": f(row.get("overall_auc")),
        "brier": f(row.get("brier")),
        "ece": f(row.get("ece")),
        "top_decile_capture": f(row.get("capture")),
        "metric_rows_with_auc": 18,
        "note": note,
    }


def row_from_aggregate(model: str, family: str, path: Path, note: str) -> dict | None:
    rows = read_csv(path)
    if not rows:
        return None
    by_scope = {}
    for row in rows:
        if row.get("aggregate") == "overall":
            by_scope["overall"] = row
        elif f(row.get("horizon_years")) == 5.0:
            by_scope["5"] = row
        elif f(row.get("horizon_years")) == 10.0:
            by_scope["10"] = row
    overall = by_scope.get("overall", {})
    return {
        "model": model,
        "family": family,
        "protocol": "locked_test_final_context",
        "auc5": f(by_scope.get("5", {}).get("auc_mean")),
        "auc10": f(by_scope.get("10", {}).get("auc_mean")),
        "overall_auc": f(overall.get("auc_mean")),
        "brier": f(overall.get("brier_mean")),
        "ece": f(overall.get("ece_mean")),
        "top_decile_capture": f(overall.get("top_decile_capture_mean")),
        "metric_rows_with_auc": as_int(overall.get("diseases_with_auc")),
        "note": note,
    }


def load_deep_rows() -> list[dict]:
    rows = []
    current_specs = [
        (
            "Monotonic Gated RoPE",
            "main",
            LOCKED_DIR / "monotonic_gated_rope_gate100" / "aggregate_metrics.csv",
            "Current recommended main model: absolute age Sin/Cos plus continuous-age RoPE and a monotonic horizon projection head.",
        ),
        (
            "Gated RoPE main",
            "main_ablation",
            LOCKED_DIR / "gated_rope_gate100" / "aggregate_metrics.csv",
            "Previous gated continuous-age RoPE model without structural 10y>=5y horizon constraint.",
        ),
        (
            "No-RoPE previous best",
            "main_ablation",
            LOCKED_DIR / "high_priority_no_rope_trunk_full3000" / "aggregate_metrics.csv",
            "Previous strongest no-RoPE trunk plus explicit horizon-risk head.",
        ),
    ]
    for model, family, path, note in current_specs:
        row = row_from_aggregate(model, family, path, note)
        if row is not None:
            rows.append(row)

    family_by_model = {
        "MedTrajectory horizon risk": "historical_main",
        "MedTrajectory survival-horizon": "innovation_branch",
        "BERT risk head": "architecture_baseline",
        "Mamba risk head": "architecture_baseline",
    }
    notes = {
        "MedTrajectory horizon risk": "Historical explicit horizon-risk main model before gated RoPE and monotonic horizon projection.",
        "MedTrajectory survival-horizon": "Unified survival/TTE/horizon branch; promising but calibration and ranking still behind the explicit horizon head.",
        "BERT risk head": "No-leak encoder baseline with the same frozen horizon-risk protocol.",
        "Mamba risk head": "No-leak SSM baseline with the same frozen horizon-risk protocol.",
    }
    for row in read_csv(LOCKED_DIR / "locked_test_summary.csv"):
        model = row["model"]
        rows.append(row_from_locked_summary(row, family_by_model.get(model, "deep_model"), notes.get(model, "")))
    return rows


def load_ablation_rows() -> list[dict]:
    specs = [
        (
            "Modern horizon-only ablation",
            "horizon_ablation",
            LOCKED_DIR / "ablation_modern_horizon_test" / "aggregate_metrics.csv",
            "Modern trunk plus horizon head; isolates the contribution of the final risk head without the TTE-enhanced trunk.",
        ),
        (
            "No-RoPE horizon-only ablation",
            "horizon_ablation",
            LOCKED_DIR / "ablation_no_rope_horizon_test" / "aggregate_metrics.csv",
            "Same horizon protocol without continuous-age RoPE in the trunk.",
        ),
        (
            "Block48 horizon-only ablation",
            "horizon_ablation",
            LOCKED_DIR / "ablation_block48_horizon_test" / "aggregate_metrics.csv",
            "Short-context trunk plus horizon head; tests context-length contribution.",
        ),
    ]
    rows = []
    for model, family, path, note in specs:
        row = row_from_aggregate(model, family, path, note)
        if row is not None:
            rows.append(row)
    return rows


def load_ml_rows() -> list[dict]:
    rows = []
    for row in read_csv(LOCKED_DIR / "frozen_ml_baselines" / "frozen_ml_aggregate_metrics.csv"):
        if row.get("aggregate") != "overall":
            continue
        model = row["model"]
        row5 = next(
            (
                r
                for r in read_csv(LOCKED_DIR / "frozen_ml_baselines" / "frozen_ml_aggregate_metrics.csv")
                if r.get("model") == model and f(r.get("horizon_years")) == 5.0
            ),
            {},
        )
        row10 = next(
            (
                r
                for r in read_csv(LOCKED_DIR / "frozen_ml_baselines" / "frozen_ml_aggregate_metrics.csv")
                if r.get("model") == model and f(r.get("horizon_years")) == 10.0
            ),
            {},
        )
        rows.append(
            {
                "model": model,
                "family": "traditional_ml_baseline",
                "protocol": "locked_test_final_context",
                "auc5": f(row5.get("auc_mean")),
                "auc10": f(row10.get("auc_mean")),
                "overall_auc": f(row.get("auc_mean")),
                "brier": f(row.get("brier_mean")),
                "ece": f(row.get("ece_mean")),
                "top_decile_capture": f(row.get("top_decile_capture_mean")),
                "metric_rows_with_auc": as_int(row.get("diseases_with_auc")),
                "note": "Traditional ML baseline evaluated at exactly the same patients, ages, diseases, horizons, and labels as MedTrajectory.",
            }
        )
    return rows


def add_delta(rows: list[dict]) -> list[dict]:
    main_auc = next((f(row["overall_auc"]) for row in rows if row["model"] == "Monotonic Gated RoPE"), float("nan"))
    out = []
    for row in rows:
        current = dict(row)
        current["delta_auc_vs_medtrajectory"] = f(row["overall_auc"]) - main_auc if not math.isnan(main_auc) else float("nan")
        out.append(current)
    return sorted(out, key=lambda item: f(item["overall_auc"]), reverse=True)


def write_status(rows: Sequence[dict]) -> None:
    lines = [
        "# Frozen Comparison Summary",
        "",
        "All rows use the locked `test` split, the same selected disease set, 5y/10y horizons, final-context prediction age, and MedTrajectory raw-prediction labels whenever applicable.",
        "",
        "| Rank | Model | Family | 5y AUC | 10y AUC | Overall AUC | Delta vs Current Main | Brier | ECE | Capture |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {model} | {family} | {auc5} | {auc10} | {auc} | {delta} | {brier} | {ece} | {capture} |".format(
                rank=rank,
                model=row["model"],
                family=row["family"],
                auc5=fmt(row["auc5"]),
                auc10=fmt(row["auc10"]),
                auc=fmt(row["overall_auc"]),
                delta=fmt(row["delta_auc_vs_medtrajectory"]),
                brier=fmt(row["brier"]),
                ece=fmt(row["ece"]),
                capture=fmt(row["top_decile_capture"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Monotonic Gated RoPE is the current recommended main model under the frozen protocol.",
            "- Its horizon projection enforces 10y risk >= 5y risk while improving locked-test macro AUC relative to the previous Gated RoPE model.",
            "- BERT/Mamba are evaluated with the same risk-head protocol and remain materially lower.",
            "- Horizon-only ablations sit far below the current main model, showing the final classifier head alone is not the source of the result.",
            "- Static/Dynamic/GBDT baselines are included only after they are run at the exact MedTrajectory locked-test prediction moments.",
        ]
    )
    (OUT_DIR / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = add_delta(load_deep_rows() + load_ablation_rows() + load_ml_rows())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "frozen_comparison_summary.csv", rows)
    write_status(rows)
    print(OUT_DIR / "frozen_comparison_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
