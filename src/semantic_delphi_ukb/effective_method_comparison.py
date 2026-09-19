from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_DIR / "tests" / "output" / "effective_method_comparison"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json_metric(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["metrics"][0]
    return {
        "patients": row.get("patients"),
        "full_top1": row.get("top1_acc"),
        "full_top5": row.get("top5_acc"),
        "full_top10": row.get("top10_acc"),
        "best_val_loss": payload.get("best_val_loss"),
    }


def f(value: object) -> float:
    if value in (None, "", "nan"):
        return float("nan")
    return float(value)


def fmt(value: object, digits: int = 4) -> str:
    value = f(value)
    return "" if value != value else f"{value:.{digits}f}"


def mean_field(rows: Iterable[dict], field: str, min_positives: int = 0) -> float:
    values = [f(row[field]) for row in rows if int(float(row.get("positives", 0) or 0)) >= min_positives]
    values = [value for value in values if value == value]
    return mean(values) if values else float("nan")


def selected_summary() -> dict[str, dict]:
    rows = read_csv(REPO_DIR / "tests" / "output" / "selected_disease_demo" / "selected_disease_metrics.csv")
    out: dict[str, dict] = {}
    for model_id in sorted({row["model_id"] for row in rows}):
        model_rows = [row for row in rows if row["model_id"] == model_id]
        out[model_id] = {
            "selected_auc_mean": mean_field(model_rows, "auc"),
            "selected_group_top10_mean": mean_field(model_rows, "group_top10"),
            "selected_capture_mean": mean_field(model_rows, "top_decile_capture"),
            "selected_source": "tests/output/selected_disease_demo/selected_disease_metrics.csv",
        }
    return out


def horizon_summary() -> dict[str, dict]:
    static_rows = read_csv(REPO_DIR / "tests" / "output" / "static_ml_baseline" / "static_ml_metrics.csv")
    dynamic_rows = read_csv(
        REPO_DIR / "tests" / "output" / "dynamic_ml_landmark_baseline" / "dynamic_ml_landmark_metrics.csv"
    )
    tte_rows = read_csv(REPO_DIR / "tests" / "output" / "selected_disease_tte_head_direct" / "tte_head_metrics.csv")
    out = {}
    for label, rows, source in [
        ("Static ML baseline", static_rows, "tests/output/static_ml_baseline/static_ml_metrics.csv"),
        (
            "Dynamic ML landmark baseline",
            dynamic_rows,
            "tests/output/dynamic_ml_landmark_baseline/dynamic_ml_landmark_metrics.csv",
        ),
        ("TTE direct head", tte_rows, "tests/output/selected_disease_tte_head_direct/tte_head_metrics.csv"),
    ]:
        for min_pos in [0, 10]:
            suffix = "" if min_pos == 0 else "_pos10"
            out[label + suffix] = {
                "tte_5_auc": mean_field([row for row in rows if f(row["horizon_years"]) == 5.0], "auc", min_pos),
                "tte_10_auc": mean_field([row for row in rows if f(row["horizon_years"]) == 10.0], "auc", min_pos),
                "tte_5_capture": mean_field(
                    [row for row in rows if f(row["horizon_years"]) == 5.0], "top_decile_capture", min_pos
                ),
                "tte_10_capture": mean_field(
                    [row for row in rows if f(row["horizon_years"]) == 10.0], "top_decile_capture", min_pos
                ),
                "horizon_source": source,
            }
    return out


def topk_rows() -> list[dict]:
    rows: list[dict] = []
    selected = selected_summary()
    horizon = horizon_summary()
    summary_selected = {
        "Old Exp2": {
            **selected["exp2"],
            "selected_source": "tests/output/selected_disease_demo/selected_disease_metrics.csv",
        },
        "Modern baseline (RoPE, block128)": {
            "selected_auc_mean": 0.7912,
            "selected_group_top10_mean": 0.4617,
            "selected_capture_mean": 0.5460,
            "selected_source": "MEDTRAJECTORY_CURRENT_RESULTS_SUMMARY.md",
        },
        "TTE w=0.05, h=10 (prior)": {
            "selected_auc_mean": 0.7872,
            "selected_group_top10_mean": 0.4708,
            "selected_capture_mean": 0.5160,
            "selected_source": "MEDTRAJECTORY_CURRENT_RESULTS_SUMMARY.md",
        },
    }

    original = read_json_metric(REPO_DIR / "tests" / "output" / "model_compare_explicit_split" / "delphi_demo_val.json")
    rows.append(
        {
            "stage": "0",
            "method": "Original Delphi demo anchor",
            "architecture": "Original Delphi checkpoint on demo/simulated data",
            "status": "context anchor, not fair direct baseline",
            **original,
            "source": "tests/output/model_compare_explicit_split/delphi_demo_val.json",
        }
    )

    early = [
        ("1", "Disease-only explicit split", "diagnosis-only semantic Delphi", "disease_only_explicit_split_test.json", "disease_only"),
        ("2", "Exp1 multitype", "diagnosis + procedure", "exp1_explicit_split_test.json", "exp1"),
        ("3", "Exp2 multitype / Old Exp2", "diagnosis + procedure + cancer + death", "exp2_explicit_split_test.json", "exp2"),
    ]
    for stage, method, architecture, filename, selected_key in early:
        metric = read_json_metric(REPO_DIR / "tests" / "output" / "model_compare_explicit_split" / filename)
        rows.append(
            {
                "stage": stage,
                "method": method,
                "architecture": architecture,
                "status": "fair explicit-split baseline",
                **metric,
                **selected[selected_key],
                "source": "tests/output/model_compare_explicit_split/" + filename,
            }
        )

    for row in read_csv(REPO_DIR / "tests" / "output" / "medtrajectory_tte_sweep_summary" / "medtrajectory_tte_sweep_summary.csv"):
        method = row["model"]
        status = "main next-event model" if "Modern baseline" in method else "ablation or TTE sweep"
        extra = horizon.get("TTE direct head", {}) if method.startswith("TTE w=0.05") else {}
        selected_extra = summary_selected.get(method, {})
        rows.append(
            {
                "stage": "4" if method.startswith("Modern") or method == "Old Exp2" else "6",
                "method": method,
                "architecture": "modern trunk / ablation / TTE variant",
                "status": status,
                "full_top1": row["full_top1"],
                "full_top5": row["full_top5"],
                "full_top10": row["full_top10"],
                "diag_top1": row["diag_top1"],
                "diag_top5": row["diag_top5"],
                "diag_top10": row["diag_top10"],
                "best_val_loss": row["best_val_loss"],
                "tte_5_auc": row.get("tte_5_auc") or extra.get("tte_5_auc"),
                "tte_10_auc": row.get("tte_10_auc") or extra.get("tte_10_auc"),
                "tte_5_capture": row.get("tte_5_top_decile_capture") or extra.get("tte_5_capture"),
                "tte_10_capture": row.get("tte_10_top_decile_capture") or extra.get("tte_10_capture"),
                **selected_extra,
                "source": "tests/output/medtrajectory_tte_sweep_summary/medtrajectory_tte_sweep_summary.csv",
            }
        )

    rows.append(
        {
            "stage": "5",
            "method": "Type-softmax",
            "architecture": "type-aware output head",
            "status": "diagnostic negative result",
            "full_top1": 0.2126,
            "full_top5": 0.3869,
            "full_top10": 0.4648,
            "diag_top1": 0.2559,
            "diag_top5": 0.4596,
            "diag_top10": 0.5472,
            "source": "MEDTRAJECTORY_CURRENT_RESULTS_SUMMARY.md",
        }
    )

    rows.append(
        {
            "stage": "7",
            "method": "Static ML baseline",
            "architecture": "static UKB fields only; weighted ridge/logistic fallback",
            "status": "traditional ML risk baseline",
            **horizon["Static ML baseline"],
            "source": "tests/output/static_ml_baseline/static_ml_metrics.csv",
        }
    )
    rows.append(
        {
            "stage": "8",
            "method": "Dynamic ML landmark baseline",
            "architecture": "static + engineered event-history landmark features",
            "status": "strong traditional dynamic baseline",
            **horizon["Dynamic ML landmark baseline"],
            "source": "tests/output/dynamic_ml_landmark_baseline/dynamic_ml_landmark_metrics.csv",
        }
    )
    return sorted(rows, key=lambda row: (float(row.get("stage") or 99), row["method"]))


def demo_disease_rows() -> list[dict]:
    wanted = {
        "ischemic_heart_disease": "IHD",
        "cerebrovascular_disease": "CVD",
        "chronic_kidney_disease": "CKD",
        "copd": "COPD",
    }
    static = read_csv(REPO_DIR / "tests" / "output" / "static_ml_baseline" / "static_ml_metrics.csv")
    dynamic = read_csv(REPO_DIR / "tests" / "output" / "dynamic_ml_landmark_baseline" / "dynamic_ml_landmark_metrics.csv")
    tte = read_csv(REPO_DIR / "tests" / "output" / "selected_disease_tte_head_direct" / "tte_head_metrics.csv")
    selected = read_csv(REPO_DIR / "tests" / "output" / "selected_disease_demo" / "selected_disease_metrics.csv")
    selected_exp2 = {row["disease_id"]: row for row in selected if row["model_id"] == "exp2"}
    static_by_key = {(row["disease_id"], f(row["horizon_years"])): row for row in static}
    dynamic_by_key = {(row["disease_id"], f(row["horizon_years"])): row for row in dynamic}
    tte_by_key = {(row["disease_id"], f(row["horizon_years"])): row for row in tte}
    out = []
    for disease_id, short_name in wanted.items():
        for horizon in [5.0, 10.0]:
            s = static_by_key[(disease_id, horizon)]
            dyn = dynamic_by_key[(disease_id, horizon)]
            t = tte_by_key[(disease_id, horizon)]
            exp2 = selected_exp2[disease_id]
            out.append(
                {
                    "disease": short_name,
                    "disease_id": disease_id,
                    "horizon_years": horizon,
                    "exp2_next_token_group_auc": exp2["auc"],
                    "static_auc": s["auc"],
                    "dynamic_ml_auc": dyn["auc"],
                    "tte_direct_auc": t["auc"],
                    "static_top_decile_capture": s["top_decile_capture"],
                    "dynamic_ml_top_decile_capture": dyn["top_decile_capture"],
                    "tte_top_decile_capture": t["top_decile_capture"],
                    "static_positives": s["positives"],
                    "tte_prediction_moments": t["prediction_moments"],
                }
            )
    return out


def write_markdown(path: Path, rows: Sequence[dict], demo_rows: Sequence[dict]) -> None:
    lines = [
        "# Effective Method Comparison",
        "",
        "This table compares the useful methods by task family instead of treating TTE as the only baseline.",
        "",
        "Important: next-event Top-K, selected-disease next-token group AUC, and horizon risk AUC are different tasks.",
        "",
        "## Main Comparison",
        "",
        "| Stage | Method | Full Top10 | Diagnosis Top10 | Selected AUC | 5y Risk AUC | 10y Risk AUC | Interpretation |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        interpretation = row.get("status", "")
        lines.append(
            "| {stage} | {method} | {full_top10} | {diag_top10} | {sel_auc} | {auc5} | {auc10} | {status} |".format(
                stage=row.get("stage", ""),
                method=row["method"],
                full_top10=fmt(row.get("full_top10")),
                diag_top10=fmt(row.get("diag_top10")),
                sel_auc=fmt(row.get("selected_auc_mean")),
                auc5=fmt(row.get("tte_5_auc")),
                auc10=fmt(row.get("tte_10_auc")),
                status=interpretation,
            )
        )

    lines.extend(
        [
            "",
            "## Four Demo Diseases",
            "",
            "| Disease | Horizon | Exp2 group AUC | Static AUC | Dynamic ML AUC | TTE direct AUC | Dynamic capture | TTE capture | Positives |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in demo_rows:
        lines.append(
            "| {disease} | {horizon:g} | {exp2} | {static} | {dyn} | {tte} | {dcap} | {tcap} | {pos} |".format(
                disease=row["disease"],
                horizon=f(row["horizon_years"]),
                exp2=fmt(row["exp2_next_token_group_auc"]),
                static=fmt(row["static_auc"]),
                dyn=fmt(row["dynamic_ml_auc"]),
                tte=fmt(row["tte_direct_auc"]),
                dcap=fmt(row["dynamic_ml_top_decile_capture"]),
                tcap=fmt(row["tte_top_decile_capture"]),
                pos=row["static_positives"],
            )
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- The largest jump is not TTE; it is disease-only to multitype Exp1/Exp2, because adding procedures/cancer/death changes the sequence modeling problem.",
            "- Modern trunk changes are real but modest: better full Top-K and validation loss, with RoPE helping full sequence metrics more than strict diagnosis Top10.",
            "- TTE multitask is the best next-event/diagnosis Top-K variant, but selected-disease logits remain mixed.",
            "- Dynamic ML landmark features are a strong baseline and narrow the gap substantially; TTE is not globally dominant over engineered trajectory features.",
            "- Direct TTE head is the clinically aligned risk view; its clearest advantage over dynamic ML is CKD, while some cardiovascular outcomes are close or favor engineered features.",
            "- Static ML is a necessary baseline: it proves age/body/static UKB fields already explain meaningful risk, so claims should emphasize incremental trajectory value, not absolute dominance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = topk_rows()
    demo_rows = demo_disease_rows()
    write_csv(OUTPUT_DIR / "effective_method_main_table.csv", rows)
    write_csv(OUTPUT_DIR / "effective_method_demo_diseases.csv", demo_rows)
    write_markdown(OUTPUT_DIR / "effective_method_comparison.md", rows, demo_rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "rows": len(rows), "demo_rows": len(demo_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
