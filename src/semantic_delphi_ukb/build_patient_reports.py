from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo" / "patient_future_demo_tte_multitask"
OUT_DIR = ROOT / "demo" / "patient_reports"
LOCKED_RAW = ROOT / "results" / "locked_test_horizon_risk" / "medtrajectory_test" / "raw_predictions.csv"


DISEASE_LABELS = {
    "CKD": "Chronic kidney disease",
    "IHD": "Ischemic heart disease",
    "CVD": "Cerebrovascular disease",
    "COPD": "Chronic obstructive pulmonary disease",
}


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def load_locked_scores() -> dict[tuple[int, str, str], dict[str, str]]:
    out = {}
    for row in read_csv(LOCKED_RAW):
        key = (int(row["patient_index"]), row["disease_id"], f"{float(row['horizon_years']):g}")
        out[key] = row
    return out


def recent_history(events: Sequence[dict], baseline_age: float, limit: int = 8) -> list[dict]:
    visible = [event for event in events if float(event["age_years"]) <= baseline_age + 1e-9]
    return visible[-limit:]


def top_selected(data: dict, limit: int = 5) -> list[dict]:
    rows = list(data.get("top_selected_diseases", []))
    return rows[:limit]


def matched_locked_risks(data: dict, locked: dict[tuple[int, str, str], dict[str, str]]) -> list[dict]:
    patient = int(data["split_patient_index"])
    rows = []
    for disease in top_selected(data, limit=5):
        disease_id = disease["disease_id"]
        r5 = locked.get((patient, disease_id, "5"))
        r10 = locked.get((patient, disease_id, "10"))
        rows.append(
            {
                "disease_id": disease_id,
                "disease": disease["disease"],
                "best_token": disease["best_token"],
                "rank": disease["rank"],
                "rank_among_diagnosis_tokens": disease["rank_among_diagnosis_tokens"],
                "risk_5y": r5.get("score", "") if r5 else "",
                "label_5y": r5.get("label", "") if r5 else "",
                "risk_10y": r10.get("score", "") if r10 else "",
                "label_10y": r10.get("label", "") if r10 else "",
                "appeared_in_future": disease.get("appeared_in_future", False),
                "years_after_baseline": disease.get("years_after_baseline"),
            }
        )
    return rows


def event_label(event: dict) -> str:
    return f"{fmt(event.get('age_years'), 1)}y `{event.get('token', '')}` ({event.get('event_type', '')})"


def write_report(data: dict, locked: dict[tuple[int, str, str], dict[str, str]]) -> dict:
    disease_key = data["case_id"].split("-", 1)[0]
    report_path = OUT_DIR / f"{data['case_id']}_patient_report.md"
    risks = matched_locked_risks(data, locked)
    history = recent_history(data.get("visible_history", []), float(data["baseline_age_years"]))
    future_events = data.get("future_events", [])[:6]
    actual_scored = data.get("actual_future_scored_events", [])[:6]
    lines = [
        f"# Patient Report Demo: {data['case_id']}",
        "",
        "用途: 科研辅助/队列风险分层，不作为临床诊断依据。",
        "",
        "## Snapshot",
        "",
        f"- Split: `{data['split']}`; redacted patient index: `{data['split_patient_index']}`.",
        f"- Baseline age: `{fmt(data['baseline_age_years'], 1)}` years; visible events: `{data['num_events']}`.",
        f"- Hidden target: `{data['target_token']}` ({data['target_disease']}); first hidden event occurs after `{fmt(data['target_gap_days'], 1)}` days.",
        f"- Demo disease focus: `{disease_key}` / {DISEASE_LABELS.get(disease_key, data['target_disease'])}.",
        "",
        "## Locked-Test Horizon Risk",
        "",
        "The 5y/10y scores below are looked up from the locked-test final-context evaluator for the same redacted patient index. The demo hidden-event trajectory uses its own earlier cutpoint, so future-observed status is shown separately and should not be read as the locked-test label.",
        "",
        "| Disease | Best token | Rank | 5y risk score | 10y risk score | Future observed in demo |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in risks:
        observed = "yes" if row["appeared_in_future"] else "no"
        if row["years_after_baseline"] not in ("", None):
            observed += f" ({fmt(row['years_after_baseline'], 2)}y)"
        lines.append(
            "| {disease} | `{token}` | {rank} | {r5} | {r10} | {obs} |".format(
                disease=row["disease"],
                token=row["best_token"],
                rank=row["rank"],
                r5=fmt(row["risk_5y"], 4),
                r10=fmt(row["risk_10y"], 4),
                obs=observed,
            )
        )
    lines.extend(
        [
            "",
            "## Key Visible History",
            "",
        ]
    )
    for event in history:
        lines.append(f"- {event_label(event)}")
    lines.extend(
        [
            "",
            "## Hidden Follow-Up Events For Demo Verification",
            "",
        ]
    )
    for event in future_events:
        lines.append(f"- {event_label(event)}")
    lines.extend(
        [
            "",
            "## Model Evidence At Baseline",
            "",
        ]
    )
    for item in actual_scored:
        lines.append(
            f"- Future `{item['token']}` at {fmt(item['age_years'], 1)}y had baseline rank `{item['rank_within_event_type']}` within `{item['event_type']}` candidates."
        )
    lines.extend(
        [
            "",
            "## Interpretation Caveat",
            "",
            "The report is a presentation demo built from held-out trajectory examples. Scores are model outputs for risk ranking and cohort stratification; they are not calibrated clinical probabilities for diagnosis or treatment decisions.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    focus = risks[0] if risks else {}
    return {
        "case_id": data["case_id"],
        "patient_index": data["split_patient_index"],
        "baseline_age_years": data["baseline_age_years"],
        "target_token": data["target_token"],
        "target_rank": data.get("target_rank_among_diagnosis_tokens", ""),
        "top_selected_disease": focus.get("disease", ""),
        "top_selected_risk_5y": focus.get("risk_5y", ""),
        "top_selected_risk_10y": focus.get("risk_10y", ""),
        "report_md": str(report_path.relative_to(ROOT)),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    locked = load_locked_scores()
    summary = []
    for path in sorted(DEMO_DIR.glob("*_demo.json")):
        summary.append(write_report(load_json(path), locked))
    write_csv(OUT_DIR / "patient_report_index.csv", summary)
    (OUT_DIR / "README.md").write_text(
        "# Patient Report Demo\n\n"
        "One-page Markdown patient reports generated from held-out demo JSON files and locked-test MedTrajectory risk scores when available.\n\n"
        "Disclaimer: 科研辅助/队列风险分层，不作为临床诊断依据。\n",
        encoding="utf-8",
    )
    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
