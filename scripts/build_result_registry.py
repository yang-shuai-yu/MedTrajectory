#!/usr/bin/env python3
"""Build a line-level result registry from frozen result sources.

The registry is deliberately append-like: old results remain available with
``status=superseded``/``diagnostic`` while the audit decides what can be used
in a primary comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "record_id", "result_class", "task", "metric", "value", "lower_ci", "upper_ci",
    "model", "checkpoint", "seed", "split", "dataset_profile", "horizon_years",
    "landmark_protocol", "censoring", "disease_panel", "aggregation", "comparison_group",
    "source_file", "source_locator", "status", "comparability", "supersedes",
    "freeze_manifest", "notes",
]
CLASSES = {"lm_logits", "horizon_head", "other_medical", "literature_reference"}


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _record(raw: Dict[str, Any], source: str = "", locator: str = "") -> Dict[str, Any]:
    row = {field: raw.get(field) for field in FIELDS}
    row["record_id"] = raw.get("record_id") or _slug(
        f"{raw.get('split','na')}_{raw.get('model','na')}_{raw.get('task','na')}_{raw.get('metric','na')}_{raw.get('aggregation','na')}"
    )
    row["source_file"] = raw.get("source_file") or source
    row["source_locator"] = raw.get("source_locator") or locator
    row["status"] = raw.get("status") or "primary"
    row["comparability"] = raw.get("comparability") or "within_protocol"
    row["aggregation"] = raw.get("aggregation") or "unspecified"
    row["comparison_group"] = raw.get("comparison_group") or "unspecified"
    row["result_class"] = raw.get("result_class")
    if row["result_class"] not in CLASSES:
        raise ValueError(f"invalid result_class for {row['record_id']}: {row['result_class']}")
    if row["value"] is not None:
        row["value"] = float(row["value"])
    for key in ("lower_ci", "upper_ci"):
        if row[key] is not None:
            row[key] = float(row[key])
    return row


def _formal_records(spec: Dict[str, Any], source: Path) -> Iterable[Dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    mapping = {
        "lm_auc": ("lm_logits", "next_event_medical_auc", "auc"),
        "risk_auc": ("horizon_head", "medical_horizon_auc", "auc"),
        "long_auc": ("other_medical", "longitudinal_auc", "auc"),
    }
    for model, model_payload in payload["per_experiment"].items():
        for seed, metrics in model_payload.items():
            if seed == "aggregate":
                for key, stats in metrics.items():
                    result_class, task, metric = mapping[key]
                    yield _record({
                        "record_id": f"formal_{model}_aggregate_{metric}_{key}",
                        "result_class": result_class, "task": task, "metric": metric,
                        "value": stats["mean"], "lower_ci": stats.get("ci95_low"),
                        "upper_ci": stats.get("ci95_high"), "model": model,
                        "split": spec["split"], "dataset_profile": spec["dataset_profile"],
                        "seed": None, "horizon_years": "1,5,10" if key == "risk_auc" else None,
                        "landmark_protocol": "official_medical_auc_v1" if key == "lm_auc" else "paper_medical_horizon_v1",
                        "censoring": "paper_censor_aware_v1" if key == "risk_auc" else "not_applicable",
                        "disease_panel": "selected_diseases", "aggregation": "mean_across_seeds",
                        "comparison_group": f"{spec['comparison_group_prefix']}_{result_class}",
                        "status": "primary", "comparability": "within_protocol",
                        "notes": f"n={stats.get('n', 3)}; SD={stats.get('sd')}"
                    }, str(source), f"per_experiment.{model}.aggregate.{key}")
            else:
                for key, value in metrics.items():
                    result_class, task, metric = mapping[key]
                    yield _record({
                        "record_id": f"formal_{model}_seed{seed}_{key}",
                        "result_class": result_class, "task": task, "metric": metric,
                        "value": value, "model": model, "seed": int(seed),
                        "split": spec["split"], "dataset_profile": spec["dataset_profile"],
                        "horizon_years": "1,5,10" if key == "risk_auc" else None,
                        "landmark_protocol": "official_medical_auc_v1" if key == "lm_auc" else "paper_medical_horizon_v1",
                        "censoring": "paper_censor_aware_v1" if key == "risk_auc" else "not_applicable",
                        "disease_panel": "selected_diseases", "aggregation": "per_seed",
                        "comparison_group": f"{spec['comparison_group_prefix']}_{result_class}",
                        "status": "primary", "comparability": "within_protocol",
                    }, str(source), f"per_experiment.{model}.{seed}.{key}")
    for contrast_key, stats in payload.get("paired", {}).items():
        contrast, key = contrast_key.split(":", 1)
        result_class, task, metric = mapping[key]
        group = f"{spec['comparison_group_prefix']}_paired_{result_class}"
        base = {
            "result_class": result_class, "task": f"validation_paired_{task}", "model": contrast,
            "split": spec["split"], "dataset_profile": spec["dataset_profile"],
            "horizon_years": "1,5,10" if key == "risk_auc" else None,
            "landmark_protocol": "official_medical_auc_v1" if key == "lm_auc" else "paper_medical_horizon_v1",
            "censoring": "paper_censor_aware_v1" if key == "risk_auc" else "not_applicable",
            "disease_panel": "selected_diseases", "aggregation": "paired_seed_summary",
            "comparison_group": group, "status": "descriptive", "comparability": "within_protocol",
        }
        yield _record({**base, "record_id": f"formal_paired_{_slug(contrast_key)}_mean", "metric": "delta_auc",
                       "value": stats["mean"], "lower_ci": stats.get("ci95_low"), "upper_ci": stats.get("ci95_high"),
                       "notes": f"p_two_sided={stats.get('p_two_sided')}; n={stats.get('n')}"},
                      str(source), f"paired.{contrast_key}.mean")
        yield _record({**base, "record_id": f"formal_paired_{_slug(contrast_key)}_p", "metric": "p_two_sided",
                       "value": stats["p_two_sided"]}, str(source), f"paired.{contrast_key}.p_two_sided")


def _test_records(spec: Dict[str, Any], source: Path) -> Iterable[Dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for model, metrics in payload["models"].items():
        if "strata" in metrics:
            yield _record({
                "record_id": f"test_{model}_strata_count", "result_class": "horizon_head",
                "task": "locked_test_protocol_coverage", "metric": "strata_count", "value": metrics["strata"],
                "model": model, "seed": int(model.rsplit("seed", 1)[1]), "split": spec["split"],
                "dataset_profile": spec["dataset_profile"], "horizon_years": spec["horizon_years"],
                "landmark_protocol": spec["landmark_protocol"], "censoring": spec["censoring"],
                "disease_panel": "selected_diseases", "aggregation": "macro",
                "comparison_group": spec["comparison_group"], "status": "primary",
                "comparability": "locked_test_same_landmarks", "freeze_manifest": spec["freeze_manifest"],
            }, str(source), f"models.{model}.strata")
        for metric in ("auc", "auprc", "brier", "ece"):
            if metric not in metrics:
                continue
            task = "locked_test_horizon_risk" if metric in {"auc", "auprc"} else "locked_test_calibration"
            yield _record({
                "record_id": f"test_{model}_{metric}", "result_class": "horizon_head",
                "task": task, "metric": metric, "value": metrics[metric], "model": model,
                "seed": int(model.rsplit("seed", 1)[1]), "split": spec["split"],
                "dataset_profile": spec["dataset_profile"], "horizon_years": spec["horizon_years"],
                "landmark_protocol": spec["landmark_protocol"], "censoring": spec["censoring"],
                "disease_panel": "selected_diseases", "aggregation": "macro",
                "comparison_group": spec["comparison_group"], "status": "primary",
                "comparability": "locked_test_same_landmarks", "freeze_manifest": spec["freeze_manifest"],
            }, str(source), f"models.{model}.{metric}")
    for contrast, stats in payload.get("paired_bootstrap", {}).items():
        yield _record({
            "record_id": f"test_paired_{_slug(contrast)}", "result_class": "horizon_head",
            "task": "locked_test_paired_bootstrap", "metric": "delta_auc", "value": stats["delta_auc"],
            "lower_ci": stats.get("ci95_low"), "upper_ci": stats.get("ci95_high"), "model": contrast,
            "split": spec["split"], "dataset_profile": spec["dataset_profile"],
            "horizon_years": spec["horizon_years"], "landmark_protocol": spec["landmark_protocol"],
            "censoring": spec["censoring"], "disease_panel": "selected_diseases",
            "aggregation": "paired_patient_bootstrap", "comparison_group": spec["comparison_group"],
            "status": "primary", "comparability": "locked_test_same_landmarks",
            "freeze_manifest": spec["freeze_manifest"], "notes": f"Holm p={stats.get('p_holm')}; n={stats.get('bootstrap_used')}"
        }, str(source), f"paired_bootstrap.{contrast}")
        for metric in ("p_two_sided", "p_holm", "bootstrap_used", "common_rows", "common_patients"):
            if metric not in stats:
                continue
            yield _record({
                "record_id": f"test_paired_{_slug(contrast)}_{metric}", "result_class": "horizon_head",
                "task": "locked_test_paired_bootstrap", "metric": metric, "value": stats[metric],
                "model": contrast, "split": spec["split"], "dataset_profile": spec["dataset_profile"],
                "horizon_years": spec["horizon_years"], "landmark_protocol": spec["landmark_protocol"],
                "censoring": spec["censoring"], "disease_panel": "selected_diseases",
                "aggregation": "paired_patient_bootstrap", "comparison_group": spec["comparison_group"],
                "status": "primary", "comparability": "locked_test_same_landmarks",
                "freeze_manifest": spec["freeze_manifest"],
            }, str(source), f"paired_bootstrap.{contrast}.{metric}")


def _topk_records(spec: Dict[str, Any], source: Path) -> Iterable[Dict[str, Any]]:
    with source.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            for column, raw_value in row.items():
                if not raw_value or not any(token in column for token in ("hit_rate", "recall", "precision", "f1", "capture")):
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                model = row.get("model", "unknown")
                disease = row.get("disease_id", "overall")
                split = row.get(spec.get("split_column", ""), "") or spec.get("split", "unknown")
                dataset_profile = row.get("data_profile", "") or spec.get("dataset_profile", "unknown")
                result_class = spec.get("result_class", "other_medical")
                task = spec.get("task", "disease_retrieval_topk")
                comparison_group = spec.get("comparison_group", "unspecified")
                freeze_manifest = spec.get("freeze_manifest") if split == "test" else None
                yield _record({
                    "record_id": f"topk_{_slug(split)}_{_slug(model)}_{_slug(disease)}_{column}",
                    "result_class": result_class, "task": task,
                    "metric": f"topk.{column}", "value": value, "model": model,
                    "split": split, "dataset_profile": dataset_profile,
                    "horizon_years": "5,10" if "5y" in column or "10y" in column else None,
                    "landmark_protocol": spec.get("landmark_protocol", "final_context"),
                    "censoring": spec.get("censoring", "not_applicable"),
                    "disease_panel": spec.get("disease_panel", "expanded_disease_panel"),
                    "aggregation": "per_disease", "comparison_group": comparison_group,
                    "status": spec.get("status", "exploratory"),
                    "comparability": spec.get("comparability", "within_protocol"),
                    "freeze_manifest": freeze_manifest,
                    "notes": f"reporting_tier={row.get('reporting_tier')}; disease_id={disease}",
                }, str(source), f"line {line_no}; column {column}")


def _shared_topk_records(spec: Dict[str, Any], source: Path) -> Iterable[Dict[str, Any]]:
    with source.open(newline="", encoding="utf-8") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            split = row["split"]
            yield _record({
                "record_id": f"shared_topk_{_slug(split)}_{_slug(row['model'])}_{_slug(row['metric'])}",
                "result_class": "lm_logits", "task": "shared_landmark_next_diagnosis_topk",
                "metric": f"topk.{row['metric']}", "value": row["value"],
                "model": row["model"], "seed": 42, "split": split,
                "dataset_profile": row["profile"], "landmark_protocol": spec["landmark_protocol"],
                "censoring": "not_applicable", "disease_panel": "selected_diseases",
                "aggregation": "shared_target_micro", "comparison_group": spec["comparison_group"],
                "status": spec["status"], "comparability": spec["comparability"],
                "freeze_manifest": spec["freeze_manifest"] if split == "test" else None,
                "notes": f"n_targets={row['n_targets']}; target aligned by code_norm",
            }, str(source), f"line {line_no}")


def _shared_topk_paired_records(spec: Dict[str, Any], source: Path) -> Iterable[Dict[str, Any]]:
    with source.open(newline="", encoding="utf-8") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            split = row["split"]
            base = {
                "result_class": "lm_logits", "task": "shared_landmark_next_diagnosis_topk_paired",
                "model": row["comparison"], "split": split, "dataset_profile": "cross_profile",
                "landmark_protocol": spec["landmark_protocol"], "censoring": "not_applicable",
                "disease_panel": "selected_diseases", "aggregation": "paired_patient_bootstrap",
                "comparison_group": spec["comparison_group"], "status": spec["status"],
                "comparability": spec["comparability"],
                "freeze_manifest": spec["freeze_manifest"] if split == "test" else None,
            }
            yield _record({
                **base, "record_id": f"shared_topk_paired_{_slug(split)}_{_slug(row['comparison'])}_{_slug(row['metric'])}_delta",
                "metric": f"delta_{row['metric']}", "value": row["delta_mean"],
                "lower_ci": row["ci95_low"], "upper_ci": row["ci95_high"],
                "notes": f"p_two_sided={row['p_two_sided']}; p_holm={row['p_holm']}; common_rows={row['common_rows']}; common_patients={row['common_patients']}",
            }, str(source), f"line {line_no}; delta")
            for p_metric in ("p_two_sided", "p_holm"):
                yield _record({
                    **base, "record_id": f"shared_topk_paired_{_slug(split)}_{_slug(row['comparison'])}_{_slug(row['metric'])}_{p_metric}",
                    "metric": f"{row['metric']}.{p_metric}", "value": row[p_metric],
                    "notes": f"common_rows={row['common_rows']}; common_patients={row['common_patients']}",
                }, str(source), f"line {line_no}; {p_metric}")


def build(config_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = [_record(item) for item in config.get("records", [])]
    dispatch = {
        "formal_comparison_json": _formal_records,
        "locked_test_comparison_json": _test_records,
        "topk_csv": _topk_records,
        "shared_topk_csv": _shared_topk_records,
        "shared_topk_paired_csv": _shared_topk_paired_records,
    }
    for spec in config.get("imports", []):
        source = _path(spec["path"])
        if not source.exists():
            raise FileNotFoundError(f"registry source does not exist: {source}")
        rows.extend(dispatch[spec["format"]](spec, source))
    seen = set()
    for row in rows:
        if row["record_id"] in seen:
            raise ValueError(f"duplicate record_id: {row['record_id']}")
        seen.add(row["record_id"])
        for field in ("source_file", "source_locator", "comparison_group", "status", "comparability"):
            if not row.get(field):
                raise ValueError(f"missing {field}: {row['record_id']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result_registry.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    with (output_dir / "result_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_protocol_v1/result_registry_sources.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or _path(config["output_dir"])
    rows = build(args.config, output_dir)
    print(f"wrote {len(rows)} records to {output_dir}")


if __name__ == "__main__":
    main()
