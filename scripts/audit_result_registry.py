#!/usr/bin/env python3
"""Audit result registry rows for protocol/category mixing and traceability."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "record_id", "result_class", "task", "metric", "value", "split", "dataset_profile",
    "comparison_group", "source_file", "source_locator", "status", "comparability",
)
CLASSES = {"lm_logits", "horizon_head", "other_medical", "literature_reference"}


def _load(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    ids = set()
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    exploratory_test_groups_without_freeze = set()
    for row in rows:
        record_id = row.get("record_id")
        if record_id in ids:
            findings.append({"severity": "error", "rule": "duplicate_record_id", "record_id": record_id})
        ids.add(record_id)
        missing = [field for field in REQUIRED if row.get(field) in (None, "")]
        if missing:
            findings.append({"severity": "error", "rule": "required_field_missing", "record_id": record_id, "fields": missing})
        if row.get("result_class") not in CLASSES:
            findings.append({"severity": "error", "rule": "invalid_result_class", "record_id": record_id, "value": row.get("result_class")})
        groups[row.get("comparison_group", "")].append(row)

        task = row.get("task", "")
        if task == "fixed_context_control_auc" and row.get("status") == "primary":
            findings.append({"severity": "error", "rule": "fixed_context_control_as_primary", "record_id": record_id})
        if row.get("dataset_profile") == "legacy" or "legacy_non_censoring_protocol" in row.get("comparability", ""):
            if "legacy_non_censoring_protocol" not in row.get("comparability", ""):
                findings.append({"severity": "error", "rule": "legacy_missing_protocol_label", "record_id": record_id})
            if row.get("status") == "primary":
                findings.append({"severity": "error", "rule": "legacy_primary_result", "record_id": record_id})
        if row.get("result_class") == "literature_reference":
            if row.get("split") != "reference" or "external_reference" not in row.get("comparability", ""):
                findings.append({"severity": "error", "rule": "literature_not_isolated", "record_id": record_id})
        if row.get("split") == "test" and not row.get("freeze_manifest"):
            if row.get("status") == "primary":
                findings.append({"severity": "error", "rule": "test_missing_freeze_manifest", "record_id": record_id})
            else:
                exploratory_test_groups_without_freeze.add(row.get("comparison_group", ""))
        if row.get("status") in {"superseded", "diagnostic"} and row.get("status") == "primary":
            findings.append({"severity": "error", "rule": "non_primary_status_conflict", "record_id": record_id})

    for group, group_rows in groups.items():
        classes = sorted({row.get("result_class") for row in group_rows})
        if len(classes) > 1:
            findings.append({
                "severity": "error", "rule": "cross_category_table_mix", "comparison_group": group,
                "result_classes": classes, "record_ids": [row.get("record_id") for row in group_rows],
                "message": "LM logits, horizon head, other medical tasks, and literature references require separate tables.",
            })
    for group in sorted(exploratory_test_groups_without_freeze):
        findings.append({
            "severity": "warning", "rule": "exploratory_test_missing_freeze_manifest",
            "comparison_group": group,
            "message": "Exploratory test rows are archived but are not eligible for locked-test primary claims.",
        })

    # A metric namespace is allowed to repeat across categories only when the
    # comparison groups are different; this catches accidental broad tables.
    metric_split_groups: Dict[tuple, set] = defaultdict(set)
    for row in rows:
        metric_split_groups[(row.get("split"), row.get("metric"))].add(row.get("comparison_group"))
    namespace_collisions = [
        {"split": split, "metric": metric, "comparison_groups": sorted(groupset)}
        for (split, metric), groupset in metric_split_groups.items() if len(groupset) > 1
    ]

    errors = sum(item["severity"] == "error" for item in findings)
    warnings = sum(item["severity"] == "warning" for item in findings)
    return {
        "ok": errors == 0,
        "record_count": len(rows),
        "class_counts": {name: sum(row.get("result_class") == name for row in rows) for name in sorted(CLASSES)},
        "status_counts": {name: sum(row.get("status") == name for row in rows) for name in sorted({row.get("status") for row in rows})},
        "errors": errors,
        "warnings": warnings,
        "findings": findings,
        "metric_namespace_repeats_across_separate_groups": namespace_collisions,
    }


def write_summary(audit_result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Result Registry Audit",
        "",
        f"- OK: `{audit_result['ok']}`",
        f"- Records: `{audit_result['record_count']}`",
        f"- Errors: `{audit_result['errors']}`",
        f"- Warnings: `{audit_result['warnings']}`",
        "",
        "## Category Counts",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in audit_result["class_counts"].items())
    lines.extend(["", "## Findings", ""])
    if not audit_result["findings"]:
        lines.append("No findings.")
    else:
        for finding in audit_result["findings"]:
            lines.append(f"- **{finding['severity']}** `{finding['rule']}`: `{finding.get('record_id', finding.get('comparison_group', 'group'))}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "results/paper_protocol_v1/result_registry/result_registry.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strict", action="store_true", help="return exit code 2 when errors are found")
    args = parser.parse_args()
    result = audit(_load(args.registry))
    output_dir = args.output_dir or args.registry.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cross_category_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(result, output_dir / "registry_summary.md")
    print(json.dumps({key: result[key] for key in ("ok", "record_count", "errors", "warnings", "class_counts")}, ensure_ascii=False))
    if args.strict and not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
