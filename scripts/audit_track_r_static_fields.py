"""Audit Track R baseline fields without writing model-ready data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--ukb-extract-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=0, help="audit all rows by default")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_raw(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else str(number)


def audit_field(
    path: Path,
    field_id: str,
    instance: str,
    coding: dict[str, str],
    max_rows: int,
    *,
    coding_is_exhaustive: bool = True,
    continuous_diagnostics: bool = False,
) -> dict:
    expected_column = f"{field_id}-{instance}"
    counts: Counter[str] = Counter()
    rows = nonmissing = 0
    nonnumeric: Counter[str] = Counter()
    nonpositive = below_15 = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "eid" not in (reader.fieldnames or []) or expected_column not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain eid and {expected_column}")
        for row in reader:
            rows += 1
            raw = canonical_raw(row.get(expected_column, ""))
            if raw:
                nonmissing += 1
                counts[raw] += 1
                if continuous_diagnostics:
                    try:
                        numeric = float(raw)
                    except ValueError:
                        nonnumeric[raw] += 1
                    else:
                        if numeric <= 0:
                            nonpositive += 1
                        elif numeric < 15:
                            below_15 += 1
            if max_rows and rows >= max_rows:
                break
    unexpected = sorted(value for value in counts if coding_is_exhaustive and coding and value not in coding)
    result = {
        "field_id": int(field_id),
        "instance": instance,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": rows,
        "nonmissing": nonmissing,
        "coverage": nonmissing / rows if rows else 0.0,
        "observed_coding_counts": dict(sorted(counts.items())),
        "registered_coding": coding,
        "coding_is_exhaustive": coding_is_exhaustive,
        "unexpected_codes": unexpected,
        "passed": bool(rows and nonmissing and not unexpected),
    }
    if continuous_diagnostics:
        result["continuous_diagnostics"] = {
            "nonnumeric_counts": dict(sorted(nonnumeric.items())),
            "nonpositive_sentinel_count": nonpositive,
            "positive_below_15_count": below_15,
            "below_15_policy": "retain_in_WHO_lt18.5_bin_and_report_as_privacy_truncation_diagnostic",
            "manual_review_required": True,
        }
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    results = {}
    for name, spec in protocol["static_fields"].items():
        field_id = str(spec["field_id"])
        path = args.ukb_extract_dir / f"{field_id}-0.csv"
        if not path.is_file():
            results[name] = {
                "field_id": int(field_id),
                "path": str(path),
                "passed": False,
                "error": "source_file_missing",
            }
            continue
        results[name] = audit_field(
            path,
            field_id,
            spec["instance"],
            spec.get("coding", {}),
            args.max_rows,
            coding_is_exhaustive=spec.get("coding_is_exhaustive", True),
            continuous_diagnostics=spec.get("source_value_type") == "continuous",
        )
    anchor_id = str(protocol["static_prefix"]["age_anchor"]["field_id"])
    anchor_path = args.ukb_extract_dir / f"{anchor_id}-0.csv"
    if anchor_path.is_file():
        results["static_anchor_age"] = audit_field(
            anchor_path, anchor_id, protocol["static_prefix"]["age_anchor"]["instance"], {}, args.max_rows
        )
        results["static_anchor_age"]["passed"] = results["static_anchor_age"]["nonmissing"] > 0
        results["static_anchor_age"].pop("unexpected_codes", None)
    else:
        results["static_anchor_age"] = {"field_id": int(anchor_id), "path": str(anchor_path), "passed": False, "error": "source_file_missing"}
    payload = {
        "protocol_id": protocol["protocol_id"],
        "audit_only": True,
        "manual_review_checklist": {
            "protocol_status_must_remain_draft_until_manual_review": True,
            "confirm_smoking_20116_observed_codes_match_registered_coding": True,
            "confirm_alcohol_1558_observed_codes_match_registered_seven_categories": True,
            "confirm_alcohol_codes_2_and_3_are_not_merged": True,
            "review_bmi_observed_coding_counts_even_when_unexpected_codes_is_empty": True,
            "review_bmi_nonnumeric_nonpositive_and_positive_below_15_diagnostics": True,
            "manual_status_transition_after_pass_only": "ready_for_build_after_static_field_audit",
            "automatic_protocol_status_transition_forbidden": True
        },
        "max_rows": args.max_rows,
        "fields": results,
        "passed": all(item.get("passed", False) for item in results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": payload["passed"]}))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
