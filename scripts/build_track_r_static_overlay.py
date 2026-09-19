"""Build the Track R static-prefix overlay after the field audit passes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
sys.path.insert(0, str(SRC))

from semantic_delphi_ukb.track_r_contract import bmi_category, load_track_r_protocol, static_token_keys  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--ukb-extract-dir", type=Path, required=True)
    p.add_argument("--source-data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def canonical_raw(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return ""
    return str(int(number)) if number.is_integer() else str(number)


def load_field(path: Path, column: str) -> dict[int, str]:
    values = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values[int(row["eid"])] = canonical_raw(row.get(column, ""))
    return values


def load_eids(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [int(row["eid"]) for row in csv.DictReader(handle)]


def link(source: Path, target: Path) -> None:
    target.symlink_to(source.resolve())


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    if protocol.get("status") != "ready_for_build_after_static_field_audit":
        raise ValueError("protocol status must be ready_for_build_after_static_field_audit before building")
    audit = json.loads(args.audit.read_text(encoding="utf-8-sig"))
    if audit.get("protocol_id") != protocol["protocol_id"] or not audit.get("passed"):
        raise ValueError("a full passing static-field audit is required")
    source = args.source_data_dir or Path(protocol["source_data_dir"])
    output = args.output_dir or Path(protocol["output_data_dir"])
    if output.exists():
        raise FileExistsError(f"Track R output already exists: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Track R staging output already exists: {staging}")

    fields = protocol["static_fields"]
    raw = {}
    for name, spec in fields.items():
        field_id = str(spec["field_id"])
        raw[name] = load_field(args.ukb_extract_dir / f"{field_id}-0.csv", f"{field_id}-{spec['instance']}")
    anchor_spec = protocol["static_prefix"]["age_anchor"]
    anchor_id = str(anchor_spec["field_id"])
    anchor = load_field(args.ukb_extract_dir / f"{anchor_id}-0.csv", f"{anchor_id}-{anchor_spec['instance']}")

    source_manifest = json.loads((source / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    source_embeddings = np.load(source / source_manifest["semantic_output"]).astype(np.float32)
    source_vocab_size = source_embeddings.shape[0]
    all_keys = []
    for name in protocol["static_prefix"]["logical_order"]:
        registered_categories = fields[name].get("categories") or fields[name].get("bins") or sorted(set(fields[name]["coding"].values()))
        for category in registered_categories:
            key = f"static:{name}:{category}"
            if key not in all_keys:
                all_keys.append(key)
    token_id = {key: source_vocab_size + index for index, key in enumerate(all_keys)}
    dynamic_bos_token_id = source_vocab_size + len(all_keys)
    mask_token_id = dynamic_bos_token_id + 1

    staging.mkdir(parents=True)
    try:
        for name in ("train", "val", "test"):
            eids = load_eids(source / f"{name}_patient_index.csv")
            prefix_rows = []
            anchor_rows = []
            fallback_count = anchor_after_first_count = 0
            category_counts = {field: Counter() for field in protocol["static_prefix"]["logical_order"]}
            data = np.memmap(source / f"{name}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
            first_age = {}
            for eid, age_days, _token in data:
                first_age.setdefault(int(eid), float(age_days))
            for eid in eids:
                raw_anchor = anchor.get(eid, "")
                categories = {
                    "sex": fields["sex"]["coding"].get(raw["sex"].get(eid, ""), "missing"),
                    "bmi": bmi_category(raw["bmi"].get(eid, "")),
                    "smoking": fields["smoking"]["coding"].get(raw["smoking"].get(eid, ""), "missing"),
                    "alcohol": fields["alcohol"]["coding"].get(raw["alcohol"].get(eid, ""), "missing"),
                }
                if not raw_anchor:
                    categories.update({"bmi": "missing", "smoking": "missing", "alcohol": "missing"})
                for field, category in categories.items():
                    category_counts[field][category] += 1
                prefix_rows.append([token_id[key] for key in static_token_keys(categories)])
                if raw_anchor:
                    anchor_days = max(1.0, float(raw_anchor) * 365.25)
                else:
                    anchor_days = max(1.0, first_age.get(eid, 1.0))
                    fallback_count += 1
                if anchor_days > first_age.get(eid, anchor_days):
                    anchor_after_first_count += 1
                anchor_rows.append(anchor_days)
            np.save(staging / f"{name}_static_prefix_token_ids.npy", np.asarray(prefix_rows, dtype=np.int64))
            np.save(staging / f"{name}_static_anchor_age_days.npy", np.asarray(anchor_rows, dtype=np.float32))
            for suffix in (".bin", "_patient_index.csv", "_followup_end_age_days.npy"):
                link(source / f"{name}{suffix}", staging / f"{name}{suffix}")
            np.save(staging / f"{name}_static.npy", np.zeros((len(eids), 0), dtype=np.float32))
            protocol.setdefault("build_statistics", {})[name] = {
                "patients": len(eids),
                "anchor_fallback_count": fallback_count,
                "anchor_after_first_dynamic_event_count": anchor_after_first_count,
                "static_category_counts": {
                    field: dict(sorted(counts.items())) for field, counts in category_counts.items()
                },
            }
        link(source / "vocab", staging / "vocab")
        link(source / "labels.csv", staging / "labels.csv")
        extended_embeddings = np.vstack((source_embeddings, np.zeros((len(all_keys) + 2, source_embeddings.shape[1]), dtype=np.float32)))
        np.save(staging / "semantic_input_embeddings_64d.npy", extended_embeddings)
        manifest = {
            **source_manifest,
            "protocol_id": protocol["protocol_id"],
            "source_dynamic_data_dir": str(source.resolve()),
            "source_dynamic_manifest": str((source / "prepare_manifest.json").resolve()),
            "semantic_output": "semantic_input_embeddings_64d.npy",
            "vocab_size": int(extended_embeddings.shape[0]),
            "static_feature_order": [],
            "static_prefix": protocol["static_prefix"],
            "static_token_ids": token_id,
            "dynamic_bos_token_id": dynamic_bos_token_id,
            "dynamic_bos": protocol["dynamic_bos"],
            "mask_token_id": mask_token_id,
            "dynamic_vocab_size": source_vocab_size,
            "static_fields": fields,
            "redundancy_policy": protocol["redundancy_policy"],
            "loss_contract": protocol["loss_contract"],
            "build_statistics": protocol["build_statistics"],
            "field_audit": str(args.audit.resolve()),
            "field_audit_manual_review": audit.get("manual_review_checklist"),
            "bmi_below_15_policy": fields["bmi"]["below_15_policy"],
            "bmi_privacy_truncation_note": fields["bmi"]["privacy_truncation_note"],
            "bmi_audit_diagnostics": audit.get("fields", {}).get("bmi", {}).get("continuous_diagnostics"),
        }
        (staging / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "vocab_size": manifest["vocab_size"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
