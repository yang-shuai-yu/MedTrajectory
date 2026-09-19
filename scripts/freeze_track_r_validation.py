"""Freeze Track R only after validation rows are complete and exactly paired."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.track_r_rows import read_json  # noqa: E402


EXPECTED_MODELS = (
    "A0-TokenStatic",
    "A0-noStatic",
    "A1-TokenStatic",
    "A1-noStatic",
    "Logistic-R",
    "Cox-R",
    "MDRMF-Clinical-R",
    "Med-BERT-Paper",
    "Med-BERT-Matched-S",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--data-manifest", type=Path, required=True)
    p.add_argument("--field-audit", type=Path, required=True)
    p.add_argument("--capacity-report", type=Path, required=True)
    p.add_argument("--validation-landmarks", type=Path, required=True)
    p.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT=VALIDATION_ROWS; repeat for all models")
    p.add_argument("--output", type=Path, required=True)
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key(row: dict) -> tuple:
    return (
        int(row["patient_index"]),
        str(row["disease_id"]),
        float(row["horizon_years"]),
        str(row["sex"]),
        float(row["age_start_years"]),
    )


def key_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for key in sorted(row_key(row) for row in rows):
        digest.update((json.dumps(key, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def label_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=row_key):
        digest.update((json.dumps((row_key(row), int(row["label"])), separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8-sig"))
    audit = json.loads(args.field_audit.read_text(encoding="utf-8-sig"))
    capacity = json.loads(args.capacity_report.read_text(encoding="utf-8-sig"))
    landmarks = json.loads(args.validation_landmarks.read_text(encoding="utf-8-sig"))
    if protocol.get("protocol_id") != "track_r_v2_1" or not audit.get("passed"):
        raise ValueError("protocol and static-field audit are not freeze-ready")
    if data_manifest.get("dynamic_bos") != protocol.get("dynamic_bos"):
        raise ValueError("data manifest dynamic BOS contract does not match protocol")
    if not capacity.get("matched_block_passed"):
        raise ValueError("Med-BERT-Matched-S failed the registered block-capacity tolerance")
    if landmarks.get("split") != "val" or any("position" not in item for item in landmarks.get("landmarks", [])):
        raise ValueError("validation landmarks must be fixed-position val landmarks")
    models = {}
    hashes = set()
    label_hashes = set()
    names = []
    for spec in args.model:
        name, checkpoint_raw, rows_raw = spec.split("=", 2)
        checkpoint, rows_path = Path(checkpoint_raw), Path(rows_raw)
        rows = read_json(rows_path)
        current_hash = key_hash(rows)
        current_label_hash = label_hash(rows)
        hashes.add(current_hash)
        label_hashes.add(current_label_hash)
        names.append(name)
        models[name] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "validation_rows": str(rows_path.resolve()),
            "validation_rows_sha256": sha256(rows_path),
            "paired_row_key_sha256": current_hash,
            "paired_label_sha256": current_label_hash,
            "row_count": len(rows),
        }
    if tuple(names) != EXPECTED_MODELS:
        raise ValueError(f"models must be supplied in registered order: {EXPECTED_MODELS}")
    if len(hashes) != 1 or len(label_hashes) != 1:
        raise ValueError("validation prediction rows or labels are not exactly paired across all models")
    payload = {
        "manifest_version": "track_r_validation_freeze_v2",
        "protocol_id": protocol["protocol_id"],
        "test_result_class": "exploratory",
        "test_used_for_selection": False,
        "validation_reporting_policy": protocol["validation_reporting_policy"],
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256(args.protocol)},
        "data_manifest": {"path": str(args.data_manifest.resolve()), "sha256": sha256(args.data_manifest)},
        "field_audit": {"path": str(args.field_audit.resolve()), "sha256": sha256(args.field_audit)},
        "capacity_report": {"path": str(args.capacity_report.resolve()), "sha256": sha256(args.capacity_report)},
        "validation_landmarks": {"path": str(args.validation_landmarks.resolve()), "sha256": sha256(args.validation_landmarks)},
        "paired_row_key_sha256": next(iter(hashes)),
        "paired_label_sha256": next(iter(label_hashes)),
        "models": models,
        "loss_contract": protocol["loss_contract"],
        "dynamic_bos_contract": protocol["dynamic_bos"],
        "dynamic_bos_token_id": int(data_manifest["dynamic_bos_token_id"]),
        "sequence_lengths": protocol["sequence_lengths"],
        "static_anchor_contract": protocol["static_prefix"]["age_anchor"],
        "redundancy_policy": protocol["redundancy_policy"],
    }
    if args.output.exists():
        raise FileExistsError(f"freeze manifest already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": len(models), "row_key_sha256": payload["paired_row_key_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
