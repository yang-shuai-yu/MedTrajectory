"""Build a deterministic, model-independent Track G evaluation cohort manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.evaluate_track_g_generation import eligible_cases, load_split  # noqa: E402
from semantic_delphi_ukb.selected_disease_demo import load_labels  # noqa: E402
from semantic_delphi_ukb.track_g_contract import assert_split_allowed, load_track_g_protocol  # noqa: E402
from semantic_delphi_ukb.track_r_batch import validate_track_r_data_manifest  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=REPO_DIR / "configs/track_g_v1/TRACK_G_v1.json")
    value.add_argument("--data-dir", type=Path, default=None)
    value.add_argument("--split", choices=("val", "test"), default="val")
    value.add_argument("--max-patients", type=int, default=None)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_g_protocol(args.protocol, REPO_DIR)
    assert_split_allowed(protocol, args.split)
    source = protocol["_source_track_r"]
    data_dir = args.data_dir or Path(source["output_data_dir"])
    validate_track_r_data_manifest(data_dir, source)
    labels = load_labels(data_dir / "labels.csv")
    data, p2i, _ = load_split(data_dir, args.split)
    settings = dict(protocol["generation_evaluation"])
    settings["dynamic_context_length"] = int(protocol["matched_input_contract"]["dynamic_context_length"])
    evaluation = settings["validation" if args.split == "val" else "locked_test"]
    max_patients = int(args.max_patients if args.max_patients is not None else evaluation["max_patients"])
    cases = eligible_cases(data, p2i, labels, settings, max_patients, int(settings["sampling_seed"]))
    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "split": args.split,
        "selection_seed": settings["sampling_seed"],
        "cases": [{"patient_index": patient, "cut_index": cut} for patient, cut in cases],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "patient_count": len(cases), "split": args.split}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
