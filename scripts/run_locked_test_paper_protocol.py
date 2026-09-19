"""Run the one-shot locked-test paired comparison after all gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.compare_horizon_control_tasks import (  # noqa: E402
    _load_compact_inputs,
    main as compare_main,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-manifest", type=Path, required=True)
    p.add_argument("--freeze-verification", type=Path, required=True)
    p.add_argument("--split-audit", type=Path, required=True)
    p.add_argument("--consistency", action="append", required=True, help="validation consistency JSON; repeat per required model")
    p.add_argument("--require-consistency", action="append", default=[], help="consistency names that must pass")
    p.add_argument("--input", action="append", required=True, help="name=rows.json; repeat for models")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--landmark-manifest", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--memory-efficient", action="store_true")
    p.add_argument("--allow-exploratory-gate-override", action="store_true")
    return p


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pairing_audit(inputs):
    """Check that paired rows share landmarks and censor-aware labels."""
    indexed = {}
    for name, raw_path in inputs:
        rows = _read(Path(raw_path))
        table = {}
        duplicate_keys = 0
        for row in rows:
            age = row.get("age_start_years")
            item = (
                int(row["patient_index"]),
                str(row["disease_id"]),
                float(row["horizon_years"]),
                str(row.get("sex", "")),
                None if age is None else float(age),
            )
            if item in table:
                duplicate_keys += 1
            table[item] = row
        indexed[name] = table
        indexed[name]["__duplicate_count__"] = duplicate_keys
    duplicate_counts = {name: int(table.pop("__duplicate_count__")) for name, table in indexed.items()}
    audit = {"models": {name: {"rows": len(table), "duplicate_keys": duplicate_counts[name]} for name, table in indexed.items()}, "comparisons": {}}
    names = list(indexed)
    for left_name, right_name in ((names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))):
        left, right = indexed[left_name], indexed[right_name]
        common = sorted(set(left) & set(right))
        label_mismatch = sum(int(left[item]["label"]) != int(right[item]["label"]) for item in common)
        age_mismatch = sum(float(left[item].get("prediction_age_days", float("nan"))) != float(right[item].get("prediction_age_days", float("nan")) ) for item in common)
        audit["comparisons"][f"{left_name} - {right_name}"] = {
            "common_rows": len(common),
            "left_only_rows": len(set(left) - set(right)),
            "right_only_rows": len(set(right) - set(left)),
            "label_mismatch_rows": label_mismatch,
            "prediction_age_mismatch_rows": age_mismatch,
        }
    return audit


def _pairing_audit_compact(inputs):
    specs = [f"{name}={path}" for name, path in inputs]
    compact = _load_compact_inputs(specs)
    rows = int(len(compact["patient_ids"]))
    names = list(compact["names"])
    return {
        "models": {name: {"rows": rows, "duplicate_keys": 0} for name in names},
        "comparisons": {
            f"{left} - {right}": {
                "common_rows": rows,
                "left_only_rows": 0,
                "right_only_rows": 0,
                "label_mismatch_rows": 0,
                "prediction_age_mismatch_rows": 0,
            }
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        },
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protected_outputs = (
        "locked_test_gate.json", "pairing_audit.json", "comparison.json",
        "comparison.md", "calibration_bins.csv",
    )
    existing = [args.out_dir / name for name in protected_outputs if (args.out_dir / name).exists()]
    if existing:
        raise FileExistsError(f"locked-test outputs already exist: {existing}")
    freeze = _read(args.freeze_manifest)
    verification = _read(args.freeze_verification)
    split_audit = _read(args.split_audit)
    gate_errors = []
    if not verification.get("ok"):
        gate_errors.append("freeze_verification_not_ok")
    if verification.get("freeze_manifest_sha256") != hashlib.sha256(args.freeze_manifest.read_bytes()).hexdigest():
        gate_errors.append("freeze_verification_manifest_hash_mismatch")
    if freeze.get("manifest_version") != "paper_medical_control_locked_test_v2":
        gate_errors.append("freeze_manifest_version_missing_or_invalid")
    if freeze.get("ok") is False:
        gate_errors.append("freeze_manifest_not_ok")
    effect_gate_record = freeze.get("frozen_files", {}).get("validation_effect_gate", {})
    if effect_gate_record.get("path"):
        effect_gate = _read(Path(effect_gate_record["path"]))
        if not bool(effect_gate.get("ok")):
            if not args.allow_exploratory_gate_override:
                gate_errors.append("validation_effect_gate_not_ok")
            elif freeze.get("run_class") != "exploratory_gate_override":
                gate_errors.append("exploratory_gate_override_not_frozen")
            elif not bool(freeze.get("exploratory_override", {}).get("authorization", {}).get("authorized")):
                gate_errors.append("exploratory_gate_override_not_authorized")
    if not bool(split_audit.get("disjoint", {}).get("ok")):
        gate_errors.append("split_audit_not_ok")
    if not bool(split_audit.get("expected_test_eids", {}).get("ok")):
        gate_errors.append("split_audit_test_eids_not_verified")
    if freeze.get("split_audit_sha256") != hashlib.sha256(args.split_audit.read_bytes()).hexdigest():
        gate_errors.append("split_audit_hash_mismatch")
    if "test" not in split_audit.get("splits", {}):
        gate_errors.append("test_split_not_audited")
    consistency_payloads = {}
    consistency_paths = {}
    for raw_path in args.consistency:
        path = Path(raw_path)
        if not path.exists():
            gate_errors.append(f"missing_consistency:{path}")
            continue
        if path.stem in consistency_payloads:
            gate_errors.append(f"duplicate_consistency_name:{path.stem}")
            continue
        consistency_payloads[path.stem] = _read(path)
        consistency_paths[path.stem] = path
        if not bool(consistency_payloads[path.stem].get("ok")):
            gate_errors.append(f"validation_consistency_not_ok:{path.stem}")
    missing_consistency = sorted(set(args.require_consistency) - set(consistency_payloads))
    gate_errors.extend(f"missing_required_consistency:{name}" for name in missing_consistency)
    if "test" not in freeze.get("splits", {}):
        gate_errors.append("freeze_manifest_has_no_test")
    test_count = freeze.get("splits", {}).get("test", {}).get("patient_count")
    declared_test_count = freeze.get("test_eids", {}).get("patient_count")
    if test_count is not None and declared_test_count is not None and int(test_count) != int(declared_test_count):
        gate_errors.append("freeze_test_count_mismatch")
    if not freeze.get("checkpoints"):
        gate_errors.append("freeze_manifest_has_no_checkpoints")
    frozen_checkpoint_paths = {
        str(Path(value.get("path", "")).resolve())
        for value in freeze.get("checkpoints", {}).values()
        if value.get("path")
    }
    frozen_model_data_dirs = {
        name: str(value.get("data_dir", ""))
        for name, value in freeze.get("model_data_dirs", {}).items()
    }
    frozen_consistency = freeze.get("frozen_files", {}).get("validation_consistency", {})
    if len(consistency_payloads) != 1:
        gate_errors.append("exactly_one_validation_consistency_required")
    for name, payload in consistency_payloads.items():
        path = consistency_paths[name]
        if str(path.resolve()) != str(Path(frozen_consistency.get("path", "")).resolve()):
            gate_errors.append("validation_consistency_not_frozen")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != frozen_consistency.get("sha256"):
            gate_errors.append("validation_consistency_hash_mismatch")
        binding = payload.get("protocol_binding", {})
        protocol = freeze.get("protocol", {})
        if not binding.get("ok"):
            gate_errors.append("validation_consistency_protocol_binding_not_ok")
        if binding.get("protocol_json", {}).get("sha256") != freeze.get("protocol_source", {}).get("sha256"):
            gate_errors.append("validation_consistency_protocol_hash_mismatch")
        if [float(value) for value in binding.get("horizons_years", [])] != [float(value) for value in protocol.get("horizons_years", [])]:
            gate_errors.append("validation_consistency_horizons_mismatch")
        if [float(value) for value in binding.get("age_groups_years", [])] != [float(value) for value in protocol.get("age_groups_years", [])]:
            gate_errors.append("validation_consistency_age_groups_mismatch")
        if binding.get("medical_protocol") != protocol.get("medical_protocol"):
            gate_errors.append("validation_consistency_medical_protocol_mismatch")
        if bool(binding.get("sex_stratified")) != bool(protocol.get("sex_stratified")):
            gate_errors.append("validation_consistency_sex_stratification_mismatch")
        if binding.get("case_control") != protocol.get("case_control"):
            gate_errors.append("validation_consistency_case_control_mismatch")
        if float(binding.get("age_bin_width_years", 0.0)) != float(protocol.get("age_bin_width_years", 0.0)):
            gate_errors.append("validation_consistency_age_bin_width_mismatch")
        controls = binding.get("controls", {})
        if set(controls) != set(freeze.get("checkpoints", {})):
            gate_errors.append("validation_consistency_controls_mismatch")
        for model_name, source in controls.items():
            frozen_checkpoint = freeze.get("checkpoints", {}).get(model_name, {}).get("path", "")
            if str(Path(source.get("checkpoint", "")).resolve()) != str(Path(frozen_checkpoint).resolve()):
                gate_errors.append(f"validation_consistency_checkpoint_mismatch:{model_name}")
            frozen_data_dir = frozen_model_data_dirs.get(model_name, "")
            if str(Path(source.get("data_dir", "")).resolve()) != str(Path(frozen_data_dir).resolve()):
                gate_errors.append(f"validation_consistency_data_dir_mismatch:{model_name}")
        frozen_val_landmarks = freeze.get("frozen_files", {}).get("shared_validation_landmarks", {}).get("path", "")
        if str(Path(binding.get("shared_validation_landmark_manifest", "")).resolve()) != str(Path(frozen_val_landmarks).resolve()):
            gate_errors.append("validation_consistency_landmark_mismatch")
        official = binding.get("official_aggregates", {})
        frozen_official = freeze.get("frozen_files", {}).get("validation_official_aggregates", {})
        if str(Path(official.get("path", "")).resolve()) != str(Path(frozen_official.get("path", "")).resolve()) or official.get("sha256") != frozen_official.get("sha256"):
            gate_errors.append("validation_consistency_official_aggregates_mismatch")
    if not bool(freeze.get("protocol", {}).get("checkpoint_training_excludes_test", False)):
        gate_errors.append("checkpoint_training_test_leakage_not_proven")
    if args.landmark_manifest is not None:
        if not args.landmark_manifest.exists():
            gate_errors.append(f"missing_landmark_manifest:{args.landmark_manifest}")
        frozen_landmark = freeze.get("frozen_files", {}).get("shared_test_landmarks", {})
        if str(args.landmark_manifest.resolve()) != str(Path(frozen_landmark.get("path", "")).resolve()):
            gate_errors.append("landmark_manifest_not_frozen")
        elif frozen_landmark.get("sha256"):
            digest = hashlib.sha256(args.landmark_manifest.read_bytes()).hexdigest()
            if digest != frozen_landmark["sha256"]:
                gate_errors.append("landmark_manifest_hash_mismatch")
    input_specs = []
    for spec in args.input:
        _name, raw_path = spec.split("=", 1)
        input_specs.append((_name, raw_path))
        summary_path = Path(raw_path).with_name("summary.json")
        if not summary_path.exists():
            gate_errors.append(f"missing_summary:{summary_path}")
            continue
        summary = _read(summary_path)
        if summary.get("split") != "test":
            gate_errors.append(f"input_not_test:{raw_path}")
        if summary.get("protocol") != "paper_medical_control_v1":
            gate_errors.append(f"input_protocol_mismatch:{raw_path}")
        checkpoint_path = summary.get("checkpoint")
        if not checkpoint_path or str(Path(checkpoint_path).resolve()) not in frozen_checkpoint_paths:
            gate_errors.append(f"input_checkpoint_not_frozen:{raw_path}")
        if _name in frozen_model_data_dirs:
            input_data_dir = summary.get("data_dir")
            if not input_data_dir or str(Path(input_data_dir).resolve()) != str(Path(frozen_model_data_dirs[_name]).resolve()):
                gate_errors.append(f"input_data_dir_not_frozen:{raw_path}")
        if not Path(raw_path).exists():
            gate_errors.append(f"missing_rows:{raw_path}")
        if args.landmark_manifest is not None:
            expected_landmark = str(args.landmark_manifest.resolve())
            observed_landmark = str(summary.get("protocol_signature", {}).get("shared_landmark_manifest", ""))
            if observed_landmark != expected_landmark:
                gate_errors.append(f"input_landmark_manifest_mismatch:{raw_path}")
    if not gate_errors:
        try:
            pairing_audit = _pairing_audit_compact(input_specs) if args.memory_efficient else _pairing_audit(input_specs)
        except ValueError as exc:
            gate_errors.append(f"pairing_audit_failed:{exc}")
            pairing_audit = None
        if pairing_audit is not None:
            (args.out_dir / "pairing_audit.json").write_text(json.dumps(pairing_audit, indent=2), encoding="utf-8")
    if not gate_errors and pairing_audit is not None:
        for name, values in pairing_audit["models"].items():
            if values["duplicate_keys"]:
                gate_errors.append(f"duplicate_paired_keys:{name}:{values['duplicate_keys']}")
        for comparison, values in pairing_audit["comparisons"].items():
            if values["left_only_rows"] or values["right_only_rows"]:
                gate_errors.append(
                    f"paired_row_set_mismatch:{comparison}:{values['left_only_rows']}:{values['right_only_rows']}"
                )
            if values["label_mismatch_rows"]:
                gate_errors.append(f"paired_label_mismatch:{comparison}:{values['label_mismatch_rows']}")
            if values["prediction_age_mismatch_rows"]:
                gate_errors.append(f"paired_prediction_age_mismatch:{comparison}:{values['prediction_age_mismatch_rows']}")
    gate_payload = {"protocol": "paper_medical_control_v1", "split": "test", "gate_errors": gate_errors, "ok": not gate_errors, "bootstrap": args.bootstrap, "seed": args.seed, "consistency_files": args.consistency, "freeze_verification": str(args.freeze_verification.resolve())}
    (args.out_dir / "locked_test_gate.json").write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
    if gate_errors:
        print(json.dumps(gate_payload, indent=2))
        return 2
    compare_args = []
    for spec in args.input:
        compare_args.extend(["--input", spec])
    compare_args.extend(["--out-dir", str(args.out_dir), "--bootstrap", str(args.bootstrap), "--seed", str(args.seed)])
    if args.memory_efficient:
        compare_args.append("--memory-efficient")
    compare_main(compare_args)
    print(json.dumps({"ok": True, "comparison": str(args.out_dir / "comparison.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
