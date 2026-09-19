"""Build the immutable input manifest for the one-shot locked-test run.

This command only records hashes and declarations.  It never evaluates a
checkpoint and it requires an explicit human confirmation that training did not
read the locked test patients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST_VERSION = "paper_medical_control_locked_test_v2"
REQUIRED_ASSETS = {
    "medical_control_evaluator",
    "comparison_evaluator",
    "validation_consistency",
    "validation_official_aggregates",
    "vocab",
    "shared_validation_landmarks",
    "shared_test_landmarks",
}
REQUIRED_PROTOCOL_FIELDS = {
    "age_groups_years",
    "age_bin_width_years",
    "bootstrap",
    "bootstrap_seed",
    "calibration_bins",
    "checkpoint_selection_rule",
    "data_protocol_id",
    "horizons_years",
    "medical_protocol",
    "case_control",
    "sex_stratified",
}
REQUIRED_CAROPE_SOURCE_FILES = {
    "src/semantic_delphi_ukb/car_rope_model.py",
    "src/semantic_delphi_ukb/train_car_rope.py",
    "src/semantic_delphi_ukb/train_car_rope_pretraining.py",
    "src/semantic_delphi_ukb/multitype_batch.py",
    "src/semantic_delphi_ukb/train_architecture_risk_heads.py",
    "src/semantic_delphi_ukb/tte_targets.py",
    "scripts/training_monitor.py",
}

from semantic_delphi_ukb.paper_protocol_audit import (  # noqa: E402
    build_freeze_manifest,
    eid_set_sha256,
    read_eid_csv,
    read_split_eids,
    sha256_file,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-audit", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--diseases-yaml", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="name=/path/to/checkpoint")
    p.add_argument("--asset", action="append", default=[], help="name=/path/to/frozen asset")
    p.add_argument("--require-asset", action="append", default=[], help="asset name that must be present")
    p.add_argument("--model-data-dir", action="append", default=[], help="model_name=/path/to/profile; repeatable")
    p.add_argument("--model-split-audit", action="append", default=[], help="model_name=/path/to/split_audit.json; repeatable")
    p.add_argument("--protocol-json", type=Path, required=True)
    p.add_argument("--test-eids-csv", type=Path, required=True)
    p.add_argument("--confirm-checkpoint-excludes-test", action="store_true")
    p.add_argument(
        "--exploratory-override",
        type=Path,
        default=None,
        help="Explicit authorization JSON for proceeding when validation_effect_gate.ok is false",
    )
    p.add_argument("--out", type=Path, required=True)
    return p


def _specs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"invalid name=path specification: {value!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if name in result:
            raise ValueError(f"duplicate frozen name: {name}")
        result[name] = path
    return result


def _dir_specs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"invalid name=directory specification: {value!r}")
        path = Path(raw_path)
        if not path.is_dir():
            raise NotADirectoryError(path)
        if name in result:
            raise ValueError(f"duplicate frozen name: {name}")
        result[name] = path
    return result


def _eid_sequence_hash(data_dir: Path, split: str = "test") -> str:
    eids = read_split_eids(data_dir / f"{split}_patient_index.csv")
    digest = hashlib.sha256()
    digest.update(("\n".join(str(eid) for eid in eids) + "\n").encode("ascii"))
    return digest.hexdigest()


def _source_manifest_evidence(path: Path) -> tuple[dict, list[str]]:
    if not path.is_file():
        return {"path": str(path.resolve())}, ["source_manifest_missing"]
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    files = {
        str(item.get("path")).replace("\\", "/"): str(item.get("sha256"))
        for item in payload.get("files", [])
    }
    errors = []
    if payload.get("manifest_version") != "source_snapshot_v2":
        errors.append("source_manifest_version_mismatch")
    for relative in sorted(REQUIRED_CAROPE_SOURCE_FILES):
        source = ROOT / relative
        if relative not in files:
            errors.append(f"source_file_not_recorded:{relative}")
        elif not source.is_file() or sha256_file(source) != files[relative]:
            errors.append(f"source_file_hash_mismatch:{relative}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "manifest_version": payload.get("manifest_version"),
        "files": files,
    }, errors


def _training_evidence(
    checkpoint: Path,
    data_dir: Path,
    expected_eid_set_sha256: str,
    expected_data_protocol_id: str,
) -> tuple[dict, list[str]]:
    errors = []
    resolved_config = checkpoint.parent.parent / "resolved_config.json"
    data_manifest = data_dir / "prepare_manifest.json"
    evidence = {
        "resolved_config": str(resolved_config.resolve()),
        "data_manifest": str(data_manifest.resolve()),
    }
    if not resolved_config.is_file():
        errors.append("resolved_config_missing")
    if not data_manifest.is_file():
        errors.append("training_data_manifest_missing")
    if errors:
        return evidence, errors
    resolved = json.loads(resolved_config.read_text(encoding="utf-8-sig"))
    run_source, source_errors = _source_manifest_evidence(checkpoint.parent.parent / "source_manifest.json")
    errors.extend(source_errors)
    configured_data_dir = Path(str(resolved.get("config", {}).get("data_dir", "")))
    if not configured_data_dir.is_absolute():
        configured_data_dir = (ROOT / configured_data_dir).resolve()
    if configured_data_dir != data_dir.resolve():
        errors.append("resolved_config_data_dir_mismatch")
    data_payload = json.loads(data_manifest.read_text(encoding="utf-8-sig"))
    locked_source = data_payload.get("locked_test_source", {})
    observed_set_hash = locked_source.get("eid_set_sha256")
    if observed_set_hash != expected_eid_set_sha256:
        errors.append("training_data_locked_test_eid_set_mismatch")
    if data_payload.get("protocol_id") != expected_data_protocol_id:
        errors.append("training_data_protocol_id_mismatch")
    if int(data_payload.get("split", {}).get("test_participants", 0)) <= 0:
        errors.append("training_data_manifest_has_no_test_partition")
    evidence.update({
        "resolved_config_sha256": sha256_file(resolved_config),
        "data_manifest_sha256": sha256_file(data_manifest),
        "data_protocol_id": data_payload.get("protocol_id"),
        "locked_test_eid_set_sha256": observed_set_hash,
        "source_manifest": run_source,
    })
    init_checkpoint = Path(str(resolved.get("config", {}).get("init_from_ckpt", "")))
    if not init_checkpoint.is_absolute():
        init_checkpoint = (ROOT / init_checkpoint).resolve()
    if not init_checkpoint.is_file():
        errors.append("pretraining_checkpoint_missing")
    else:
        pretraining_source, pretraining_source_errors = _source_manifest_evidence(
            init_checkpoint.parent.parent / "source_manifest.json"
        )
        errors.extend(f"pretraining:{item}" for item in pretraining_source_errors)
        evidence["pretraining_checkpoint"] = {
            "path": str(init_checkpoint.resolve()),
            "sha256": sha256_file(init_checkpoint),
        }
        evidence["pretraining_source_manifest"] = pretraining_source
    return evidence, errors


def _validation_binding(
    assets: dict[str, Path],
    protocol_path: Path,
    protocol: dict,
    checkpoints: dict[str, Path],
    model_data_dirs: dict[str, Path],
) -> tuple[dict, list[str]]:
    errors = []
    consistency_path = assets.get("validation_consistency")
    if consistency_path is None:
        return {}, ["validation_consistency_missing"]
    payload = json.loads(consistency_path.read_text(encoding="utf-8-sig"))
    binding = payload.get("protocol_binding", {})
    if not payload.get("ok") or not binding.get("ok"):
        errors.append("validation_consistency_binding_not_ok")
    if binding.get("protocol_json", {}).get("sha256") != sha256_file(protocol_path):
        errors.append("validation_consistency_protocol_hash_mismatch")
    if [float(value) for value in binding.get("horizons_years", [])] != [float(value) for value in protocol.get("horizons_years", [])]:
        errors.append("validation_consistency_horizons_mismatch")
    if [float(value) for value in binding.get("age_groups_years", [])] != [float(value) for value in protocol.get("age_groups_years", [])]:
        errors.append("validation_consistency_age_groups_mismatch")
    if binding.get("medical_protocol") != protocol.get("medical_protocol"):
        errors.append("validation_consistency_medical_protocol_mismatch")
    if bool(binding.get("sex_stratified")) != bool(protocol.get("sex_stratified")):
        errors.append("validation_consistency_sex_stratification_mismatch")
    if binding.get("case_control") != protocol.get("case_control"):
        errors.append("validation_consistency_case_control_mismatch")
    if float(binding.get("age_bin_width_years", 0.0)) != float(protocol.get("age_bin_width_years", 0.0)):
        errors.append("validation_consistency_age_bin_width_mismatch")
    controls = binding.get("controls", {})
    if set(controls) != set(checkpoints):
        errors.append("validation_consistency_controls_must_match_checkpoints")
    for name, checkpoint in checkpoints.items():
        source = controls.get(name, {})
        if str(Path(source.get("checkpoint", "")).resolve()) != str(checkpoint.resolve()):
            errors.append(f"validation_consistency_checkpoint_mismatch:{name}")
        if name in model_data_dirs and str(Path(source.get("data_dir", "")).resolve()) != str(model_data_dirs[name].resolve()):
            errors.append(f"validation_consistency_data_dir_mismatch:{name}")
        summary_path = Path(source.get("path", ""))
        if not summary_path.is_file() or sha256_file(summary_path) != source.get("sha256"):
            errors.append(f"validation_consistency_summary_hash_mismatch:{name}")
    landmark_path = assets.get("shared_validation_landmarks")
    if landmark_path is not None and str(Path(binding.get("shared_validation_landmark_manifest", "")).resolve()) != str(landmark_path.resolve()):
        errors.append("validation_consistency_landmark_mismatch")
    official = binding.get("official_aggregates", {})
    official_path = assets.get("validation_official_aggregates")
    if official_path is not None:
        if str(Path(official.get("path", "")).resolve()) != str(official_path.resolve()):
            errors.append("validation_official_aggregates_path_mismatch")
        if official.get("sha256") != sha256_file(official_path):
            errors.append("validation_official_aggregates_hash_mismatch")
    return binding, errors


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    audit = json.loads(args.split_audit.read_text(encoding="utf-8-sig"))
    splits = audit.get("splits", {})
    required = ("train", "val", "test")
    errors = []
    if not all(name in splits for name in required):
        errors.append("split_audit_missing_train_val_test")
    if not bool(audit.get("disjoint", {}).get("ok")):
        errors.append("split_audit_not_disjoint")
    if not bool(audit.get("expected_test_eids", {}).get("ok")):
        errors.append("split_audit_test_eids_not_verified")
    test_audit = splits.get("test", {})
    if not bool(test_audit.get("ok")) or int(test_audit.get("patient_count", 0)) == 0:
        errors.append("test_split_not_valid")
    if not args.test_eids_csv.is_file():
        errors.append("test_eids_csv_missing")
    protocol = json.loads(args.protocol_json.read_text(encoding="utf-8-sig"))
    protocol = dict(protocol)
    protocol["checkpoint_training_excludes_test"] = bool(args.confirm_checkpoint_excludes_test)
    if not args.confirm_checkpoint_excludes_test:
        errors.append("checkpoint_training_exclusion_not_confirmed")
    if args.test_eids_csv.is_file():
        protocol["test_eids_sha256"] = sha256_file(args.test_eids_csv)
    errors.extend(
        f"protocol_field_missing:{name}"
        for name in sorted(REQUIRED_PROTOCOL_FIELDS - set(protocol))
    )
    checkpoints = _specs(args.checkpoint)
    assets = _specs(args.asset)
    required_assets = REQUIRED_ASSETS | set(args.require_asset)
    errors.extend(f"required_asset_missing:{name}" for name in sorted(required_assets - set(assets)))
    effect_gate = None
    effect_gate_path = assets.get("validation_effect_gate")
    if effect_gate_path is not None:
        effect_gate = json.loads(effect_gate_path.read_text(encoding="utf-8-sig"))
        if not bool(effect_gate.get("ok")) and args.exploratory_override is None:
            errors.append("validation_effect_gate_not_ok")
    exploratory_override = None
    if args.exploratory_override is not None:
        if not args.exploratory_override.is_file():
            errors.append("exploratory_override_missing")
        else:
            exploratory_override = json.loads(args.exploratory_override.read_text(encoding="utf-8-sig"))
            if not bool(exploratory_override.get("authorized")):
                errors.append("exploratory_override_not_authorized")
            if exploratory_override.get("scope") != "CARoPE_locked_test_v2_stage_D":
                errors.append("exploratory_override_scope_invalid")
            if not str(exploratory_override.get("reason", "")).strip():
                errors.append("exploratory_override_reason_missing")
            if effect_gate is None:
                errors.append("exploratory_override_requires_validation_effect_gate")
            elif bool(effect_gate.get("ok")):
                errors.append("exploratory_override_requires_failed_validation_effect_gate")
            elif str(Path(exploratory_override.get("failed_gate_path", "")).resolve()) != str(effect_gate_path.resolve()):
                errors.append("exploratory_override_gate_path_mismatch")
    model_data_dirs = _dir_specs(args.model_data_dir)
    model_audit_paths = _specs(args.model_split_audit)
    if set(model_data_dirs) != set(checkpoints):
        errors.append("model_data_dirs_must_match_checkpoint_names")
    if set(model_audit_paths) != set(checkpoints):
        errors.append("model_split_audits_must_match_checkpoint_names")
    model_audits = {}
    for name, path in model_audit_paths.items():
        model_audits[name] = json.loads(path.read_text(encoding="utf-8-sig"))
        if not bool(model_audits[name].get("disjoint", {}).get("ok")):
            errors.append(f"model_split_audit_not_ok:{name}")
        if not bool(model_audits[name].get("expected_test_eids", {}).get("ok")):
            errors.append(f"model_split_audit_test_eids_not_verified:{name}")
        if int(model_audits[name].get("splits", {}).get("test", {}).get("patient_count", 0)) != int(test_audit.get("patient_count", 0)):
            errors.append(f"model_test_count_mismatch:{name}")
    model_test_eid_hashes = {
        name: _eid_sequence_hash(path, "test")
        for name, path in model_data_dirs.items()
    }
    if len(set(model_test_eid_hashes.values())) > 1:
        errors.append("model_test_eid_sequences_differ")
    expected_eids = read_eid_csv(args.test_eids_csv) if args.test_eids_csv.is_file() else []
    expected_eid_set_hash = eid_set_sha256(expected_eids) if expected_eids else ""
    training_evidence = {}
    for name, checkpoint in checkpoints.items():
        if name not in model_data_dirs:
            continue
        evidence, evidence_errors = _training_evidence(
            checkpoint,
            model_data_dirs[name],
            expected_eid_set_hash,
            str(protocol.get("data_protocol_id", "")),
        )
        training_evidence[name] = evidence
        errors.extend(f"checkpoint_training_evidence:{name}:{item}" for item in evidence_errors)
    validation_binding, validation_errors = _validation_binding(
        assets, args.protocol_json, protocol, checkpoints, model_data_dirs
    )
    errors.extend(validation_errors)
    if errors:
        payload = {"ok": False, "errors": errors, "split_audit": str(args.split_audit.resolve())}
        write_json(args.out, payload)
        print(json.dumps(payload, indent=2))
        return 2
    manifest = build_freeze_manifest(
        args.data_dir,
        splits,
        checkpoints,
        args.diseases_yaml,
        protocol,
        assets,
        model_data_dirs,
        model_audits,
    )
    manifest["manifest_version"] = MANIFEST_VERSION
    manifest["protocol_source"] = {
        "path": str(args.protocol_json.resolve()),
        "sha256": sha256_file(args.protocol_json),
    }
    manifest["test_eids"] = {
        "path": str(args.test_eids_csv.resolve()),
        "sha256": sha256_file(args.test_eids_csv),
        "eid_set_sha256": expected_eid_set_hash,
        "patient_count": int(test_audit["patient_count"]),
    }
    manifest["split_audit_sha256"] = sha256_file(args.split_audit)
    manifest["split_audit_path"] = str(args.split_audit.resolve())
    manifest["model_split_audit_files"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in model_audit_paths.items()
    }
    manifest["model_test_eid_sequence_sha256"] = model_test_eid_hashes
    manifest["checkpoint_training_evidence"] = training_evidence
    manifest["validation_binding"] = validation_binding
    manifest["validation_sources"] = {
        "control_summaries": {
            name: {"path": source.get("path"), "sha256": source.get("sha256")}
            for name, source in validation_binding.get("controls", {}).items()
        },
        "official_aggregates": validation_binding.get("official_aggregates", {}),
    }
    manifest["run_class"] = "exploratory_gate_override" if exploratory_override is not None else "formal_locked_test"
    manifest["formal_validation_gate_passed"] = None if effect_gate is None else bool(effect_gate.get("ok"))
    if exploratory_override is not None:
        manifest["exploratory_override"] = {
            "path": str(args.exploratory_override.resolve()),
            "sha256": sha256_file(args.exploratory_override),
            "authorization": exploratory_override,
        }
    manifest["ok"] = True
    write_json(args.out, manifest)
    print(json.dumps({"ok": True, "out": str(args.out), "checkpoints": sorted(checkpoints), "assets": sorted(assets)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
