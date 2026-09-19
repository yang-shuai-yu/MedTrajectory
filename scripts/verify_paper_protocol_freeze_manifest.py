"""Re-hash every frozen input before any locked-test evaluator reads test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.paper_protocol_audit import sha256_file, write_json  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p


def _check_file(errors: list[str], cache: dict[Path, str], label: str, path_value: str, expected: str) -> None:
    path = Path(path_value).resolve()
    if not path.is_file():
        errors.append(f"missing:{label}:{path}")
        return
    if path not in cache:
        cache[path] = sha256_file(path)
    if cache[path] != expected:
        errors.append(f"sha256_mismatch:{label}:{path}")


def _check_record(errors: list[str], cache: dict[Path, str], label: str, record: dict) -> None:
    if record.get("path") and record.get("sha256"):
        _check_file(errors, cache, label, str(record["path"]), str(record["sha256"]))
    else:
        errors.append(f"invalid_frozen_record:{label}")


def _check_split_hashes(errors: list[str], cache: dict[Path, str], label: str, data_dir: Path, splits: dict) -> None:
    for split_name, split in splits.items():
        for filename, expected in split.get("sha256", {}).items():
            _check_file(errors, cache, f"{label}:{split_name}:{filename}", str(data_dir / filename), str(expected))


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"verification output already exists: {args.out}")
    manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    digest_cache: dict[Path, str] = {}
    if manifest.get("manifest_version") != "paper_medical_control_locked_test_v2" or not manifest.get("ok"):
        errors.append("freeze_manifest_not_ready")
    for name, record in manifest.get("checkpoints", {}).items():
        _check_record(errors, digest_cache, f"checkpoint:{name}", record)
    _check_record(errors, digest_cache, "diseases_yaml", manifest.get("diseases_yaml", {}))
    _check_record(errors, digest_cache, "protocol_source", manifest.get("protocol_source", {}))
    for name, record in manifest.get("frozen_files", {}).items():
        _check_record(errors, digest_cache, f"frozen_file:{name}", record)
    if manifest.get("run_class") == "exploratory_gate_override":
        override = manifest.get("exploratory_override", {})
        _check_record(errors, digest_cache, "exploratory_override", override)
        authorization = override.get("authorization", {})
        if not bool(authorization.get("authorized")):
            errors.append("exploratory_override_not_authorized")
        if authorization.get("scope") != "CARoPE_locked_test_v2_stage_D":
            errors.append("exploratory_override_scope_invalid")
    _check_record(errors, digest_cache, "test_eids", manifest.get("test_eids", {}))
    split_audit_path = manifest.get("split_audit_path")
    if split_audit_path and manifest.get("split_audit_sha256"):
        _check_file(errors, digest_cache, "split_audit", split_audit_path, manifest["split_audit_sha256"])
    else:
        errors.append("split_audit_path_or_hash_missing")
    _check_split_hashes(errors, digest_cache, "primary_data", Path(manifest.get("data_dir", "")), manifest.get("splits", {}))
    for name, audit in manifest.get("model_split_audits", {}).items():
        _check_split_hashes(errors, digest_cache, f"model_data:{name}", Path(audit.get("data_dir", "")), audit.get("splits", {}))
    for name, record in manifest.get("model_split_audit_files", {}).items():
        _check_record(errors, digest_cache, f"model_split_audit:{name}", record)
    for name, evidence in manifest.get("checkpoint_training_evidence", {}).items():
        _check_file(errors, digest_cache, f"resolved_config:{name}", evidence.get("resolved_config", ""), evidence.get("resolved_config_sha256", ""))
        _check_file(errors, digest_cache, f"training_data_manifest:{name}", evidence.get("data_manifest", ""), evidence.get("data_manifest_sha256", ""))
        _check_record(errors, digest_cache, f"source_manifest:{name}", evidence.get("source_manifest", {}))
        _check_record(errors, digest_cache, f"pretraining_checkpoint:{name}", evidence.get("pretraining_checkpoint", {}))
        _check_record(errors, digest_cache, f"pretraining_source_manifest:{name}", evidence.get("pretraining_source_manifest", {}))
    for name, record in manifest.get("validation_sources", {}).get("control_summaries", {}).items():
        _check_record(errors, digest_cache, f"validation_summary:{name}", record)
    _check_record(
        errors,
        digest_cache,
        "validation_official_aggregates",
        manifest.get("validation_sources", {}).get("official_aggregates", {}),
    )
    payload = {
        "ok": not errors,
        "freeze_manifest": str(args.freeze_manifest.resolve()),
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        "files_verified": len(digest_cache),
        "errors": errors,
    }
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
