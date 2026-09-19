"""Compare the new medical-protocol control summaries with formal AUC output.

The check is intended for validation only.  It compares the same disease and
horizon rows, reports count/score deltas, and returns non-zero when the
protocols disagree beyond the configured tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", action="append", required=True, help="name=medical-control summary.json")
    p.add_argument("--official-aggregates", type=Path, required=True, help="calibration_auc_aggregates.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-abs-delta", type=float, default=0.03)
    p.add_argument("--require", action="append", default=[], help="required control names")
    p.add_argument("--protocol-json", type=Path, default=None, help="v2 protocol declaration used to bind horizons and strata")
    return p


def _load_specs(values):
    output = {}
    for value in values:
        name, path = value.split("=", 1)
        source = Path(path)
        output[name] = {
            "path": source,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "payload": json.loads(source.read_text(encoding="utf-8")),
        }
    return output


def _float_list(values):
    return [float(value) for value in (values or [])]


def _official_rows(payload: dict, model_name: str) -> list[dict]:
    models = payload.get("models")
    if isinstance(models, dict):
        model = models.get(model_name, {})
        return list(model.get("horizon_risk_age_sex_macro", []))
    return list(payload.get("horizon_risk_age_sex_macro", []))


def _protocol_binding(args, controls, official_path: Path, official_payload: dict) -> dict:
    errors = []
    if args.protocol_json is None:
        return {"ok": False, "errors": ["protocol_json_not_provided"]}
    protocol = json.loads(args.protocol_json.read_text(encoding="utf-8-sig"))
    expected_horizons = _float_list(protocol.get("horizons_years", []))
    expected_ages = _float_list(protocol.get("age_groups_years", []))
    expected_medical_protocol = str(protocol.get("medical_protocol", ""))
    expected_sex_stratified = bool(protocol.get("sex_stratified"))
    expected_case_control = str(protocol.get("case_control", ""))
    expected_age_bin_width = float(protocol.get("age_bin_width_years", 0.0))
    if not expected_horizons:
        errors.append("protocol_horizons_missing")
    if not expected_ages:
        errors.append("protocol_age_groups_missing")
    official_horizons_by_model = {}
    for name in controls:
        rows = _official_rows(official_payload, name)
        horizons = sorted({float(row["horizon_years"]) for row in rows})
        official_horizons_by_model[name] = horizons
        if horizons != sorted(expected_horizons):
            errors.append(
                f"official_aggregate_horizons_mismatch:{name}"
                if isinstance(official_payload.get("models"), dict)
                else "official_aggregate_horizons_mismatch"
            )
    official_models = official_payload.get("models")
    if isinstance(official_models, dict) and set(official_models) != set(controls):
        errors.append("official_aggregate_models_mismatch")
    control_sources = {}
    shared_landmarks = set()
    for name, source in controls.items():
        payload = source["payload"]
        signature = payload.get("protocol_signature", {})
        horizons = _float_list(payload.get("horizons_years", []))
        age_groups = _float_list(payload.get("age_groups", []))
        landmark = signature.get("shared_landmark_manifest")
        checkpoint = payload.get("checkpoint")
        data_dir = payload.get("data_dir")
        if payload.get("split") != "val":
            errors.append(f"control_not_validation:{name}")
        if payload.get("protocol") != expected_medical_protocol:
            errors.append(f"control_protocol_mismatch:{name}")
        if horizons != expected_horizons:
            errors.append(f"control_horizons_mismatch:{name}")
        if age_groups != expected_ages or _float_list(signature.get("age_groups_years", [])) != expected_ages:
            errors.append(f"control_age_groups_mismatch:{name}")
        if bool(signature.get("sex_stratified")) != expected_sex_stratified:
            errors.append(f"control_sex_stratification_mismatch:{name}")
        if str(signature.get("case_control", "")) != expected_case_control:
            errors.append(f"control_case_control_mismatch:{name}")
        if float(signature.get("age_bin_width_years", 0.0)) != expected_age_bin_width:
            errors.append(f"control_age_bin_width_mismatch:{name}")
        if not landmark:
            errors.append(f"control_shared_landmark_missing:{name}")
        else:
            shared_landmarks.add(str(Path(landmark).resolve()))
        if not checkpoint:
            errors.append(f"control_checkpoint_missing:{name}")
        if not data_dir:
            errors.append(f"control_data_dir_missing:{name}")
        control_sources[name] = {
            "path": str(source["path"].resolve()),
            "sha256": source["sha256"],
            "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
            "data_dir": str(Path(data_dir).resolve()) if data_dir else None,
            "shared_landmark_manifest": str(Path(landmark).resolve()) if landmark else None,
            "horizons_years": horizons,
            "age_groups_years": age_groups,
        }
    if len(shared_landmarks) != 1:
        errors.append("control_shared_landmarks_differ")
    return {
        "ok": not errors,
        "errors": errors,
        "protocol_json": {
            "path": str(args.protocol_json.resolve()),
            "sha256": hashlib.sha256(args.protocol_json.read_bytes()).hexdigest(),
        },
        "medical_protocol": expected_medical_protocol,
        "horizons_years": expected_horizons,
        "age_groups_years": expected_ages,
        "age_bin_width_years": expected_age_bin_width,
        "sex_stratified": expected_sex_stratified,
        "case_control": expected_case_control,
        "shared_validation_landmark_manifest": next(iter(shared_landmarks), None),
        "controls": control_sources,
        "official_aggregates": {
            "path": str(official_path.resolve()),
            "sha256": hashlib.sha256(official_path.read_bytes()).hexdigest(),
            "horizons_years_by_model": official_horizons_by_model,
            "model_sources": {
                name: {"path": value.get("path"), "sha256": value.get("sha256")}
                for name, value in (official_models or {}).items()
            },
        },
    }


def _official_by_key(payload):
    rows = payload.get("horizon_risk_age_sex_macro", [])
    return {(str(row["disease_id"]), float(row["horizon_years"])): row for row in rows}


def _control_by_key(payload):
    grouped = defaultdict(list)
    for row in payload.get("summary", {}).get("cells", []):
        raw_auc = row.get("auc")
        if raw_auc is None:
            continue
        try:
            auc = float(raw_auc)
        except (TypeError, ValueError):
            continue
        if math.isfinite(auc):
            grouped[(str(row["disease_id"]), float(row["horizon_years"]))].append(row)
    return {key: sum(float(item["auc"]) for item in values) / len(values) for key, values in grouped.items()}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    controls = _load_specs(args.control)
    official_payload = json.loads(args.official_aggregates.read_text(encoding="utf-8"))
    missing_required = sorted(set(args.require) - set(controls))
    reports = {}
    all_ok = not missing_required
    for name, source in controls.items():
        payload = source["payload"]
        control = _control_by_key(payload)
        official = _official_by_key({"horizon_risk_age_sex_macro": _official_rows(official_payload, name)})
        rows = []
        for key in sorted(set(control) | set(official)):
            if key not in control or key not in official:
                rows.append({"disease_id": key[0], "horizon_years": key[1], "status": "missing", "control_auc": control.get(key), "official_auc": official.get(key, {}).get("auc")})
                continue
            delta = float(control[key] - float(official[key]["auc"]))
            rows.append({"disease_id": key[0], "horizon_years": key[1], "status": "ok" if abs(delta) <= args.max_abs_delta else "mismatch", "control_auc": control[key], "official_auc": float(official[key]["auc"]), "delta": delta})
        ok = all(row["status"] == "ok" for row in rows) and bool(rows)
        reports[name] = {"ok": ok, "max_abs_delta": max((abs(float(row.get("delta", 0.0))) for row in rows if "delta" in row), default=float("nan")), "rows": rows}
        all_ok = all_ok and ok
    binding = _protocol_binding(args, controls, args.official_aggregates, official_payload)
    if args.protocol_json is not None:
        all_ok = all_ok and bool(binding["ok"])
    result = {"protocol": "paper_medical_control_v1", "official_aggregates": str(args.official_aggregates.resolve()), "max_abs_delta": args.max_abs_delta, "missing_required": missing_required, "ok": all_ok, "protocol_binding": binding, "controls": reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"ok": all_ok, "out": str(args.out), "controls": {name: value["ok"] for name, value in reports.items()}}, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
