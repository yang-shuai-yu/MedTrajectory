from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_DIR / "data" / "paper_protocol_v1"
DEFAULT_CONFIG_ROOT = REPO_DIR / "configs" / "paper_protocol_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate paper_protocol_v1 data and configs without launching training.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def patient_ids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["eid"]) for row in csv.DictReader(handle)}


def check_profile(data_dir: Path, expected_profile: str) -> list[str]:
    errors: list[str] = []
    manifest_path = data_dir / "prepare_manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = load_json(manifest_path)
    if manifest.get("profile") != expected_profile:
        errors.append(f"profile mismatch for {data_dir}")
    if manifest.get("ethnicity_model_input") is not False:
        errors.append(f"ethnicity must not be a model input for {expected_profile}")
    if manifest.get("static_feature_order") != ["sex_id"]:
        errors.append(f"unexpected static features for {expected_profile}")
    if manifest.get("training_started") is not False:
        errors.append(f"data manifest says training started for {expected_profile}")

    split_ids: dict[str, set[int]] = {}
    for split in ("train", "val", "longitudinal"):
        required = [
            data_dir / f"{split}.bin",
            data_dir / f"{split}_static.npy",
            data_dir / f"{split}_followup_end_age_days.npy",
            data_dir / f"{split}_patient_index.csv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            errors.extend(f"missing {path}" for path in missing)
            continue
        split_ids[split] = patient_ids(required[3])
        static = np.load(required[1], mmap_mode="r")
        followup = np.load(required[2], mmap_mode="r")
        expected = int(manifest["split_summaries"][split]["patients"])
        if static.shape != (expected, 1):
            errors.append(f"{split} static shape mismatch for {expected_profile}")
        if followup.shape != (expected,):
            errors.append(f"{split} followup shape mismatch for {expected_profile}")
        if len(split_ids[split]) != expected:
            errors.append(f"{split} patient index count mismatch for {expected_profile}")
    if split_ids.get("train", set()) & split_ids.get("val", set()):
        errors.append(f"train/val participant overlap for {expected_profile}")
    if not (data_dir / "longitudinal_outcomes.csv").exists():
        errors.append(f"missing longitudinal outcomes for {expected_profile}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors: list[str] = []
    common_path = args.config_root / "common.json"
    if not common_path.exists():
        errors.append(f"missing {common_path}")
        common = {}
    else:
        common = load_json(common_path)
    if common.get("split_seed") != 1337:
        errors.append("paper split seed must remain 1337")
    if common.get("training_seeds") != [42, 43, 44]:
        errors.append("training seeds must be [42, 43, 44]")
    if common.get("posttraining", {}).get("target_protocol") != "paper_censor_aware_v1":
        errors.append("posttraining target protocol is not censor-aware")
    pretraining = common.get("pretraining", {})
    if pretraining.get("encoder_family") != "causal_delphi":
        errors.append("pretraining encoder is not causal Delphi")
    if pretraining.get("objective") != "all_position_next_event_and_time":
        errors.append("pretraining objective is not all-position next-event/time")
    posttraining = common.get("posttraining", {})
    if posttraining.get("horizons_years") != [1, 5, 10]:
        errors.append("posttraining horizons must be [1, 5, 10]")
    if posttraining.get("monotonic_horizon_risk") is not True:
        errors.append("posttraining risk head must be monotonic")
    if float(posttraining.get("next_event_loss_weight", -1)) != 0.2:
        errors.append("next-event auxiliary loss weight must be 0.2")
    if int(posttraining.get("head_warmup_iters", 0)) <= 0:
        errors.append("head warm-up must freeze the trunk for a positive number of iterations")
    if float(posttraining.get("joint_trunk_learning_rate", 1)) >= float(posttraining.get("joint_head_learning_rate", 0)):
        errors.append("joint trunk learning rate must be lower than head learning rate")
    survival = common.get("survival_horizon", {})
    if survival.get("enabled_experiments") != ["P0", "P3"] or survival.get("target_protocol") != "paper_censor_aware_v1":
        errors.append("survival-horizon must be censor-aware and enabled for P0/P3")
    evaluation = common.get("evaluation", {})
    if evaluation.get("primary_auc") != "delphi2m_medical_auc" or evaluation.get("implementation") != "DeLong":
        errors.append("primary AUC is not the Delphi-2M DeLong medical AUC")
    required_entrypoints = [
        pretraining.get("entrypoint"), posttraining.get("entrypoint"), survival.get("entrypoint"),
        evaluation.get("medical_auc_entrypoint"), evaluation.get("longitudinal_entrypoint"),
    ]
    for entrypoint in required_entrypoints:
        if not entrypoint or not (REPO_DIR / str(entrypoint)).exists():
            errors.append(f"missing protocol entrypoint: {entrypoint}")
    required_run_files = set(common.get("run_contract", {}).get("required_files", []))
    expected_run_files = {
        "resolved_config.json", "source_manifest.json", "stdout.log", "metrics.jsonl", "status.json",
        "tensorboard", "checkpoints/last.pt", "checkpoints/best_val_loss.pt",
    }
    if required_run_files != expected_run_files or common.get("run_contract", {}).get("resume_required") is not True:
        errors.append("run contract is incomplete or not resumable")
    for experiment_key in ("P0", "P1", "P2", "P3", "P4"):
        if not (REPO_DIR / "scripts" / f"run_{experiment_key}_seed1337.sh").exists():
            errors.append(f"missing launcher for {experiment_key}")
    if common.get("launch_state") != "prepared_not_started":
        errors.append("launch_state must remain prepared_not_started")

    experiment_rows = []
    checked_profiles: set[str] = set()
    for config_path in sorted(args.config_root.glob("P*.json")):
        config = load_json(config_path)
        profile = str(config["dataset_profile"])
        if config.get("encoder") != "causal_delphi":
            errors.append(f"{config_path.name} is not configured for causal Delphi")
        data_dir = args.data_root / profile
        if profile not in checked_profiles:
            errors.extend(check_profile(data_dir, profile))
            checked_profiles.add(profile)
        experiment_rows.append(
            {
                "experiment_id": config["experiment_id"],
                "dataset_profile": profile,
                "encoder": config["encoder"],
                "continuous_age_rope": bool(config["continuous_age_rope"]),
            }
        )
    if len(experiment_rows) != 5:
        errors.append("expected exactly five P0-P4 experiment configs")

    report = {
        "status": "READY" if not errors else "BLOCKED",
        "training_started": False,
        "experiments": experiment_rows,
        "checked_profiles": sorted(checked_profiles),
        "errors": errors,
    }
    output = args.output or (args.data_root / "preflight_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
