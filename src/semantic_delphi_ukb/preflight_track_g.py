"""Validate Track G protocol, data contract, and checkpoint metadata without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.track_g_contract import (  # noqa: E402
    accepted_checkpoint_protocol_hashes,
    load_track_g_protocol,
)
from semantic_delphi_ukb.track_r_batch import validate_track_r_data_manifest  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=REPO_DIR / "configs/track_g_v1/TRACK_G_v1.json")
    value.add_argument("--data-dir", type=Path, default=None)
    value.add_argument("--require-trained-baselines", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_g_protocol(args.protocol, REPO_DIR)
    source = protocol["_source_track_r"]
    data_dir = args.data_dir or Path(source["output_data_dir"])
    validate_track_r_data_manifest(data_dir, source)
    checked = []
    for seed in protocol["seeds"]:
        for model in protocol["models"]:
            if model["train"] and not args.require_trained_baselines:
                continue
            path = REPO_DIR / model["checkpoint"].format(seed=seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if model["train"]:
                if checkpoint.get("protocol_manifest_sha256") not in accepted_checkpoint_protocol_hashes(protocol):
                    raise ValueError(f"Track G checkpoint protocol hash mismatch: {path}")
                if checkpoint.get("track_g_family") != model["family"]:
                    raise ValueError(f"Track G checkpoint family mismatch: {path}")
                required = ("optimizer", "scheduler", "random_state", "iteration", "global_step")
                if any(key not in checkpoint for key in required):
                    raise ValueError(f"Track G checkpoint is not fully resumable: {path}")
            else:
                if checkpoint.get("protocol_manifest_sha256") != protocol["source_track_r_protocol_sha256"]:
                    raise ValueError(f"source Track R checkpoint protocol hash mismatch: {path}")
                if checkpoint.get("stage") != "pretraining":
                    raise ValueError(f"generation must use a pretraining checkpoint: {path}")
            checked.append(str(path))
    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "data_dir": str(data_dir),
        "require_trained_baselines": args.require_trained_baselines,
        "checked_checkpoints": checked,
        "locked_test_read": protocol["locked_test_read"],
        "test_authorized": protocol["test_authorized"],
        "passed": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
