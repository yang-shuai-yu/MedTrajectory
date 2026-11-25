"""Export a trusted Track R A2 checkpoint without optimizer, RNG or patient data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def export(checkpoint, expected_sha256, labels_path, protocol_path, sampler_path, output):
    if sha256(checkpoint) != expected_sha256:
        raise ValueError("Checkpoint SHA-256 differs from the trusted provenance record")
    # These are the only non-tensor types in the trusted training checkpoints.
    # PurePosixPath makes Linux path metadata readable on Windows; it is discarded.
    reconstruct = np._core.multiarray._reconstruct if hasattr(np, "_core") else np.core.multiarray._reconstruct
    allowed = [(PurePosixPath, "pathlib.PosixPath"),
               (reconstruct, "numpy.core.multiarray._reconstruct"),
               np.ndarray, np.dtype, type(np.dtype("uint32"))]
    with torch.serialization.safe_globals(allowed):
        original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = original["model_args"]
    if not (original["stage"] == "pretraining" and config["use_age_encoding"]
            and config["use_age_rope"] and config["age_rope_variant"] == "additive_v2_2"
            and original["config"]["seed"] == 42):
        raise ValueError("This release exports the seed42 A2 pretraining model only")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8-sig"))
    selection = json.loads(Path(sampler_path).read_text(encoding="utf-8-sig"))
    with Path(labels_path).open(encoding="utf-8-sig", newline="") as stream:
        labels = [row[0] for row in csv.reader(stream)]
    if len(labels) != 8424 or labels[:2] != ["Padding", "No event"] or len(set(labels)) != len(labels):
        raise ValueError("Unexpected dynamic vocabulary")
    keys = []
    for field in protocol["static_prefix"]["logical_order"]:
        spec = protocol["static_fields"][field]
        categories = spec.get("categories") or spec.get("bins") or sorted(set(spec["coding"].values()))
        keys.extend(f"static:{field}:{category}" for category in categories)
    static_ids = {key: len(labels) + index for index, key in enumerate(keys)}
    bos_id = len(labels) + len(keys)
    if bos_id != original["config"]["track_r_bos_token_id"] or bos_id + 2 != config["vocab_size"]:
        raise ValueError("Static vocabulary or BOS does not match the trained checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = {key.removeprefix("_orig_mod."): value.detach().cpu().contiguous().clone()
             for key, value in original["model"].items()}
    # Historical training constructed a zero-input static MLP even with static_dim=0.
    # Static information in A2 is supplied as tokens; this MLP is never invoked.
    # Current public source omits it. Remove only its four known, unused tensors.
    removed = []
    if config["static_dim"] == 0:
        removed = sorted(key for key in state if key.startswith("static_encoder."))
        expected = ["static_encoder.0.bias", "static_encoder.0.weight",
                    "static_encoder.3.bias", "static_encoder.3.weight"]
        if removed != expected:
            raise ValueError("Unexpected historical static branch; review export compatibility")
        for key in removed:
            del state[key]
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Nonfinite model weights")
    torch.save(state, output / "model.pt")
    save_json(output / "config.json", config)
    save_json(output / "vocabulary.json", {
        "dynamic_labels": labels, "static_token_ids": static_ids,
        "static_order": protocol["static_prefix"]["logical_order"],
        "bos_token_id": bos_id, "mask_token_id": bos_id + 1,
    })
    selected = selection["selected_by_family"]["carope"]
    files = {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
             for name in ("model.pt", "config.json", "vocabulary.json")}
    manifest = {
        "format_version": 1, "model_id": "medtrajectory-ukb-a2-seed42-generation-v1",
        "architecture": "absolute age + additive continuous relative-time attention",
        "training_seed": 42, "training_stage": "pretraining",
        "training_iteration": int(original["iteration"]),
        "omitted_unused_parameters": removed,
        "source_checkpoint_sha256": expected_sha256,
        "source_labels_sha256": sha256(labels_path),
        "source_protocol_sha256": sha256(protocol_path),
        "source_sampler_sha256": sha256(sampler_path),
        "dynamic_context_length": int(original["config"]["dynamic_context_length"]),
        "sampler": {key: selected[key] for key in ("temperature", "top_p", "death_logit_bias")},
        "minimum_wait_days": 1.0,
        "files": files,
        "scope": "Multitype trajectory generation; no calibrated disease or mortality risk endpoint.",
    }
    save_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "checkpoint-sha256", "labels", "protocol", "sampler", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = export(args.checkpoint, args.checkpoint_sha256, args.labels,
                    args.protocol, args.sampler, args.output)
    print(json.dumps(result, indent=2))
