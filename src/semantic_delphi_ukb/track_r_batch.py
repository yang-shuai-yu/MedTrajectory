from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from semantic_delphi_ukb.multitype_batch import get_batch
from semantic_delphi_ukb.track_r_contract import exact_time_loss_mask, prepend_static_context


STATIC_CONDITIONING_MODES = ("none", "legacy-sex-residual", "categorical-residual")


def _manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))


def track_r_static_dim(data_dir: Path, mode: str) -> int:
    if mode not in STATIC_CONDITIONING_MODES:
        raise ValueError(f"unknown Track R static conditioning mode: {mode}")
    if mode == "none":
        return 0
    if mode == "legacy-sex-residual":
        return 1
    return len(_manifest(data_dir)["static_token_ids"])


def load_track_r_static_features(
    data_dir: Path,
    split: str,
    mode: str,
    max_patients: int = 0,
) -> np.ndarray:
    prefix_ids = np.load(data_dir / f"{split}_static_prefix_token_ids.npy").astype(np.int64)
    if max_patients > 0:
        prefix_ids = prefix_ids[:max_patients]
    manifest = _manifest(data_dir)
    token_ids = manifest["static_token_ids"]
    if mode == "none":
        return np.zeros((len(prefix_ids), 0), dtype=np.float32)
    if mode == "legacy-sex-residual":
        sex = np.full(len(prefix_ids), -3.0, dtype=np.float32)
        sex[np.any(prefix_ids == int(token_ids["static:sex:female"]), axis=1)] = 0.0
        sex[np.any(prefix_ids == int(token_ids["static:sex:male"]), axis=1)] = 1.0
        return sex[:, None]
    if mode != "categorical-residual":
        raise ValueError(f"unknown Track R static conditioning mode: {mode}")
    ordered_ids = np.asarray(list(token_ids.values()), dtype=np.int64)
    return np.any(prefix_ids[:, :, None] == ordered_ids[None, None, :], axis=1).astype(np.float32)


def load_track_r_assets(data_dir: Path, split: str, max_patients: int = 0) -> tuple[np.ndarray, np.ndarray]:
    prefix_ids = np.load(data_dir / f"{split}_static_prefix_token_ids.npy").astype(np.int64)
    anchor_ages = np.load(data_dir / f"{split}_static_anchor_age_days.npy").astype(np.float32)
    if prefix_ids.ndim != 2 or anchor_ages.shape != (prefix_ids.shape[0],):
        raise ValueError(f"Track R static assets are misaligned for split={split}")
    if max_patients > 0:
        prefix_ids = prefix_ids[:max_patients]
        anchor_ages = anchor_ages[:max_patients]
    return prefix_ids, anchor_ages


def load_track_r_bos_token_id(data_dir: Path) -> int:
    manifest = _manifest(data_dir)
    return int(manifest["dynamic_bos_token_id"])


def get_track_r_batch(
    ix,
    data,
    p2i,
    static_matrix,
    prefix_ids: np.ndarray,
    anchor_ages: np.ndarray,
    *,
    include_static_prefix: bool,
    dynamic_context_length: int,
    device: str,
    select: str,
    padding: str,
    no_event_token_rate: int,
    cut_batch: bool,
    bos_token_id: int,
    static_conditioning: str = "none",
):
    x, age, y, target_age, _legacy_static = get_batch(
        ix,
        data,
        p2i,
        static_matrix,
        block_size=dynamic_context_length,
        device=device,
        select=select,
        padding=padding,
        no_event_token_rate=no_event_token_rate,
        cut_batch=cut_batch,
    )
    ix_np = ix.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(ix) else np.asarray(ix, dtype=np.int64)
    anchors = torch.as_tensor(anchor_ages[ix_np], dtype=torch.float32, device=device)
    if include_static_prefix:
        tokens = torch.as_tensor(prefix_ids[ix_np], dtype=torch.long, device=device)
    else:
        tokens = torch.empty((x.size(0), 0), dtype=torch.long, device=device)
    x, age, y, target_age, static_mask, bos_mask, next_mask = prepend_static_context(
        x, age, y, target_age, tokens, anchors, bos_token_id
    )
    time_mask = exact_time_loss_mask(next_mask, age, target_age, bos_mask)
    static_features = None
    static_feature_mask = None
    if static_conditioning != "none":
        static_features = torch.as_tensor(static_matrix[ix_np], dtype=torch.float32, device=device)
        if static_conditioning == "categorical-residual":
            static_feature_mask = (x > 0) & (age >= anchors[:, None])
        elif static_conditioning != "legacy-sex-residual":
            raise ValueError(f"unknown Track R static conditioning mode: {static_conditioning}")
    return x, age, y, target_age, static_features, static_mask, bos_mask, next_mask, time_mask, static_feature_mask


def validate_track_r_data_manifest(data_dir: Path, protocol: dict) -> dict:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    expected_data_protocol = protocol.get("data_protocol_id", protocol["protocol_id"])
    if manifest.get("protocol_id") != expected_data_protocol:
        raise ValueError("Track R data manifest protocol_id mismatch")
    if manifest.get("loss_contract") != protocol["loss_contract"]:
        raise ValueError("Track R data loss contract does not match the frozen protocol")
    if manifest.get("static_prefix", {}).get("age_anchor") != protocol["static_prefix"]["age_anchor"]:
        raise ValueError("Track R static anchor contract mismatch")
    if manifest.get("dynamic_bos") != protocol["dynamic_bos"]:
        raise ValueError("Track R dynamic BOS contract mismatch")
    if not isinstance(manifest.get("dynamic_bos_token_id"), int):
        raise ValueError("Track R data manifest is missing dynamic_bos_token_id")
    return manifest
