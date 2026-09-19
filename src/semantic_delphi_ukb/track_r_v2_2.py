from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from semantic_delphi_ukb.track_r_contract import exact_time_loss_mask, prepend_static_context


def canonical_sha256(payload: dict) -> str:
    value = dict(payload)
    value.pop("sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def positive_consecutive_gaps(data: np.ndarray, p2i: np.ndarray) -> np.ndarray:
    chunks = []
    for start, length in np.asarray(p2i, dtype=np.int64):
        ages = np.asarray(data[int(start): int(start + length), 1], dtype=np.float64)
        ages = np.sort(ages, kind="stable")
        gaps = np.diff(ages)
        if gaps.size:
            chunks.append(gaps[gaps > 0])
    if not chunks:
        raise ValueError("train split contains no positive consecutive event gaps")
    return np.concatenate(chunks)


def load_wavelength_manifest(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    scale_factor_bounds: Optional[Sequence[float]] = None,
    require_log_scale_match: bool = False,
) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    actual_sha256 = canonical_sha256(manifest)
    if manifest.get("sha256") != actual_sha256:
        raise ValueError("RoPE wavelength manifest SHA-256 mismatch")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("RoPE wavelength manifest does not match protocol expected_sha256")
    if manifest.get("source_split") != "train":
        raise ValueError("RoPE wavelength manifest must be derived from train only")
    if require_log_scale_match:
        if scale_factor_bounds is None:
            raise ValueError("protocol scale_factor_bounds are required for manifest cross-check")
        log_bounds = tuple(float(value) for value in manifest["log_scale_bounds"])
        scale_bounds = tuple(float(value) for value in scale_factor_bounds)
        if len(log_bounds) != 2 or len(scale_bounds) != 2:
            raise ValueError("RoPE scale bounds must each contain exactly two values")
        if not all(
            math.isclose(math.exp(log_value), scale_value, rel_tol=0.0, abs_tol=1e-12)
            for log_value, scale_value in zip(log_bounds, scale_bounds)
        ):
            raise ValueError("manifest log_scale_bounds disagree with protocol scale_factor_bounds")
    return manifest


def build_wavelength_manifest(
    gaps_days: Sequence[float],
    *,
    n_heads: int = 8,
    head_dim: int = 8,
) -> dict:
    gaps = np.asarray(gaps_days, dtype=np.float64)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if not gaps.size:
        raise ValueError("wavelength construction requires positive finite train gaps")
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    p1, p95 = np.percentile(gaps, [1.0, 95.0])
    wavelength_lo = max(2.0 * float(p1), 7.0)
    wavelength_hi = 1.5 * float(p95)
    if wavelength_hi <= wavelength_lo:
        raise ValueError("train gap distribution does not define a non-empty wavelength range")
    n_frequencies = head_dim // 2
    grid = np.geomspace(wavelength_lo, wavelength_hi, n_heads * n_frequencies)
    # Head h receives h, h+n_heads, ... from the global short-to-long grid.
    head_major = grid.reshape(n_frequencies, n_heads).T.reshape(-1)
    manifest = {
        "schema_version": "track_r_v2_2_wavelengths_v1",
        "source_split": "train",
        "gap_contract": "positive_consecutive_dynamic_event_gap_days",
        "p1_gap_days": float(p1),
        "p95_gap_days": float(p95),
        "wavelength_lo_days": wavelength_lo,
        "wavelength_hi_days": wavelength_hi,
        "n_heads": int(n_heads),
        "frequencies_per_head": int(n_frequencies),
        "allocation": "global_logspace_then_interleave_across_heads",
        "wavelengths_days_global_sorted": grid.tolist(),
        "wavelengths_days_head_major": head_major.tolist(),
        "log_scale_bounds": [float(np.log(0.5)), float(np.log(2.0))],
    }
    manifest["sha256"] = canonical_sha256(manifest)
    return manifest


def write_wavelength_manifest(path: Path, manifest: dict) -> Path:
    if manifest.get("sha256") != canonical_sha256(manifest):
        raise ValueError("refusing to write an invalid wavelength manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8-sig"))
        if existing != manifest:
            raise FileExistsError(f"refusing to replace a different wavelength manifest: {path}")
        return path
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def deterministic_patient_batch(
    data: np.ndarray,
    p2i: np.ndarray,
    prefix_ids: np.ndarray,
    anchor_ages: np.ndarray,
    patient_index: int,
    *,
    dynamic_context_length: int,
    bos_token_id: int,
    device: str = "cpu",
) -> dict:
    start, length = (int(value) for value in p2i[int(patient_index)])
    retained = min(length, dynamic_context_length + 1)
    context_window_start = length - retained
    rows = np.asarray(data[start + context_window_start: start + length])
    if len(rows) < 2:
        return {}
    original_ordinals = np.arange(context_window_start, length, dtype=np.int64)
    order = np.argsort(rows[:, 1], kind="stable")
    rows = rows[order]
    original_ordinals = original_ordinals[order]
    token = torch.as_tensor(rows[:, 2].astype(np.int64) + 1, dtype=torch.long, device=device)[None]
    event_age = torch.as_tensor(rows[:, 1].astype(np.float32), dtype=torch.float32, device=device)[None]
    x, age = token[:, :-1], event_age[:, :-1]
    targets, targets_age = token[:, 1:], event_age[:, 1:]
    prefix = torch.as_tensor(prefix_ids[int(patient_index)], dtype=torch.long, device=device)[None]
    anchor = torch.as_tensor([anchor_ages[int(patient_index)]], dtype=torch.float32, device=device)
    x, age, targets, targets_age, static_mask, bos_mask, next_mask = prepend_static_context(
        x, age, targets, targets_age, prefix, anchor, bos_token_id
    )
    time_mask = exact_time_loss_mask(next_mask, age, targets_age, bos_mask)
    prefix_length = prefix.size(1)
    local_positions = np.arange(prefix_length + 1, x.size(1), dtype=np.int64)
    return {
        "x": x,
        "age": age,
        "targets": targets,
        "targets_age": targets_age,
        "static_mask": static_mask,
        "bos_mask": bos_mask,
        "next_event_mask": next_mask,
        "time_loss_mask": time_mask,
        "context_window_start": context_window_start,
        "local_positions": local_positions,
        "source_ordinals": original_ordinals[:-1],
        "target_ordinals": original_ordinals[1:],
    }


def event_time_nll_rows(
    logits: torch.Tensor,
    idx: torch.Tensor,
    age: torch.Tensor,
    targets: torch.Tensor,
    targets_age: torch.Tensor,
    attn_mask: torch.Tensor,
    *,
    ignore_tokens: Sequence[int],
    t_min: float,
    mask_ties: bool,
    next_event_mask: torch.Tensor,
    time_loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ignored = list(ignore_tokens) + [1]
    filtered = logits.clone()
    filtered[..., ignored] = -torch.inf
    targets_flat = targets.reshape(-1)
    valid = targets_flat != -1
    for token_id in ignored:
        valid &= targets_flat != token_id
    valid &= next_event_mask.reshape(-1).bool()
    raw_lse = torch.logsumexp(filtered, dim=-1)
    log_rate = -torch.log(torch.exp(-raw_lse) + float(t_min))
    dt_eff = torch.clamp(targets_age - age, min=1.0)
    if mask_ties:
        gather_index = (
            attn_mask
            * torch.arange(idx.size(1), device=idx.device, dtype=torch.float32).view(1, 1, 1, -1)
        ).max(-1).indices.squeeze(1).squeeze(1)
        dt_eff = torch.gather(dt_eff, -1, gather_index)
    log_dt = -torch.log(dt_eff + float(t_min))
    nll = -(log_rate - torch.exp(log_rate - log_dt))
    time_valid = valid & time_loss_mask.reshape(-1).bool()
    return nll.reshape(-1), time_valid
