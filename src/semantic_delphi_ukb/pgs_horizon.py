from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions
from semantic_delphi_ukb.multitype_batch import get_batch
from semantic_delphi_ukb.train_architecture_risk_heads import (
    binary_auc,
    build_horizon_targets,
    top_decile_stats,
)
from semantic_delphi_ukb.tte_model import HorizonRiskConfig, HorizonRiskMedTrajectory
from utils import get_p2i


@dataclass
class PredictionCache:
    base_logits: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    pgs_features: np.ndarray
    pgs_available: np.ndarray
    row_indices: np.ndarray


class PGSResidualAdapter(nn.Module):
    """Disease-specific genetic residual added after the frozen trajectory model."""

    def __init__(self, pgs_dim: int, hidden_dim: int, num_horizons: int, num_targets: int, dropout: float = 0.1):
        super().__init__()
        self.pgs_dim = int(pgs_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_horizons = int(num_horizons)
        self.num_targets = int(num_targets)
        self.encoder = nn.Sequential(
            nn.Linear(self.pgs_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(self.hidden_dim, self.num_horizons * self.num_targets)
        self.gate_logits = nn.Parameter(torch.full((self.num_targets,), -4.0))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        base_logits: torch.Tensor,
        pgs_features: torch.Tensor,
        pgs_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self.delta_head(self.encoder(pgs_features)).view(
            -1, self.num_horizons, self.num_targets
        )
        gate = torch.sigmoid(self.gate_logits).view(1, 1, self.num_targets)
        delta = delta * gate * pgs_available.view(-1, 1, 1)
        fused = torch.cummax(base_logits + delta, dim=1).values
        return fused, delta, gate.squeeze(0).squeeze(0)


def load_base_model(checkpoint_path: Path, device: str) -> tuple[HorizonRiskMedTrajectory, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HorizonRiskMedTrajectory(HorizonRiskConfig(**checkpoint["model_args"]))
    state = checkpoint["model"].copy()
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[len("_orig_mod.") :]] = state.pop(key)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def load_split(data_dir: Path, pgs_dir: Path, split: str):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy", mmap_mode="r")
    pgs = np.load(pgs_dir / f"{split}_pgs.npy", mmap_mode="r")
    available = np.load(pgs_dir / f"{split}_pgs_available.npy", mmap_mode="r")
    p2i = get_p2i(data)
    expected = len(p2i)
    if len(static) != expected or len(pgs) != expected or len(available) != expected:
        raise RuntimeError(
            f"{split} alignment mismatch: p2i={expected}, static={len(static)}, pgs={len(pgs)}, available={len(available)}"
        )
    return data, p2i, static, pgs, available.astype(bool)


@torch.no_grad()
def collect_final_context_cache(
    model: HorizonRiskMedTrajectory,
    data: np.ndarray,
    p2i: np.ndarray,
    static: np.ndarray,
    pgs: np.ndarray,
    available: np.ndarray,
    patient_disease_ages: np.ndarray,
    patient_last_ages: np.ndarray,
    horizons: Sequence[float],
    target_indices: Sequence[int],
    *,
    device: str,
    block_size: int,
    batch_size: int,
    available_only: bool,
    row_mask: np.ndarray | None = None,
) -> PredictionCache:
    selected = available.copy() if available_only else np.ones(len(p2i), dtype=bool)
    if row_mask is not None:
        if len(row_mask) != len(selected):
            raise RuntimeError(f"row mask length {len(row_mask)} does not match patient rows {len(selected)}")
        selected &= row_mask.astype(bool)
    rows = np.flatnonzero(selected)
    base_parts = []
    label_parts = []
    mask_parts = []
    pgs_parts = []
    available_parts = []
    row_parts = []
    target_index = torch.as_tensor(target_indices, dtype=torch.long, device=device)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        ix = torch.as_tensor(batch_rows, dtype=torch.long)
        x, age, y, target_age, static_features = get_batch(
            ix,
            data,
            p2i,
            static,
            select="right",
            padding="none",
            block_size=block_size,
            device=device,
            cut_batch=True,
        )
        _, _, _, _, risk_logits = model(
            x,
            age,
            static_features,
            y,
            target_age,
            validation_loss_mode=True,
        )
        keep, pos = last_prediction_positions(x, y)
        labels, mask, _ = build_horizon_targets(
            ix,
            x,
            age,
            y,
            patient_disease_ages,
            patient_last_ages,
            horizons,
            device,
        )
        batch_index = torch.arange(len(ix), device=device)
        selected_logits = risk_logits[batch_index, pos].index_select(2, target_index)
        selected_labels = labels.index_select(2, target_index)
        selected_mask = mask.index_select(2, target_index) * keep.view(-1, 1, 1)
        base_parts.append(selected_logits.cpu().numpy().astype(np.float32))
        label_parts.append(selected_labels.cpu().numpy().astype(np.float32))
        mask_parts.append(selected_mask.cpu().numpy().astype(bool))
        pgs_parts.append(np.asarray(pgs[batch_rows], dtype=np.float32))
        available_parts.append(np.asarray(available[batch_rows], dtype=bool))
        row_parts.append(batch_rows)

    return PredictionCache(
        base_logits=np.concatenate(base_parts),
        labels=np.concatenate(label_parts),
        mask=np.concatenate(mask_parts),
        pgs_features=np.concatenate(pgs_parts),
        pgs_available=np.concatenate(available_parts),
        row_indices=np.concatenate(row_parts),
    )


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(bool)
    hit_ranks = np.flatnonzero(ranked) + 1
    return float((np.arange(1, positives + 1, dtype=np.float64) / hit_ranks).mean())


def prediction_metrics(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40.0, 40.0)))
    aucs = []
    auprcs = []
    briers = []
    captures = []
    positives = 0
    observations = 0
    for horizon_index in range(logits.shape[1]):
        for target_index in range(logits.shape[2]):
            valid = mask[:, horizon_index, target_index]
            scores = probabilities[valid, horizon_index, target_index]
            outcome = labels[valid, horizon_index, target_index].astype(np.int8)
            positives += int(outcome.sum())
            observations += len(outcome)
            if len(np.unique(outcome)) == 2:
                aucs.append(binary_auc(scores, outcome))
                auprcs.append(average_precision(scores, outcome))
            if len(outcome):
                briers.append(float(np.mean((scores - outcome) ** 2)))
                capture, _, _ = top_decile_stats(scores, outcome)
                if not math.isnan(capture):
                    captures.append(capture)
    return {
        "auroc_macro": float(np.mean(aucs)) if aucs else float("nan"),
        "auprc_macro": float(np.mean(auprcs)) if auprcs else float("nan"),
        "brier_macro": float(np.mean(briers)) if briers else float("nan"),
        "top_decile_capture_macro": float(np.mean(captures)) if captures else float("nan"),
        "positives": float(positives),
        "observations": float(observations),
    }


def adapter_logits(adapter: PGSResidualAdapter, cache: PredictionCache, device: str, batch_size: int) -> np.ndarray:
    adapter.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(cache.base_logits), batch_size):
            stop = start + batch_size
            base = torch.as_tensor(cache.base_logits[start:stop], device=device)
            features = torch.as_tensor(cache.pgs_features[start:stop], device=device)
            available = torch.as_tensor(cache.pgs_available[start:stop], dtype=torch.float32, device=device)
            fused, _, _ = adapter(base, features, available)
            parts.append(fused.cpu().numpy().astype(np.float32))
    return np.concatenate(parts)
