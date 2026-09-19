from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AgeEncoding
from semantic_delphi_ukb.modern_model import ContinuousAgeRoPE


@dataclass
class ArchitectureBaselineConfig:
    block_size: int = 128
    vocab_size: int = 8424
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 64
    dropout: float = 0.1
    t_min: float = 0.1
    static_dim: int = 10
    static_hidden_dim: int = 64
    static_dropout: float = 0.05


def last_prediction_positions(x: torch.Tensor, y: torch.Tensor, ignore_tokens: tuple[int, ...] = (0, 1)) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (x > 0) & (y >= 0)
    for token in ignore_tokens:
        valid &= y != token
    indices = torch.arange(x.size(1), device=x.device).view(1, -1).expand_as(x)
    pos = torch.where(valid, indices, torch.full_like(indices, -1)).max(dim=1).values
    keep = pos >= 0
    pos = torch.clamp(pos, min=0)
    return keep, pos


def history_only_inputs(x: torch.Tensor, age: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask tokens after the final prediction point so bidirectional encoders cannot leak targets."""
    _, pos = last_prediction_positions(x, y)
    future = torch.arange(x.size(1), device=x.device).view(1, -1) > pos.view(-1, 1)
    x_in = x.masked_fill(future, 0)
    age_in = age.masked_fill(future, -10000.0)
    return x_in, age_in


class HistoryMaskBERTBaseline(nn.Module):
    """BERT-style history encoder evaluated only at the final observed context position.

    This avoids the usual bidirectional leakage problem: the encoder can see the
    whole observed history window, but targets are only the next event after that
    window, not intermediate positions with future tokens still visible.
    """

    def __init__(self, config: ArchitectureBaselineConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.wae = AgeEncoding(config)
        self.static_encoder = nn.Sequential(
            nn.Linear(config.static_dim, config.static_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.static_dropout),
            nn.Linear(config.static_hidden_dim, config.n_embd),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layer)
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, idx: torch.Tensor, age: torch.Tensor, static_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.wte(idx.clamp_min(0)) + self.wae(age.unsqueeze(-1))
        if static_features is not None:
            x = x + self.static_encoder(static_features).unsqueeze(1)
        key_padding_mask = idx <= 0
        return self.norm(self.encoder(x, src_key_padding_mask=key_padding_mask))

    def forward(self, idx: torch.Tensor, age: torch.Tensor, static_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encode(idx, age, static_features))


class RoPEBERTSelfAttention(nn.Module):
    def __init__(self, config: ArchitectureBaselineConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = ContinuousAgeRoPE(self.head_dim)

    def forward(self, x: torch.Tensor, age: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, age)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(key_padding_mask.view(batch_size, 1, 1, seq_len), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.proj(y))


class RoPEBERTBlock(nn.Module):
    def __init__(self, config: ArchitectureBaselineConfig):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.attn = RoPEBERTSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, age: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), age, key_padding_mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class HistoryMaskRoPEBERTBaseline(nn.Module):
    """BERT-style final-context encoder with continuous-age RoPE applied to self-attention q/k."""

    def __init__(self, config: ArchitectureBaselineConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.wae = AgeEncoding(config)
        self.static_encoder = nn.Sequential(
            nn.Linear(config.static_dim, config.static_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.static_dropout),
            nn.Linear(config.static_hidden_dim, config.n_embd),
        )
        self.blocks = nn.ModuleList([RoPEBERTBlock(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, idx: torch.Tensor, age: torch.Tensor, static_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.wte(idx.clamp_min(0)) + self.wae(age.unsqueeze(-1))
        if static_features is not None:
            x = x + self.static_encoder(static_features).unsqueeze(1)
        key_padding_mask = idx <= 0
        for block in self.blocks:
            x = block(x, age, key_padding_mask)
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return self.norm(x)

    def forward(self, idx: torch.Tensor, age: torch.Tensor, static_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encode(idx, age, static_features))


class MambaHistoryBaseline(nn.Module):
    """Mamba sequence baseline with the same final-context prediction protocol."""

    def __init__(self, config: ArchitectureBaselineConfig):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:  # pragma: no cover - depends on remote env
            raise RuntimeError("mamba-ssm is not installed; install it before running --model mamba.") from exc
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.wae = AgeEncoding(config)
        self.static_encoder = nn.Sequential(
            nn.Linear(config.static_dim, config.static_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.static_dropout),
            nn.Linear(config.static_hidden_dim, config.n_embd),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.n_embd),
                    Mamba(d_model=config.n_embd, d_state=16, d_conv=4, expand=2),
                    nn.Dropout(config.dropout),
                )
                for _ in range(config.n_layer)
            ]
        )
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, idx: torch.Tensor, age: torch.Tensor, static_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.wte(idx.clamp_min(0)) + self.wae(age.unsqueeze(-1))
        if static_features is not None:
            x = x + self.static_encoder(static_features).unsqueeze(1)
        x = x.masked_fill((idx <= 0).unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = x + block(x)
            x = x.masked_fill((idx <= 0).unsqueeze(-1), 0.0)
        return self.norm(x)

    def forward(self, idx: torch.Tensor, age: torch.Tensor, static_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encode(idx, age, static_features))


class HorizonRiskHead(nn.Module):
    def __init__(self, n_embd: int, num_diseases: int, num_horizons: int):
        super().__init__()
        self.head = nn.Linear(n_embd, num_diseases * num_horizons)
        self.num_diseases = num_diseases
        self.num_horizons = num_horizons
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, hidden: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_index, pos]
        logits = self.head(pooled)
        return logits.view(hidden.size(0), self.num_horizons, self.num_diseases)


def final_context_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    age: torch.Tensor,
    y: torch.Tensor,
    target_age: torch.Tensor,
    t_min: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    keep, pos = last_prediction_positions(x, y)
    if not bool(keep.any()):
        zero = logits.sum() * 0.0
        return zero, {"loss_ce": zero, "loss_dt": zero, "loss": zero}
    batch_index = torch.arange(x.size(0), device=x.device)[keep]
    pos = pos[keep]
    gathered_logits = logits[batch_index, pos]
    targets = y[batch_index, pos]
    loss_ce = F.cross_entropy(gathered_logits, targets)
    lse = torch.logsumexp(gathered_logits, dim=-1)
    lse = -torch.log(torch.exp(-lse) + t_min)
    dt = torch.clamp(target_age[batch_index, pos] - age[batch_index, pos], min=1.0)
    ldt = -torch.log(dt + t_min)
    loss_dt = -(lse - torch.exp(lse - ldt)).mean()
    loss = loss_ce + loss_dt
    return loss, {"loss_ce": loss_ce, "loss_dt": loss_dt, "loss": loss}


@torch.no_grad()
def final_context_topk(
    logits: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    topk: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    keep, pos = last_prediction_positions(x, y)
    if not bool(keep.any()):
        return {f"top{k}": float("nan") for k in topk} | {"n": 0}
    batch_index = torch.arange(x.size(0), device=x.device)[keep]
    scores = logits[batch_index, pos]
    targets = y[batch_index, pos]
    max_k = min(max(topk), scores.size(-1))
    pred = torch.topk(scores, k=max_k, dim=-1).indices
    out = {"n": int(targets.numel())}
    for k in topk:
        kk = min(k, max_k)
        out[f"top{k}"] = float((pred[:, :kk] == targets.unsqueeze(1)).any(dim=1).float().mean().item())
    return out
