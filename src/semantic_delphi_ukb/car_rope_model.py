"""Continuous-age relative rotary MedTrajectory model.

This module is intentionally separate from the frozen paper_protocol_v1 models.
It is a new-method candidate and must be validated before any checkpoint or
locked result is replaced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_delphi_ukb.modern_model import (
    ModernMultitypeSemanticDelphi,
    ModernMultitypeSemanticDelphiConfig,
    RMSNorm,
    SwiGLU,
)


class MultiScaleAgeRoPE(nn.Module):
    """Learnable multi-scale rotary phases for irregular age intervals."""

    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        base: float = 10000.0,
        initial_scales: tuple[float, ...] = (0.25, 1.0, 4.0),
        initial_gate: float = 0.1,
        learn_inner_gate: bool = True,
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for rotary encoding")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be between 0 and 1")
        if not initial_scales or any(value <= 0 for value in initial_scales):
            raise ValueError("initial_scales must contain positive values")
        n_freq = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        scale_grid = torch.tensor(initial_scales, dtype=torch.float32)
        grid = torch.linspace(0, len(scale_grid) - 1, n_freq)
        lower = grid.floor().long().clamp_max(len(scale_grid) - 1)
        upper = grid.ceil().long().clamp_max(len(scale_grid) - 1)
        weight = grid - lower.float()
        interpolated = scale_grid[lower] * (1.0 - weight) + scale_grid[upper] * weight
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.log_scale = nn.Parameter(interpolated.log().repeat(n_heads, 1))
        if learn_inner_gate:
            self.gate_logit = nn.Parameter(
                torch.full((n_heads,), math.log(initial_gate / (1.0 - initial_gate)))
            )
        else:
            self.register_parameter("gate_logit", None)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def _phase(self, age_days: torch.Tensor) -> torch.Tensor:
        if age_days.ndim == 2:
            age_years = age_days.float()[:, None, :, None]
        elif age_days.ndim == 3:
            age_years = age_days.float()[..., None]
        else:
            raise ValueError("age_days must have shape [B,T] or [B,H,T]")
        scales = self.log_scale.exp()[None, :, None, :]
        phase = age_years / 365.25 * self.inv_freq[None, None, None, :] * scales
        return torch.repeat_interleave(phase, repeats=2, dim=-1)

    @staticmethod
    def _rotate(values: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        left = values[..., 0::2]
        right = values[..., 1::2]
        rotated = torch.stack((-right, left), dim=-1).flatten(-2)
        return values * phase.cos() + rotated * phase.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_age_days: torch.Tensor,
        k_age_days: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape or q.ndim != 4 or q.shape[1] != self.n_heads or q.shape[-1] != self.head_dim:
            raise ValueError("q and k must have shape [B,n_heads,T,head_dim]")
        k_age_days = q_age_days if k_age_days is None else k_age_days
        # Scale the phase itself instead of interpolating with the identity.
        # A full orthogonal rotation preserves q-k dot products under a common
        # translation of the age origin; vector interpolation does not.
        phase_q = self._phase(q_age_days).to(dtype=q.dtype)
        phase_k = self._phase(k_age_days).to(dtype=k.dtype)
        if self.gate_logit is not None:
            gate = self.gate_logit.sigmoid().view(1, self.n_heads, 1, 1).to(dtype=q.dtype)
            phase_q = phase_q * gate
            phase_k = phase_k * gate.to(dtype=k.dtype)
        q_rot = self._rotate(q, phase_q)
        k_rot = self._rotate(k, phase_k)
        return q_rot, k_rot


class AdditiveAgeRoPEV22(nn.Module):
    """Track R v2.2 head-specific clinical-time RoPE with bounded frequencies."""

    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        wavelengths_days: tuple[float, ...],
        max_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for rotary encoding")
        n_freq = head_dim // 2
        expected = n_heads * n_freq
        if len(wavelengths_days) != expected:
            raise ValueError(
                f"additive v2.2 requires {expected} head-specific wavelengths, got {len(wavelengths_days)}"
            )
        if any(value <= 0 for value in wavelengths_days):
            raise ValueError("v2.2 wavelengths must be positive")
        if max_scale <= 1.0:
            raise ValueError("v2.2 max_scale must exceed 1")
        wavelengths = torch.tensor(wavelengths_days, dtype=torch.float32).view(n_heads, n_freq)
        self.register_buffer("base_inv_freq_days", 2.0 * math.pi / wavelengths)
        self.log_scale_raw = nn.Parameter(torch.zeros(n_heads, n_freq))
        self.log_scale_bound = math.log(float(max_scale))
        self.n_heads = n_heads
        self.head_dim = head_dim

    def bounded_log_scale(self) -> torch.Tensor:
        return self.log_scale_bound * torch.tanh(self.log_scale_raw)

    def scale_factor(self) -> torch.Tensor:
        return self.bounded_log_scale().exp()

    def effective_wavelengths_days(self) -> torch.Tensor:
        return (2.0 * math.pi / self.base_inv_freq_days) / self.scale_factor()

    def _phase(self, age_days: torch.Tensor) -> torch.Tensor:
        if age_days.ndim != 2:
            raise ValueError("v2.2 attention ages must have shape [batch,sequence]")
        frequency = self.base_inv_freq_days * self.scale_factor()
        centered_age = age_days.float() - age_days.float()[:, :1]
        phase = centered_age[:, None, :, None] * frequency[None, :, None, :]
        return torch.repeat_interleave(phase, repeats=2, dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, age_days: torch.Tensor):
        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("q and k must have shape [batch,heads,sequence,head_dim]")
        if q.shape[1] != self.n_heads or q.shape[-1] != self.head_dim:
            raise ValueError("q/k shape does not match the v2.2 RoPE configuration")
        phase = self._phase(age_days)
        return (
            MultiScaleAgeRoPE._rotate(q, phase.to(dtype=q.dtype)),
            MultiScaleAgeRoPE._rotate(k, phase.to(dtype=k.dtype)),
        )


class CARoPECausalSelfAttention(nn.Module):
    def __init__(self, config: "CARoPEConfig") -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.additive_rope_v2_2 = None
        self.capture_additive_diagnostics = False
        self.last_additive_relative_values = None
        if config.age_rope_variant == "additive_v2_2":
            self.additive_rope_v2_2 = AdditiveAgeRoPEV22(
                self.head_dim,
                config.n_head,
                config.rope_wavelengths_days,
                max_scale=config.rope_max_scale,
            )
            self.additive_rope_v2_2.log_scale_raw.requires_grad_(config.use_age_rope)
        elif config.age_rope_variant != "legacy":
            raise ValueError("age_rope_variant must be legacy or additive_v2_2")
        self.rope = (
            MultiScaleAgeRoPE(
                self.head_dim,
                config.n_head,
                base=config.rope_base,
                initial_scales=config.rope_scales,
                initial_gate=config.rope_initial_gate,
            )
            if config.use_age_rope and config.age_rope_variant == "legacy"
            else None
        )
        self.residual_rope = None
        self.residual_gate_logit = None
        self.residual_alpha_override: Optional[float] = None
        if config.residual_rope_mode != "none":
            if config.use_age_rope:
                raise ValueError("residual RoPE cannot be combined with the legacy inner RoPE")
            if not config.use_age_encoding:
                raise ValueError("residual RoPE requires the A0 absolute-age Sin/Cos base")
            self.residual_rope = MultiScaleAgeRoPE(
                self.head_dim,
                config.n_head,
                base=config.rope_base,
                initial_scales=config.rope_scales,
                initial_gate=config.rope_initial_gate,
                learn_inner_gate=False,
            )
            if config.residual_rope_mode == "learned":
                alpha = float(config.residual_rope_initial_alpha)
                if not 0.0 < alpha < 1.0:
                    raise ValueError("residual_rope_initial_alpha must be between 0 and 1")
                self.residual_gate_logit = nn.Parameter(
                    torch.full((config.n_head,), math.log(alpha / (1.0 - alpha)))
                )
            elif config.residual_rope_mode != "fixed":
                raise ValueError("residual_rope_mode must be none, learned, or fixed")
            if not 0.0 <= float(config.residual_rope_fixed_alpha) <= 1.0:
                raise ValueError("residual_rope_fixed_alpha must be in [0,1]")

    def residual_alpha(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.residual_rope is None:
            raise RuntimeError("residual alpha requested for a non-residual attention layer")
        if self.residual_alpha_override is not None:
            value = self.residual_alpha_override
            return torch.full((self.n_head,), value, dtype=dtype, device=device)
        if self.residual_gate_logit is not None:
            return self.residual_gate_logit.sigmoid().to(dtype=dtype, device=device)
        return torch.full(
            (self.n_head,),
            float(self.config.residual_rope_fixed_alpha),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _masked_attention(logits: torch.Tensor, attn_mask: torch.Tensor, dropout: nn.Module) -> torch.Tensor:
        logits = logits.masked_fill(attn_mask == 0, float("-inf"))
        return dropout(F.softmax(logits, dim=-1))

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        reshape = lambda value: value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        return reshape(q), reshape(k), reshape(v)

    def _output(self, attention: torch.Tensor, values: torch.Tensor, channels: int) -> torch.Tensor:
        batch_size, _, seq_len, _ = values.shape
        y = (attention @ values).transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))

    def additive_pair_logits(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        age: torch.Tensor,
        clinical_dynamic_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.additive_rope_v2_2 is None:
            raise RuntimeError("additive pair logits requested for a non-v2.2 attention layer")
        scale = 1.0 / math.sqrt(self.head_dim)
        base_logits = (q @ k.transpose(-2, -1)) * scale
        rope_q, rope_k = self.additive_rope_v2_2(q, k, age)
        rope_logits = (rope_q @ rope_k.transpose(-2, -1)) * scale
        pair_mask = clinical_dynamic_mask[:, None, :, None] & clinical_dynamic_mask[:, None, None, :]
        return base_logits, rope_logits, torch.where(pair_mask, rope_logits, base_logits), pair_mask

    def forward(
        self,
        x: torch.Tensor,
        age: torch.Tensor,
        attn_mask: torch.Tensor,
        *,
        clean_x: Optional[torch.Tensor] = None,
        clinical_dynamic_mask: Optional[torch.Tensor] = None,
        phase_anchor_age_days: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, channels = x.shape
        q, k, v = self._project(x)
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.residual_rope is not None:
            if clean_x is None or clinical_dynamic_mask is None or phase_anchor_age_days is None:
                raise ValueError("residual RoPE requires clean_x, clinical_dynamic_mask, and phase anchor")
            if clean_x.shape != x.shape or clinical_dynamic_mask.shape != x.shape[:2]:
                raise ValueError("residual RoPE clean stream or dynamic mask shape mismatch")
            if phase_anchor_age_days.shape != (batch_size,):
                raise ValueError("phase_anchor_age_days must have shape [batch]")

            clean_q, clean_k, clean_v = self._project(clean_x)
            base_logits = (q @ k.transpose(-2, -1)) * scale
            centered_age = age - phase_anchor_age_days[:, None].to(dtype=age.dtype)
            rope_q, rope_k = self.residual_rope(clean_q, clean_k, centered_age)
            rope_logits = (rope_q @ rope_k.transpose(-2, -1)) * scale
            pair_mask = (
                clinical_dynamic_mask[:, None, :, None]
                & clinical_dynamic_mask[:, None, None, :]
            )
            alpha = self.residual_alpha(dtype=base_logits.dtype, device=base_logits.device).view(
                1, self.n_head, 1, 1
            )
            logits = torch.where(pair_mask, base_logits + alpha * (rope_logits - base_logits), base_logits)
            att = self._masked_attention(logits, attn_mask, self.attn_dropout)

            clean_logits = (clean_q @ clean_k.transpose(-2, -1)) * scale
            clean_att = self._masked_attention(clean_logits, attn_mask, self.attn_dropout)
            return self._output(att, v, channels), att, self._output(clean_att, clean_v, channels)

        if self.additive_rope_v2_2 is not None and self.config.use_age_rope:
            if clinical_dynamic_mask is None:
                raise ValueError("additive v2.2 RoPE requires the explicit clinical dynamic mask")
            base_logits, rope_logits, logits, pair_mask = self.additive_pair_logits(
                q, k, age, clinical_dynamic_mask
            )
            if self.capture_additive_diagnostics:
                eligible = pair_mask & attn_mask.bool()
                relative = (rope_logits - base_logits).abs() / (base_logits.abs() + 1e-8)
                self.last_additive_relative_values = relative[eligible.expand_as(relative)].detach().cpu()
            att = self._masked_attention(logits, attn_mask, self.attn_dropout)
            return self._output(att, v, channels), att, None

        if self.rope is not None:
            q, k = self.rope(q, k, age)
        att = self._masked_attention((q @ k.transpose(-2, -1)) * scale, attn_mask, self.attn_dropout)
        return self._output(att, v, channels), att, None


class CARoPEBlock(nn.Module):
    def __init__(self, config: "CARoPEConfig") -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = CARoPECausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        age: torch.Tensor,
        attn_mask: torch.Tensor,
        *,
        clean_x: Optional[torch.Tensor] = None,
        clinical_dynamic_mask: Optional[torch.Tensor] = None,
        phase_anchor_age_days: Optional[torch.Tensor] = None,
    ):
        normalized_clean = self.norm_1(clean_x) if clean_x is not None else None
        y, att, clean_y = self.attn(
            self.norm_1(x),
            age,
            attn_mask,
            clean_x=normalized_clean,
            clinical_dynamic_mask=clinical_dynamic_mask,
            phase_anchor_age_days=phase_anchor_age_days,
        )
        x = x + y
        x = x + self.mlp(self.norm_2(x))
        if clean_x is not None:
            if clean_y is None:
                raise RuntimeError("residual attention did not return the clean-stream update")
            clean_x = clean_x + clean_y
            clean_x = clean_x + self.mlp(self.norm_2(clean_x))
        return x, att, clean_x


@dataclass
class CARoPEConfig(ModernMultitypeSemanticDelphiConfig):
    """Configuration for the new method; defaults intentionally differ from Delphi-compatible P0."""

    num_tte_tasks: int = 10
    num_horizons: int = 2
    tte_loss_weight: float = 0.1
    horizon_risk_loss_weight: float = 1.0
    monotonic_horizon_risk: bool = True
    use_age_encoding: bool = False
    use_age_rope: bool = True
    age_rope_variant: str = "legacy"
    rope_wavelengths_days: tuple[float, ...] = ()
    rope_max_scale: float = 2.0
    rope_base: float = 10000.0
    rope_scales: tuple[float, ...] = (0.25, 1.0, 4.0)
    rope_initial_gate: float = 0.1
    time_gap_loss_weight: float = 0.2
    horizon_years: tuple[float, ...] = (1.0, 5.0)
    use_relative_horizon_query: bool = True
    static_fusion_stage: str = "pre_transformer"
    residual_rope_mode: str = "none"
    residual_rope_initial_alpha: float = 0.01
    residual_rope_fixed_alpha: float = 1.0
    residual_rope_phase_origin: str = "recruitment_age"


class CARoPEHorizonMedTrajectory(ModernMultitypeSemanticDelphi):
    """CARoPE trunk plus target-age rotated horizon queries and time-gap loss."""

    def __init__(self, config: CARoPEConfig, pretrained_token_embeddings=None) -> None:
        if config.residual_rope_mode not in ("none", "learned", "fixed"):
            raise ValueError("residual_rope_mode must be none, learned, or fixed")
        if config.residual_rope_mode != "none" and config.residual_rope_phase_origin != "recruitment_age":
            raise ValueError("residual RoPE phase origin must be recruitment_age")
        super().__init__(config, pretrained_token_embeddings=pretrained_token_embeddings)
        self.config = config
        self.transformer.h = nn.ModuleList([CARoPEBlock(config) for _ in range(config.n_layer)])
        for block in self.transformer.h:
            block.apply(self._init_weights)
        self.tte_head = nn.Linear(config.n_embd, config.num_tte_tasks)
        self.time_gap_head = nn.Linear(config.n_embd, 1)
        if config.use_relative_horizon_query:
            self.horizon_query = nn.Parameter(torch.empty(config.num_horizons, config.n_embd))
            self.horizon_feature_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.horizon_task_head = nn.Linear(config.n_embd, config.num_tte_tasks)
            self.future_rope = MultiScaleAgeRoPE(
                config.n_embd,
                config.num_horizons,
                base=config.rope_base,
                initial_scales=config.rope_scales,
                initial_gate=config.rope_initial_gate,
            )
            for module in (self.horizon_feature_proj, self.horizon_task_head):
                module.apply(self._init_weights)
            nn.init.normal_(self.horizon_query, mean=0.0, std=0.02)
        else:
            self.horizon_risk_head = nn.Linear(
                config.n_embd, config.num_horizons * config.num_tte_tasks, bias=True
            )
            self.horizon_risk_head.apply(self._init_weights)
        for module in (self.tte_head, self.time_gap_head):
            module.apply(self._init_weights)

    def set_residual_rope_alpha_override(self, value: Optional[float]) -> None:
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError("residual RoPE alpha override must be in [0,1]")
        residual_layers = 0
        for block in self.transformer.h:
            if block.attn.residual_rope is not None:
                block.attn.residual_alpha_override = None if value is None else float(value)
                residual_layers += 1
        if value is not None and residual_layers == 0:
            raise ValueError("alpha override requires a residual RoPE checkpoint")

    def residual_rope_diagnostics(self) -> dict:
        layers = []
        for layer_index, block in enumerate(self.transformer.h):
            attention = block.attn
            if attention.residual_rope is None:
                continue
            alpha = attention.residual_alpha(
                dtype=torch.float32,
                device=attention.residual_rope.log_scale.device,
            ).detach().cpu()
            scales = attention.residual_rope.log_scale.exp().detach().cpu()
            layers.append({
                "layer": layer_index,
                "alpha_by_head": alpha.tolist(),
                "scale_by_head_and_frequency": scales.tolist(),
                "alpha_mean": float(alpha.mean()),
                "scale_mean": float(scales.mean()),
            })
        return {
            "mode": self.config.residual_rope_mode,
            "phase_origin": self.config.residual_rope_phase_origin,
            "inner_gate": False if layers else None,
            "alpha_override": next(
                (
                    block.attn.residual_alpha_override
                    for block in self.transformer.h
                    if block.attn.residual_rope is not None
                ),
                None,
            ),
            "layers": layers,
        }

    def additive_rope_v2_2_diagnostics(self) -> dict:
        layers = []
        all_saturated = []
        for layer_index, block in enumerate(self.transformer.h):
            rope = block.attn.additive_rope_v2_2
            if rope is None:
                continue
            scale = rope.scale_factor().detach().cpu()
            wavelengths = rope.effective_wavelengths_days().detach().cpu()
            saturated = (scale <= 0.505) | (scale >= 1.98)
            all_saturated.append(saturated.reshape(-1))
            layers.append({
                "layer": layer_index,
                "scale_factor_by_head_and_frequency": scale.tolist(),
                "effective_wavelengths_days": wavelengths.tolist(),
                "saturated_fraction": float(saturated.float().mean()),
            })
        return {
            "variant": self.config.age_rope_variant,
            "enabled": bool(self.config.use_age_rope),
            "scale_bounds": [0.5, float(self.config.rope_max_scale)],
            "saturation_bounds": [0.505, 1.98],
            "saturated_fraction": (
                float(torch.cat(all_saturated).float().mean()) if all_saturated else None
            ),
            "layers": layers,
        }

    def set_additive_diagnostics_capture(self, enabled: bool) -> None:
        for block in self.transformer.h:
            attention = block.attn
            attention.capture_additive_diagnostics = bool(enabled)
            if not enabled:
                attention.last_additive_relative_values = None

    def build_track_r_attention_mask(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        targets_age: Optional[torch.Tensor],
        static_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_mask = self.build_attention_mask(idx, age, targets_age)
        if static_token_mask is None:
            return attn_mask
        from semantic_delphi_ukb.track_r_contract import apply_static_temporal_visibility

        attn_mask = apply_static_temporal_visibility(attn_mask, age, static_token_mask)
        valid = (idx > 0).view(idx.size(0), 1, 1, idx.size(1)) & (
            idx > 0
        ).view(idx.size(0), 1, idx.size(1), 1)
        diag = torch.eye(idx.size(1), device=idx.device, dtype=torch.bool).view(
            1, 1, idx.size(1), idx.size(1)
        )
        return attn_mask | (valid & diag)

    def _next_event_loss(
        self,
        logits,
        idx,
        age,
        targets,
        targets_age,
        attn_mask,
        validation_loss_mode,
        next_event_mask=None,
        time_loss_mask=None,
    ):
        if targets is None:
            return None
        if targets_age is None:
            raise ValueError("targets_age is required when targets are provided")
        ignored = list(self.config.ignore_tokens)
        if validation_loss_mode:
            ignored.append(1)
            logits = logits.clone()
            logits[..., ignored] = -torch.inf
        targets_flat = targets.reshape(-1)
        valid = targets_flat != -1
        for token_id in ignored:
            valid &= targets_flat != token_id
        if next_event_mask is not None:
            if next_event_mask.shape != targets.shape:
                raise ValueError("next_event_mask must match targets")
            valid &= next_event_mask.reshape(-1).bool()
        if not bool(valid.any()):
            zero = logits.sum() * 0.0
            return {"loss_ce": zero, "loss_dt": zero, "loss": zero}
        loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[valid], targets_flat[valid], ignore_index=-1)
        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + self.config.t_min)
        dt = torch.clamp(targets_age - age, min=1.0)
        if self.config.mask_ties:
            gather_index = (
                attn_mask
                * torch.arange(0, idx.size(1), device=idx.device, dtype=torch.float32).view(1, 1, 1, -1)
            ).max(-1).indices.squeeze(1).squeeze(1)
            dt = torch.gather(dt, -1, gather_index)
        ldt = -torch.log(dt + self.config.t_min).view(-1)
        loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
        time_valid = valid
        if time_loss_mask is not None:
            if time_loss_mask.shape != targets.shape:
                raise ValueError("time_loss_mask must match targets")
            time_valid = time_valid & time_loss_mask.reshape(-1).bool()
        loss_dt = loss_dt[time_valid].mean() if bool(time_valid.any()) else logits.sum() * 0.0
        return {"loss_ce": loss_ce, "loss_dt": loss_dt, "loss": loss_ce + loss_dt}

    @staticmethod
    def _tte_loss(logits, event, duration, mask):
        hazard = F.softplus(logits) + 1e-6
        duration = duration.clamp_min(1.0 / 365.25)
        nll = hazard * duration - event * torch.log(hazard)
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)

    def _horizon_logits(self, hidden: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        if not self.config.use_relative_horizon_query:
            raw = self.horizon_risk_head(hidden).view(
                hidden.size(0), hidden.size(1), self.config.num_horizons, self.config.num_tte_tasks
            )
            return self._monotonic_horizon_logits(raw)
        batch_size, seq_len, _ = hidden.shape
        horizon_days = torch.as_tensor(self.config.horizon_years, device=hidden.device, dtype=age.dtype) * 365.25
        query = self.horizon_query[:, None, :].expand(self.config.num_horizons, seq_len, -1)
        query = query.unsqueeze(0).expand(batch_size, -1, -1, -1)
        key = hidden.unsqueeze(1).expand(-1, self.config.num_horizons, -1, -1)
        current_age = age[:, None, :].expand(-1, self.config.num_horizons, -1)
        target_age = current_age + horizon_days[None, :, None]
        query, key = self.future_rope(query, key, target_age, current_age)
        query_even, query_odd = query[..., 0::2], query[..., 1::2]
        key_even, key_odd = key[..., 0::2], key[..., 1::2]
        pair_dot = query_even * key_even + query_odd * key_odd
        pair_cross = query_even * key_odd - query_odd * key_even
        relative = torch.stack((pair_dot, pair_cross), dim=-1).flatten(-2)
        base = hidden.unsqueeze(1).expand(-1, self.config.num_horizons, -1, -1)
        features = self.static_norm(base + self.horizon_feature_proj(relative))
        raw = self.horizon_task_head(features).permute(0, 2, 1, 3)
        return self._monotonic_horizon_logits(raw)

    def _monotonic_horizon_logits(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.config.monotonic_horizon_risk or raw.size(2) <= 1:
            return raw
        first = raw[:, :, :1, :]
        increments = F.softplus(raw[:, :, 1:, :])
        return torch.cat([first, first + torch.cumsum(increments, dim=2)], dim=2)

    def forward(
        self,
        idx,
        age,
        static_features=None,
        targets=None,
        targets_age=None,
        validation_loss_mode: bool = False,
        tte_event=None,
        tte_duration=None,
        tte_mask=None,
        horizon_event=None,
        horizon_mask=None,
        static_token_mask=None,
        next_event_mask=None,
        time_loss_mask=None,
        static_feature_mask=None,
        bos_token_mask=None,
    ):
        token_x = self.token_input_proj(self.transformer.token_drop(self.transformer.wte(idx)))
        clean_x = token_x if self.config.residual_rope_mode != "none" else None
        x = self.apply_age_encoding(token_x, age)
        static_bias = None
        if static_features is not None:
            if self.static_encoder is None:
                raise ValueError("static_features were supplied to a model with static_dim=0")
            static_bias = self.static_norm(self.static_encoder(static_features)).unsqueeze(1)
            if static_feature_mask is not None:
                if static_feature_mask.shape != idx.shape:
                    raise ValueError("static_feature_mask must match token_ids")
                static_bias = static_bias * static_feature_mask.to(static_bias.dtype).unsqueeze(-1)
            if self.config.static_fusion_stage not in ("pre_transformer", "post_transformer"):
                raise ValueError("static_fusion_stage must be pre_transformer or post_transformer")
        if static_bias is not None and self.config.static_fusion_stage == "pre_transformer":
            x = x + static_bias
        x = self.transformer.drop(x)
        if clean_x is not None:
            clean_x = self.transformer.drop(clean_x)
        attn_mask = self.build_track_r_attention_mask(
            idx,
            age,
            targets_age if targets is not None else None,
            static_token_mask,
        )
        clinical_dynamic_mask = None
        phase_anchor_age_days = None
        needs_dynamic_mask = clean_x is not None or (
            self.config.age_rope_variant == "additive_v2_2" and self.config.use_age_rope
        )
        if needs_dynamic_mask:
            if static_token_mask is None or bos_token_mask is None:
                raise ValueError("clinical RoPE requires explicit static and BOS token masks")
            if static_token_mask.shape != idx.shape or bos_token_mask.shape != idx.shape:
                raise ValueError("static and BOS token masks must match token_ids")
            clinical_dynamic_mask = (idx > 1) & ~static_token_mask & ~bos_token_mask
        if clean_x is not None:
            static_count = static_token_mask.sum(dim=1)
            if bool((static_count == 0).any()):
                raise ValueError("recruitment-age phase origin requires the static prefix")
            anchor_sum = (age * static_token_mask.to(age.dtype)).sum(dim=1)
            phase_anchor_age_days = anchor_sum / static_count.to(age.dtype)
            static_age_delta = (age - phase_anchor_age_days[:, None]).abs()
            if bool((static_age_delta.masked_select(static_token_mask) > 0.25).any()):
                raise ValueError("static prefix ages disagree with the recruitment-age phase origin")
        attentions = []
        for block in self.transformer.h:
            x, att, clean_x = block(
                x,
                age,
                attn_mask,
                clean_x=clean_x,
                clinical_dynamic_mask=clinical_dynamic_mask,
                phase_anchor_age_days=phase_anchor_age_days,
            )
            attentions.append(att)
        x = self.transformer.norm_f(x)
        if static_bias is not None and self.config.static_fusion_stage == "post_transformer":
            x = x + static_bias
        logits = self.lm_head(x)
        if self.static_to_logits is not None and static_bias is not None:
            logits = logits + self.static_to_logits(static_bias).expand_as(logits)
        parts = self._next_event_loss(
            logits,
            idx,
            age,
            targets,
            targets_age,
            attn_mask,
            validation_loss_mode,
            next_event_mask=next_event_mask,
            time_loss_mask=time_loss_mask,
        )
        tte_logits = self.tte_head(x)
        if tte_event is not None and tte_duration is not None and tte_mask is not None:
            loss_tte = self._tte_loss(tte_logits, tte_event, tte_duration, tte_mask)
            if parts is None:
                parts = {"loss_tte": loss_tte, "loss": self.config.tte_loss_weight * loss_tte}
            else:
                parts["loss_tte"] = loss_tte
                parts["loss"] = parts["loss"] + self.config.tte_loss_weight * loss_tte
        risk_logits = self._horizon_logits(x, age)
        gap_logits = self.time_gap_head(x).squeeze(-1)
        if targets is not None and targets_age is not None:
            gap_target = torch.log1p(torch.clamp(targets_age - age, min=0.0) / 365.25)
            gap_mask = (targets > 1) & (targets_age > age) & (targets_age > -1000)
            if time_loss_mask is not None:
                gap_mask &= time_loss_mask.bool()
            if bool(gap_mask.any()):
                gap_loss = F.smooth_l1_loss(gap_logits[gap_mask], gap_target[gap_mask])
                if parts is None:
                    parts = {"loss": self.config.time_gap_loss_weight * gap_loss}
                else:
                    parts["loss_gap"] = gap_loss
                    parts["loss"] = parts["loss"] + self.config.time_gap_loss_weight * gap_loss
        if horizon_event is not None and horizon_mask is not None:
            risk_loss = F.binary_cross_entropy_with_logits(risk_logits, horizon_event, reduction="none")
            risk_loss = (risk_loss * horizon_mask).sum() / horizon_mask.sum().clamp_min(1.0)
            if parts is None:
                parts = {"loss": self.config.horizon_risk_loss_weight * risk_loss}
            else:
                parts["loss_horizon"] = risk_loss
                parts["loss"] = parts["loss"] + self.config.horizon_risk_loss_weight * risk_loss
        return logits, parts, torch.stack(attentions), tte_logits, risk_logits
