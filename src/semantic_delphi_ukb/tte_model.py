from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_delphi_ukb.modern_model import ModernMultitypeSemanticDelphi, ModernMultitypeSemanticDelphiConfig


@dataclass
class TTEConfig(ModernMultitypeSemanticDelphiConfig):
    num_tte_tasks: int = 10
    tte_loss_weight: float = 0.1
    tte_horizon_years: float = 10.0


@dataclass
class HorizonRiskConfig(TTEConfig):
    num_horizons: int = 2
    horizon_risk_loss_weight: float = 1.0
    monotonic_horizon_risk: bool = False


@dataclass
class SurvivalHorizonConfig(HorizonRiskConfig):
    survival_bins: int = 10
    survival_bin_years: float = 1.0
    survival_loss_weight: float = 1.0
    survival_pos_weight: float = 1.0
    survival_focal_gamma: float = 0.0


@dataclass
class FutureDiseaseSetConfig(HorizonRiskConfig):
    future_set_loss_weight: float = 1.0
    future_set_pos_weight: float = 1.0
    future_set_dice_weight: float = 0.0


class TTEMultitaskDelphi(ModernMultitypeSemanticDelphi):
    def __init__(self, config: TTEConfig, pretrained_token_embeddings: Optional[np.ndarray] = None):
        super().__init__(config, pretrained_token_embeddings=pretrained_token_embeddings)
        self.tte_head = nn.Linear(config.n_embd, config.num_tte_tasks, bias=True)
        self._init_weights(self.tte_head)

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
    ):
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = self.apply_age_encoding(x, age)

        static_bias = None
        if static_features is not None:
            static_bias = self.static_norm(self.static_encoder(static_features)).unsqueeze(1)
            if self.config.fusion_mode == "prepend":
                x = torch.cat([static_bias, x[:, 1:, :]], dim=1)
            else:
                x = x + static_bias
        x = self.transformer.drop(x)

        attn_mask = self.build_attention_mask(idx, age, targets_age if targets is not None else None)
        att = []
        for block in self.transformer.h:
            x, a = block(x, age, attn_mask)
            att.append(a)
        x = self.transformer.norm_f(x)
        logits = self.lm_head(x)
        if self.static_to_logits is not None and static_bias is not None:
            logits = logits + self.static_to_logits(static_bias).expand_as(logits)

        loss = self._next_event_loss(logits, idx, age, targets, targets_age, attn_mask, validation_loss_mode)
        tte_logits = self.tte_head(x)
        if tte_event is not None and tte_duration is not None and tte_mask is not None:
            loss_tte = self._tte_loss(tte_logits, tte_event, tte_duration, tte_mask)
            if loss is None:
                loss = {"loss_tte": loss_tte}
            else:
                loss["loss_tte"] = loss_tte
                loss["loss"] = loss["loss_ce"] + loss["loss_dt"] + self.config.tte_loss_weight * loss_tte
        return logits, loss, torch.stack(att), tte_logits

    def _next_event_loss(self, logits, idx, age, targets, targets_age, attn_mask, validation_loss_mode):
        if targets is None:
            return None
        ignored_tokens = self.config.ignore_tokens.copy()
        if validation_loss_mode:
            ignored_tokens += [1]
            logits[..., ignored_tokens] = -torch.inf
        targets_flat = targets.reshape(-1)
        pass_tokens = targets_flat != -1
        for token_id in ignored_tokens:
            pass_tokens &= targets_flat != token_id

        loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[pass_tokens], targets_flat[pass_tokens], ignore_index=-1)
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
        loss_dt = torch.mean(loss_dt[pass_tokens])
        return {"loss_ce": loss_ce, "loss_dt": loss_dt, "loss": loss_ce + loss_dt}

    def _tte_loss(self, logits, event, duration, mask):
        hazard = F.softplus(logits) + 1e-6
        duration = duration.clamp_min(1.0 / 365.25)
        nll = hazard * duration - event * torch.log(hazard)
        denom = mask.sum().clamp_min(1.0)
        return (nll * mask).sum() / denom


class HorizonRiskMedTrajectory(TTEMultitaskDelphi):
    """MedTrajectory/TTE with an explicit 5y/10y horizon risk-ranking head."""

    def __init__(self, config: HorizonRiskConfig, pretrained_token_embeddings: Optional[np.ndarray] = None):
        super().__init__(config, pretrained_token_embeddings=pretrained_token_embeddings)
        self.horizon_risk_head = nn.Linear(config.n_embd, config.num_horizons * config.num_tte_tasks, bias=True)
        self._init_weights(self.horizon_risk_head)

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
    ):
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = self.apply_age_encoding(x, age)

        static_bias = None
        if static_features is not None:
            static_bias = self.static_norm(self.static_encoder(static_features)).unsqueeze(1)
            if self.config.fusion_mode == "prepend":
                x = torch.cat([static_bias, x[:, 1:, :]], dim=1)
            else:
                x = x + static_bias
        x = self.transformer.drop(x)

        attn_mask = self.build_attention_mask(idx, age, targets_age if targets is not None else None)
        att = []
        for block in self.transformer.h:
            x, a = block(x, age, attn_mask)
            att.append(a)
        x = self.transformer.norm_f(x)

        logits = self.lm_head(x)
        if self.static_to_logits is not None and static_bias is not None:
            logits = logits + self.static_to_logits(static_bias).expand_as(logits)

        loss = self._next_event_loss(logits, idx, age, targets, targets_age, attn_mask, validation_loss_mode)

        tte_logits = self.tte_head(x)
        if tte_event is not None and tte_duration is not None and tte_mask is not None:
            loss_tte = self._tte_loss(tte_logits, tte_event, tte_duration, tte_mask)
            if loss is None:
                loss = {"loss_tte": loss_tte, "loss": self.config.tte_loss_weight * loss_tte}
            else:
                loss["loss_tte"] = loss_tte
                loss["loss"] = loss["loss"] + self.config.tte_loss_weight * loss_tte

        raw_risk_logits = self.horizon_risk_head(x).view(
            idx.size(0),
            idx.size(1),
            self.config.num_horizons,
            self.config.num_tte_tasks,
        )
        risk_logits = self._horizon_risk_logits(raw_risk_logits)
        if horizon_event is not None and horizon_mask is not None:
            loss_horizon = self._horizon_risk_loss(risk_logits, horizon_event, horizon_mask)
            if loss is None:
                loss = {"loss_horizon": loss_horizon, "loss": self.config.horizon_risk_loss_weight * loss_horizon}
            else:
                loss["loss_horizon"] = loss_horizon
                loss["loss"] = loss["loss"] + self.config.horizon_risk_loss_weight * loss_horizon

        return logits, loss, torch.stack(att), tte_logits, risk_logits

    def _horizon_risk_loss(self, logits, event, mask):
        raw = F.binary_cross_entropy_with_logits(logits, event, reduction="none")
        return (raw * mask).sum() / mask.sum().clamp_min(1.0)

    def _horizon_risk_logits(self, raw_logits):
        if not self.config.monotonic_horizon_risk or raw_logits.size(2) <= 1:
            return raw_logits
        first = raw_logits[:, :, :1, :]
        positive_increments = F.softplus(raw_logits[:, :, 1:, :])
        return torch.cat([first, first + torch.cumsum(positive_increments, dim=2)], dim=2)


class SurvivalHorizonMedTrajectory(TTEMultitaskDelphi):
    """Unified discrete survival head whose hazards imply 5y/10y horizon risks."""

    def __init__(self, config: SurvivalHorizonConfig, pretrained_token_embeddings: Optional[np.ndarray] = None):
        super().__init__(config, pretrained_token_embeddings=pretrained_token_embeddings)
        self.survival_head = nn.Linear(config.n_embd, config.num_tte_tasks * config.survival_bins, bias=True)
        self._init_weights(self.survival_head)

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
        survival_event=None,
        survival_mask=None,
    ):
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = self.apply_age_encoding(x, age)

        static_bias = None
        if static_features is not None:
            static_bias = self.static_norm(self.static_encoder(static_features)).unsqueeze(1)
            if self.config.fusion_mode == "prepend":
                x = torch.cat([static_bias, x[:, 1:, :]], dim=1)
            else:
                x = x + static_bias
        x = self.transformer.drop(x)

        attn_mask = self.build_attention_mask(idx, age, targets_age if targets is not None else None)
        att = []
        for block in self.transformer.h:
            x, a = block(x, age, attn_mask)
            att.append(a)
        x = self.transformer.norm_f(x)

        logits = self.lm_head(x)
        if self.static_to_logits is not None and static_bias is not None:
            logits = logits + self.static_to_logits(static_bias).expand_as(logits)

        loss = self._next_event_loss(logits, idx, age, targets, targets_age, attn_mask, validation_loss_mode)

        tte_logits = self.tte_head(x)
        if tte_event is not None and tte_duration is not None and tte_mask is not None:
            loss_tte = self._tte_loss(tte_logits, tte_event, tte_duration, tte_mask)
            if loss is None:
                loss = {"loss_tte": loss_tte, "loss": self.config.tte_loss_weight * loss_tte}
            else:
                loss["loss_tte"] = loss_tte
                loss["loss"] = loss["loss"] + self.config.tte_loss_weight * loss_tte

        survival_logits = self.survival_head(x).view(
            idx.size(0),
            idx.size(1),
            self.config.num_tte_tasks,
            self.config.survival_bins,
        )
        if survival_event is not None and survival_mask is not None:
            loss_survival = self._survival_loss(survival_logits, survival_event, survival_mask)
            if loss is None:
                loss = {"loss_survival": loss_survival, "loss": self.config.survival_loss_weight * loss_survival}
            else:
                loss["loss_survival"] = loss_survival
                loss["loss"] = loss["loss"] + self.config.survival_loss_weight * loss_survival

        return logits, loss, torch.stack(att), tte_logits, survival_logits

    def horizon_risk_from_survival(self, survival_logits, horizons_years):
        hazard = torch.sigmoid(survival_logits)
        survival = torch.cumprod(1.0 - hazard.clamp(max=1.0 - 1e-6), dim=-1)
        risks = []
        for horizon_years in horizons_years:
            bins = max(1, min(self.config.survival_bins, int(math.ceil(float(horizon_years) / self.config.survival_bin_years))))
            risks.append(1.0 - survival[..., bins - 1])
        return torch.stack(risks, dim=-2)

    def _survival_loss(self, logits, event, mask):
        raw = F.binary_cross_entropy_with_logits(logits, event, reduction="none")
        if self.config.survival_pos_weight != 1.0:
            pos_weight = torch.full_like(raw, float(self.config.survival_pos_weight))
            raw = raw * torch.where(event > 0.5, pos_weight, torch.ones_like(raw))
        if self.config.survival_focal_gamma > 0.0:
            prob = torch.sigmoid(logits)
            pt = torch.where(event > 0.5, prob, 1.0 - prob)
            raw = raw * (1.0 - pt).pow(float(self.config.survival_focal_gamma))
        return (raw * mask).sum() / mask.sum().clamp_min(1.0)


class FutureDiseaseSetMedTrajectory(TTEMultitaskDelphi):
    """Auxiliary future disease-set head for constraining rollout false positives."""

    def __init__(self, config: FutureDiseaseSetConfig, pretrained_token_embeddings: Optional[np.ndarray] = None):
        super().__init__(config, pretrained_token_embeddings=pretrained_token_embeddings)
        self.future_set_head = nn.Linear(config.n_embd, config.num_horizons * config.num_tte_tasks, bias=True)
        self._init_weights(self.future_set_head)

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
        future_set_event=None,
        future_set_mask=None,
    ):
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = self.apply_age_encoding(x, age)

        static_bias = None
        if static_features is not None:
            static_bias = self.static_norm(self.static_encoder(static_features)).unsqueeze(1)
            if self.config.fusion_mode == "prepend":
                x = torch.cat([static_bias, x[:, 1:, :]], dim=1)
            else:
                x = x + static_bias
        x = self.transformer.drop(x)

        attn_mask = self.build_attention_mask(idx, age, targets_age if targets is not None else None)
        att = []
        for block in self.transformer.h:
            x, a = block(x, age, attn_mask)
            att.append(a)
        x = self.transformer.norm_f(x)

        logits = self.lm_head(x)
        if self.static_to_logits is not None and static_bias is not None:
            logits = logits + self.static_to_logits(static_bias).expand_as(logits)

        loss = self._next_event_loss(logits, idx, age, targets, targets_age, attn_mask, validation_loss_mode)

        tte_logits = self.tte_head(x)
        if tte_event is not None and tte_duration is not None and tte_mask is not None:
            loss_tte = self._tte_loss(tte_logits, tte_event, tte_duration, tte_mask)
            if loss is None:
                loss = {"loss_tte": loss_tte, "loss": self.config.tte_loss_weight * loss_tte}
            else:
                loss["loss_tte"] = loss_tte
                loss["loss"] = loss["loss"] + self.config.tte_loss_weight * loss_tte

        future_logits = self.future_set_head(x).view(
            idx.size(0),
            idx.size(1),
            self.config.num_horizons,
            self.config.num_tte_tasks,
        )
        if future_set_event is not None and future_set_mask is not None:
            loss_future = self._future_set_loss(future_logits, future_set_event, future_set_mask)
            if loss is None:
                loss = {"loss_future_set": loss_future, "loss": self.config.future_set_loss_weight * loss_future}
            else:
                loss["loss_future_set"] = loss_future
                loss["loss"] = loss["loss"] + self.config.future_set_loss_weight * loss_future

        return logits, loss, torch.stack(att), tte_logits, future_logits

    def _future_set_loss(self, logits, event, mask):
        raw = F.binary_cross_entropy_with_logits(logits, event, reduction="none")
        if self.config.future_set_pos_weight != 1.0:
            raw = raw * torch.where(
                event > 0.5,
                torch.full_like(raw, float(self.config.future_set_pos_weight)),
                torch.ones_like(raw),
            )
        bce = (raw * mask).sum() / mask.sum().clamp_min(1.0)
        if self.config.future_set_dice_weight <= 0.0:
            return bce

        prob = torch.sigmoid(logits)
        intersection = (prob * event * mask).sum(dim=-1)
        denom = ((prob + event) * mask).sum(dim=-1).clamp_min(1e-6)
        dice = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6))
        active = (mask.sum(dim=-1) > 0).float()
        dice_loss = (dice * active).sum() / active.sum().clamp_min(1.0)
        return bce + float(self.config.future_set_dice_weight) * dice_loss
