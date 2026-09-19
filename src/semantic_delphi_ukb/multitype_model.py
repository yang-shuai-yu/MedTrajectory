from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from semantic_delphi_ukb.semantic_model import SemanticDelphi, SemanticDelphiConfig


@dataclass
class MultitypeSemanticDelphiConfig(SemanticDelphiConfig):
    static_dim: int = 10
    static_hidden_dim: int = 64
    static_dropout: float = 0.0
    fusion_mode: str = "residual"
    fuse_static_to_logits: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])


class MultitypeSemanticDelphi(SemanticDelphi):
    def __init__(self, config: MultitypeSemanticDelphiConfig, pretrained_token_embeddings: Optional[np.ndarray] = None):
        super().__init__(config, pretrained_token_embeddings=pretrained_token_embeddings)
        self.config = config
        self.static_encoder = nn.Sequential(
            nn.Linear(config.static_dim, config.static_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.static_dropout),
            nn.Linear(config.static_hidden_dim, config.n_embd),
        )
        self.static_ln = nn.LayerNorm(config.n_embd)
        if config.fuse_static_to_logits:
            self.static_to_logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        else:
            self.static_to_logits = None

    def forward(self, idx, age, static_features=None, targets=None, targets_age=None, validation_loss_mode: bool = False):
        device = idx.device
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb

        static_bias = None
        if static_features is not None:
            static_bias = self.static_ln(self.static_encoder(static_features)).unsqueeze(1)
            if self.config.fusion_mode == "prepend":
                x = torch.cat([static_bias, x[:, 1:, :]], dim=1)
            else:
                x = x + static_bias

        x = self.transformer.drop(x)

        attn_mask = (idx > 0).view(idx.size(0), 1, 1, idx.size(1)) * (idx > 0).view(idx.size(0), 1, idx.size(1), 1)
        attn_mask *= torch.tril(torch.ones(idx.size(1), idx.size(1), device=device))[None, None, :, :] > 0
        if targets is not None and self.config.mask_ties:
            attn_mask *= age.view(idx.size(0), 1, 1, idx.size(1)) != targets_age.view(idx.size(0), 1, idx.size(1), 1)
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(idx.size(1), device=device)) > 0
        attn_mask = attn_mask + (idx == 0).view(idx.size(0), 1, 1, idx.size(1)) * torch.diag(
            torch.ones(idx.size(1), device=device)
        ) > 0
        attn_mask *= torch.tril(torch.ones(idx.size(1), idx.size(1), device=device))[None, None, :, :] > 0

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        logits = self.lm_head(x)
        if self.static_to_logits is not None and static_bias is not None:
            logits = logits + self.static_to_logits(static_bias).expand_as(logits)

        if targets is not None:
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits[..., ignored_tokens] = -torch.inf
            targets = targets.reshape(-1)
            pass_tokens = targets != -1
            for token_id in ignored_tokens:
                pass_tokens *= targets != token_id

            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[pass_tokens], targets[pass_tokens], ignore_index=-1)

            lse = torch.logsumexp(logits, -1)
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            dt = torch.clamp(targets_age - age, min=1.0)
            if self.config.mask_ties:
                dt = torch.gather(
                    dt,
                    -1,
                    (
                        attn_mask
                        * torch.arange(0, idx.size(1), device=device, dtype=torch.float32).view(1, 1, 1, -1)
                    )
                    .max(-1)
                    .indices.squeeze(1).squeeze(1),
                )
            ldt = -torch.log(dt + self.config.t_min).view(-1)
            loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
            loss_dt = torch.mean(loss_dt[pass_tokens])
            loss = {"loss_ce": loss_ce, "loss_dt": loss_dt}
        else:
            loss = None

        return logits, loss, att
