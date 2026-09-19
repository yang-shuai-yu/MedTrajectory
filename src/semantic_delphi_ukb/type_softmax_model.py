from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import AgeEncoding
from semantic_delphi_ukb.modern_model import ModernBlock, ModernMultitypeSemanticDelphiConfig, RMSNorm


@dataclass
class TypeSoftmaxMultitypeConfig(ModernMultitypeSemanticDelphiConfig):
    event_type_count: int = 5
    type_loss_weight: float = 1.0
    token_loss_weight: float = 1.0
    dt_loss_weight: float = 1.0


class TypeSoftmaxMultitypeDelphi(nn.Module):
    """Shared trajectory trunk with event-type prediction plus within-type token softmax.

    The raw token head is still vocabulary-sized so generation and Top-K evaluation can
    use a single score table. The training cross-entropy is factorized into:
    p(event_type | history) and p(token | event_type, history).
    """

    def __init__(
        self,
        config: TypeSoftmaxMultitypeConfig,
        pretrained_token_embeddings: Optional[np.ndarray] = None,
        token_type_ids: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.config = config
        if token_type_ids is None:
            raise ValueError("token_type_ids is required for type-softmax training.")
        token_type_tensor = torch.as_tensor(token_type_ids, dtype=torch.long)
        if int(token_type_tensor.numel()) != int(config.vocab_size):
            raise ValueError(f"token_type_ids length {token_type_tensor.numel()} != vocab_size {config.vocab_size}")
        if int(token_type_tensor.max().item()) >= int(config.event_type_count):
            raise ValueError("token_type_ids contains an id >= event_type_count.")
        self.register_buffer("token_type_ids", token_type_tensor, persistent=True)
        type_token_mask = torch.zeros(config.event_type_count, config.vocab_size, dtype=torch.bool)
        for type_id in range(config.event_type_count):
            type_token_mask[type_id] = token_type_tensor == type_id
        self.register_buffer("type_token_mask", type_token_mask, persistent=True)

        if pretrained_token_embeddings is not None:
            if pretrained_token_embeddings.shape != (config.vocab_size, config.n_embd):
                raise ValueError(
                    "Pretrained matrix shape mismatch: "
                    f"matrix={pretrained_token_embeddings.shape} config={(config.vocab_size, config.n_embd)}"
                )
            wte = nn.Embedding.from_pretrained(
                torch.as_tensor(pretrained_token_embeddings, dtype=torch.float32),
                freeze=config.freeze_input_embeddings,
                padding_idx=0,
            )
        else:
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)

        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                wet=nn.Embedding(config.event_type_count, config.n_embd),
                wae=AgeEncoding(config),
                token_drop=nn.Dropout(config.token_dropout),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([ModernBlock(config) for _ in range(config.n_layer)]),
                norm_f=RMSNorm(config.n_embd),
            )
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(config.static_dim, config.static_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.static_dropout),
            nn.Linear(config.static_hidden_dim, config.n_embd),
        )
        self.static_norm = RMSNorm(config.n_embd)
        self.token_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.type_head = nn.Linear(config.n_embd, config.event_type_count, bias=False)
        self.apply(self._init_weights)
        if pretrained_token_embeddings is not None:
            with torch.no_grad():
                self.transformer.wte.weight.copy_(torch.as_tensor(pretrained_token_embeddings, dtype=torch.float32))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def build_attention_mask(self, idx: torch.Tensor, age: torch.Tensor, targets_age: Optional[torch.Tensor]) -> torch.Tensor:
        device = idx.device
        batch_size, seq_len = idx.shape
        tril = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        attn_mask = (idx > 0).view(batch_size, 1, 1, seq_len) & (idx > 0).view(batch_size, 1, seq_len, 1)
        attn_mask &= tril
        if targets_age is not None and self.config.mask_ties:
            attn_mask &= age.view(batch_size, 1, 1, seq_len) != targets_age.view(batch_size, 1, seq_len, 1)
            diag = torch.diag(torch.ones(seq_len, device=device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
            attn_mask |= (attn_mask.sum(-1, keepdim=True) == 0) & diag
        diag = torch.diag(torch.ones(seq_len, device=device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        attn_mask |= (idx == 0).view(batch_size, 1, 1, seq_len) & diag
        attn_mask &= tril
        return attn_mask

    def combine_logits(self, token_logits: torch.Tensor, type_logits: torch.Tensor) -> torch.Tensor:
        valid_type_tokens = self.token_type_ids >= 0
        safe_type_ids = self.token_type_ids.clamp(min=0)
        type_bias = torch.gather(
            type_logits,
            -1,
            safe_type_ids.view(1, 1, -1).expand(token_logits.size(0), token_logits.size(1), -1),
        )
        combined = token_logits + type_bias
        combined[..., ~valid_type_tokens] = -torch.inf
        return combined

    def forward(
        self,
        idx,
        age,
        static_features=None,
        targets=None,
        targets_age=None,
        validation_loss_mode: bool = False,
    ):
        tok_emb = self.transformer.wte(idx)
        input_type_ids = self.token_type_ids[idx].clamp(min=0)
        type_emb = self.transformer.wet(input_type_ids)
        type_emb = type_emb.masked_fill((idx == 0).unsqueeze(-1), 0.0)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + type_emb + age_emb
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

        token_logits = self.token_head(x)
        type_logits = self.type_head(x)
        combined_logits = self.combine_logits(token_logits, type_logits)

        loss = None
        if targets is not None:
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                combined_logits[..., ignored_tokens] = -torch.inf
            targets_flat = targets.reshape(-1)
            pass_tokens = targets_flat != -1
            for token_id in ignored_tokens:
                pass_tokens &= targets_flat != token_id
            valid_targets = targets_flat >= 0
            target_types = torch.full_like(targets_flat, -1)
            target_types[valid_targets] = self.token_type_ids[targets_flat[valid_targets]]
            pass_tokens &= target_types >= 0
            n_tokens = pass_tokens.sum()

            token_logits_flat = token_logits.reshape(-1, token_logits.size(-1))[pass_tokens]
            targets_valid = targets_flat[pass_tokens]
            target_types_valid = target_types[pass_tokens]
            candidate_mask = self.type_token_mask[target_types_valid]
            token_logits_flat = token_logits_flat.masked_fill(~candidate_mask, -torch.inf)
            loss_ce = F.cross_entropy(token_logits_flat, targets_valid, ignore_index=-1)
            loss_type = F.cross_entropy(type_logits.reshape(-1, type_logits.size(-1))[pass_tokens], target_types_valid)

            lse = torch.logsumexp(combined_logits, -1)
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
            total = (
                self.config.token_loss_weight * loss_ce
                + self.config.type_loss_weight * loss_type
                + self.config.dt_loss_weight * loss_dt
            )
            loss = {"loss": total, "loss_ce": loss_ce, "loss_type": loss_type, "loss_dt": loss_dt, "n_tokens": n_tokens}
        return combined_logits, loss, torch.stack(att)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        import inspect

        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and not name.endswith("wte.weight"):
                decay.append(param)
            else:
                no_decay.append(param)
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        extra_args = dict(fused=True) if use_fused else dict()
        print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
