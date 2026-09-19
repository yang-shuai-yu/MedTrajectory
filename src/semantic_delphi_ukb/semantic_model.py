from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import AgeEncoding, Block, LayerNorm


@dataclass
class SemanticDelphiConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 1.0
    bias: bool = True
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])
    freeze_input_embeddings: bool = False
    tie_input_output_embeddings: bool = False


class SemanticDelphi(nn.Module):
    def __init__(self, config: SemanticDelphiConfig, pretrained_token_embeddings: Optional[np.ndarray] = None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if pretrained_token_embeddings is not None:
            if int(pretrained_token_embeddings.shape[0]) != int(config.vocab_size):
                raise ValueError(
                    f"Pretrained matrix vocab mismatch: matrix={pretrained_token_embeddings.shape[0]} config={config.vocab_size}"
                )
            if int(pretrained_token_embeddings.shape[1]) != int(config.n_embd):
                raise ValueError(
                    f"Pretrained matrix dim mismatch: matrix={pretrained_token_embeddings.shape[1]} config={config.n_embd}"
                )
            embedding_weight = torch.as_tensor(pretrained_token_embeddings, dtype=torch.float32)
            wte = nn.Embedding.from_pretrained(
                embedding_weight,
                freeze=config.freeze_input_embeddings,
                padding_idx=0,
            )
        else:
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)

        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                wae=AgeEncoding(config),
                token_drop=nn.Dropout(config.token_dropout),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if pretrained_token_embeddings is not None:
            with torch.no_grad():
                self.transformer.wte.weight.copy_(torch.as_tensor(pretrained_token_embeddings, dtype=torch.float32))

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        if config.tie_input_output_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, age, targets=None, targets_age=None, validation_loss_mode: bool = False):
        device = idx.device
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
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

        if targets is not None:
            logits = self.lm_head(x)

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
            logits = self.lm_head(x[:, :, :])
            loss = None

        return logits, loss, att

    def adjust_block_size(self, block_size: int):
        for block in self.transformer.h:
            block.attn.bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, module in self.named_modules():
            for pn, _ in module.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    no_decay.add(fpn)

        if self.config.tie_input_output_embeddings and "lm_head.weight" in decay:
            decay.remove("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)
        )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, age, max_new_tokens=100, max_age=85 * 365.25, no_repeat=True, termination_tokens=None, top_k=None):
        if termination_tokens is None:
            warnings.warn("When using a custom dataset, consider changing the `termination_tokens` argument.")
            termination_tokens = [1269]
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=idx.device)
        mask_time = -10000

        if max_new_tokens == -1:
            max_new_tokens = 128

        for _ in range(max_new_tokens):
            logits, _, _ = self(idx, age)
            logits = logits[:, -1, :]
            logits[:, self.config.ignore_tokens] = -torch.inf

            if no_repeat:
                fill = idx.clone()
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)

            t_next = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(), min=0, max=365 * 80).min(1)
            idx_next = t_next[1][:, None]
            age_next = age[..., [-1]] + t_next[0][:, None]

            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)

            if torch.logical_or(torch.isin(idx, termination_tokens).any(-1), age_next > max_age).all():
                break

        pad = (torch.cumsum(torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1) > 1) + (age > max_age)

        logits, _, _ = self(idx, age)
        idx[pad] = 0
        age[pad] = mask_time

        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack([logits[:, j].scatter_(1, fill[:, : j + 1], -torch.inf) for j in range(fill.shape[1])]).transpose(0, 1)

        return idx, age, logits
