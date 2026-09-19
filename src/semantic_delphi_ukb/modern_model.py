from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import AgeEncoding


class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class ContinuousAgeRoPE(nn.Module):
    """Rotary q/k phase from real age in years, not from integer token position."""

    def __init__(self, head_dim: int, base: float = 10000.0, scale: float = 1.0, gate: float = 1.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        if gate < 0.0 or gate > 1.0:
            raise ValueError("age_rope_gate must be between 0.0 and 1.0.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scale = float(scale)
        self.gate = float(gate)

    def forward(self, q: torch.Tensor, k: torch.Tensor, age_days: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = (age_days.float() / 365.25 * self.scale).unsqueeze(-1) * self.inv_freq
        theta = torch.repeat_interleave(theta, repeats=2, dim=-1).unsqueeze(1)
        cos = theta.cos().to(dtype=q.dtype)
        sin = theta.sin().to(dtype=q.dtype)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        if self.gate == 1.0:
            return q_rot, k_rot
        return q + self.gate * (q_rot - q), k + self.gate * (k_rot - k)


class ModernCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rope = (
            ContinuousAgeRoPE(self.head_dim, scale=config.age_rope_scale, gate=config.age_rope_gate)
            if config.use_age_rope
            else None
        )

    def forward(self, x: torch.Tensor, age: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, age)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y)), att


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.swiglu_hidden_mult * config.n_embd)
        self.w12 = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.dropout(self.w3(F.silu(gate) * value))


class ModernBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = ModernCausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, age: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, att = self.attn(self.norm_1(x), age, attn_mask)
        x = x + y
        x = x + self.mlp(self.norm_2(x))
        return x, att


@dataclass
class ModernMultitypeSemanticDelphiConfig:
    block_size: int = 128
    vocab_size: int = 8424
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 64
    dropout: float = 0.1
    token_dropout: float = 0.0
    t_min: float = 0.1
    bias: bool = False
    mask_ties: bool = True
    ignore_tokens: list = field(default_factory=lambda: [0])
    freeze_input_embeddings: bool = False
    tie_input_output_embeddings: bool = False
    use_age_encoding: bool = True
    use_age_rope: bool = True
    age_rope_scale: float = 1.0
    age_rope_gate: float = 1.0
    swiglu_hidden_mult: float = 8.0 / 3.0
    static_dim: int = 10
    static_hidden_dim: int = 64
    static_dropout: float = 0.05
    fusion_mode: str = "residual"
    fuse_static_to_logits: bool = False
    semantic_embedding_dim: Optional[int] = None


class ModernMultitypeSemanticDelphi(nn.Module):
    def __init__(
        self,
        config: ModernMultitypeSemanticDelphiConfig,
        pretrained_token_embeddings: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.config = config
        semantic_dim = int(config.semantic_embedding_dim or config.n_embd)
        if pretrained_token_embeddings is not None:
            if pretrained_token_embeddings.shape != (config.vocab_size, semantic_dim):
                raise ValueError(
                    "Pretrained matrix shape mismatch: "
                    f"matrix={pretrained_token_embeddings.shape} config={(config.vocab_size, semantic_dim)}"
                )
            wte = nn.Embedding.from_pretrained(
                torch.as_tensor(pretrained_token_embeddings, dtype=torch.float32),
                freeze=config.freeze_input_embeddings,
                padding_idx=0,
            )
        else:
            wte = nn.Embedding(config.vocab_size, semantic_dim, padding_idx=0)

        if config.tie_input_output_embeddings and semantic_dim != config.n_embd:
            raise ValueError("input/output embeddings cannot be tied when semantic_embedding_dim != n_embd")
        self.token_input_proj = (
            nn.Linear(semantic_dim, config.n_embd, bias=False)
            if semantic_dim != config.n_embd
            else nn.Identity()
        )

        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                wae=AgeEncoding(config),
                token_drop=nn.Dropout(config.token_dropout),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([ModernBlock(config) for _ in range(config.n_layer)]),
                norm_f=RMSNorm(config.n_embd),
            )
        )
        self.static_encoder = (
            nn.Sequential(
                nn.Linear(config.static_dim, config.static_hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.static_dropout),
                nn.Linear(config.static_hidden_dim, config.n_embd),
            )
            if config.static_dim > 0
            else None
        )
        self.static_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.static_to_logits = (
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
            if config.fuse_static_to_logits and config.static_dim > 0
            else None
        )

        self.apply(self._init_weights)
        if pretrained_token_embeddings is not None:
            with torch.no_grad():
                self.transformer.wte.weight.copy_(torch.as_tensor(pretrained_token_embeddings, dtype=torch.float32))
        if config.tie_input_output_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
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

    def apply_age_encoding(self, x: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "use_age_encoding", True):
            return x + self.transformer.wae(age.unsqueeze(-1))
        return x

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

    def forward(
        self,
        idx,
        age,
        static_features=None,
        targets=None,
        targets_age=None,
        validation_loss_mode: bool = False,
    ):
        tok_emb = self.transformer.token_drop(self.transformer.wte(idx)) * (1 - self.config.token_dropout)
        x = self.token_input_proj(tok_emb)
        x = self.apply_age_encoding(x, age)

        static_bias = None
        if static_features is not None:
            if self.static_encoder is None:
                raise ValueError("static_features were supplied to a model with static_dim=0")
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

        loss = None
        if targets is not None:
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
            loss = {"loss_ce": loss_ce, "loss_dt": torch.mean(loss_dt[pass_tokens])}
        return logits, loss, torch.stack(att)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
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
