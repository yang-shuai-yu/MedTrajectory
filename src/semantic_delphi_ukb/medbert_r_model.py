from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MedBERTRConfig:
    vocab_size: int
    max_sequence_length: int = 133
    age_vocab_size: int = 121
    n_layer: int = 6
    n_head: int = 6
    hidden_size: int = 192
    intermediate_size: int = 64
    dropout: float = 0.1
    num_diseases: int = 10
    num_horizons: int = 3
    tie_mlm_decoder: bool = True


class MedBERTR(nn.Module):
    """Med-BERT method adaptation for the frozen Track R inputs."""

    def __init__(self, config: MedBERTRConfig) -> None:
        super().__init__()
        if config.hidden_size % config.n_head:
            raise ValueError("hidden_size must be divisible by n_head")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.age_embedding = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.token_type_embedding = nn.Embedding(2, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_head,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layer, norm=nn.LayerNorm(config.hidden_size))
        self.mlm_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        if config.tie_mlm_decoder:
            self.mlm_decoder.weight = self.token_embedding.weight
        self.risk_head = nn.Linear(config.hidden_size, config.num_diseases * config.num_horizons)

    def embeddings(self, token_ids: torch.Tensor, age_days: torch.Tensor, static_token_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device).view(1, -1)
        age_years = torch.clamp(torch.floor(age_days / 365.25), min=0, max=self.config.age_vocab_size - 1).long()
        token_type = static_token_mask.long()
        hidden = (
            self.token_embedding(token_ids)
            + self.age_embedding(age_years)
            + self.position_embedding(positions)
            + self.token_type_embedding(token_type)
        )
        return self.embedding_dropout(self.embedding_norm(hidden))

    def encode(
        self,
        token_ids: torch.Tensor,
        age_days: torch.Tensor,
        static_token_mask: torch.Tensor,
        attention_allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embeddings(token_ids, age_days, static_token_mask)
        padding_mask = token_ids == 0
        if attention_allowed is not None:
            if attention_allowed.ndim != 3 or attention_allowed.shape != (
                token_ids.size(0), token_ids.size(1), token_ids.size(1)
            ):
                raise ValueError("attention_allowed must have shape [batch,sequence,sequence]")
            # PyTorch still evaluates padded query rows. An all-masked row
            # produces NaN attention and can contaminate valid tokens in later
            # encoder layers. Give every such row one valid key; padded tokens
            # remain excluded as keys and from all objectives.
            no_key = ~attention_allowed.any(dim=-1)
            valid_key = token_ids > 0
            first_valid = valid_key.to(torch.int64).argmax(dim=-1)
            rows = no_key.nonzero(as_tuple=False)
            if rows.numel():
                attention_allowed = attention_allowed.clone()
                attention_allowed[rows[:, 0], rows[:, 1], first_valid[rows[:, 0]]] = True
            batch_head_mask = (~attention_allowed).repeat_interleave(self.config.n_head, dim=0)
            return self.encoder(hidden, mask=batch_head_mask, src_key_padding_mask=padding_mask)
        return self.encoder(hidden, src_key_padding_mask=padding_mask)

    def forward(
        self,
        token_ids: torch.Tensor,
        age_days: torch.Tensor,
        static_token_mask: torch.Tensor,
        attention_allowed: torch.Tensor | None = None,
        mlm_labels: torch.Tensor | None = None,
        prediction_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(token_ids, age_days, static_token_mask, attention_allowed=attention_allowed)
        output = {"hidden": hidden}
        mlm_logits = self.mlm_decoder(self.mlm_transform(hidden)) + self.mlm_bias
        output["mlm_logits"] = mlm_logits
        if mlm_labels is not None:
            valid = (mlm_labels >= 0) & ~static_token_mask
            output["loss_mlm"] = (
                F.cross_entropy(mlm_logits[valid], mlm_labels[valid])
                if bool(valid.any())
                else mlm_logits.sum() * 0.0
            )
        if prediction_positions is not None:
            rows = torch.arange(token_ids.size(0), device=token_ids.device)
            raw = self.risk_head(hidden[rows, prediction_positions])
            raw = raw.view(token_ids.size(0), self.config.num_horizons, self.config.num_diseases)
            first = raw[:, :1]
            output["risk_logits"] = torch.cat((first, first + torch.cumsum(F.softplus(raw[:, 1:]), dim=1)), dim=1)
        return output


def parameter_report(model: MedBERTR) -> dict[str, int]:
    embedding_parameters = sum(p.numel() for module in (
        model.token_embedding,
        model.age_embedding,
        model.position_embedding,
        model.token_type_embedding,
    ) for p in module.parameters())
    transformer_block_parameters = sum(p.numel() for p in model.encoder.layers.parameters())
    total_parameters = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_parameters,
        "total_trainable_parameters": total_trainable,
        "embedding_parameters": embedding_parameters,
        "transformer_block_parameters": transformer_block_parameters,
        "other_parameters": total_parameters - embedding_parameters - transformer_block_parameters,
    }
