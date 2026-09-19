from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory
from semantic_delphi_ukb.paper_run import strip_compiled_prefix


MATCHED_FAMILIES = ("ethos_matched", "foresight_matched")


@dataclass
class TrackGMatchedConfig(CARoPEConfig):
    track_g_family: str = "foresight_matched"
    ethos_gap_bucket_upper_days: tuple[float, ...] = (
        1.0,
        7.0,
        30.0,
        90.0,
        365.25,
        1826.25,
        3652.5,
    )


def inter_event_gap_days(age: torch.Tensor) -> torch.Tensor:
    gap = torch.zeros_like(age)
    if age.size(1) > 1:
        delta = age[:, 1:] - age[:, :-1]
        valid = (age[:, 1:] > -5000.0) & (age[:, :-1] > -5000.0) & (delta >= 0.0)
        gap[:, 1:] = torch.where(valid, delta, torch.zeros_like(delta))
    return gap


class TrackGMatchedTimelineModel(CARoPEHorizonMedTrajectory):
    """Matched causal baselines on the frozen Track R input and loss contract.

    These are task-matched implementations, not claims of exact paper reproduction.
    """

    def __init__(
        self,
        config: TrackGMatchedConfig,
        pretrained_token_embeddings: np.ndarray | None = None,
    ) -> None:
        if config.track_g_family not in MATCHED_FAMILIES:
            raise ValueError(f"unknown Track G family: {config.track_g_family}")
        if config.use_age_encoding or config.use_age_rope:
            raise ValueError("matched baselines replace CARoPE age encodings with their registered embedding")
        super().__init__(config, pretrained_token_embeddings=pretrained_token_embeddings)
        if config.track_g_family == "ethos_matched":
            boundaries = torch.as_tensor(config.ethos_gap_bucket_upper_days, dtype=torch.float32)
            if boundaries.ndim != 1 or not bool((boundaries[1:] > boundaries[:-1]).all()):
                raise ValueError("ETHOS gap bucket boundaries must be strictly increasing")
            self.register_buffer("ethos_gap_boundaries", boundaries, persistent=True)
            self.time_embedding = nn.Embedding(len(boundaries) + 1, config.n_embd)
            self.position_embedding = None
            self.time_embedding.apply(self._init_weights)
        else:
            self.register_buffer("ethos_gap_boundaries", torch.empty(0), persistent=True)
            self.time_embedding = None
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
            self.position_embedding.apply(self._init_weights)

    def apply_age_encoding(self, x: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        if self.config.track_g_family == "ethos_matched":
            buckets = torch.bucketize(inter_event_gap_days(age), self.ethos_gap_boundaries)
            return x + self.time_embedding(buckets)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.position_embedding(positions)[None, :, :]


def build_matched_model(
    family: str,
    *,
    block_size: int,
    vocab_size: int,
    semantic_embedding_dim: int,
    pretrained_token_embeddings: np.ndarray,
    num_diseases: int,
    architecture: Mapping,
    horizons_years: Sequence[float],
) -> TrackGMatchedTimelineModel:
    if family not in MATCHED_FAMILIES:
        raise ValueError(f"unknown Track G matched family: {family}")
    config = TrackGMatchedConfig(
        block_size=int(block_size),
        vocab_size=int(vocab_size),
        n_layer=int(architecture["n_layer"]),
        n_head=int(architecture["n_head"]),
        n_embd=int(architecture["n_embd"]),
        semantic_embedding_dim=int(semantic_embedding_dim),
        dropout=float(architecture["dropout"]),
        static_dim=0,
        static_hidden_dim=int(architecture["n_embd"]),
        num_tte_tasks=int(num_diseases),
        num_horizons=len(horizons_years),
        horizon_years=tuple(float(value) for value in horizons_years),
        use_age_encoding=False,
        use_age_rope=False,
        use_relative_horizon_query=False,
        time_gap_loss_weight=0.0,
        track_g_family=family,
        ethos_gap_bucket_upper_days=tuple(
            float(value) for value in architecture["ethos_gap_bucket_upper_days"]
        ),
    )
    return TrackGMatchedTimelineModel(config, pretrained_token_embeddings=pretrained_token_embeddings)


def checkpoint_state(
    checkpoint: Mapping,
    *,
    expected_family: str | None = None,
) -> tuple[CARoPEHorizonMedTrajectory, str]:
    family = str(checkpoint.get("track_g_family", "carope"))
    if expected_family is not None and family != expected_family:
        raise ValueError(f"checkpoint family mismatch: expected {expected_family}, got {family}")
    model_args = dict(checkpoint["model_args"])
    if family in MATCHED_FAMILIES:
        config = TrackGMatchedConfig(**model_args)
        model = TrackGMatchedTimelineModel(config)
    elif family == "carope":
        model = CARoPEHorizonMedTrajectory(CARoPEConfig(**model_args))
    else:
        raise ValueError(f"unsupported Track G checkpoint family: {family}")
    model.load_state_dict(strip_compiled_prefix(checkpoint["model"]), strict=True)
    return model, family


def model_config_payload(model: CARoPEHorizonMedTrajectory) -> dict:
    return asdict(model.config)
