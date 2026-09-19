from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from semantic_delphi_ukb.modern_model import ModernMultitypeSemanticDelphi, ModernMultitypeSemanticDelphiConfig
from semantic_delphi_ukb.selected_disease_demo import LoadedModel, ModelSpec, load_labels, load_token_codes
from utils import get_p2i


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def modern_model_spec() -> ModelSpec:
    return ModelSpec(
        model_id="modern",
        display_name="MedTrajectory modern baseline",
        model_type="multitype",
        ckpt_path=REPO_DIR / "ckpt" / "MedTrajectory_exp2_modern_baseline" / "ckpt.pt",
        data_dir=REPO_DIR / "data" / "ukb_semantic_multitype_explicit_split",
    )


def load_modern_model(split: str, device: str) -> LoadedModel:
    spec = modern_model_spec()
    checkpoint = torch.load(spec.ckpt_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    conf = ModernMultitypeSemanticDelphiConfig(**checkpoint["model_args"])
    model = ModernMultitypeSemanticDelphi(conf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    labels = load_labels(spec.data_dir / "labels.csv")
    token_codes, diagnosis_tokens = load_token_codes(spec)
    data = np.memmap(spec.data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static_matrix = np.load(spec.data_dir / f"{split}_static.npy").astype(np.float32)

    return LoadedModel(
        spec=spec,
        model=model,
        block_size=int(conf.block_size),
        no_event_token_rate=int(checkpoint_config.get("no_event_token_rate", 5)),
        labels=labels,
        token_codes=token_codes,
        diagnosis_tokens=sorted(diagnosis_tokens),
        data=data,
        p2i=get_p2i(data),
        static_matrix=static_matrix,
    )


def modern_checkpoint_meta() -> dict:
    spec = modern_model_spec()
    checkpoint = torch.load(spec.ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    return {
        "ckpt_path": str(spec.ckpt_path),
        "model_family": config.get("model_family", "medtrajectory_modern_baseline"),
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "dataset": config.get("dataset", "ukb_semantic_multitype_explicit_split"),
    }


def resolve_vocab_csv() -> Path:
    manifest = json.loads((modern_model_spec().data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    return Path(str(manifest["vocab_csv"]))
