"""Evaluate fixed-horizon control tasks on identical patient/disease rows.

Modes:
  lm       next-token LM logits grouped to disease scores (no risk head)
  explicit explicit horizon-risk head
  survival discrete survival-horizon head
  linear   frozen-trunk linear probe checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions  # noqa: E402
from semantic_delphi_ukb.horizon_control_metrics import summarize_scores  # noqa: E402
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.tte_model import (  # noqa: E402
    HorizonRiskConfig,
    HorizonRiskMedTrajectory,
    SurvivalHorizonConfig,
    SurvivalHorizonMedTrajectory,
)
from semantic_delphi_ukb.modern_model import ModernMultitypeSemanticDelphi, ModernMultitypeSemanticDelphiConfig  # noqa: E402
from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    build_horizon_targets,
    build_patient_disease_ages,
    load_followup_end_ages,
    load_selected_disease_token_groups,
    load_split,
    parse_horizons,
    resolve_diseases_yaml,
)
from semantic_delphi_ukb.evaluate_horizon_risk_locked_test import patient_batches  # noqa: E402
from utils import get_p2i  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["lm", "explicit", "survival", "linear"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="Risk/survival checkpoint; trunk source for lm/linear.")
    p.add_argument("--probe-checkpoint", type=Path, default=None, help="Linear probe state for --mode linear.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--diseases-yaml", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--horizons", default="1,5,10")
    p.add_argument("--max-patients", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--calibration-bins", type=int, default=10)
    p.add_argument("--lm-aggregation", choices=["logsumexp", "max"], default="logsumexp")
    return p


def load_model(mode: str, checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = dict(checkpoint["model_args"])
    state_dict = dict(checkpoint["model"])
    for key in list(state_dict):
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod.") :]] = state_dict.pop(key)
    has_survival = any(key.startswith("survival_head.") for key in state_dict)
    has_horizon = any(key.startswith("horizon_risk_head.") for key in state_dict)
    is_car_rope = checkpoint.get("model_family") == "CARoPE_v1" or "rope_scales" in model_args
    if is_car_rope:
        if mode not in {"lm", "explicit", "linear"}:
            raise ValueError(f"CARoPE checkpoint does not support mode={mode}")
        model = CARoPEHorizonMedTrajectory(CARoPEConfig(**model_args))
    elif mode == "survival" or (mode == "lm" and has_survival):
        model = SurvivalHorizonMedTrajectory(SurvivalHorizonConfig(**model_args))
    elif mode == "linear" and has_survival:
        model = SurvivalHorizonMedTrajectory(SurvivalHorizonConfig(**model_args))
    elif mode == "explicit" or (mode == "lm" and has_horizon) or (mode == "linear" and has_horizon):
        model = HorizonRiskMedTrajectory(HorizonRiskConfig(**model_args))
    elif mode in {"lm", "linear"}:
        model = ModernMultitypeSemanticDelphi(ModernMultitypeSemanticDelphiConfig(**model_args))
    else:
        raise ValueError(f"checkpoint does not contain the requested {mode} head")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), checkpoint


def captured_hidden(model, x, age, static):
    captured = []
    if isinstance(model, SurvivalHorizonMedTrajectory):
        module = model.survival_head
    elif hasattr(model, "horizon_risk_head"):
        module = model.horizon_risk_head
    else:
        module = model.lm_head
    handle = module.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    try:
        model(x, age, static)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("model did not expose a hidden state")
    return captured[-1]


def disease_scores_from_lm(logits: torch.Tensor, token_groups: list[list[int]], aggregation: str = "logsumexp") -> torch.Tensor:
    chunks = []
    for tokens in token_groups:
        valid = [token for token in tokens if 0 <= int(token) < logits.size(-1)]
        if not valid:
            chunks.append(torch.full(logits.shape[:-1] + (1,), -torch.inf, device=logits.device))
        else:
            values = logits[..., valid]
            chunks.append((torch.max(values, dim=-1, keepdim=True).values if aggregation == "max" else torch.logsumexp(values, dim=-1, keepdim=True)))
    return torch.cat(chunks, dim=-1)


def append_rows(rows, mode, ix, keep, scores, labels, mask, diseases, horizons):
    for batch_idx, patient_idx in enumerate(ix.tolist()):
        if not bool(keep[batch_idx]):
            continue
        for horizon_idx, horizon in enumerate(horizons):
            for disease_idx, disease in enumerate(diseases):
                if not bool(mask[batch_idx, horizon_idx, disease_idx]):
                    continue
                rows.append({
                    "model": mode,
                    "patient_index": int(patient_idx),
                    "horizon_years": float(horizon),
                    "disease_id": disease.disease_id,
                    "label": int(labels[batch_idx, horizon_idx, disease_idx]),
                    "score": float(scores[batch_idx, horizon_idx, disease_idx]),
                })


@torch.no_grad()
def evaluate(args):
    horizons = parse_horizons(args.horizons)
    data, p2i, static = load_split(args.data_dir, args.split, args.max_patients)
    diseases, token_groups = load_selected_disease_token_groups(resolve_diseases_yaml(args.diseases_yaml), args.data_dir)
    vocab_size = int(json.loads((args.data_dir / "prepare_manifest.json").read_text(encoding="utf-8")).get("vocab_size", int(data[:, 2].max()) + 2))
    patient_ages, observed_last = build_patient_disease_ages(data, p2i, token_groups, vocab_size)
    followup_end = load_followup_end_ages(args.data_dir, args.split, observed_last, args.max_patients)
    model, checkpoint = load_model(args.mode, args.checkpoint, args.device)
    probe = None
    if args.mode == "linear":
        if args.probe_checkpoint is None:
            raise ValueError("--probe-checkpoint is required for --mode linear")
        payload = torch.load(args.probe_checkpoint, map_location=args.device, weights_only=False)
        expected_ids = [d.disease_id for d in diseases]
        if payload.get("horizons") != horizons or payload.get("disease_ids") != expected_ids:
            raise ValueError("linear probe checkpoint horizons/disease_ids do not match this evaluator")
        probe = torch.nn.Linear(int(payload["hidden_dim"]), len(horizons) * len(diseases)).to(args.device)
        probe.load_state_dict(payload["state_dict"], strict=True)
        probe.eval()

    rows = []
    for ix in patient_batches(len(p2i), args.batch_size):
        x, age, y, target_age, static_batch = get_batch(
            ix, data, p2i, static, select="left", padding="regular", block_size=args.block_size,
            device=args.device, cut_batch=False,
        )
        keep, pos = last_prediction_positions(x, y)
        labels, mask, _ = build_horizon_targets(
            ix, x, age, y, patient_ages, followup_end, horizons, args.device, censor_aware=True
        )
        batch_idx = torch.arange(x.size(0), device=args.device)
        if args.mode == "lm":
            lm_logits = model(x, age, static_batch)[0]
            score = disease_scores_from_lm(lm_logits[batch_idx, pos], token_groups, args.lm_aggregation)
            score = score[:, None, :].expand(-1, len(horizons), -1)
        elif args.mode == "explicit":
            risk_logits = model(x, age, static_batch)[4]
            score = torch.sigmoid(risk_logits[batch_idx, pos])
        elif args.mode == "survival":
            survival_logits = model(x, age, static_batch)[4]
            score = model.horizon_risk_from_survival(survival_logits[batch_idx, pos], horizons)
        else:
            hidden = captured_hidden(model, x, age, static_batch)
            score = torch.sigmoid(probe(hidden[batch_idx, pos]).view(x.size(0), len(horizons), len(diseases)))
        append_rows(rows, args.mode, ix.cpu().numpy(), keep.cpu().numpy(), score.cpu().numpy(), labels.cpu().numpy(), mask.cpu().numpy(), diseases, horizons)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_rows = []
    for horizon in horizons:
        for disease in diseases:
            selected = [row for row in rows if row["horizon_years"] == float(horizon) and row["disease_id"] == disease.disease_id]
            if not selected:
                continue
            scores = np.asarray([row["score"] for row in selected], dtype=np.float64)
            labels = np.asarray([row["label"] for row in selected], dtype=np.int8)
            summary_rows.append({
                "model": args.mode,
                "horizon_years": horizon,
                "disease_id": disease.disease_id,
                **summarize_scores(scores, labels, args.calibration_bins, probability_scores=args.mode != "lm"),
            })
    (args.out_dir / "summary.json").write_text(json.dumps({
        "mode": args.mode,
        "split": args.split,
        "rows": len(rows),
        "score_scale": "raw_lm_logit" if args.mode == "lm" else "probability_sigmoid",
        "metrics": summary_rows,
    }, indent=2), encoding="utf-8")
    return rows


def main():
    args = parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
