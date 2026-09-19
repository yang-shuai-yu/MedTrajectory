from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions  # noqa: E402
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.tte_model import SurvivalHorizonConfig, SurvivalHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    build_survival_horizon_batch,
    disease_specs_payload,
    load_selected_disease_token_groups,
)
from semantic_delphi_ukb.train_architecture_risk_heads import binary_auc, parse_horizons, top_decile_stats  # noqa: E402
from semantic_delphi_ukb.train_medtrajectory_horizon_risk import (  # noqa: E402
    resolve_diseases_yaml,
    resolve_init_ckpt,
    selected_next_event_loss,
)
from utils import get_p2i  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MedTrajectory with a unified discrete survival-horizon head.")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--init-from-ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "medtrajectory_survival_horizon")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument("--survival-bins", type=int, default=10)
    parser.add_argument("--survival-bin-years", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--next-event-loss-weight", type=float, default=0.2)
    parser.add_argument("--tte-loss-weight", type=float, default=0.0)
    parser.add_argument("--survival-loss-weight", type=float, default=1.0)
    parser.add_argument("--survival-pos-weight", type=float, default=1.0)
    parser.add_argument("--survival-focal-gamma", type=float, default=0.0)
    parser.add_argument("--aux-horizon-loss-weight", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser


def load_split(data_dir: Path, split: str, max_patients: int):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if max_patients > 0:
        p2i = p2i[:max_patients]
        static = static[:max_patients]
    return data, p2i, static


def make_model(args: argparse.Namespace, diseases_count: int) -> SurvivalHorizonMedTrajectory:
    checkpoint = torch.load(resolve_init_ckpt(args.init_from_ckpt), map_location=args.device, weights_only=False)
    model_args = checkpoint["model_args"].copy()
    model_args.update(
        num_tte_tasks=diseases_count,
        num_horizons=len(parse_horizons(args.horizons)),
        tte_loss_weight=args.tte_loss_weight,
        survival_loss_weight=args.survival_loss_weight,
        survival_pos_weight=args.survival_pos_weight,
        survival_focal_gamma=args.survival_focal_gamma,
        survival_bins=args.survival_bins,
        survival_bin_years=args.survival_bin_years,
    )
    if args.dropout is not None:
        model_args["dropout"] = args.dropout
    model = SurvivalHorizonMedTrajectory(SurvivalHorizonConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"survival_head.weight", "survival_head.bias"}
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if set(missing) - allowed_missing:
        raise RuntimeError(f"Unexpected missing checkpoint keys: {missing}")
    return model


def final_position_survival_loss(model, survival_logits, pos, event, mask):
    batch_index = torch.arange(survival_logits.size(0), device=survival_logits.device)
    return model._survival_loss(survival_logits[batch_index, pos], event[batch_index, pos], mask[batch_index, pos])


def final_position_aux_horizon_loss(model, survival_logits, pos, labels, mask, horizons):
    batch_index = torch.arange(survival_logits.size(0), device=survival_logits.device)
    risk = model.horizon_risk_from_survival(survival_logits[batch_index, pos], horizons)
    risk = risk.clamp(min=1e-6, max=1.0 - 1e-6)
    raw = F.binary_cross_entropy(risk, labels, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def estimate(model, data, p2i, static, patient_disease_ages, patient_last_ages, diseases, horizons, args):
    model.eval()
    losses = []
    buckets = {
        float(horizon): [{"scores": [], "labels": []} for _ in diseases]
        for horizon in horizons
    }
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,))
        x, age, y, target_age, s = get_batch(
            ix,
            data,
            p2i,
            static,
            block_size=args.block_size,
            device=args.device,
            padding="random",
            select="random",
            cut_batch=True,
        )
        survival_event, survival_mask, horizon_event, horizon_mask = build_survival_horizon_batch(
            ix,
            age,
            x > 1,
            patient_disease_ages,
            patient_last_ages,
            args.survival_bins,
            args.survival_bin_years,
            horizons,
            args.device,
        )
        logits, _, _, _, survival_logits = model(x, age, s, y, target_age, validation_loss_mode=True)
        _, pos = last_prediction_positions(x, y)
        losses.append(float(final_position_survival_loss(model, survival_logits, pos, survival_event, survival_mask).detach().cpu()))
        batch_index = torch.arange(survival_logits.size(0), device=survival_logits.device)
        risk = model.horizon_risk_from_survival(survival_logits[batch_index, pos], horizons).detach().cpu().numpy()
        labels_np = horizon_event[batch_index, pos].detach().cpu().numpy()
        mask_np = horizon_mask[batch_index, pos].detach().cpu().numpy().astype(bool)
        for horizon_idx, horizon in enumerate(horizons):
            for disease_idx in range(len(diseases)):
                valid = mask_np[:, horizon_idx, disease_idx]
                if not valid.any():
                    continue
                bucket = buckets[float(horizon)][disease_idx]
                bucket["scores"].append(risk[:, horizon_idx, disease_idx][valid])
                bucket["labels"].append(labels_np[:, horizon_idx, disease_idx][valid])
    model.train()

    rows = []
    auc_values = []
    capture_values = []
    for horizon in horizons:
        for disease_idx, disease in enumerate(diseases):
            bucket = buckets[float(horizon)][disease_idx]
            if bucket["scores"]:
                scores = np.concatenate(bucket["scores"]).astype(np.float64)
                labels = np.concatenate(bucket["labels"]).astype(np.int8)
            else:
                scores = np.asarray([], dtype=np.float64)
                labels = np.asarray([], dtype=np.int8)
            positives = int(labels.sum())
            negatives = int(labels.size - positives)
            auc = binary_auc(scores, labels) if positives > 0 and negatives > 0 else float("nan")
            capture, top_rate, lift = top_decile_stats(scores, labels)
            if not math.isnan(auc):
                auc_values.append(auc)
            if not math.isnan(capture):
                capture_values.append(capture)
            rows.append(
                {
                    "horizon_years": float(horizon),
                    "disease_id": disease.disease_id,
                    "name": disease.name,
                    "name_cn": disease.name_cn,
                    "positives": positives,
                    "negatives": negatives,
                    "auc": auc,
                    "top_decile_capture": capture,
                    "top_decile_event_rate": top_rate,
                    "top_decile_lift": lift,
                }
            )
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc_mean": float(np.mean(auc_values)) if auc_values else float("nan"),
        "top_decile_capture_mean": float(np.mean(capture_values)) if capture_values else float("nan"),
        "rows": rows,
    }


def self_test(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    model = make_model(args, diseases_count=10).to(args.device)
    batch_size = 4
    seq_len = min(args.block_size, 16)
    x = torch.randint(2, 128, (batch_size, seq_len), device=args.device)
    age = torch.sort(torch.randint(1000, 30000, (batch_size, seq_len), device=args.device).float(), dim=1).values
    y = torch.randint(2, 128, (batch_size, seq_len), device=args.device)
    target_age = age + torch.randint(1, 1000, (batch_size, seq_len), device=args.device).float()
    static = torch.randn(batch_size, int(model.config.static_dim), device=args.device)
    _, _, _, _, survival_logits = model(x, age, static, y, target_age)
    _, pos = last_prediction_positions(x, y)
    event = torch.zeros(batch_size, seq_len, 10, args.survival_bins, device=args.device)
    mask = torch.ones_like(event)
    loss = final_position_survival_loss(model, survival_logits, pos, event, mask)
    loss.backward()
    risk = model.horizon_risk_from_survival(survival_logits[torch.arange(batch_size, device=args.device), pos], [5, 10])
    print(json.dumps({"self_test": True, "loss": float(loss.detach().cpu()), "risk_shape": list(risk.shape)}, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    data_dir = args.data_dir or (REPO_DIR / "data" / args.dataset)
    diseases, token_groups = load_selected_disease_token_groups(diseases_yaml)
    train_data, train_p2i, train_static = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(data_dir, "val", args.max_patients)
    model = make_model(args, diseases_count=len(diseases)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_tte_ages, train_last_ages = build_patient_disease_ages(train_data, train_p2i, token_groups, int(model.config.vocab_size))
    val_tte_ages, val_last_ages = build_patient_disease_ages(val_data, val_p2i, token_groups, int(model.config.vocab_size))

    run_config = vars(args).copy()
    run_config["horizons"] = horizons
    run_config["diseases"] = disease_specs_payload(diseases)
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    best_auc = -float("inf")
    history = []
    for iteration in range(args.max_iters + 1):
        if iteration % args.eval_interval == 0:
            metrics = estimate(model, val_data, val_p2i, val_static, val_tte_ages, val_last_ages, diseases, horizons, args)
            row = {
                "iter": iteration,
                "val_loss": metrics["loss"],
                "val_auc_mean": metrics["auc_mean"],
                "val_top_decile_capture_mean": metrics["top_decile_capture_mean"],
            }
            history.append(row)
            print(json.dumps(row, indent=2))
            if metrics["auc_mean"] > best_auc:
                best_auc = metrics["auc_mean"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_args": model.config.__dict__.copy(),
                        "iter_num": iteration,
                        "best_val_auc_mean": best_auc,
                        "config": run_config,
                    },
                    args.out_dir / "ckpt.pt",
                )
                (args.out_dir / "risk_metrics_rows.json").write_text(json.dumps(metrics["rows"], ensure_ascii=False, indent=2), encoding="utf-8")
        if iteration == args.max_iters:
            break

        ix = torch.randint(len(train_p2i), (args.batch_size,))
        x, age, y, target_age, s = get_batch(
            ix,
            train_data,
            train_p2i,
            train_static,
            block_size=args.block_size,
            device=args.device,
            padding="random",
            select="random",
            cut_batch=True,
        )
        survival_event, survival_mask, horizon_event, horizon_mask = build_survival_horizon_batch(
            ix,
            age,
            x > 1,
            train_tte_ages,
            train_last_ages,
            args.survival_bins,
            args.survival_bin_years,
            horizons,
            args.device,
        )
        logits, _, _, _, survival_logits = model(x, age, s, y, target_age)
        _, pos = last_prediction_positions(x, y)
        loss = args.survival_loss_weight * final_position_survival_loss(model, survival_logits, pos, survival_event, survival_mask)
        if args.aux_horizon_loss_weight > 0:
            loss = loss + args.aux_horizon_loss_weight * final_position_aux_horizon_loss(
                model,
                survival_logits,
                pos,
                horizon_event[torch.arange(horizon_event.size(0), device=horizon_event.device), pos],
                horizon_mask[torch.arange(horizon_mask.size(0), device=horizon_mask.device), pos],
                horizons,
            )
        if args.next_event_loss_weight > 0:
            loss = loss + args.next_event_loss_weight * selected_next_event_loss(model, logits, x, age, y, target_age)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
