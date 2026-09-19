from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import EXTERNAL_ROOT
except ImportError:  # executed from inside the source tree
    from paths import EXTERNAL_ROOT

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions  # noqa: E402
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.tte_model import HorizonRiskConfig, HorizonRiskMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    disease_specs_payload,
    load_selected_disease_token_groups,
)
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    binary_auc,
    build_horizon_targets,
    parse_horizons,
    top_decile_stats,
)
from utils import get_p2i  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MedTrajectory with an explicit final-context horizon risk head.")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--init-from-ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "medtrajectory_horizon_risk")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--next-event-loss-weight", type=float, default=0.2)
    parser.add_argument("--tte-loss-weight", type=float, default=0.0)
    parser.add_argument("--horizon-risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-age-encoding", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--use-age-rope", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--age-rope-gate", type=float, default=None)
    parser.add_argument("--monotonic-horizon-risk", action="store_true")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser


def resolve_diseases_yaml(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    root_path = REPO_DIR / "selected_diseases.yaml"
    if root_path.exists():
        return root_path
    return REPO_DIR / "docs" / "selected_diseases.yaml"


def resolve_init_ckpt(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    local = REPO_DIR / "ckpt" / "MedTrajectory_exp2_tte_multitask" / "ckpt.pt"
    if local.exists():
        return local
    remote_reference = Path(str(EXTERNAL_ROOT / "ckpt" / "MedTrajectory_exp2_tte_multitask" / "ckpt.pt"))
    return remote_reference


def load_split(data_dir: Path, split: str, max_patients: int):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if max_patients > 0:
        p2i = p2i[:max_patients]
        static = static[:max_patients]
    return data, p2i, static


def make_model(args: argparse.Namespace, diseases_count: int, horizons_count: int) -> HorizonRiskMedTrajectory:
    checkpoint = torch.load(resolve_init_ckpt(args.init_from_ckpt), map_location=args.device, weights_only=False)
    model_args = checkpoint["model_args"].copy()
    model_args.update(
        num_tte_tasks=diseases_count,
        num_horizons=horizons_count,
        tte_loss_weight=args.tte_loss_weight,
        horizon_risk_loss_weight=args.horizon_risk_loss_weight,
        monotonic_horizon_risk=args.monotonic_horizon_risk,
    )
    if args.use_age_encoding != "auto":
        model_args["use_age_encoding"] = args.use_age_encoding == "true"
    if args.use_age_rope != "auto":
        model_args["use_age_rope"] = args.use_age_rope == "true"
    if args.age_rope_gate is not None:
        model_args["age_rope_gate"] = args.age_rope_gate
    if args.dropout is not None:
        model_args["dropout"] = args.dropout
    model = HorizonRiskMedTrajectory(HorizonRiskConfig(**model_args))
    state_dict = checkpoint["model"].copy()
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    current_state = model.state_dict()
    skipped_mismatched = []
    for key in list(state_dict.keys()):
        if key in current_state and tuple(state_dict[key].shape) != tuple(current_state[key].shape):
            skipped_mismatched.append(
                {
                    "key": key,
                    "checkpoint_shape": list(state_dict[key].shape),
                    "model_shape": list(current_state[key].shape),
                }
            )
            state_dict.pop(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"horizon_risk_head.weight", "horizon_risk_head.bias"}
    if args.tte_loss_weight == 0.0:
        allowed_missing.update({"tte_head.weight", "tte_head.bias"})
    for item in skipped_mismatched:
        allowed_missing.add(item["key"])
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if set(missing) - allowed_missing:
        raise RuntimeError(f"Unexpected missing checkpoint keys: {missing}")
    model._skipped_mismatched_keys = skipped_mismatched  # type: ignore[attr-defined]
    return model


def final_position_horizon_loss(logits: torch.Tensor, pos: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_index = torch.arange(logits.size(0), device=logits.device)
    final_logits = logits[batch_index, pos]
    raw = F.binary_cross_entropy_with_logits(final_logits, labels, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def selected_next_event_loss(model, logits, x, age, y, target_age) -> torch.Tensor:
    loss = model._next_event_loss(logits, x, age, y, target_age, model.build_attention_mask(x, age, target_age), True)
    return loss["loss"]


@torch.no_grad()
def estimate(model, data, p2i, static, patient_disease_ages, patient_last_ages, diseases, horizons, args):
    model.eval()
    losses = []
    buckets = {
        float(horizon): [
            {"scores": [], "labels": []}
            for _ in diseases
        ]
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
        logits, _, _, _, risk_logits = model(x, age, s, y, target_age, validation_loss_mode=True)
        _, pos = last_prediction_positions(x, y)
        labels, mask, _ = build_horizon_targets(ix, x, age, y, patient_disease_ages, patient_last_ages, horizons, args.device)
        losses.append(float(final_position_horizon_loss(risk_logits, pos, labels, mask).detach().cpu()))
        batch_index = torch.arange(risk_logits.size(0), device=risk_logits.device)
        scores_np = torch.sigmoid(risk_logits[batch_index, pos]).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        for horizon_idx, horizon in enumerate(horizons):
            for disease_idx in range(len(diseases)):
                valid = mask_np[:, horizon_idx, disease_idx]
                if not valid.any():
                    continue
                bucket = buckets[float(horizon)][disease_idx]
                bucket["scores"].append(scores_np[:, horizon_idx, disease_idx][valid])
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
    diseases_count = 10
    model = make_model(args, diseases_count=diseases_count, horizons_count=2).to(args.device)
    batch_size = 4
    seq_len = min(args.block_size, 16)
    x = torch.randint(2, 128, (batch_size, seq_len), device=args.device)
    age = torch.sort(torch.randint(1000, 30000, (batch_size, seq_len), device=args.device).float(), dim=1).values
    y = torch.randint(2, 128, (batch_size, seq_len), device=args.device)
    target_age = age + torch.randint(1, 1000, (batch_size, seq_len), device=args.device).float()
    static = torch.randn(batch_size, int(model.config.static_dim), device=args.device)
    _, _, _, _, risk_logits = model(x, age, static, y, target_age)
    _, pos = last_prediction_positions(x, y)
    labels = torch.randint(0, 2, (batch_size, 2, diseases_count), device=args.device).float()
    mask = torch.ones_like(labels)
    loss = final_position_horizon_loss(risk_logits, pos, labels, mask)
    loss.backward()
    payload = {"self_test": True, "loss": float(loss.detach().cpu())}
    if args.monotonic_horizon_risk:
        deltas = risk_logits[:, :, 1:, :] - risk_logits[:, :, :-1, :]
        payload["monotonic_horizon_risk"] = bool(torch.all(deltas >= -1e-7).detach().cpu())
        if not payload["monotonic_horizon_risk"]:
            raise RuntimeError("monotonic horizon risk self-test failed")
    print(json.dumps(payload, indent=2))
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
    diseases, token_groups = load_selected_disease_token_groups(diseases_yaml, data_dir)

    train_data, train_p2i, train_static = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(data_dir, "val", args.max_patients)
    model = make_model(args, diseases_count=len(diseases), horizons_count=len(horizons)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_tte_ages, train_last_ages = build_patient_disease_ages(train_data, train_p2i, token_groups, int(model.config.vocab_size))
    val_tte_ages, val_last_ages = build_patient_disease_ages(val_data, val_p2i, token_groups, int(model.config.vocab_size))

    run_config = vars(args).copy()
    run_config["horizons"] = horizons
    run_config["diseases"] = disease_specs_payload(diseases)
    run_config["skipped_mismatched_keys"] = getattr(model, "_skipped_mismatched_keys", [])
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
        labels, mask, _ = build_horizon_targets(ix, x, age, y, train_tte_ages, train_last_ages, horizons, args.device)
        logits, _, _, _, risk_logits = model(x, age, s, y, target_age)
        _, pos = last_prediction_positions(x, y)
        loss = args.horizon_risk_loss_weight * final_position_horizon_loss(risk_logits, pos, labels, mask)
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
