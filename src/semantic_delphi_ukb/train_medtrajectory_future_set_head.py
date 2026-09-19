from __future__ import annotations

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
from semantic_delphi_ukb.tte_model import FutureDiseaseSetConfig, FutureDiseaseSetMedTrajectory  # noqa: E402
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
from semantic_delphi_ukb.train_medtrajectory_horizon_risk import (  # noqa: E402
    resolve_diseases_yaml,
    resolve_init_ckpt,
    selected_next_event_loss,
)
from utils import get_p2i  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MedTrajectory with a future disease-set auxiliary head.")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--init-from-ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "medtrajectory_future_set_head")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=600)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--next-event-loss-weight", type=float, default=0.2)
    parser.add_argument("--tte-loss-weight", type=float, default=0.0)
    parser.add_argument("--future-set-loss-weight", type=float, default=1.0)
    parser.add_argument("--future-set-pos-weight", type=float, default=2.0)
    parser.add_argument("--future-set-dice-weight", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260630)
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


def normalize_state_dict(state_dict: dict) -> dict:
    state_dict = dict(state_dict)
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    return state_dict


def make_model(args: argparse.Namespace, diseases_count: int, horizons_count: int) -> FutureDiseaseSetMedTrajectory:
    checkpoint = torch.load(resolve_init_ckpt(args.init_from_ckpt), map_location=args.device, weights_only=False)
    model_args = checkpoint["model_args"].copy()
    model_args.update(
        num_tte_tasks=diseases_count,
        num_horizons=horizons_count,
        tte_loss_weight=args.tte_loss_weight,
        future_set_loss_weight=args.future_set_loss_weight,
        future_set_pos_weight=args.future_set_pos_weight,
        future_set_dice_weight=args.future_set_dice_weight,
    )
    if args.dropout is not None:
        model_args["dropout"] = args.dropout
    model = FutureDiseaseSetMedTrajectory(FutureDiseaseSetConfig(**model_args))
    missing, unexpected = model.load_state_dict(normalize_state_dict(checkpoint["model"]), strict=False)
    allowed_missing = {"future_set_head.weight", "future_set_head.bias"}
    if args.tte_loss_weight == 0.0:
        allowed_missing.update({"tte_head.weight", "tte_head.bias"})
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if set(missing) - allowed_missing:
        raise RuntimeError(f"Unexpected missing checkpoint keys: {missing}")
    return model


def final_position_future_set_loss(model, logits: torch.Tensor, pos: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_index = torch.arange(logits.size(0), device=logits.device)
    return model._future_set_loss(logits[batch_index, pos], labels, mask)


def topk_set_stats(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray, k: int) -> dict[str, float]:
    precision_values = []
    recall_values = []
    jaccard_values = []
    hit_values = []
    for row_scores, row_labels, row_mask in zip(scores, labels, mask):
        valid = np.flatnonzero(row_mask.astype(bool))
        if valid.size == 0:
            continue
        kk = min(int(k), int(valid.size))
        chosen = valid[np.argsort(-row_scores[valid], kind="mergesort")[:kk]]
        truth = set(valid[row_labels[valid] > 0.5].tolist())
        pred = set(chosen.tolist())
        inter = len(pred & truth)
        precision_values.append(inter / max(len(pred), 1))
        recall_values.append(inter / len(truth) if truth else float("nan"))
        union = pred | truth
        jaccard_values.append(inter / len(union) if union else float("nan"))
        hit_values.append(float(inter > 0))
    return {
        f"precision_at_{k}": safe_mean(precision_values),
        f"recall_at_{k}": safe_mean(recall_values),
        f"jaccard_at_{k}": safe_mean(jaccard_values),
        f"hit_at_{k}": safe_mean(hit_values),
    }


def safe_mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else float("nan")


@torch.no_grad()
def estimate(model, data, p2i, static, patient_disease_ages, patient_last_ages, diseases, horizons, args):
    model.eval()
    losses = []
    buckets = {float(horizon): [{"scores": [], "labels": []} for _ in diseases] for horizon in horizons}
    all_scores = {float(horizon): [] for horizon in horizons}
    all_labels = {float(horizon): [] for horizon in horizons}
    all_masks = {float(horizon): [] for horizon in horizons}

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
        logits, _, _, _, future_logits = model(x, age, s, y, target_age, validation_loss_mode=True)
        _ = logits
        _, pos = last_prediction_positions(x, y)
        labels, mask, _ = build_horizon_targets(ix, x, age, y, patient_disease_ages, patient_last_ages, horizons, args.device)
        losses.append(float(final_position_future_set_loss(model, future_logits, pos, labels, mask).detach().cpu()))

        batch_index = torch.arange(future_logits.size(0), device=future_logits.device)
        scores_np = torch.sigmoid(future_logits[batch_index, pos]).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        for horizon_idx, horizon in enumerate(horizons):
            all_scores[float(horizon)].append(scores_np[:, horizon_idx, :])
            all_labels[float(horizon)].append(labels_np[:, horizon_idx, :])
            all_masks[float(horizon)].append(mask_np[:, horizon_idx, :])
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
            if math.isfinite(auc):
                auc_values.append(auc)
            if math.isfinite(capture):
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

    set_rows = []
    for horizon in horizons:
        scores = np.concatenate(all_scores[float(horizon)], axis=0)
        labels = np.concatenate(all_labels[float(horizon)], axis=0)
        mask = np.concatenate(all_masks[float(horizon)], axis=0)
        flat_valid = mask.astype(bool).reshape(-1)
        flat_scores = scores.reshape(-1)[flat_valid].astype(np.float64)
        flat_labels = labels.reshape(-1)[flat_valid].astype(np.int8)
        positives = int(flat_labels.sum())
        negatives = int(flat_labels.size - positives)
        row = {
            "horizon_years": float(horizon),
            "samples": int(scores.shape[0]),
            "labels": int(flat_labels.size),
            "positives": positives,
            "positive_rate": float(flat_labels.mean()) if flat_labels.size else float("nan"),
            "micro_auc": binary_auc(flat_scores, flat_labels) if positives > 0 and negatives > 0 else float("nan"),
        }
        row.update(topk_set_stats(scores, labels, mask, 3))
        row.update(topk_set_stats(scores, labels, mask, 5))
        set_rows.append(row)

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc_mean": float(np.mean(auc_values)) if auc_values else float("nan"),
        "top_decile_capture_mean": float(np.mean(capture_values)) if capture_values else float("nan"),
        "rows": rows,
        "set_rows": set_rows,
    }


def self_test(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    horizons = parse_horizons(args.horizons)
    model = make_model(args, diseases_count=10, horizons_count=len(horizons)).to(args.device)
    batch_size = 4
    seq_len = min(args.block_size, 16)
    x = torch.randint(2, 128, (batch_size, seq_len), device=args.device)
    age = torch.sort(torch.randint(1000, 30000, (batch_size, seq_len), device=args.device).float(), dim=1).values
    y = torch.randint(2, 128, (batch_size, seq_len), device=args.device)
    target_age = age + torch.randint(1, 1000, (batch_size, seq_len), device=args.device).float()
    static = torch.randn(batch_size, int(model.config.static_dim), device=args.device)
    _, _, _, _, future_logits = model(x, age, static, y, target_age)
    _, pos = last_prediction_positions(x, y)
    labels = torch.randint(0, 2, (batch_size, len(horizons), 10), device=args.device).float()
    mask = torch.ones_like(labels)
    loss = final_position_future_set_loss(model, future_logits, pos, labels, mask)
    loss.backward()
    print(json.dumps({"self_test": True, "loss": float(loss.detach().cpu()), "logits_shape": list(future_logits.shape)}, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    data_dir = args.data_dir or (REPO_DIR / "data" / args.dataset)
    diseases, token_groups = load_selected_disease_token_groups(diseases_yaml)

    train_data, train_p2i, train_static = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(data_dir, "val", args.max_patients)
    model = make_model(args, diseases_count=len(diseases), horizons_count=len(horizons)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_tte_ages, train_last_ages = build_patient_disease_ages(train_data, train_p2i, token_groups, int(model.config.vocab_size))
    val_tte_ages, val_last_ages = build_patient_disease_ages(val_data, val_p2i, token_groups, int(model.config.vocab_size))

    run_config = vars(args).copy()
    run_config["horizons"] = horizons
    run_config["diseases"] = disease_specs_payload(diseases)
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    best_score = -float("inf")
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
            for set_row in metrics["set_rows"]:
                prefix = f"h{int(set_row['horizon_years'])}"
                row[f"{prefix}_micro_auc"] = set_row["micro_auc"]
                row[f"{prefix}_recall_at_3"] = set_row["recall_at_3"]
                row[f"{prefix}_jaccard_at_3"] = set_row["jaccard_at_3"]
                row[f"{prefix}_recall_at_5"] = set_row["recall_at_5"]
                row[f"{prefix}_jaccard_at_5"] = set_row["jaccard_at_5"]
            history.append(row)
            print(json.dumps(row, indent=2))
            score = row.get("h10_jaccard_at_3", metrics["auc_mean"])
            if math.isfinite(float(score)) and float(score) > best_score:
                best_score = float(score)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_args": model.config.__dict__.copy(),
                        "iter_num": iteration,
                        "best_val_future_set_score": best_score,
                        "config": run_config,
                    },
                    args.out_dir / "ckpt.pt",
                )
                (args.out_dir / "risk_metrics_rows.json").write_text(json.dumps(metrics["rows"], ensure_ascii=False, indent=2), encoding="utf-8")
                (args.out_dir / "future_set_metrics_rows.json").write_text(json.dumps(metrics["set_rows"], ensure_ascii=False, indent=2), encoding="utf-8")
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
        logits, _, _, _, future_logits = model(x, age, s, y, target_age)
        _, pos = last_prediction_positions(x, y)
        loss = args.future_set_loss_weight * final_position_future_set_loss(model, future_logits, pos, labels, mask)
        if args.next_event_loss_weight > 0:
            loss = loss + args.next_event_loss_weight * selected_next_event_loss(model, logits, x, age, y, target_age)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    status = [
        "# Future disease-set head status",
        "",
        "This is a P1 innovation branch, not yet the frozen main model.",
        f"Best validation future-set score: {best_score:.6f}",
        "",
        "Outputs:",
        "- `history.json`: validation learning curve",
        "- `future_set_metrics_rows.json`: horizon-level set metrics",
        "- `risk_metrics_rows.json`: disease/horizon AUC rows",
        "- `ckpt.pt`: best checkpoint by 10y Jaccard@3 when available",
        "",
    ]
    (args.out_dir / "STATUS.md").write_text("\n".join(status), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
