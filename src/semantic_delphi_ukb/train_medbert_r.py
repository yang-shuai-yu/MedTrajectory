"""Train Med-BERT-R with MLM pretraining or Track R horizon fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(path))

from semantic_delphi_ukb.medbert_r_model import MedBERTR, MedBERTRConfig, parameter_report  # noqa: E402
from semantic_delphi_ukb.paper_run import seed_everything, write_source_manifest  # noqa: E402
from semantic_delphi_ukb.track_r_batch import get_track_r_batch, load_track_r_assets, validate_track_r_data_manifest  # noqa: E402
from semantic_delphi_ukb.track_r_contract import (  # noqa: E402
    load_track_r_protocol,
    mask_medbert_inputs,
    medbert_attention_allowed,
)
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    build_horizon_targets,
    load_followup_end_ages,
)
from semantic_delphi_ukb.train_car_rope import load_split  # noqa: E402
from semantic_delphi_ukb.tte_targets import build_patient_disease_ages, load_selected_disease_token_groups  # noqa: E402
from training_monitor import RunMonitor  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("pretraining", "horizon"), required=True)
    p.add_argument("--variant", choices=("Med-BERT-Paper", "Med-BERT-Matched-S"), required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/paper_protocol_v1/TRACK_R_v2_1.json")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--init-from-ckpt", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-iters", type=int, default=100000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--no-tensorboard", action="store_true")
    return p


def variant_config(protocol: dict, name: str, vocab_size: int, num_diseases: int) -> MedBERTRConfig:
    item = next(value for value in protocol["medbert_variants"] if value["name"] == name)
    return MedBERTRConfig(
        vocab_size=vocab_size,
        max_sequence_length=protocol["dynamic_context_length"] + protocol["static_prefix"]["fixed_length"] + 1,
        n_layer=item["n_layer"],
        n_head=item["n_head"],
        hidden_size=item["hidden_size"],
        intermediate_size=item["intermediate_size"],
        num_diseases=num_diseases,
        num_horizons=len(protocol["horizons_years"]),
    )


def batch(ix, data, p2i, static, prefix, anchor, protocol, manifest, device, select):
    x, age, y, target_age, _, static_mask, bos_mask, _, _, _ = get_track_r_batch(
        ix,
        data,
        p2i,
        static,
        prefix,
        anchor,
        include_static_prefix=True,
        dynamic_context_length=protocol["dynamic_context_length"],
        device=device,
        select=select,
        padding="regular",
        no_event_token_rate=5,
        cut_batch=True,
        bos_token_id=manifest["dynamic_bos_token_id"],
    )
    return x, age, y, target_age, static_mask, bos_mask


def mlm_loss(model, x, age, static_mask, bos_mask, manifest, generator):
    masked, labels = mask_medbert_inputs(
        x,
        static_mask,
        bos_mask,
        manifest["mask_token_id"],
        manifest["dynamic_vocab_size"],
        generator,
        probability=0.35,
    )
    allowed = medbert_attention_allowed(masked, age, static_mask)
    return model(masked, age, static_mask, attention_allowed=allowed, mlm_labels=labels)["loss_mlm"]


def horizon_objective(model, ix, x, age, y, static_mask, bos_mask, patient_ages, followup, horizons, device):
    horizon_y = y.masked_fill(static_mask | bos_mask, -1)
    labels, observed, _ = build_horizon_targets(ix, x, age, horizon_y, patient_ages, followup, horizons, device, censor_aware=True)
    valid_dynamic = (x > 0) & (y > 1) & ~static_mask & ~bos_mask
    positions = torch.where(
        valid_dynamic,
        torch.arange(x.size(1), device=x.device).view(1, -1),
        torch.full_like(x, -1),
    ).max(1).values
    keep = positions >= 0
    positions = positions.clamp_min(0)
    observed = observed * keep[:, None, None]
    # Zero all tokens after the landmark before entering the bidirectional encoder.
    future = torch.arange(x.size(1), device=x.device).view(1, -1) > positions[:, None]
    x_input = x.masked_fill(future, 0)
    static_input = static_mask & ~future
    allowed = medbert_attention_allowed(x_input, age, static_input)
    logits = model(x_input, age, static_input, attention_allowed=allowed, prediction_positions=positions)["risk_logits"]
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = (raw * observed).sum() / observed.sum().clamp_min(1.0)
    return loss, logits, labels, observed


@torch.no_grad()
def evaluate(model, stage, data, p2i, static, prefix, anchor, protocol, manifest, patient_ages, followup, args, generator):
    model.eval()
    losses = []
    risk_scores, risk_labels, risk_masks = [], [], []
    horizons = tuple(float(value) for value in protocol["horizons_years"])
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,), generator=generator, device=args.device)
        x, age, y, _, static_mask, bos_mask = batch(ix, data, p2i, static, prefix, anchor, protocol, manifest, args.device, "random")
        if stage == "pretraining":
            loss = mlm_loss(model, x, age, static_mask, bos_mask, manifest, generator)
        else:
            loss, logits, labels, observed = horizon_objective(
                model, ix, x, age, y, static_mask, bos_mask, patient_ages, followup, horizons, args.device
            )
            risk_scores.append(torch.sigmoid(logits).cpu().numpy())
            risk_labels.append(labels.cpu().numpy())
            risk_masks.append(observed.cpu().numpy().astype(bool))
        losses.append(float(loss.cpu()))
    model.train()
    metrics = {"val_loss": float(np.mean(losses))}
    if stage == "horizon":
        from semantic_delphi_ukb.horizon_control_metrics import binary_auc

        scores = np.concatenate(risk_scores)
        labels = np.concatenate(risk_labels)
        masks = np.concatenate(risk_masks)
        aucs = []
        for horizon in range(scores.shape[1]):
            for disease in range(scores.shape[2]):
                valid = masks[:, horizon, disease]
                metric = binary_auc(scores[valid, horizon, disease], labels[valid, horizon, disease])
                if np.isfinite(metric):
                    aucs.append(metric)
        metrics["val_horizon_auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
    return metrics


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.run_dir.exists() and any(path.name != "stdout.log" for path in args.run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {args.run_dir}")
    protocol = load_track_r_protocol(args.protocol)
    manifest = validate_track_r_data_manifest(args.data_dir, protocol)
    seed_everything(args.seed)
    diseases, token_groups = load_selected_disease_token_groups(ROOT / protocol["diseases_yaml"], args.data_dir)
    train_data, train_p2i, train_static = load_split(args.data_dir, "train", 0)
    val_data, val_p2i, val_static = load_split(args.data_dir, "val", 0)
    train_prefix, train_anchor = load_track_r_assets(args.data_dir, "train")
    val_prefix, val_anchor = load_track_r_assets(args.data_dir, "val")
    model = MedBERTR(variant_config(protocol, args.variant, manifest["vocab_size"], len(diseases))).to(args.device)
    if args.init_from_ckpt is not None:
        checkpoint = torch.load(args.init_from_ckpt, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    elif args.stage == "horizon":
        raise ValueError("horizon stage requires --init-from-ckpt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_ages = train_followup = val_ages = val_followup = None
    if args.stage == "horizon":
        train_ages, train_last = build_patient_disease_ages(train_data, train_p2i, token_groups, manifest["vocab_size"])
        val_ages, val_last = build_patient_disease_ages(val_data, val_p2i, token_groups, manifest["vocab_size"])
        train_followup = load_followup_end_ages(args.data_dir, "train", train_last, 0)
        val_followup = load_followup_end_ages(args.data_dir, "val", val_last, 0)
    monitor = RunMonitor(args.run_dir, {**vars(args), "model_config": model.config.__dict__, "protocol": protocol}, enable_tensorboard=not args.no_tensorboard)
    (args.run_dir / "model_size.json").write_text(json.dumps(parameter_report(model), indent=2), encoding="utf-8")
    write_source_manifest(args.run_dir, ROOT, Path(__file__), extra_paths=(ROOT / "src/semantic_delphi_ukb/medbert_r_model.py", ROOT / "src/semantic_delphi_ukb/track_r_contract.py"))
    generator = torch.Generator(device=torch.device(args.device).type)
    generator.manual_seed(args.seed)
    best_loss = float("inf")
    best_auc = -float("inf")
    try:
        monitor.mark_running(stage=args.stage, iteration=0)
        for iteration in range(args.max_iters + 1):
            if iteration % args.eval_interval == 0:
                metrics = evaluate(model, args.stage, val_data, val_p2i, val_static, val_prefix, val_anchor, protocol, manifest, val_ages, val_followup, args, generator)
                if args.stage == "pretraining":
                    improved = metrics["val_loss"] < best_loss
                    best_loss = min(best_loss, metrics["val_loss"])
                else:
                    current_auc = metrics["val_horizon_auc_mean"]
                    improved = bool(np.isfinite(current_auc) and current_auc > best_auc)
                    best_auc = max(best_auc, current_auc) if np.isfinite(current_auc) else best_auc
                row = {"iteration": iteration, **metrics, "best_val_loss": best_loss, "best_val_horizon_auc_mean": best_auc}
                monitor.log_epoch(row)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration, "config": model.config.__dict__}
                monitor.save_checkpoint(state, "last.pt")
                if improved:
                    monitor.save_checkpoint(state, "best_val_loss.pt" if args.stage == "pretraining" else "best_val_horizon_auc.pt")
                monitor.update_status(status="running", **row)
            if iteration == args.max_iters:
                break
            ix = torch.randint(len(train_p2i), (args.batch_size,), generator=generator, device=args.device)
            x, age, y, _, static_mask, bos_mask = batch(ix, train_data, train_p2i, train_static, train_prefix, train_anchor, protocol, manifest, args.device, "random")
            if args.stage == "pretraining":
                loss = mlm_loss(model, x, age, static_mask, bos_mask, manifest, generator)
            else:
                loss, _, _, _ = horizon_objective(
                    model, ix, x, age, y, static_mask, bos_mask, train_ages, train_followup,
                    tuple(protocol["horizons_years"]), args.device,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if iteration % 25 == 0:
                monitor.log_step(iteration, {"loss/train": loss, **monitor.system_metrics()})
        monitor.mark_finished(stage=args.stage, iteration=args.max_iters, best_val_loss=best_loss, best_val_horizon_auc_mean=best_auc)
    except BaseException as exc:
        monitor.mark_failed(exc, stage=args.stage)
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
