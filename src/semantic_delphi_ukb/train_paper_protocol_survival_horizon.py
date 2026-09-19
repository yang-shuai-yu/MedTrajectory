from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions  # noqa: E402
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.paper_run import (  # noqa: E402
    TwoStageRiskScheduler, load_protocol_config, resolve_path, restore_random_state,
    seed_everything, strip_compiled_prefix, write_source_manifest,
)
from semantic_delphi_ukb.train_architecture_risk_heads import load_followup_end_ages  # noqa: E402
from semantic_delphi_ukb.train_medtrajectory_horizon_risk import (  # noqa: E402
    resolve_diseases_yaml, selected_next_event_loss,
)
from semantic_delphi_ukb.train_paper_protocol_risk_heads import (  # noqa: E402
    load_split, summarize_auxiliary_metrics,
)
from semantic_delphi_ukb.tte_model import SurvivalHorizonConfig, SurvivalHorizonMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages, build_survival_horizon_batch, disease_specs_payload,
    load_selected_disease_token_groups,
)
from training_monitor import RunMonitor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Censor-aware survival-horizon posttraining for paper P0/P3.")
    parser.add_argument("--common-config", type=Path, default=REPO_DIR / "configs/paper_protocol_v1/common.json")
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--init-from-ckpt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--head-warmup-iters", type=int, default=None)
    parser.add_argument("--warmup-head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--joint-trunk-learning-rate", type=float, default=None)
    parser.add_argument("--joint-head-learning-rate", type=float, default=None)
    parser.add_argument("--next-event-loss-weight", type=float, default=None)
    parser.add_argument("--survival-bins", type=int, default=10)
    parser.add_argument("--survival-bin-years", type=float, default=1.0)
    parser.add_argument("--survival-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def make_model(checkpoint: dict, diseases: int, device: str, args) -> SurvivalHorizonMedTrajectory:
    model_args = dict(checkpoint["model_args"])
    model_args.update(num_tte_tasks=diseases, tte_loss_weight=0.0, survival_bins=args.survival_bins,
                      survival_bin_years=args.survival_bin_years, survival_loss_weight=1.0)
    model = SurvivalHorizonMedTrajectory(SurvivalHorizonConfig(**model_args))
    missing, unexpected = model.load_state_dict(strip_compiled_prefix(checkpoint["model"]), strict=False)
    allowed = {"tte_head.weight", "tte_head.bias", "survival_head.weight", "survival_head.bias"}
    if unexpected or set(missing) - allowed:
        raise RuntimeError(f"incompatible pretraining checkpoint: missing={missing}, unexpected={unexpected}")
    return model.to(device)


def set_phase(model, head_only: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("survival_head.") or (not head_only and not name.startswith("tte_head."))


def final_survival_loss(model, logits, pos, event, mask):
    batch = torch.arange(logits.size(0), device=logits.device)
    return model._survival_loss(logits[batch, pos], event[batch, pos], mask[batch, pos])


@torch.no_grad()
def estimate(model, data, p2i, static, disease_ages, followup, horizons, args):
    model.eval()
    survival_losses, next_losses, scores, labels, masks = [], [], [], [], []
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,))
        x, age, y, target_age, s = get_batch(ix, data, p2i, static, block_size=model.config.block_size,
            device=args.device, padding="random", select="random", no_event_token_rate=args.no_event_token_rate, cut_batch=True)
        event, survival_mask, horizon_event, horizon_mask = build_survival_horizon_batch(
            ix, age, x > 1, disease_ages, followup, args.survival_bins, args.survival_bin_years,
            horizons, args.device, censor_aware=True)
        lm_logits, _, _, _, survival_logits = model(x, age, s, y, target_age, validation_loss_mode=True)
        _, pos = last_prediction_positions(x, y)
        survival_losses.append(float(final_survival_loss(model, survival_logits, pos, event, survival_mask).cpu()))
        next_losses.append(float(selected_next_event_loss(model, lm_logits, x, age, y, target_age).cpu()))
        batch = torch.arange(len(ix), device=survival_logits.device)
        scores.append(model.horizon_risk_from_survival(survival_logits[batch, pos], horizons).cpu().numpy())
        labels.append(horizon_event[batch, pos].cpu().numpy())
        masks.append(horizon_mask[batch, pos].cpu().numpy().astype(bool))
    model.train()
    return {"survival_loss": float(np.mean(survival_losses)), "next_event_loss": float(np.mean(next_losses)),
            **summarize_auxiliary_metrics(np.concatenate(scores), np.concatenate(labels), np.concatenate(masks))}


def self_test(args) -> int:
    config = SurvivalHorizonConfig(block_size=12, vocab_size=64, n_layer=2, n_head=4, n_embd=32,
        static_dim=1, num_tte_tasks=3, survival_bins=10, survival_bin_years=1.0)
    model = SurvivalHorizonMedTrajectory(config).to(args.device)
    risks = model.horizon_risk_from_survival(torch.randn(4, 3, 10, device=args.device), [1, 5, 10])
    if not bool(torch.all(risks[:, 1:] >= risks[:, :-1])):
        raise RuntimeError("survival-derived risks are not monotonic")
    print(json.dumps({"self_test": True, "strictly_non_decreasing": True, "horizons": [1, 5, 10]}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)
    resolved = load_protocol_config(args.common_config, args.experiment_config)
    experiment_key = resolved["experiment"]["experiment_id"].split("_", 1)[0]
    enabled = resolved["common"]["survival_horizon"]["enabled_experiments"]
    if experiment_key not in enabled:
        raise ValueError(f"survival-horizon is configured only for {enabled}, not {experiment_key}")
    post = resolved["common"]["posttraining"]
    horizons = [float(v) for v in resolved["common"]["survival_horizon"]["horizons_years"]]
    args.head_warmup_iters = int(args.head_warmup_iters if args.head_warmup_iters is not None else post["head_warmup_iters"])
    args.joint_trunk_learning_rate = float(args.joint_trunk_learning_rate or post["joint_trunk_learning_rate"])
    args.joint_head_learning_rate = float(args.joint_head_learning_rate or post["joint_head_learning_rate"])
    args.next_event_loss_weight = float(args.next_event_loss_weight if args.next_event_loss_weight is not None else post["next_event_loss_weight"])
    data_dir = args.data_dir or resolve_path(REPO_DIR, resolved["experiment"]["data_dir"])
    args.seed = int(args.seed if args.seed is not None else resolved["common"]["training_seeds"][0])
    seed_everything(args.seed)
    diseases, token_groups = load_selected_disease_token_groups(resolve_diseases_yaml(args.diseases_yaml), data_dir)
    checkpoint = torch.load(args.init_from_ckpt, map_location=args.device, weights_only=False)
    model = make_model(checkpoint, len(diseases), args.device, args)
    train_data, train_p2i, train_static = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(data_dir, "val", args.max_patients)
    train_ages, train_last = build_patient_disease_ages(train_data, train_p2i, token_groups, model.config.vocab_size)
    val_ages, val_last = build_patient_disease_ages(val_data, val_p2i, token_groups, model.config.vocab_size)
    train_followup = load_followup_end_ages(data_dir, "train", train_last, args.max_patients)
    val_followup = load_followup_end_ages(data_dir, "val", val_last, args.max_patients)
    trunk = [p for n, p in model.named_parameters() if not n.startswith(("survival_head.", "tte_head."))]
    head = list(model.survival_head.parameters())
    optimizer = torch.optim.AdamW([{"params": trunk}, {"params": head}], weight_decay=args.weight_decay)
    scheduler = TwoStageRiskScheduler(optimizer, args.head_warmup_iters, args.max_iters,
        args.joint_trunk_learning_rate, args.joint_head_learning_rate, args.warmup_head_learning_rate)
    config = {**vars(args), **resolved, "data_dir": str(data_dir), "horizons": horizons, "diseases": disease_specs_payload(diseases)}
    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    write_source_manifest(args.run_dir, REPO_DIR, Path(__file__))
    start_iter, global_step, best_val_loss = 0, 0, float("inf")
    if args.resume is not None:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(strip_compiled_prefix(state["model"])); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); restore_random_state(state["random_state"])
        start_iter, global_step, best_val_loss = int(state["iteration"]) + 1, int(state["global_step"]), float(state["best_val_loss"])
    last_iter = start_iter - 1
    try:
        monitor.mark_running(phase="head_warmup", iteration=start_iter, global_step=global_step)
        for iteration in range(start_iter, args.max_iters + 1):
            last_iter = iteration; head_only = iteration < args.head_warmup_iters
            phase = "head_warmup" if head_only else "joint_finetuning"; set_phase(model, head_only); scheduler.step(iteration)
            if iteration % args.eval_interval == 0:
                metrics = estimate(model, val_data, val_p2i, val_static, val_ages, val_followup, horizons, args)
                val_loss = metrics["survival_loss"] + args.next_event_loss_weight * metrics["next_event_loss"]
                improved = val_loss < best_val_loss; best_val_loss = min(best_val_loss, val_loss)
                row = {"iteration": iteration, "global_step": global_step, "phase": phase, "val_loss": val_loss,
                    "val_survival_loss": metrics["survival_loss"], "val_next_event_loss": metrics["next_event_loss"],
                    "val_aux_auprc_macro": metrics["auprc_macro"], "val_aux_brier_macro": metrics["brier_macro"],
                    "val_aux_calibration_intercept_macro": metrics["calibration_intercept_macro"],
                    "val_aux_calibration_slope_macro": metrics["calibration_slope_macro"], "best_val_loss": best_val_loss}
                monitor.log_epoch(row)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": None, "iteration": iteration, "global_step": global_step, "best_val_loss": best_val_loss,
                    "model_args": model.config.__dict__.copy(), "model_family": "paper_causal_survival_horizon_v1",
                    "config": config, "random_state": monitor.capture_random_state()}
                monitor.save_checkpoint(state, "last.pt")
                if improved: monitor.save_checkpoint(state, "best_val_loss.pt")
                monitor.update_status(status="running", **row); print(json.dumps(row, sort_keys=True), flush=True)
            if iteration == args.max_iters: break
            started = time.perf_counter(); ix = torch.randint(len(train_p2i), (args.batch_size,))
            x, age, y, target_age, s = get_batch(ix, train_data, train_p2i, train_static, block_size=model.config.block_size,
                device=args.device, padding="random", select="random", no_event_token_rate=args.no_event_token_rate, cut_batch=True)
            event, survival_mask, _, _ = build_survival_horizon_batch(ix, age, x > 1, train_ages, train_followup,
                args.survival_bins, args.survival_bin_years, horizons, args.device, censor_aware=True)
            lm_logits, _, _, _, survival_logits = model(x, age, s, y, target_age); _, pos = last_prediction_positions(x, y)
            survival_loss = final_survival_loss(model, survival_logits, pos, event, survival_mask)
            next_loss = selected_next_event_loss(model, lm_logits, x, age, y, target_age)
            loss = args.survival_loss_weight * survival_loss + args.next_event_loss_weight * next_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); optimizer.step(); global_step += 1
            if global_step % args.log_every == 0:
                elapsed = time.perf_counter() - started
                monitor.log_step(global_step, {"loss/train_total": loss, "loss/train_survival": survival_loss,
                    "loss/train_next_event": next_loss, "optimization/trunk_learning_rate": optimizer.param_groups[0]["lr"],
                    "optimization/head_learning_rate": optimizer.param_groups[1]["lr"], "optimization/gradient_norm": grad_norm,
                    "performance/step_time_seconds": elapsed, "performance/samples_per_second": args.batch_size / max(elapsed, 1e-9),
                    **monitor.system_metrics()})
        monitor.mark_finished(phase="finished", iteration=last_iter, global_step=global_step, best_val_loss=best_val_loss)
    except BaseException as exc:
        monitor.mark_failed(exc, iteration=last_iter, global_step=global_step); raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
