from __future__ import annotations

import argparse
import json
import math
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
    TwoStageRiskScheduler,
    load_protocol_config,
    resolve_path,
    restore_random_state,
    seed_everything,
    strip_compiled_prefix,
    write_source_manifest,
)
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    build_horizon_targets,
    load_followup_end_ages,
    parse_horizons,
)
from semantic_delphi_ukb.train_medtrajectory_horizon_risk import (  # noqa: E402
    final_position_horizon_loss,
    resolve_diseases_yaml,
    selected_next_event_loss,
)
from semantic_delphi_ukb.tte_model import HorizonRiskConfig, HorizonRiskMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    disease_specs_payload,
    load_selected_disease_token_groups,
)
from training_monitor import RunMonitor  # noqa: E402
from utils import get_p2i  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Censor-aware monotonic risk posttraining for paper_protocol_v1.")
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
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--next-event-loss-weight", type=float, default=None)
    parser.add_argument("--risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
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


def make_model(checkpoint: dict, disease_count: int, horizons: list[float], device: str) -> HorizonRiskMedTrajectory:
    model_args = dict(checkpoint["model_args"])
    model_args.update(
        num_tte_tasks=disease_count,
        num_horizons=len(horizons),
        tte_loss_weight=0.0,
        horizon_risk_loss_weight=1.0,
        monotonic_horizon_risk=True,
    )
    model = HorizonRiskMedTrajectory(HorizonRiskConfig(**model_args))
    state = strip_compiled_prefix(checkpoint["model"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {"tte_head.weight", "tte_head.bias", "horizon_risk_head.weight", "horizon_risk_head.bias"}
    if unexpected or set(missing) - allowed:
        raise RuntimeError(f"incompatible pretraining checkpoint: missing={missing}, unexpected={unexpected}")
    return model.to(device)


def set_training_phase(model: HorizonRiskMedTrajectory, head_only: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("horizon_risk_head.") or (not head_only and not name.startswith("tte_head."))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def calibration_fit(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if labels.min() == labels.max():
        return float("nan"), float("nan")
    x = np.column_stack([np.ones(len(probabilities)), np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))])
    beta = np.zeros(2, dtype=np.float64)
    for _ in range(25):
        eta = np.clip(x @ beta, -30, 30)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(fitted * (1.0 - fitted), 1e-6, None)
        hessian = x.T @ (weights[:, None] * x)
        update = np.linalg.solve(hessian + np.eye(2) * 1e-8, x.T @ (labels - fitted))
        beta += update
        if np.linalg.norm(update) < 1e-7:
            break
    return float(beta[0]), float(beta[1])


def summarize_auxiliary_metrics(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    auprc, brier, intercepts, slopes = [], [], [], []
    for horizon_idx in range(scores.shape[1]):
        for disease_idx in range(scores.shape[2]):
            valid = mask[:, horizon_idx, disease_idx]
            cell_labels = labels[valid, horizon_idx, disease_idx]
            if len(cell_labels) == 0:
                continue
            cell_scores = scores[valid, horizon_idx, disease_idx]
            brier.append(float(np.mean((cell_scores - cell_labels) ** 2)))
            ap = average_precision(cell_scores, cell_labels)
            if math.isfinite(ap):
                auprc.append(ap)
            intercept, slope = calibration_fit(cell_scores, cell_labels)
            if math.isfinite(slope):
                intercepts.append(intercept)
                slopes.append(slope)
    return {
        "auprc_macro": float(np.mean(auprc)) if auprc else float("nan"),
        "brier_macro": float(np.mean(brier)) if brier else float("nan"),
        "calibration_intercept_macro": float(np.mean(intercepts)) if intercepts else float("nan"),
        "calibration_slope_macro": float(np.mean(slopes)) if slopes else float("nan"),
    }


@torch.no_grad()
def estimate(model, data, p2i, static, disease_ages, followup_ages, horizons, args) -> dict[str, float]:
    model.eval()
    risk_losses, next_losses, score_parts, label_parts, mask_parts = [], [], [], [], []
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,))
        x, age, y, target_age, s = get_batch(
            ix, data, p2i, static, block_size=model.config.block_size, device=args.device,
            padding="random", select="random", no_event_token_rate=args.no_event_token_rate, cut_batch=True,
        )
        logits, _, _, _, risk_logits = model(x, age, s, y, target_age, validation_loss_mode=True)
        _, pos = last_prediction_positions(x, y)
        labels, mask, _ = build_horizon_targets(
            ix, x, age, y, disease_ages, followup_ages, horizons, args.device, censor_aware=True
        )
        risk_losses.append(float(final_position_horizon_loss(risk_logits, pos, labels, mask).cpu()))
        next_losses.append(float(selected_next_event_loss(model, logits, x, age, y, target_age).cpu()))
        batch_index = torch.arange(len(ix), device=risk_logits.device)
        score_parts.append(torch.sigmoid(risk_logits[batch_index, pos]).cpu().numpy())
        label_parts.append(labels.cpu().numpy())
        mask_parts.append(mask.cpu().numpy().astype(bool))
    model.train()
    auxiliary = summarize_auxiliary_metrics(
        np.concatenate(score_parts), np.concatenate(label_parts), np.concatenate(mask_parts)
    )
    return {
        "risk_loss": float(np.mean(risk_losses)),
        "next_event_loss": float(np.mean(next_losses)),
        **auxiliary,
    }


def self_test(args: argparse.Namespace) -> int:
    config = HorizonRiskConfig(
        block_size=12, vocab_size=64, n_layer=2, n_head=4, n_embd=32, static_dim=1,
        num_tte_tasks=3, num_horizons=3, monotonic_horizon_risk=True,
    )
    model = HorizonRiskMedTrajectory(config).to(args.device)
    x = torch.randint(2, 64, (4, 12), device=args.device)
    age = torch.sort(torch.rand(4, 12, device=args.device) * 30000, dim=1).values
    y = torch.randint(2, 64, (4, 12), device=args.device)
    logits, _, _, _, risk_logits = model(x, age, torch.randn(4, 1, device=args.device), y, age + 10)
    next_loss = selected_next_event_loss(model, logits, x, age, y, age + 10)
    delta = risk_logits[:, :, 1:] - risk_logits[:, :, :-1]
    if not bool(torch.all(delta > 0)):
        raise RuntimeError("risk logits are not strictly monotonic")
    (risk_logits.mean() + 0.2 * next_loss).backward()
    set_training_phase(model, True)
    trunk_frozen = all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith(("horizon_risk_head.", "tte_head.")))
    print(json.dumps({"self_test": True, "strictly_monotonic": True, "trunk_frozen": trunk_frozen, "next_event_weight": 0.2}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)
    resolved = load_protocol_config(args.common_config, args.experiment_config)
    post = resolved["common"]["posttraining"]
    horizons = [float(value) for value in post["horizons_years"]]
    args.head_warmup_iters = int(args.head_warmup_iters if args.head_warmup_iters is not None else post["head_warmup_iters"])
    args.joint_trunk_learning_rate = float(args.joint_trunk_learning_rate or post["joint_trunk_learning_rate"])
    args.joint_head_learning_rate = float(args.joint_head_learning_rate or post["joint_head_learning_rate"])
    args.next_event_loss_weight = float(args.next_event_loss_weight if args.next_event_loss_weight is not None else post["next_event_loss_weight"])
    data_dir = args.data_dir or resolve_path(REPO_DIR, resolved["experiment"]["data_dir"])
    args.seed = int(args.seed if args.seed is not None else resolved["common"]["training_seeds"][0])
    seed_everything(args.seed)
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    diseases, token_groups = load_selected_disease_token_groups(diseases_yaml, data_dir)
    checkpoint = torch.load(args.init_from_ckpt, map_location=args.device, weights_only=False)
    model = make_model(checkpoint, len(diseases), horizons, args.device)
    train_data, train_p2i, train_static = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(data_dir, "val", args.max_patients)
    train_ages, train_last = build_patient_disease_ages(train_data, train_p2i, token_groups, model.config.vocab_size)
    val_ages, val_last = build_patient_disease_ages(val_data, val_p2i, token_groups, model.config.vocab_size)
    train_followup = load_followup_end_ages(data_dir, "train", train_last, args.max_patients)
    val_followup = load_followup_end_ages(data_dir, "val", val_last, args.max_patients)
    trunk = [p for n, p in model.named_parameters() if not n.startswith(("horizon_risk_head.", "tte_head."))]
    head = list(model.horizon_risk_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": trunk, "lr": args.joint_trunk_learning_rate},
        {"params": head, "lr": args.warmup_head_learning_rate},
    ], weight_decay=args.weight_decay)
    scheduler = TwoStageRiskScheduler(
        optimizer, args.head_warmup_iters, args.max_iters, args.joint_trunk_learning_rate,
        args.joint_head_learning_rate, args.warmup_head_learning_rate,
    )
    config = {**vars(args), **resolved, "data_dir": str(data_dir), "horizons": horizons, "diseases": disease_specs_payload(diseases)}
    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    write_source_manifest(args.run_dir, REPO_DIR, Path(__file__))
    start_iter, global_step, best_val_loss = 0, 0, float("inf")
    if args.resume is not None:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(strip_compiled_prefix(state["model"]))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_iter, global_step = int(state["iteration"]) + 1, int(state["global_step"])
        best_val_loss = float(state["best_val_loss"])
        restore_random_state(state["random_state"])
    last_iter = start_iter - 1
    try:
        monitor.mark_running(phase="head_warmup", iteration=start_iter, global_step=global_step)
        for iteration in range(start_iter, args.max_iters + 1):
            last_iter = iteration
            head_only = iteration < args.head_warmup_iters
            phase = "head_warmup" if head_only else "joint_finetuning"
            set_training_phase(model, head_only)
            scheduler.step(iteration)
            if iteration % args.eval_interval == 0:
                metrics = estimate(model, val_data, val_p2i, val_static, val_ages, val_followup, horizons, args)
                val_loss = metrics["risk_loss"] + args.next_event_loss_weight * metrics["next_event_loss"]
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                row = {"iteration": iteration, "global_step": global_step, "phase": phase, "val_loss": val_loss,
                       "val_risk_loss": metrics["risk_loss"], "val_next_event_loss": metrics["next_event_loss"],
                       "val_aux_auprc_macro": metrics["auprc_macro"], "val_aux_brier_macro": metrics["brier_macro"],
                       "val_aux_calibration_intercept_macro": metrics["calibration_intercept_macro"],
                       "val_aux_calibration_slope_macro": metrics["calibration_slope_macro"],
                       "trunk_learning_rate": optimizer.param_groups[0]["lr"], "head_learning_rate": optimizer.param_groups[1]["lr"],
                       "best_val_loss": best_val_loss}
                monitor.log_epoch(row)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "scaler": None, "iteration": iteration, "global_step": global_step, "best_val_loss": best_val_loss,
                         "model_args": model.config.__dict__.copy(), "model_family": "paper_causal_horizon_risk_v1",
                         "config": config, "random_state": monitor.capture_random_state()}
                monitor.save_checkpoint(state, "last.pt")
                if improved:
                    monitor.save_checkpoint(state, "best_val_loss.pt")
                monitor.update_status(status="running", **row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if iteration == args.max_iters:
                break
            step_start = time.perf_counter()
            ix = torch.randint(len(train_p2i), (args.batch_size,))
            x, age, y, target_age, s = get_batch(ix, train_data, train_p2i, train_static,
                block_size=model.config.block_size, device=args.device, padding="random", select="random",
                no_event_token_rate=args.no_event_token_rate, cut_batch=True)
            labels, mask, _ = build_horizon_targets(ix, x, age, y, train_ages, train_followup, horizons, args.device, censor_aware=True)
            logits, _, _, _, risk_logits = model(x, age, s, y, target_age)
            _, pos = last_prediction_positions(x, y)
            risk_loss = final_position_horizon_loss(risk_logits, pos, labels, mask)
            next_loss = selected_next_event_loss(model, logits, x, age, y, target_age)
            loss = args.risk_loss_weight * risk_loss + args.next_event_loss_weight * next_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            global_step += 1
            if global_step % args.log_every == 0:
                elapsed = time.perf_counter() - step_start
                monitor.log_step(global_step, {"loss/train_total": loss, "loss/train_risk": risk_loss,
                    "loss/train_next_event": next_loss, "optimization/trunk_learning_rate": optimizer.param_groups[0]["lr"],
                    "optimization/head_learning_rate": optimizer.param_groups[1]["lr"], "optimization/gradient_norm": grad_norm,
                    "performance/step_time_seconds": elapsed, "performance/samples_per_second": args.batch_size / max(elapsed, 1e-9),
                    **monitor.system_metrics()})
        monitor.mark_finished(phase="finished", iteration=last_iter, global_step=global_step, best_val_loss=best_val_loss)
    except BaseException as exc:
        monitor.mark_failed(exc, iteration=last_iter, global_step=global_step)
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
