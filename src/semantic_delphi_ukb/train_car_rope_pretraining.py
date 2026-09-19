"""CARoPE all-position pretraining stage.

This is the first stage of the new-method protocol. It trains only on train
and monitors val; the resulting CARoPE checkpoint is consumed by
``train_car_rope.py --init-from-ckpt`` for horizon fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.car_rope_model import CARoPEConfig  # noqa: E402
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    STATIC_CONDITIONING_MODES,
    load_track_r_static_features,
)
from semantic_delphi_ukb.paper_run import cosine_warmup_scheduler, seed_everything, write_source_manifest  # noqa: E402
from semantic_delphi_ukb.tte_targets import disease_specs_payload, load_selected_disease_token_groups  # noqa: E402
from semantic_delphi_ukb.train_car_rope import (  # noqa: E402
    build_training_batch,
    load_rope_wavelengths,
    load_split,
    make_model,
    parse_horizons,
    parse_positive_floats,
    resolve_diseases_yaml,
    write_additive_rope_diagnostics,
    write_model_size_manifest,
)
from training_monitor import RunMonitor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain CARoPE on train and monitor validation.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data/paper_protocol_v1/multitype")
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-learning-rate", type=float, default=6e-5)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--time-gap-loss-weight", type=float, default=0.2)
    parser.add_argument("--next-event-loss-weight", type=float, default=0.2)
    parser.add_argument("--horizon-risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-age-encoding", choices=("true", "false"), default="false")
    parser.add_argument("--use-age-rope", choices=("true", "false"), default="true")
    parser.add_argument("--use-relative-horizon-query", choices=("true", "false"), default="true")
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--rope-scales", default="0.25,1,4")
    parser.add_argument("--rope-initial-gate", type=float, default=0.1)
    parser.add_argument("--age-rope-variant", choices=("legacy", "additive_v2_2"), default="legacy")
    parser.add_argument("--rope-wavelengths-manifest", type=Path, default=None)
    parser.add_argument("--rope-max-scale", type=float, default=2.0)
    parser.add_argument("--residual-rope-mode", choices=("none",), default="none")
    parser.add_argument("--residual-rope-initial-alpha", type=float, default=0.01)
    parser.add_argument("--residual-rope-fixed-alpha", type=float, default=1.0)
    parser.add_argument("--residual-rope-phase-origin", choices=("recruitment_age",), default="recruitment_age")
    parser.add_argument("--horizons", default="1,5")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--track-r-protocol", type=Path, default=None)
    parser.add_argument("--include-static-prefix", choices=("true", "false"), default="false")
    parser.add_argument("--static-conditioning", choices=STATIC_CONDITIONING_MODES, default="none")
    parser.add_argument(
        "--static-fusion-stage",
        choices=("pre_transformer", "post_transformer"),
        default="pre_transformer",
    )
    return parser


@torch.no_grad()
def evaluate(model, data, p2i, static, args, prefix_ids=None, anchor_ages=None):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,))
        (
            x, age, y, target_age, s, static_mask, bos_mask,
            next_mask, time_mask, static_feature_mask,
        ) = build_training_batch(
            ix, data, p2i, static, args, padding="regular", select="random",
            prefix_ids=prefix_ids, anchor_ages=anchor_ages,
        )
        _, parts, *_ = model(
            x, age, s, y, target_age, validation_loss_mode=True,
            static_token_mask=static_mask, next_event_mask=next_mask, time_loss_mask=time_mask,
            static_feature_mask=static_feature_mask,
            bos_token_mask=bos_mask,
        )
        gap_loss = parts.get("loss_gap", torch.zeros((), device=age.device))
        losses.append(float((parts["loss_ce"] + parts["loss_dt"] + model.config.time_gap_loss_weight * gap_loss).cpu()))
    model.train()
    return {"val_pretraining_loss": float(sum(losses) / max(len(losses), 1))}


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.horizons = parse_horizons(args.horizons)
    args.rope_scales = parse_positive_floats(args.rope_scales, "--rope-scales")
    args.rope_wavelengths_days = ()
    if args.self_test:
        # Reuse the model-level self-test without touching the dataset.
        from semantic_delphi_ukb.train_car_rope import self_test
        return self_test(args)
    args.dynamic_context_length = args.block_size
    protocol = None
    if args.track_r_protocol is not None:
        from semantic_delphi_ukb.track_r_batch import load_track_r_assets, load_track_r_bos_token_id, validate_track_r_data_manifest
        from semantic_delphi_ukb.track_r_contract import load_track_r_protocol

        protocol = load_track_r_protocol(args.track_r_protocol)
        args.track_r_protocol_sha256 = protocol.get("protocol_manifest_sha256")
        validate_track_r_data_manifest(args.data_dir, protocol)
        args.dynamic_context_length = int(protocol["dynamic_context_length"])
        args.track_r_bos_token_id = load_track_r_bos_token_id(args.data_dir)
        args.block_size = args.dynamic_context_length + 1 + (
            int(protocol["static_prefix"]["fixed_length"]) if args.include_static_prefix == "true" else 0
        )
        if args.static_conditioning != "none" and args.include_static_prefix == "true":
            raise ValueError("static residual conditioning cannot be combined with static prefix tokens")
    elif args.static_conditioning != "none":
        raise ValueError("--static-conditioning is only valid with --track-r-protocol")
    args.rope_wavelengths_days = load_rope_wavelengths(
        args.rope_wavelengths_manifest,
        protocol.get("wavelength_contract") if protocol is not None else None,
        require_expected=args.age_rope_variant == "additive_v2_2",
    )
    if args.age_rope_variant == "additive_v2_2":
        if args.track_r_protocol is None or args.include_static_prefix != "true":
            raise ValueError("additive v2.2 RoPE is registered only for Track R TokenStatic")
        if not args.rope_wavelengths_days:
            raise ValueError("additive v2.2 RoPE requires --rope-wavelengths-manifest")
    if args.run_dir.exists() and any(p.name != "stdout.log" for p in args.run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {args.run_dir}")
    seed_everything(args.seed)
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    diseases, _token_groups = load_selected_disease_token_groups(diseases_yaml, args.data_dir)
    train_data, train_p2i, train_static = load_split(args.data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(args.data_dir, "val", args.max_patients)
    train_prefix = train_anchor = val_prefix = val_anchor = None
    if protocol is not None:
        train_prefix, train_anchor = load_track_r_assets(args.data_dir, "train", args.max_patients)
        val_prefix, val_anchor = load_track_r_assets(args.data_dir, "val", args.max_patients)
        train_static = load_track_r_static_features(
            args.data_dir, "train", args.static_conditioning, args.max_patients
        )
        val_static = load_track_r_static_features(
            args.data_dir, "val", args.static_conditioning, args.max_patients
        )
    model = make_model(args, args.data_dir, len(diseases)).to(args.device)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.99), "cuda" if "cuda" in args.device else "cpu")
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_iters, args.max_iters, args.min_learning_rate / args.learning_rate)
    config = {**vars(args), "data_dir": str(args.data_dir), "model_args": model.config.__dict__.copy(), "diseases": disease_specs_payload(diseases), "stage": "pretraining"}
    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    write_model_size_manifest(args.run_dir, model)
    write_source_manifest(args.run_dir, REPO_DIR, Path(__file__), extra_paths=(
        REPO_DIR / "src/semantic_delphi_ukb/car_rope_model.py",
        REPO_DIR / "src/semantic_delphi_ukb/train_car_rope.py",
        REPO_DIR / "src/semantic_delphi_ukb/train_car_rope_pretraining.py",
        REPO_DIR / "src/semantic_delphi_ukb/multitype_batch.py",
        REPO_DIR / "src/semantic_delphi_ukb/track_r_batch.py",
        REPO_DIR / "src/semantic_delphi_ukb/track_r_contract.py",
        REPO_DIR / "src/semantic_delphi_ukb/train_architecture_risk_heads.py",
        REPO_DIR / "src/semantic_delphi_ukb/tte_targets.py",
        REPO_DIR / "scripts/training_monitor.py",
    ))
    best_loss = float("inf")
    last_iter = -1
    try:
        monitor.mark_running(phase="pretraining", iteration=0, global_step=0)
        for iteration in range(args.max_iters + 1):
            last_iter = iteration
            if iteration % args.eval_interval == 0:
                metrics = evaluate(model, val_data, val_p2i, val_static, args, val_prefix, val_anchor)
                improved = metrics["val_pretraining_loss"] < best_loss
                best_loss = min(best_loss, metrics["val_pretraining_loss"])
                row = {"iteration": iteration, **metrics, "best_val_pretraining_loss": best_loss}
                monitor.log_epoch(row)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "iteration": iteration, "model_args": model.config.__dict__.copy(),
                         "model_family": "CARoPE_additive_v2_2" if model.config.age_rope_variant == "additive_v2_2" else "CARoPE_v1",
                         "stage": "pretraining",
                         "protocol_manifest_sha256": getattr(args, "track_r_protocol_sha256", None),
                         "config": config}
                monitor.save_checkpoint(state, "last.pt")
                if improved:
                    monitor.save_checkpoint(state, "best_val_loss.pt")
                monitor.update_status(status="running", **row)
                print(json.dumps(row, sort_keys=True), flush=True)
                write_additive_rope_diagnostics(args.run_dir, model)
            if iteration == args.max_iters:
                break
            step_start = time.perf_counter()
            ix = torch.randint(len(train_p2i), (args.batch_size,))
            (
                x, age, y, target_age, s, static_mask, bos_mask,
                next_mask, time_mask, static_feature_mask,
            ) = build_training_batch(
                ix, train_data, train_p2i, train_static, args, padding="regular", select="random",
                prefix_ids=train_prefix, anchor_ages=train_anchor,
            )
            _, parts, *_ = model(
                x, age, s, y, target_age,
                static_token_mask=static_mask, next_event_mask=next_mask, time_loss_mask=time_mask,
                static_feature_mask=static_feature_mask,
                bos_token_mask=bos_mask,
            )
            gap_loss = parts.get("loss_gap", torch.zeros((), device=age.device))
            loss = parts["loss_ce"] + parts["loss_dt"] + model.config.time_gap_loss_weight * gap_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if iteration % args.log_every == 0:
                monitor.log_step(iteration, {"loss/train_total": loss, "loss/train_time_gap": gap_loss,
                                              "optimization/gradient_norm": grad_norm, "performance/step_time_seconds": time.perf_counter() - step_start,
                                              **monitor.system_metrics()})
        monitor.mark_finished(phase="finished", iteration=last_iter, global_step=last_iter, best_val_pretraining_loss=best_loss)
    except BaseException as exc:
        monitor.mark_failed(exc, iteration=last_iter, global_step=last_iter)
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
