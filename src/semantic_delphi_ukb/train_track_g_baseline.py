"""Train the ETHOS-Matched and Foresight-Matched Track G baselines."""

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

from semantic_delphi_ukb.paper_run import (  # noqa: E402
    cosine_warmup_scheduler,
    restore_random_state,
    seed_everything,
    write_source_manifest,
)
from semantic_delphi_ukb.track_g_contract import (  # noqa: E402
    TRAINED_TRACK_G_FAMILIES,
    assert_training_allowed,
    load_track_g_protocol,
    resolve_repo_path,
)
from semantic_delphi_ukb.track_g_models import (  # noqa: E402
    build_matched_model,
    model_config_payload,
)
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    load_track_r_assets,
    load_track_r_bos_token_id,
    load_track_r_static_features,
    validate_track_r_data_manifest,
)
from semantic_delphi_ukb.train_car_rope import build_training_batch, load_split  # noqa: E402
from semantic_delphi_ukb.tte_targets import load_selected_disease_token_groups  # noqa: E402
from training_monitor import RunMonitor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_DIR / "configs/track_g_v1/TRACK_G_v1.json",
    )
    parser.add_argument("--model", choices=TRAINED_TRACK_G_FAMILIES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _runtime_args(protocol: dict, args: argparse.Namespace) -> argparse.Namespace:
    source = protocol["_source_track_r"]
    training = protocol["training"]
    args.track_r_protocol = protocol["_source_track_r_path"]
    args.track_r_protocol_sha256 = source["protocol_manifest_sha256"]
    args.dynamic_context_length = int(protocol["matched_input_contract"]["dynamic_context_length"])
    args.block_size = args.dynamic_context_length + int(source["static_prefix"]["fixed_length"]) + 1
    args.include_static_prefix = "true"
    args.static_conditioning = "none"
    args.no_event_token_rate = int(training["no_event_token_rate"])
    args.batch_size = int(training["batch_size"])
    args.eval_interval = int(training["eval_interval"])
    args.eval_iters = int(training["eval_iters"])
    args.log_every = int(training["log_every"])
    args.learning_rate = float(training["learning_rate"])
    args.min_learning_rate = float(training["min_learning_rate"])
    args.warmup_iters = int(training["warmup_iters"])
    args.weight_decay = float(training["weight_decay"])
    args.gradient_clip_norm = float(training["gradient_clip_norm"])
    args.max_iters = int(args.max_iters if args.max_iters is not None else training["max_iters"])
    return args


def _load_embeddings(data_dir: Path) -> np.ndarray:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    return np.load(data_dir / manifest["semantic_output"]).astype(np.float32)


def _make_model(protocol: dict, data_dir: Path, family: str, num_diseases: int):
    source = protocol["_source_track_r"]
    embeddings = _load_embeddings(data_dir)
    return build_matched_model(
        family,
        block_size=int(protocol["matched_input_contract"]["dynamic_context_length"])
        + int(source["static_prefix"]["fixed_length"])
        + 1,
        vocab_size=int(embeddings.shape[0]),
        semantic_embedding_dim=int(embeddings.shape[1]),
        pretrained_token_embeddings=embeddings,
        num_diseases=num_diseases,
        architecture=protocol["architecture"],
        horizons_years=source["horizons_years"],
    )


@torch.no_grad()
def evaluate(model, data, p2i, static, prefix, anchors, args) -> dict[str, float]:
    model.eval()
    total = ce = dt = 0.0
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,))
        batch = build_training_batch(
            ix,
            data,
            p2i,
            static,
            args,
            padding="regular",
            select="random",
            prefix_ids=prefix,
            anchor_ages=anchors,
        )
        x, age, targets, targets_age, features, static_mask, bos_mask, next_mask, time_mask, feature_mask = batch
        _, parts, *_ = model(
            x,
            age,
            features,
            targets,
            targets_age,
            validation_loss_mode=True,
            static_token_mask=static_mask,
            bos_token_mask=bos_mask,
            next_event_mask=next_mask,
            time_loss_mask=time_mask,
            static_feature_mask=feature_mask,
        )
        ce += float(parts["loss_ce"].cpu())
        dt += float(parts["loss_dt"].cpu())
        total += float((parts["loss_ce"] + parts["loss_dt"]).cpu())
    model.train()
    scale = 1.0 / max(args.eval_iters, 1)
    return {
        "val_loss": total * scale,
        "val_loss_ce": ce * scale,
        "val_loss_dt": dt * scale,
    }


def checkpoint_payload(model, optimizer, scheduler, monitor, args, protocol, iteration, best_loss, config):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": None,
        "iteration": int(iteration),
        "global_step": int(iteration),
        "best_val_loss": float(best_loss),
        "model_args": model_config_payload(model),
        "track_g_family": args.model,
        "stage": "pretraining",
        "seed": int(args.seed),
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "source_track_r_protocol_sha256": protocol["source_track_r_protocol_sha256"],
        "config": config,
        "random_state": monitor.capture_random_state(),
    }


def resume_global_step(checkpoint: dict) -> int:
    """Return completed optimizer steps; checkpoints are saved before the next step."""
    step = int(checkpoint["global_step"])
    if int(checkpoint["iteration"]) != step:
        raise ValueError("checkpoint iteration/global_step mismatch")
    return step


def self_test(device: str = "cpu") -> int:
    architecture = {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
        "dropout": 0.0,
        "ethos_gap_bucket_upper_days": [1.0, 7.0, 30.0],
    }
    embeddings = np.random.default_rng(7).normal(size=(32, 8)).astype(np.float32)
    for family in TRAINED_TRACK_G_FAMILIES:
        model = build_matched_model(
            family,
            block_size=8,
            vocab_size=32,
            semantic_embedding_dim=8,
            pretrained_token_embeddings=embeddings,
            num_diseases=2,
            architecture=architecture,
            horizons_years=(1.0, 5.0),
        ).to(device)
        x = torch.tensor([[20, 21, 22, 23, 24, 4, 5, 6]], device=device)
        age = torch.tensor([[200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 130.0, 160.0]], device=device)
        targets = torch.tensor([[-1, -1, -1, -1, 4, 5, 6, 7]], device=device)
        target_age = torch.tensor([[-10000.0] * 4 + [100.0, 130.0, 160.0, 190.0]], device=device)
        static = torch.zeros_like(x, dtype=torch.bool); static[:, :4] = True
        bos = torch.zeros_like(x, dtype=torch.bool); bos[:, 4] = True
        next_mask = targets > 1
        time_mask = next_mask & ~bos & (target_age > age)
        _, parts, *_ = model(
            x,
            age,
            None,
            targets,
            target_age,
            static_token_mask=static,
            bos_token_mask=bos,
            next_event_mask=next_mask,
            time_loss_mask=time_mask,
        )
        parts["loss"].backward()
        if not bool(torch.isfinite(parts["loss"])):
            raise AssertionError(f"non-finite self-test loss for {family}")
    print(json.dumps({"self_test": True, "families": list(TRAINED_TRACK_G_FAMILIES)}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args.device)
    protocol = load_track_g_protocol(args.protocol, REPO_DIR)
    assert_training_allowed(protocol)
    if args.seed not in protocol["seeds"]:
        raise ValueError(f"unregistered Track G seed: {args.seed}")
    args = _runtime_args(protocol, args)
    source = protocol["_source_track_r"]
    data_dir = args.data_dir or Path(source["output_data_dir"])
    validate_track_r_data_manifest(data_dir, source)
    args.track_r_bos_token_id = load_track_r_bos_token_id(data_dir)

    if args.resume is None:
        if args.run_dir.exists() and any(path.name != "stdout.log" for path in args.run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {args.run_dir}")
    elif not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "stdout.log").touch(exist_ok=True)

    seed_everything(args.seed)
    diseases_yaml = resolve_repo_path(REPO_DIR, source["diseases_yaml"])
    diseases, _ = load_selected_disease_token_groups(diseases_yaml, data_dir)
    train_data, train_p2i, _ = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, _ = load_split(data_dir, "val", args.max_patients)
    train_prefix, train_anchor = load_track_r_assets(data_dir, "train", args.max_patients)
    val_prefix, val_anchor = load_track_r_assets(data_dir, "val", args.max_patients)
    train_static = load_track_r_static_features(data_dir, "train", "none", args.max_patients)
    val_static = load_track_r_static_features(data_dir, "val", "none", args.max_patients)

    model = _make_model(protocol, data_dir, args.model, len(diseases)).to(args.device)
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (0.9, 0.99),
        "cuda" if "cuda" in args.device else "cpu",
    )
    scheduler = cosine_warmup_scheduler(
        optimizer,
        args.warmup_iters,
        args.max_iters,
        args.min_learning_rate / args.learning_rate,
    )
    config = {
        **vars(args),
        "protocol": str(args.protocol),
        "data_dir": str(data_dir),
        "model_args": model_config_payload(model),
    }
    monitor = RunMonitor(
        args.run_dir,
        None if args.resume is not None else config,
        enable_tensorboard=not args.no_tensorboard,
    )

    start_iteration = 0
    best_loss = float("inf")
    if args.resume is not None:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        if state.get("track_g_family") != args.model or int(state.get("seed", -1)) != args.seed:
            raise ValueError("resume checkpoint model/seed mismatch")
        if state.get("protocol_manifest_sha256") != protocol["protocol_manifest_sha256"]:
            raise ValueError("resume checkpoint protocol hash mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_random_state(state["random_state"])
        start_iteration = resume_global_step(state)
        best_loss = float(state["best_val_loss"])

    if args.resume is None:
        write_source_manifest(
            args.run_dir,
            REPO_DIR,
            Path(__file__),
            extra_paths=(
                REPO_DIR / "src/semantic_delphi_ukb/track_g_contract.py",
                REPO_DIR / "src/semantic_delphi_ukb/track_g_models.py",
                REPO_DIR / "src/semantic_delphi_ukb/track_r_batch.py",
                args.protocol,
                protocol["_source_track_r_path"],
            ),
        )

    last_iteration = start_iteration
    resumed_at_checkpoint = args.resume is not None
    try:
        monitor.mark_running(
            phase="pretraining",
            model=args.model,
            seed=args.seed,
            iteration=last_iteration,
            global_step=last_iteration,
        )
        iteration = start_iteration
        while iteration <= args.max_iters:
            last_iteration = iteration
            should_evaluate = iteration % args.eval_interval == 0 or iteration == args.max_iters
            if resumed_at_checkpoint and iteration == start_iteration:
                should_evaluate = False
            if should_evaluate:
                metrics = evaluate(
                    model,
                    val_data,
                    val_p2i,
                    val_static,
                    val_prefix,
                    val_anchor,
                    args,
                )
                improved = metrics["val_loss"] < best_loss
                best_loss = min(best_loss, metrics["val_loss"])
                row = {
                    "iteration": iteration,
                    "global_step": iteration,
                    **metrics,
                    "best_val_loss": best_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                monitor.log_epoch(row)
                state = checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    monitor,
                    args,
                    protocol,
                    iteration,
                    best_loss,
                    config,
                )
                monitor.save_checkpoint(state, "last.pt")
                if improved:
                    monitor.save_checkpoint(state, "best_val_loss.pt")
                monitor.update_status(status="running", model=args.model, seed=args.seed, **row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if iteration == args.max_iters:
                break

            started = time.perf_counter()
            ix = torch.randint(len(train_p2i), (args.batch_size,))
            batch = build_training_batch(
                ix,
                train_data,
                train_p2i,
                train_static,
                args,
                padding="regular",
                select="random",
                prefix_ids=train_prefix,
                anchor_ages=train_anchor,
            )
            x, age, targets, targets_age, features, static_mask, bos_mask, next_mask, time_mask, feature_mask = batch
            _, parts, *_ = model(
                x,
                age,
                features,
                targets,
                targets_age,
                static_token_mask=static_mask,
                bos_token_mask=bos_mask,
                next_event_mask=next_mask,
                time_loss_mask=time_mask,
                static_feature_mask=feature_mask,
            )
            loss = parts["loss_ce"] + parts["loss_dt"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            elapsed = max(time.perf_counter() - started, 1e-9)
            completed_step = iteration + 1
            if completed_step % args.log_every == 0:
                monitor.log_step(
                    completed_step,
                    {
                        "loss/train_total": loss,
                        "loss/train_ce": parts["loss_ce"],
                        "loss/train_dt": parts["loss_dt"],
                        "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                        "optimization/gradient_norm": grad_norm,
                        "performance/step_time_seconds": elapsed,
                        "performance/samples_per_second": args.batch_size / elapsed,
                        **monitor.system_metrics(),
                    },
                )
            iteration = completed_step
            last_iteration = iteration
            resumed_at_checkpoint = False
        monitor.mark_finished(
            phase="finished",
            model=args.model,
            seed=args.seed,
            iteration=last_iteration,
            global_step=last_iteration,
            best_val_loss=best_loss,
        )
    except BaseException as exc:
        monitor.mark_failed(
            exc,
            model=args.model,
            seed=args.seed,
            iteration=last_iteration,
            global_step=last_iteration,
        )
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
