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

from semantic_delphi_ukb.modern_model import (  # noqa: E402
    ModernMultitypeSemanticDelphi,
    ModernMultitypeSemanticDelphiConfig,
)
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.paper_run import (  # noqa: E402
    cosine_warmup_scheduler,
    load_protocol_config,
    resolve_path,
    restore_random_state,
    seed_everything,
    strip_compiled_prefix,
    write_source_manifest,
)
from training_monitor import RunMonitor  # noqa: E402
from utils import get_p2i  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal all-position Delphi pretraining for paper_protocol_v1.")
    parser.add_argument("--common-config", type=Path, default=REPO_DIR / "configs/paper_protocol_v1/common.json")
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-learning-rate", type=float, default=6e-5)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
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


def make_model(args: argparse.Namespace, resolved: dict, data_dir: Path) -> ModernMultitypeSemanticDelphi:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    embeddings = np.load(data_dir / manifest["semantic_output"]).astype(np.float32)
    experiment = resolved["experiment"]
    common = resolved["common"]
    config = ModernMultitypeSemanticDelphiConfig(
        block_size=int(common["pretraining"]["block_size"]),
        vocab_size=int(embeddings.shape[0]),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=int(embeddings.shape[1]),
        dropout=args.dropout,
        static_dim=len(manifest["static_feature_order"]),
        static_hidden_dim=int(embeddings.shape[1]),
        use_age_encoding=bool(experiment["age_sincos"]),
        use_age_rope=bool(experiment["continuous_age_rope"]),
    )
    return ModernMultitypeSemanticDelphi(config, embeddings)


@torch.no_grad()
def estimate(model, data, p2i, static, args) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_ce": 0.0, "loss_dt": 0.0}
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,))
        x, age, y, target_age, s = get_batch(
            ix, data, p2i, static, block_size=model.config.block_size, device=args.device,
            padding="random", select="random", no_event_token_rate=args.no_event_token_rate, cut_batch=True,
        )
        _, parts, _ = model(x, age, s, y, target_age, validation_loss_mode=True)
        totals["loss_ce"] += float(parts["loss_ce"].detach().cpu())
        totals["loss_dt"] += float(parts["loss_dt"].detach().cpu())
    model.train()
    totals = {key: value / args.eval_iters for key, value in totals.items() if key != "loss"}
    totals["loss"] = totals["loss_ce"] + totals["loss_dt"]
    return totals


def self_test(args: argparse.Namespace) -> int:
    torch.manual_seed(7)
    config = ModernMultitypeSemanticDelphiConfig(
        block_size=12, vocab_size=64, n_layer=2, n_head=4, n_embd=32, static_dim=1, use_age_rope=True
    )
    model = ModernMultitypeSemanticDelphi(config).to(args.device)
    x = torch.randint(2, 64, (4, 12), device=args.device)
    age = torch.sort(torch.rand(4, 12, device=args.device) * 30000, dim=1).values
    y = torch.randint(2, 64, (4, 12), device=args.device)
    target_age = age + 30
    _, loss, _ = model(x, age, torch.randn(4, 1, device=args.device), y, target_age)
    total = loss["loss_ce"] + loss["loss_dt"]
    total.backward()
    print(json.dumps({"self_test": True, "all_position_targets": int((y > 1).sum()), "loss": float(total)}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)
    resolved = load_protocol_config(args.common_config, args.experiment_config)
    data_dir = args.data_dir or resolve_path(REPO_DIR, resolved["experiment"]["data_dir"])
    if args.run_dir.exists() and args.resume is None:
        unexpected = [p.name for p in args.run_dir.iterdir() if p.name != "stdout.log"]
        if unexpected:
            raise FileExistsError(f"run directory is not empty: {args.run_dir}")
    seed = int(args.seed if args.seed is not None else resolved["common"]["training_seeds"][0])
    args.seed = seed
    seed_everything(seed)
    train_data, train_p2i, train_static = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(data_dir, "val", args.max_patients)
    model = make_model(args, resolved, data_dir).to(args.device)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.99), "cuda" if "cuda" in args.device else "cpu")
    scheduler = cosine_warmup_scheduler(
        optimizer, args.warmup_iters, args.max_iters, args.min_learning_rate / args.learning_rate
    )
    config = {**vars(args), **resolved, "data_dir": str(data_dir), "model_args": model.config.__dict__.copy()}
    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    write_source_manifest(args.run_dir, REPO_DIR, Path(__file__))
    start_iter = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(strip_compiled_prefix(checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = int(checkpoint["iteration"]) + 1
        global_step = int(checkpoint["global_step"])
        best_val_loss = float(checkpoint["best_val_loss"])
        restore_random_state(checkpoint["random_state"])
    last_iter = start_iter - 1
    try:
        monitor.mark_running(phase="pretraining", iteration=start_iter, global_step=global_step)
        for iteration in range(start_iter, args.max_iters + 1):
            last_iter = iteration
            if iteration % args.eval_interval == 0:
                metrics = estimate(model, val_data, val_p2i, val_static, args)
                improved = metrics["loss"] < best_val_loss
                if improved:
                    best_val_loss = metrics["loss"]
                row = {
                    "iteration": iteration, "global_step": global_step, "val_loss": metrics["loss"],
                    "val_next_event_ce": metrics["loss_ce"], "val_time_loss": metrics["loss_dt"],
                    "learning_rate": optimizer.param_groups[0]["lr"], "best_val_loss": best_val_loss,
                }
                monitor.log_epoch(row)
                state = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": None, "iteration": iteration, "global_step": global_step,
                    "best_val_loss": best_val_loss, "model_args": model.config.__dict__.copy(),
                    "model_family": "paper_causal_delphi_v1", "config": config,
                    "random_state": monitor.capture_random_state(),
                }
                monitor.save_checkpoint(state, "last.pt")
                if improved:
                    monitor.save_checkpoint(state, "best_val_loss.pt")
                monitor.update_status(status="running", phase="pretraining", **row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if iteration == args.max_iters:
                break
            step_start = time.perf_counter()
            ix = torch.randint(len(train_p2i), (args.batch_size,))
            x, age, y, target_age, s = get_batch(
                ix, train_data, train_p2i, train_static, block_size=model.config.block_size, device=args.device,
                padding="random", select="random", no_event_token_rate=args.no_event_token_rate, cut_batch=True,
            )
            _, parts, _ = model(x, age, s, y, target_age)
            loss = parts["loss_ce"] + parts["loss_dt"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % args.log_every == 0:
                elapsed = time.perf_counter() - step_start
                monitor.log_step(global_step, {
                    "loss/train_total": loss, "loss/train_next_event_ce": parts["loss_ce"],
                    "loss/train_time": parts["loss_dt"], "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                    "optimization/gradient_norm": grad_norm, "performance/step_time_seconds": elapsed,
                    "performance/samples_per_second": args.batch_size / max(elapsed, 1e-9), **monitor.system_metrics(),
                })
        monitor.mark_finished(phase="finished", iteration=last_iter, global_step=global_step, best_val_loss=best_val_loss)
    except BaseException as exc:
        monitor.mark_failed(exc, iteration=last_iter, global_step=global_step)
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
