"""Train an ETHOS-Matched / Foresight-Matched timeline baseline on MIMIC Track-R data.

Reuses the frozen matched-baseline model family (`build_matched_model`) and the MIMIC Track-R
data contract, but drives training with the same optimizer/schedule/loop as the CARoPE
pretraining entry point so that A0/A2 and the matched baseline differ only in the model family.
"""
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

from semantic_delphi_ukb.paper_run import cosine_warmup_scheduler, seed_everything  # noqa: E402
from semantic_delphi_ukb.track_g_models import build_matched_model, model_config_payload  # noqa: E402
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    load_track_r_assets,
    load_track_r_bos_token_id,
    load_track_r_static_features,
    validate_track_r_data_manifest,
)
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402
from semantic_delphi_ukb.train_car_rope import build_training_batch, load_split  # noqa: E402
from semantic_delphi_ukb.tte_targets import load_selected_disease_token_groups  # noqa: E402
from training_monitor import RunMonitor  # noqa: E402

FAMILIES = ("ethos_matched", "foresight_matched")
ETHOS_BUCKETS = (1.0, 7.0, 30.0, 90.0, 365.25, 1826.25, 3652.5)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=FAMILIES, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--track-r-protocol", type=Path, required=True)
    p.add_argument("--diseases-yaml", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-iters", type=int, default=100000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--min-learning-rate", type=float, default=6e-5)
    p.add_argument("--warmup-iters", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--horizons", default="1,5")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--no-tensorboard", action="store_true")
    return p


def make_model(args, data_dir: Path, num_diseases: int):
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    embeddings = np.load(data_dir / manifest["semantic_output"]).astype(np.float32)
    protocol = load_track_r_protocol(args.track_r_protocol)
    block_size = int(protocol["dynamic_context_length"]) + 1 + int(protocol["static_prefix"]["fixed_length"])
    return build_matched_model(
        args.model,
        block_size=block_size,
        vocab_size=int(embeddings.shape[0]),
        semantic_embedding_dim=int(embeddings.shape[1]),
        pretrained_token_embeddings=embeddings,
        num_diseases=num_diseases,
        architecture={
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": int(embeddings.shape[1]),
            "dropout": args.dropout,
            "ethos_gap_bucket_upper_days": list(ETHOS_BUCKETS),
        },
        horizons_years=[float(x) for x in str(args.horizons).split(",")],
    )


@torch.no_grad()
def evaluate(model, data, p2i, static, prefix, anchors, args):
    model.eval()
    total = ce = dt = 0.0
    for _ in range(args.eval_iters):
        ix = torch.randint(len(p2i), (args.batch_size,))
        batch = build_training_batch(ix, data, p2i, static, args, padding="regular", select="random",
                                     prefix_ids=prefix, anchor_ages=anchors)
        x, age, targets, targets_age, features, static_mask, bos_mask, next_mask, time_mask, feature_mask = batch
        _, parts, *_ = model(x, age, features, targets, targets_age, validation_loss_mode=True,
                             static_token_mask=static_mask, bos_token_mask=bos_mask,
                             next_event_mask=next_mask, time_loss_mask=time_mask,
                             static_feature_mask=feature_mask)
        ce += float(parts["loss_ce"].cpu()); dt += float(parts["loss_dt"].cpu())
        total += float((parts["loss_ce"] + parts["loss_dt"]).cpu())
    model.train()
    s = 1.0 / max(args.eval_iters, 1)
    return {"val_loss": total * s, "val_loss_ce": ce * s, "val_loss_dt": dt * s}


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_track_r_protocol(args.track_r_protocol)
    args.track_r_protocol_sha256 = protocol.get("protocol_manifest_sha256")
    args.dynamic_context_length = int(protocol["dynamic_context_length"])
    args.block_size = args.dynamic_context_length + 1 + int(protocol["static_prefix"]["fixed_length"])
    args.include_static_prefix = "true"
    args.static_conditioning = "none"
    args.no_event_token_rate = 5
    args.track_r_protocol = args.track_r_protocol
    if args.run_dir.exists() and any(p.name != "stdout.log" for p in args.run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {args.run_dir}")
    seed_everything(args.seed)
    diseases, _ = load_selected_disease_token_groups(args.diseases_yaml, args.data_dir)
    train_data, train_p2i, _ = load_split(args.data_dir, "train", 0)
    val_data, val_p2i, _ = load_split(args.data_dir, "val", 0)
    train_prefix, train_anchor = load_track_r_assets(args.data_dir, "train", 0)
    val_prefix, val_anchor = load_track_r_assets(args.data_dir, "val", 0)
    train_static = load_track_r_static_features(args.data_dir, "train", "none", 0)
    val_static = load_track_r_static_features(args.data_dir, "val", "none", 0)
    args.track_r_bos_token_id = load_track_r_bos_token_id(args.data_dir)

    model = make_model(args, args.data_dir, len(diseases)).to(args.device)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.99),
                                           "cuda" if "cuda" in args.device else "cpu")
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_iters, args.max_iters,
                                        args.min_learning_rate / args.learning_rate)
    config = {**vars(args), "data_dir": str(args.data_dir), "model_args": model_config_payload(model)}
    monitor = RunMonitor(args.run_dir, config, enable_tensorboard=not args.no_tensorboard)
    best = float("inf")
    last = -1
    print("number of parameters: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    try:
        monitor.mark_running(phase="pretraining", iteration=0, global_step=0)
        for it in range(args.max_iters + 1):
            last = it
            if it % args.eval_interval == 0:
                m = evaluate(model, val_data, val_p2i, val_static, val_prefix, val_anchor, args)
                improved = m["val_loss"] < best
                best = min(best, m["val_loss"])
                row = {"iteration": it, **m, "best_val_loss": best}
                monitor.log_epoch(row)
                state = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "iteration": it,
                    "model_args": model_config_payload(model), "stage": "pretraining",
                    "track_g_family": args.model, "seed": int(args.seed),
                    "protocol_manifest_sha256": protocol.get("protocol_manifest_sha256"),
                    "config": config,
                }
                monitor.save_checkpoint(state, "last.pt")
                if improved:
                    monitor.save_checkpoint(state, "best_val_loss.pt")
                monitor.update_status(status="running", **row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if it == args.max_iters:
                break
            start = time.perf_counter()
            ix = torch.randint(len(train_p2i), (args.batch_size,))
            batch = build_training_batch(ix, train_data, train_p2i, train_static, args,
                                         padding="regular", select="random",
                                         prefix_ids=train_prefix, anchor_ages=train_anchor)
            x, age, targets, targets_age, features, static_mask, bos_mask, next_mask, time_mask, feature_mask = batch
            _, parts, *_ = model(x, age, features, targets, targets_age,
                                 static_token_mask=static_mask, bos_token_mask=bos_mask,
                                 next_event_mask=next_mask, time_loss_mask=time_mask,
                                 static_feature_mask=feature_mask)
            loss = parts["loss_ce"] + parts["loss_dt"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            if it % args.log_every == 0:
                monitor.log_step(it, {"loss/train_total": loss, "optimization/gradient_norm": grad,
                                      "performance/step_time_seconds": time.perf_counter() - start})
        monitor.mark_finished(phase="finished", iteration=last, global_step=last, best_val_loss=best)
    except BaseException as exc:
        monitor.mark_failed(exc, iteration=last, global_step=last)
        raise
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
