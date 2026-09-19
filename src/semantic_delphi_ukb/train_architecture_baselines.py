from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import get_p2i
from semantic_delphi_ukb.architecture_baselines import (
    ArchitectureBaselineConfig,
    HistoryMaskBERTBaseline,
    HistoryMaskRoPEBERTBaseline,
    MambaHistoryBaseline,
    final_context_loss,
    final_context_topk,
    history_only_inputs,
)
from semantic_delphi_ukb.multitype_batch import get_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BERT-style or Mamba architecture baselines on MedTrajectory data.")
    parser.add_argument("--model", choices=["bert", "bert_rope", "mamba"], default="bert")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser


def make_model(args: argparse.Namespace, vocab_size: int, static_dim: int):
    config = ArchitectureBaselineConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        static_dim=static_dim,
    )
    if args.model == "bert":
        return HistoryMaskBERTBaseline(config)
    if args.model == "bert_rope":
        return HistoryMaskRoPEBERTBaseline(config)
    return MambaHistoryBaseline(config)


def self_test(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    vocab_size = 128
    static_dim = 10
    model = make_model(args, vocab_size=vocab_size, static_dim=static_dim).to(args.device)
    batch_size = 4
    seq_len = min(args.block_size, 16)
    x = torch.randint(2, vocab_size, (batch_size, seq_len), device=args.device)
    age = torch.sort(torch.randint(1000, 30000, (batch_size, seq_len), device=args.device).float(), dim=1).values
    y = torch.randint(2, vocab_size, (batch_size, seq_len), device=args.device)
    target_age = age + torch.randint(1, 365, (batch_size, seq_len), device=args.device).float()
    static = torch.randn(batch_size, static_dim, device=args.device)
    x_in, age_in = history_only_inputs(x, age, y)
    logits = model(x_in, age_in, static)
    loss, parts = final_context_loss(logits, x, age, y, target_age)
    loss.backward()
    print(json.dumps({"self_test": True, "model": args.model, "loss": float(loss.detach().cpu())}, indent=2))
    return 0


def load_split(data_dir: Path, split: str, max_patients: int):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if max_patients > 0:
        p2i = p2i[:max_patients]
        static = static[:max_patients]
    return data, p2i, static


def load_vocab_size(data_dir: Path, train_data: np.ndarray, val_data: np.ndarray) -> int:
    manifest_path = data_dir / "prepare_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "vocab_size" in manifest:
            return int(manifest["vocab_size"])
    return int(max(train_data[:, 2].max(), val_data[:, 2].max())) + 2


@torch.no_grad()
def estimate(model, data, p2i, static, args):
    model.eval()
    totals = {"n": 0, "top1": 0.0, "top5": 0.0, "top10": 0.0, "loss": 0.0}
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
        x_in, age_in = history_only_inputs(x, age, y)
        logits = model(x_in, age_in, s)
        loss, _ = final_context_loss(logits, x, age, y, target_age)
        topk = final_context_topk(logits, x, y)
        n = topk["n"]
        if n == 0:
            continue
        totals["n"] += n
        totals["loss"] += float(loss.item()) * n
        for key in ["top1", "top5", "top10"]:
            totals[key] += topk[key] * n
    model.train()
    if totals["n"] == 0:
        return {key: float("nan") for key in ["loss", "top1", "top5", "top10"]} | {"n": 0}
    return {key: totals[key] / totals["n"] for key in ["loss", "top1", "top5", "top10"]} | {"n": totals["n"]}


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)

    torch.manual_seed(args.seed)
    data_dir = args.data_dir or (REPO_DIR / "data" / args.dataset)
    out_dir = args.out_dir or (REPO_DIR / "results" / "architecture_baselines" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data, train_p2i, train_static = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(data_dir, "val", args.max_patients)
    vocab_size = load_vocab_size(data_dir, train_data, val_data)
    static_dim = int(train_static.shape[1])
    model = make_model(args, vocab_size=vocab_size, static_dim=static_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    history = []
    for iteration in range(args.max_iters + 1):
        if iteration % args.eval_interval == 0:
            val_metrics = estimate(model, val_data, val_p2i, val_static, args)
            row = {"iter": iteration, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            print(json.dumps(row, indent=2))
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_name": args.model,
                        "model_args": vars(args),
                        "vocab_size": vocab_size,
                        "static_dim": static_dim,
                        "best_val_loss": best_val,
                        "iter_num": iteration,
                    },
                    out_dir / "ckpt.pt",
                )
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
        x_in, age_in = history_only_inputs(x, age, y)
        logits = model(x_in, age_in, s)
        loss, _ = final_context_loss(logits, x, age, y, target_age)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
