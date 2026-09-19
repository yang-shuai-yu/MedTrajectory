from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from semantic_delphi_ukb.semantic_model import SemanticDelphi, SemanticDelphiConfig
from utils import get_p2i


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a semantic Delphi checkpoint on train/val/test splits.")
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0, help="0 means evaluate all patients in the split.")
    parser.add_argument("--topk", type=str, default="1,5,10")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def resolve_ckpt_path(args: argparse.Namespace) -> Path:
    if args.ckpt_path is not None:
        return args.ckpt_path
    if args.out_dir is None:
        raise ValueError("Provide either --ckpt-path or --out-dir.")
    return args.out_dir / "ckpt.pt"


def resolve_data_dir(args: argparse.Namespace, checkpoint_config: Dict[str, object]) -> Path:
    if args.data_dir is not None:
        return args.data_dir
    if args.dataset is not None:
        return REPO_DIR / "data" / args.dataset
    dataset_name = checkpoint_config.get("dataset")
    if not dataset_name:
        raise ValueError("Could not infer dataset from checkpoint config. Provide --data-dir or --dataset.")
    return REPO_DIR / "data" / str(dataset_name)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.memmap(path, dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    return data, p2i


def parse_topk(spec: str) -> List[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    values = [value for value in values if value > 0]
    if not values:
        raise ValueError("topk must contain at least one positive integer.")
    return values


def get_batch_safe(
    ix: List[int],
    data: np.ndarray,
    p2i: np.ndarray,
    block_size: int,
    device: str,
    no_event_token_rate: int,
    padding: str = "regular",
    cut_batch: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_time = -10000.0

    x = torch.tensor([p2i[i].tolist() for i in ix], dtype=torch.long)
    ix_tensor = torch.tensor(ix, dtype=torch.long)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(ix_tensor.sum().item()))

    traj_start_idx = x[:, 0]
    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1).tolist()
    batch_idx = np.arange(block_size + 1)[None, :] + np.asarray(traj_start_idx, dtype=np.int64)[:, None]

    mask_src = data[:, 0][batch_idx].astype(np.int64)
    patient_src = data[p2i[np.asarray(ix, dtype=np.int64)][:, 0], 0].astype(np.int64)[:, None]
    mask = torch.tensor((mask_src == patient_src).tolist(), dtype=torch.bool)

    tokens = torch.tensor(data[:, 2][batch_idx].astype(np.int64).tolist(), dtype=torch.long)
    ages = torch.tensor(data[:, 1][batch_idx].astype(np.float32).tolist(), dtype=torch.float32)

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    if padding.lower() == "none" or no_event_token_rate in (0, None):
        pad = torch.ones(len(ix), 0)
    elif padding == "regular":
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == "random":
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError

    m = ages.max(1, keepdim=True).values
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.long)])
    ages = torch.hstack([ages, pad])

    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    tokens = tokens + 1

    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]

    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)

    if device == "cuda":
        x, a, y, b = [tensor.pin_memory().to(device, non_blocking=True) for tensor in [x, a, y, b]]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
    return x, a, y, b


def evaluate_split(
    model: SemanticDelphi,
    data: np.ndarray,
    p2i: np.ndarray,
    split_name: str,
    batch_size: int,
    block_size: int,
    device: str,
    no_event_token_rate: int,
    ignore_tokens: List[int],
    max_patients: int,
    topk_values: List[int],
) -> Dict[str, float]:
    model.eval()
    ignored = list(ignore_tokens) + [1]
    total_ce = 0.0
    total_dt = 0.0
    total_targets = 0
    topk_hits = {k: 0 for k in topk_values}
    num_patients = len(p2i) if max_patients <= 0 else min(len(p2i), max_patients)

    with torch.no_grad():
        for start in range(0, num_patients, batch_size):
            stop = min(start + batch_size, num_patients)
            ix = list(range(start, stop))
            x, a, y, b = get_batch_safe(
                ix,
                data,
                p2i,
                block_size=block_size,
                device=device,
                no_event_token_rate=no_event_token_rate,
                padding="regular",
                cut_batch=True,
            )
            logits, loss, _ = model(x, a, y, b, validation_loss_mode=True)

            targets = y.reshape(-1)
            pass_tokens = targets != -1
            for token_id in ignored:
                pass_tokens *= targets != token_id
            n_targets = int(pass_tokens.sum().item())
            if n_targets == 0:
                continue

            total_ce += float(loss["loss_ce"].item()) * n_targets
            total_dt += float(loss["loss_dt"].item()) * n_targets
            total_targets += n_targets

            logits_valid = logits.reshape(-1, logits.size(-1))[pass_tokens]
            targets_valid = targets[pass_tokens]
            max_k = min(max(topk_values), int(logits_valid.size(-1)))
            topk_indices = logits_valid.topk(max_k, dim=-1).indices
            for k in topk_values:
                capped_k = min(k, max_k)
                topk_hits[k] += int(
                    topk_indices[:, :capped_k].eq(targets_valid.unsqueeze(-1)).any(dim=-1).sum().item()
                )

    if total_targets == 0:
        raise RuntimeError(f"No valid targets found for split={split_name}.")

    mean_ce = total_ce / total_targets
    mean_dt = total_dt / total_targets
    metrics = {
        "split": split_name,
        "patients": num_patients,
        "rows": int(data.shape[0]),
        "eval_targets": total_targets,
        "loss_ce": mean_ce,
        "loss_dt": mean_dt,
        "loss_total": mean_ce + mean_dt,
        "perplexity": math.exp(mean_ce) if mean_ce < 20 else float("inf"),
    }
    for k in topk_values:
        metrics[f"top{k}_acc"] = topk_hits[k] / total_targets
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_path = resolve_ckpt_path(args)
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)

    checkpoint_config = checkpoint.get("config", {})
    data_dir = resolve_data_dir(args, checkpoint_config)

    model_args = checkpoint["model_args"]
    conf = SemanticDelphiConfig(**model_args)
    model = SemanticDelphi(conf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    no_event_token_rate = int(checkpoint_config.get("no_event_token_rate", 5))
    ignore_tokens = list(getattr(conf, "ignore_tokens", [0]))
    block_size = int(conf.block_size)
    topk_values = parse_topk(args.topk)

    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    metrics = []
    for split_name in split_names:
        split_path = data_dir / f"{split_name}.bin"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        data, p2i = load_split(split_path)
        metrics.append(
            evaluate_split(
                model=model,
                data=data,
                p2i=p2i,
                split_name=split_name,
                batch_size=args.batch_size,
                block_size=block_size,
                device=args.device,
                no_event_token_rate=no_event_token_rate,
                ignore_tokens=ignore_tokens,
                max_patients=args.max_patients,
                topk_values=topk_values,
            )
        )

    payload = {
        "ckpt_path": str(ckpt_path),
        "data_dir": str(data_dir),
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "metrics": metrics,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
