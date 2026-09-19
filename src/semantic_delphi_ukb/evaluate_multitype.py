from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F

from utils import get_p2i
from semantic_delphi_ukb.multitype_batch import get_batch
from semantic_delphi_ukb.multitype_model import MultitypeSemanticDelphi, MultitypeSemanticDelphiConfig


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a multitype semantic Delphi checkpoint.")
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--token-vocab-csv", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--topk", type=str, default="1,5,10")
    parser.add_argument(
        "--target-event-types",
        type=str,
        default=None,
        help="Comma-separated event types to keep as evaluation targets, e.g. diagnosis.",
    )
    parser.add_argument(
        "--candidate-event-types",
        type=str,
        default=None,
        help="Comma-separated event types to keep as prediction candidates, e.g. diagnosis.",
    )
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


def resolve_token_vocab_csv(args: argparse.Namespace, data_dir: Path) -> Optional[Path]:
    if args.token_vocab_csv is not None:
        return args.token_vocab_csv
    manifest_path = data_dir / "prepare_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        vocab_csv = manifest.get("vocab_csv")
        if vocab_csv:
            return Path(str(vocab_csv))
    fallback = data_dir / "dynamic_token_vocab.csv"
    if fallback.exists():
        return fallback
    return None


def parse_topk(spec: str) -> List[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    values = [value for value in values if value > 0]
    if not values:
        raise ValueError("topk must contain at least one positive integer.")
    return values


def parse_event_types(spec: Optional[str]) -> Optional[Set[str]]:
    if spec is None:
        return None
    values = {item.strip() for item in spec.split(",") if item.strip()}
    return values or None


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.memmap(path, dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    return data, p2i


def load_token_event_types(vocab_csv: Optional[Path], vocab_size: int) -> List[str]:
    event_types = ["unknown"] * vocab_size
    if vocab_csv is None or not vocab_csv.exists():
        return event_types
    with vocab_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            token_id = int(row["token_id"])
            if 0 <= token_id < vocab_size:
                event_types[token_id] = row["event_type"].strip()
    return event_types


def build_event_type_lookup(
    token_event_types: Sequence[str],
    allowed_event_types: Optional[Set[str]],
    device: str,
) -> Optional[torch.Tensor]:
    if allowed_event_types is None:
        return None
    lookup = torch.zeros(len(token_event_types), dtype=torch.bool, device=device)
    for token_id, event_type in enumerate(token_event_types):
        if event_type in allowed_event_types:
            lookup[token_id] = True
    return lookup


def build_target_mask(
    targets: torch.Tensor,
    ignored_tokens: Sequence[int],
    target_lookup: Optional[torch.Tensor],
) -> torch.Tensor:
    pass_tokens = targets != -1
    for token_id in ignored_tokens:
        pass_tokens &= targets != token_id
    if target_lookup is not None:
        target_ok = torch.zeros_like(pass_tokens, dtype=torch.bool)
        valid = targets >= 0
        if valid.any():
            target_ok[valid] = target_lookup[targets[valid]]
        pass_tokens &= target_ok
    return pass_tokens


def apply_candidate_filter(logits: torch.Tensor, candidate_lookup: Optional[torch.Tensor]) -> torch.Tensor:
    if candidate_lookup is None:
        return logits
    filtered = logits.clone()
    filtered[..., ~candidate_lookup] = -torch.inf
    return filtered


def build_attn_mask(
    idx: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    mask_ties: bool,
) -> torch.Tensor:
    device = idx.device
    batch_size, seq_len = idx.shape
    tril = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
    attn_mask = (idx > 0).view(batch_size, 1, 1, seq_len) & (idx > 0).view(batch_size, 1, seq_len, 1)
    attn_mask &= tril
    if mask_ties:
        attn_mask &= age.view(batch_size, 1, 1, seq_len) != targets_age.view(batch_size, 1, seq_len, 1)
        diag = torch.diag(torch.ones(seq_len, device=device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        attn_mask |= (attn_mask.sum(-1, keepdim=True) == 0) & diag
    diag = torch.diag(torch.ones(seq_len, device=device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
    attn_mask |= (idx == 0).view(batch_size, 1, 1, seq_len) & diag
    attn_mask &= tril
    return attn_mask


def compute_time_loss_terms(
    logits: torch.Tensor,
    idx: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    t_min: float,
    mask_ties: bool,
) -> torch.Tensor:
    lse = torch.logsumexp(logits, -1)
    lse = -torch.log(torch.exp(-lse) + t_min)
    dt = torch.clamp(targets_age - age, min=1.0)
    if mask_ties:
        attn_mask = build_attn_mask(idx, age, targets_age, mask_ties=True)
        gather_index = (
            attn_mask
            * torch.arange(0, idx.size(1), device=idx.device, dtype=torch.float32).view(1, 1, 1, -1)
        ).max(-1).indices.squeeze(1).squeeze(1)
        dt = torch.gather(dt, -1, gather_index)
    ldt = -torch.log(dt + t_min)
    loss_dt = -(lse - torch.exp(lse - ldt))
    return loss_dt.reshape(-1)


def evaluate_split(
    model: MultitypeSemanticDelphi,
    data: np.ndarray,
    p2i: np.ndarray,
    static_matrix: np.ndarray,
    split_name: str,
    batch_size: int,
    block_size: int,
    device: str,
    no_event_token_rate: int,
    ignore_tokens: List[int],
    max_patients: int,
    topk_values: List[int],
    target_lookup: Optional[torch.Tensor],
    candidate_lookup: Optional[torch.Tensor],
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
            x, a, y, b, s = get_batch(
                ix,
                data,
                p2i,
                static_matrix,
                block_size=block_size,
                device=device,
                no_event_token_rate=no_event_token_rate,
                padding="regular",
                cut_batch=True,
            )
            logits, _, _ = model(x, a, s, y, b, validation_loss_mode=True)
            logits = apply_candidate_filter(logits, candidate_lookup)

            targets = y.reshape(-1)
            pass_tokens = build_target_mask(targets, ignored, target_lookup)
            n_targets = int(pass_tokens.sum().item())
            if n_targets == 0:
                continue

            logits_valid = logits.reshape(-1, logits.size(-1))[pass_tokens]
            targets_valid = targets[pass_tokens]
            ce_sum = F.cross_entropy(logits_valid, targets_valid, reduction="sum")
            dt_terms = compute_time_loss_terms(
                logits=logits,
                idx=x,
                age=a,
                targets_age=b,
                t_min=float(model.config.t_min),
                mask_ties=bool(model.config.mask_ties),
            )
            total_ce += float(ce_sum.item())
            total_dt += float(dt_terms[pass_tokens].sum().item())
            total_targets += n_targets

            max_k = min(max(topk_values), int(logits_valid.size(-1)))
            topk_indices = logits_valid.topk(max_k, dim=-1).indices
            for k in topk_values:
                capped_k = min(k, max_k)
                topk_hits[k] += int(topk_indices[:, :capped_k].eq(targets_valid.unsqueeze(-1)).any(dim=-1).sum().item())

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

    conf = MultitypeSemanticDelphiConfig(**checkpoint["model_args"])
    model = MultitypeSemanticDelphi(conf)
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
    token_vocab_csv = resolve_token_vocab_csv(args, data_dir)
    target_event_types = parse_event_types(args.target_event_types)
    candidate_event_types = parse_event_types(args.candidate_event_types)
    token_event_types = load_token_event_types(token_vocab_csv, int(conf.vocab_size))
    target_lookup = build_event_type_lookup(token_event_types, target_event_types, args.device)
    candidate_lookup = build_event_type_lookup(token_event_types, candidate_event_types, args.device)

    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    metrics = []
    for split_name in split_names:
        split_path = data_dir / f"{split_name}.bin"
        static_path = data_dir / f"{split_name}_static.npy"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        if not static_path.exists():
            raise FileNotFoundError(f"Static matrix not found: {static_path}")
        data, p2i = load_split(split_path)
        static_matrix = np.load(static_path).astype(np.float32)
        metrics.append(
            evaluate_split(
                model=model,
                data=data,
                p2i=p2i,
                static_matrix=static_matrix,
                split_name=split_name,
                batch_size=args.batch_size,
                block_size=block_size,
                device=args.device,
                no_event_token_rate=no_event_token_rate,
                ignore_tokens=ignore_tokens,
                max_patients=args.max_patients,
                topk_values=topk_values,
                target_lookup=target_lookup,
                candidate_lookup=candidate_lookup,
            )
        )

    payload = {
        "ckpt_path": str(ckpt_path),
        "data_dir": str(data_dir),
        "token_vocab_csv": None if token_vocab_csv is None else str(token_vocab_csv),
        "target_event_types": None if target_event_types is None else sorted(target_event_types),
        "candidate_event_types": None if candidate_event_types is None else sorted(candidate_event_types),
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
