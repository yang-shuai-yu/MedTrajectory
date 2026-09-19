from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import (  # noqa: E402
    ArchitectureBaselineConfig,
    HistoryMaskBERTBaseline,
    MambaHistoryBaseline,
    history_only_inputs,
    last_prediction_positions,
)
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.tte_model import HorizonRiskConfig, HorizonRiskMedTrajectory  # noqa: E402
from utils import get_p2i  # noqa: E402


DEFAULT_SPECS = [
    (
        "Monotonic Gated RoPE main",
        "medtrajectory",
        "monotonic",
        REPO_DIR / "results" / "monotonic_horizon" / "monotonic_gated_rope_gate100" / "ckpt.pt",
    ),
    (
        "Gated RoPE main",
        "medtrajectory",
        "gated",
        REPO_DIR / "results" / "gated_rope_ablation" / "gate100" / "ckpt.pt",
    ),
    (
        "No-RoPE previous best",
        "medtrajectory",
        "no_rope",
        REPO_DIR / "results" / "high_priority_ablation" / "no_rope_trunk_full3000" / "ckpt.pt",
    ),
    (
        "Old TTE+RoPE main",
        "medtrajectory",
        "old_tte",
        REPO_DIR / "results" / "high_priority_ablation" / "tte_trunk_aux02" / "ckpt.pt",
    ),
    (
        "BERT next-event no-leak",
        "bert",
        "bert",
        REPO_DIR / "results" / "architecture_baselines" / "bert_full_noleak" / "ckpt.pt",
    ),
    (
        "Mamba next-event no-leak",
        "mamba",
        "mamba",
        REPO_DIR / "results" / "architecture_baselines" / "mamba_full_noleak" / "ckpt.pt",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locked-test final-context state-generation metrics.")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--token-vocab-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "age_state_evaluation")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--topk", type=str, default="1,5,10")
    parser.add_argument("--selection", choices=["right", "random", "left"], default="right")
    parser.add_argument("--padding", choices=["regular", "random", "none"], default="regular")
    parser.add_argument("--models", type=str, default="monotonic,gated,bert,mamba")
    parser.add_argument("--medtrajectory-checkpoint", type=Path, default=None)
    parser.add_argument("--medtrajectory-name", type=str, default="MedTrajectory custom")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_topk(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values:
        raise ValueError("topk must not be empty")
    return values


def resolve_data_dir(args: argparse.Namespace) -> Path:
    return args.data_dir or (REPO_DIR / "data" / args.dataset)


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


def load_token_event_types(vocab_csv: Optional[Path], vocab_size: int) -> list[str]:
    event_types = ["unknown"] * vocab_size
    if vocab_csv is None or not vocab_csv.exists():
        return event_types
    with vocab_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            token_id = int(row["token_id"])
            if 0 <= token_id < vocab_size:
                event_types[token_id] = row.get("event_type", "unknown").strip() or "unknown"
    return event_types


def build_event_type_candidate_mask(
    token_event_types: Sequence[str],
    vocab_size: int,
    event_type: str,
    device: str,
) -> torch.Tensor:
    values = [
        idx < len(token_event_types) and token_event_types[idx] == event_type
        for idx in range(vocab_size)
    ]
    return torch.tensor(values, dtype=torch.bool, device=device)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_split(data_dir: Path, split: str, max_patients: int):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if max_patients > 0:
        p2i = p2i[:max_patients]
        static = static[:max_patients]
    return data, p2i, static


def normalize_state_dict(state_dict: dict) -> dict:
    state_dict = dict(state_dict)
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    return state_dict


def load_medtrajectory(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HorizonRiskMedTrajectory(HorizonRiskConfig(**checkpoint["model_args"]))
    model.load_state_dict(normalize_state_dict(checkpoint["model"]), strict=True)
    return model.to(device), checkpoint


def load_architecture(model_kind: str, path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint["model_args"]
    config = ArchitectureBaselineConfig(
        block_size=int(args.get("block_size", checkpoint.get("block_size", 128))),
        vocab_size=int(checkpoint["vocab_size"]),
        n_layer=int(args["n_layer"]),
        n_head=int(args["n_head"]),
        n_embd=int(args["n_embd"]),
        dropout=float(args["dropout"]),
        static_dim=int(checkpoint["static_dim"]),
    )
    model = HistoryMaskBERTBaseline(config) if model_kind == "bert" else MambaHistoryBaseline(config)
    model.load_state_dict(normalize_state_dict(checkpoint["model"]), strict=True)
    return model.to(device), checkpoint


def final_dt_loss(logits: torch.Tensor, age: torch.Tensor, target_age: torch.Tensor, t_min: float = 0.1) -> torch.Tensor:
    lse = torch.logsumexp(logits, dim=-1)
    lse = -torch.log(torch.exp(-lse) + t_min)
    dt = torch.clamp(target_age - age, min=1.0)
    ldt = -torch.log(dt + t_min)
    return -(lse - torch.exp(lse - ldt))


@torch.no_grad()
def collect_model_rows(
    model_name: str,
    model_kind: str,
    ckpt_path: Path,
    data,
    p2i,
    static,
    token_event_types: Sequence[str],
    args: argparse.Namespace,
    topk: Sequence[int],
) -> tuple[list[dict], dict]:
    if model_kind == "medtrajectory":
        model, checkpoint = load_medtrajectory(ckpt_path, args.device)
        vocab_size = int(model.config.vocab_size)
        t_min = float(model.config.t_min)
    else:
        model, checkpoint = load_architecture(model_kind, ckpt_path, args.device)
        vocab_size = int(model.config.vocab_size)
        t_min = float(getattr(model.config, "t_min", 0.1))
    model.eval()
    diagnosis_candidate_mask = build_event_type_candidate_mask(
        token_event_types,
        vocab_size,
        "diagnosis",
        args.device,
    )

    rows = []
    batches = [torch.arange(start, min(start + args.batch_size, len(p2i)), dtype=torch.long) for start in range(0, len(p2i), args.batch_size)]
    if args.max_patients > 0:
        batches = [batch for batch in batches if int(batch[0]) < args.max_patients]
    padding = None if args.padding == "none" else args.padding
    max_k = max(topk)
    for batch_ix in batches:
        x, age, y, target_age, s = get_batch(
            batch_ix,
            data,
            p2i,
            static,
            block_size=args.block_size,
            device=args.device,
            padding=padding,
            select=args.selection,
            cut_batch=True,
        )
        if model_kind == "medtrajectory":
            logits, _, _, _, _ = model(x, age, s, y, target_age, validation_loss_mode=True)
        else:
            x_in, age_in = history_only_inputs(x, age, y)
            logits = model(x_in, age_in, s)
        keep, pos = last_prediction_positions(x, y)
        if not bool(keep.any()):
            continue
        batch_index = torch.arange(x.size(0), device=args.device)[keep]
        pos_keep = pos[keep]
        final_logits = logits[batch_index, pos_keep]
        final_targets = y[batch_index, pos_keep]
        final_age = age[batch_index, pos_keep]
        final_target_age = target_age[batch_index, pos_keep]
        ce = F.cross_entropy(final_logits, final_targets, reduction="none")
        dt_loss = final_dt_loss(final_logits, final_age, final_target_age, t_min=t_min)
        k = min(max_k, final_logits.size(-1))
        top_idx = torch.topk(final_logits, k=k, dim=-1).indices
        diagnosis_top_idx = None
        if bool(diagnosis_candidate_mask.any().item()):
            diagnosis_logits = final_logits.clone()
            diagnosis_logits[:, ~diagnosis_candidate_mask] = -torch.inf
            diagnosis_k = min(max_k, int(diagnosis_candidate_mask.sum().item()))
            diagnosis_top_idx = torch.topk(diagnosis_logits, k=diagnosis_k, dim=-1).indices
        probs = torch.softmax(final_logits, dim=-1)
        target_prob = probs.gather(1, final_targets.view(-1, 1)).squeeze(1)
        for row_idx in range(final_targets.numel()):
            target = int(final_targets[row_idx].item())
            event_type = token_event_types[target] if 0 <= target < len(token_event_types) else "unknown"
            record = {
                "model": model_name,
                "patient_index": int(batch_ix[keep.cpu()][row_idx].item()),
                "prediction_age_years": float(final_age[row_idx].detach().cpu().item()) / 365.25,
                "target_delta_years": float((final_target_age[row_idx] - final_age[row_idx]).detach().cpu().item()) / 365.25,
                "target_token": target,
                "target_event_type": event_type,
                "cross_entropy": float(ce[row_idx].detach().cpu().item()),
                "target_probability": float(target_prob[row_idx].detach().cpu().item()),
                "delta_time_loss": float(dt_loss[row_idx].detach().cpu().item()),
            }
            pred_row = top_idx[row_idx]
            for top in topk:
                kk = min(top, pred_row.numel())
                record[f"top{top}_hit"] = int(bool((pred_row[:kk] == target).any().item()))
            for top in topk:
                key = f"diagnosis_candidate_top{top}_hit"
                if event_type == "diagnosis" and diagnosis_top_idx is not None:
                    kk = min(top, diagnosis_top_idx.size(1))
                    record[key] = int(bool((diagnosis_top_idx[row_idx, :kk] == target).any().item()))
                else:
                    record[key] = ""
            rows.append(record)
    return rows, checkpoint


def summarize(rows: list[dict], topk: Sequence[int]) -> list[dict]:
    out = []
    groups = {
        "all": rows,
        "diagnosis_targets": [row for row in rows if row["target_event_type"] == "diagnosis"],
    }
    for group_name, group in groups.items():
        if not group:
            continue
        by_model = sorted({row["model"] for row in group})
        for model in by_model:
            model_rows = [row for row in group if row["model"] == model]
            row = {
                "model": model,
                "target_scope": group_name,
                "n": len(model_rows),
                "mean_cross_entropy": float(np.mean([float(item["cross_entropy"]) for item in model_rows])),
                "mean_target_probability": float(np.mean([float(item["target_probability"]) for item in model_rows])),
                "mean_delta_time_loss": float(np.mean([float(item["delta_time_loss"]) for item in model_rows])),
                "median_target_delta_years": float(np.median([float(item["target_delta_years"]) for item in model_rows])),
            }
            for top in topk:
                row[f"top{top}_accuracy"] = float(np.mean([int(item[f"top{top}_hit"]) for item in model_rows]))
                if group_name == "diagnosis_targets":
                    row[f"diagnosis_candidate_top{top}_accuracy"] = float(
                        np.mean([int(item[f"diagnosis_candidate_top{top}_hit"]) for item in model_rows])
                    )
                else:
                    row[f"diagnosis_candidate_top{top}_accuracy"] = float("nan")
            out.append(row)
    return out


def selected_specs(args: argparse.Namespace):
    if args.medtrajectory_checkpoint is not None:
        return [(args.medtrajectory_name, "medtrajectory", args.medtrajectory_checkpoint)]
    aliases = {item.strip().lower() for item in args.models.split(",") if item.strip()}
    if "all" in aliases:
        aliases.update(alias for _, _, alias, _ in DEFAULT_SPECS)
    selected = []
    for name, kind, alias, path in DEFAULT_SPECS:
        if alias in aliases or name.lower() in aliases:
            selected.append((name, kind, path))
    return selected


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(args)
    data, p2i, static = load_split(data_dir, args.split, args.max_patients)
    topk = parse_topk(args.topk)
    specs = selected_specs(args)
    if not specs:
        raise ValueError("No models selected.")
    max_vocab_size = 0
    checkpoints = {}
    for name, kind, path in specs:
        if not path.exists():
            print(f"[skip missing] {name}: {path}", file=sys.stderr)
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints[name] = checkpoint
        if "model_args" in checkpoint and "vocab_size" in checkpoint["model_args"]:
            max_vocab_size = max(max_vocab_size, int(checkpoint["model_args"]["vocab_size"]))
        if "vocab_size" in checkpoint:
            max_vocab_size = max(max_vocab_size, int(checkpoint["vocab_size"]))
    token_event_types = load_token_event_types(resolve_token_vocab_csv(args, data_dir), max_vocab_size + 1)

    raw_rows = []
    used_specs = []
    for name, kind, path in specs:
        if not path.exists():
            continue
        rows, checkpoint = collect_model_rows(name, kind, path, data, p2i, static, token_event_types, args, topk)
        raw_rows.extend(rows)
        used_specs.append(
            {
                "model": name,
                "kind": kind,
                "checkpoint": str(path),
                "iter_num": checkpoint.get("iter_num"),
                "best_val_loss": checkpoint.get("best_val_loss"),
                "best_val_auc_mean": checkpoint.get("best_val_auc_mean"),
                "rows": len(rows),
            }
        )
    summary_rows = summarize(raw_rows, topk)
    write_csv(args.out_dir / "state_generation_raw.csv", raw_rows)
    write_csv(args.out_dir / "state_generation_summary.csv", summary_rows)
    (args.out_dir / "state_generation_run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "data_dir": str(data_dir),
                "models": used_specs,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(raw_rows), "models": used_specs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
