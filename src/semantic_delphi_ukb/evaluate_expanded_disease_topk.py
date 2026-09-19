from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.evaluate_state_generation_locked_test import (  # noqa: E402
    DEFAULT_SPECS,
    load_architecture,
    load_medtrajectory,
    load_split,
    load_token_event_types,
    parse_topk,
    resolve_data_dir,
    resolve_token_vocab_csv,
    selected_specs,
)
from semantic_delphi_ukb.architecture_baselines import (  # noqa: E402
    history_only_inputs,
    last_prediction_positions,
)
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    load_token_codes,
    parse_selected_diseases,
    token_ids_for_disease,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate expanded disease-panel Top-K hits from next-event logits.")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--token-vocab-csv", type=Path, default=None)
    parser.add_argument(
        "--diseases-yaml",
        type=Path,
        default=REPO_DIR / "docs" / "expanded_disease_panel_ukb_icd10.yaml",
    )
    parser.add_argument("--counts-wide", type=Path, default=REPO_DIR / "results" / "expanded_disease_panel" / "expanded_disease_panel_counts_wide.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "expanded_disease_panel_topk")
    parser.add_argument("--prefix", type=str, default="expanded_disease_topk")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--topk", type=str, default="1,5,10")
    parser.add_argument("--selection", choices=["right", "random", "left"], default="right")
    parser.add_argument("--padding", choices=["regular", "random", "none"], default="regular")
    parser.add_argument("--models", type=str, default="monotonic,gated,no_rope,old_tte,bert,mamba")
    parser.add_argument("--medtrajectory-checkpoint", type=Path, default=None)
    parser.add_argument("--medtrajectory-name", type=str, default="MedTrajectory custom")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def read_counts_tiers(path: Optional[Path]) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["disease_id"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_disease_membership(data_dir: Path, diseases_yaml: Path):
    diseases = parse_selected_diseases(diseases_yaml)
    token_codes = load_token_codes(data_dir)
    token_groups = [token_ids_for_disease(disease, token_codes) for disease in diseases]
    token_to_diseases: dict[int, list[int]] = defaultdict(list)
    for disease_idx, tokens in enumerate(token_groups):
        for token in tokens:
            token_to_diseases[int(token)].append(disease_idx)
    token_sets = [set(map(int, tokens)) for tokens in token_groups]
    return diseases, token_groups, token_sets, token_to_diseases


def topk_hit(values: Sequence[int], target: int, k: int) -> int:
    return int(target in set(values[:k]))


def group_hit(values: Sequence[int], group_tokens: set[int], k: int) -> int:
    return int(bool(set(values[:k]) & group_tokens))


@torch.no_grad()
def collect_model_rows(
    model_name: str,
    model_kind: str,
    ckpt_path: Path,
    data,
    p2i,
    static,
    token_event_types: Sequence[str],
    diseases,
    token_sets: Sequence[set[int]],
    token_to_diseases: dict[int, list[int]],
    args: argparse.Namespace,
    topk: Sequence[int],
) -> tuple[list[dict], dict]:
    if model_kind == "medtrajectory":
        model, checkpoint = load_medtrajectory(ckpt_path, args.device)
        vocab_size = int(model.config.vocab_size)
    else:
        model, checkpoint = load_architecture(model_kind, ckpt_path, args.device)
        vocab_size = int(model.config.vocab_size)
    model.eval()

    diagnosis_mask = torch.tensor(
        [idx < len(token_event_types) and token_event_types[idx] == "diagnosis" for idx in range(vocab_size)],
        dtype=torch.bool,
        device=args.device,
    )
    max_k = max(topk)
    max_full_k = min(max_k, vocab_size)
    max_diag_k = min(max_k, int(diagnosis_mask.sum().item()))
    rows = []
    batches = [
        torch.arange(start, min(start + args.batch_size, len(p2i)), dtype=torch.long)
        for start in range(0, len(p2i), args.batch_size)
    ]
    padding = None if args.padding == "none" else args.padding
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
        targets = y[batch_index, pos_keep]
        final_logits = logits[batch_index, pos_keep]
        full_top = torch.topk(final_logits, k=max_full_k, dim=-1).indices.detach().cpu().numpy()
        diagnosis_logits = final_logits.clone()
        diagnosis_logits[:, ~diagnosis_mask] = -torch.inf
        diag_top = torch.topk(diagnosis_logits, k=max_diag_k, dim=-1).indices.detach().cpu().numpy()
        ages_years = (age[batch_index, pos_keep].detach().cpu().numpy() / 365.25).astype(float)
        target_delta_years = ((target_age[batch_index, pos_keep] - age[batch_index, pos_keep]).detach().cpu().numpy() / 365.25).astype(float)
        patient_indices = batch_ix[keep.cpu()].numpy()
        for row_idx, raw_target in enumerate(targets.detach().cpu().numpy().tolist()):
            target = int(raw_target)
            if target >= len(token_event_types) or token_event_types[target] != "diagnosis":
                continue
            memberships = token_to_diseases.get(target, [])
            if not memberships:
                continue
            full_values = [int(x) for x in full_top[row_idx].tolist()]
            diag_values = [int(x) for x in diag_top[row_idx].tolist()]
            for disease_idx in memberships:
                disease = diseases[disease_idx]
                record = {
                    "model": model_name,
                    "split": args.split,
                    "patient_index": int(patient_indices[row_idx]),
                    "prediction_age_years": float(ages_years[row_idx]),
                    "target_delta_years": float(target_delta_years[row_idx]),
                    "target_token": target,
                    "disease_id": disease.disease_id,
                    "name": disease.name,
                    "name_cn": disease.name_cn,
                    "category": disease.category,
                    "icd10": ";".join(disease.ranges),
                    "disease_token_count": len(token_sets[disease_idx]),
                }
                for k in topk:
                    kk_full = min(k, len(full_values))
                    kk_diag = min(k, len(diag_values))
                    record[f"full_exact_top{k}_hit"] = topk_hit(full_values, target, kk_full)
                    record[f"diagnosis_exact_top{k}_hit"] = topk_hit(diag_values, target, kk_diag)
                    record[f"diagnosis_group_top{k}_hit"] = group_hit(diag_values, token_sets[disease_idx], kk_diag)
                rows.append(record)
    return rows, checkpoint


def summarize(rows: list[dict], topk: Sequence[int], tier_rows: dict[str, dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["disease_id"])].append(row)
    summary = []
    for (model, disease_id), items in sorted(grouped.items()):
        first = items[0]
        tier = tier_rows.get(disease_id, {})
        out = {
            "model": model,
            "disease_id": disease_id,
            "name": first["name"],
            "name_cn": first["name_cn"],
            "category": first["category"],
            "icd10": first["icd10"],
            "reporting_tier": tier.get("reporting_tier", ""),
            "test_5y_final_context_positives": tier.get("test_5y_final_context_positives", ""),
            "test_10y_final_context_positives": tier.get("test_10y_final_context_positives", ""),
            "n_next_diagnosis_targets": len(items),
            "disease_token_count": first["disease_token_count"],
        }
        for k in topk:
            for metric in ["full_exact", "diagnosis_exact", "diagnosis_group"]:
                key = f"{metric}_top{k}_hit"
                out[f"{key}_rate"] = float(np.mean([int(item[key]) for item in items]))
        summary.append(out)

    if rows:
        by_model: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_model[row["model"]].append(row)
        for model, items in sorted(by_model.items()):
            out = {
                "model": model,
                "disease_id": "overall_panel_targets",
                "name": "Overall disease-panel diagnosis targets",
                "name_cn": "扩展疾病面板诊断目标总体",
                "category": "overall",
                "icd10": "",
                "reporting_tier": "overall",
                "test_5y_final_context_positives": "",
                "test_10y_final_context_positives": "",
                "n_next_diagnosis_targets": len(items),
                "disease_token_count": "",
            }
            for k in topk:
                for metric in ["full_exact", "diagnosis_exact", "diagnosis_group"]:
                    key = f"{metric}_top{k}_hit"
                    out[f"{key}_rate"] = float(np.mean([int(item[key]) for item in items]))
            summary.append(out)
    return summary


def selected_model_specs(args: argparse.Namespace):
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
    specs = selected_model_specs(args)
    if not specs:
        raise ValueError("No models selected.")

    max_vocab_size = 0
    for _, _, path in specs:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model_args" in checkpoint and "vocab_size" in checkpoint["model_args"]:
            max_vocab_size = max(max_vocab_size, int(checkpoint["model_args"]["vocab_size"]))
        if "vocab_size" in checkpoint:
            max_vocab_size = max(max_vocab_size, int(checkpoint["vocab_size"]))
    token_event_types = load_token_event_types(resolve_token_vocab_csv(args, data_dir), max_vocab_size + 1)
    diseases, token_groups, token_sets, token_to_diseases = build_disease_membership(data_dir, args.diseases_yaml)
    tier_rows = read_counts_tiers(args.counts_wide)

    raw_rows = []
    used_specs = []
    for name, kind, path in specs:
        if not path.exists():
            print(f"[skip missing] {name}: {path}", file=sys.stderr)
            continue
        rows, checkpoint = collect_model_rows(
            name,
            kind,
            path,
            data,
            p2i,
            static,
            token_event_types,
            diseases,
            token_sets,
            token_to_diseases,
            args,
            topk,
        )
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
    summary_rows = summarize(raw_rows, topk, tier_rows)
    raw_path = args.out_dir / f"{args.prefix}_raw.csv"
    summary_path = args.out_dir / f"{args.prefix}_summary.csv"
    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary_rows)
    (args.out_dir / f"{args.prefix}_run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "data_dir": str(data_dir),
                "diseases_yaml": str(args.diseases_yaml),
                "counts_wide": str(args.counts_wide),
                "models": used_specs,
                "num_panel_diseases": len(diseases),
                "num_panel_token_groups": sum(1 for tokens in token_groups if tokens),
                "raw_path": str(raw_path),
                "summary_path": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "raw_rows": len(raw_rows), "models": used_specs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
