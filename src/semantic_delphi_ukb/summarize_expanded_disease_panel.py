from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import EXTERNAL_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import EXTERNAL_ROOT, REPO_ROOT

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import last_prediction_positions  # noqa: E402
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    load_selected_disease_token_groups,
)
from semantic_delphi_ukb.train_architecture_risk_heads import build_horizon_targets  # noqa: E402
from utils import get_p2i  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize positive counts for an expanded UKB ICD-10 disease panel.")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--diseases-yaml",
        type=Path,
        default=REPO_DIR / "docs" / "expanded_disease_panel_ukb_icd10.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_DIR / "results" / "expanded_disease_panel",
    )
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument("--headline-min-test-positives", type=int, default=20)
    parser.add_argument("--exploratory-min-test-positives", type=int, default=5)
    parser.add_argument("--prediction-position", choices=["final_context"], default="final_context")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--selection", choices=["right", "random"], default="right")
    parser.add_argument("--padding", choices=["regular", "random", "none"], default="regular")
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def parse_float_list(spec: str) -> list[float]:
    values = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not values:
        raise ValueError("expected at least one value")
    return values


def parse_str_list(spec: str) -> list[str]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one split")
    return values


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        return args.data_dir
    local = REPO_DIR / "data" / args.dataset
    if local.exists():
        return local
    remote_project = Path(str(REPO_ROOT / "data")) / args.dataset
    if remote_project.exists():
        return remote_project
    remote_reference = Path(str(EXTERNAL_ROOT / "data")) / args.dataset
    return remote_reference


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    return data, p2i, static


def protocol_positive_counts(
    data: np.ndarray,
    p2i: np.ndarray,
    static: np.ndarray,
    horizons: Sequence[float],
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    patient_last_ages: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[float, list[int]], dict[float, list[int]], dict[float, list[int]]]:
    num_diseases = len(patient_disease_ages[0])
    positives = {float(h): [0 for _ in range(num_diseases)] for h in horizons}
    valid = {float(h): [0 for _ in range(num_diseases)] for h in horizons}
    with_any_positive = {float(h): [0 for _ in range(num_diseases)] for h in horizons}
    padding = None if args.padding == "none" else args.padding
    batches = [
        torch.arange(start, min(start + args.batch_size, len(p2i)), dtype=torch.long)
        for start in range(0, len(p2i), args.batch_size)
    ]
    for ix in batches:
        x, age, y, _, _ = get_batch(
            ix,
            data,
            p2i,
            static,
            block_size=args.block_size,
            device=args.device,
            padding=padding,
            select=args.selection,
            cut_batch=True,
        )
        keep, _ = last_prediction_positions(x, y)
        labels, mask, _ = build_horizon_targets(
            ix, x, age, y, patient_disease_ages, patient_last_ages, horizons, args.device
        )
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        keep_np = keep.detach().cpu().numpy().astype(bool)
        for batch_idx, keep_flag in enumerate(keep_np):
            if not keep_flag:
                continue
            for horizon_idx, horizon in enumerate(horizons):
                h = float(horizon)
                for disease_idx in range(num_diseases):
                    if not mask_np[batch_idx, horizon_idx, disease_idx]:
                        continue
                    valid[h][disease_idx] += 1
                    if labels_np[batch_idx, horizon_idx, disease_idx] > 0.5:
                        positives[h][disease_idx] += 1
                        with_any_positive[h][disease_idx] += 1
    return positives, valid, with_any_positive


def patient_level_positive_counts(
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    horizons: Sequence[float],
) -> dict[str, list[int]]:
    any_history = [0 for _ in patient_disease_ages[0]]
    by_horizon = {f"ever_within_{int(h)}y_from_first": [0 for _ in patient_disease_ages[0]] for h in horizons}
    for per_disease in patient_disease_ages:
        for disease_idx, ages in enumerate(per_disease):
            if len(ages) == 0:
                continue
            any_history[disease_idx] += 1
            first_age = float(ages[0])
            for horizon in horizons:
                horizon_days = float(horizon) * 365.25
                if np.any((ages - first_age) <= horizon_days):
                    by_horizon[f"ever_within_{int(horizon)}y_from_first"][disease_idx] += 1
    return {"ever_in_split": any_history, **by_horizon}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_panel_rows(args: argparse.Namespace, data_dir: Path, splits: Sequence[str], horizons: Sequence[float]) -> tuple[list[dict], list[dict]]:
    diseases, token_groups = load_selected_disease_token_groups(args.diseases_yaml)
    token_group_sizes = [len(tokens) for tokens in token_groups]
    long_rows: list[dict] = []
    wide: dict[str, dict] = {}

    for split in splits:
        data, p2i, static = load_split(data_dir, split)
        vocab_size = int(max(data[:, 2].max(), max(max(tokens, default=0) for tokens in token_groups))) + 2
        patient_disease_ages, patient_last_ages = build_patient_disease_ages(data, p2i, token_groups, vocab_size)
        final_counts, valid_counts, _ = protocol_positive_counts(
            data, p2i, static, horizons, patient_disease_ages, patient_last_ages, args
        )
        patient_counts = patient_level_positive_counts(patient_disease_ages, horizons)
        for disease_idx, disease in enumerate(diseases):
            key = disease.disease_id
            record = wide.setdefault(
                key,
                {
                    "disease_id": disease.disease_id,
                    "name": disease.name,
                    "name_cn": disease.name_cn,
                    "category": disease.category,
                    "icd10": ";".join(disease.ranges),
                    "token_count": token_group_sizes[disease_idx],
                },
            )
            record[f"{split}_patients"] = len(p2i)
            record[f"{split}_ever_positive"] = patient_counts["ever_in_split"][disease_idx]
            for horizon in horizons:
                h = int(horizon)
                positives = final_counts[float(horizon)][disease_idx]
                valid = valid_counts[float(horizon)][disease_idx]
                record[f"{split}_{h}y_final_context_positives"] = positives
                record[f"{split}_{h}y_final_context_valid"] = valid
                long_rows.append(
                    {
                        "split": split,
                        "horizon_years": horizon,
                        "disease_id": disease.disease_id,
                        "name": disease.name,
                        "name_cn": disease.name_cn,
                        "category": disease.category,
                        "icd10": ";".join(disease.ranges),
                        "token_count": token_group_sizes[disease_idx],
                        "patients": len(p2i),
                        "ever_positive": patient_counts["ever_in_split"][disease_idx],
                        "final_context_valid": valid,
                        "final_context_positives": positives,
                        "final_context_event_rate": positives / max(1, valid),
                    }
                )
    wide_rows = list(wide.values())
    for row in wide_rows:
        test_counts = [
            int(row.get(f"test_{int(h)}y_final_context_positives", 0))
            for h in horizons
        ]
        min_test = min(test_counts) if test_counts else 0
        max_test = max(test_counts) if test_counts else 0
        if row["token_count"] == 0:
            tier = "no_token_coverage"
        elif min_test >= args.headline_min_test_positives:
            tier = "headline"
        elif max_test >= args.exploratory_min_test_positives:
            tier = "exploratory"
        else:
            tier = "low_count"
        row["reporting_tier"] = tier
        row["min_test_final_context_positives"] = min_test
        row["max_test_final_context_positives"] = max_test
    wide_rows.sort(key=lambda r: (r["reporting_tier"], r["category"], r["disease_id"]))
    return long_rows, wide_rows


def write_summary(path: Path, args: argparse.Namespace, wide_rows: list[dict]) -> None:
    tiers: dict[str, list[dict]] = {}
    for row in wide_rows:
        tiers.setdefault(str(row["reporting_tier"]), []).append(row)
    lines = [
        "# Expanded UKB ICD-10 Disease Panel Summary",
        "",
        f"- Disease YAML: `{args.diseases_yaml}`",
        f"- Label protocol: `get_batch(select={args.selection}, padding={args.padding}, block_size={args.block_size})` + existing locked-test `build_horizon_targets`",
        f"- Headline threshold: min test positives across horizons >= `{args.headline_min_test_positives}`",
        f"- Exploratory threshold: max test positives across horizons >= `{args.exploratory_min_test_positives}`",
        "",
        "## Tier counts",
        "",
    ]
    for tier in ["headline", "exploratory", "low_count", "no_token_coverage"]:
        lines.append(f"- `{tier}`: {len(tiers.get(tier, []))}")
    lines.extend(["", "## Headline candidates", "", "| Disease | Category | ICD-10 | Tokens | Test min/max positives |", "|---|---|---|---:|---:|"])
    for row in tiers.get("headline", []):
        lines.append(
            f"| {row['name']} | {row['category']} | {row['icd10']} | {row['token_count']} | "
            f"{row['min_test_final_context_positives']} / {row['max_test_final_context_positives']} |"
        )
    lines.extend(["", "## Exploratory candidates", "", "| Disease | Category | ICD-10 | Tokens | Test min/max positives |", "|---|---|---|---:|---:|"])
    for row in tiers.get("exploratory", []):
        lines.append(
            f"| {row['name']} | {row['category']} | {row['icd10']} | {row['token_count']} | "
            f"{row['min_test_final_context_positives']} / {row['max_test_final_context_positives']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(args)
    splits = parse_str_list(args.splits)
    horizons = parse_float_list(args.horizons)
    long_rows, wide_rows = make_panel_rows(args, data_dir, splits, horizons)
    write_csv(args.out_dir / "expanded_disease_panel_counts_long.csv", long_rows)
    write_csv(args.out_dir / "expanded_disease_panel_counts_wide.csv", wide_rows)
    write_summary(args.out_dir / "STATUS.md", args, wide_rows)
    (args.out_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "data_dir": str(data_dir),
                "splits": splits,
                "horizons": horizons,
                "num_diseases": len(wide_rows),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "num_diseases": len(wide_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
