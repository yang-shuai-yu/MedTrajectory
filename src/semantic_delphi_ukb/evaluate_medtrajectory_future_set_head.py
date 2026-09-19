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

from semantic_delphi_ukb.tte_model import FutureDiseaseSetConfig, FutureDiseaseSetMedTrajectory  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    disease_specs_payload,
    load_selected_disease_token_groups,
)
from semantic_delphi_ukb.train_architecture_risk_heads import parse_horizons  # noqa: E402
from semantic_delphi_ukb.train_medtrajectory_future_set_head import (  # noqa: E402
    estimate,
    load_split,
    normalize_state_dict,
)
from semantic_delphi_ukb.train_medtrajectory_horizon_risk import resolve_diseases_yaml  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a future disease-set MedTrajectory checkpoint.")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--horizons", type=str, default="")
    parser.add_argument("--seed", type=int, default=20260701)
    return parser


def load_model(ckpt_path: Path, device: str) -> tuple[FutureDiseaseSetMedTrajectory, dict]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = FutureDiseaseSetConfig(**checkpoint["model_args"])
    model = FutureDiseaseSetMedTrajectory(config)
    model.load_state_dict(normalize_state_dict(checkpoint["model"]), strict=True)
    return model.to(device), checkpoint


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(args.ckpt, args.device)
    ckpt_config = checkpoint.get("config", {})
    horizons = parse_horizons(args.horizons) if args.horizons else [float(x) for x in ckpt_config.get("horizons", [5.0, 10.0])]
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    data_dir = args.data_dir or (REPO_DIR / "data" / args.dataset)
    diseases, token_groups = load_selected_disease_token_groups(diseases_yaml)
    data, p2i, static = load_split(data_dir, args.split, args.max_patients)
    patient_disease_ages, patient_last_ages = build_patient_disease_ages(data, p2i, token_groups, int(model.config.vocab_size))

    metrics = estimate(model, data, p2i, static, patient_disease_ages, patient_last_ages, diseases, horizons, args)
    payload = {
        "split": args.split,
        "ckpt": str(args.ckpt),
        "iter_num": checkpoint.get("iter_num"),
        "best_val_future_set_score": checkpoint.get("best_val_future_set_score"),
        "loss": metrics["loss"],
        "auc_mean": metrics["auc_mean"],
        "top_decile_capture_mean": metrics["top_decile_capture_mean"],
        "horizons": horizons,
        "diseases": disease_specs_payload(diseases),
    }
    (args.out_dir / "eval_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (args.out_dir / "future_set_metrics_rows.json").write_text(json.dumps(metrics["set_rows"], ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "risk_metrics_rows.json").write_text(json.dumps(metrics["rows"], ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Future disease-set locked evaluation",
        "",
        f"Split: `{args.split}`",
        f"Checkpoint: `{args.ckpt}`",
        f"Best validation future-set score: `{checkpoint.get('best_val_future_set_score')}`",
        "",
        "| Horizon | Micro AUC | Recall@3 | Precision@3 | Jaccard@3 | Recall@5 | Jaccard@5 | Positive rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["set_rows"]:
        lines.append(
            "| {horizon_years:.0f}y | {micro_auc:.4f} | {recall_at_3:.4f} | {precision_at_3:.4f} | {jaccard_at_3:.4f} | {recall_at_5:.4f} | {jaccard_at_5:.4f} | {positive_rate:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this branch evaluates a future disease-set filter/reranker. It is not yet the final autoregressive generator.",
            "",
        ]
    )
    (args.out_dir / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
