from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import DATA_ROOT, EXTERNAL_ROOT
except ImportError:  # executed from inside the source tree
    from paths import DATA_ROOT, EXTERNAL_ROOT

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.pgs_horizon import (  # noqa: E402
    PGSResidualAdapter,
    adapter_logits,
    collect_final_context_cache,
    load_base_model,
    load_split,
    prediction_metrics,
)
from semantic_delphi_ukb.train_architecture_risk_heads import binary_auc, parse_horizons  # noqa: E402
from semantic_delphi_ukb.tte_targets import (  # noqa: E402
    build_patient_disease_ages,
    load_selected_disease_token_groups,
)


DEFAULT_DATA_DIR = Path(str(EXTERNAL_ROOT / "data" / "ukb_semantic_multitype_explicit_split"))
DEFAULT_PGS_DIR = Path(str(DATA_ROOT / "pgs_i21_v1"))
DEFAULT_BASE_CKPT = REPO_DIR / "results" / "monotonic_horizon" / "monotonic_gated_rope_gate100" / "ckpt.pt"


def load_token_groups_compat(diseases_yaml: Path, data_dir: Path):
    try:
        return load_selected_disease_token_groups(diseases_yaml, data_dir)
    except TypeError as exc:
        if "positional argument" not in str(exc):
            raise
        return load_selected_disease_token_groups(diseases_yaml)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired locked-split evaluation of the PGS residual adapter.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pgs-dir", type=Path, default=DEFAULT_PGS_DIR)
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "docs" / "selected_diseases.yaml")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--row-mask", type=Path, default=None)
    parser.add_argument("--cohort", choices=["complete", "all"], default="complete")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-batch-size", type=int, default=256)
    parser.add_argument("--adapter-batch-size", type=int, default=4096)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def macro_auc(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray, rows: np.ndarray) -> float:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits[rows].astype(np.float64), -40.0, 40.0)))
    aucs = []
    for horizon_index in range(logits.shape[1]):
        for target_index in range(logits.shape[2]):
            valid = mask[rows, horizon_index, target_index]
            outcome = labels[rows, horizon_index, target_index][valid].astype(np.int8)
            scores = probabilities[:, horizon_index, target_index][valid]
            if len(np.unique(outcome)) == 2:
                aucs.append(binary_auc(scores, outcome))
    return float(np.mean(aucs)) if aucs else float("nan")


def paired_bootstrap(
    base_logits: np.ndarray,
    fused_logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        rows = rng.integers(0, len(labels), size=len(labels), endpoint=False)
        base_auc = macro_auc(base_logits, labels, mask, rows)
        fused_auc = macro_auc(fused_logits, labels, mask, rows)
        if np.isfinite(base_auc) and np.isfinite(fused_auc):
            deltas.append(fused_auc - base_auc)
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "bootstrap_samples_requested": samples,
        "bootstrap_samples_valid": len(values),
        "delta_auroc_mean": float(values.mean()),
        "delta_auroc_ci_low": float(np.quantile(values, 0.025)),
        "delta_auroc_ci_high": float(np.quantile(values, 0.975)),
    }


def write_metric_rows(path: Path, base_metrics: dict, fused_metrics: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *base_metrics.keys()])
        writer.writeheader()
        writer.writerow({"model": "Monotonic Gated RoPE", **base_metrics})
        writer.writerow({"model": "Monotonic Gated RoPE + PGS", **fused_metrics})


def write_raw_rows(path: Path, cache, fused_logits, target_ids, horizons) -> None:
    base_prob = 1.0 / (1.0 + np.exp(-np.clip(cache.base_logits.astype(np.float64), -40.0, 40.0)))
    fused_prob = 1.0 / (1.0 + np.exp(-np.clip(fused_logits.astype(np.float64), -40.0, 40.0)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "horizon_years",
                "disease_id",
                "label",
                "valid",
                "pgs_available",
                "base_score",
                "pgs_score",
            ],
        )
        writer.writeheader()
        for patient_index, row_index in enumerate(cache.row_indices):
            for horizon_index, horizon in enumerate(horizons):
                for target_index, disease_id in enumerate(target_ids):
                    writer.writerow(
                        {
                            "row_index": int(row_index),
                            "horizon_years": float(horizon),
                            "disease_id": disease_id,
                            "label": int(cache.labels[patient_index, horizon_index, target_index]),
                            "valid": int(cache.mask[patient_index, horizon_index, target_index]),
                            "pgs_available": int(cache.pgs_available[patient_index]),
                            "base_score": float(base_prob[patient_index, horizon_index, target_index]),
                            "pgs_score": float(fused_prob[patient_index, horizon_index, target_index]),
                        }
                    )


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.adapter_checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    target_ids = list(config["target_disease_ids"])
    target_indices = [int(value) for value in config["target_disease_indices"]]
    horizons = [float(value) for value in config["horizons"]]

    diseases, token_groups = load_token_groups_compat(args.diseases_yaml, args.data_dir)
    for disease_id, target_index in zip(target_ids, target_indices):
        if diseases[target_index].disease_id != disease_id:
            raise RuntimeError("Disease ordering differs from the training checkpoint")
    base_model, _ = load_base_model(args.base_checkpoint, args.device)
    data, p2i, static, pgs, available = load_split(args.data_dir, args.pgs_dir, args.split)
    patient_disease_ages, patient_last_ages = build_patient_disease_ages(
        data, p2i, token_groups, int(base_model.config.vocab_size)
    )
    cache = collect_final_context_cache(
        base_model,
        data,
        p2i,
        static,
        pgs,
        available,
        patient_disease_ages,
        patient_last_ages,
        horizons,
        target_indices,
        device=args.device,
        block_size=int(base_model.config.block_size),
        batch_size=args.base_batch_size,
        available_only=args.cohort == "complete",
        row_mask=np.load(args.row_mask).astype(bool) if args.row_mask is not None else None,
    )
    adapter = PGSResidualAdapter(
        cache.pgs_features.shape[1],
        int(config["hidden_dim"]),
        len(horizons),
        len(target_ids),
        float(config["dropout"]),
    ).to(args.device)
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    fused_logits = adapter_logits(adapter, cache, args.device, args.adapter_batch_size)
    base_metrics = prediction_metrics(cache.base_logits, cache.labels, cache.mask)
    fused_metrics = prediction_metrics(fused_logits, cache.labels, cache.mask)
    bootstrap = paired_bootstrap(
        cache.base_logits,
        fused_logits,
        cache.labels,
        cache.mask,
        args.bootstrap,
        args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_metric_rows(args.out_dir / "metrics.csv", base_metrics, fused_metrics)
    write_raw_rows(args.out_dir / "raw_predictions.csv", cache, fused_logits, target_ids, horizons)
    summary = {
        "split": args.split,
        "cohort": args.cohort,
        "row_mask": str(args.row_mask) if args.row_mask is not None else None,
        "patients": len(cache.row_indices),
        "pgs_available": int(cache.pgs_available.sum()),
        "target_disease_ids": target_ids,
        "horizons": horizons,
        "base": base_metrics,
        "pgs": fused_metrics,
        "paired_bootstrap": bootstrap,
        "pgs_gate": torch.sigmoid(adapter.gate_logits).detach().cpu().tolist(),
        "monotonic_fused": bool(np.all(fused_logits[:, 1:, :] >= fused_logits[:, :-1, :] - 1e-7)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
