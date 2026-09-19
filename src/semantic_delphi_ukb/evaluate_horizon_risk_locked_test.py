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

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import (  # noqa: E402
    ArchitectureBaselineConfig,
    HistoryMaskBERTBaseline,
    HistoryMaskRoPEBERTBaseline,
    HorizonRiskHead,
    MambaHistoryBaseline,
    history_only_inputs,
    last_prediction_positions,
)
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from semantic_delphi_ukb.tte_model import (  # noqa: E402
    HorizonRiskConfig,
    HorizonRiskMedTrajectory,
    SurvivalHorizonConfig,
    SurvivalHorizonMedTrajectory,
)
from semantic_delphi_ukb.train_architecture_risk_heads import (  # noqa: E402
    binary_auc,
    build_horizon_targets,
    build_patient_disease_ages,
    load_selected_disease_token_groups,
    parse_horizons,
    top_decile_stats,
)
from utils import get_p2i  # noqa: E402


DEFAULT_CKPTS = {
    "medtrajectory": REPO_DIR / "results" / "medtrajectory_horizon_risk" / "full_bce_aux02" / "ckpt.pt",
    "survival": REPO_DIR / "results" / "medtrajectory_survival_horizon" / "weighted_aux_smoke600" / "ckpt.pt",
    "bert": REPO_DIR / "results" / "architecture_risk_heads" / "bert_full" / "ckpt.pt",
    "bert_rope": REPO_DIR / "results" / "architecture_risk_heads" / "bert_rope_full" / "ckpt.pt",
    "mamba": REPO_DIR / "results" / "architecture_risk_heads" / "mamba_full" / "ckpt.pt",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locked-split horizon risk evaluation with CI and calibration.")
    parser.add_argument("--model", choices=["medtrajectory", "survival", "bert", "bert_rope", "mamba"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--eval-all", action="store_true", help="Evaluate each patient in the split once, in order.")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--selection", choices=["random", "right"], default="random")
    parser.add_argument("--padding", choices=["random", "regular", "none"], default="random")
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--use-age-encoding", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--use-age-rope", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--age-rope-gate", type=float, default=None)
    parser.add_argument("--monotonic-horizon-risk", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def resolve_diseases_yaml(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    root_path = REPO_DIR / "selected_diseases.yaml"
    if root_path.exists():
        return root_path
    return REPO_DIR / "docs" / "selected_diseases.yaml"


def load_split(data_dir: Path, split: str, max_patients: int):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if max_patients > 0:
        p2i = p2i[:max_patients]
        static = static[:max_patients]
    return data, p2i, static


def safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(bool)
    hit_ranks = np.flatnonzero(ranked) + 1
    precision_at_hits = np.arange(1, positives + 1, dtype=np.float64) / hit_ranks
    return float(precision_at_hits.mean())


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def calibration_stats(scores: np.ndarray, labels: np.ndarray, bins: int) -> tuple[float, list[dict]]:
    if labels.size == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count:
            mean_pred = float(scores[mask].mean())
            event_rate = float(labels[mask].mean())
            ece += count / labels.size * abs(mean_pred - event_rate)
        else:
            mean_pred = float("nan")
            event_rate = float("nan")
        rows.append(
            {
                "bin": idx,
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": count,
                "mean_score": mean_pred,
                "event_rate": event_rate,
            }
        )
    return float(ece), rows


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float]:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if bootstrap <= 0 or positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    n = labels.size
    for _ in range(bootstrap):
        sample = rng.integers(0, n, n)
        y = labels[sample]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aucs.append(binary_auc(scores[sample], y))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def top_decile_event_count(scores: np.ndarray, labels: np.ndarray) -> int:
    if labels.size == 0:
        return 0
    order = np.argsort(-scores, kind="mergesort")
    top_n = max(1, int(math.ceil(labels.size * 0.10)))
    return int(labels[order[:top_n]].sum())


def apply_temporal_overrides(model_args: dict, args: argparse.Namespace) -> dict:
    model_args = model_args.copy()
    if args.use_age_encoding != "auto":
        model_args["use_age_encoding"] = args.use_age_encoding == "true"
    if args.use_age_rope != "auto":
        model_args["use_age_rope"] = args.use_age_rope == "true"
    if args.age_rope_gate is not None:
        model_args["age_rope_gate"] = args.age_rope_gate
    if args.monotonic_horizon_risk != "auto":
        model_args["monotonic_horizon_risk"] = args.monotonic_horizon_risk == "true"
    return model_args


def load_medtrajectory_model(checkpoint_path: Path, device: str, args: argparse.Namespace):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = apply_temporal_overrides(checkpoint["model_args"], args)
    model = HorizonRiskMedTrajectory(HorizonRiskConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device), checkpoint


def load_survival_model(checkpoint_path: Path, device: str, args: argparse.Namespace):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = apply_temporal_overrides(checkpoint["model_args"], args)
    model = SurvivalHorizonMedTrajectory(SurvivalHorizonConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device), checkpoint


def load_architecture_model(model_name: str, checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    if model_name == "bert":
        encoder = HistoryMaskBERTBaseline(config)
    elif model_name == "bert_rope":
        encoder = HistoryMaskRoPEBERTBaseline(config)
    else:
        encoder = MambaHistoryBaseline(config)
    risk_head = HorizonRiskHead(config.n_embd, len(checkpoint["diseases"]), len(checkpoint["horizons"]))
    encoder.load_state_dict(checkpoint["model"], strict=True)
    risk_head.load_state_dict(checkpoint["risk_head"], strict=True)
    return encoder.to(device), risk_head.to(device), checkpoint


@torch.no_grad()
def collect_predictions(
    args: argparse.Namespace,
    data,
    p2i,
    static,
    patient_disease_ages,
    patient_last_ages,
    diseases,
    horizons,
):
    checkpoint_path = args.checkpoint or DEFAULT_CKPTS[args.model]
    raw_rows = []
    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)
    batches = patient_batches(len(p2i), args.batch_size) if args.eval_all else None

    if args.model == "medtrajectory":
        model, checkpoint = load_medtrajectory_model(checkpoint_path, args.device, args)
        model.eval()
        model_id = "MedTrajectory horizon risk"
        iterator = enumerate(batches) if batches is not None else ((i, torch.randint(len(p2i), (args.batch_size,), generator=rng)) for i in range(args.eval_iters))
        for eval_iter, ix in iterator:
            x, age, y, target_age, s = get_batch(
                ix,
                data,
                p2i,
                static,
                block_size=args.block_size,
                device=args.device,
                padding=None if args.padding == "none" else args.padding,
                select=args.selection,
                cut_batch=True,
            )
            _, _, _, _, risk_logits = model(x, age, s, y, target_age, validation_loss_mode=True)
            keep, pos = last_prediction_positions(x, y)
            labels, mask, durations = build_horizon_targets(
                ix, x, age, y, patient_disease_ages, patient_last_ages, horizons, args.device
            )
            batch_index = torch.arange(risk_logits.size(0), device=args.device)
            scores = torch.sigmoid(risk_logits[batch_index, pos]).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy().astype(bool)
            durations_np = durations.detach().cpu().numpy()
            ages_np = age[batch_index, pos].detach().cpu().numpy() / 365.25
            append_prediction_rows(
                raw_rows,
                model_id,
                args.split,
                eval_iter,
                ix.numpy(),
                keep.detach().cpu().numpy().astype(bool),
                ages_np,
                scores,
                labels_np,
                mask_np,
                durations_np,
                diseases,
                horizons,
            )
    elif args.model == "survival":
        model, checkpoint = load_survival_model(checkpoint_path, args.device, args)
        model.eval()
        model_id = "MedTrajectory survival-horizon"
        iterator = enumerate(batches) if batches is not None else ((i, torch.randint(len(p2i), (args.batch_size,), generator=rng)) for i in range(args.eval_iters))
        for eval_iter, ix in iterator:
            x, age, y, target_age, s = get_batch(
                ix,
                data,
                p2i,
                static,
                block_size=args.block_size,
                device=args.device,
                padding=None if args.padding == "none" else args.padding,
                select=args.selection,
                cut_batch=True,
            )
            _, _, _, _, survival_logits = model(x, age, s, y, target_age, validation_loss_mode=True)
            keep, pos = last_prediction_positions(x, y)
            labels, mask, durations = build_horizon_targets(
                ix, x, age, y, patient_disease_ages, patient_last_ages, horizons, args.device
            )
            batch_index = torch.arange(survival_logits.size(0), device=args.device)
            scores = model.horizon_risk_from_survival(survival_logits[batch_index, pos], horizons).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy().astype(bool)
            durations_np = durations.detach().cpu().numpy()
            ages_np = age[batch_index, pos].detach().cpu().numpy() / 365.25
            append_prediction_rows(
                raw_rows,
                model_id,
                args.split,
                eval_iter,
                ix.numpy(),
                keep.detach().cpu().numpy().astype(bool),
                ages_np,
                scores,
                labels_np,
                mask_np,
                durations_np,
                diseases,
                horizons,
            )
    else:
        encoder, risk_head, checkpoint = load_architecture_model(args.model, checkpoint_path, args.device)
        encoder.eval()
        risk_head.eval()
        model_ids = {
            "bert": "BERT risk head",
            "bert_rope": "BERT+RoPE risk head",
            "mamba": "Mamba risk head",
        }
        model_id = model_ids[args.model]
        iterator = enumerate(batches) if batches is not None else ((i, torch.randint(len(p2i), (args.batch_size,), generator=rng)) for i in range(args.eval_iters))
        for eval_iter, ix in iterator:
            x, age, y, target_age, s = get_batch(
                ix,
                data,
                p2i,
                static,
                block_size=args.block_size,
                device=args.device,
                padding=None if args.padding == "none" else args.padding,
                select=args.selection,
                cut_batch=True,
            )
            x_in, age_in = history_only_inputs(x, age, y)
            hidden = encoder.encode(x_in, age_in, s)
            keep, pos = last_prediction_positions(x, y)
            logits = risk_head(hidden, pos)
            labels, mask, durations = build_horizon_targets(
                ix, x, age, y, patient_disease_ages, patient_last_ages, horizons, args.device
            )
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy().astype(bool)
            durations_np = durations.detach().cpu().numpy()
            ages_np = age[torch.arange(age.size(0), device=args.device), pos].detach().cpu().numpy() / 365.25
            append_prediction_rows(
                raw_rows,
                model_id,
                args.split,
                eval_iter,
                ix.numpy(),
                keep.detach().cpu().numpy().astype(bool),
                ages_np,
                scores,
                labels_np,
                mask_np,
                durations_np,
                diseases,
                horizons,
            )
    return raw_rows, checkpoint_path, checkpoint


def patient_batches(n_patients: int, batch_size: int) -> list[torch.Tensor]:
    return [torch.arange(start, min(start + batch_size, n_patients), dtype=torch.long) for start in range(0, n_patients, batch_size)]


def append_prediction_rows(
    raw_rows,
    model_id: str,
    split: str,
    eval_iter: int,
    patient_indices: np.ndarray,
    keep: np.ndarray,
    ages_years: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    durations: np.ndarray,
    diseases,
    horizons,
):
    for batch_idx, patient_idx in enumerate(patient_indices.tolist()):
        if not keep[batch_idx]:
            continue
        for horizon_idx, horizon in enumerate(horizons):
            for disease_idx, disease in enumerate(diseases):
                if not mask[batch_idx, horizon_idx, disease_idx]:
                    continue
                raw_rows.append(
                    {
                        "model": model_id,
                        "split": split,
                        "eval_iter": eval_iter,
                        "patient_index": int(patient_idx),
                        "prediction_age_years": float(ages_years[batch_idx]),
                        "horizon_years": float(horizon),
                        "disease_id": disease.disease_id,
                        "name": disease.name,
                        "name_cn": disease.name_cn,
                        "label": int(labels[batch_idx, horizon_idx, disease_idx]),
                        "score": float(scores[batch_idx, horizon_idx, disease_idx]),
                        "duration_years": float(durations[batch_idx, horizon_idx, disease_idx]),
                    }
                )


def summarize(raw_rows: list[dict], diseases, horizons, bootstrap: int, seed: int, bins: int):
    summary_rows = []
    calibration_rows = []
    for horizon in horizons:
        for disease in diseases:
            rows = [
                row
                for row in raw_rows
                if float(row["horizon_years"]) == float(horizon) and row["disease_id"] == disease.disease_id
            ]
            scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
            labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
            positives = int(labels.sum())
            negatives = int(labels.size - positives)
            auc = binary_auc(scores, labels) if positives > 0 and negatives > 0 else float("nan")
            auc_low, auc_high = bootstrap_ci(
                scores, labels, bootstrap=bootstrap, seed=seed + int(float(horizon) * 10) + len(summary_rows)
            )
            capture, top_rate, lift = top_decile_stats(scores, labels)
            ece, cal_rows = calibration_stats(scores, labels, bins)
            for cal in cal_rows:
                cal.update(
                    {
                        "horizon_years": float(horizon),
                        "disease_id": disease.disease_id,
                        "name": disease.name,
                        "name_cn": disease.name_cn,
                    }
                )
                calibration_rows.append(cal)
            summary_rows.append(
                {
                    "horizon_years": float(horizon),
                    "disease_id": disease.disease_id,
                    "name": disease.name,
                    "name_cn": disease.name_cn,
                    "prediction_moments": int(labels.size),
                    "positives": positives,
                    "negatives": negatives,
                    "auc": auc,
                    "auc_ci_low": auc_low,
                    "auc_ci_high": auc_high,
                    "average_precision": average_precision(scores, labels),
                    "brier": brier_score(scores, labels),
                    "ece": ece,
                    "top_decile_capture": capture,
                    "top_decile_event_rate": top_rate,
                    "top_decile_event_count": top_decile_event_count(scores, labels),
                    "top_decile_lift": lift,
                    "mean_score": float(scores.mean()) if scores.size else float("nan"),
                    "event_rate": float(labels.mean()) if labels.size else float("nan"),
                }
            )
    aggregate_rows = []
    for horizon in horizons:
        rows = [
            row
            for row in summary_rows
            if float(row["horizon_years"]) == float(horizon) and not math.isnan(safe_float(row["auc"]))
        ]
        aggregate_rows.append(aggregate_metric_row(rows, float(horizon), "horizon"))
    aggregate_rows.append(
        aggregate_metric_row([row for row in summary_rows if not math.isnan(safe_float(row["auc"]))], float("nan"), "overall")
    )
    return summary_rows, aggregate_rows, calibration_rows


def aggregate_metric_row(rows: list[dict], horizon: float, label: str) -> dict:
    def avg(key: str) -> float:
        values = [safe_float(row.get(key)) for row in rows if not math.isnan(safe_float(row.get(key)))]
        return float(np.mean(values)) if values else float("nan")

    return {
        "aggregate": label,
        "horizon_years": horizon,
        "diseases_with_auc": len(rows),
        "auc_mean": avg("auc"),
        "auc_ci_low_mean": avg("auc_ci_low"),
        "auc_ci_high_mean": avg("auc_ci_high"),
        "average_precision_mean": avg("average_precision"),
        "brier_mean": avg("brier"),
        "ece_mean": avg("ece"),
        "top_decile_capture_mean": avg("top_decile_capture"),
        "event_rate_mean": avg("event_rate"),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_status(path: Path, args, checkpoint_path, summary_rows, aggregate_rows):
    lines = [
        f"# Locked {args.split} Horizon Risk Evaluation: {args.model}",
        "",
        f"- Split: `{args.split}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Selection: `{args.selection}`; padding: `{args.padding}`; eval iters: `{args.eval_iters}`; batch size: `{args.batch_size}`",
        f"- Bootstrap replicates: `{args.bootstrap}`; calibration bins: `{args.calibration_bins}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Scope | Horizon | Diseases | AUC mean | AUC CI mean | Brier mean | ECE mean | Top-decile capture mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        horizon = "all" if math.isnan(safe_float(row["horizon_years"])) else f"{row['horizon_years']:g}"
        lines.append(
            "| {scope} | {horizon} | {n} | {auc:.4f} | {lo:.4f}-{hi:.4f} | {brier:.4f} | {ece:.4f} | {cap:.4f} |".format(
                scope=row["aggregate"],
                horizon=horizon,
                n=row["diseases_with_auc"],
                auc=safe_float(row["auc_mean"]),
                lo=safe_float(row["auc_ci_low_mean"]),
                hi=safe_float(row["auc_ci_high_mean"]),
                brier=safe_float(row["brier_mean"]),
                ece=safe_float(row["ece_mean"]),
                cap=safe_float(row["top_decile_capture_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Disease Metrics",
            "",
            "| Horizon | Disease | Positives | AUC | 95% CI | Brier | ECE | Top-decile capture |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {h:g} | {name} | {pos} | {auc:.4f} | {lo:.4f}-{hi:.4f} | {brier:.4f} | {ece:.4f} | {cap:.4f} |".format(
                h=safe_float(row["horizon_years"]),
                name=row["name"],
                pos=int(row["positives"]),
                auc=safe_float(row["auc"]),
                lo=safe_float(row["auc_ci_low"]),
                hi=safe_float(row["auc_ci_high"]),
                brier=safe_float(row["brier"]),
                ece=safe_float(row["ece"]),
                cap=safe_float(row["top_decile_capture"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = args.data_dir or (REPO_DIR / "data" / args.dataset)
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    out_dir = args.out_dir or (REPO_DIR / "results" / "locked_test_horizon_risk" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)
    diseases, token_groups = load_selected_disease_token_groups(diseases_yaml, data_dir)
    data, p2i, static = load_split(data_dir, args.split, args.max_patients)
    vocab_size = int(max(data[:, 2].max(), max(max(tokens, default=0) for tokens in token_groups))) + 2
    patient_disease_ages, patient_last_ages = build_patient_disease_ages(data, p2i, token_groups, vocab_size)
    raw_rows, checkpoint_path, checkpoint = collect_predictions(
        args, data, p2i, static, patient_disease_ages, patient_last_ages, diseases, horizons
    )
    summary_rows, aggregate_rows, calibration_rows = summarize(
        raw_rows, diseases, horizons, args.bootstrap, args.seed, args.calibration_bins
    )
    write_csv(out_dir / "raw_predictions.csv", raw_rows)
    write_csv(out_dir / "risk_metrics_rows.csv", summary_rows)
    write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(out_dir / "calibration_bins.csv", calibration_rows)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "checkpoint": str(checkpoint_path),
                "data_dir": str(data_dir),
                "diseases_yaml": str(diseases_yaml),
                "out_dir": str(out_dir),
                "checkpoint_iter": checkpoint.get("iter_num"),
                "checkpoint_best_val_auc_mean": checkpoint.get("best_val_auc_mean"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_status(out_dir / "STATUS.md", args, checkpoint_path, summary_rows, aggregate_rows)
    print(json.dumps({"out_dir": str(out_dir), "raw_rows": len(raw_rows), "aggregates": aggregate_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
