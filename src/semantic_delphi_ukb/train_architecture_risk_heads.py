from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
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
    HistoryMaskRoPEBERTBaseline,
    HorizonRiskHead,
    MambaHistoryBaseline,
    history_only_inputs,
    last_prediction_positions,
)
from semantic_delphi_ukb.multitype_batch import get_batch  # noqa: E402
from utils import get_p2i  # noqa: E402


@dataclass
class DiseaseSpec:
    disease_id: str
    name: str
    name_cn: str
    category: str
    ranges: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train horizon disease risk heads on BERT/Mamba history encoders.")
    parser.add_argument("--model", choices=["bert", "bert_rope", "mamba"], default="bert")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--diseases-yaml", type=Path, default=None)
    parser.add_argument("--init-from-ckpt", type=Path, default=None)
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
    parser.add_argument("--risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--next-event-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--horizons", type=str, default="5,10")
    parser.add_argument(
        "--target-protocol",
        choices=["legacy_non_censoring", "paper_censor_aware_v1"],
        default="legacy_non_censoring",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser


def parse_horizons(spec: str) -> list[float]:
    values = sorted({float(item.strip()) for item in spec.split(",") if item.strip()})
    if not values:
        raise ValueError("at least one horizon is required")
    return values


def resolve_diseases_yaml(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    root_path = REPO_DIR / "selected_diseases.yaml"
    if root_path.exists():
        return root_path
    return REPO_DIR / "docs" / "selected_diseases.yaml"


def parse_selected_diseases(path: Path) -> list[DiseaseSpec]:
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw_items = payload.get("diseases", payload)
    except Exception:
        raw_items = parse_simple_yaml(path)
    return [
        DiseaseSpec(
            disease_id=str(item["id"]),
            name=str(item["name"]),
            name_cn=str(item.get("name_cn", item["name"])),
            category=str(item.get("category", "")),
            ranges=[str(x).strip().upper() for x in item["icd10"]],
        )
        for item in raw_items
    ]


def parse_simple_yaml(path: Path) -> list[dict]:
    items: list[dict] = []
    current: Optional[dict] = None
    reading_codes = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "diseases:":
            continue
        if line.startswith("- id:"):
            if current is not None:
                items.append(current)
            current = {"id": clean_scalar(line.split(":", 1)[1])}
            reading_codes = False
            continue
        if current is None:
            continue
        if reading_codes and line.startswith("- "):
            current.setdefault("icd10", []).append(clean_scalar(line[2:]))
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() == "icd10":
                current["icd10"] = []
                reading_codes = True
            else:
                current[key.strip()] = clean_scalar(value)
                reading_codes = False
    if current is not None:
        items.append(current)
    return items


def clean_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def load_token_codes(data_dir: Path) -> dict[int, str]:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    vocab_csv = Path(str(manifest.get("vocab_csv", "vocab/dynamic_token_vocab.csv")))
    if not vocab_csv.is_absolute():
        vocab_csv = data_dir / vocab_csv
    token_codes: dict[int, str] = {}
    with vocab_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type", "").strip() == "diagnosis" and row.get("code_norm", "").strip():
                token_codes[int(row["token_id"])] = row["code_norm"].strip().upper()
    return token_codes


def token_ids_for_disease(disease: DiseaseSpec, token_codes: dict[int, str]) -> list[int]:
    return sorted(token_id for token_id, code in token_codes.items() if any(code_matches_range(code, spec) for spec in disease.ranges))


def code_matches_range(code: str, spec: str) -> bool:
    code = code.strip().upper()
    spec = spec.strip().upper()
    if "-" not in spec:
        return code == spec or code.startswith(spec + ".")
    start, stop = [part.strip() for part in spec.split("-", 1)]
    parsed_code = parse_three_char_code(code)
    parsed_start = parse_three_char_code(start)
    parsed_stop = parse_three_char_code(stop)
    if parsed_code is None or parsed_start is None or parsed_stop is None:
        return False
    letter, number = parsed_code
    start_letter, start_number = parsed_start
    stop_letter, stop_number = parsed_stop
    return letter == start_letter == stop_letter and start_number <= number <= stop_number


def parse_three_char_code(code: str):
    match = re.match(r"^([A-Z])([0-9]{2})", code)
    return (match.group(1), int(match.group(2))) if match else None


def load_selected_disease_token_groups(path: Path, data_dir: Path) -> tuple[list[DiseaseSpec], list[list[int]]]:
    diseases = parse_selected_diseases(path)
    token_codes = load_token_codes(data_dir)
    return diseases, [token_ids_for_disease(disease, token_codes) for disease in diseases]


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    return float((ranks[labels.astype(bool)].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def top_decile_stats(scores: np.ndarray, labels: np.ndarray):
    if len(scores) == 0 or labels.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    top_n = max(1, int(math.ceil(len(scores) * 0.10)))
    top_labels = labels[order[:top_n]]
    capture = float(top_labels.sum() / labels.sum())
    top_rate = float(top_labels.mean())
    baseline = float(labels.mean())
    return capture, top_rate, top_rate / baseline if baseline > 0 else float("nan")


def build_patient_disease_ages(
    data: np.ndarray,
    p2i: np.ndarray,
    token_groups: Sequence[Sequence[int]],
    vocab_size: int,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    token_to_diseases: list[list[int]] = [[] for _ in range(vocab_size)]
    for disease_idx, tokens in enumerate(token_groups):
        for token in tokens:
            if 0 <= int(token) < vocab_size:
                token_to_diseases[int(token)].append(disease_idx)

    patient_disease_ages: list[list[np.ndarray]] = []
    patient_last_ages = np.zeros(len(p2i), dtype=np.float32)
    for start, length in p2i:
        rows = data[int(start) : int(start) + int(length)]
        per_disease = [[] for _ in token_groups]
        if len(rows):
            patient_last_ages[len(patient_disease_ages)] = float(rows[-1, 1])
        for _, age_days, raw_token in rows:
            token_id = int(raw_token) + 1
            if 0 <= token_id < vocab_size:
                for disease_idx in token_to_diseases[token_id]:
                    per_disease[disease_idx].append(float(age_days))
        patient_disease_ages.append([np.asarray(values, dtype=np.float32) for values in per_disease])
    return patient_disease_ages, patient_last_ages


def load_split(data_dir: Path, split: str, max_patients: int):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    p2i = get_p2i(data)
    if max_patients > 0:
        p2i = p2i[:max_patients]
        static = static[:max_patients]
    return data, p2i, static


def load_followup_end_ages(data_dir: Path, split: str, observed_last_ages: np.ndarray, max_patients: int) -> np.ndarray:
    path = data_dir / f"{split}_followup_end_age_days.npy"
    if not path.exists():
        return observed_last_ages
    values = np.load(path).astype(np.float32)
    if max_patients > 0:
        values = values[:max_patients]
    if len(values) != len(observed_last_ages):
        raise ValueError(f"{path} is not aligned with {split}.bin")
    return values


def load_vocab_size(data_dir: Path, train_data: np.ndarray, val_data: np.ndarray) -> int:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    return int(manifest.get("vocab_size", int(max(train_data[:, 2].max(), val_data[:, 2].max())) + 2))


def make_encoder(args: argparse.Namespace, vocab_size: int, static_dim: int):
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


def build_horizon_targets(
    ix: Sequence[int] | torch.Tensor,
    x: torch.Tensor,
    age: torch.Tensor,
    y: torch.Tensor,
    patient_disease_ages: Sequence[Sequence[np.ndarray]],
    patient_last_ages: np.ndarray,
    horizons: Sequence[float],
    device: str,
    censor_aware: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ix_list = [int(i) for i in ix.tolist()] if torch.is_tensor(ix) else [int(i) for i in ix]
    keep, pos = last_prediction_positions(x, y)
    ages_np = age.detach().cpu().numpy()
    num_diseases = len(patient_disease_ages[0])
    labels = np.zeros((len(ix_list), len(horizons), num_diseases), dtype=np.float32)
    mask = np.zeros_like(labels)
    durations = np.ones_like(labels)

    for batch_idx, patient_idx in enumerate(ix_list):
        if not bool(keep[batch_idx]):
            continue
        current_age = float(ages_np[batch_idx, int(pos[batch_idx].item())])
        last_age = float(patient_last_ages[patient_idx])
        if current_age <= -5000.0 or current_age >= last_age:
            continue
        for horizon_idx, horizon_years in enumerate(horizons):
            horizon_days = float(horizon_years) * 365.25
            censor_days = max(1.0, min(last_age - current_age, horizon_days))
            for disease_idx, disease_ages in enumerate(patient_disease_ages[patient_idx]):
                next_idx = bisect.bisect_right(disease_ages, current_age)
                event = 0.0
                duration = censor_days
                if next_idx < len(disease_ages):
                    delta = float(disease_ages[next_idx]) - current_age
                    if 0.0 < delta <= horizon_days:
                        event = 1.0
                        duration = max(1.0, delta)
                if censor_aware and event == 0.0 and last_age - current_age < horizon_days:
                    continue
                labels[batch_idx, horizon_idx, disease_idx] = event
                durations[batch_idx, horizon_idx, disease_idx] = duration / 365.25
                mask[batch_idx, horizon_idx, disease_idx] = 1.0

    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
        torch.tensor(durations, dtype=torch.float32, device=device),
    )


def risk_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def estimate(model, risk_head, data, p2i, static, patient_disease_ages, patient_last_ages, diseases, horizons, args):
    model.eval()
    risk_head.eval()
    losses = []
    buckets = {
        float(horizon): [
            {"scores": [], "labels": [], "durations": []}
            for _ in diseases
        ]
        for horizon in horizons
    }
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
        hidden = model.encode(x_in, age_in, s)
        _, pos = last_prediction_positions(x, y)
        logits = risk_head(hidden, pos)
        labels, mask, durations = build_horizon_targets(
            ix,
            x,
            age,
            y,
            patient_disease_ages,
            patient_last_ages,
            horizons,
            args.device,
            censor_aware=args.target_protocol == "paper_censor_aware_v1",
        )
        losses.append(float(risk_loss(logits, labels, mask).detach().cpu()))
        scores_np = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        durations_np = durations.detach().cpu().numpy()
        for horizon_idx, horizon in enumerate(horizons):
            for disease_idx in range(len(diseases)):
                valid = mask_np[:, horizon_idx, disease_idx]
                if not valid.any():
                    continue
                bucket = buckets[float(horizon)][disease_idx]
                bucket["scores"].append(scores_np[:, horizon_idx, disease_idx][valid])
                bucket["labels"].append(labels_np[:, horizon_idx, disease_idx][valid])
                bucket["durations"].append(durations_np[:, horizon_idx, disease_idx][valid])
    model.train()
    risk_head.train()

    rows = []
    auc_values = []
    capture_values = []
    for horizon in horizons:
        for disease_idx, disease in enumerate(diseases):
            bucket = buckets[float(horizon)][disease_idx]
            if bucket["scores"]:
                scores = np.concatenate(bucket["scores"]).astype(np.float64)
                labels = np.concatenate(bucket["labels"]).astype(np.int8)
            else:
                scores = np.asarray([], dtype=np.float64)
                labels = np.asarray([], dtype=np.int8)
            positives = int(labels.sum())
            negatives = int(labels.size - positives)
            auc = (
                binary_auc(scores, labels)
                if args.target_protocol == "legacy_non_censoring" and positives > 0 and negatives > 0
                else float("nan")
            )
            capture, top_rate, lift = top_decile_stats(scores, labels)
            if not math.isnan(auc):
                auc_values.append(auc)
            if not math.isnan(capture):
                capture_values.append(capture)
            rows.append(
                {
                    "horizon_years": float(horizon),
                    "disease_id": disease.disease_id,
                    "name": disease.name,
                    "name_cn": disease.name_cn,
                    "positives": positives,
                    "negatives": negatives,
                    "auc": auc,
                    "top_decile_capture": capture,
                    "top_decile_event_rate": top_rate,
                    "top_decile_lift": lift,
                }
            )
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc_mean": float(np.mean(auc_values)) if auc_values else float("nan"),
        "top_decile_capture_mean": float(np.mean(capture_values)) if capture_values else float("nan"),
        "rows": rows,
    }


def self_test(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    args.device = args.device
    horizons = parse_horizons(args.horizons)
    diseases = parse_selected_diseases(resolve_diseases_yaml(args.diseases_yaml))[:3]
    model = make_encoder(args, vocab_size=128, static_dim=10).to(args.device)
    risk_head = HorizonRiskHead(args.n_embd, len(diseases), len(horizons)).to(args.device)
    batch_size = 4
    seq_len = min(args.block_size, 16)
    x = torch.randint(2, 128, (batch_size, seq_len), device=args.device)
    age = torch.sort(torch.randint(1000, 30000, (batch_size, seq_len), device=args.device).float(), dim=1).values
    y = torch.randint(2, 128, (batch_size, seq_len), device=args.device)
    static = torch.randn(batch_size, 10, device=args.device)
    _, pos = last_prediction_positions(x, y)
    hidden = model.encode(*history_only_inputs(x, age, y), static)
    logits = risk_head(hidden, pos)
    labels = torch.randint(0, 2, logits.shape, device=args.device).float()
    mask = torch.ones_like(labels)
    loss = risk_loss(logits, labels, mask)
    loss.backward()
    print(json.dumps({"self_test": True, "model": args.model, "loss": float(loss.detach().cpu())}, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args)

    torch.manual_seed(args.seed)
    horizons = parse_horizons(args.horizons)
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    data_dir = args.data_dir or (REPO_DIR / "data" / args.dataset)
    diseases, token_groups = load_selected_disease_token_groups(diseases_yaml, data_dir)
    out_dir = args.out_dir or (REPO_DIR / "results" / "architecture_risk_heads" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data, train_p2i, train_static = load_split(data_dir, "train", args.max_patients)
    val_data, val_p2i, val_static = load_split(data_dir, "val", args.max_patients)
    vocab_size = load_vocab_size(data_dir, train_data, val_data)
    static_dim = int(train_static.shape[1])
    model = make_encoder(args, vocab_size=vocab_size, static_dim=static_dim).to(args.device)
    if args.init_from_ckpt is not None:
        checkpoint = torch.load(args.init_from_ckpt, map_location=args.device, weights_only=False)
        state_dict = dict(checkpoint["model"])
        for key in list(state_dict):
            if key.startswith("_orig_mod."):
                state_dict[key[len("_orig_mod.") :]] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=True)
    risk_head = HorizonRiskHead(args.n_embd, len(diseases), len(horizons)).to(args.device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(risk_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_tte_ages, train_last_ages = build_patient_disease_ages(train_data, train_p2i, token_groups, vocab_size)
    val_tte_ages, val_last_ages = build_patient_disease_ages(val_data, val_p2i, token_groups, vocab_size)
    if args.target_protocol == "paper_censor_aware_v1":
        train_last_ages = load_followup_end_ages(data_dir, "train", train_last_ages, args.max_patients)
        val_last_ages = load_followup_end_ages(data_dir, "val", val_last_ages, args.max_patients)

    best_auc = -float("inf")
    best_loss = float("inf")
    history = []
    for iteration in range(args.max_iters + 1):
        if iteration % args.eval_interval == 0:
            metrics = estimate(
                model,
                risk_head,
                val_data,
                val_p2i,
                val_static,
                val_tte_ages,
                val_last_ages,
                diseases,
                horizons,
                args,
            )
            row = {
                "iter": iteration,
                "val_loss": metrics["loss"],
                "val_auc_mean": metrics["auc_mean"],
                "val_top_decile_capture_mean": metrics["top_decile_capture_mean"],
            }
            history.append(row)
            print(json.dumps(row, indent=2))
            improved = (
                metrics["loss"] < best_loss
                if args.target_protocol == "paper_censor_aware_v1"
                else metrics["auc_mean"] > best_auc
            )
            if improved:
                best_loss = min(best_loss, metrics["loss"])
                if not math.isnan(metrics["auc_mean"]):
                    best_auc = max(best_auc, metrics["auc_mean"])
                torch.save(
                    {
                        "model": model.state_dict(),
                        "risk_head": risk_head.state_dict(),
                        "model_name": args.model,
                        "model_args": vars(args),
                        "vocab_size": vocab_size,
                        "static_dim": static_dim,
                        "diseases": [d.__dict__ for d in diseases],
                        "horizons": horizons,
                        "best_val_auc_mean": best_auc,
                        "best_val_loss": best_loss,
                        "selection_metric": (
                            "val_loss" if args.target_protocol == "paper_censor_aware_v1" else "legacy_window_auc"
                        ),
                        "iter_num": iteration,
                    },
                    out_dir / "ckpt.pt",
                )
                (out_dir / "risk_metrics_rows.json").write_text(json.dumps(metrics["rows"], ensure_ascii=False, indent=2), encoding="utf-8")
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
        hidden = model.encode(x_in, age_in, s)
        _, pos = last_prediction_positions(x, y)
        logits = risk_head(hidden, pos)
        labels, mask, _ = build_horizon_targets(
            ix,
            x,
            age,
            y,
            train_tte_ages,
            train_last_ages,
            horizons,
            args.device,
            censor_aware=args.target_protocol == "paper_censor_aware_v1",
        )
        loss = args.risk_loss_weight * risk_loss(logits, labels, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(risk_head.parameters()), 1.0)
        optimizer.step()

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
