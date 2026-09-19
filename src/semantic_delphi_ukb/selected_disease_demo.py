from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from semantic_delphi_ukb.evaluate_test import get_batch_safe
from semantic_delphi_ukb.multitype_batch import get_batch as get_multitype_batch
from semantic_delphi_ukb.multitype_model import MultitypeSemanticDelphi, MultitypeSemanticDelphiConfig
from semantic_delphi_ukb.semantic_model import SemanticDelphi, SemanticDelphiConfig
from utils import get_p2i


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]


@dataclass
class DiseaseSpec:
    disease_id: str
    name: str
    name_cn: str
    category: str
    ranges: List[str]


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    model_type: str
    ckpt_path: Path
    data_dir: Path
    token_vocab_csv: Optional[Path] = None


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: torch.nn.Module
    block_size: int
    no_event_token_rate: int
    labels: List[str]
    token_codes: Dict[int, str]
    diagnosis_tokens: List[int]
    data: np.ndarray
    p2i: np.ndarray
    static_matrix: Optional[np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate selected disease groups across disease-only, exp1, and exp2.")
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "selected_diseases.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "tests" / "output" / "selected_disease_demo")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--topk", type=str, default="1,5,10")
    return parser


def parse_selected_diseases(path: Path) -> List[DiseaseSpec]:
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw_items = payload.get("diseases", payload)
    except Exception:
        raw_items = parse_simple_yaml(path)

    diseases = []
    for item in raw_items:
        diseases.append(
            DiseaseSpec(
                disease_id=str(item["id"]),
                name=str(item["name"]),
                name_cn=str(item.get("name_cn", item["name"])),
                category=str(item.get("category", "")),
                ranges=[str(x).strip().upper() for x in item["icd10"]],
            )
        )
    return diseases


def parse_simple_yaml(path: Path) -> List[dict]:
    items: List[dict] = []
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
            key = key.strip()
            if key == "icd10":
                current["icd10"] = []
                reading_codes = True
            else:
                current[key] = clean_scalar(value)
                reading_codes = False
    if current is not None:
        items.append(current)
    return items


def clean_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def default_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            model_id="disease_only",
            display_name="Disease-only",
            model_type="semantic",
            ckpt_path=REPO_DIR / "ckpt" / "Delphi_semantic_icd64_explicit_split" / "ckpt.pt",
            data_dir=REPO_DIR / "data" / "ukb_semantic_icd64_explicit_split",
        ),
        ModelSpec(
            model_id="exp1",
            display_name="Exp1",
            model_type="multitype",
            ckpt_path=REPO_DIR / "ckpt" / "Delphi_semantic_icd64_multitype_exp1_explicit_split" / "ckpt.pt",
            data_dir=REPO_DIR / "data" / "ukb_semantic_multitype_exp1_explicit_split",
        ),
        ModelSpec(
            model_id="exp2",
            display_name="Exp2",
            model_type="multitype",
            ckpt_path=REPO_DIR / "ckpt" / "Delphi_semantic_icd64_multitype_explicit_split" / "ckpt.pt",
            data_dir=REPO_DIR / "data" / "ukb_semantic_multitype_explicit_split",
        ),
    ]


def load_model(spec: ModelSpec, split: str, device: str) -> LoadedModel:
    checkpoint = torch.load(spec.ckpt_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    if spec.model_type == "semantic":
        conf = SemanticDelphiConfig(**checkpoint["model_args"])
        model = SemanticDelphi(conf)
    elif spec.model_type == "multitype":
        conf = MultitypeSemanticDelphiConfig(**checkpoint["model_args"])
        model = MultitypeSemanticDelphi(conf)
    else:
        raise ValueError(f"Unknown model_type: {spec.model_type}")

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    labels = load_labels(spec.data_dir / "labels.csv")
    token_codes, diagnosis_tokens = load_token_codes(spec)
    split_path = spec.data_dir / f"{split}.bin"
    data = np.memmap(split_path, dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    static_matrix = None
    if spec.model_type == "multitype":
        static_matrix = np.load(spec.data_dir / f"{split}_static.npy").astype(np.float32)

    return LoadedModel(
        spec=spec,
        model=model,
        block_size=int(conf.block_size),
        no_event_token_rate=int(checkpoint_config.get("no_event_token_rate", 5)),
        labels=labels,
        token_codes=token_codes,
        diagnosis_tokens=sorted(diagnosis_tokens),
        data=data,
        p2i=p2i,
        static_matrix=static_matrix,
    )


def load_labels(path: Path) -> List[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def load_token_codes(spec: ModelSpec) -> Tuple[Dict[int, str], List[int]]:
    if spec.model_type == "multitype":
        vocab_csv = resolve_token_vocab_csv(spec)
        token_codes: Dict[int, str] = {}
        diagnosis_tokens: List[int] = []
        with vocab_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                token_id = int(row["token_id"])
                event_type = (row.get("event_type") or "").strip()
                code = (row.get("code_norm") or "").strip().upper()
                if event_type == "diagnosis" and code:
                    token_codes[token_id] = code
                    diagnosis_tokens.append(token_id)
        return token_codes, diagnosis_tokens

    token_codes = {}
    diagnosis_tokens = []
    for token_id, label in enumerate(load_labels(spec.data_dir / "labels.csv")):
        code = extract_icd_code(label)
        if code:
            token_codes[token_id] = code
            diagnosis_tokens.append(token_id)
    return token_codes, diagnosis_tokens


def resolve_token_vocab_csv(spec: ModelSpec) -> Path:
    if spec.token_vocab_csv is not None:
        return spec.token_vocab_csv
    manifest_path = spec.data_dir / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Path(str(manifest["vocab_csv"]))


def extract_icd_code(label: str) -> str:
    match = re.match(r"^([A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?)", label.strip().upper())
    return match.group(1) if match else ""


def token_ids_for_disease(disease: DiseaseSpec, token_codes: Dict[int, str]) -> List[int]:
    return sorted(token_id for token_id, code in token_codes.items() if code_matches_any(code, disease.ranges))


def code_matches_any(code: str, ranges: Sequence[str]) -> bool:
    return any(code_matches_range(code, spec) for spec in ranges)


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


def parse_three_char_code(code: str) -> Optional[Tuple[str, int]]:
    match = re.match(r"^([A-Z])([0-9]{2})", code)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def parse_topk(spec: str) -> List[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    return [value for value in values if value > 0]


def evaluate_loaded_model(
    loaded: LoadedModel,
    diseases: Sequence[DiseaseSpec],
    batch_size: int,
    max_patients: int,
    topk_values: Sequence[int],
    device: str,
) -> Tuple[List[dict], List[dict]]:
    vocab_size = len(loaded.labels)
    diagnosis_tokens = [token for token in loaded.diagnosis_tokens if 0 <= token < vocab_size]
    diagnosis_lookup = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    diagnosis_lookup[torch.tensor(diagnosis_tokens, dtype=torch.long, device=device)] = True
    candidate_tensor = torch.tensor(diagnosis_tokens, dtype=torch.long, device=device)
    disease_tokens = {d.disease_id: token_ids_for_disease(d, loaded.token_codes) for d in diseases}
    disease_lookups = {
        d.disease_id: build_lookup(tokens, vocab_size, device)
        for d in diseases
        for tokens in [disease_tokens[d.disease_id]]
    }
    state = {
        d.disease_id: {
            "scores": [],
            "labels": [],
            "positive_count": 0,
            "rank_sum": 0.0,
            "rank_count": 0,
            "topk_hits": {int(k): 0 for k in topk_values},
        }
        for d in diseases
    }

    num_patients = len(loaded.p2i) if max_patients <= 0 else min(len(loaded.p2i), max_patients)
    with torch.no_grad():
        for start in range(0, num_patients, batch_size):
            stop = min(start + batch_size, num_patients)
            ix = list(range(start, stop))
            logits, targets = forward_batch(loaded, ix, device)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            valid = flat_targets >= 0
            valid &= flat_targets < vocab_size
            target_is_diag = torch.zeros_like(valid, dtype=torch.bool)
            if valid.any():
                target_is_diag[valid] = diagnosis_lookup[flat_targets[valid]]
            if not target_is_diag.any():
                continue

            logits_valid = flat_logits[target_is_diag]
            targets_valid = flat_targets[target_is_diag]
            logits_valid = filter_to_candidates(logits_valid, diagnosis_lookup)
            topk_indices = logits_valid.topk(min(max(topk_values), len(diagnosis_tokens)), dim=-1).indices
            candidate_scores = logits_valid[:, candidate_tensor]

            for disease in diseases:
                tokens = [t for t in disease_tokens[disease.disease_id] if 0 <= t < vocab_size]
                if not tokens:
                    continue
                lookup = disease_lookups[disease.disease_id]
                token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
                group_scores = logits_valid[:, token_tensor].max(dim=1).values
                positives = lookup[targets_valid]
                current = state[disease.disease_id]
                current["scores"].append(group_scores.cpu().numpy().astype(np.float64))
                current["labels"].append(positives.cpu().numpy().astype(np.int8))

                n_pos = int(positives.sum().item())
                if n_pos == 0:
                    continue
                current["positive_count"] += n_pos
                positive_scores = group_scores[positives]
                positive_candidate_scores = candidate_scores[positives]
                ranks = (positive_candidate_scores > positive_scores.unsqueeze(1)).sum(dim=1).float() + 1.0
                current["rank_sum"] += float(ranks.sum().item())
                current["rank_count"] += int(ranks.numel())
                for k in topk_values:
                    hits = lookup[topk_indices[positives, : int(k)]].any(dim=1).sum().item()
                    current["topk_hits"][int(k)] += int(hits)

    metrics_rows = []
    band_rows = []
    for disease in diseases:
        tokens = disease_tokens[disease.disease_id]
        current = state[disease.disease_id]
        if current["scores"]:
            scores = np.concatenate(current["scores"])
            labels = np.concatenate(current["labels"]).astype(np.int8)
        else:
            scores = np.asarray([], dtype=np.float64)
            labels = np.asarray([], dtype=np.int8)
        metrics = summarize_metrics(loaded.spec, disease, tokens, scores, labels, current, topk_values)
        metrics_rows.append(metrics)
        band_rows.extend(build_risk_bands(metrics, scores, labels))
    return metrics_rows, band_rows


def build_lookup(tokens: Sequence[int], vocab_size: int, device: str) -> torch.Tensor:
    lookup = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    valid_tokens = [token for token in tokens if 0 <= token < vocab_size]
    if valid_tokens:
        lookup[torch.tensor(valid_tokens, dtype=torch.long, device=device)] = True
    return lookup


def forward_batch(loaded: LoadedModel, ix: Sequence[int], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if loaded.spec.model_type == "semantic":
        x, a, y, b = get_batch_safe(
            list(ix),
            loaded.data,
            loaded.p2i,
            block_size=loaded.block_size,
            device=device,
            no_event_token_rate=loaded.no_event_token_rate,
            padding="regular",
            cut_batch=True,
        )
        logits, _, _ = loaded.model(x, a, y, b, validation_loss_mode=True)
        return logits, y

    if loaded.static_matrix is None:
        raise RuntimeError("Multitype model requires static_matrix.")
    x, a, y, b, s = get_multitype_batch(
        ix,
        loaded.data,
        loaded.p2i,
        loaded.static_matrix,
        block_size=loaded.block_size,
        device=device,
        no_event_token_rate=loaded.no_event_token_rate,
        padding="regular",
        cut_batch=True,
    )
    logits, _, _ = loaded.model(x, a, s, y, b, validation_loss_mode=True)
    return logits, y


def filter_to_candidates(logits: torch.Tensor, diagnosis_lookup: torch.Tensor) -> torch.Tensor:
    filtered = logits.clone()
    filtered[:, ~diagnosis_lookup] = -torch.inf
    return filtered


def summarize_metrics(
    spec: ModelSpec,
    disease: DiseaseSpec,
    tokens: Sequence[int],
    scores: np.ndarray,
    labels: np.ndarray,
    state: dict,
    topk_values: Sequence[int],
) -> dict:
    positives = int(labels.sum()) if labels.size else 0
    total = int(labels.size)
    negatives = total - positives
    baseline_rate = positives / total if total else float("nan")
    auc_value = binary_auc(scores, labels) if positives > 0 and negatives > 0 else float("nan")
    top_capture, top_rate, lift = top_decile_stats(scores, labels)
    row = {
        "disease_id": disease.disease_id,
        "name": disease.name,
        "name_cn": disease.name_cn,
        "category": disease.category,
        "icd10": ";".join(disease.ranges),
        "model_id": spec.model_id,
        "model": spec.display_name,
        "matched_tokens": len(tokens),
        "prediction_moments": total,
        "positives": positives,
        "negatives": negatives,
        "baseline_event_rate": baseline_rate,
        "auc": auc_value,
        "top_decile_capture": top_capture,
        "top_decile_event_rate": top_rate,
        "top_decile_lift": lift,
        "mean_group_rank": state["rank_sum"] / state["rank_count"] if state["rank_count"] else float("nan"),
    }
    for k in topk_values:
        row[f"group_top{k}"] = state["topk_hits"][int(k)] / positives if positives else float("nan")
    return row


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        avg_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = avg_rank
        start = stop
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    rank_sum_pos = ranks[labels.astype(bool)].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def top_decile_stats(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    if len(scores) == 0 or labels.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    top_n = max(1, int(math.ceil(len(scores) * 0.10)))
    top_labels = labels[order[:top_n]]
    capture = float(top_labels.sum() / labels.sum())
    top_rate = float(top_labels.mean())
    baseline = float(labels.mean())
    lift = top_rate / baseline if baseline > 0 else float("nan")
    return capture, top_rate, lift


def build_risk_bands(base_row: dict, scores: np.ndarray, labels: np.ndarray) -> List[dict]:
    if len(scores) == 0:
        return []
    order = np.argsort(-scores, kind="mergesort")
    bands = [("high_top_10", 0.0, 0.10), ("middle_10_50", 0.10, 0.50), ("low_50_100", 0.50, 1.0)]
    rows = []
    for band, start_q, stop_q in bands:
        start = int(math.floor(len(scores) * start_q))
        stop = int(math.ceil(len(scores) * stop_q))
        idx = order[start:stop]
        if len(idx) == 0:
            continue
        positives = int(labels[idx].sum())
        total = int(len(idx))
        rows.append(
            {
                "disease_id": base_row["disease_id"],
                "name_cn": base_row["name_cn"],
                "model_id": base_row["model_id"],
                "model": base_row["model"],
                "band": band,
                "n": total,
                "positives": positives,
                "event_rate": positives / total,
                "lift_vs_baseline": (positives / total) / base_row["baseline_event_rate"]
                if base_row["baseline_event_rate"] > 0
                else float("nan"),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, diseases: Sequence[DiseaseSpec], metrics_rows: Sequence[dict], topk_values: Sequence[int]) -> None:
    by_key = {(row["disease_id"], row["model_id"]): row for row in metrics_rows}
    lines = [
        "# 有限疾病专题验证",
        "",
        "## 疾病清单",
        "",
        "| 疾病 | ICD-10 | 类别 |",
        "|---|---|---|",
    ]
    for disease in diseases:
        lines.append(f"| {disease.name_cn} ({disease.name}) | {'; '.join(disease.ranges)} | {disease.category} |")

    lines.extend(
        [
            "",
            "## 模型对比",
            "",
            "| 疾病 | Exp2 阳性数 | AUC disease-only | AUC Exp1 | AUC Exp2 | Top10 disease-only | Top10 Exp1 | Top10 Exp2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    top10_key = "group_top10" if 10 in topk_values else f"group_top{max(topk_values)}"
    for disease in diseases:
        d0 = by_key.get((disease.disease_id, "disease_only"), {})
        e1 = by_key.get((disease.disease_id, "exp1"), {})
        e2 = by_key.get((disease.disease_id, "exp2"), {})
        lines.append(
            "| {name} | {pos} | {auc0} | {auc1} | {auc2} | {t0} | {t1} | {t2} |".format(
                name=disease.name_cn,
                pos=_fmt_int(e2.get("positives")),
                auc0=_fmt_float(d0.get("auc")),
                auc1=_fmt_float(e1.get("auc")),
                auc2=_fmt_float(e2.get("auc")),
                t0=_fmt_float(d0.get(top10_key)),
                t1=_fmt_float(e1.get(top10_key)),
                t2=_fmt_float(e2.get(top10_key)),
            )
        )

    lines.extend(
        [
            "",
            "## 结果说明",
            "",
            "- AUC 基于诊断目标时刻计算；阳性表示下一次诊断属于该疾病组。",
            "- Group Top-K 表示模型 Top-K 诊断候选中至少有一个 ICD code 属于该疾病组。",
            "- Top-decile capture 表示疾病组得分最高的 10% 样本覆盖了多少阳性时刻。",
            "- 该结果是回顾性模型验证，不代表临床部署结论。",
            "- 阳性数很少的疾病仅适合作为补充观察，不建议作为主要展示证据。",
            "",
            "## 输出文件",
            "",
            "- `selected_disease_metrics.csv`",
            "- `selected_disease_risk_bands.csv`",
            "- `selected_disease_details.json`",
            "- `selected_disease_auc.svg`",
            "- `selected_disease_top_decile_capture.svg`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_float(value: object) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(x) else f"{x:.4f}"


def _fmt_int(value: object) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def write_bar_svg(path: Path, rows: Sequence[dict], metric: str, title: str) -> None:
    diseases = list(dict.fromkeys(row["name_cn"] for row in rows))
    models = ["disease_only", "exp1", "exp2"]
    model_labels = {"disease_only": "Disease", "exp1": "Exp1", "exp2": "Exp2"}
    colors = {"disease_only": "#64748b", "exp1": "#0f766e", "exp2": "#b45309"}
    by_key = {(row["name_cn"], row["model_id"]): row for row in rows}
    width = max(980, 120 + len(diseases) * 92)
    height = 460
    left, top, bottom = 70, 48, 110
    plot_h = height - top - bottom
    plot_w = width - left - 40
    max_value = max([float(row.get(metric, 0) or 0) for row in rows] + [1.0])
    max_value = min(1.0, max_value * 1.08) if max_value <= 1.0 else max_value * 1.08
    group_w = plot_w / max(len(diseases), 1)
    bar_w = min(18, group_w / 5)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial, sans-serif;font-size:12px;fill:#1f2937}.title{font-size:20px;font-weight:700}.axis{stroke:#94a3b8;stroke-width:1}</style>',
        f'<text x="{left}" y="28" class="title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
    ]
    for tick in np.linspace(0, max_value, 5):
        y = top + plot_h - (tick / max_value) * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>')
    for i, disease in enumerate(diseases):
        cx = left + i * group_w + group_w / 2
        for j, model_id in enumerate(models):
            value = float(by_key.get((disease, model_id), {}).get(metric, 0) or 0)
            h = (value / max_value) * plot_h if max_value else 0
            x = cx + (j - 1) * (bar_w + 3) - bar_w / 2
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[model_id]}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{html.escape(disease[:8])}</text>')
    legend_x = left
    legend_y = height - 32
    for j, model_id in enumerate(models):
        x = legend_x + j * 120
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{colors[model_id]}"/>')
        parts.append(f'<text x="{x + 18}" y="{legend_y}">{model_labels[model_id]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    diseases = parse_selected_diseases(args.diseases_yaml)
    topk_values = parse_topk(args.topk)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[dict] = []
    all_bands: List[dict] = []
    for spec in default_model_specs():
        loaded = load_model(spec, split=args.split, device=args.device)
        metrics, bands = evaluate_loaded_model(
            loaded=loaded,
            diseases=diseases,
            batch_size=args.batch_size,
            max_patients=args.max_patients,
            topk_values=topk_values,
            device=args.device,
        )
        all_metrics.extend(metrics)
        all_bands.extend(bands)

    write_csv(args.output_dir / "selected_disease_metrics.csv", all_metrics)
    write_csv(args.output_dir / "selected_disease_risk_bands.csv", all_bands)
    (args.output_dir / "selected_disease_details.json").write_text(
        json.dumps({"diseases": [d.__dict__ for d in diseases], "metrics": all_metrics, "risk_bands": all_bands}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_md(args.output_dir / "selected_disease_summary.md", diseases, all_metrics, topk_values)
    write_bar_svg(args.output_dir / "selected_disease_auc.svg", all_metrics, "auc", "有限疾病 AUC 对比")
    write_bar_svg(
        args.output_dir / "selected_disease_top_decile_capture.svg",
        all_metrics,
        "top_decile_capture",
        "最高 10% 得分覆盖率",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(all_metrics)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
