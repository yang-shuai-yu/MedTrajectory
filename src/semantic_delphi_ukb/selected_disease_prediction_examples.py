from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from semantic_delphi_ukb.evaluate_test import get_batch_safe
from semantic_delphi_ukb.multitype_batch import get_batch as get_multitype_batch
from semantic_delphi_ukb.selected_disease_demo import (
    DiseaseSpec,
    LoadedModel,
    default_model_specs,
    filter_to_candidates,
    load_model,
    parse_selected_diseases,
    token_ids_for_disease,
)
from semantic_delphi_ukb.modern_selected_utils import load_modern_model, modern_model_spec
from semantic_delphi_ukb.tte_selected_utils import load_tte_model, tte_model_spec


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create selected-disease prediction-vs-label examples for presentation."
    )
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "selected_diseases.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "tests" / "output" / "selected_disease_demo")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--models", type=str, default="exp2")
    parser.add_argument("--samples-per-disease", type=int, default=3)
    parser.add_argument("--misses-per-disease", type=int, default=1)
    parser.add_argument("--markdown-model", type=str, default="exp2")
    return parser


def selected_specs(model_ids: Sequence[str]):
    wanted = set(model_ids)
    available = default_model_specs() + [modern_model_spec(), tte_model_spec()]
    specs = [spec for spec in available if spec.model_id in wanted]
    missing = wanted - {spec.model_id for spec in specs}
    if missing:
        raise ValueError(f"Unknown model id(s): {', '.join(sorted(missing))}")
    return specs


def load_selected_model(spec, split: str, device: str) -> LoadedModel:
    if spec.model_id == "modern":
        return load_modern_model(split=split, device=device)
    if spec.model_id == "tte_multitask":
        return load_tte_model(split=split, device=device)
    return load_model(spec, split=split, device=device)


def parse_model_ids(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["exp2"]


def forward_batch_with_context(
    loaded: LoadedModel,
    ix: Sequence[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return logits, y, a, b

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
    return logits, y, a, b


def collect_model_examples(
    loaded: LoadedModel,
    diseases: Sequence[DiseaseSpec],
    batch_size: int,
    max_patients: int,
    samples_per_disease: int,
    misses_per_disease: int,
    device: str,
) -> List[dict]:
    vocab_size = len(loaded.labels)
    diagnosis_tokens = [token for token in loaded.diagnosis_tokens if 0 <= token < vocab_size]
    diagnosis_lookup = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    diagnosis_lookup[torch.tensor(diagnosis_tokens, dtype=torch.long, device=device)] = True
    candidate_tensor = torch.tensor(diagnosis_tokens, dtype=torch.long, device=device)

    disease_tokens = {
        disease.disease_id: [token for token in token_ids_for_disease(disease, loaded.token_codes) if 0 <= token < vocab_size]
        for disease in diseases
    }
    valid_diseases = [disease for disease in diseases if disease_tokens[disease.disease_id]]
    if not valid_diseases:
        return []

    disease_token_sets = {
        disease.disease_id: set(disease_tokens[disease.disease_id])
        for disease in valid_diseases
    }
    disease_token_tensors = {
        disease.disease_id: torch.tensor(disease_tokens[disease.disease_id], dtype=torch.long, device=device)
        for disease in valid_diseases
    }
    name_by_id = {disease.disease_id: disease for disease in valid_diseases}

    buckets: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    num_patients = len(loaded.p2i) if max_patients <= 0 else min(len(loaded.p2i), max_patients)

    loaded.model.eval()
    with torch.no_grad():
        for start in range(0, num_patients, batch_size):
            stop = min(start + batch_size, num_patients)
            ix = list(range(start, stop))
            logits, targets, pred_ages, target_ages = forward_batch_with_context(loaded, ix, device)
            batch_size_actual, seq_len = targets.shape

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
            filtered_logits = filter_to_candidates(logits_valid, diagnosis_lookup)
            diag_scores = logits_valid[:, candidate_tensor]

            disease_scores = []
            for disease in valid_diseases:
                token_tensor = disease_token_tensors[disease.disease_id]
                disease_scores.append(logits_valid[:, token_tensor].max(dim=1).values)
            disease_scores_tensor = torch.stack(disease_scores, dim=1)
            top_group_scores, top_group_indices = disease_scores_tensor.topk(
                min(5, len(valid_diseases)),
                dim=1,
            )
            top_diag_tokens = filtered_logits.topk(min(5, len(diagnosis_tokens)), dim=1).indices

            valid_mask_np = target_is_diag.cpu().numpy()
            patient_indices = np.repeat(np.asarray(ix, dtype=np.int64), seq_len)[valid_mask_np]
            pred_age_values = pred_ages.reshape(-1)[target_is_diag].cpu().numpy() / 365.25
            target_age_values = target_ages.reshape(-1)[target_is_diag].cpu().numpy() / 365.25
            targets_np = targets_valid.cpu().numpy().astype(int)
            disease_scores_np = disease_scores_tensor.cpu().numpy()
            top_group_scores_np = top_group_scores.cpu().numpy()
            top_group_indices_np = top_group_indices.cpu().numpy()
            top_diag_tokens_np = top_diag_tokens.cpu().numpy().astype(int)
            diag_scores_np = diag_scores.cpu().numpy()

            for row_idx, target_token in enumerate(targets_np):
                actual_disease_ids = [
                    disease.disease_id
                    for disease in valid_diseases
                    if int(target_token) in disease_token_sets[disease.disease_id]
                ]
                top_disease_ids = [
                    valid_diseases[int(idx)].disease_id
                    for idx in top_group_indices_np[row_idx].tolist()
                ]
                expected_disease_id = top_disease_ids[0]
                expected_score = float(top_group_scores_np[row_idx, 0])
                common_fields = {
                    "model_id": loaded.spec.model_id,
                    "model": loaded.spec.display_name,
                    "split_patient_index": int(patient_indices[row_idx]),
                    "prediction_age_years": float(pred_age_values[row_idx]),
                    "target_age_years": float(target_age_values[row_idx]),
                    "expected_disease_id": expected_disease_id,
                    "expected_disease": disease_name(name_by_id[expected_disease_id]),
                    "expected_score": expected_score,
                    "top5_selected_diseases": " | ".join(disease_name(name_by_id[disease_id]) for disease_id in top_disease_ids),
                    "actual_selected_diseases": " | ".join(
                        disease_name(name_by_id[disease_id]) for disease_id in actual_disease_ids
                    )
                    if actual_disease_ids
                    else "Other diagnosis",
                    "actual_code": loaded.token_codes.get(int(target_token), ""),
                    "actual_label": label_for_token(loaded, int(target_token)),
                    "top5_diagnosis_labels": " | ".join(
                        compact_label(loaded, int(token)) for token in top_diag_tokens_np[row_idx].tolist()
                    ),
                }

                expected_rank = rank_against_diagnoses(diag_scores_np[row_idx], expected_score)
                top_row = {
                    **common_fields,
                    "row_type": "top_selected_prediction",
                    "display_disease_id": expected_disease_id,
                    "display_disease": disease_name(name_by_id[expected_disease_id]),
                    "display_icd10": ";".join(name_by_id[expected_disease_id].ranges),
                    "target_disease_score": expected_score,
                    "target_disease_rank_among_diagnoses": expected_rank,
                    "actual_is_display_disease": int(expected_disease_id in actual_disease_ids),
                    "top1_hit_for_display_disease": int(expected_disease_id in actual_disease_ids),
                    "top5_hit_for_display_disease": int(expected_disease_id in actual_disease_ids),
                }
                buckets[(loaded.spec.model_id, expected_disease_id, "top_selected_prediction")].append(top_row)

                for disease_id in actual_disease_ids:
                    disease_idx = next(i for i, disease in enumerate(valid_diseases) if disease.disease_id == disease_id)
                    score = float(disease_scores_np[row_idx, disease_idx])
                    rank = rank_against_diagnoses(diag_scores_np[row_idx], score)
                    hit_top1 = int(top_disease_ids[0] == disease_id)
                    hit_top5 = int(disease_id in top_disease_ids)
                    positive_row = {
                        **common_fields,
                        "row_type": "actual_positive_case",
                        "display_disease_id": disease_id,
                        "display_disease": disease_name(name_by_id[disease_id]),
                        "display_icd10": ";".join(name_by_id[disease_id].ranges),
                        "target_disease_score": score,
                        "target_disease_rank_among_diagnoses": rank,
                        "actual_is_display_disease": 1,
                        "top1_hit_for_display_disease": hit_top1,
                        "top5_hit_for_display_disease": hit_top5,
                    }
                    buckets[(loaded.spec.model_id, disease_id, "actual_positive_case")].append(positive_row)
                    if not hit_top5:
                        miss_row = dict(positive_row)
                        miss_row["row_type"] = "actual_positive_low_score_miss"
                        buckets[(loaded.spec.model_id, disease_id, "actual_positive_low_score_miss")].append(miss_row)

    selected_rows: List[dict] = []
    for disease in valid_diseases:
        for row_type, limit, reverse in [
            ("top_selected_prediction", samples_per_disease, True),
            ("actual_positive_case", samples_per_disease, True),
            ("actual_positive_low_score_miss", misses_per_disease, False),
        ]:
            rows = buckets.get((loaded.spec.model_id, disease.disease_id, row_type), [])
            rows = sorted(rows, key=lambda row: float(row["target_disease_score"]), reverse=reverse)
            selected_rows.extend(rows[: max(0, limit)])

    for idx, row in enumerate(selected_rows, start=1):
        row["case_id"] = f"{loaded.spec.model_id}_{idx:04d}"
    return selected_rows


def rank_against_diagnoses(diagnosis_scores: np.ndarray, score: float) -> int:
    return int((diagnosis_scores > score).sum() + 1)


def label_for_token(loaded: LoadedModel, token_id: int) -> str:
    if 0 <= token_id < len(loaded.labels):
        return loaded.labels[token_id].replace("\n", " ").strip()
    return str(token_id)


def compact_label(loaded: LoadedModel, token_id: int) -> str:
    code = loaded.token_codes.get(token_id, "")
    label = label_for_token(loaded, token_id)
    if code and code not in label:
        return f"{code} {label}"
    return label


def disease_name(disease: DiseaseSpec) -> str:
    return f"{disease.name_cn} ({disease.name})"


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "case_id",
        "row_type",
        "model_id",
        "model",
        "display_disease_id",
        "display_disease",
        "display_icd10",
        "split_patient_index",
        "prediction_age_years",
        "target_age_years",
        "expected_disease_id",
        "expected_disease",
        "expected_score",
        "target_disease_score",
        "target_disease_rank_among_diagnoses",
        "actual_is_display_disease",
        "top1_hit_for_display_disease",
        "top5_hit_for_display_disease",
        "actual_selected_diseases",
        "actual_code",
        "actual_label",
        "top5_selected_diseases",
        "top5_diagnosis_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[dict], focus_model: str) -> None:
    focus_rows = [row for row in rows if row["model_id"] == focus_model]
    lines = [
        "# 有限疾病模型预期与实际 label 对比",
        "",
        "## 口径",
        "",
        "- 模型预期：在同一个历史预测时点，selected diseases 中模型打分最高的疾病组。",
        "- 实际 label：该位置真实的下一个 diagnosis token；如果属于 selected diseases，会显示对应疾病组，否则显示 Other diagnosis。",
        "- 分数：目标疾病组内 ICD-10 token 的最大 raw logit，只用于同模型内排序展示，不是临床概率。",
        "- 年龄：prediction age 是模型打分时点，target age 是真实 next diagnosis 发生年龄。",
        "",
        f"## 展示样本（{focus_model}）",
        "",
        "| 疾病 | 场景 | 预测年龄 | 目标年龄 | 模型预期 | 实际 label | 分数 | 诊断内排名 | Top-5 selected | 命中 |",
        "|---|---|---:|---:|---|---|---:|---:|---|---:|",
    ]
    for row in sorted(focus_rows, key=lambda item: (item["display_disease_id"], item["row_type"], item["case_id"])):
        lines.append(
            "| {disease} | {row_type} | {pred_age} | {target_age} | {expected} | {actual} | {score} | {rank} | {top5} | {hit} |".format(
                disease=escape_pipe(str(row["display_disease"])),
                row_type=display_row_type(str(row["row_type"])),
                pred_age=fmt_float(row["prediction_age_years"]),
                target_age=fmt_float(row["target_age_years"]),
                expected=escape_pipe(str(row["expected_disease"])),
                actual=escape_pipe(f'{row["actual_selected_diseases"]}; {row["actual_label"]}'),
                score=fmt_float(row["target_disease_score"]),
                rank=row["target_disease_rank_among_diagnoses"],
                top5=escape_pipe(str(row["top5_selected_diseases"])),
                hit=row["top5_hit_for_display_disease"],
            )
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `selected_disease_prediction_comparison.csv`：完整样本表，可筛选模型、疾病和场景。",
            "- `selected_disease_prediction_comparison.md`：面向汇报的简表。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_row_type(row_type: str) -> str:
    mapping = {
        "top_selected_prediction": "模型高分预测",
        "actual_positive_case": "实际阳性样本",
        "actual_positive_low_score_miss": "未命中样本",
    }
    return mapping.get(row_type, row_type)


def escape_pipe(value: str) -> str:
    return value.replace("|", "\\|")


def fmt_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.3f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    diseases = parse_selected_diseases(args.diseases_yaml)
    model_ids = parse_model_ids(args.models)
    rows: List[dict] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in selected_specs(model_ids):
        loaded = load_selected_model(spec, split=args.split, device=args.device)
        rows.extend(
            collect_model_examples(
                loaded=loaded,
                diseases=diseases,
                batch_size=args.batch_size,
                max_patients=args.max_patients,
                samples_per_disease=args.samples_per_disease,
                misses_per_disease=args.misses_per_disease,
                device=args.device,
            )
        )

    csv_path = args.output_dir / "selected_disease_prediction_comparison.csv"
    md_path = args.output_dir / "selected_disease_prediction_comparison.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, args.markdown_model)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "models": model_ids,
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
