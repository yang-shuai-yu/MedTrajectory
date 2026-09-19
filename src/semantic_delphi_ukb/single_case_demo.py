from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from semantic_delphi_ukb.evaluate_test import get_batch_safe
from semantic_delphi_ukb.multitype_batch import get_batch as get_multitype_batch
from semantic_delphi_ukb.selected_disease_demo import (
    DiseaseSpec,
    LoadedModel,
    build_lookup,
    default_model_specs,
    filter_to_candidates,
    load_model,
    parse_selected_diseases,
    token_ids_for_disease,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
MASK_TIME = -10000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible single-case selected disease demo.")
    parser.add_argument("--case-id", type=str, default="AF-001")
    parser.add_argument("--model-id", type=str, default="exp2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split-patient-index", type=int, default=749)
    parser.add_argument("--disease-id", type=str, default="atrial_fibrillation")
    parser.add_argument("--target-token", type=str, default="diag:I48")
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "selected_diseases.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "tests" / "output" / "single_case_demo")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--include-no-event", action="store_true")
    return parser


def select_model_spec(model_id: str):
    for spec in default_model_specs():
        if spec.model_id == model_id:
            return spec
    raise ValueError(f"Unknown model_id={model_id!r}")


def forward_one(loaded: LoadedModel, row_index: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ix = [int(row_index)]
    if loaded.spec.model_type == "semantic":
        x, a, y, b = get_batch_safe(
            ix,
            loaded.data,
            loaded.p2i,
            block_size=loaded.block_size,
            device=device,
            no_event_token_rate=loaded.no_event_token_rate,
            padding="regular",
            cut_batch=True,
        )
        logits, _, _ = loaded.model(x, a, y, b, validation_loss_mode=True)
        return logits, x, a, y, b

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
    return logits, x, a, y, b


def token_label(loaded: LoadedModel, token_id: int) -> str:
    if 0 <= token_id < len(loaded.labels):
        return loaded.labels[token_id]
    return str(token_id)


def find_prediction_position(
    loaded: LoadedModel,
    targets: torch.Tensor,
    target_ages: torch.Tensor,
    target_token: str,
) -> int:
    labels = [token_label(loaded, int(token_id)) for token_id in targets[0].detach().cpu().tolist()]
    valid_positions = [
        idx
        for idx, label in enumerate(labels)
        if label == target_token and float(target_ages[0, idx].detach().cpu().item()) > MASK_TIME / 2
    ]
    if not valid_positions:
        available = sorted({label for label in labels if label not in {"Padding", ""}})[:20]
        raise RuntimeError(f"Target token {target_token!r} not found in batch targets. Example labels: {available}")
    return valid_positions[0]


def visible_history(
    loaded: LoadedModel,
    inputs: torch.Tensor,
    input_ages: torch.Tensor,
    position: int,
    include_no_event: bool,
) -> List[dict]:
    rows = []
    for idx in range(position + 1):
        token_id = int(inputs[0, idx].detach().cpu().item())
        age_days = float(input_ages[0, idx].detach().cpu().item())
        if token_id <= 0 or age_days <= MASK_TIME / 2:
            continue
        label = token_label(loaded, token_id)
        if not include_no_event and label == "No event":
            continue
        rows.append(
            {
                "age_years": age_days / 365.25,
                "token": label,
                "event_type": label.split(":", 1)[0] if ":" in label else label,
            }
        )
    return rows


def top_diagnoses(loaded: LoadedModel, logits_at_position: torch.Tensor, topn: int, device: str) -> List[dict]:
    diagnosis_tokens = [token for token in loaded.diagnosis_tokens if 0 <= token < len(loaded.labels)]
    diagnosis_lookup = torch.zeros(len(loaded.labels), dtype=torch.bool, device=device)
    diagnosis_lookup[torch.tensor(diagnosis_tokens, dtype=torch.long, device=device)] = True
    filtered = filter_to_candidates(logits_at_position.unsqueeze(0), diagnosis_lookup)[0]
    top = filtered.topk(min(int(topn), len(diagnosis_tokens))).indices.detach().cpu().tolist()
    return [
        {
            "rank": rank,
            "token": token_label(loaded, int(token_id)),
            "score": float(logits_at_position[int(token_id)].detach().cpu().item()),
        }
        for rank, token_id in enumerate(top, start=1)
    ]


def top_selected_diseases(
    loaded: LoadedModel,
    diseases: Sequence[DiseaseSpec],
    logits_at_position: torch.Tensor,
    topn: int,
    device: str,
) -> List[dict]:
    rows = []
    for disease in diseases:
        tokens = [token for token in token_ids_for_disease(disease, loaded.token_codes) if 0 <= token < len(loaded.labels)]
        if not tokens:
            continue
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        score = float(logits_at_position[token_tensor].max().detach().cpu().item())
        rows.append(
            {
                "disease_id": disease.disease_id,
                "disease": disease_name(disease),
                "icd10": ";".join(disease.ranges),
                "score": score,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows[: int(topn)], start=1):
        row["rank"] = rank
    return rows[: int(topn)]


def actual_selected_diseases(loaded: LoadedModel, diseases: Sequence[DiseaseSpec], target_token_id: int, device: str) -> List[str]:
    out = []
    for disease in diseases:
        tokens = [token for token in token_ids_for_disease(disease, loaded.token_codes) if 0 <= token < len(loaded.labels)]
        lookup = build_lookup(tokens, len(loaded.labels), device)
        if bool(lookup[target_token_id].detach().cpu().item()):
            out.append(disease_name(disease))
    return out


def disease_name(disease: DiseaseSpec) -> str:
    return f"{disease.name_cn} ({disease.name})"


def read_patient_index(data_dir: Path, split: str, row_index: int) -> dict:
    path = data_dir / f"{split}_patient_index.csv"
    if not path.exists():
        return {"row_index": row_index, "eid": "", "num_events": ""}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["row_index"]) == row_index:
                return row
    return {"row_index": row_index, "eid": "", "num_events": ""}


def fmt(value: float) -> str:
    return "" if math.isnan(float(value)) else f"{float(value):.3f}"


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_timeline_svg(path: Path, payload: dict) -> None:
    history = payload["history"]
    ages = [float(row["age_years"]) for row in history] + [float(payload["target_age_years"])]
    min_age = math.floor(min(ages)) - 1
    max_age = math.ceil(max(ages)) + 1
    if max_age <= min_age:
        max_age = min_age + 1

    width, height = 1120, 360
    left, right = 90, 70
    axis_y = 235
    plot_w = width - left - right

    def x_for(age: float) -> float:
        return left + (float(age) - min_age) / (max_age - min_age) * plot_w

    colors = {
        "diag": "#2563eb",
        "proc": "#d97706",
        "cancer": "#be123c",
        "death": "#525252",
        "target": "#dc2626",
        "prediction": "#0891b2",
    }
    lane_y = {"diag": 135, "proc": 188, "cancer": 82, "death": 82}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        (
            "<style>"
            "text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#172033;font-size:14px}"
            ".title{font-size:24px;font-weight:700}.sub{fill:#596275;font-size:13px}"
            ".label{font-size:12px}.axis{stroke:#8b95a7;stroke-width:2}.grid{stroke:#e5e7eb;stroke-width:1}"
            "</style>"
        ),
        f'<text x="{left}" y="38" class="title">Demo Case {escape(payload["case_id"])}: timeline</text>',
        f'<text x="{left}" y="62" class="sub">Only events before the prediction point are visible to the model.</text>',
        f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" class="axis"/>',
    ]

    for tick in range(min_age, max_age + 1, max(1, (max_age - min_age) // 6)):
        x = x_for(tick)
        parts.append(f'<line x1="{x:.1f}" y1="82" x2="{x:.1f}" y2="{axis_y + 8}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{axis_y + 30}" text-anchor="middle" class="sub">{tick}</text>')
    parts.append(f'<text x="{width - right}" y="{axis_y + 54}" text-anchor="end" class="sub">Age (years)</text>')

    for row in history:
        age = float(row["age_years"])
        token = str(row["token"])
        event_type = str(row["event_type"])
        x = x_for(age)
        y = lane_y.get(event_type, 160)
        fill = colors.get(event_type, "#64748b")
        is_prediction = abs(age - float(payload["prediction_age_years"])) < 1e-3
        radius = 10 if is_prediction else 7
        stroke = colors["prediction"] if is_prediction else "#ffffff"
        stroke_w = 4 if is_prediction else 2
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{y + radius}" x2="{x:.1f}" y2="{axis_y}" stroke="{fill}" stroke-width="1.5" stroke-dasharray="3 3"/>')
        label = f'{age:.3f}y {token}'
        anchor = "middle"
        dy = -18 if event_type != "proc" else 28
        parts.append(f'<text x="{x:.1f}" y="{y + dy}" text-anchor="{anchor}" class="label">{escape(label)}</text>')
        if is_prediction:
            parts.append(f'<text x="{x:.1f}" y="{y - 36}" text-anchor="middle" class="label" fill="{colors["prediction"]}">prediction point</text>')

    target_x = x_for(float(payload["target_age_years"]))
    target_y = 292
    parts.append(f'<polygon points="{target_x:.1f},{target_y - 13} {target_x + 12:.1f},{target_y + 10} {target_x - 12:.1f},{target_y + 10}" fill="{colors["target"]}"/>')
    parts.append(f'<line x1="{target_x:.1f}" y1="{target_y - 13}" x2="{target_x:.1f}" y2="{axis_y}" stroke="{colors["target"]}" stroke-width="1.5" stroke-dasharray="3 3"/>')
    parts.append(
        f'<text x="{target_x:.1f}" y="{target_y + 34}" text-anchor="middle" class="label">'
        f'{float(payload["target_age_years"]):.3f}y {escape(payload["target_token"])} (actual next label)</text>'
    )

    legend_x, legend_y = left, 325
    legend = [("diagnosis", colors["diag"]), ("procedure", colors["proc"]), ("actual next label", colors["target"])]
    for i, (name, color) in enumerate(legend):
        x = legend_x + i * 160
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{x + 14}" y="{legend_y + 5}" class="sub">{escape(name)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bar_svg(path: Path, rows: Sequence[dict], title: str, label_key: str) -> None:
    width = 1120
    row_h = 58
    top = 92
    left = 270
    height = top + max(1, len(rows)) * row_h + 48
    scores = [float(row["score"]) for row in rows]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    span = max(max_score - min_score, 1e-6)
    bar_max = width - left - 130

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        (
            "<style>"
            "text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#172033;font-size:14px}"
            ".title{font-size:24px;font-weight:700}.sub{fill:#596275;font-size:13px}"
            ".rank{font-size:18px;font-weight:700;fill:#0f172a}.label{font-size:14px}"
            "</style>"
        ),
        f'<text x="70" y="38" class="title">{escape(title)}</text>',
        '<text x="70" y="62" class="sub">Bars show relative score within the top candidates; raw logits are shown on the right.</text>',
    ]

    for i, row in enumerate(rows):
        y = top + i * row_h
        score = float(row["score"])
        frac = (score - min_score) / span if span else 1.0
        bar_w = max(20.0, 80.0 + frac * (bar_max - 80.0))
        fill = "#dc2626" if i == 0 else "#2563eb"
        label = str(row[label_key])
        parts.append(f'<text x="70" y="{y + 23}" class="rank">#{int(row["rank"])}</text>')
        parts.append(f'<text x="118" y="{y + 23}" class="label">{escape(label[:46])}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="32" rx="8" fill="{fill}" opacity="0.88"/>')
        parts.append(f'<text x="{left + bar_w + 14:.1f}" y="{y + 22}" class="sub">logit {score:.3f}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_visuals(output_dir: Path, case_id: str, payload: dict) -> Dict[str, str]:
    paths = {
        "timeline_svg": output_dir / f"{case_id}_timeline.svg",
        "top_diagnoses_svg": output_dir / f"{case_id}_top_diagnoses.svg",
        "top_selected_diseases_svg": output_dir / f"{case_id}_top_selected_diseases.svg",
    }
    write_timeline_svg(paths["timeline_svg"], payload)
    write_bar_svg(paths["top_diagnoses_svg"], payload["top_diagnoses"], "Top-5 Diagnosis Candidates", "token")
    write_bar_svg(paths["top_selected_diseases_svg"], payload["top_selected_diseases"], "Top-5 Selected Disease Groups", "disease")
    return {key: str(path) for key, path in paths.items()}


def append_visual_links(path: Path, payload: dict) -> None:
    lines = [
        "",
        "## 可视化图表",
        "",
        f"![Timeline]({Path(payload['visuals']['timeline_svg']).name})",
        "",
        f"![Top-5 Diagnosis]({Path(payload['visuals']['top_diagnoses_svg']).name})",
        "",
        f"![Top-5 Selected Diseases]({Path(payload['visuals']['top_selected_diseases_svg']).name})",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        f"# Single Case Demo: {payload['case_id']}",
        "",
        "## 任务口径",
        "",
        "该 demo 重新加载训练好的模型并对单个测试集病例运行推理。模型在历史预测点上预测下一条诊断事件，不是固定 5 年风险预测。",
        "",
        "## 病例概览",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 模型 | {payload['model']} |",
        f"| 疾病组 | {payload['disease']} |",
        f"| 预测年龄 | {fmt(payload['prediction_age_years'])} 岁 |",
        f"| 真实下一诊断年龄 | {fmt(payload['target_age_years'])} 岁 |",
        f"| 时间间隔 | {fmt(payload['gap_days'])} 天 |",
        f"| 真实 label | `{payload['target_token']}` |",
        f"| Top-1 诊断候选 | `{payload['top_diagnoses'][0]['token']}` |",
        f"| 是否 Top-1 命中 | {'是' if payload['top1_hit'] else '否'} |",
        "",
        "## 历史输入轨迹",
        "",
        "| 年龄 | 输入 token | 事件类型 |",
        "|---:|---|---|",
    ]
    for row in payload["history"]:
        lines.append(f"| {fmt(row['age_years'])} | `{row['token']}` | {row['event_type']} |")

    lines.extend(
        [
            "",
            "## 模型 Top-5 诊断候选",
            "",
            "| 排名 | 诊断 token | raw logit |",
            "|---:|---|---:|",
        ]
    )
    for row in payload["top_diagnoses"]:
        lines.append(f"| {row['rank']} | `{row['token']}` | {fmt(row['score'])} |")

    lines.extend(
        [
            "",
            "## 模型 Top-5 Selected Disease 候选",
            "",
            "| 排名 | 疾病组 | ICD-10 | raw logit |",
            "|---:|---|---|---:|",
        ]
    )
    for row in payload["top_selected_diseases"]:
        lines.append(f"| {row['rank']} | {row['disease']} | `{row['icd10']}` | {fmt(row['score'])} |")

    lines.extend(
        [
            "",
            "## 汇报句式",
            "",
            (
                f"在 {fmt(payload['prediction_age_years'])} 岁的历史预测点上，模型只能看到此前的医疗事件；"
                f"模型将 `{payload['top_diagnoses'][0]['token']}` 排为下一诊断 Top-1，"
                f"真实下一诊断在约 {fmt(payload['gap_days'])} 天后发生，为 `{payload['target_token']}`。"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    disease_specs = parse_selected_diseases(args.diseases_yaml)
    disease_by_id: Dict[str, DiseaseSpec] = {disease.disease_id: disease for disease in disease_specs}
    if args.disease_id not in disease_by_id:
        raise ValueError(f"Unknown disease_id={args.disease_id!r}")

    loaded = load_model(select_model_spec(args.model_id), split=args.split, device=args.device)
    with torch.no_grad():
        logits, inputs, input_ages, targets, target_ages = forward_one(loaded, args.split_patient_index, args.device)

    position = find_prediction_position(loaded, targets, target_ages, args.target_token)
    logits_at_position = logits[0, position]
    target_token_id = int(targets[0, position].detach().cpu().item())
    prediction_age_years = float(input_ages[0, position].detach().cpu().item()) / 365.25
    target_age_years = float(target_ages[0, position].detach().cpu().item()) / 365.25
    top_diag = top_diagnoses(loaded, logits_at_position, args.topn, args.device)
    top_diseases = top_selected_diseases(loaded, disease_specs, logits_at_position, args.topn, args.device)
    actual_groups = actual_selected_diseases(loaded, disease_specs, target_token_id, args.device)
    patient_row = read_patient_index(loaded.spec.data_dir, args.split, args.split_patient_index)

    payload = {
        "case_id": args.case_id,
        "model_id": loaded.spec.model_id,
        "model": loaded.spec.display_name,
        "split": args.split,
        "split_patient_index": int(args.split_patient_index),
        "eid_redacted": bool(patient_row.get("eid")),
        "num_events": patient_row.get("num_events", ""),
        "disease_id": args.disease_id,
        "disease": disease_name(disease_by_id[args.disease_id]),
        "target_token": token_label(loaded, target_token_id),
        "actual_selected_diseases": actual_groups,
        "prediction_age_years": prediction_age_years,
        "target_age_years": target_age_years,
        "gap_days": (target_age_years - prediction_age_years) * 365.25,
        "target_position": int(position),
        "history": visible_history(loaded, inputs, input_ages, position, args.include_no_event),
        "top_diagnoses": top_diag,
        "top_selected_diseases": top_diseases,
        "top1_hit": bool(top_diag and top_diag[0]["token"] == token_label(loaded, target_token_id)),
    }
    payload["visuals"] = write_visuals(args.output_dir, args.case_id, payload)

    json_path = args.output_dir / f"{args.case_id}_demo.json"
    md_path = args.output_dir / f"{args.case_id}_demo.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    append_visual_links(md_path, payload)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "visuals": payload["visuals"],
                "top1_hit": payload["top1_hit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
