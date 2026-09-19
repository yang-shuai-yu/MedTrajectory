from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from semantic_delphi_ukb.selected_disease_demo import (
    DiseaseSpec,
    LoadedModel,
    default_model_specs,
    load_model,
    parse_selected_diseases,
    token_ids_for_disease,
)
from semantic_delphi_ukb.single_case_demo import (
    find_prediction_position,
    forward_one,
    read_patient_index,
    token_label,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
MASK_TIME = -10000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single-patient future-trajectory demo for Exp2.")
    parser.add_argument("--case-id", type=str, default="AF-FUTURE-001")
    parser.add_argument("--model-id", type=str, default="exp2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split-patient-index", type=int, default=2028)
    parser.add_argument("--target-token", type=str, default="diag:I48")
    parser.add_argument("--disease-id", type=str, default="atrial_fibrillation")
    parser.add_argument("--diseases-yaml", type=Path, default=REPO_DIR / "selected_diseases.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "tests" / "output" / "patient_future_demo")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topn", type=int, default=10)
    return parser


def select_model_spec(model_id: str):
    for spec in default_model_specs():
        if spec.model_id == model_id:
            return spec
    raise ValueError(f"Unknown model_id={model_id!r}")


def event_type(token: str) -> str:
    return token.split(":", 1)[0] if ":" in token else token


def event_type_name(value: str) -> str:
    return {
        "diag": "诊断",
        "proc": "手术/操作",
        "cancer": "癌症登记",
        "death": "死亡登记",
    }.get(value, value)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def disease_display(disease: DiseaseSpec) -> str:
    return f"{disease.name} ({';'.join(disease.ranges)})"


def token_ids_by_prefix(loaded: LoadedModel, prefix: str) -> List[int]:
    if prefix == "diag":
        return [token for token in loaded.diagnosis_tokens if 0 <= token < len(loaded.labels)]
    marker = f"{prefix}:"
    return [idx for idx, label in enumerate(loaded.labels) if label.startswith(marker)]


def rank_token(logits_at_position: torch.Tensor, token_id: int, candidate_ids: Sequence[int]) -> Tuple[Optional[int], float]:
    score = float(logits_at_position[int(token_id)].detach().cpu().item())
    valid = [int(token) for token in candidate_ids if 0 <= int(token) < logits_at_position.numel()]
    if not valid or int(token_id) not in valid:
        return None, score
    candidate_tensor = torch.tensor(valid, dtype=torch.long, device=logits_at_position.device)
    rank = int((logits_at_position[candidate_tensor] > score).sum().detach().cpu().item() + 1)
    return rank, score


def top_tokens(
    loaded: LoadedModel,
    logits_at_position: torch.Tensor,
    prefix: str,
    topn: int,
) -> List[dict]:
    candidates = token_ids_by_prefix(loaded, prefix)
    if not candidates:
        return []
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits_at_position.device)
    scores = logits_at_position[candidate_tensor]
    top = scores.topk(min(int(topn), len(candidates))).indices.detach().cpu().tolist()
    rows = []
    for rank, candidate_offset in enumerate(top, start=1):
        token_id = int(candidates[int(candidate_offset)])
        rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": token_label(loaded, token_id),
                "event_type": prefix,
                "score": float(logits_at_position[token_id].detach().cpu().item()),
            }
        )
    return rows


def patient_events(loaded: LoadedModel, row_index: int) -> List[dict]:
    start, length = loaded.p2i[int(row_index)]
    rows = loaded.data[int(start) : int(start) + int(length)]
    events = []
    for _, age_days, raw_token in rows:
        token_id = int(raw_token) + 1
        token = token_label(loaded, token_id)
        events.append(
            {
                "age_days": int(age_days),
                "age_years": float(age_days) / 365.25,
                "token_id": token_id,
                "token": token,
                "event_type": event_type(token),
                "event_type_name": event_type_name(event_type(token)),
            }
        )
    return events


def first_matching_event(events: Sequence[dict], token_ids: Sequence[int]) -> Optional[dict]:
    wanted = {int(token_id) for token_id in token_ids}
    for event in events:
        if int(event["token_id"]) in wanted:
            return event
    return None


def annotate_exact_future_matches(rows: Sequence[dict], future_events: Sequence[dict], baseline_age: float) -> List[dict]:
    first_by_token: Dict[str, dict] = {}
    for event in future_events:
        first_by_token.setdefault(str(event["token"]), event)

    out = []
    for row in rows:
        matched = first_by_token.get(str(row["token"]))
        enriched = dict(row)
        enriched["appeared_in_future"] = matched is not None
        enriched["first_future_age_years"] = float(matched["age_years"]) if matched else None
        enriched["years_after_baseline"] = (float(matched["age_years"]) - baseline_age) if matched else None
        out.append(enriched)
    return out


def selected_disease_scores(
    loaded: LoadedModel,
    diseases: Sequence[DiseaseSpec],
    logits_at_position: torch.Tensor,
    topn: int,
) -> List[dict]:
    diagnosis_ids = token_ids_by_prefix(loaded, "diag")
    rows = []
    for disease in diseases:
        tokens = [token for token in token_ids_for_disease(disease, loaded.token_codes) if 0 <= token < len(loaded.labels)]
        if not tokens:
            continue
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=logits_at_position.device)
        scores = logits_at_position[token_tensor]
        best_offset = int(scores.argmax().detach().cpu().item())
        best_token = int(tokens[best_offset])
        rank, score = rank_token(logits_at_position, best_token, diagnosis_ids)
        rows.append(
            {
                "disease_id": disease.disease_id,
                "disease": disease_display(disease),
                "icd10": ";".join(disease.ranges),
                "best_token": token_label(loaded, best_token),
                "score": score,
                "rank_among_diagnosis_tokens": rank,
                "token_ids": tokens,
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    out = []
    for rank, row in enumerate(rows[: int(topn)], start=1):
        item = dict(row)
        item["rank"] = rank
        item.pop("token_ids", None)
        out.append(item)
    return out


def annotate_selected_future_matches(
    rows: Sequence[dict],
    diseases: Sequence[DiseaseSpec],
    loaded: LoadedModel,
    future_events: Sequence[dict],
    baseline_age: float,
) -> List[dict]:
    by_id = {disease.disease_id: disease for disease in diseases}
    out = []
    for row in rows:
        disease = by_id[str(row["disease_id"])]
        tokens = token_ids_for_disease(disease, loaded.token_codes)
        matched = first_matching_event(future_events, tokens)
        enriched = dict(row)
        enriched["appeared_in_future"] = matched is not None
        enriched["first_future_token"] = str(matched["token"]) if matched else ""
        enriched["first_future_age_years"] = float(matched["age_years"]) if matched else None
        enriched["years_after_baseline"] = (float(matched["age_years"]) - baseline_age) if matched else None
        out.append(enriched)
    return out


def actual_future_scores(
    loaded: LoadedModel,
    logits_at_position: torch.Tensor,
    future_events: Sequence[dict],
) -> List[dict]:
    candidate_cache = {
        "diag": token_ids_by_prefix(loaded, "diag"),
        "cancer": token_ids_by_prefix(loaded, "cancer"),
        "death": token_ids_by_prefix(loaded, "death"),
    }
    rows = []
    for event in future_events:
        kind = str(event["event_type"])
        if kind not in candidate_cache:
            continue
        rank, score = rank_token(logits_at_position, int(event["token_id"]), candidate_cache[kind])
        rows.append(
            {
                "age_years": event["age_years"],
                "token": event["token"],
                "event_type": kind,
                "event_type_name": event["event_type_name"],
                "score_at_baseline": score,
                "rank_within_event_type": rank,
            }
        )
    return rows


def write_timeline_svg(path: Path, payload: dict) -> None:
    events = payload["all_events"]
    baseline_age = float(payload["baseline_age_years"])
    ages = [float(event["age_years"]) for event in events] + [baseline_age]
    min_age = math.floor(min(ages)) - 1
    max_age = math.ceil(max(ages)) + 1
    if max_age <= min_age:
        max_age = min_age + 1

    width, height = 1280, 430
    left, right = 85, 65
    top, axis_y = 78, 315
    plot_w = width - left - right

    def x_for(age: float) -> float:
        return left + (float(age) - min_age) / (max_age - min_age) * plot_w

    colors = {
        "diag": "#2563eb",
        "proc": "#d97706",
        "cancer": "#be123c",
        "death": "#111827",
        "baseline": "#0891b2",
        "future": "#e0f2fe",
    }
    lane_y = {"cancer": 112, "death": 112, "diag": 172, "proc": 232}
    baseline_x = x_for(baseline_age)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        (
            "<style>"
            "text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#172033;font-size:14px}"
            ".title{font-size:25px;font-weight:700}.sub{fill:#596275;font-size:13px}"
            ".label{font-size:12px}.axis{stroke:#8b95a7;stroke-width:2}.grid{stroke:#e5e7eb;stroke-width:1}"
            "</style>"
        ),
        f'<rect x="{baseline_x:.1f}" y="{top}" width="{width - right - baseline_x:.1f}" height="{axis_y - top}" fill="{colors["future"]}" opacity="0.55"/>',
        f'<text x="{left}" y="38" class="title">具体患者未来轨迹 demo: {escape(payload["case_id"])}</text>',
        f'<text x="{left}" y="62" class="sub">基线左侧为模型可见历史；右侧为推理时隐藏、事后用于对照的真实未来轨迹。</text>',
        f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" class="axis"/>',
        f'<line x1="{baseline_x:.1f}" y1="{top - 8}" x2="{baseline_x:.1f}" y2="{axis_y + 12}" stroke="{colors["baseline"]}" stroke-width="3"/>',
        f'<text x="{baseline_x + 8:.1f}" y="{top - 15}" class="sub">baseline {baseline_age:.3f}y</text>',
    ]

    step = max(1, (max_age - min_age) // 7)
    for tick in range(min_age, max_age + 1, step):
        x = x_for(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{axis_y + 8}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{axis_y + 30}" text-anchor="middle" class="sub">{tick}</text>')
    parts.append(f'<text x="{width - right}" y="{axis_y + 54}" text-anchor="end" class="sub">Age (years)</text>')

    key_tokens = set(payload["key_future_tokens"]) | {payload["target_token"]}
    for event in events:
        age = float(event["age_years"])
        kind = str(event["event_type"])
        x = x_for(age)
        y = lane_y.get(kind, 200)
        fill = colors.get(kind, "#64748b")
        future = age > baseline_age
        radius = 8 if future else 6
        stroke = "#ffffff" if future else "#475569"
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{y + radius}" x2="{x:.1f}" y2="{axis_y}" stroke="{fill}" stroke-width="1" stroke-dasharray="3 4" opacity="0.65"/>')
        should_label = kind in {"diag", "cancer", "death"} or str(event["token"]) in key_tokens
        if should_label:
            dy = -18 if kind in {"diag", "cancer", "death"} else 28
            label = f'{age:.3f}y {event["token"]}'
            parts.append(f'<text x="{x:.1f}" y="{y + dy}" text-anchor="middle" class="label">{escape(label)}</text>')

    legend = [("诊断", colors["diag"]), ("操作", colors["proc"]), ("癌症登记", colors["cancer"]), ("死亡登记", colors["death"])]
    for i, (name, color) in enumerate(legend):
        x = left + i * 150
        y = 386
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{x + 14}" y="{y + 5}" class="sub">{escape(name)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_prediction_bars_svg(path: Path, payload: dict) -> None:
    sections = [
        ("Top diagnosis candidates", payload["top_diagnoses"], "token"),
        ("Top selected disease groups", payload["top_selected_diseases"], "disease"),
        ("Top death candidates", payload["top_death"], "token"),
        ("Top cancer candidates", payload["top_cancer"], "token"),
    ]
    row_h = 38
    section_gap = 52
    width = 1280
    left = 330
    y = 92
    height = 130 + sum(max(1, len(rows)) * row_h + section_gap for _, rows, _ in sections)
    bar_max = width - left - 220

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        (
            "<style>"
            "text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#172033;font-size:14px}"
            ".title{font-size:25px;font-weight:700}.sub{fill:#596275;font-size:13px}"
            ".section{font-size:18px;font-weight:700}.rank{font-weight:700}"
            "</style>"
        ),
        f'<text x="70" y="38" class="title">Baseline ranked candidates</text>',
        '<text x="70" y="62" class="sub">Raw logits are only used for ranking within this model; green bars mean the token/group appears later in the hidden future.</text>',
    ]

    for title, rows, label_key in sections:
        parts.append(f'<text x="70" y="{y}" class="section">{escape(title)}</text>')
        y += 18
        scores = [float(row["score"]) for row in rows]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        span = max(max_score - min_score, 1e-6)
        if not rows:
            parts.append(f'<text x="92" y="{y + 24}" class="sub">No candidate tokens in this model vocabulary.</text>')
            y += row_h + section_gap
            continue
        for row in rows:
            y += row_h
            score = float(row["score"])
            frac = (score - min_score) / span
            bar_w = max(26.0, 110.0 + frac * (bar_max - 110.0))
            matched = bool(row.get("appeared_in_future"))
            fill = "#16a34a" if matched else "#2563eb"
            label = str(row[label_key])
            suffix = "observed later" if matched else "not observed later"
            parts.append(f'<text x="70" y="{y}" class="rank">#{int(row["rank"])}</text>')
            parts.append(f'<text x="120" y="{y}">{escape(label[:54])}</text>')
            parts.append(f'<rect x="{left}" y="{y - 20}" width="{bar_w:.1f}" height="26" rx="8" fill="{fill}" opacity="0.88"/>')
            parts.append(f'<text x="{left + bar_w + 12:.1f}" y="{y}" class="sub">logit {score:.3f}; {escape(suffix)}</text>')
        y += section_gap

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_future_scores_svg(path: Path, payload: dict) -> None:
    rows = payload["actual_future_scored_events"]
    width = 1280
    row_h = 42
    top = 92
    height = top + max(1, len(rows)) * row_h + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        (
            "<style>"
            "text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#172033;font-size:14px}"
            ".title{font-size:25px;font-weight:700}.sub{fill:#596275;font-size:13px}"
            ".head{font-weight:700;fill:#0f172a}.line{stroke:#e5e7eb;stroke-width:1}"
            "</style>"
        ),
        '<text x="70" y="38" class="title">Actual future events scored at baseline</text>',
        '<text x="70" y="62" class="sub">Ranks show where each true future token stood among same-type candidates at the baseline step.</text>',
        '<text x="75" y="92" class="head">Age</text>',
        '<text x="185" y="92" class="head">Actual future token</text>',
        '<text x="520" y="92" class="head">Type</text>',
        '<text x="700" y="92" class="head">Baseline logit</text>',
        '<text x="900" y="92" class="head">Rank within type</text>',
    ]
    if not rows:
        parts.append('<text x="75" y="130" class="sub">No diagnosis/cancer/death events after baseline.</text>')
    for i, row in enumerate(rows, start=1):
        y = top + i * row_h
        parts.append(f'<line x1="70" y1="{y - 28}" x2="{width - 70}" y2="{y - 28}" class="line"/>')
        parts.append(f'<text x="75" y="{y}">{fmt(row["age_years"])}</text>')
        parts.append(f'<text x="185" y="{y}">{escape(row["token"])}</text>')
        parts.append(f'<text x="520" y="{y}">{escape(row["event_type_name"])}</text>')
        parts.append(f'<text x="700" y="{y}">{fmt(row["score_at_baseline"])}</text>')
        rank = row["rank_within_event_type"]
        parts.append(f'<text x="900" y="{y}">{escape(rank if rank is not None else "")}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_visuals(output_dir: Path, case_id: str, payload: dict) -> Dict[str, str]:
    paths = {
        "timeline_svg": output_dir / f"{case_id}_timeline.svg",
        "prediction_bars_svg": output_dir / f"{case_id}_prediction_bars.svg",
        "future_scores_svg": output_dir / f"{case_id}_future_scores.svg",
    }
    write_timeline_svg(paths["timeline_svg"], payload)
    write_prediction_bars_svg(paths["prediction_bars_svg"], payload)
    write_future_scores_svg(paths["future_scores_svg"], payload)
    return {key: str(path) for key, path in paths.items()}


def write_markdown(path: Path, payload: dict) -> None:
    first_hidden = payload["first_hidden_event"]
    death_summary = ", ".join(event["token"] for event in payload["future_death_events"]) or "未观察到死亡登记"
    cancer_summary = ", ".join(event["token"] for event in payload["future_cancer_events"]) or "基线后未观察到癌症登记"

    lines = [
        f"# 具体患者未来轨迹 Demo: {payload['case_id']}",
        "",
        "## Demo 定位",
        "",
        "该 demo 使用训练好的 Exp2 多事件模型，在一个具体测试集患者的基线时点隐藏其后续事件，并把模型在基线时点给出的 ranked token 与真实未来轨迹对照。",
        "",
        "注意：当前模型是序列 next-token 模型；这里展示的是基线时点的一步候选排序与后续真实轨迹的对应关系，不是固定 5 年绝对风险预测。",
        "",
        "## 病例概览",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 模型 | {payload['model']} |",
        f"| 数据 split | {payload['split']} |",
        f"| split_patient_index | {payload['split_patient_index']} |",
        f"| 患者 ID | 已脱敏，不在汇报材料中展示 |",
        f"| 基线年龄 | {fmt(payload['baseline_age_years'])} 岁 |",
        f"| 首个隐藏真实事件 | `{first_hidden['token']}` at {fmt(first_hidden['age_years'])} 岁 |",
        f"| 目标展示疾病 | {payload['target_disease']} |",
        f"| 目标疾病首次事件是否诊断 Top-5 命中 | {'是' if payload['target_in_top5_diagnosis'] else '否'} |",
        f"| 基线后随访长度 | {fmt(payload['followup_years'])} 年 |",
        f"| 基线后死亡信息 | {death_summary} |",
        f"| 基线后癌症信息 | {cancer_summary} |",
        "",
        "## 可视化",
        "",
        f"![Patient timeline]({Path(payload['visuals']['timeline_svg']).name})",
        "",
        f"![Baseline ranked candidates]({Path(payload['visuals']['prediction_bars_svg']).name})",
        "",
        f"![Actual future event scores]({Path(payload['visuals']['future_scores_svg']).name})",
        "",
        "## 基线前模型可见历史",
        "",
        "| 年龄 | token | 类型 |",
        "|---:|---|---|",
    ]
    for event in payload["visible_history"]:
        lines.append(f"| {fmt(event['age_years'])} | `{event['token']}` | {event['event_type_name']} |")

    lines.extend(
        [
            "",
            "## 基线后隐藏真实轨迹",
            "",
            "| 年龄 | token | 类型 |",
            "|---:|---|---|",
        ]
    )
    for event in payload["future_events"]:
        lines.append(f"| {fmt(event['age_years'])} | `{event['token']}` | {event['event_type_name']} |")

    lines.extend(
        [
            "",
            "## 基线模型输出：诊断 token",
            "",
            "| Rank | token | raw logit | 后续是否出现 | 首次出现年龄 |",
            "|---:|---|---:|---|---:|",
        ]
    )
    for row in payload["top_diagnoses"]:
        lines.append(
            f"| {row['rank']} | `{row['token']}` | {fmt(row['score'])} | "
            f"{'是' if row['appeared_in_future'] else '否'} | {fmt(row['first_future_age_years'])} |"
        )

    lines.extend(
        [
            "",
            "## 基线模型输出：有限疾病组",
            "",
            "| Rank | 疾病组 | ICD-10 | 代表 token | raw logit | 诊断内排名 | 后续是否出现 | 首次出现 |",
            "|---:|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in payload["top_selected_diseases"]:
        lines.append(
            f"| {row['rank']} | {row['disease']} | `{row['icd10']}` | `{row['best_token']}` | "
            f"{fmt(row['score'])} | {row['rank_among_diagnosis_tokens']} | "
            f"{'是' if row['appeared_in_future'] else '否'} | {row['first_future_token']} {fmt(row['first_future_age_years'])} |"
        )

    lines.extend(
        [
            "",
            "## 基线模型输出：死亡和癌症 token",
            "",
            "| 类型 | Rank | token | raw logit | 后续是否出现 | 首次出现年龄 |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for group_name, rows in [("死亡", payload["top_death"]), ("癌症", payload["top_cancer"])]:
        for row in rows:
            lines.append(
                f"| {group_name} | {row['rank']} | `{row['token']}` | {fmt(row['score'])} | "
                f"{'是' if row['appeared_in_future'] else '否'} | {fmt(row['first_future_age_years'])} |"
            )

    lines.extend(
        [
            "",
            "## 真实未来事件在基线时的得分",
            "",
            "| 年龄 | 真实 token | 类型 | baseline raw logit | 同类型候选内排名 |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for row in payload["actual_future_scored_events"]:
        lines.append(
            f"| {fmt(row['age_years'])} | `{row['token']}` | {row['event_type_name']} | "
            f"{fmt(row['score_at_baseline'])} | {row['rank_within_event_type']} |"
        )

    lines.extend(
        [
            "",
            "## 汇报口径",
            "",
            (
                f"在 {fmt(payload['baseline_age_years'])} 岁基线时点，模型只能看到此前病史；真实未来首个隐藏事件为 "
                f"`{first_hidden['token']}`。模型对该目标 token 的诊断内排名为 "
                f"{payload['target_rank_among_diagnosis_tokens']}，因此这是一个"
                f"{' Top-5 命中' if payload['target_in_top5_diagnosis'] else '未 Top-5 命中'}的目标疾病展示病例。"
            ),
            "",
            (
                f"该患者基线后真实轨迹还出现了 {payload['future_diagnosis_summary']}，并最终记录 "
                f"{death_summary}。因此该 demo 可以展示具体患者层面的“模型候选排序”和“真实未来疾病/死亡轨迹”对照。"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict:
    disease_specs = parse_selected_diseases(args.diseases_yaml)
    disease_by_id = {disease.disease_id: disease for disease in disease_specs}
    if args.disease_id not in disease_by_id:
        raise ValueError(f"Unknown disease_id={args.disease_id!r}")

    loaded = load_model(select_model_spec(args.model_id), split=args.split, device=args.device)
    with torch.no_grad():
        logits, inputs, input_ages, targets, target_ages = forward_one(loaded, args.split_patient_index, args.device)

    position = find_prediction_position(loaded, targets, target_ages, args.target_token)
    logits_at_position = logits[0, position]
    target_token_id = int(targets[0, position].detach().cpu().item())
    baseline_age = float(input_ages[0, position].detach().cpu().item()) / 365.25
    target_age = float(target_ages[0, position].detach().cpu().item()) / 365.25
    all_events = patient_events(loaded, args.split_patient_index)
    visible = [event for event in all_events if float(event["age_years"]) <= baseline_age]
    future = [event for event in all_events if float(event["age_years"]) > baseline_age]
    if not future:
        raise RuntimeError("Selected patient has no future events after the baseline prediction point.")

    top_diag = annotate_exact_future_matches(top_tokens(loaded, logits_at_position, "diag", args.topn), future, baseline_age)
    top_death = annotate_exact_future_matches(top_tokens(loaded, logits_at_position, "death", args.topn), future, baseline_age)
    top_cancer = annotate_exact_future_matches(top_tokens(loaded, logits_at_position, "cancer", args.topn), future, baseline_age)
    top_diseases = annotate_selected_future_matches(
        selected_disease_scores(loaded, disease_specs, logits_at_position, args.topn),
        disease_specs,
        loaded,
        future,
        baseline_age,
    )

    diagnosis_ids = token_ids_by_prefix(loaded, "diag")
    target_rank, target_score = rank_token(logits_at_position, target_token_id, diagnosis_ids)
    patient_row = read_patient_index(loaded.spec.data_dir, args.split, args.split_patient_index)
    future_death = [event for event in future if event["event_type"] == "death"]
    future_cancer = [event for event in future if event["event_type"] == "cancer"]
    future_diag_tokens = sorted({event["token"] for event in future if event["event_type"] == "diag"})
    key_future_tokens = [args.target_token] + [event["token"] for event in future_death + future_cancer]
    scored_future = actual_future_scores(loaded, logits_at_position, future)

    payload = {
        "case_id": args.case_id,
        "model_id": loaded.spec.model_id,
        "model": loaded.spec.display_name,
        "split": args.split,
        "split_patient_index": int(args.split_patient_index),
        "eid_redacted": bool(patient_row.get("eid")),
        "num_events": patient_row.get("num_events", ""),
        "target_token": token_label(loaded, target_token_id),
        "target_token_id": target_token_id,
        "target_disease": disease_display(disease_by_id[args.disease_id]),
        "baseline_age_years": baseline_age,
        "target_age_years": target_age,
        "target_gap_days": (target_age - baseline_age) * 365.25,
        "target_rank_among_diagnosis_tokens": target_rank,
        "target_score": target_score,
        "target_in_top5_diagnosis": bool(target_rank is not None and target_rank <= 5),
        "followup_years": float(future[-1]["age_years"]) - baseline_age,
        "first_hidden_event": future[0],
        "visible_history": visible,
        "future_events": future,
        "all_events": all_events,
        "future_death_events": future_death,
        "future_cancer_events": future_cancer,
        "future_diagnosis_summary": ", ".join(f"`{token}`" for token in future_diag_tokens) or "无诊断事件",
        "key_future_tokens": key_future_tokens,
        "top_diagnoses": top_diag,
        "top_selected_diseases": top_diseases,
        "top_death": top_death,
        "top_cancer": top_cancer,
        "actual_future_scored_events": scored_future,
    }
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_payload(args)
    payload["visuals"] = write_visuals(args.output_dir, args.case_id, payload)

    json_path = args.output_dir / f"{args.case_id}_demo.json"
    md_path = args.output_dir / f"{args.case_id}_demo.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "visuals": payload["visuals"],
                "split_patient_index": payload["split_patient_index"],
                "target_rank_among_diagnosis_tokens": payload["target_rank_among_diagnosis_tokens"],
                "target_in_top5_diagnosis": payload["target_in_top5_diagnosis"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
