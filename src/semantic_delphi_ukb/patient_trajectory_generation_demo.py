from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from semantic_delphi_ukb.patient_future_demo import (
    event_type,
    event_type_name,
    patient_events,
    select_model_spec,
)
from semantic_delphi_ukb.selected_disease_demo import LoadedModel, load_model
from semantic_delphi_ukb.single_case_demo import find_prediction_position, forward_one, read_patient_index, token_label


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
MASK_TIME = -10000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate stochastic patient trajectories from an Exp2 baseline.")
    parser.add_argument("--case-id", type=str, default="AF-GEN-001")
    parser.add_argument("--model-id", type=str, default="exp2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split-patient-index", type=int, default=2028)
    parser.add_argument("--target-token", type=str, default="diag:I48")
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "demo" / "patient_trajectory_generation_demo")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-rollouts", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-age-years", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--scope", choices=["full", "clinical"], default="full")
    parser.add_argument("--allow-repeat", action="store_true")
    return parser


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


def first_event(events: Sequence[dict], token: str) -> Optional[dict]:
    for event in events:
        if str(event["token"]) == token:
            return event
    return None


def first_event_with_prefix(events: Sequence[dict], prefix: str) -> Optional[dict]:
    marker = f"{prefix}:"
    for event in events:
        if str(event["token"]).startswith(marker):
            return event
    return None


def candidate_ids_for_scope(loaded: LoadedModel, scope: str) -> List[int]:
    if scope == "clinical":
        prefixes = ("diag:", "cancer:", "death:")
        return [idx for idx, label in enumerate(loaded.labels) if label.startswith(prefixes)]
    return [idx for idx, label in enumerate(loaded.labels) if idx > 1 and label not in {"", "Padding", "No event"}]


def prepare_baseline_context(
    loaded: LoadedModel,
    row_index: int,
    target_token: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float, int]:
    with torch.no_grad():
        _, inputs, input_ages, targets, target_ages = forward_one(loaded, row_index, device)
    position = find_prediction_position(loaded, targets, target_ages, target_token)
    context_tokens = inputs[0, : position + 1].detach().clone()
    context_ages = input_ages[0, : position + 1].detach().clone()
    valid = (context_tokens > 0) & (context_ages > MASK_TIME / 2)
    context_tokens = context_tokens[valid].unsqueeze(0).to(device)
    context_ages = context_ages[valid].unsqueeze(0).to(device)

    static_features = None
    if loaded.spec.model_type == "multitype":
        if loaded.static_matrix is None:
            raise RuntimeError("Multitype trajectory generation requires static_matrix.")
        static_features = torch.tensor(loaded.static_matrix[[int(row_index)]], dtype=torch.float32, device=device)

    baseline_age_years = float(context_ages[0, -1].detach().cpu().item()) / 365.25
    target_token_id = int(targets[0, position].detach().cpu().item())
    return context_tokens, context_ages, static_features, baseline_age_years, target_token_id


def forward_last_logits(
    loaded: LoadedModel,
    tokens: torch.Tensor,
    ages: torch.Tensor,
    static_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if loaded.spec.model_type == "multitype":
        logits, _, _ = loaded.model(tokens, ages, static_features)
    else:
        logits, _, _ = loaded.model(tokens, ages)
    return logits[:, -1, :]


def rollout_many(
    loaded: LoadedModel,
    context_tokens: torch.Tensor,
    context_ages: torch.Tensor,
    static_features: Optional[torch.Tensor],
    scope: str,
    num_rollouts: int,
    max_new_tokens: int,
    max_age_years: float,
    seed: int,
    no_repeat: bool,
    device: str,
) -> List[dict]:
    n = int(num_rollouts)
    tokens = context_tokens.repeat(n, 1).to(device)
    ages = context_ages.repeat(n, 1).to(device)
    static_batch = static_features.repeat(n, 1).to(device) if static_features is not None else None
    active = torch.ones(n, dtype=torch.bool, device=device)
    generated: List[List[dict]] = [[] for _ in range(n)]
    stop_reasons = ["max_new_tokens"] * n

    allowed_ids = candidate_ids_for_scope(loaded, scope)
    allowed = torch.zeros(len(loaded.labels), dtype=torch.bool, device=device)
    allowed[torch.tensor(allowed_ids, dtype=torch.long, device=device)] = True
    for token_id in loaded.model.config.ignore_tokens:
        if 0 <= int(token_id) < len(loaded.labels):
            allowed[int(token_id)] = False
    if len(loaded.labels) > 1:
        allowed[1] = False

    generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    max_age_days = float(max_age_years) * 365.25
    max_wait_days = 365.25 * 80.0

    for _ in range(int(max_new_tokens)):
        active_ids = torch.where(active)[0]
        if int(active_ids.numel()) == 0:
            break
        forward_tokens = tokens[active_ids, -loaded.block_size :]
        forward_ages = ages[active_ids, -loaded.block_size :]
        forward_static = static_batch[active_ids] if static_batch is not None else None
        with torch.no_grad():
            logits = forward_last_logits(loaded, forward_tokens, forward_ages, forward_static)
        logits[:, ~allowed] = -torch.inf
        if no_repeat:
            seen = tokens[active_ids].clone()
            seen[seen < 2] = 0
            logits.scatter_(1, seen, -torch.inf)

        uniform = torch.rand(logits.shape, generator=generator, device=logits.device).clamp_min(1e-12)
        waiting = -torch.log(uniform) * torch.exp(-logits)
        waiting = torch.clamp(waiting, min=0.0, max=max_wait_days)
        waiting = waiting.masked_fill(~torch.isfinite(logits), torch.inf)
        delta_days, next_tokens = waiting.min(dim=1)
        next_ages = forward_ages[:, -1] + delta_days

        appended_tokens = torch.zeros((n, 1), dtype=torch.long, device=device)
        appended_ages = torch.full((n, 1), MASK_TIME, dtype=torch.float32, device=device)
        appended_tokens[active_ids, 0] = next_tokens
        appended_ages[active_ids, 0] = next_ages
        tokens = torch.cat([tokens, appended_tokens], dim=1)
        ages = torch.cat([ages, appended_ages], dim=1)

        for j, rollout_id in enumerate(active_ids.detach().cpu().tolist()):
            age_years = float(next_ages[j].detach().cpu().item()) / 365.25
            token_id = int(next_tokens[j].detach().cpu().item())
            token = token_label(loaded, token_id)
            if age_years > max_age_years:
                stop_reasons[rollout_id] = "max_age"
                active[rollout_id] = False
                continue
            generated[rollout_id].append(
                {
                    "step": len(generated[rollout_id]) + 1,
                    "age_years": age_years,
                    "token_id": token_id,
                    "token": token,
                    "event_type": event_type(token),
                    "event_type_name": event_type_name(event_type(token)),
                }
            )
            if token.startswith("death:"):
                stop_reasons[rollout_id] = "death"
                active[rollout_id] = False

    return [
        {
            "rollout_id": rollout_id,
            "stop_reason": stop_reasons[rollout_id],
            "generated_events": events,
        }
        for rollout_id, events in enumerate(generated)
    ]


def rollout_summary_rows(rollouts: Sequence[dict], actual: dict) -> List[dict]:
    exact_death_token = actual["exact_death_token"]
    rows = []
    for rollout in rollouts:
        events = rollout["generated_events"]
        i48 = first_event(events, "diag:I48")
        i50 = first_event(events, "diag:I50")
        exact_death = first_event(events, exact_death_token) if exact_death_token else None
        any_death = first_event_with_prefix(events, "death")
        rows.append(
            {
                "rollout_id": int(rollout["rollout_id"]),
                "stop_reason": rollout["stop_reason"],
                "num_generated_events": len(events),
                "hit_diag_i48": int(i48 is not None),
                "age_diag_i48": i48["age_years"] if i48 else None,
                "hit_diag_i50": int(i50 is not None),
                "age_diag_i50": i50["age_years"] if i50 else None,
                "hit_any_death": int(any_death is not None),
                "generated_death_token": any_death["token"] if any_death else "",
                "generated_death_age_years": any_death["age_years"] if any_death else None,
                "hit_exact_death": int(exact_death is not None),
                "exact_death_age_years": exact_death["age_years"] if exact_death else None,
                "death_age_error_years": abs(float(any_death["age_years"]) - float(actual["death_age_years"]))
                if any_death and actual["death_age_years"] is not None
                else None,
            }
        )
    return rows


def mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return sum(clean) / len(clean) if clean else None


def summarize(rows: Sequence[dict], actual: dict) -> dict:
    n = len(rows)
    death_ages = [row["generated_death_age_years"] for row in rows if row["generated_death_age_years"] is not None]
    exact_death_ages = [row["exact_death_age_years"] for row in rows if row["exact_death_age_years"] is not None]
    return {
        "num_rollouts": n,
        "hit_diag_i48": sum(int(row["hit_diag_i48"]) for row in rows),
        "hit_diag_i48_rate": sum(int(row["hit_diag_i48"]) for row in rows) / n if n else float("nan"),
        "hit_diag_i50": sum(int(row["hit_diag_i50"]) for row in rows),
        "hit_diag_i50_rate": sum(int(row["hit_diag_i50"]) for row in rows) / n if n else float("nan"),
        "hit_any_death": sum(int(row["hit_any_death"]) for row in rows),
        "hit_any_death_rate": sum(int(row["hit_any_death"]) for row in rows) / n if n else float("nan"),
        "hit_exact_death": sum(int(row["hit_exact_death"]) for row in rows),
        "hit_exact_death_rate": sum(int(row["hit_exact_death"]) for row in rows) / n if n else float("nan"),
        "generated_death_age_mean": mean(death_ages),
        "exact_death_age_mean": mean(exact_death_ages),
        "actual_death_age_years": actual["death_age_years"],
    }


def choose_representative_rollout(rows: Sequence[dict], rollouts: Sequence[dict], actual: dict) -> dict:
    actual_death_age = actual["death_age_years"]

    def score(row: dict) -> float:
        value = 1000.0 * int(row["hit_exact_death"])
        value += 100.0 * int(row["hit_any_death"])
        if row["generated_death_age_years"] is not None and actual_death_age is not None:
            value -= 10.0 * abs(float(row["generated_death_age_years"]) - float(actual_death_age))
        value += 10.0 * int(row["hit_diag_i48"])
        value += 8.0 * int(row["hit_diag_i50"])
        return value

    best_row = max(rows, key=score)
    return rollouts[int(best_row["rollout_id"])]


def build_actual_payload(future_events: Sequence[dict]) -> dict:
    death_event = first_event_with_prefix(future_events, "death")
    return {
        "future_diag_tokens": sorted({event["token"] for event in future_events if event["event_type"] == "diag"}),
        "death_age_years": death_event["age_years"] if death_event else None,
        "exact_death_token": death_event["token"] if death_event else "",
        "future_death_events": [event for event in future_events if event["event_type"] == "death"],
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "rollout_id",
        "stop_reason",
        "num_generated_events",
        "hit_diag_i48",
        "age_diag_i48",
        "hit_diag_i50",
        "age_diag_i50",
        "hit_any_death",
        "generated_death_token",
        "generated_death_age_years",
        "hit_exact_death",
        "exact_death_age_years",
        "death_age_error_years",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_hit_rate_svg(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    bars = [
        ("diag:I48", summary["hit_diag_i48"], summary["hit_diag_i48_rate"], "#2563eb"),
        ("diag:I50", summary["hit_diag_i50"], summary["hit_diag_i50_rate"], "#0891b2"),
        ("any death:*", summary["hit_any_death"], summary["hit_any_death_rate"], "#111827"),
        (payload["actual"]["exact_death_token"], summary["hit_exact_death"], summary["hit_exact_death_rate"], "#be123c"),
    ]
    width, height = 1080, 390
    left, top = 245, 95
    bar_max = 650
    row_h = 58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        (
            "<style>"
            "text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#172033;font-size:14px}"
            ".title{font-size:25px;font-weight:700}.sub{fill:#596275;font-size:13px}"
            ".label{font-size:15px}.rank{font-size:16px;font-weight:700}"
            "</style>"
        ),
        '<text x="70" y="38" class="title">Trajectory generation key-event hit rates</text>',
        f'<text x="70" y="62" class="sub">N={summary["num_rollouts"]} stochastic rollouts; full-vocab generation stops at death, max age, or max steps.</text>',
    ]
    for i, (label, count, rate, color) in enumerate(bars):
        y = top + i * row_h
        width_value = max(4.0, float(rate) * bar_max)
        parts.append(f'<text x="70" y="{y + 23}" class="label">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{width_value:.1f}" height="32" rx="8" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{left + width_value + 14:.1f}" y="{y + 22}" class="sub">{count}/{summary["num_rollouts"]} ({rate * 100:.1f}%)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_death_age_svg(path: Path, payload: dict) -> None:
    rows = payload["rollout_rows"]
    ages = [float(row["generated_death_age_years"]) for row in rows if row["generated_death_age_years"] is not None]
    actual_age = payload["actual"]["death_age_years"]
    width, height = 1080, 400
    left, right, top, bottom = 80, 60, 88, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        (
            "<style>"
            "text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#172033;font-size:14px}"
            ".title{font-size:25px;font-weight:700}.sub{fill:#596275;font-size:13px}"
            ".axis{stroke:#8b95a7;stroke-width:2}.grid{stroke:#e5e7eb;stroke-width:1}"
            "</style>"
        ),
        '<text x="70" y="38" class="title">Generated death-age distribution</text>',
        '<text x="70" y="62" class="sub">Only rollouts that generated a death:* token are included.</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]
    if not ages:
        parts.append(f'<text x="{left + 20}" y="{top + 80}" class="sub">No generated death token.</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return

    min_age = math.floor(min(min(ages), float(actual_age or min(ages)))) - 1
    max_age = math.ceil(max(max(ages), float(actual_age or max(ages)))) + 1
    bins = max(4, min(12, max_age - min_age))
    counts = [0] * bins
    for age in ages:
        idx = min(bins - 1, max(0, int((age - min_age) / (max_age - min_age) * bins)))
        counts[idx] += 1
    max_count = max(counts)
    bar_gap = 6
    bar_w = (plot_w - (bins - 1) * bar_gap) / bins
    for i, count in enumerate(counts):
        frac = count / max_count if max_count else 0.0
        bar_h = frac * plot_h
        x = left + i * (bar_w + bar_gap)
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#2563eb" opacity="0.82"/>')
        label_age = min_age + (i + 0.5) * (max_age - min_age) / bins
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="sub">{label_age:.0f}</text>')
        if count:
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="sub">{count}</text>')
    if actual_age is not None:
        x_actual = left + (float(actual_age) - min_age) / (max_age - min_age) * plot_w
        parts.append(f'<line x1="{x_actual:.1f}" y1="{top}" x2="{x_actual:.1f}" y2="{top + plot_h}" stroke="#dc2626" stroke-width="3"/>')
        parts.append(f'<text x="{x_actual + 8:.1f}" y="{top + 16}" class="sub">actual death {float(actual_age):.3f}y</text>')
    parts.append(f'<text x="{left + plot_w}" y="{height - 20}" text-anchor="end" class="sub">Age at generated death event</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_trajectory_svg(path: Path, payload: dict) -> None:
    baseline_age = float(payload["baseline_age_years"])
    actual_future = payload["future_events"]
    generated = payload["representative_rollout"]["generated_events"]
    ages = [baseline_age] + [float(event["age_years"]) for event in actual_future + generated]
    min_age = math.floor(min(ages)) - 1
    max_age = math.ceil(max(ages)) + 1
    if max_age <= min_age:
        max_age = min_age + 1

    width, height = 1280, 430
    left, right = 90, 70
    plot_w = width - left - right
    axis_y = 345
    actual_y = 145
    generated_y = 245

    def x_for(age: float) -> float:
        return left + (float(age) - min_age) / (max_age - min_age) * plot_w

    colors = {"diag": "#2563eb", "proc": "#d97706", "cancer": "#be123c", "death": "#111827", "baseline": "#0891b2"}
    key_tokens = {"diag:I48", "diag:I50", payload["actual"]["exact_death_token"]}
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
        '<text x="70" y="38" class="title">Actual future vs representative generated trajectory</text>',
        '<text x="70" y="62" class="sub">Representative rollout is selected by key-event matches and death-age closeness.</text>',
        f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" class="axis"/>',
        f'<line x1="{left}" y1="{actual_y}" x2="{width - right}" y2="{actual_y}" stroke="#cbd5e1" stroke-width="2"/>',
        f'<line x1="{left}" y1="{generated_y}" x2="{width - right}" y2="{generated_y}" stroke="#cbd5e1" stroke-width="2"/>',
        f'<text x="{left}" y="{actual_y - 20}" class="sub">Actual hidden future</text>',
        f'<text x="{left}" y="{generated_y - 20}" class="sub">Generated rollout #{payload["representative_rollout"]["rollout_id"]}</text>',
    ]
    baseline_x = x_for(baseline_age)
    parts.append(f'<line x1="{baseline_x:.1f}" y1="92" x2="{baseline_x:.1f}" y2="{axis_y + 8}" stroke="{colors["baseline"]}" stroke-width="3"/>')
    parts.append(f'<text x="{baseline_x + 8:.1f}" y="105" class="sub">baseline {baseline_age:.3f}y</text>')

    step = max(1, (max_age - min_age) // 7)
    for tick in range(min_age, max_age + 1, step):
        x = x_for(tick)
        parts.append(f'<line x1="{x:.1f}" y1="92" x2="{x:.1f}" y2="{axis_y + 8}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{axis_y + 30}" text-anchor="middle" class="sub">{tick}</text>')

    for lane_name, events, y_base in [("actual", actual_future, actual_y), ("generated", generated, generated_y)]:
        for event in events:
            age = float(event["age_years"])
            token = str(event["token"])
            kind = str(event["event_type"])
            x = x_for(age)
            radius = 8 if token in key_tokens or kind == "death" else 5
            fill = colors.get(kind, "#64748b")
            parts.append(f'<circle cx="{x:.1f}" cy="{y_base}" r="{radius}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            should_label = token in key_tokens or kind in {"death", "cancer"} or (lane_name == "actual" and kind == "diag")
            if should_label:
                dy = -18 if lane_name == "actual" else 28
                parts.append(f'<text x="{x:.1f}" y="{y_base + dy}" text-anchor="middle" class="label">{escape(token)} {age:.2f}y</text>')

    parts.append(f'<text x="{width - right}" y="{height - 20}" text-anchor="end" class="sub">Age (years)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_visuals(output_dir: Path, case_id: str, payload: dict) -> Dict[str, str]:
    paths = {
        "hit_rate_svg": output_dir / f"{case_id}_key_event_hits.svg",
        "death_age_svg": output_dir / f"{case_id}_death_age_distribution.svg",
        "trajectory_svg": output_dir / f"{case_id}_representative_trajectory.svg",
    }
    write_hit_rate_svg(paths["hit_rate_svg"], payload)
    write_death_age_svg(paths["death_age_svg"], payload)
    write_trajectory_svg(paths["trajectory_svg"], payload)
    return {key: str(path) for key, path in paths.items()}


def write_markdown(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    actual = payload["actual"]
    representative = payload["representative_row"]
    lines = [
        f"# 自回归轨迹生成 Demo: {payload['case_id']}",
        "",
        "## Demo 定位",
        "",
        "该 demo 从同一个具体测试集患者的基线时点出发，让 Exp2 模型连续自回归生成后续事件轨迹。每生成一个 token，就把该 token 和生成年龄追加回输入，再预测下一步。",
        "",
        "这与前一个 baseline ranking demo 不同：这里不是一次 forward 同时解释多个未来事件，而是多步 rollout。结果具有随机性，因此用多次 rollout 的命中率和分布展示，不把单条轨迹解释为确定性临床预测。",
        "",
        "## 生成设置",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 模型 | {payload['model']} |",
        f"| 数据 split | {payload['split']} |",
        f"| split_patient_index | {payload['split_patient_index']} |",
        f"| 患者 ID | 已脱敏，不展示原始 eid |",
        f"| 基线年龄 | {fmt(payload['baseline_age_years'])} 岁 |",
        f"| 生成范围 | {payload['scope']} |",
        f"| rollout 次数 | {summary['num_rollouts']} |",
        f"| 最大生成步数 | {payload['max_new_tokens']} |",
        f"| 最大年龄 | {fmt(payload['max_age_years'])} 岁 |",
        f"| 是否禁止重复 token | {'是' if payload['no_repeat'] else '否'} |",
        f"| 随机种子 | {payload['seed']} |",
        "",
        "## 真实隐藏未来",
        "",
        "| 年龄 | token | 类型 |",
        "|---:|---|---|",
    ]
    for event in payload["future_events"]:
        lines.append(f"| {fmt(event['age_years'])} | `{event['token']}` | {event['event_type_name']} |")

    lines.extend(
        [
            "",
            "## 关键事件生成命中",
            "",
            "| 事件 | 命中次数 | 命中率 | 真实年龄 | 生成年龄均值 |",
            "|---|---:|---:|---:|---:|",
            f"| `diag:I48` | {summary['hit_diag_i48']} / {summary['num_rollouts']} | {summary['hit_diag_i48_rate'] * 100:.1f}% | 69.700 |  |",
            f"| `diag:I50` | {summary['hit_diag_i50']} / {summary['num_rollouts']} | {summary['hit_diag_i50_rate'] * 100:.1f}% | 72.101 |  |",
            f"| 任意 `death:*` | {summary['hit_any_death']} / {summary['num_rollouts']} | {summary['hit_any_death_rate'] * 100:.1f}% | {fmt(actual['death_age_years'])} | {fmt(summary['generated_death_age_mean'])} |",
            f"| `{actual['exact_death_token']}` | {summary['hit_exact_death']} / {summary['num_rollouts']} | {summary['hit_exact_death_rate'] * 100:.1f}% | {fmt(actual['death_age_years'])} | {fmt(summary['exact_death_age_mean'])} |",
            "",
            "## 可视化",
            "",
            f"![Key event hit rates]({Path(payload['visuals']['hit_rate_svg']).name})",
            "",
            f"![Death age distribution]({Path(payload['visuals']['death_age_svg']).name})",
            "",
            f"![Representative trajectory]({Path(payload['visuals']['trajectory_svg']).name})",
            "",
            "## 代表性生成轨迹",
            "",
            f"代表性 rollout：`#{representative['rollout_id']}`。选择规则是优先匹配真实死亡 token，并尽量接近真实死亡年龄；房颤和心衰通过上方命中率统计展示。",
            "",
            "| 步数 | 年龄 | token | 类型 |",
            "|---:|---:|---|---|",
        ]
    )
    for event in payload["representative_rollout"]["generated_events"]:
        lines.append(f"| {event['step']} | {fmt(event['age_years'])} | `{event['token']}` | {event['event_type_name']} |")

    lines.extend(
        [
            "",
            "## 汇报口径",
            "",
            (
                f"该患者真实未来出现 `diag:I48`、`diag:I50`，并在 {fmt(actual['death_age_years'])} 岁记录 "
                f"`{actual['exact_death_token']}`。在 {summary['num_rollouts']} 次自回归生成中，"
                f"`diag:I48` 命中 {summary['hit_diag_i48']} 次，`diag:I50` 命中 {summary['hit_diag_i50']} 次，"
                f"任意死亡 token 命中 {summary['hit_any_death']} 次，精确死亡 token `{actual['exact_death_token']}` "
                f"命中 {summary['hit_exact_death']} 次。"
            ),
            "",
            "因此，该 demo 可以作为“具体患者未来轨迹生成”的展示材料；但它仍是随机生成实验，不应表述为确定性预测或临床概率。",
            "",
            "## 输出文件",
            "",
            f"- `{Path(payload['paths']['json']).name}`：完整 JSON，包括每条 rollout 的生成序列。",
            f"- `{Path(payload['paths']['rollout_csv']).name}`：每条 rollout 的命中摘要表。",
            f"- `{Path(payload['paths']['markdown']).name}`：当前汇报说明。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict:
    loaded = load_model(select_model_spec(args.model_id), split=args.split, device=args.device)
    context_tokens, context_ages, static_features, baseline_age, _ = prepare_baseline_context(
        loaded,
        args.split_patient_index,
        args.target_token,
        args.device,
    )
    all_events = patient_events(loaded, args.split_patient_index)
    visible_history = [event for event in all_events if float(event["age_years"]) <= baseline_age]
    future_events = [event for event in all_events if float(event["age_years"]) > baseline_age]
    actual = build_actual_payload(future_events)
    rollouts = rollout_many(
        loaded=loaded,
        context_tokens=context_tokens,
        context_ages=context_ages,
        static_features=static_features,
        scope=args.scope,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        max_age_years=args.max_age_years,
        seed=args.seed,
        no_repeat=not args.allow_repeat,
        device=args.device,
    )
    rows = rollout_summary_rows(rollouts, actual)
    summary = summarize(rows, actual)
    representative = choose_representative_rollout(rows, rollouts, actual)
    representative_row = rows[int(representative["rollout_id"])]
    patient_row = read_patient_index(loaded.spec.data_dir, args.split, args.split_patient_index)

    return {
        "case_id": args.case_id,
        "model_id": loaded.spec.model_id,
        "model": loaded.spec.display_name,
        "split": args.split,
        "split_patient_index": int(args.split_patient_index),
        "eid_redacted": bool(patient_row.get("eid")),
        "num_events": patient_row.get("num_events", ""),
        "baseline_age_years": baseline_age,
        "target_token": args.target_token,
        "scope": args.scope,
        "num_rollouts": int(args.num_rollouts),
        "max_new_tokens": int(args.max_new_tokens),
        "max_age_years": float(args.max_age_years),
        "seed": int(args.seed),
        "no_repeat": not args.allow_repeat,
        "visible_history": visible_history,
        "future_events": future_events,
        "actual": actual,
        "rollouts": rollouts,
        "rollout_rows": rows,
        "summary": summary,
        "representative_rollout": representative,
        "representative_row": representative_row,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_payload(args)
    payload["visuals"] = write_visuals(args.output_dir, args.case_id, payload)
    json_path = args.output_dir / f"{args.case_id}_demo.json"
    csv_path = args.output_dir / f"{args.case_id}_rollouts.csv"
    md_path = args.output_dir / f"{args.case_id}_demo.md"
    payload["paths"] = {"json": str(json_path), "rollout_csv": str(csv_path), "markdown": str(md_path)}

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, payload["rollout_rows"])
    write_markdown(md_path, payload)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "rollout_csv": str(csv_path),
                "visuals": payload["visuals"],
                "summary": payload["summary"],
                "representative_rollout_id": payload["representative_rollout"]["rollout_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
