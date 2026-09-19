from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


REPO_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create diagnostics for the survival-constrained trajectory generation innovation branch."
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=REPO_DIR / "results" / "rollout_generation_benchmark_full",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_DIR / "results" / "survival_constrained_generation_branch",
    )
    parser.add_argument(
        "--target-death-rate",
        type=float,
        default=None,
        help="External/validation death rate for global death-gate scaling. If omitted, uses observed rollout cohort rate as an oracle diagnostic only.",
    )
    parser.add_argument("--bins", type=int, default=5)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def brier(scores: Sequence[float], labels: Sequence[int]) -> float:
    return safe_mean([(float(s) - int(y)) ** 2 for s, y in zip(scores, labels)])


def ece(scores: Sequence[float], labels: Sequence[int], bins: int) -> float:
    rows = calibration_bins(scores, labels, bins)
    total = sum(int(row["n"]) for row in rows)
    if total <= 0:
        return float("nan")
    return sum((int(row["n"]) / total) * float(row["abs_calibration_error"]) for row in rows)


def calibration_bins(scores: Sequence[float], labels: Sequence[int], bins: int) -> list[dict]:
    rows = []
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels) if math.isfinite(float(score))]
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        if idx == bins - 1:
            selected = [(s, y) for s, y in pairs if lo <= s <= hi]
        else:
            selected = [(s, y) for s, y in pairs if lo <= s < hi]
        if not selected:
            continue
        mean_pred = safe_mean([s for s, _ in selected])
        observed = safe_mean([y for _, y in selected])
        rows.append(
            {
                "bin": idx,
                "prob_low": lo,
                "prob_high": hi,
                "n": len(selected),
                "mean_predicted_death_prob": mean_pred,
                "observed_death_rate": observed,
                "abs_calibration_error": abs(mean_pred - observed),
            }
        )
    return rows


def group_by_model(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    return dict(grouped)


def death_gate_rows(patient_rows: Sequence[dict[str, str]], target_death_rate: float | None, bins: int) -> tuple[list[dict], list[dict]]:
    out = []
    bin_rows = []
    for model, rows in group_by_model(patient_rows).items():
        raw = [safe_float(row["predicted_death_prob"]) for row in rows]
        labels = [int(safe_float(row["actual_death"], 0.0)) for row in rows]
        observed = safe_mean(labels)
        target = float(target_death_rate) if target_death_rate is not None else observed
        raw_mean = safe_mean(raw)
        scale = target / raw_mean if raw_mean > 0 and math.isfinite(raw_mean) else 0.0
        adjusted = [min(1.0, max(0.0, score * scale)) for score in raw]
        source = "external_or_validation_target" if target_death_rate is not None else "oracle_test_diagnostic"
        suggested_bias = math.log(scale) if scale > 0 and math.isfinite(scale) else float("-inf")
        out.append(
            {
                "model": model,
                "patients": len(rows),
                "target_source": source,
                "observed_death_rate": observed,
                "target_death_rate": target,
                "raw_predicted_death_prob_mean": raw_mean,
                "death_gate_scale": scale,
                "suggested_death_logit_bias": suggested_bias,
                "gated_predicted_death_prob_mean": safe_mean(adjusted),
                "raw_death_brier": brier(raw, labels),
                "gated_death_brier": brier(adjusted, labels),
                "raw_death_ece": ece(raw, labels, bins),
                "gated_death_ece": ece(adjusted, labels, bins),
                "relative_death_prob_reduction": 1.0 - safe_mean(adjusted) / raw_mean if raw_mean > 0 else float("nan"),
            }
        )
        for row in calibration_bins(raw, labels, bins):
            row = {"model": model, "variant": "raw", **row}
            bin_rows.append(row)
        for row in calibration_bins(adjusted, labels, bins):
            row = {"model": model, "variant": "global_death_gate", **row}
            bin_rows.append(row)
    return out, bin_rows


def estimate_generated_diag_size(actual_diag_count: float, recall: float, jaccard: float) -> float:
    if actual_diag_count <= 0 or recall <= 0 or jaccard <= 0:
        return float("nan")
    intersection = recall * actual_diag_count
    return max(0.0, intersection / jaccard - actual_diag_count + intersection)


def jaccard_rows(patient_rows: Sequence[dict[str, str]]) -> list[dict]:
    out = []
    for model, rows in group_by_model(patient_rows).items():
        actual_counts = [safe_float(row["actual_future_diag_count"]) for row in rows]
        union_recall = [safe_float(row["diag_recall_union"]) for row in rows]
        mean_recall = [safe_float(row["diag_recall_mean_rollout"]) for row in rows]
        union_j = [safe_float(row["diag_jaccard_union"]) for row in rows]
        mean_j = [safe_float(row["diag_jaccard_mean_rollout"]) for row in rows]
        estimated_sizes = [
            estimate_generated_diag_size(a, r, j)
            for a, r, j in zip(actual_counts, union_recall, union_j)
        ]
        over_factors = [
            p / max(a, 1.0)
            for p, a in zip(estimated_sizes, actual_counts)
            if math.isfinite(p) and math.isfinite(a)
        ]
        false_positive_gap = [
            max(0.0, r - j)
            for r, j in zip(union_recall, union_j)
            if math.isfinite(r) and math.isfinite(j)
        ]
        out.append(
            {
                "model": model,
                "patients": len(rows),
                "actual_future_diag_count_mean": safe_mean(actual_counts),
                "diag_recall_union_mean": safe_mean(union_recall),
                "diag_recall_mean_rollout": safe_mean(mean_recall),
                "diag_jaccard_union_mean": safe_mean(union_j),
                "diag_jaccard_mean_rollout": safe_mean(mean_j),
                "estimated_generated_diag_set_size_mean": safe_mean(estimated_sizes),
                "estimated_diag_overgeneration_factor": safe_mean(over_factors),
                "oracle_future_set_gate_jaccard_ceiling": safe_mean(union_recall),
                "false_positive_jaccard_gap": safe_mean(false_positive_gap),
                "multi_rollout_pool_recall_gain": safe_mean(
                    [u - m for u, m in zip(union_recall, mean_recall) if math.isfinite(u) and math.isfinite(m)]
                ),
                "multi_rollout_pool_jaccard_gain": safe_mean(
                    [u - m for u, m in zip(union_j, mean_j) if math.isfinite(u) and math.isfinite(m)]
                ),
            }
        )
    return out


def priority_rows() -> list[dict]:
    return [
        {
            "priority": "P0",
            "task": "Validation-calibrated death gate + stop/censor rule",
            "status": "implemented as diagnostic; sampling hook added in rollout script",
            "why": "Death should be an absorbing survival endpoint, not an unconstrained ordinary token.",
            "deliverable": "death_gate_calibration_summary.csv; use --death-logit-bias for gated reruns",
        },
        {
            "priority": "P1",
            "task": "Future disease set auxiliary head",
            "status": "design specified; training branch next",
            "why": "Jaccard is dominated by false positives; a set head can constrain which diseases are plausible within 5y/10y.",
            "deliverable": "future multi-label BCE/Dice/Tversky objective on trunk state",
        },
        {
            "priority": "P2",
            "task": "Rollout reranking",
            "status": "diagnostic decomposition implemented",
            "why": "Multiple rollouts increase recall but can also inflate false positives; reranking should choose trajectories consistent with risk and set heads.",
            "deliverable": "rerank score = likelihood + disease-set consistency + horizon consistency - overgeneration penalty",
        },
        {
            "priority": "P3",
            "task": "Survival-derived death sampling",
            "status": "model-level path available via survival-horizon branch",
            "why": "Death generation should be sampled from calibrated hazard bins and terminate the sequence.",
            "deliverable": "replace raw death token sampling with survival hazard gate",
        },
        {
            "priority": "P4",
            "task": "Hierarchical Jaccard reporting",
            "status": "planned",
            "why": "Exact ICD-token Jaccard is too strict for clinically related alternatives.",
            "deliverable": "exact ICD, ICD-3/chapter, and disease-group Jaccard",
        },
    ]


def draw_branch_charts(out_dir: Path, death_rows: Sequence[dict], jaccard: Sequence[dict]) -> dict[str, str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return {}

    def font(size: int, bold: bool = False):
        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    def shorten(model: str) -> str:
        return (
            model.replace("MedTrajectory ", "MedTraj\n")
            .replace("Exp2 multitype internal anchor", "Exp2\ninternal")
            .replace(" pseudo-rollout", "\npseudo")
        )

    def grouped(path: Path, title: str, rows: Sequence[dict], keys: list[tuple[str, str]], ymax: float = 1.0):
        width, height = 1500, 850
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        title_font = font(34, True)
        label_font = font(18)
        value_font = font(16, True)
        draw.text((50, 30), title, fill=(11, 37, 69), font=title_font)
        left, right, top, bottom = 95, width - 55, 130, height - 180
        draw.line((left, bottom, right, bottom), fill=(80, 90, 100), width=2)
        draw.line((left, top, left, bottom), fill=(80, 90, 100), width=2)
        for i in range(6):
            val = ymax * i / 5
            y = bottom - (bottom - top) * val / max(ymax, 1e-9)
            draw.line((left, y, right, y), fill=(230, 235, 241), width=1)
            draw.text((left - 72, y - 10), f"{val:.2f}", fill=(91, 103, 112), font=label_font)
        colors = [(46, 116, 181), (188, 88, 88), (61, 153, 142)]
        group_w = (right - left) / max(1, len(rows))
        bar_w = min(52, (group_w - 34) / max(1, len(keys)))
        for mi, row in enumerate(rows):
            gx = left + mi * group_w + 18
            for ki, (key, _) in enumerate(keys):
                val = safe_float(row.get(key), 0.0)
                x0 = gx + ki * (bar_w + 10)
                x1 = x0 + bar_w
                y0 = bottom - (bottom - top) * val / max(ymax, 1e-9)
                draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=colors[ki % len(colors)])
                txt = f"{val:.3f}"
                box = draw.textbbox((0, 0), txt, font=value_font)
                draw.text((x0 + (bar_w - (box[2] - box[0])) / 2, y0 - 24), txt, fill=(11, 37, 69), font=value_font)
            for li, line in enumerate(shorten(row["model"]).split("\n")):
                box = draw.textbbox((0, 0), line, font=label_font)
                draw.text((gx + 38 - (box[2] - box[0]) / 2, bottom + 18 + li * 22), line, fill=(40, 48, 58), font=label_font)
        legend_x = 80
        for ki, (_, label) in enumerate(keys):
            x = legend_x + ki * 360
            draw.rounded_rectangle((x, height - 62, x + 28, height - 44), radius=4, fill=colors[ki % len(colors)])
            draw.text((x + 38, height - 64), label, fill=(40, 48, 58), font=label_font)
        img.save(path)

    charts = {
        "death_gate_chart": str(out_dir / "death_gate_before_after.png"),
        "jaccard_ceiling_chart": str(out_dir / "jaccard_current_vs_ceiling.png"),
    }
    grouped(
        Path(charts["death_gate_chart"]),
        "Death probability before/after global survival gate",
        death_rows,
        [
            ("raw_predicted_death_prob_mean", "raw generated death prob"),
            ("gated_predicted_death_prob_mean", "gated death prob"),
            ("observed_death_rate", "observed death rate"),
        ],
        ymax=1.0,
    )
    grouped(
        Path(charts["jaccard_ceiling_chart"]),
        "Diagnosis Jaccard and future-set gate ceiling",
        jaccard,
        [
            ("diag_jaccard_union_mean", "current union Jaccard"),
            ("oracle_future_set_gate_jaccard_ceiling", "set-gated ceiling"),
            ("diag_recall_union_mean", "future disease recall"),
        ],
        ymax=1.0,
    )
    return charts


def markdown_table(rows: Sequence[dict], columns: Sequence[str], digits: int = 4) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.{digits}f}" if math.isfinite(value) else "")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, death_rows: Sequence[dict], jaccard: Sequence[dict], priority: Sequence[dict], charts: dict[str, str]) -> None:
    lines = [
        "# Survival-constrained trajectory generation branch",
        "",
        "## Purpose",
        "",
        "This innovation branch addresses two rollout weaknesses: low exact diagnosis Jaccard and over-generated death events. It keeps the existing MedTrajectory trunk, then adds survival constraints and future-disease-set constraints around autoregressive rollout.",
        "",
        "Important boundary: when no external validation death rate is supplied, the death gate uses the observed rollout cohort death rate as an oracle diagnostic. That quantifies the calibration problem and the expected gain, but it should not be reported as a locked-test final result.",
        "",
        "## Priority order",
        "",
        markdown_table(priority, ["priority", "task", "status", "deliverable"], digits=4),
        "",
        "## P0 death gate diagnostic",
        "",
        markdown_table(
            death_rows,
            [
                "model",
                "observed_death_rate",
                "raw_predicted_death_prob_mean",
                "death_gate_scale",
                "suggested_death_logit_bias",
                "gated_predicted_death_prob_mean",
                "raw_death_brier",
                "gated_death_brier",
                "raw_death_ece",
                "gated_death_ece",
            ],
        ),
        "",
        "## P1/P2 Jaccard decomposition",
        "",
        markdown_table(
            jaccard,
            [
                "model",
                "diag_jaccard_union_mean",
                "diag_recall_union_mean",
                "estimated_diag_overgeneration_factor",
                "oracle_future_set_gate_jaccard_ceiling",
                "false_positive_jaccard_gap",
                "multi_rollout_pool_recall_gain",
            ],
        ),
        "",
        "## Figures",
        "",
    ]
    if charts:
        lines += [
            f"![Death gate before/after]({Path(charts['death_gate_chart']).name})",
            "",
            f"![Jaccard current vs ceiling]({Path(charts['jaccard_ceiling_chart']).name})",
            "",
        ]
    lines += [
        "## Next executable steps",
        "",
        "1. Rerun rollout with a validation-selected `--death-logit-bias` to validate death-gated sampling without using test labels.",
        "2. Train a future disease set auxiliary head with multi-label BCE plus Dice/Tversky loss.",
        "3. Use the set head and horizon-risk head to rerank multiple rollouts.",
        "4. Replace raw death token sampling with survival-horizon hazard sampling.",
        "5. Add exact ICD, ICD-3/chapter, and disease-group Jaccard to the benchmark.",
        "",
    ]
    (out_dir / "SURVIVAL_CONSTRAINED_GENERATION_BRANCH.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patient_path = args.rollout_dir / "rollout_patient_metrics.csv"
    if not patient_path.exists():
        raise FileNotFoundError(patient_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patient_rows = read_csv(patient_path)
    death_rows, bin_rows = death_gate_rows(patient_rows, args.target_death_rate, args.bins)
    jaccard = jaccard_rows(patient_rows)
    priority = priority_rows()
    charts = draw_branch_charts(args.out_dir, death_rows, jaccard)
    write_csv(args.out_dir / "death_gate_calibration_summary.csv", death_rows)
    write_csv(args.out_dir / "death_gate_calibration_bins.csv", bin_rows)
    write_csv(args.out_dir / "jaccard_decomposition_summary.csv", jaccard)
    write_csv(args.out_dir / "priority_plan.csv", priority)
    write_report(args.out_dir, death_rows, jaccard, priority, charts)
    config = {
        "rollout_dir": str(args.rollout_dir),
        "out_dir": str(args.out_dir),
        "target_death_rate": args.target_death_rate,
        "death_gate_target_source": "external_or_validation_target" if args.target_death_rate is not None else "oracle_test_diagnostic",
        "bins": args.bins,
        "charts": charts,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
