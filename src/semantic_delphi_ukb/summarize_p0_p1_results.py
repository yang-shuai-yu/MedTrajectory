from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence


REPO_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize P0 death-gate and P1 future-set innovation results.")
    parser.add_argument("--root", type=Path, default=REPO_DIR)
    parser.add_argument("--out-doc", type=Path, default=REPO_DIR / "docs" / "P0_P1_INNOVATION_RESULTS_CN.md")
    parser.add_argument("--p0-dir", type=Path, default=REPO_DIR / "results" / "survival_constrained_generation_p0")
    parser.add_argument("--p1-dir", type=Path, default=REPO_DIR / "results" / "medtrajectory_future_set_head")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: object, digits: int = 4) -> str:
    out = safe_float(value)
    return f"{out:.{digits}f}" if math.isfinite(out) else ""


def collect_p0(p0_dir: Path) -> tuple[list[dict], dict | None, list[dict]]:
    tags = [("z0", 0.0), ("m1", -1.0), ("m1p5", -1.5), ("m2", -2.0), ("m2p5", -2.5), ("m3", -3.0)]
    rows = []
    for tag, bias in tags:
        path = p0_dir / f"val_bias_{tag}" / "rollout_summary.csv"
        if not path.exists():
            continue
        row = read_csv(path)[0]
        rows.append({"split": "val", "bias": bias, "tag": tag, **row})
    best = min(
        rows,
        key=lambda row: (
            safe_float(row.get("death_brier_mean")),
            safe_float(row.get("death_ece")),
            -safe_float(row.get("diag_recall_union_mean")),
        ),
    ) if rows else None
    test_rows = []
    for name in ["test_bias_m3_locked_nomamba", "test_bias_m3_locked_mamba"]:
        test_path = p0_dir / name / "rollout_summary.csv"
        if test_path.exists():
            test_rows.extend(read_csv(test_path))
    return rows, best, test_rows


def collect_p1(p1_dir: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    history_300 = read_json(p1_dir / "smoke_300" / "history.json") if (p1_dir / "smoke_300" / "history.json").exists() else []
    history_1000 = read_json(p1_dir / "smoke_1000" / "history.json") if (p1_dir / "smoke_1000" / "history.json").exists() else []
    test_rows = read_json(p1_dir / "smoke_1000_test" / "future_set_metrics_rows.json") if (p1_dir / "smoke_1000_test" / "future_set_metrics_rows.json").exists() else []
    test_summary = read_json(p1_dir / "smoke_1000_test" / "eval_summary.json") if (p1_dir / "smoke_1000_test" / "eval_summary.json").exists() else {}
    return history_300, history_1000, test_rows, test_summary


def chart_assets_dir(root: Path) -> Path:
    out = root / "docs" / "report_assets"
    out.mkdir(parents=True, exist_ok=True)
    return out


def draw_line_chart(path: Path, title: str, series: list[tuple[str, list[tuple[float, float]]]], y_label: str = "") -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

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

    clean = [(label, [(x, y) for x, y in points if math.isfinite(y)]) for label, points in series]
    all_x = [x for _, points in clean for x, _ in points]
    all_y = [y for _, points in clean for _, y in points]
    if not all_x or not all_y:
        return
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(0.0, min(all_y)), max(all_y)
    if max_y <= min_y:
        max_y = min_y + 1.0
    max_y *= 1.08

    width, height = 1400, 820
    left, right, top, bottom = 110, width - 70, 115, height - 145
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(18)
    value_font = font(17, True)
    draw.text((48, 30), title, fill=(18, 35, 59), font=title_font)
    if y_label:
        draw.text((48, 78), y_label, fill=(85, 96, 109), font=label_font)
    draw.line((left, bottom, right, bottom), fill=(86, 96, 106), width=2)
    draw.line((left, top, left, bottom), fill=(86, 96, 106), width=2)
    for i in range(6):
        value = min_y + (max_y - min_y) * i / 5
        y = bottom - (bottom - top) * (value - min_y) / (max_y - min_y)
        draw.line((left, y, right, y), fill=(228, 233, 239), width=1)
        draw.text((left - 82, y - 10), f"{value:.2f}", fill=(85, 96, 109), font=label_font)

    colors = [(46, 116, 181), (61, 153, 142), (188, 88, 88), (235, 154, 76)]
    for idx, (label, points) in enumerate(clean):
        if not points:
            continue
        color = colors[idx % len(colors)]
        xy = []
        for x, yv in points:
            px = left + (right - left) * (x - min_x) / max(max_x - min_x, 1e-9)
            py = bottom - (bottom - top) * (yv - min_y) / max(max_y - min_y, 1e-9)
            xy.append((px, py))
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=4)
        for px, py in xy:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        lx = 90 + idx * 310
        ly = height - 70
        draw.rounded_rectangle((lx, ly, lx + 28, ly + 18), radius=4, fill=color)
        draw.text((lx + 40, ly - 3), label, fill=(45, 54, 64), font=value_font)
    img.save(path)


def draw_bar_chart(path: Path, title: str, labels: list[str], metrics: list[tuple[str, list[float]]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

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

    values = [value for _, vals in metrics for value in vals if math.isfinite(value)]
    if not values:
        return
    ymax = max(values) * 1.15
    if ymax <= 0:
        ymax = 1.0
    width, height = 1400, 820
    left, right, top, bottom = 105, width - 65, 120, height - 170
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(18)
    value_font = font(16, True)
    draw.text((48, 35), title, fill=(18, 35, 59), font=title_font)
    draw.line((left, bottom, right, bottom), fill=(86, 96, 106), width=2)
    draw.line((left, top, left, bottom), fill=(86, 96, 106), width=2)
    for i in range(6):
        value = ymax * i / 5
        y = bottom - (bottom - top) * value / ymax
        draw.line((left, y, right, y), fill=(228, 233, 239), width=1)
        draw.text((left - 82, y - 10), f"{value:.2f}", fill=(85, 96, 109), font=label_font)
    colors = [(46, 116, 181), (61, 153, 142), (188, 88, 88), (235, 154, 76)]
    group_w = (right - left) / max(len(labels), 1)
    bar_w = min(54, (group_w - 34) / max(len(metrics), 1))
    for i, label in enumerate(labels):
        gx = left + i * group_w + 24
        for j, (_, vals) in enumerate(metrics):
            value = vals[i] if i < len(vals) else float("nan")
            if not math.isfinite(value):
                continue
            x0 = gx + j * (bar_w + 10)
            x1 = x0 + bar_w
            y0 = bottom - (bottom - top) * value / ymax
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=colors[j % len(colors)])
            txt = f"{value:.3f}"
            box = draw.textbbox((0, 0), txt, font=value_font)
            draw.text((x0 + (bar_w - (box[2] - box[0])) / 2, y0 - 24), txt, fill=(18, 35, 59), font=value_font)
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((gx + 55 - (box[2] - box[0]) / 2, bottom + 20), label, fill=(45, 54, 64), font=label_font)
    for j, (name, _) in enumerate(metrics):
        lx = 90 + j * 300
        ly = height - 70
        draw.rounded_rectangle((lx, ly, lx + 28, ly + 18), radius=4, fill=colors[j % len(colors)])
        draw.text((lx + 40, ly - 3), name, fill=(45, 54, 64), font=value_font)
    img.save(path)


def write_report(path: Path, root: Path, p0_rows: list[dict], p0_best: dict | None, p0_test: list[dict], history_1000: list[dict], p1_test_rows: list[dict], p1_summary: dict, charts: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P0/P1 创新分支执行结果",
        "",
        "本文档汇总两个最高优先级任务：P0 survival/death-gate 校准，以及 P1 future disease-set auxiliary head。所有 P0 参数选择只使用 validation split；locked test 只用于最终报告。",
        "",
        "## P0：validation-calibrated death gate",
        "",
        "### 验证集 bias sweep",
        "",
        "| Bias | Pred death | Actual death | Brier | ECE | Disease recall | Diagnosis Jaccard | Mean length |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p0_rows:
        lines.append(
            f"| {fmt(row['bias'], 1)} | {fmt(row['predicted_death_prob_mean'])} | {fmt(row['actual_death_rate'])} | {fmt(row['death_brier_mean'])} | {fmt(row['death_ece'])} | {fmt(row['diag_recall_union_mean'])} | {fmt(row['diag_jaccard_mean_rollout'])} | {fmt(row['generated_length_mean'])} |"
        )
    if p0_best:
        lines += [
            "",
            f"选择规则：先最小化 validation death Brier，再比较 ECE，最后保留更高 disease recall。由此锁定 `--death-logit-bias {fmt(p0_best['bias'], 1)}`。",
            "",
        ]
    lines += [
        "### Locked test 结果",
        "",
        "| Model | Disease recall | Diagnosis Jaccard | Top-10 first disease | Pred death | Actual death | Death Brier | Death ECE | Diagnosis JS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p0_test:
        lines.append(
            f"| {row['model']} | {fmt(row['diag_recall_union_mean'])} | {fmt(row['diag_jaccard_mean_rollout'])} | {fmt(row['top10_first_diag_hit_rate'])} | {fmt(row['predicted_death_prob_mean'])} | {fmt(row['actual_death_rate'])} | {fmt(row['death_brier_mean'])} | {fmt(row['death_ece'])} | {fmt(row['diagnosis_token_js_divergence'])} |"
        )
    lines += [
        "",
        "Mamba 说明：历史 Mamba baseline 和本次 P0 Mamba rollout 均使用独立 Python 3.10 Mamba 环境运行；默认 `ysy_delphi` 环境没有 `mamba_ssm`，不能直接加载 Mamba baseline。因此复现实验时需要显式切换到 Mamba 环境。",
        "",
    ]
    if charts.get("p0"):
        rel = charts["p0"].relative_to(path.parent)
        lines += [f"![P0 death-gate sweep]({rel.as_posix()})", ""]

    lines += [
        "## P1：future disease-set auxiliary head",
        "",
        "实现内容：在 MedTrajectory trunk 最终状态上增加 5y/10y 多标签疾病集合预测头，损失为 BCE + positive weight + Dice。它当前定位是 rollout 过滤器/重排序器的基础，不是已经替换主生成模型的最终版本。",
        "",
        "### Validation 学习曲线（1000-step run）",
        "",
        "| Iter | 5y micro-AUC | 5y Recall@3 | 5y Jaccard@3 | 10y micro-AUC | 10y Recall@3 | 10y Jaccard@3 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history_1000:
        lines.append(
            f"| {int(row['iter'])} | {fmt(row.get('h5_micro_auc'))} | {fmt(row.get('h5_recall_at_3'))} | {fmt(row.get('h5_jaccard_at_3'))} | {fmt(row.get('h10_micro_auc'))} | {fmt(row.get('h10_recall_at_3'))} | {fmt(row.get('h10_jaccard_at_3'))} |"
        )
    lines += [
        "",
        f"Best checkpoint 来自 iteration `{p1_summary.get('iter_num', '')}`，validation best 10y Jaccard@3 = `{fmt(p1_summary.get('best_val_future_set_score'))}`。",
        "",
        "### Locked test 结果",
        "",
        "| Horizon | Micro-AUC | Recall@3 | Precision@3 | Jaccard@3 | Recall@5 | Jaccard@5 | Positive rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p1_test_rows:
        lines.append(
            f"| {fmt(row['horizon_years'], 0)}y | {fmt(row['micro_auc'])} | {fmt(row['recall_at_3'])} | {fmt(row['precision_at_3'])} | {fmt(row['jaccard_at_3'])} | {fmt(row['recall_at_5'])} | {fmt(row['jaccard_at_5'])} | {fmt(row['positive_rate'])} |"
        )
    lines += [
        "",
        "解释：P1 在 test 上 micro-AUC 约 0.882，说明未来疾病集合排序信号存在；但 precision/Jaccard 很低，主要因为正例率只有约 0.27%-0.29%。因此它适合进入下一步 rollout reranking / soft constraint，而不是现在直接宣称为主生成模型。",
        "",
    ]
    if charts.get("p1_curve"):
        rel = charts["p1_curve"].relative_to(path.parent)
        lines += [f"![P1 validation learning curve]({rel.as_posix()})", ""]
    if charts.get("p1_test"):
        rel = charts["p1_test"].relative_to(path.parent)
        lines += [f"![P1 locked test metrics]({rel.as_posix()})", ""]

    lines += [
        "## 当前结论",
        "",
        "1. P0 已完成：`bias=-3.0` 是 validation-locked 的 survival gate 参数，locked test 上 MedTrajectory death probability 接近真实死亡率，同时 disease recall 高于 Exp2/BERT/Mamba。",
        "2. P1 已完成 smoke + locked-test 验证：future-set head 学到了未来疾病集合排序，但还需要 threshold calibration 和 rollout reranking 才能真正改善生成 Jaccard。",
        "3. 下一步优先级：保留 Mamba 独立环境复现实验记录，并在更大 patient/rollout 设置下复核；把 future-set score 接入 rollout reranker；增加 hierarchical Jaccard（ICD exact / ICD-3 / disease group）来减少过严 exact-token 指标造成的误读。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    root = args.root
    p0_rows, p0_best, p0_test = collect_p0(args.p0_dir)
    history_300, history_1000, p1_test_rows, p1_summary = collect_p1(args.p1_dir)
    _ = history_300

    write_csv(args.p0_dir / "p0_bias_sweep_summary.csv", p0_rows)
    if p0_best:
        (args.p0_dir / "p0_validation_selection.json").write_text(
            json.dumps(
                {
                    "selection_split": "val",
                    "selection_rule": "min death_brier_mean, then death_ece, then higher diag_recall_union_mean",
                    "selected": p0_best,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    assets = chart_assets_dir(root)
    charts = {
        "p0": assets / "p0_death_gate_sweep.png",
        "p1_curve": assets / "p1_future_set_learning_curve.png",
        "p1_test": assets / "p1_future_set_locked_test.png",
    }
    draw_line_chart(
        charts["p0"],
        "P0 validation death-gate sweep",
        [
            ("pred death", [(safe_float(row["bias"]), safe_float(row["predicted_death_prob_mean"])) for row in p0_rows]),
            ("actual death", [(safe_float(row["bias"]), safe_float(row["actual_death_rate"])) for row in p0_rows]),
            ("death brier", [(safe_float(row["bias"]), safe_float(row["death_brier_mean"])) for row in p0_rows]),
            ("death ece", [(safe_float(row["bias"]), safe_float(row["death_ece"])) for row in p0_rows]),
        ],
        y_label="Validation split; selected by Brier/ECE",
    )
    draw_line_chart(
        charts["p1_curve"],
        "P1 future-set validation learning curve",
        [
            ("5y micro-AUC", [(safe_float(row["iter"]), safe_float(row.get("h5_micro_auc"))) for row in history_1000]),
            ("10y micro-AUC", [(safe_float(row["iter"]), safe_float(row.get("h10_micro_auc"))) for row in history_1000]),
            ("5y Jaccard@3", [(safe_float(row["iter"]), safe_float(row.get("h5_jaccard_at_3"))) for row in history_1000]),
            ("10y Jaccard@3", [(safe_float(row["iter"]), safe_float(row.get("h10_jaccard_at_3"))) for row in history_1000]),
        ],
        y_label="Best checkpoint selected by validation 10y Jaccard@3",
    )
    draw_bar_chart(
        charts["p1_test"],
        "P1 future-set locked-test metrics",
        [f"{fmt(row['horizon_years'], 0)}y" for row in p1_test_rows],
        [
            ("micro-AUC", [safe_float(row["micro_auc"]) for row in p1_test_rows]),
            ("recall@3", [safe_float(row["recall_at_3"]) for row in p1_test_rows]),
            ("jaccard@3", [safe_float(row["jaccard_at_3"]) for row in p1_test_rows]),
        ],
    )
    write_report(args.out_doc, root, p0_rows, p0_best, p0_test, history_1000, p1_test_rows, p1_summary, charts)
    print(args.out_doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
