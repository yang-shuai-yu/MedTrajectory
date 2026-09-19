from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_DIR / "results" / "expanded_disease_panel_figures"
PREFERRED_HORIZON_MODEL = "MedTrajectory monotonic 38d full3000"
FALLBACK_HORIZON_MODEL = "MedTrajectory monotonic 38d smoke1000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create lightweight figures for expanded disease panel presentation.")
    parser.add_argument("--topk-csv", type=Path, default=REPO_DIR / "results" / "expanded_disease_panel_topk" / "expanded_disease_topk_comparison.csv")
    parser.add_argument("--horizon-tier-csv", type=Path, default=REPO_DIR / "results" / "expanded_horizon_risk" / "comparison" / "expanded_horizon_risk_tier_summary.csv")
    parser.add_argument("--counts-wide", type=Path, default=REPO_DIR / "results" / "expanded_disease_panel" / "expanded_disease_panel_counts_wide.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def bar_chart(path: Path, title: str, subtitle: str, values: list[tuple[str, float]], y_max: float = 1.0) -> None:
    w, h = 1400, 820
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = font(38, True)
    sub_font = font(21)
    tick_font = font(19)
    val_font = font(20, True)
    d.text((60, 34), title, font=title_font, fill=(11, 37, 69))
    d.text((60, 86), subtitle, font=sub_font, fill=(91, 103, 112))
    left, right, top, bottom = 105, w - 50, 145, h - 160
    d.line((left, bottom, right, bottom), fill=(65, 75, 90), width=2)
    d.line((left, top, left, bottom), fill=(65, 75, 90), width=2)
    for i in range(6):
        val = y_max * i / 5
        y = bottom - (bottom - top) * val / y_max
        d.line((left, y, right, y), fill=(229, 234, 241), width=1)
        label = f"{val:.1f}"
        tw = d.textlength(label, font=tick_font)
        d.text((left - tw - 14, y - 10), label, font=tick_font, fill=(91, 103, 112))
    colors = [(44, 123, 182), (0, 166, 147), (247, 160, 77), (190, 87, 88), (115, 128, 150), (112, 91, 160)]
    n = len(values)
    gap = 34
    bar_w = (right - left - gap * (n + 1)) / max(1, n)
    for idx, (label, val) in enumerate(values):
        x0 = left + gap + idx * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = bottom - (bottom - top) * min(val, y_max) / y_max
        d.rounded_rectangle((x0, y0, x1, bottom), radius=7, fill=colors[idx % len(colors)])
        txt = f"{val:.4f}"
        tw = d.textlength(txt, font=val_font)
        d.text((x0 + (bar_w - tw) / 2, y0 - 30), txt, font=val_font, fill=(11, 37, 69))
        y = bottom + 20
        for line in label.split("\n"):
            tw = d.textlength(line, font=tick_font)
            d.text((x0 + (bar_w - tw) / 2, y), line, font=tick_font, fill=(40, 48, 58))
            y += 24
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def grouped_bar_chart(path: Path, title: str, subtitle: str, groups: list[str], series: list[str], values: list[list[float]]) -> None:
    w, h = 1450, 850
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_font = font(38, True)
    sub_font = font(21)
    tick_font = font(18)
    val_font = font(17, True)
    d.text((60, 34), title, font=title_font, fill=(11, 37, 69))
    d.text((60, 86), subtitle, font=sub_font, fill=(91, 103, 112))
    left, right, top, bottom = 105, w - 50, 150, h - 170
    d.line((left, bottom, right, bottom), fill=(65, 75, 90), width=2)
    d.line((left, top, left, bottom), fill=(65, 75, 90), width=2)
    for i in range(6):
        val = i / 5
        y = bottom - (bottom - top) * val
        d.line((left, y, right, y), fill=(229, 234, 241), width=1)
        label = f"{val:.1f}"
        tw = d.textlength(label, font=tick_font)
        d.text((left - tw - 14, y - 10), label, font=tick_font, fill=(91, 103, 112))
    colors = [(44, 123, 182), (0, 166, 147), (247, 160, 77)]
    legend_x = right - 360
    for si, name in enumerate(series):
        y = 86 + si * 30
        d.rounded_rectangle((legend_x, y, legend_x + 28, y + 18), radius=4, fill=colors[si])
        d.text((legend_x + 40, y - 4), name, font=tick_font, fill=(40, 48, 58))
    group_gap = 45
    group_w = (right - left - group_gap * (len(groups) + 1)) / len(groups)
    bar_gap = 8
    bar_w = (group_w - bar_gap * (len(series) - 1)) / len(series)
    for gi, group in enumerate(groups):
        gx = left + group_gap + gi * (group_w + group_gap)
        for si, val in enumerate(values[gi]):
            x0 = gx + si * (bar_w + bar_gap)
            x1 = x0 + bar_w
            y0 = bottom - (bottom - top) * val
            d.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=colors[si])
            txt = f"{val:.3f}"
            tw = d.textlength(txt, font=val_font)
            d.text((x0 + (bar_w - tw) / 2, y0 - 24), txt, font=val_font, fill=(11, 37, 69))
        y = bottom + 20
        for line in group.split("\n"):
            tw = d.textlength(line, font=tick_font)
            d.text((gx + (group_w - tw) / 2, y), line, font=tick_font, fill=(40, 48, 58))
            y += 23
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def choose_horizon_model(horizon_rows: list[dict]) -> str:
    available = {row["model"] for row in horizon_rows}
    if PREFERRED_HORIZON_MODEL in available:
        return PREFERRED_HORIZON_MODEL
    if FALLBACK_HORIZON_MODEL in available:
        return FALLBACK_HORIZON_MODEL
    raise KeyError(
        f"Neither {PREFERRED_HORIZON_MODEL!r} nor {FALLBACK_HORIZON_MODEL!r} "
        "is present in the horizon tier summary."
    )


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    topk = read_csv(args.topk_csv)
    horizon = read_csv(args.horizon_tier_csv)
    counts = read_csv(args.counts_wide)

    overall_topk = [
        row for row in topk
        if row["disease_id"] == "overall_panel_targets"
    ]
    order = [
        "Monotonic Gated RoPE main",
        "Gated RoPE main",
        "Old TTE+RoPE main",
        "No-RoPE previous best",
        "Mamba next-event no-leak",
        "BERT next-event no-leak",
    ]
    by_model = {row["model"]: row for row in overall_topk}
    bar_chart(
        args.out_dir / "expanded_panel_group_top10_overall.png",
        "Expanded Disease Panel: Diagnosis Group Top10",
        "n=582 next-diagnosis targets from evidence-based ICD-10 disease panel",
        [(name.replace(" ", "\n", 2), float(by_model[name]["diagnosis_group_top10_hit_rate"])) for name in order if name in by_model],
    )

    med = [r for r in topk if r["model"] == "Monotonic Gated RoPE main" and r["reporting_tier"] == "headline"]
    med.sort(key=lambda r: float(r["diagnosis_group_top10_hit_rate"]), reverse=True)
    bar_chart(
        args.out_dir / "headline_disease_group_top10_medtrajectory.png",
        "Headline Diseases: MedTrajectory Group Top10",
        "Disease-group hit means any ICD-10 token from the same disease group appears in diagnosis Top10",
        [(r["name"].replace(" ", "\n", 2), float(r["diagnosis_group_top10_hit_rate"])) for r in med],
    )

    key_rows = {}
    for row in horizon:
        if row["horizon_years"] == "overall" and row["reporting_tier"] in {"headline", "headline_plus_exploratory"}:
            key_rows[(row["model"], row["reporting_tier"])] = row
    horizon_model = choose_horizon_model(horizon)
    groups = ["Headline\n5 diseases", "Headline +\nExploratory\n18 diseases"]
    series = ["AUC", "AP", "Top-decile capture"]
    values = []
    for tier in ["headline", "headline_plus_exploratory"]:
        row = key_rows[(horizon_model, tier)]
        values.append([
            float(row["auc_mean"]),
            float(row["average_precision_mean"]),
            float(row["top_decile_capture_mean"]),
        ])
    grouped_bar_chart(
        args.out_dir / "expanded_horizon_medtrajectory_tier_metrics.png",
        "Expanded Horizon Risk: MedTrajectory",
        f"Locked test; {horizon_model}; final-context 5y/10y average",
        groups,
        series,
        values,
    )

    headline = [r for r in counts if r["reporting_tier"] == "headline"]
    headline.sort(key=lambda r: int(r["test_5y_final_context_positives"]), reverse=True)
    bar_chart(
        args.out_dir / "headline_disease_test_positives.png",
        "Headline Disease Test Positives",
        "Disease groups with >=20 locked-test positives in both 5y and 10y labels",
        [(r["name"].replace(" ", "\n", 2), float(r["test_5y_final_context_positives"])) for r in headline],
        y_max=max(float(r["test_5y_final_context_positives"]) for r in headline) * 1.15,
    )
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
