"""Plot Figure 5A / 5B: broad-spectrum disease-prediction landscape.

5A: macro AUROC vs log10(training cases), colored by official ICD-10 chapter,
    point size ~ validation cases, y=0.5 reference line.
5B: macro AUROC distribution (box + strip) by ICD-10 chapter.

Inputs (server paths):
  results/track_r_v2_2/runs/seed42_additiverope1/risk/A2_expanded38/validation/disease_level/disease_macro_risk.csv
  results/expanded_disease_panel/expanded_disease_panel_counts_wide.csv

Usage: python scripts/plot_disease_landscape.py --out results/expanded_disease_panel_figures/
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# official ICD-10 chapter roman -> (numeric order, short label)
CHAPTER_ORDER = {
    "I": (1, "I Infectious"),
    "II": (2, "II Neoplasms"),
    "III": (3, "III Blood"),
    "IV": (4, "IV Endocrine/metabolic"),
    "V": (5, "V Mental"),
    "VI": (6, "VI Nervous"),
    "VII": (7, "VII Eye"),
    "VIII": (8, "VIII Ear"),
    "IX": (9, "IX Circulatory"),
    "X": (10, "X Respiratory"),
    "XI": (11, "XI Digestive"),
    "XII": (12, "XII Skin"),
    "XIII": (13, "XIII Musculoskeletal"),
    "XIV": (14, "XIV Genitourinary"),
}


def read_macro(path: Path) -> dict:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["macro_auc"] = float(r["macro_auc"])
            r["macro_auc_ci_low"] = _f(r["macro_auc_ci_low"])
            r["macro_auc_ci_high"] = _f(r["macro_auc_ci_high"])
            r["n_positives_sum"] = int(r["n_positives_sum"])
            rows[r["disease_id"]] = r
    return rows


def read_counts(path: Path) -> dict:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["disease_id"]] = r
    return rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--macro", type=Path,
                    default=Path("results/track_r_v2_2/runs/seed42_additiverope1/risk/A2_expanded38/validation/disease_level/disease_macro_risk.csv"))
    ap.add_argument("--counts", type=Path,
                    default=Path("results/expanded_disease_panel/expanded_disease_panel_counts_wide.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/expanded_disease_panel_figures"))
    args = ap.parse_args()

    macro = read_macro(args.macro)
    counts = read_counts(args.counts)
    args.out.mkdir(parents=True, exist_ok=True)

    # join
    recs = []
    for did, m in macro.items():
        c = counts.get(did)
        if c is None:
            continue
        train_cases = _f(c["train_ever_positive"])
        val_cases = _f(c["val_ever_positive"])
        test_cases = _f(c["test_ever_positive"])
        if not np.isfinite(train_cases) or train_cases <= 0:
            continue
        chapter = m.get("chapter", "?")
        recs.append({
            "disease_id": did,
            "name": m["name_cn"] or m["disease_name"],
            "name_en": m["disease_name"],
            "chapter": chapter,
            "macro_auc": m["macro_auc"],
            "ci_low": m["macro_auc_ci_low"],
            "ci_high": m["macro_auc_ci_high"],
            "train_cases": train_cases,
            "val_cases": val_cases,
            "test_cases": test_cases,
        })
    recs.sort(key=lambda r: (CHAPTER_ORDER.get(r["chapter"], (99, ""))[0], r["macro_auc"]))

    chapters_present = list(OrderedDict((r["chapter"], True) for r in recs).keys())
    # stable color per chapter
    cmap = plt.get_cmap("tab10")
    chapter_color = {ch: cmap(i % 10) for i, ch in enumerate(chapters_present)}

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    # ---- Figure 5A: scatter ----
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for r in recs:
        x = np.log10(r["train_cases"])
        y = r["macro_auc"]
        size = np.clip(r["val_cases"], 5, 3000)
        col = chapter_color[r["chapter"]]
        ax.scatter(x, y, s=size * 0.04, c=[col], alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
        # error bars (subtle)
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
            ax.plot([x, x], [r["ci_low"], r["ci_high"]], color=col, alpha=0.35, lw=0.8, zorder=2)
    ax.axhline(0.5, color="0.4", ls="--", lw=1, zorder=1)
    ax.set_xlabel("Training cases (log10)")
    ax.set_ylabel("Macro AUROC (validation, 1/5/10-y pooled)")
    ax.set_title("A  Disease prediction AUROC vs training frequency")
    ax.set_ylim(0.45, 0.90)

    # annotate extremes + headline
    headline_ids = {"ischemic_heart_disease", "myocardial_infarction", "hypertension",
                    "diabetes_mellitus", "chronic_kidney_disease"}
    for r in recs:
        if r["disease_id"] in headline_ids or r["macro_auc"] >= 0.838 or r["macro_auc"] <= 0.690:
            ax.annotate(r["name_en"], (np.log10(r["train_cases"]), r["macro_auc"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7.5, color="0.15")

    # legend for chapters
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=chapter_color[ch],
                          label=CHAPTER_ORDER.get(ch, (99, ch))[1]) for ch in chapters_present]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out / "figure5a_auc_vs_frequency.png", dpi=200)
    plt.close(fig)

    # ---- Figure 5B: box + strip by chapter ----
    chapter_list = sorted(chapters_present, key=lambda ch: CHAPTER_ORDER.get(ch, (99, ""))[0])
    data_by_ch = {ch: [r["macro_auc"] for r in recs if r["chapter"] == ch] for ch in chapter_list}
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    xs, labels = [], []
    for i, ch in enumerate(chapter_list):
        vals = data_by_ch[ch]
        x = i + 1
        labels.append(CHAPTER_ORDER.get(ch, (99, ch))[1])
        # strip points with jitter
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=22, c=[chapter_color[ch]], alpha=0.8,
                   edgecolors="white", linewidths=0.5, zorder=3)
        if len(vals) >= 3:
            bp = ax.boxplot(vals, positions=[x], widths=0.45, showfliers=False,
                            patch_artist=True, zorder=2)
            for patch in bp["boxes"]:
                patch.set_facecolor(chapter_color[ch])
                patch.set_alpha(0.25)
            for med in bp["medians"]:
                med.set_color("black")
        else:
            ax.plot([x], [np.median(vals)], marker="_", ms=18, color="black", zorder=4)
        xs.append(x)
    ax.axhline(0.5, color="0.4", ls="--", lw=1, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Macro AUROC (validation, 1/5/10-y pooled)")
    ax.set_title("B  AUROC distribution by ICD-10 chapter")
    ax.set_ylim(0.45, 0.90)
    fig.tight_layout()
    fig.savefig(args.out / "figure5b_auc_by_chapter.png", dpi=200)
    plt.close(fig)

    print(f"[wrote] {args.out / 'figure5a_auc_vs_frequency.png'}")
    print(f"[wrote] {args.out / 'figure5b_auc_by_chapter.png'}")
    print(f"[info] {len(recs)} diseases plotted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
