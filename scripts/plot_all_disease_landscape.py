"""Plot the 1000+ disease LM disease-onset landscape, aligned to Delphi-2M Fig 2b/2c.

A: age-sex-stratified AUC vs log10(training occurrences), colored by ICD-10 chapter
   (Delphi-2M Fig 2b style: AUC vs training frequency).
B: age-sex-stratified AUC distribution (box + strip) by ICD-10 chapter (Fig 2c).

Input: results/all_disease_lm/all_disease_lm_auc.csv (from evaluate_all_disease_lm.py).

Usage: python scripts/plot_all_disease_landscape.py --min-cases 20
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

CHAPTER_ORDER = {
    "I": (1, "I Infectious"), "II": (2, "II Neoplasms"), "III": (3, "III Blood"),
    "IV": (4, "IV Endocrine"), "V": (5, "V Mental"), "VI": (6, "VI Nervous"),
    "VII": (7, "VII Eye"), "VIII": (8, "VIII Ear"), "IX": (9, "IX Circulatory"),
    "X": (10, "X Respiratory"), "XI": (11, "XI Digestive"), "XII": (12, "XII Skin"),
    "XIII": (13, "XIII Musculoskeletal"), "XIV": (14, "XIV Genitourinary"),
    "XV": (15, "XV Pregnancy"), "XVI": (16, "XVI Perinatal"),
    "XVII": (17, "XVII Congenital"), "XVIII": (18, "XVIII Symptoms"),
    "XIX": (19, "XIX Injury"), "XX": (20, "XX External"), "XXI": (21, "XXI Factors"),
}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load(path: Path, min_cases: int) -> list[dict]:
    recs = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_pos = int(r["n_positives"])
            auc = _f(r["age_sex_auc"])
            if n_pos < min_cases or not np.isfinite(auc):
                continue
            recs.append({
                "token_key": r["token_key"],
                "code_norm": r["code_norm"],
                "chapter": r["chapter"],
                "n_positives": n_pos,
                "train_occurrences": max(int(r["train_occurrences"]), 1),
                "auc": auc,
            })
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("results/all_disease_lm/all_disease_lm_auc.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/all_disease_lm"))
    ap.add_argument("--min-cases", type=int, default=20)
    args = ap.parse_args()

    recs = load(args.csv, args.min_cases)
    recs.sort(key=lambda r: -r["auc"])
    chapters_present = list(OrderedDict((r["chapter"], True) for r in recs).keys())
    cmap = plt.get_cmap("tab20")
    chapter_color = {ch: cmap(CHAPTER_ORDER.get(ch, (99, ch))[0] % 20) for ch in chapters_present}

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    # ---- A: scatter (Delphi Fig 2b style) ----
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for r in recs:
        x = np.log10(r["train_occurrences"])
        col = chapter_color[r["chapter"]]
        ax.scatter(x, r["auc"], s=np.clip(r["n_positives"], 8, 1200) * 0.12,
                   c=[col], alpha=0.7, edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(0.5, color="0.4", ls="--", lw=1, zorder=1)
    for r in recs[:6] + recs[-6:]:
        ax.annotate(r["code_norm"], (np.log10(r["train_occurrences"]), r["auc"]),
                    textcoords="offset points", xytext=(3, 3), fontsize=6.5, color="0.2")
    ax.set_xlabel("Training occurrences (log10)")
    ax.set_ylabel("Age–sex-stratified AUROC")
    ax.set_title(f"A  Disease-onset discrimination across {len(recs)} ICD-10 codes (>= {args.min_cases} cases)")
    ax.set_ylim(0.0, 1.0)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=chapter_color[ch],
                          label=CHAPTER_ORDER.get(ch, (99, ch))[1]) for ch in chapters_present]
    ax.legend(handles=handles, fontsize=6.5, loc="lower right", framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out / "all_disease_lm_scatter.png", dpi=200)
    plt.close(fig)

    # ---- B: box + strip by chapter (Delphi Fig 2c style) ----
    ch_list = sorted(chapters_present, key=lambda ch: CHAPTER_ORDER.get(ch, (99, ""))[0])
    data_by_ch = {ch: [r["auc"] for r in recs if r["chapter"] == ch] for ch in ch_list}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    xs, labels = [], []
    for i, ch in enumerate(ch_list):
        vals = data_by_ch[ch]
        x = i + 1
        labels.append(CHAPTER_ORDER.get(ch, (99, ch))[1])
        jitter = np.random.default_rng(i).uniform(-0.14, 0.14, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=12, c=[chapter_color[ch]],
                   alpha=0.55, edgecolors="none", zorder=3)
        if len(vals) >= 5:
            bp = ax.boxplot(vals, positions=[x], widths=0.5, showfliers=False,
                            patch_artist=True, zorder=2)
            for patch in bp["boxes"]:
                patch.set_facecolor(chapter_color[ch]); patch.set_alpha(0.2)
            for med in bp["medians"]:
                med.set_color("black")
        else:
            ax.plot([x], [np.median(vals)], marker="_", ms=20, color="black", zorder=4)
        xs.append(x)
    ax.axhline(0.5, color="0.4", ls="--", lw=1, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Age–sex-stratified AUROC")
    ax.set_title(f"B  AUROC distribution by ICD-10 chapter (>= {args.min_cases} cases)")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(args.out / "all_disease_lm_by_chapter.png", dpi=200)
    plt.close(fig)

    print(f"[wrote] {args.out / 'all_disease_lm_scatter.png'} ({len(recs)} codes)")
    print(f"[wrote] {args.out / 'all_disease_lm_by_chapter.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
