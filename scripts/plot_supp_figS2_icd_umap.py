# -*- coding: utf-8 -*-
"""Supplementary Figure S2: low-dimensional UMAP of ICD-10 representation
strategies, re-rendered from the saved 64d embeddings (not composited PNGs).

- Recomputes UMAP (random_state=0, same as compare_icd_embeddings.py) per method
  from emb_{method}_64d.npy.
- 2x2 panel, one shared ICD-10 chapter legend, Delphi chapter colours.
- dpi=300 for publication.

Run on server env: <python interpreter, e.g. the project virtualenv>
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

REPO = str(REPO_ROOT)
OUT = f"{REPO}/outputs/icd_embedding_compare"
CSV = f"{OUT}/delphi_labels_chapters_colours_icd.csv"

# --- exact same record order as compare_icd_embeddings.py ---
sys.path.insert(0, f"{REPO}/scripts")
from compare_icd_embeddings import load_delphi_csv  # noqa: E402

records = [r for r in load_delphi_csv(Path(CSV)) if r.get("chapter")]
codes = [r["code"] for r in records]
chapter_names = sorted({r["chapter"] for r in records})
chapter_id = {c: i for i, c in enumerate(chapter_names)}
gold = np.asarray([chapter_id[r["chapter"]] for r in records], dtype=np.int64)
n = len(codes)
print("codes:", n, "chapters:", len(chapter_names))

# --- chapter -> colour from the Delphi CSV (hex) ---
def sniff_dicts(text):
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    return list(csv.DictReader(text.splitlines(), dialect=dialect))

rows = sniff_dicts(open(CSV, encoding="utf-8-sig").read())
chapter_color = {}
for row in rows:
    low = {str(k).strip().lower(): str(v).strip() for k, v in row.items()}
    ch = low.get("icd-10 chapter", "") or low.get("icd-10 chapter (short)", "")
    col = low.get("color", "") or low.get("colour", "")
    if ch and col and ch not in chapter_color:
        chapter_color[ch] = col

# fallback: tab20 for any chapter without an assigned colour
tab20 = [matplotlib.colors.to_hex(c) for c in plt.cm.tab20.colors]
for i, ch in enumerate(chapter_names):
    if ch not in chapter_color:
        chapter_color[ch] = tab20[i % len(tab20)]

print("chapter_colour_map:")
for ch in chapter_names:
    print("  ", ch, "->", chapter_color[ch])

methods = ["hierarchy", "gram", "pubmedbert", "qwen"]
titles = [
    "ICD hierarchy (chapter one-hot)",
    "GRAM (section one-hot)",
    "PubMedBERT (768d \u2192 PCA-64)",
    "Qwen text-embedding-v4 (1024d \u2192 PCA-64)",
]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
for ax, m, title in zip(axes.ravel(), methods, titles):
    mat = np.load(f"{OUT}/emb_{m}_64d.npy")
    assert mat.shape[0] == n, (m, mat.shape)
    emb = umap.UMAP(n_components=2, random_state=0).fit_transform(mat)
    for i, ch in enumerate(chapter_names):
        mask = gold == i
        ax.scatter(emb[mask, 0], emb[mask, 1], s=4, c=chapter_color[ch],
                   alpha=0.65, linewidths=0)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

handles = [
    plt.Line2D([], [], marker="o", linestyle="", markersize=5,
               markerfacecolor=chapter_color[ch], markeredgewidth=0, label=ch)
    for ch in chapter_names
]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.985, 0.5),
           fontsize=6.3, frameon=False, title="ICD-10 chapter",
           title_fontsize=8, handlelength=1.0)
fig.tight_layout(rect=[0, 0, 0.845, 1])

OUTPNG = f"{REPO}/scripts/figS2_icd_umap.png"
fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
print("wrote", OUTPNG)
