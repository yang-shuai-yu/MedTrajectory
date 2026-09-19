"""Supplementary Figure S1: relationship between observed mortality,
rollout death rate, and independently-estimated censor-aware death risk.

Panel A: observed (locked-test) vs rollout death probability per model.
Panel B: reliability diagram of raw rollout death probability (validation),
         pooled over the three seeds from results/track_g_v1/val_death_calibration.

Run on server env: <python interpreter, e.g. the project virtualenv>
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

REPO = str(REPO_ROOT)
BINS_CSV = f"{REPO}/results/track_g_v1/val_death_calibration/reliability_bins.csv"

# ---- Panel A: locked-test observed vs rollout death probability ----
models = ["MedTrajectory-\nAbsolute", "MedTrajectory-\nRelative",
          "ETHOS-Matched", "Foresight-Matched"]
rollout = [0.653, 0.729, 0.238, 0.079]
observed_locked = 0.122
colors = {"MedTrajectory-\nAbsolute": "#7f7f7f",
          "MedTrajectory-\nRelative": "#1f77b4",
          "ETHOS-Matched": "#2ca02c",
          "Foresight-Matched": "#d62728"}

# ---- Panel B: reliability diagram (raw variant, pooled over seeds) ----
df = pd.read_csv(BINS_CSV)
raw = df[(df["variant"] == "raw") & (df["count"] > 0)].copy()
g = raw.groupby(["model", "bin"]).apply(
    lambda d: pd.Series({
        "n": d["count"].sum(),
        "mean_prob": (d["count"] * d["mean_probability"]).sum() / d["count"].sum(),
        "obs_rate": (d["count"] * d["observed_rate"]).sum() / d["count"].sum(),
    })).reset_index()

name_map = {"A0": "MedTrajectory-Absolute", "A2": "MedTrajectory-Relative",
            "ETHOS-Matched": "ETHOS-Matched", "Foresight-Matched": "Foresight-Matched"}
color_map = {"A0": "#7f7f7f", "A2": "#1f77b4",
             "ETHOS-Matched": "#2ca02c", "Foresight-Matched": "#d62728"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.4))

# Panel A
x = np.arange(len(models))
axA.bar(x, rollout, color=[colors[m] for m in models], width=0.6)
axA.axhline(observed_locked, color="black", linestyle="--", linewidth=1.3)
axA.text(0.99, observed_locked + 0.015, "observed mortality\n(locked test, 0.122)",
         ha="right", va="bottom", fontsize=7.5)
axA.set_xticks(x)
axA.set_xticklabels(models, fontsize=7.5)
axA.set_ylabel("Rollout death probability", fontsize=9)
axA.set_ylim(0, 0.85)
axA.set_title("A  Observed vs rollout mortality", fontsize=10, loc="left")
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)

# Panel B
axB.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, alpha=0.6)
for m in ["A0", "A2", "ETHOS-Matched", "Foresight-Matched"]:
    sub = g[g["model"] == m].sort_values("mean_prob")
    axB.plot(sub["mean_prob"], sub["obs_rate"], "-o", markersize=4.5,
             linewidth=1.4, color=color_map[m], label=name_map[m])
axB.set_xlabel("Mean predicted death probability", fontsize=9)
axB.set_ylabel("Observed death rate", fontsize=9)
axB.set_xlim(0, 1)
axB.set_ylim(0, 1)
axB.set_aspect("equal", adjustable="box")
axB.set_title("B  Reliability diagram (rollout, validation)", fontsize=10, loc="left")
axB.legend(fontsize=7.5, frameon=False, loc="upper left")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

fig.tight_layout()
OUT = "figS1_death_calibration.png"
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
