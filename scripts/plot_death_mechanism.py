"""Figure 6 (Results, §5.6): death mechanism three-panel figure.

Pure plotting from hardcoded validation / locked-test numbers (zero computation).
Source: technical doc v3 §4.7 + Supplementary F (Tables 14-15).

Run on server env: <python interpreter, e.g. the project virtualenv>
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Panel A: observed vs rollout mortality ----
models = ["v1", "v2", "ETHOS", "Foresight"]
rollout_death = [0.653, 0.729, 0.238, 0.079]
observed = 0.122

# ---- Panel B: death-token ablation (v2) ----
b_jaccard = (0.0834, 0.0530)      # raw, no-death
b_countmae = (12.9335, 24.2815)   # raw, no-death

# ---- Panel C: Brier-score improvement (rollout -> mortality head), 10y N=187 ----
c_models = ["v1", "v2", "ETHOS", "Foresight"]
c_improve = [0.0920, 0.0970, 0.0235, 0.1045]          # positive = better
c_lo = [0.0379, 0.0367, -0.0078, 0.0580]
c_hi = [0.1481, 0.1552, 0.0543, 0.1522]
c_err = [[m - l for m, l in zip(c_improve, c_lo)],
         [h - m for m, h in zip(c_improve, c_hi)]]

fig = plt.figure(figsize=(10.5, 3.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.72, 0.72, 1.0], wspace=0.55)

colors = {"v1": "#7f7f7f", "v2": "#1f77b4", "ETHOS": "#2ca02c",
          "Foresight": "#d62728"}

# --- Panel A ---
axA = fig.add_subplot(gs[0, 0])
x = np.arange(len(models))
axA.bar(x, rollout_death, color=[colors[m] for m in models], width=0.62)
axA.axhline(observed, color="black", linestyle="--", linewidth=1.2)
axA.text(0.98, observed + 0.02, "observed mortality\n0.122", ha="right",
         va="bottom", fontsize=7, color="black")
axA.set_xticks(x)
axA.set_xticklabels([f"{m}" for m in models], fontsize=8)
axA.set_ylabel("Rollout death probability", fontsize=8.5)
axA.set_ylim(0, 0.85)
axA.set_title("A  Observed vs rollout mortality", fontsize=9, loc="left")
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)

# --- Panel B (two sub-panels: Jaccard + Count MAE) ---
def paired_bar(ax, pair, title, ylabel, color="#1f77b4", fmt="{:.4g}"):
    x = np.arange(2)
    ax.bar(x, pair, color=[color, "#ff7f0e"], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["Raw", "No death"], fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_ylabel(ylabel, fontsize=8.5)
    for xi, v in zip(x, pair):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axB1 = fig.add_subplot(gs[0, 1])
paired_bar(axB1, b_jaccard, "B  Death-token ablation", "Diagnosis Jaccard",
           fmt="{:.4f}")
axB1.set_ylim(0, 0.10)

axB2 = fig.add_subplot(gs[0, 2])
paired_bar(axB2, b_countmae, " ", "Event-count MAE", color="#d62728",
           fmt="{:.2f}")
axB2.set_ylim(0, 27)

# --- Panel C ---
axC = fig.add_subplot(gs[0, 3])
x = np.arange(len(c_models))
axC.bar(x, c_improve, yerr=c_err, capsize=4, color=[colors[m] for m in c_models],
        width=0.62, error_kw=dict(linewidth=1.2))
axC.axhline(0, color="black", linewidth=0.9)
axC.set_xticks(x)
axC.set_xticklabels([f"{m}" for m in c_models], fontsize=8)
axC.set_ylabel("Brier-score improvement\n(mortality head \u2212 rollout)",
               fontsize=8.5)
axC.set_ylim(-0.04, 0.20)
axC.set_title("C  Mortality-head vs rollout Brier\n(10 y, N=187)", fontsize=9,
              loc="left")
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)

OUT = "fig6_death_mechanism.png"
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
