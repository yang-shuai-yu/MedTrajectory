"""Figure 4 (Results, §5.2): forest plot of MedTrajectory_v2 vs MedTrajectory_v1
paired effect sizes on the four trajectory-fidelity metrics.

Pure plotting from hardcoded locked-test numbers (zero computation).
Source: technical doc v3 Table 4 (main text) / Table 9 (Supplementary D).

Run on server env: <python interpreter, e.g. the project virtualenv>
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# name, estimate (v2 - v1), CI low, CI high, higher-is-better, unit
METRICS = [
    ("Diagnosis Jaccard",       0.01385, 0.01065, 0.01706, True,  ""),
    ("Hit@10",                  0.03215, 0.02077, 0.04394, True,  ""),
    ("First-event time MAE",   -5.010,  -6.57,   -3.44,   False, "days"),
    ("Event-count MAE",        -0.899,  -1.007,  -0.787,  False, ""),
]

fig, axes = plt.subplots(nrows=len(METRICS), ncols=1, figsize=(6.2, 5.6))
fig.subplots_adjust(left=0.32, right=0.80, top=0.96, bottom=0.06, hspace=1.05)

for ax, (name, est, lo, hi, higher, unit) in zip(axes, METRICS):
    # point estimate + 95% CI error bar
    ax.errorbar([est], [0.0], xerr=[[est - lo], [hi - est]],
                fmt="o", color="#1f77b4", markersize=6.5,
                linewidth=1.6, capsize=5, capthick=1.6, zorder=3)
    # zero-effect reference line (centred)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, zorder=2)

    # symmetric x limits centred on zero
    m = max(abs(lo), abs(hi)) * 1.18
    ax.set_xlim(-m, m)
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.9)

    # metric name (with unit) on the left
    label = name if not unit else f"{name} ({unit})"
    ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center",
                  labelpad=10)
    ax.yaxis.set_label_position("left")

    # numeric annotations
    ax.text(est, 0.42, f"{est:+.4g}", ha="center", va="bottom", fontsize=9,
            fontweight="bold")
    ax.text(lo, -0.48, f"{lo:+.4g}", ha="center", va="top", fontsize=7.5,
            color="0.35")
    ax.text(hi, -0.48, f"{hi:+.4g}", ha="center", va="top", fontsize=7.5,
            color="0.35")

    # direction annotation: metric-aware (v2 improvement is + for higher-better,
    # - for lower-better metrics)
    if higher:
        fav_v1, fav_v2 = 0.02, 0.98     # v2 better on the right (positive)
    else:
        fav_v1, fav_v2 = 0.98, 0.02     # v2 better on the left (negative)
    ax.text(fav_v1, 0.78, "favours\nv1", transform=ax.transAxes, ha="center",
            va="top", fontsize=7, color="0.35")
    ax.text(fav_v2, 0.78, "favours\nv2", transform=ax.transAxes, ha="center",
            va="top", fontsize=7, color="0.35")

axes[-1].set_xlabel("Paired effect size (v2 \u2212 v1)", fontsize=9.5)

OUT = "fig4_relative_time_forest.png"
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
