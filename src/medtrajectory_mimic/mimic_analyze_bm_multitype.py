"""Bm-matched multitype summary: M1 / M2 / M3 with 3-seed means on an identical patient cohort."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import argparse, json, math
from pathlib import Path
from statistics import mean, pstdev

DATA = Path(str(MIMIC_ROOT))
KEYS = [("hit_at_1", "Hit@1"), ("hit_at_10", "Hit@10"), ("diagnosis_jaccard", "Diag Jaccard"),
        ("diagnosis_recall", "Diag Recall"), ("first_event_time_mae_days", "Time MAE (d)"),
        ("event_count_mae", "Count MAE"), ("sequence_edit_distance", "Seq edit"),
        ("death_brier", "Death Brier")]
STEMS = {"M1 (principal dx only)": ["bm_m1_a0", "bm_m1_a0_s43", "bm_m1_a0_s44"],
         "M2 (+ principal procedure)": ["bm_m2_a0", "bm_m2_a0_s43", "bm_m2_a0_s44"],
         "M3 (+ death)": ["bm_m3_a0", "bm_m3_a0_s43", "bm_m3_a0_s44"]}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--label", default="")
args = ap.parse_args()

print(f"### {args.label or args.tag}\n")
acc = {}
n_seeds = 0
for label, stems in STEMS.items():
    vals = {}
    ns = []
    for s in stems:
        p = DATA / args.tag / s / "summary.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())["metrics"]
        ns.append(m.get("patient_count"))
        for k, _ in KEYS:
            v = m.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                vals.setdefault(k, []).append(float(v))
    acc[label] = vals
    n_seeds = max(n_seeds, len(ns))
    print(f"({label}: seeds={len(ns)} n={ns})")
print()
print("| Metric | " + " | ".join(acc.keys()) + " |")
print("|---" + "|---:" * len(acc) + "|")
for k, lbl in KEYS:
    cells = []
    for label in acc:
        v = acc[label].get(k, [])
        cells.append("NA" if not v else (f"{mean(v):.4f} ± {pstdev(v):.4f}" if len(v) > 1 else f"{mean(v):.4f}"))
    print(f"| {lbl} | " + " | ".join(cells) + " |")
print()
print("Δ(M2−M1) and Δ(M3−M2) on the 3-seed means (identical patient cohort):\n")
print("| Metric | M1 | M2 | M3 | Δ(M2−M1) | Δ(M3−M2) |")
print("|---|---:|---:|---:|---:|---:|")
labels = list(acc.keys())
for k, lbl in KEYS:
    v1 = acc[labels[0]].get(k, [])
    v2 = acc[labels[1]].get(k, [])
    v3 = acc[labels[2]].get(k, [])
    if not (v1 and v2 and v3):
        print(f"| {lbl} | NA | NA | NA | NA | NA |")
        continue
    m1, m2, m3 = mean(v1), mean(v2), mean(v3)
    print(f"| {lbl} | {m1:.4f} | {m2:.4f} | {m3:.4f} | {m2-m1:+.4f} | {m3-m2:+.4f} |")
print()
