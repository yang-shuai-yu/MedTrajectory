try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json, math, sys
from pathlib import Path
from statistics import mean, pstdev
from math import erfc

BASE = Path(str(MIMIC_ROOT / "eval_trackr"))
PAIR = [("trackr_a0_abs", "A0 (absolute, static prefix)"), ("trackr_a2_rel", "A2 (additive_v2_2 relative)")]
KEYS = [
    ("hit_at_1", "Hit@1"), ("hit_at_10", "Hit@10"),
    ("diagnosis_jaccard", "Diag Jaccard"), ("diagnosis_recall", "Diag Recall"),
    ("first_event_time_mae_days", "Time MAE (d)"), ("event_count_mae", "Count MAE"),
    ("sequence_edit_distance", "Seq edit"), ("death_brier", "Death Brier"),
    ("duplicate_event_rate", "Dup rate"),
]


def load(name):
    s = json.loads((BASE / name / "summary.json").read_text())
    rows = [json.loads(l) for l in (BASE / name / "patient_rows.jsonl").read_text().splitlines() if l.strip()]
    return s, rows


def wilcoxon(a, b):
    diffs = [x - y for x, y in zip(a, b) if (x - y) != 0]
    n = len(diffs)
    if n < 5:
        return None
    absd = sorted(abs(d) for d in diffs)
    rm = {}
    for i, v in enumerate(absd):
        rm.setdefault(v, []).append(i + 1)
    ar = {v: mean(rs) for v, rs in rm.items()}
    wp = sum(ar[abs(d)] for d in diffs if d > 0)
    wm = sum(ar[abs(d)] for d in diffs if d < 0)
    W = min(wp, wm)
    mu = n * (n + 1) / 4
    tie = sum(len(rs) ** 3 - len(rs) for rs in rm.values())
    var = n * (n + 1) * (2 * n + 1) / 24 - tie / 48
    if var <= 0:
        return None
    z = (W - mu) / math.sqrt(var)
    return erfc(abs(z) / math.sqrt(2))


print("# MIMIC Track-R (additive_v2_2) — Experiment 2, test split\n")
rows_by = {}
for name, label in PAIR:
    s, rows = load(name)
    rows_by[name] = rows
    m = s["metrics"]
    print(f"## {label}  (n={m['patient_count']})")
    for k, lbl in KEYS:
        v = m.get(k)
        sd = None
        vals = [r.get(k) for r in rows if r.get(k) is not None and math.isfinite(r.get(k))]
        if vals:
            sd = pstdev(vals)
        print(f"  - {lbl}: " + ("NA" if v is None or not math.isfinite(v) else (f"{v:.4f}±{sd:.4f}" if sd else f"{v:.4f}")))
    print()

a_rows = {r["patient_index"]: r for r in rows_by["trackr_a0_abs"]}
r_rows = {r["patient_index"]: r for r in rows_by["trackr_a2_rel"]}
common = sorted(set(a_rows) & set(r_rows))
print(f"## Paired A2 - A0 (common patients = {len(common)})\n")
print("| Metric | A0 (Abs) | A2 (Rel) | Δ(A2-A0) | Wilcoxon p |")
print("|---|---:|---:|---:|---:|")
for k, lbl in KEYS:
    pairs = [(a_rows[p].get(k), r_rows[p].get(k)) for p in common]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 5:
        print(f"| {lbl} | NA | NA | NA | NA |")
        continue
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs])
    ps = "NA" if p is None else f"{p:.4g}"
    star = ""
    if p is not None:
        star = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
    print(f"| {lbl} | {mx:.4f} | {my:.4f} | {my-mx:+.4f} | {ps}{star} |")
