"""Compile MIMIC generation-eval summaries into a comparison table for Experiments 1 & 2."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

import sys
BASE = Path(sys.argv[1] if len(sys.argv) > 1 else str(MIMIC_ROOT / "eval_quick"))
MODELS = [
    ("M1 (diagnosis only)", "m1_abs", "Exp1"),
    ("M2 (+procedure)", "m2_abs", "Exp1"),
    ("M3 (full)", "m3_abs", "Exp1"),
    ("M3 Relative", "m3_rel", "Exp2"),
]
KEYS = [
    ("hit_at_1", "Hit@1"),
    ("hit_at_10", "Hit@10"),
    ("diagnosis_jaccard", "Diag Jac"),
    ("diagnosis_recall", "Diag Rec"),
    ("first_event_time_mae_days", "TimeMAE(d)"),
    ("event_count_mae", "CntMAE"),
    ("sequence_edit_distance", "SeqEdit"),
    ("death_brier", "DeathBrier"),
]


def mean_sd(vals):
    import math
    from statistics import mean, pstdev
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return None, None, 0
    return mean(v), pstdev(v), len(v)


def wilcoxon(a, b):
    import math
    from math import erfc
    from statistics import mean
    diffs = [x - y for x, y in zip(a, b) if (x - y) != 0]
    n = len(diffs)
    if n < 5:
        return None, None
    absd = [abs(d) for d in diffs]
    sorted_abs = sorted(absd)
    rank_map = {}
    for i, v in enumerate(sorted_abs):
        rank_map.setdefault(v, []).append(i + 1)
    avg_rank = {v: mean(rs) for v, rs in rank_map.items()}
    w_plus = sum(avg_rank[abs(d)] for d in diffs if d > 0)
    w_minus = sum(avg_rank[abs(d)] for d in diffs if d < 0)
    W = min(w_plus, w_minus)
    mu = n * (n + 1) / 4
    tie = sum(len(rs) ** 3 - len(rs) for rs in rank_map.values())
    var = n * (n + 1) * (2 * n + 1) / 24 - tie / 48
    if var <= 0:
        return W, None
    z = (W - mu) / math.sqrt(var)
    return W, erfc(abs(z) / math.sqrt(2))


def rows_of(name):
    p = BASE / name / "patient_rows.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


print(f"# MIMIC-IV generation eval (test split, n~1700/model)  base={BASE}\n")
print("## Experiment 1 & 2 (mean +/- SD across patients)\n")
print("| Model | n | " + " | ".join(l for _, l in KEYS) + " |")
print("|---|---:" + "|---:" * len(KEYS) + "|")
data = {}
for label, name, exp in MODELS:
    rows = rows_of(name)
    data[name] = rows
    if not rows:
        print(f"| {label} | - | MISSING |")
        continue
    cells = []
    for k, _ in KEYS:
        m, s, _ = mean_sd([r.get(k) for r in rows])
        cells.append("NA" if m is None else f"{m:.4f}±{s:.4f}")
    print(f"| {label} | {len(rows)} | " + " | ".join(cells) + " |")

print("\n## Experiment 2 paired (Absolute vs Relative, same patients)\n")
a, r = data.get("m3_abs"), data.get("m3_rel")
if a and r:
    am = {x["patient_index"]: x for x in a}
    rm = {x["patient_index"]: x for x in r}
    common = sorted(set(am) & set(rm))
    print(f"common patients: {len(common)}\n")
    print("| Metric | Abs | Rel | Δ(Rel-Abs) | Wilcoxon p |")
    print("|---|---:|---:|---:|---:|")
    import math
    for k, lbl in KEYS:
        pairs = [(am[p].get(k), rm[p].get(k)) for p in common]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
        if len(pairs) < 5:
            print(f"| {lbl} | NA | NA | NA | NA |")
            continue
        mx = sum(x for x, _ in pairs) / len(pairs)
        my = sum(y for _, y in pairs) / len(pairs)
        _, p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs])
        ps = "NA" if p is None else f"{p:.4g}"
        star = ""
        if p is not None:
            star = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
        print(f"| {lbl} | {mx:.4f} | {my:.4f} | {my-mx:+.4f} | {ps}{star} |")

print("\n## Experiment 1 deltas (descriptive; ablations use slightly different patient sets)\n")
m1, m2, m3 = data.get("m1_abs"), data.get("m2_abs"), data.get("m3_abs")
if m1 and m2 and m3:
    for k, lbl in KEYS:
        v1 = mean_sd([x.get(k) for x in m1])[0]
        v2 = mean_sd([x.get(k) for x in m2])[0]
        v3 = mean_sd([x.get(k) for x in m3])[0]
        f = lambda v: "NA" if v is None else f"{v:.4f}"
        print(f"- {lbl}: M1={f(v1)}  M2={f(v2)}  M3={f(v3)}")
