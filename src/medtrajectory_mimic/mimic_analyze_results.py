"""Analyze MIMIC generation-eval results: mean +/- SD across patients, Experiment 1 & 2 tables,
and a paired Wilcoxon test for Experiment 2 (Absolute vs Relative, same M3 patients)."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
import math
from pathlib import Path
from statistics import mean, pstdev

BASE = Path(str(MIMIC_ROOT / "eval"))
MODELS = [
    ("M1 (diagnosis only)", "m1_abs", "Exp1"),
    ("M2 (+procedure)", "m2_abs", "Exp1"),
    ("M3 (full)", "m3_abs", "Exp1"),
    ("M3 Relative", "m3_rel", "Exp2"),
]
KEYS = [
    ("hit_at_1", "Hit@1"),
    ("hit_at_10", "Hit@10"),
    ("diagnosis_jaccard", "Diag Jaccard"),
    ("diagnosis_recall", "Diag Recall"),
    ("first_event_time_mae_days", "Time MAE (d)"),
    ("event_count_mae", "Count MAE"),
    ("sequence_edit_distance", "Seq edit"),
    ("death_brier", "Death Brier"),
]


def load_rows(name):
    p = BASE / name / "patient_rows.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def mean_sd(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None, None, 0
    return mean(vals), pstdev(vals), len(vals)


def wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test (two-sided). Returns (statistic, p_value)."""
    diffs = [x - y for x, y in zip(a, b)]
    diffs = [d for d in diffs if d != 0]
    if len(diffs) < 5:
        return None, None
    ranks = {v: i + 1 for i, v in enumerate(sorted({abs(d) for d in diffs}))}
    # handle ties by average rank
    from collections import defaultdict
    abs_d = [abs(d) for d in diffs]
    sorted_abs = sorted(abs_d)
    rank_map = {}
    for i, v in enumerate(sorted_abs):
        rank_map.setdefault(v, []).append(i + 1)
    avg_rank = {v: mean(rs) for v, rs in rank_map.items()}
    w_plus = sum(avg_rank[abs(d)] for d in diffs if d > 0)
    w_minus = sum(avg_rank[abs(d)] for d in diffs if d < 0)
    W = min(w_plus, w_minus)
    n = len(diffs)
    # normal approximation with tie correction
    mu = n * (n + 1) / 4
    tie_corr = sum(len(rs) ** 3 - len(rs) for rs in rank_map.values())
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_corr / 48
    if var <= 0:
        return W, None
    z = (W - mu) / math.sqrt(var)
    # two-sided p via normal approx
    from math import erfc
    p = erfc(abs(z) / math.sqrt(2))
    return W, p


rows_by_model = {}
for label, name, exp in MODELS:
    rows_by_model[name] = load_rows(name)

print("# MIMIC-IV generation evaluation (test split)\n")
print("## Experiment 1 & 2 summary (mean +/- SD across patients)\n")
print("| Model | n | " + " | ".join(lbl for _, lbl in KEYS) + " |")
print("|---|---:" + "|---:" * len(KEYS) + "|")
for label, name, exp in MODELS:
    rows = rows_by_model[name]
    if rows is None:
        print(f"| {label} | - | MISSING |")
        continue
    vals = []
    n = len(rows)
    for key, _ in KEYS:
        m, s, _ = mean_sd([r.get(key) for r in rows])
        if m is None:
            vals.append("NA")
        else:
            vals.append(f"{m:.4f}±{s:.4f}")
    print(f"| {label} | {n} | " + " | ".join(vals) + " |")

print("\n## Experiment 2 paired comparison (Absolute vs Relative, same M3 patients)\n")
a_rows = rows_by_model.get("m3_abs")
r_rows = rows_by_model.get("m3_rel")
if a_rows and r_rows:
    # align by patient_index
    a_map = {r["patient_index"]: r for r in a_rows}
    r_map = {r["patient_index"]: r for r in r_rows}
    common = sorted(set(a_map) & set(r_map))
    print(f"common patients: {len(common)}\n")
    print("| Metric | Abs (mean) | Rel (mean) | Δ (Rel-Abs) | Wilcoxon p |")
    print("|---|---:|---:|---:|---:|")
    for key, lbl in KEYS:
        av = [a_map[p].get(key) for p in common]
        rv = [r_map[p].get(key) for p in common]
        # filter to patients where both are finite
        pairs = [(x, y) for x, y in zip(av, rv) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
        if len(pairs) < 5:
            print(f"| {lbl} | NA | NA | NA | NA |")
            continue
        am = mean(x for x, _ in pairs)
        rm = mean(y for _, y in pairs)
        W, p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs])
        p_str = "NA" if p is None else f"{p:.4f}"
        sig = ""
        if p is not None and p < 0.05:
            sig = " *" if p < 0.05 else ""
            sig = " **" if p < 0.01 else sig
            sig = " ***" if p < 0.001 else sig
        print(f"| {lbl} | {am:.4f} | {rm:.4f} | {rm-am:+.4f} | {p_str}{sig} |")
    print("\n* p<0.05, ** p<0.01, *** p<0.001 (two-sided Wilcoxon signed-rank, normal approx)")
