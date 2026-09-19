"""Analyze design C (visit-level composite) results: Experiment 2 paired A0 vs A2, two cohort thresholds."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json, math
from pathlib import Path
from statistics import mean, pstdev
from math import erfc

DATA = Path(str(MIMIC_ROOT))
KEYS = [("hit_at_1", "Hit@1"), ("hit_at_10", "Hit@10"), ("diagnosis_jaccard", "Diag Jaccard"),
        ("diagnosis_recall", "Diag Recall"), ("first_event_time_mae_days", "Time MAE (d)"),
        ("event_count_mae", "Count MAE"), ("sequence_edit_distance", "Seq edit"),
        ("death_brier", "Death Brier")]


def load(base, name):
    d = base / name
    if not (d / "summary.json").exists():
        return None, None
    s = json.loads((d / "summary.json").read_text())["metrics"]
    rows = [json.loads(l) for l in (d / "patient_rows.jsonl").read_text().splitlines() if l.strip()]
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
    W = min(sum(ar[abs(d)] for d in diffs if d > 0), sum(ar[abs(d)] for d in diffs if d < 0))
    mu = n * (n + 1) / 4
    tie = sum(len(rs) ** 3 - len(rs) for rs in rm.values())
    var = n * (n + 1) * (2 * n + 1) / 24 - tie / 48
    if var <= 0:
        return None
    return erfc(abs((W - mu) / math.sqrt(var)) / math.sqrt(2))


print("# Design C (one token per admission = composite of first 2 codes) — Experiment 2\n")
for tag, desc in (("ukb", "UKB-aligned cohort (baseline 0.65, >=8 history, >=3 future)"),
                  ("relaxed", "Relaxed cohort (>=3 history, >=2 future, first 2000)")):
    base = DATA / f"eval_visitC_{tag}"
    print(f"\n## {desc}\n")
    a_s, a_rows = load(base, "vC_a0")
    r_s, r_rows = load(base, "vC_a2")
    if not a_s or not r_s:
        print("MISSING results under", base)
        continue
    print("| Model | n | " + " | ".join(l for _, l in KEYS) + " |")
    print("|---|---:" + "|---:" * len(KEYS) + "|")
    for nm, s in (("A0 (absolute)", a_s), ("A2 (additive_v2_2)", r_s)):
        cells = []
        for k, _ in KEYS:
            v = s.get(k)
            cells.append("NA" if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else f"{v:.4f}")
        print(f"| {nm} | {s.get('patient_count')} | " + " | ".join(cells) + " |")
    am = {x["patient_index"]: x for x in a_rows}
    rm = {x["patient_index"]: x for x in r_rows}
    common = sorted(set(am) & set(rm))
    print(f"\n### Paired A2 − A0 (n={len(common)})\n")
    print("| Metric | A0 | A2 | Δ(A2−A0) | Wilcoxon p |")
    print("|---|---:|---:|---:|---:|")
    for k, lbl in KEYS:
        pairs = [(am[p].get(k), rm[p].get(k)) for p in common]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
        if len(pairs) < 5:
            print(f"| {lbl} | NA | NA | NA | NA |")
            continue
        mx = mean(x for x, _ in pairs); my = mean(y for _, y in pairs)
        p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs])
        star = ""
        if p is not None:
            star = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
        print(f"| {lbl} | {mx:.4f} | {my:.4f} | {my-mx:+.4f} | {'NA' if p is None else f'{p:.4g}'}{star} |")

print("\n--- end of design C analysis ---")
