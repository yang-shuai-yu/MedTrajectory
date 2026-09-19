"""Analyze visit-level B Track-R eval results: Experiment 1 (M1/M2/M3) and Experiment 2 (A0 vs A2)."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json, math
from pathlib import Path
from statistics import mean, pstdev
from math import erfc

DATA = Path(str(MIMIC_ROOT))
MODES = [("ukb", "UKB-aligned cohort (baseline 0.65, >=8 history, >=3 future)"),
         ("relaxed", "Relaxed cohort (>=3 history, >=2 future; first 2000 eligible)")]
MODELS = [("M1 principal-dx only", "vB_m1_a0"), ("M2 +principal-proc", "vB_m2_a0"),
          ("M3 +death", "vB_m3_a0"), ("M3 additive_v2_2 (A2)", "vB_m3_a2")]
KEYS = [("hit_at_1", "Hit@1"), ("hit_at_10", "Hit@10"), ("diagnosis_jaccard", "Diag Jac"),
        ("diagnosis_recall", "Diag Rec"), ("first_event_time_mae_days", "TimeMAE(d)"),
        ("event_count_mae", "CntMAE"), ("sequence_edit_distance", "SeqEdit"), ("death_brier", "DeathBrier")]


def load(mode, name):
    d = DATA / f"eval_visitB_{mode}" / name
    if not (d / "summary.json").exists():
        return None, None
    s = json.loads((d / "summary.json").read_text())
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


for mode, desc in MODES:
    print(f"\n# Visit-level design B — {desc}\n")
    cache = {}
    print("## Table: Experiment 1 & 2 (mean ± SD across patients)\n")
    print("| Model | n | " + " | ".join(l for _, l in KEYS) + " |")
    print("|---|---:" + "|---:" * len(KEYS) + "|")
    for label, name in MODELS:
        s, rows = load(mode, name)
        cache[name] = rows
        if not rows:
            print(f"| {label} | - | MISSING |")
            continue
        cells = []
        for k, _ in KEYS:
            v = [r.get(k) for r in rows if r.get(k) is not None and math.isfinite(r.get(k))]
            cells.append("NA" if not v else f"{mean(v):.4f}±{pstdev(v):.4f}")
        print(f"| {label} | {len(rows)} | " + " | ".join(cells) + " |")

    a, r = cache.get("vB_m3_a0"), cache.get("vB_m3_a2")
    if a and r:
        am = {x["patient_index"]: x for x in a}
        rm = {x["patient_index"]: x for x in r}
        common = sorted(set(am) & set(rm))
        print(f"\n## Paired A2 − A0 (n={len(common)})\n")
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

    m1, m2, m3 = cache.get("vB_m1_a0"), cache.get("vB_m2_a0"), cache.get("vB_m3_a0")
    if m1 and m2 and m3:
        print("\n## Experiment 1 deltas (M1→M2→M3, descriptive)\n")
        for k, lbl in KEYS:
            f = lambda rows: (lambda v: "NA" if not v else f"{mean(v):.4f}")([r.get(k) for r in rows if r.get(k) is not None and math.isfinite(r.get(k))])
            print(f"- {lbl}: M1={f(m1)}  M2={f(m2)}  M3={f(m3)}")
