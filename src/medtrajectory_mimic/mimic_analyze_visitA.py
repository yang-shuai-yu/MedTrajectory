try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json, math, sys
from pathlib import Path
from statistics import mean, pstdev
from math import erfc

DATA = Path(str(MIMIC_ROOT))
KEYS = [("hit_at_1", "Hit@1"), ("hit_at_10", "Hit@10"), ("diagnosis_jaccard", "Diag Jaccard"),
        ("diagnosis_recall", "Diag Recall"), ("first_event_time_mae_days", "Time MAE (d)"),
        ("event_count_mae", "Count MAE"), ("sequence_edit_distance", "Seq edit"),
        ("death_brier", "Death Brier")]


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


for tag, desc in (("ukb", "UKB-aligned (0.65, >=8 history, >=3 future)"),
                  ("relaxed", "Relaxed (>=3 history, >=2 future, first 2000)")):
    base = DATA / f"eval_visitA_{tag}"
    a_p, r_p = base / "vA_a0", base / "vA_a2"
    print(f"\n# Design A (1 token/visit = principal diagnosis) — {desc}\n")
    if not (a_p / "summary.json").exists():
        print("MISSING", a_p)
        continue
    a_s = json.loads((a_p / "summary.json").read_text())["metrics"]
    r_s = json.loads((r_p / "summary.json").read_text())["metrics"]
    rows = {}
    for nm, p in (("vA_a0", a_p), ("vA_a2", r_p)):
        rows[nm] = [json.loads(l) for l in (p / "patient_rows.jsonl").read_text().splitlines() if l.strip()]
    print("| Model | n | " + " | ".join(l for _, l in KEYS) + " |")
    print("|---|---:" + "|---:" * len(KEYS) + "|")
    for nm, s in (("A0 (absolute)", a_s), ("A2 (additive_v2_2)", r_s)):
        cells = []
        for k, _ in KEYS:
            v = s.get(k)
            cells.append("NA" if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else f"{v:.4f}")
        print(f"| {nm} | {s.get('patient_count')} | " + " | ".join(cells) + " |")
    am = {x["patient_index"]: x for x in rows["vA_a0"]}
    rm = {x["patient_index"]: x for x in rows["vA_a2"]}
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
