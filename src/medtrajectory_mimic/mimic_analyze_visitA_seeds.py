"""Design A three-seed summary (seeds 42/43/44), matching the UKB Table-6 reporting style:
per-model mean +/- SD across the three training seeds, and per-seed Delta(A2-A0) averaged across seeds.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json, math
from pathlib import Path
from statistics import mean, pstdev

DATA = Path(str(MIMIC_ROOT))
SEEDS = [("42", "vA_a0", "vA_a2"), ("43", "vA_a0_s43", "vA_a2_s43"), ("44", "vA_a0_s44", "vA_a2_s44")]
KEYS = [("hit_at_1", "Hit@1"), ("hit_at_10", "Hit@10"), ("diagnosis_jaccard", "Diag Jaccard"),
        ("diagnosis_recall", "Diag Recall"), ("first_event_time_mae_days", "Time MAE (d)"),
        ("event_count_mae", "Count MAE"), ("sequence_edit_distance", "Seq edit"),
        ("death_brier", "Death Brier")]


def load(mode, name):
    p = DATA / f"eval_visitA_seeds_{mode}" / name / "summary.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["metrics"]
    return {k: (None if m.get(k) is None or not isinstance(m.get(k), (int, float)) or not math.isfinite(m[k]) else float(m[k]))
            for k, _ in KEYS} | {"_n": m.get("patient_count")}


for mode, desc in (("ukb", "UKB-aligned cohort (baseline 0.65, >=8 history, >=3 future)"),
                   ("relaxed", "Relaxed cohort (>=3 history, >=2 future, first 2000)")):
    print(f"\n# Design A - three training seeds - {desc}\n")
    a_vals = {k: [] for k, _ in KEYS}
    r_vals = {k: [] for k, _ in KEYS}
    deltas = {k: [] for k, _ in KEYS}
    ns = []
    got = []
    for seed, a0, a2 in SEEDS:
        ma, mr = load(mode, a0), load(mode, a2)
        if ma is None or mr is None:
            print(f"(seed {seed}: MISSING)")
            continue
        got.append(seed)
        ns.append(ma["_n"])
        for k, _ in KEYS:
            if ma[k] is not None and mr[k] is not None:
                a_vals[k].append(ma[k]); r_vals[k].append(mr[k]); deltas[k].append(mr[k] - ma[k])
    if not got:
        print("no data"); continue
    print(f"seeds available: {got}   n per seed: {ns}\n")
    print("| Metric | A0 (3-seed mean ± SD) | A2 (3-seed mean ± SD) | Δ(A2−A0) mean ± SD | per-seed Δ |")
    print("|---|---:|---:|---:|---|")
    for k, lbl in KEYS:
        if len(a_vals[k]) < 2:
            print(f"| {lbl} | NA | NA | NA | NA |")
            continue
        am, asd = mean(a_vals[k]), pstdev(a_vals[k])
        rm, rsd = mean(r_vals[k]), pstdev(r_vals[k])
        dm, dsd = mean(deltas[k]), pstdev(deltas[k])
        per = ", ".join(f"{d:+.4f}" for d in deltas[k])
        print(f"| {lbl} | {am:.4f} ± {asd:.4f} | {rm:.4f} ± {rsd:.4f} | {dm:+.4f} ± {dsd:.4f} | {per} |")

    # UKB reference row for direction comparison
    print("\nUKB Table-6 reference Δ(A2−A0): Jaccard +0.0139, Hit@10 +0.0322, Time MAE −5.01 d, Count MAE −0.899")
    # count metrics whose 3-seed mean delta agrees in sign with UKB (excluding Hit@10 which is neutral)
    ukb_sign = {"diagnosis_jaccard": +1, "first_event_time_mae_days": -1, "event_count_mae": -1}
    agree = sum(1 for k, s in ukb_sign.items()
                if deltas[k] and (mean(deltas[k]) * s) > 0)
    print(f"3-seed mean Δ agrees in direction with UKB on {agree}/3 of (Jaccard, Time MAE, Count MAE).")
