"""Generic 3-seed summary for a model family on a given representation.

Usage:
  mimic_analyze_3seed_generic.py --tag <eval_dir_tag> --a0 stem42,stem43,stem44 --a2 stem42,stem43,stem44 --label "..."
Reads <DATA>/<eval_dir_tag>/<stem>/summary.json for each seed and prints the 3-seed table.
"""
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


def load(tag, stem):
    p = DATA / tag / stem / "summary.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["metrics"]
    out = {}
    for k in m:
        v = m[k]
        out[k] = float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
    out["_n"] = m.get("patient_count")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--a0", required=True)
    ap.add_argument("--a2", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    a0_list = args.a0.split(",")
    a2_list = args.a2.split(",")
    print(f"### {args.label or args.tag}\n")
    rows = []
    for name, stems in (("A0 (absolute)", a0_list), ("A2 (relative)", a2_list)):
        vals = {}
        ns = []
        for s in stems:
            m = load(args.tag, s)
            if m is None:
                continue
            ns.append(m["_n"])
            for k, _ in KEYS:
                if m.get(k) is not None:
                    vals.setdefault(k, []).append(m[k])
        rows.append((name, vals, ns))
    ns = rows[0][2] if rows else []
    print(f"seeds: {len(ns)}   n per seed: {ns}\n")
    print("| Metric | A0 (mean ± SD) | A2 (mean ± SD) | Δ(A2−A0) mean ± SD | per-seed Δ |")
    print("|---|---:|---:|---:|---|")
    for k, lbl in KEYS:
        v0 = rows[0][1].get(k, [])
        v2 = rows[1][1].get(k, []) if len(rows) > 1 else []
        if len(v0) < 1 or len(v0) != len(v2):
            print(f"| {lbl} | NA | NA | NA | NA |")
            continue
        d = [b - a for a, b in zip(v0, v2)]
        sd0 = f" ± {pstdev(v0):.4f}" if len(v0) > 1 else ""
        sd2 = f" ± {pstdev(v2):.4f}" if len(v2) > 1 else ""
        sdd = f" ± {pstdev(d):.4f}" if len(d) > 1 else ""
        per = ", ".join(f"{x:+.4f}" for x in d)
        print(f"| {lbl} | {mean(v0):.4f}{sd0} | {mean(v2):.4f}{sd2} | {mean(d):+.4f}{sdd} | {per} |")
    print()


if __name__ == "__main__":
    main()
