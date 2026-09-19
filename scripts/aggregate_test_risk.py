# -*- coding: utf-8 -*-
"""Aggregate Track R v2.2 locked-TEST risk evaluation (macro AUROC).

Reads the 9 summary.json files (3 models x 3 seeds) and prints macro AUROC per
model/seed, the A2-A0 delta, and a non-inferiority flag against margin -0.005.
The formal patient-level paired bootstrap (rows.json.gz) is a follow-up step.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import json
from pathlib import Path

ROOT = Path(str(REPO_ROOT))
OUT = ROOT / "results/track_r_v2_2/test"

MODELS = ["A0", "A2", "A2-noAge"]
SEEDS = [42, 43, 44]
MARGIN = -0.005


def macro_auc(model, seed):
    p = OUT / f"seed{seed}" / model / "summary.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return float(d["summary"]["macro"]["auc"])


def mean(xs):
    return sum(xs) / len(xs)


def main():
    table = {}
    print(f"{'model':<10} {'seed42':>10} {'seed43':>10} {'seed44':>10} {'mean':>10}")
    for m in MODELS:
        vals = [macro_auc(m, s) for s in SEEDS]
        table[m] = vals
        print(f"{m:<10} " + " ".join(f"{v:>10.5f}" for v in vals) + f" {mean(vals):>10.5f}")

    a2 = table["A2"]
    a0 = table["A0"]
    deltas = [x - y for x, y in zip(a2, a0)]
    mean_delta = mean(deltas)
    print(f"\nA2 - A0 per seed : {[round(x, 6) for x in deltas]}")
    print(f"A2 - A0 mean delta: {mean_delta:.6f}")
    print(f"Non-inferiority (mean delta > margin {MARGIN}): {mean_delta > MARGIN}")
    print("\nNOTE: formal patient-level paired bootstrap (rows.json.gz) to follow.")


if __name__ == "__main__":
    main()
