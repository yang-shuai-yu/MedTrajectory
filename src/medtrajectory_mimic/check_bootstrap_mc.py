# -*- coding: utf-8 -*-
"""Monte Carlo stability of the reported bootstrap confidence intervals.

The reported CIs are percentile intervals from 10,000 participant-level bootstrap
replicates drawn with a fixed seed (20260819).  A reviewer may reasonably ask how
much of the interval is Monte Carlo noise.  This re-runs the principal contrasts
under several different bootstrap seeds and reports the spread of the CI
endpoints, holding the data and the statistical procedure fixed.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import sys

sys.path.insert(0, str(MIMIC_ROOT))

import numpy as np  # noqa: E402

import mimic_paired_3seed as P  # noqa: E402

SEEDS = ["", "_s43", "_s44"]
BOOT_SEEDS = [20260819, 1, 2, 3, 4]


def arm(tag, stem, ddir, eid):
    return [P.load_rows(tag, stem + s, ddir if eid else None) for s in SEEDS]


def run(label, left, right, ids, metric):
    rows = []
    for bs in BOOT_SEEDS:
        P.BOOTSTRAP_SEED = bs
        res = P.paired_analysis(left, right, ids, metric)
        if res is None:
            print(f"{label}: NA")
            return
        rows.append((res["ci_low"], res["ci_high"], res["point"]))
    lows = [r[0] for r in rows]
    highs = [r[1] for r in rows]
    print(f"{label}")
    print(f"   point estimate      : {rows[0][2]:+.4f}   (identical across seeds by construction)")
    print(f"   ci95_low  spread    : {min(lows):+.4f} .. {max(lows):+.4f}  "
          f"(range {max(lows) - min(lows):.4f})")
    print(f"   ci95_high spread    : {min(highs):+.4f} .. {max(highs):+.4f}  "
          f"(range {max(highs) - min(highs):.4f})")


ids = None
for metric in ("hit_at_10", "diagnosis_jaccard", "event_count_mae"):
    m1 = arm("eval_bm3_ukb", "bm_m1_a0", "visit_Bm_m1_trackr", True)
    m2 = arm("eval_bm3_ukb", "bm_m2_a0", "visit_Bm_m2_trackr", True)
    ids = sorted(set(m1[0]) & set(m2[0]))
    run(f"Exp1 Δ(M2−M1) {metric}, UKB-aligned, n={len(ids)}", m2, m1, ids, metric)

P.BOOTSTRAP_SEED = 20260819
for metric in ("event_count_mae", "sequence_edit_distance", "diagnosis_jaccard"):
    a0 = arm("eval_visitA_seeds_ukb", "vA_a0", None, False)
    a2 = arm("eval_visitA_seeds_ukb", "vA_a2", None, False)
    ids = sorted(set(a0[0]) & set(a2[0]))
    run(f"Exp2 Design A Δ(A2−A0) {metric}, UKB-aligned, n={len(ids)}", a2, a0, ids, metric)

P.BOOTSTRAP_SEED = 20260819
for metric in ("diagnosis_jaccard", "hit_at_10"):
    a2 = arm("eval_visitA_seeds_relaxed", "vA_a2", None, False)
    a0 = arm("eval_visitA_seeds_relaxed", "vA_a0", None, False)
    ids = sorted(set(a2[0]) & set(a0[0]))
    run(f"Exp2 Design A Δ(A2−A0) {metric}, relaxed, n={len(ids)}", a2, a0, ids, metric)
