# -*- coding: utf-8 -*-
"""Explain a bootstrap-CI / Wilcoxon disagreement seen in the relaxed cohort.

For the relaxed multitype comparison, Diagnosis Jaccard has a bootstrap 95% CI
that excludes zero ([+0.0019, +0.0080]) while the paired Wilcoxon p is 0.551.
The two tests answer different questions: the bootstrap tests the MEAN of the
per-participant differences, the Wilcoxon signed-rank tests their signed ranks
and discards zero differences.  This script prints how many participants have a
non-zero difference, which is what drives the discrepancy.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import sys

sys.path.insert(0, str(MIMIC_ROOT))

import mimic_paired_3seed as P  # noqa: E402

SEEDS = ["", "_s43", "_s44"]


def arm(tag, stem, ddir, eid):
    return [P.load_rows(tag, stem + s, ddir if eid else None) for s in SEEDS]


def report(label, left, right, ids, metric):
    res = P.paired_analysis(left, right, ids, metric)
    if res is None:
        print(f"{label}: NA")
        return
    frac = res["n_nonzero_pairs"] / res["n_patients"]
    print(f"{label}")
    print(f"   mean Δ            : {res['point']:+.4f}   CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
          f"   Wilcoxon p = {res['p_wilcoxon']:.3g}")
    print(f"   participants      : {res['n_patients']}")
    print(f"   non-zero paired Δ : {res['n_nonzero_pairs']}  ({frac:.1%})")


bm = {"M1": ("bm_m1_a0", "visit_Bm_m1_trackr"),
      "M2": ("bm_m2_a0", "visit_Bm_m2_trackr"),
      "M3": ("bm_m3_a0", "visit_Bm_m3_trackr")}

for tag, label in (("eval_bm3_ukb", "UKB-aligned"), ("eval_bm3_relaxed", "relaxed")):
    arms = {k: arm(tag, *v, True) for k, v in bm.items()}
    ids = sorted(set(arms["M1"][0]) & set(arms["M2"][0]))
    for metric in ("diagnosis_jaccard", "hit_at_10", "event_count_mae"):
        report(f"[{label}] Δ(M2−M1) {metric}, n={len(ids)}",
               arms["M2"], arms["M1"], ids, metric)
