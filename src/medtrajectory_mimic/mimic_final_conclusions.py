# -*- coding: utf-8 -*-
"""Assemble the final MIMIC-IV cross-setting replication conclusions.

One command that (1) runs the eid-level cohort-match gate, (2) runs the three-seed
paired analysis for every claim and both cohort definitions, and (3) emits the
final conclusions as markdown.  Groups whose evaluations are not yet complete are
marked INCOMPLETE rather than silently skipped.

Usage:
    python mimic_final_conclusions.py > FINAL_CONCLUSIONS.md
"""
from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT

import io
import contextlib
import subprocess
import sys

sys.path.insert(0, str(MIMIC_ROOT))

from mimic_paired_3seed import DATA, METRICS, paired_analysis, load_rows  # noqa: E402

SEEDS = ["", "_s43", "_s44"]

# (representation label, data-dir per arm, note)
BM_ARMS = {
    "M1 diagnosis-only": ("bm_m1_a0", "visit_Bm_m1_trackr"),
    "M2 +principal procedure": ("bm_m2_a0", "visit_Bm_m2_trackr"),
    "M3 +death": ("bm_m3_a0", "visit_Bm_m3_trackr"),
}

COHORTS = (
    ("eval_bm3_ukb", "UKB-aligned (>=8 history, >=3 future)"),
    ("eval_bm3_relaxed", "relaxed (>=3/>=2, capped at 2000)"),
)
COHORTS_B = (
    ("eval_B3_ukb", "UKB-aligned"),
    ("eval_B3_relaxed", "relaxed"),
)
COHORTS_C = (
    ("eval_C3_ukb", "UKB-aligned"),
    ("eval_C3_relaxed", "relaxed"),
)
COHORTS_A = (
    ("eval_visitA_seeds_ukb", "UKB-aligned"),
    ("eval_visitA_seeds_relaxed", "relaxed"),
)


def load_arm(tag, stem, ddir, eid_align):
    return [load_rows(tag, stem + sfx, ddir if eid_align else None) for sfx in SEEDS]


def common_ids(*arms):
    """Intersection of participant keys across seeds and arms."""
    ok = True
    ids = None
    for arm in arms:
        for seeded in arm:
            if seeded is None:
                ok = False
                continue
            ids = set(seeded) if ids is None else (ids & set(seeded))
    return (sorted(ids) if ids else []), ok


def emit_contrast(title, left_name, left, right_name, right, ids):
    print(f"\n**Δ = {left_name} − {right_name}**  (n = {len(ids)})\n")
    print("| Metric | Effect | ± seed SD (ddof=1) | 95% CI | per-seed Δ | p (Wilcoxon) |")
    print("|---|---:|---:|---|---|---:|")
    # a single-run baseline (ETHOS-Matched) is repeated so every CARoPE seed is
    # compared against the same fixed run
    right_per_seed = right * len(left) if len(right) == 1 else right
    for metric, disp, _d in METRICS:
        res = paired_analysis(left, right_per_seed, ids, metric)
        if res is None:
            print(f"| {disp} | NA | NA | NA | NA | NA |")
            continue
        ss = res["seed_sd_ddof1"]
        sd_s = "NA" if ss != ss else f"{ss:.4f}"
        p = res["p_wilcoxon"]
        p_s = "NA" if p != p else f"{p:.3g}"
        per = ", ".join(f"{v:+.4f}" for v in res["per_seed"])
        print(f"| {disp} | {res['point']:+.4f} | {sd_s} | "
              f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}] | {per} | {p_s} |")


def main():
    print("# MIMIC-IV cross-setting replication — FINAL conclusions")
    print()
    print("Three-seed (42/43/44) paired analysis, UKB-aligned cohort as the primary "
          "protocol-matched analysis and the relaxed cohort as a sensitivity check.")
    print()

    # ---- step 1: cohort-match gate ----
    print("## 0. Cohort-match gate (`verify_eval_cohorts.py`, alignment by eid)")
    print()
    proc = subprocess.run(
        [sys.executable, str(MIMIC_ROOT / "verify_eval_cohorts.py")],
        capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        print("    " + line)
    if proc.returncode != 0:
        print()
        print("**GATE FAILED — no conclusion below may be used until this passes.**")
    print()

    # ---- conclusion 1: matched multitype ----
    print("## Conclusion 1 — matched multitype (Δ = M2 − M1)")
    print()
    for tag, label in COHORTS:
        arms = {n: load_arm(tag, stem, ddir, True) for n, (stem, ddir) in BM_ARMS.items()}
        ids, ok = common_ids(*arms.values())
        print(f"\n### {label}  (`{tag}`)\n")
        if not ok or not ids:
            print("**INCOMPLETE — evaluation not finished**")
            continue
        print("participants per arm: " + ", ".join(
            f"{n}={len(a[0])}" if a[0] else f"{n}=?" for n, a in arms.items()))
        emit_contrast("", "M2 +procedure", arms["M2 +principal procedure"],
                      "M1 diagnosis-only", arms["M1 diagnosis-only"], ids)
        emit_contrast("", "M3 +death", arms["M3 +death"],
                      "M2 +procedure", arms["M2 +principal procedure"], ids)

    # ---- conclusion 2: relative time ----
    print("\n## Conclusion 2 — continuous relative time (Δ = A2 − A0)")
    print()
    rep_specs = [
        ("Event level (seed 42 only)", "eval_quick", "m3_abs", "m3_rel", None),
        ("Event level Track-R (seed 42 only)", "eval_trackr", "trackr_a0_abs", "trackr_a2_rel", None),
        ("Design A", None, "vA_a0", "vA_a2", COHORTS_A),
        ("Design B", None, "vB_m3_a0", "vB_m3_a2", COHORTS_B),
        ("Design C", None, "vC_a0", "vC_a2", COHORTS_C),
    ]
    for label, fixed_tag, s0, s2, cohort_list in rep_specs:
        if fixed_tag is not None:
            a0 = [load_rows(fixed_tag, s0, None)]
            a2 = [load_rows(fixed_tag, s2, None)]
            ids, ok = common_ids(a0, a2)
            print(f"\n### {label}\n")
            if not ok or not ids:
                print("**INCOMPLETE**")
                continue
            emit_contrast("", "A2 relative", a2, "A0 absolute", a0, ids)
            continue
        for tag, clabel in cohort_list:
            a0 = load_arm(tag, s0, None, False)
            a2 = load_arm(tag, s2, None, False)
            ids, ok = common_ids(a0, a2)
            print(f"\n### {label} — {clabel}  (`{tag}`)\n")
            if not ok or not ids:
                print("**INCOMPLETE — evaluation not finished**")
                continue
            emit_contrast("", "A2 relative", a2, "A0 absolute", a0, ids)

    # ---- conclusion 3: ETHOS baseline ----
    print("\n## Conclusion 3 — CARoPE arms vs ETHOS-Matched")
    print()
    for (t_ethos, clabel), (t_car, _) in zip(
            (("eval_ethos_ukb", "UKB-aligned"), ("eval_ethos_relaxed", "relaxed")),
            COHORTS_A):
        ethos = load_rows(t_ethos, "ethos_matched_a", None)
        print(f"\n### {clabel}  (`{t_ethos}` vs `{t_car}`)\n")
        if ethos is None:
            print("**INCOMPLETE — ETHOS eval missing**")
            continue
        for name, stem in (("A2 relative", "vA_a2"), ("A0 absolute", "vA_a0")):
            arm = load_arm(t_car, stem, None, False)
            ids, ok = common_ids(arm, [ethos])
            if not ok or not ids:
                print(f"**{name}: INCOMPLETE**")
                continue
            emit_contrast("", name, arm, "ETHOS-Matched", [ethos], ids)
    print()


if __name__ == "__main__":
    main()
