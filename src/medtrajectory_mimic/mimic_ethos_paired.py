# -*- coding: utf-8 -*-
"""Paired comparison of the CARoPE arms against the ETHOS-Matched generative baseline.

ETHOS-Matched was evaluated on the same data dir (visit_A_trackr) and the same
participant set (574 UKB-aligned / 2000 relaxed) as the Design A arms, so
patient_index is comparable and a patient-level paired analysis is valid.

ETHOS-Matched has only one trained run, so there is no between-seed SD for it;
the three CARoPE seeds are each compared against that single baseline and the
per-seed deltas are reported alongside the pooled estimate.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import sys

sys.path.insert(0, str(MIMIC_ROOT))

from mimic_paired_3seed import METRICS, paired_analysis, load_rows, DATA  # noqa: E402

SEEDS = ["", "_s43", "_s44"]


def run(tag_ethos, tag_carope, label):
    print(f"\n### {label}")
    print(f"\ntag_ethos=`{tag_ethos}` tag_carope=`{tag_carope}`  alignment=patient_index")
    ethos = load_rows(tag_ethos, "ethos_matched_a", None)
    if ethos is None:
        print("\n**ETHOS eval missing**")
        return
    print(f"  - ETHOS-Matched: n = {len(ethos)} (single run)")
    arms = {}
    for name, stem in (("A2 relative", "vA_a2"), ("A0 absolute", "vA_a0")):
        arms[name] = [load_rows(tag_carope, stem + sfx, None) for sfx in SEEDS]
        sizes = [len(x) if x else None for x in arms[name]]
        print(f"  - {name}: per-seed n = {sizes}")

    for name, arm in arms.items():
        if any(x is None for x in arm):
            print(f"\n**{name}: incomplete**")
            continue
        ids = sorted(set(ethos) & set(arm[0]))
        for s in arm[1:]:
            ids = sorted(set(ids) & set(s))
        print(f"\n**Δ = {name} − ETHOS-Matched**  (common participants = {len(ids)})")
        print()
        print("| Metric | Δ mean | ± seed SD (ddof=1) | 95% CI | per-seed Δ | Wilcoxon p | n |")
        print("|---|---:|---:|---|---|---:|---:|")
        for metric, disp, _d in METRICS:
            res = paired_analysis(arm, [ethos] * len(arm), ids, metric)
            if res is None:
                print(f"| {disp} | NA | NA | NA | NA | NA | 0 |")
                continue
            ps = ", ".join(f"{v:+.4f}" for v in res["per_seed"])
            sd = res["seed_sd_ddof1"]
            sd_s = f"{sd:.4f}" if sd == sd else "NA"
            p = res["p_wilcoxon"]
            p_s = f"{p:.3g}" if p == p else "NA"
            print(f"| {disp} | {res['point']:+.4f} | {sd_s} | "
                  f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}] | {ps} | {p_s} | {res['n_patients']} |")


print("# CARoPE arms vs the ETHOS-Matched baseline (paired, patient level)")
print()
print("Note: ETHOS-Matched is a single trained run, so its own seed variability is "
      "not observable; the three CARoPE seeds are compared against it independently.")
run("eval_ethos_ukb", "eval_visitA_seeds_ukb", "UKB-aligned cohort (>=8 history, >=3 future)")
run("eval_ethos_relaxed", "eval_visitA_seeds_relaxed", "relaxed cohort (>=3/>=2, capped at 2000)")
