# -*- coding: utf-8 -*-
"""Three-seed PAIRED analysis for the MIMIC-IV cross-setting replication.

Why this script exists
----------------------
`mimic_analyze_bm_multitype.py` and `mimic_analyze_3seed_generic.py` report
3-seed means and deltas-of-means but perform no paired inference, and they use
`pstdev` (population SD, ddof=0) whereas the UK Biobank generation table in the
paper uses the SAMPLE SD (ddof=1, verified numerically against Supplementary
Table S5).  This script supplies:

  * patient alignment by eid, not by patient_index.  patient_index is a position
    into the representation's own .bin; the three Bm representations live in
    three different data dirs whose orderings differ (only 132/9519 eids sit at
    the same position in visit_Bm_m1 vs visit_Bm_m2), so aligning on
    patient_index silently compares different people.
  * a point estimate = unweighted mean of the per-seed paired differences
    (identical to the patient-then-seed order because the participant sets match)
  * between-seed sample SD (ddof=1), matching the UKB convention
  * a patient-level percentile bootstrap with the SAME resample used for all
    seeds within a replicate (the Track G `finalize_track_g.py` scheme)
  * a paired Wilcoxon signed-rank test on the per-participant seed-averaged
    deltas, as a secondary p-value

Nothing here changes any evaluation output; it only re-analyses patient_rows.jsonl.
"""
from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Sequence

import numpy as np

DATA = Path(str(MIMIC_ROOT))

METRICS = [
    ("hit_at_10", "Hit@10", "higher"),
    ("hit_at_1", "Hit@1", "higher"),
    ("diagnosis_jaccard", "Diag Jaccard", "higher"),
    ("diagnosis_recall", "Diag Recall", "higher"),
    ("first_event_time_mae_days", "First-event MAE (d)", "lower"),
    ("event_count_mae", "Event-count MAE", "lower"),
    ("sequence_edit_distance", "Sequence edit", "lower"),
]
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260819
PERCENTILE = (2.5, 97.5)

_index_cache: dict[str, list[str]] = {}


def index_to_eid(data_dir: str, split: str = "test") -> list[str] | None:
    key = f"{data_dir}|{split}"
    if key not in _index_cache:
        p = DATA / data_dir / f"{split}_patient_index.csv"
        if not p.exists():
            _index_cache[key] = []
        else:
            with p.open(encoding="utf-8", newline="") as fh:
                _index_cache[key] = [str(r["eid"]) for r in csv.DictReader(fh)]
    return _index_cache[key] or None


def load_rows(tag: str, stem: str, data_dir: str | None) -> dict[str, dict] | None:
    path = DATA / tag / stem / "patient_rows.jsonl"
    if not path.exists():
        return None
    idx = index_to_eid(data_dir) if data_dir else None
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pi = int(row["patient_index"])
            if idx is not None:
                if pi >= len(idx):
                    raise ValueError(f"{tag}/{stem}: patient_index {pi} out of range")
                key = idx[pi]
            else:
                key = str(pi)
            out[key] = row
    return out


def finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))


def paired_analysis(left: Sequence[dict], right: Sequence[dict], ids: Sequence[str],
                    metric: str):
    """left/right are lists (one per seed) of {eid: row}."""
    n_seeds = len(left)
    usable = [k for k in ids
              if all(finite(left[s].get(k, {}).get(metric)) and finite(right[s].get(k, {}).get(metric))
                     for s in range(n_seeds))]
    if not usable:
        return None
    deltas_by_seed = np.array(
        [[float(left[s][k][metric]) - float(right[s][k][metric]) for k in usable]
         for s in range(n_seeds)], dtype=np.float64)
    per_seed = [float(v.mean()) for v in deltas_by_seed]
    point = float(np.mean(per_seed))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(usable)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, n, size=n)
        samples[i] = float(np.mean([v[draw].mean() for v in deltas_by_seed]))
    lo, hi = (float(x) for x in np.percentile(samples, PERCENTILE))

    per_patient = deltas_by_seed.mean(axis=0)
    try:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(per_patient, zero_method="wilcox")
        pval = float(pval)
    except Exception:
        pval = float("nan")
    n_nonzero = int(np.count_nonzero(per_patient))
    return {
        "point": point,
        "per_seed": per_seed,
        "seed_sd_ddof1": statistics.stdev(per_seed) if len(per_seed) > 1 else float("nan"),
        "ci_low": lo, "ci_high": hi,
        "p_wilcoxon": pval,
        "n_patients": n,
        "n_nonzero_pairs": n_nonzero,
    }


def report(label: str, tag: str, arms: dict, contrasts: list, eid_align: bool):
    loaded = {}
    for arm_name, specs in arms.items():
        per_seed = []
        for stem, ddir in specs:
            per_seed.append(load_rows(tag, stem, ddir if eid_align else None))
        loaded[arm_name] = per_seed

    missing = [f"{a}[s{i}]" for a, v in loaded.items() for i, x in enumerate(v) if x is None]
    print(f"\n### {label}")
    print(f"\ntag=`{tag}`  arms={list(arms)}  alignment={'eid' if eid_align else 'patient_index'}")
    if missing:
        print(f"\n**INCOMPLETE — missing: {missing}**")
        return
    for arm_name, seeds in loaded.items():
        sizes = [len(s) for s in seeds]
        print(f"  - {arm_name}: per-seed n = {sizes}")

    for left_name, right_name in contrasts:
        left, right = loaded[left_name], loaded[right_name]
        common = set(left[0])
        for s in left[1:]:
            common &= set(s)
        for s in right:
            common &= set(s)
        ids = sorted(common)
        print(f"\n**Δ = {left_name} − {right_name}**  (common participants = {len(ids)})")
        print()
        print("| Metric | Δ mean | ± seed SD (ddof=1) | 95% CI (patient bootstrap) | per-seed Δ | Wilcoxon p | n |")
        print("|---|---:|---:|---|---|---:|---:|")
        for metric, disp, _direction in METRICS:
            res = paired_analysis(left, right, ids, metric)
            if res is None:
                print(f"| {disp} | NA | NA | NA | NA | NA | 0 |")
                continue
            ps = ", ".join(f"{v:+.4f}" for v in res["per_seed"])
            sd = res["seed_sd_ddof1"]
            sd_s = f"{sd:.4f}" if math.isfinite(sd) else "NA"
            p = res["p_wilcoxon"]
            p_s = f"{p:.3g}" if math.isfinite(p) else "NA"
            print(f"| {disp} | {res['point']:+.4f} | {sd_s} | "
                  f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}] | {ps} | {p_s} | "
                  f"{res['n_patients']} |")


def main():
    print("# MIMIC-IV three-seed PAIRED analysis")
    print()
    print(f"Bootstrap: {BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}, "
          f"percentile {list(PERCENTILE)}; the same participant resample is used for all "
          f"seeds within a replicate.  Between-seed SD is the sample SD (ddof=1).")

    def s(sfx):
        return sfx

    for tag, lab in (("eval_bm3_ukb", "UKB-aligned cohort (>=8 history, >=3 future)"),
                     ("eval_bm3_relaxed", "relaxed cohort (>=3/>=2, capped at 2000)")):
        arms = {
            "M1 diagnosis-only": [(f"bm_m1_a0{s('') if i == 0 else f'_s{42 + i}'}", "visit_Bm_m1_trackr")
                                  for i in range(3)],
            "M2 +procedure": [(f"bm_m2_a0{s('') if i == 0 else f'_s{42 + i}'}", "visit_Bm_m2_trackr")
                              for i in range(3)],
            "M3 +death": [(f"bm_m3_a0{s('') if i == 0 else f'_s{42 + i}'}", "visit_Bm_m3_trackr")
                          for i in range(3)],
        }
        report(f"Experiment 1 — matched multitype, {lab}", tag, arms,
               [("M2 +procedure", "M1 diagnosis-only"), ("M3 +death", "M2 +procedure")], eid_align=True)

    for tag, lab in (("eval_B3_ukb", "UKB-aligned cohort"),
                     ("eval_B3_relaxed", "relaxed cohort")):
        arms = {"A0 absolute": [(f"vB_m3_a0{s('') if i == 0 else f'_s{42 + i}'}", None) for i in range(3)],
                "A2 relative": [(f"vB_m3_a2{s('') if i == 0 else f'_s{42 + i}'}", None) for i in range(3)]}
        report(f"Experiment 2 — Design B (principal dx + principal procedure), {lab}", tag, arms,
               [("A2 relative", "A0 absolute")], eid_align=False)

    for tag, lab in (("eval_C3_ukb", "UKB-aligned cohort"),
                     ("eval_C3_relaxed", "relaxed cohort")):
        arms = {"A0 absolute": [(f"vC_a0{s('') if i == 0 else f'_s{42 + i}'}", None) for i in range(3)],
                "A2 relative": [(f"vC_a2{s('') if i == 0 else f'_s{42 + i}'}", None) for i in range(3)]}
        report(f"Experiment 2 — Design C (first-2-code composite), {lab}", tag, arms,
               [("A2 relative", "A0 absolute")], eid_align=False)

    for tag, lab in (("eval_visitA_seeds_ukb", "UKB-aligned cohort"),
                     ("eval_visitA_seeds_relaxed", "relaxed cohort")):
        arms = {"A0 absolute": [(f"vA_a0{s('') if i == 0 else f'_s{42 + i}'}", None) for i in range(3)],
                "A2 relative": [(f"vA_a2{s('') if i == 0 else f'_s{42 + i}'}", None) for i in range(3)]}
        report(f"Experiment 2 — Design A (principal diagnosis only), {lab}", tag, arms,
               [("A2 relative", "A0 absolute")], eid_align=False)

    # Event level exists for seed 42 only; both pairs share one data dir, so
    # patient_index is comparable within each pair.
    report("Experiment 2 — event level (multitype M3, seed 42 only)", "eval_quick",
           {"A0 absolute": [("m3_abs", None)], "A2 relative": [("m3_rel", None)]},
           [("A2 relative", "A0 absolute")], eid_align=False)
    report("Experiment 2 — event level (Track-R, seed 42 only)", "eval_trackr",
           {"A0 absolute": [("trackr_a0_abs", None)], "A2 relative": [("trackr_a2_rel", None)]},
           [("A2 relative", "A0 absolute")], eid_align=False)


if __name__ == "__main__":
    main()
