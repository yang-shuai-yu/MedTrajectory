"""Recompute the same-day adjacent-pair fraction on the FULL test split.

The draft asserts 93.3% (event level), 29.6% (Design B), ~1.0% (Design A) and
~1.0% (Design C).  A 400-patient sample gave 93.5 / 32.4 / 0.9 / 1.6, so the full
split must be measured before those numbers are quoted again.  This is the central
motivating quantity of the whole analysis, so it is worth getting exactly right.

Also reported: the fraction restricted to the matched Bm cohorts used in
Experiment 1, and for the relaxed cohort, since the reported fractions should
describe the cohorts actually analysed.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import csv
import json
from pathlib import Path

import numpy as np

DATA = Path(str(MIMIC_ROOT))

REPS = [
    ("Event-level", "multitype"),
    ("Design B (principal dx + principal proc)", "visit_B_m3_trackr"),
    ("Design A (principal dx only)", "visit_A_trackr"),
    ("Design C (first-2-code composite)", "visit_C_trackr"),
]


def fractions(ddir: str, split: str, only=None):
    f = DATA / ddir / f"{split}.bin"
    if not f.exists():
        return None
    arr = np.memmap(f, dtype=np.uint32, mode="r").reshape(-1, 3)
    index = DATA / ddir / f"{split}_patient_index.csv"
    eids = None
    if index.exists():
        with index.open(encoding="utf-8", newline="") as fh:
            eids = [str(r["eid"]) for r in csv.DictReader(fh)]
    same = total = 0
    patients = 0
    kept = 0
    i = 0
    n = arr.shape[0]
    while i < n:
        pid = arr[i, 0]
        j = i
        while j < n and arr[j, 0] == pid:
            j += 1
        patients += 1
        use = True
        if only is not None:
            use = eids is not None and pid < len(eids) and eids[pid] in only
        if use:
            kept += 1
            ages = arr[i:j, 1].astype(np.int64)
            if len(ages) > 1:
                same += int((np.diff(ages) == 0).sum())
                total += len(ages) - 1
        i = j
    return same, total, patients, kept


print("=== FULL test split ===")
print(f"{'representation':44s} {'same-day':>9s} {'pairs':>12s} {'patients':>9s}")
for label, ddir in REPS:
    r = fractions(ddir, "test")
    if r is None:
        print(f"{label:44s} test.bin missing")
        continue
    same, total, patients, _ = r
    print(f"{label:44s} {same / total:8.2%} {same:6d}/{total:<6d} {patients:>9d}")

print("\n=== restricted to the matched Bm UKB cohort (541 eids) ===")
ukb = DATA / "matched_bm_ukb.txt"
if ukb.exists():
    ids = {l.strip() for l in ukb.read_text().splitlines() if l.strip()}
    for label, ddir in REPS[:2]:
        r = fractions(ddir, "test", ids)
        if r is None:
            continue
        same, total, patients, kept = r
        frac = same / total if total else float("nan")
        print(f"{label:44s} {frac:8.2%} {same:6d}/{total:<6d} kept={kept}")

print("\n=== full TRAIN split (for reference) ===")
for label, ddir in REPS:
    r = fractions(ddir, "train")
    if r is None:
        print(f"{label:44s} train.bin missing")
        continue
    same, total, patients, _ = r
    print(f"{label:44s} {same / total:8.2%} {same:6d}/{total:<6d} {patients:>9d}")
