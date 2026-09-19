"""Verify the numbers asserted in the draft's Data section against the actual data.

Checks vocabulary sizes for the four representations, plus the raw MIMIC-IV counts
if the source CSV headers are still available.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import csv
import json
from collections import Counter
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))

REPS = [
    ("Event-level", "multitype"),
    ("Design B (principal dx + principal proc)", "visit_B_m3_trackr"),
    ("Design A (principal dx only)", "visit_A_trackr"),
    ("Design C (first-2-code composite)", "visit_C_trackr"),
]

print("=== vocabulary sizes asserted vs actual ===")
ASSERTED = {"Event-level": 4152, "Design B (principal dx + principal proc)": 4152,
            "Design A (principal dx only)": 1786,
            "Design C (first-2-code composite)": 3192}
for label, ddir in REPS:
    d = DATA / ddir
    man_path = d / "prepare_manifest.json"
    if not man_path.exists():
        print(f"  {label:44s} {ddir}: no prepare_manifest.json")
        continue
    man = json.loads(man_path.read_text(encoding="utf-8-sig"))
    vs = man.get("vocab_size")
    vcsv = man.get("vocab_csv")
    rows = None
    cand = d / str(vcsv) if vcsv else None
    if cand and cand.exists():
        rows = sum(1 for _ in csv.DictReader(cand.open(encoding="utf-8", newline="")))
    note = ""
    if rows is not None and vs is not None:
        note = f" (vocab_size - 1 = {vs - 1}, csv rows = {rows})"
    a = ASSERTED.get(label)
    flag = "OK " if (a is not None and (vs == a or rows == a or (vs is not None and vs - 1 == a))) else "?? "
    print(f"  {flag}{label:44s} asserted={a}  manifest vocab_size={vs}{note}")

print("\n=== event-type composition per representation ===")
for label, ddir in REPS:
    d = DATA / ddir
    man_path = d / "prepare_manifest.json"
    if not man_path.exists():
        continue
    man = json.loads(man_path.read_text(encoding="utf-8-sig"))
    vcsv = man.get("vocab_csv")
    cand = d / str(vcsv) if vcsv else None
    if not (cand and cand.exists()):
        print(f"  {label:44s} vocab csv not found")
        continue
    rows = list(csv.DictReader(cand.open(encoding="utf-8", newline="")))
    et = Counter(r.get("event_type", "?") for r in rows)
    print(f"  {label:44s} {dict(et)}")

print("\n=== same-day adjacent-pair fraction (test split, first 400 patients) ===")
import numpy as np

for label, ddir in REPS:
    d = DATA / ddir
    f = d / "test.bin"
    if not f.exists():
        print(f"  {label:44s} test.bin missing")
        continue
    arr = np.memmap(f, dtype=np.uint32, mode="r").reshape(-1, 3)
    start = 0
    same = total = 0
    patients = 0
    i = 0
    while i < arr.shape[0] and patients < 400:
        pid = arr[i, 0]
        j = i
        while j < arr.shape[0] and arr[j, 0] == pid:
            j += 1
        ages = arr[i:j, 1].astype(np.int64)
        if len(ages) > 1:
            same += int((np.diff(ages) == 0).sum())
            total += len(ages) - 1
        patients += 1
        i = j
    frac = same / total if total else float("nan")
    print(f"  {label:44s} {frac:.1%}  ({same}/{total} adjacent pairs, {patients} patients)")
