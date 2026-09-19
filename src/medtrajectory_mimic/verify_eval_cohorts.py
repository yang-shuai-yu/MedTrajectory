"""Verify that every model compared in results_bc_3seeds.md was evaluated on an
IDENTICAL participant set.

Two traps this guards against:

1. Equal case counts do not prove equal case membership.  Without --eid-file the
   eligibility filter (>=8 history + >=3 future events) selects 541 participants
   for the diagnosis-only representation but 837 for the procedure-bearing ones,
   and the extra participants have more events by construction.

2. patient_index is a position into the representation's own .bin, so it is only
   comparable between runs that share a data dir.  The three Bm representations
   live in three different dirs with different participant orderings (only
   132/9519 eids sit at the same position in visit_Bm_m1 vs visit_Bm_m2), so the
   comparison must map patient_index -> eid through each dir's
   {split}_patient_index.csv.  Comparing raw patient_index produces a spurious
   "intersection = 36" failure.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import csv
import json
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))
SPLIT = "test"

# (tag, seed, [(stem, data_dir), ...])
GROUPS = []
for tag in ("eval_bm3_ukb", "eval_bm3_relaxed"):
    for seed in (42, 43, 44):
        suffix = "" if seed == 42 else f"_s{seed}"
        GROUPS.append((tag, f"seed{seed}", [
            (f"bm_m1_a0{suffix}", "visit_Bm_m1_trackr"),
            (f"bm_m2_a0{suffix}", "visit_Bm_m2_trackr"),
            (f"bm_m3_a0{suffix}", "visit_Bm_m3_trackr"),
        ]))
for seed in (42, 43, 44):
    suffix = "" if seed == 42 else f"_s{seed}"
    GROUPS.append(("eval_B3_ukb", f"seed{seed}",
                   [(f"vB_m3_a0{suffix}", "visit_B_m3_trackr"),
                    (f"vB_m3_a2{suffix}", "visit_B_m3_trackr")]))
    GROUPS.append(("eval_B3_relaxed", f"seed{seed}",
                   [(f"vB_m3_a0{suffix}", "visit_B_m3_trackr"),
                    (f"vB_m3_a2{suffix}", "visit_B_m3_trackr")]))
    GROUPS.append(("eval_C3_ukb", f"seed{seed}",
                   [(f"vC_a0{suffix}", "visit_C_trackr"),
                    (f"vC_a2{suffix}", "visit_C_trackr")]))
    GROUPS.append(("eval_C3_relaxed", f"seed{seed}",
                   [(f"vC_a0{suffix}", "visit_C_trackr"),
                    (f"vC_a2{suffix}", "visit_C_trackr")]))

_index_cache: dict[Path, list[str]] = {}


def index_to_eid(data_dir: Path):
    if data_dir not in _index_cache:
        p = data_dir / f"{SPLIT}_patient_index.csv"
        if not p.exists():
            _index_cache[data_dir] = []
        else:
            with p.open(encoding="utf-8", newline="") as fh:
                _index_cache[data_dir] = [str(r["eid"]) for r in csv.DictReader(fh)]
    return _index_cache[data_dir]


def used_eids(tag: str, stem: str, data_dir: str):
    rows = DATA / tag / stem / "patient_rows.jsonl"
    if not rows.exists():
        return None
    idx = index_to_eid(DATA / data_dir)
    if not idx:
        return f"NO_INDEX_CSV({data_dir})"
    out = set()
    with rows.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pi = int(json.loads(line)["patient_index"])
            if pi >= len(idx):
                return f"INDEX_OUT_OF_RANGE({pi}>={len(idx)})"
            out.add(idx[pi])
    return out


failures = 0
pending = 0
print("=== cohort-match verification (by eid) ===")
for tag, seed_label, specs in GROUPS:
    got = {stem: used_eids(tag, stem, ddir) for stem, ddir in specs}
    if any(v is None for v in got.values()):
        pending += 1
        print(f"  PENDING {tag}/{seed_label}: missing "
              f"{[k for k, v in got.items() if v is None]}")
        continue
    if any(isinstance(v, str) for v in got.values()):
        failures += 1
        print(f"  ERROR   {tag}/{seed_label}: {got}")
        continue
    keys = list(got)
    ref = got[keys[0]]
    same = all(got[k] == ref for k in keys)
    sizes = {k: len(v) for k, v in got.items()}
    if same:
        print(f"  OK      {tag}/{seed_label}: n={len(ref)} identical  {sizes}")
    else:
        failures += 1
        inter = set.intersection(*got.values())
        union = set.union(*got.values())
        print(f"  FAIL    {tag}/{seed_label}: sets differ  {sizes}  "
              f"intersection={len(inter)} union={len(union)}")

print()
if failures:
    print(f"RESULT: {failures} cohort mismatch(es) -- these comparisons are NOT paired")
    raise SystemExit(1)
if pending:
    print(f"RESULT: {pending} group(s) still pending; no mismatch detected so far")
    raise SystemExit(0)
print("RESULT: every compared model used an identical participant set")
