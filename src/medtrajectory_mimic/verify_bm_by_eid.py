"""Compare Bm representations by real eid, not by positional patient_index.

patient_index is a position into each representation's own .bin, so it is only
comparable across runs that share a data dir.  For the Bm multitype comparison the
three representations live in three different data dirs (visit_Bm_m{1,2,3}_trackr),
so the check must map patient_index -> eid through that dir's
{split}_patient_index.csv before comparing participant sets.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import csv
import json
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))


def index_to_eid(data_dir: Path, split: str):
    p = data_dir / f"{split}_patient_index.csv"
    if not p.exists():
        return None
    with p.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [str(r["eid"]) for r in rows]


def used_eids(tag: str, stem: str, data_dir: Path):
    rows = DATA / tag / stem / "patient_rows.jsonl"
    if not rows.exists():
        return None
    idx = index_to_eid(DATA / data_dir, "test")
    if idx is None:
        return "NO_INDEX_CSV"
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


GROUPS = []
for seed in (42, 43, 44):
    suffix = "" if seed == 42 else f"_s{seed}"
    GROUPS.append((f"seed{seed}", [
        (f"bm_m1_a0{suffix}", "visit_Bm_m1_trackr"),
        (f"bm_m2_a0{suffix}", "visit_Bm_m2_trackr"),
        (f"bm_m3_a0{suffix}", "visit_Bm_m3_trackr"),
    ]))

print("=== eid-based cohort check for Experiment 1 (eval_bm3_ukb) ===")
for seed_label, specs in GROUPS:
    got = {}
    for stem, ddir in specs:
        got[stem] = used_eids("eval_bm3_ukb", stem, ddir)
    if any(v is None for v in got.values()):
        print(f"  PENDING {seed_label}: {[k for k, v in got.items() if v is None]}")
        continue
    if any(isinstance(v, str) for v in got.values()):
        print(f"  ERROR   {seed_label}: {got}")
        continue
    sizes = {k: len(v) for k, v in got.items()}
    keys = list(got)
    inter = set.intersection(*got.values())
    union = set.union(*got.values())
    same = all(got[k] == got[keys[0]] for k in keys)
    status = "OK  " if same else "FAIL"
    print(f"  {status} {seed_label}: sizes={sizes} intersection={len(inter)} union={len(union)} "
          f"identical={same}")

print("\n=== sanity: do the three data dirs share the same index ordering? ===")
ref = None
for ddir in ("visit_Bm_m1_trackr", "visit_Bm_m2_trackr", "visit_Bm_m3_trackr"):
    idx = index_to_eid(DATA / ddir, "test")
    if idx is None:
        print(f"  {ddir}: no test_patient_index.csv")
        continue
    if ref is None:
        ref, ref_name = idx, ddir
        print(f"  {ddir}: {len(idx)} rows (reference)")
    else:
        shared_positions = sum(1 for a, b in zip(ref, idx) if a == b)
        print(f"  {ddir}: {len(idx)} rows, same-eid-same-position = {shared_positions}/{min(len(ref), len(idx))}")

mb = DATA / "matched_bm_ukb.txt"
if mb.exists():
    ids = {l.strip() for l in mb.read_text().splitlines() if l.strip()}
    print(f"\nmatched_bm_ukb.txt has {len(ids)} eids")
