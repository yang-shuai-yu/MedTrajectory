"""Estimate the remaining evaluation wall time from completed patient counts."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))

print("=== completed summaries ===")
for p in sorted(DATA.glob("eval_*/*/summary.json")):
    d = json.loads(p.read_text())
    keys = {k: d.get(k) for k in ("patient_count", "num_rollouts", "split", "cohort") if k in d}
    m = d.get("metrics", d)
    print(f"  {p.parent.parent.name:18s}/{p.parent.name:14s} {keys}  "
          f"jaccard={m.get('diagnosis_jaccard')}")
    top = list(d)[:12]
    if p.parent.parent.name == "eval_bm3_ukb" and p.parent.name == "bm_m1_a0":
        print(f"      full keys: {top}")

print("\n=== case counts for the not-yet-run tags (from the test splits) ===")
import numpy as np

def split_sizes(ddir):
    man = json.loads((DATA / ddir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    out = {}
    for split in ("train", "val", "test"):
        p = DATA / ddir / f"{split}_patient_ids.npy"
        if p.exists():
            out[split] = int(len(np.load(p)))
    return out

for ddir in ("visit_B_m3_trackr", "visit_C_trackr",
             "visit_Bm_m1_trackr", "visit_Bm_m2_trackr", "visit_Bm_m3_trackr"):
    try:
        print(f"  {ddir:22s} {split_sizes(ddir)}")
    except Exception as exc:
        print(f"  {ddir:22s} ERROR {exc}")

for name in ("matched_bm_ukb.txt", "matched_bm_relaxed.txt"):
    p = DATA / name
    if p.exists():
        n = len([l for l in p.read_text().split() if l.strip()])
        print(f"  {name:22s} {n} participants")
