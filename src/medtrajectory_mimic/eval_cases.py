"""Case counts for completed Bm evals, to calibrate the remaining ETA."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))

print("=== num_cases for completed evals ===")
for tag in ("eval_bm3_ukb", "eval_bm3_relaxed"):
    d = DATA / tag
    for p in sorted(d.glob("*/summary.json")):
        blob = json.loads(p.read_text())
        m = blob.get("metrics", {})
        print(f"  {tag:18s}/{p.parent.name:14s} num_cases={blob.get('num_cases')}  "
              f"jaccard={m.get('diagnosis_jaccard'):.6f}  hit@10={m.get('hit_at_10')}")

print("\n=== full metric dict of one completed eval ===")
p = DATA / "eval_bm3_ukb" / "bm_m1_a0" / "summary.json"
blob = json.loads(p.read_text())
print(f"  num_cases={blob.get('num_cases')}  eid_file={blob.get('eid_file')}")
for k, v in blob.get("metrics", {}).items():
    print(f"    {k:32s} {v}")
