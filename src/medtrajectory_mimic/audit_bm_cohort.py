"""Audit the cohort of every Bm multitype evaluation: is the patient set actually matched?"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))

for tag in ("eval_bm_ukb", "eval_bm_relaxed", "eval_bm3_ukb", "eval_bm3_relaxed"):
    d = DATA / tag
    if not d.exists():
        continue
    print(f"=== {tag} ===")
    for p in sorted(d.glob("*/summary.json")):
        b = json.loads(p.read_text())
        print(f"   {p.parent.name:16s} num_cases={str(b.get('num_cases')):>6s}  "
              f"eid_file={b.get('eid_file')}  "
              f"jaccard={b.get('metrics', {}).get('diagnosis_jaccard'):.6f}")

print("\n=== matched cohort files ===")
for name in ("matched_bm_ukb.txt", "matched_bm_relaxed.txt"):
    p = DATA / name
    if p.exists():
        ids = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        print(f"   {name:24s} {len(ids)} ids  head={ids[:3]}")

print("\n=== the two old matched runs (if they exist) ===")
for cand in ("eval_bm_matched_ukb", "eval_bm_matched_relaxed", "eval_bm_pair_ukb"):
    d = DATA / cand
    print(f"   {cand}: exists={d.exists()}")

# what does the eval driver actually pass for --eid-file?
drv = DATA / "mimic_run_eval_bc_seeds.sh"
print("\n=== mode invocations in the eval driver ===")
for line in drv.read_text().splitlines():
    if line.strip().startswith("mode ") or "--eid-file" in line or "${eid:" in line:
        print("   " + line.strip())
