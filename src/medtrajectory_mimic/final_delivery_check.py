"""Final delivery check: is every element of the objective actually in place?"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import re
import subprocess
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))
DOCS = Path(r"E:\class\Doctor\term4\Comp1\MedTrajectory_AI_Competition\docs")

print("=== evaluation completeness ===")
tags = ["eval_bm3_ukb", "eval_bm3_relaxed", "eval_B3_ukb", "eval_B3_relaxed",
        "eval_C3_ukb", "eval_C3_relaxed"]
total = 0
for t in tags:
    n = len(list((DATA / t).glob("*/summary.json"))) if (DATA / t).exists() else 0
    total += n
    print(f"  {t:20s} {n}")
print(f"  TOTAL {total}/42")

print("\n=== deliverables on the server ===")
for name in ("FINAL_CONCLUSIONS.md", "paired_3seed.md", "paired_ethos.md",
             "results_bc_3seeds.md"):
    p = DATA / name
    print(f"  {name:26s} {'OK' if p.exists() else 'MISSING':8s}"
          + (f" {p.stat().st_size} bytes, {len(p.read_text(errors='replace').splitlines())} lines"
             if p.exists() else ""))

print("\n=== results_bc_3seeds.md: disclosure present? ===")
r = DATA / "results_bc_3seeds.md"
if r.exists():
    txt = r.read_text(errors="replace")
    for probe in ("DISCLOSED COHORT SHORTFALL", "Part A — final conclusions",
                  "Part B — descriptive appendix", "INCOMPLETE"):
        print(f"  {probe:40s} {'present' if probe in txt else 'ABSENT'}")

print("\n=== documents: any placeholder left? ===")
for name in ("MIMIC_external_validation_draft.md", "MIMIC外部验证_技术报告_v1.md"):
    p = DOCS / name
    t = p.read_text(encoding="utf-8")
    pending = [l.strip()[:90] for l in t.splitlines() if "[PENDING" in l or "TO BE FILLED" in l]
    print(f"  {name}")
    print(f"     lines={len(t.splitlines())}  placeholders={len(pending)}")
    for l in pending:
        print(f"       {l}")

print("\n=== gate status ===")
out = subprocess.run(["<python interpreter, e.g. the project virtualenv>",
                      str(DATA / "verify_eval_cohorts.py")],
                     capture_output=True, text=True).stdout
print("  " + out.strip().splitlines()[-1])
fails = [l for l in out.splitlines() if "FAIL" in l]
oks = [l for l in out.splitlines() if l.strip().startswith("OK")]
print(f"  groups OK={len(oks)}  FAIL={len(fails)}")
for l in fails:
    print("   " + l.strip()[:130])
