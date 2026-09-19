"""Status of the B/C three-seed evaluation chain."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, WORKSPACE_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, WORKSPACE_ROOT
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))
TAGS = ["eval_bm3_ukb", "eval_bm3_relaxed", "eval_B3_ukb", "eval_B3_relaxed",
        "eval_C3_ukb", "eval_C3_relaxed"]
EXPECTED = {"eval_bm3": 9, "eval_B3": 6, "eval_C3": 6}

print(f"now {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")

done_total = 0
expected_total = 0
for tag in TAGS:
    d = DATA / tag
    n = len(list(d.glob("*/summary.json"))) if d.exists() else 0
    base = tag.rsplit("_", 1)[0]
    exp = EXPECTED.get(base, 0)
    done_total += n
    expected_total += exp
    print(f"  {tag:20s} {n:2d}/{exp} summaries")
print(f"  TOTAL {done_total}/{expected_total}")

print("\nrecent eval stdout (last lines of the 3 newest):")
logs = sorted(DATA.glob("eval_*/*.stdout.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
for p in logs:
    try:
        text = p.read_text(errors="replace").strip().splitlines()
    except Exception as exc:
        print(f"  {p.name}: unreadable ({exc})")
        continue
    age = int((datetime.now().timestamp() - p.stat().st_mtime))
    print(f"  -- {p.parent.name}/{p.name}  last_write={age}s ago")
    for line in text[-3:]:
        print(f"       {line[:160]}")

proc = subprocess.run(["ps", "-eo", "etimes,args"], capture_output=True, text=True).stdout
evals = [l for l in proc.splitlines() if "generation_eval" in l and "grep" not in l]
print(f"\nrunning eval processes: {len(evals)}")
for line in evals[:8]:
    parts = line.strip().split(None, 1)
    if len(parts) == 2:
        secs, cmd = parts
        out = ""
        for token in cmd.split():
            if token.startswith("eval_"):
                out = token
        print(f"  {int(secs):5d}s  {out}")

res = DATA / "results_bc_3seeds.md"
print(f"\nresults_bc_3seeds.md exists: {res.exists()}"
      + (f"  ({res.stat().st_size} bytes)" if res.exists() else ""))

sens = Path(str(WORKSPACE_ROOT / "risk_ni_sensitivity.log"))
if sens.exists():
    print("\n=== risk non-inferiority sensitivity ===")
    for line in sens.read_text().splitlines()[-10:]:
        print("  " + line)
