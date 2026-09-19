"""Health check before the long relaxed-cohort evaluation completes."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
import os
import subprocess
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


print("=== disk ===")
print(sh("df -h . | tail -2"))
print("=== GPU ===")
print(sh("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total "
         "--format=csv,noheader"))
print("=== cpu quota / load ===")
print("  " + sh("cat /sys/fs/cgroup/cpu.max").strip())
print("  " + sh("uptime").strip())
print("=== eval processes ===")
out = sh("ps -eo pid,etimes,pcpu,args")
n = 0
for line in out.splitlines():
    if "generation_eval_trackr.py" not in line:
        continue
    parts = line.split(None, 3)
    if len(parts) < 4:
        continue
    n += 1
print(f"  {n} running")

print("\n=== stderr/traceback scan in every in-flight eval log ===")
bad = []
for p in DATA.glob("eval_*/*.stdout.log"):
    try:
        txt = p.read_text(errors="replace")
    except Exception:
        continue
    low = txt.lower()
    if "traceback" in low or "error" in low or "cuda out of memory" in low:
        bad.append(p)
for p in bad:
    age = int(os.path.getmtime(DATA / "bc_eval_rerun3.log") - p.stat().st_mtime) if False else None
    print(f"  {p.parent.name}/{p.name}")
    for line in p.read_text(errors="replace").splitlines()[-6:]:
        print("      " + line[:150])
if not bad:
    print("  none")

print("\n=== driver log tail ===")
log = DATA / "bc_eval_rerun3.log"
if log.exists():
    for line in log.read_text(errors="replace").splitlines()[-6:]:
        print("  " + line[:150])

print("\n=== free space needed vs used by one 2000-patient eval ===")
d = DATA / "eval_bm3_relaxed" / "bm_m1_a0"
if d.exists():
    print("  partial output dir contents:", [x.name for x in d.iterdir()])
else:
    print("  (no output written yet -- rows are written at the end)")
