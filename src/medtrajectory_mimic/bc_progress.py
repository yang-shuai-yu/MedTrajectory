"""Report training progress and per-step timing for the BC seed runs."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
import subprocess
import sys
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))
RUNS = sys.argv[1:] or [
    "vB_m3_a0_s44", "vB_m3_a2_s44",
    "vC_a0_s43", "vC_a2_s43",
    "vC_a0_s44", "vC_a2_s44",
]

# process start times, to derive throughput from iteration/elapsed
out = subprocess.run(
    ["ps", "-eo", "etimes,args"], capture_output=True, text=True
).stdout
elapsed = {}
for line in out.splitlines():
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        continue
    secs, cmd = parts
    for name in RUNS:
        if f"runs/{name} " in cmd + " ":
            elapsed[name] = int(secs)

tot = 0
for name in RUNS:
    p = DATA / "runs" / name / "status.json"
    if not p.exists():
        print(f"{name:16s} no status.json")
        continue
    d = json.loads(p.read_text())
    it = int(d["iteration"])
    sm = float(d.get("performance_step_time_seconds", -1)) * 1000
    el = elapsed.get(name)
    thr = f"{it / el:6.2f} steps/s" if el else "   n/a"
    eta = f"{(100000 - it) * (el / it) / 3600:5.2f} h" if el and it else "  n/a"
    tot += it
    print(f"{name:16s} iter={it:6d}  step_ms={sm:7.2f}  elapsed={el or 0:5d}s  "
          f"{thr}  ETA={eta}  gpu={d.get('gpu_memory_reserved_gb', 0):.2f}GB")
print(f"{'TOTAL':16s} {tot} steps completed of 600000")
