"""Measure actual evaluation throughput before and after the concurrency change."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
import subprocess
import time
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))
TAGS = ["eval_bm3_ukb", "eval_bm3_relaxed", "eval_B3_ukb", "eval_B3_relaxed",
        "eval_C3_ukb", "eval_C3_relaxed"]

TOTAL_REMAINING_PATIENTS = {
    "eval_bm3_ukb": 0,
    "eval_bm3_relaxed": 9 * 2000,
    "eval_B3_ukb": 4 * 949,
    "eval_B3_relaxed": 6 * 2000,
    "eval_C3_ukb": 2 * 604,
    "eval_C3_relaxed": 6 * 2000,
}


def snapshot():
    done = {}
    for tag in TAGS:
        d = DATA / tag
        done[tag] = len(list(d.glob("*/summary.json"))) if d.exists() else 0
    return done


def cpu_cores():
    out = subprocess.run(
        ["ps", "-eo", "pcpu,args"], capture_output=True, text=True).stdout
    total = 0.0
    n = 0
    for line in out.splitlines():
        if "generation_eval" in line and "grep" not in line:
            try:
                total += float(line.split()[0])
                n += 1
            except ValueError:
                pass
    return n, total / 100.0


d0 = snapshot()
n0 = sum(d0.values())
t0 = time.time()
cores_n, cores = cpu_cores()
print(f"t0: {n0} summaries done, {cores_n} evals running using {cores:.1f} cores")

WINDOW = 600
time.sleep(WINDOW)
d1 = snapshot()
n1 = sum(d1.values())
dt = time.time() - t0
print(f"after {dt:.0f}s: {n1} summaries done (+{n1 - n0})")

newly = []
for tag in TAGS:
    if d1[tag] > d0[tag]:
        newly.append(f"{tag}+{d1[tag] - d0[tag]}")
print("new completions:", newly or "(none finished in this window)")

# throughput in patient-evals/s, using per-tag cohort sizes of completed jobs
SIZES = {
    ("eval_bm3_ukb", "bm"): 541, ("eval_bm3_relaxed", "bm"): 2000,
    ("eval_B3_ukb", "B"): 949, ("eval_B3_relaxed", "B"): 2000,
    ("eval_C3_ukb", "C"): 604, ("eval_C3_relaxed", "C"): 2000,
}
work = 0
for tag in TAGS:
    delta = d1[tag] - d0[tag]
    if not delta:
        continue
    key = (tag, "bm" if tag.startswith("eval_bm3") else ("B" if "_B3_" in tag else "C"))
    work += delta * SIZES[key]

if work:
    rate = work / dt
    print(f"work completed in window: {work} patient-evals")
    print(f"THROUGHPUT: {rate:.2f} patient-evals/s  ({rate * 60:.0f}/min)")
    remaining = sum(TOTAL_REMAINING_PATIENTS.values()) - work
    print(f"remaining ~{remaining} patient-evals -> {remaining / rate / 3600:.2f} h")
else:
    print("no job finished in the window; cannot compute a rate yet")
    print("(each relaxed job is 2000 patients; at the old 1.87 s/patient "
          "a single one takes ~62 min, so a short window may see nothing)")
