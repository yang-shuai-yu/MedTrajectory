"""Characterise the eval limit: per-process rate vs concurrency."""
import subprocess
import time

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

out = sh("ps -eo pcpu,args | grep generation_eval | grep -v grep")
n = 0
cores = 0.0
for line in out.splitlines():
    try:
        cores += float(line.split()[0]) / 100.0
        n += 1
    except ValueError:
        pass
print(f"running evals: {n}   CPU cores used: {cores:.1f} / 20")

print("\nGPU samples (12 s):")
for _ in range(6):
    print("  " + sh("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,"
                    "memory.used --format=csv,noheader").strip())
    time.sleep(2)

print("\nGPU processes:")
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"))

print("\ncompleted eval dirs:")
print(sh("cd MIMIC_ROOT && "
         "for t in eval_bm3_ukb eval_bm3_relaxed eval_B3_ukb eval_B3_relaxed "
         "eval_C3_ukb eval_C3_relaxed; do "
         "printf '%-20s %s\\n' $t $(find $t -name summary.json 2>/dev/null | wc -l); done"))
