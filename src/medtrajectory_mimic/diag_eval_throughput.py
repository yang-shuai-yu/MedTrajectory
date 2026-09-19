"""Diagnose what actually limits the evaluation throughput."""
import subprocess
import time

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

print("=== eval processes ===")
out = sh("ps -eo pid,etime,pcpu,rss,args | grep generation_eval | grep -v grep")
procs = []
for line in out.splitlines():
    parts = line.split(None, 4)
    if len(parts) < 5:
        continue
    pid, etime, pcpu, rss, cmd = parts
    tag = ""
    toks = cmd.split()
    for i, t in enumerate(toks):
        if t == "--out-dir":
            tag = toks[i + 1].split("/")[-2] + "/" + toks[i + 1].split("/")[-1]
    procs.append((pid, etime, float(pcpu), int(rss) / 1e6, tag))
for pid, etime, pcpu, rss, tag in procs:
    print(f"  pid={pid:7s} elapsed={etime:>8s} cpu={pcpu:6.1f}%  rss={rss:.2f}GB  {tag}")
print(f"  --> {len(procs)} evals, total CPU = {sum(p[2] for p in procs)/100:.1f} cores")

print("\n=== GPU samples over 12 s ===")
for _ in range(6):
    row = sh("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used "
             "--format=csv,noheader").strip()
    print(f"  {row}")
    time.sleep(2)

print("\n=== cgroup CPU headroom ===")
print("  " + sh("head -1 /sys/fs/cgroup/cpu.stat").strip())
print("  " + sh("cat /sys/fs/cgroup/cpu.max").strip() + "  (quota/period -> cores)")
print("  load: " + sh("uptime").strip())
print("  affinity-visible nproc: " + sh("nproc").strip())

print("\n=== anything else running ===")
print(sh("ps -eo pcpu,args --sort=-pcpu | head -6"))
