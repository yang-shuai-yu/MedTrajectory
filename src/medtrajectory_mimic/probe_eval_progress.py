"""Are the relaxed evaluations progressing or stuck?  Compare wall vs CPU time."""
import subprocess
import time

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

print("=== eval processes: wall vs CPU ===")
out = sh("ps -eo pid,etimes,times,pcpu,args")
rows = []
for line in out.splitlines():
    if "generation_eval_trackr.py" not in line:
        continue
    if "grep" in line.split()[-1]:
        continue
    parts = line.split(None, 4)
    if len(parts) < 5:
        continue
    pid, etimes, cputime, pcpu, cmd = parts
    tag = ""
    toks = cmd.split()
    for i, t in enumerate(toks):
        if t == "--out-dir":
            tag = "/".join(toks[i + 1].split("/")[-2:])
    rows.append((pid, int(etimes), cputime, float(pcpu), tag))
    print("  pid={} wall={}s cpu={} pcpu={}  {}".format(pid, etimes, cputime, pcpu, tag))
print(f"  count = {len(rows)}")

print("\n=== progress check: CPU-time delta over 60 s ===")
snap1 = {}
for line in sh("ps -eo pid,times").splitlines()[1:]:
    p = line.split()
    if len(p) == 2:
        snap1[p[0]] = p[1]
time.sleep(60)
snap2 = {}
for line in sh("ps -eo pid,times").splitlines()[1:]:
    p = line.split()
    if len(p) == 2:
        snap2[p[0]] = p[1]

def to_sec(v):
    if ":" in v:
        parts = [float(x) for x in v.split(":")]
        s = 0.0
        for x in parts:
            s = s * 60 + x
        return s
    return float(v)

active = 0
for pid, _w, _c, _p, tag in rows:
    a, b = snap1.get(pid), snap2.get(pid)
    if a is None or b is None:
        print(f"  {tag}: process disappeared")
        continue
    d = to_sec(b) - to_sec(a)
    active += d > 30
    print(f"  {tag}: consumed {d:.1f} CPU-s in 60 s wall -> {'WORKING' if d > 30 else 'STALLED'}")
print(f"  {active}/{len(rows)} actively computing")
