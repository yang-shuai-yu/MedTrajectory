"""Stop the running eval chain by explicit PID (never pkill -f, which self-matches
the ssh command line and kills the session)."""
import os
import signal
import subprocess
import time

TARGETS = ("mimic_run_bc_seeds_fast.sh", "mimic_run_eval_bc_seeds.sh",
           "mimic_generation_eval_trackr.py")

def snapshot():
    out = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid, ppid, cmd = int(parts[0]), int(parts[1]), parts[2]
        if pid == os.getpid():
            continue
        if any(t in cmd for t in TARGETS):
            found.append((pid, ppid, cmd))
    return found

procs = snapshot()
print("=== matching processes ===")
for pid, ppid, cmd in procs:
    print(f"  pid={pid:8d} ppid={ppid:8d}  {cmd[:120]}")

if not procs:
    print("  (none)")
    raise SystemExit(0)

# Kill shells first so they cannot start new evals, then the eval workers.
shells = [p for p in procs if "eval_bc_seeds.sh" in p[2] or "bc_seeds_fast.sh" in p[2]]
workers = [p for p in procs if p not in shells]

for pid, _ppid, cmd in shells:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  TERM shell pid={pid}")
    except ProcessLookupError:
        print(f"  shell pid={pid} already gone")
time.sleep(2)
for pid, _ppid, cmd in workers:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  TERM worker pid={pid}")
    except ProcessLookupError:
        print(f"  worker pid={pid} already gone")

time.sleep(6)
print("\n=== survivors ===")
left = snapshot()
for pid, ppid, cmd in left:
    print(f"  pid={pid:8d}  {cmd[:120]}")
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"    -> KILL")
    except ProcessLookupError:
        pass
if not left:
    print("  (none)")

print("\n=== GPU ===")
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
