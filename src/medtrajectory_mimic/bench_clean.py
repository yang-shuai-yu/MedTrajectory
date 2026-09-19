"""Clean (idle-machine) characterisation of the MIMIC Track-R training step.

Produces three things:
  1. bit-identity certificate: identical batches across torch thread counts
  2. clean per-step wall/CPU time vs thread count (idle GPU, idle CPU)
  3. pure GPU time per step (forward+backward+optimizer, pre-built batches)
so that the optimal number of concurrent training processes can be derived.
"""
from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(str(REPO_ROOT))
for _p in (REPO, REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(MIMIC_ROOT))
from bench_step_threads import DATA, THREADS, setup_args  # noqa: E402

from semantic_delphi_ukb.train_car_rope import (  # noqa: E402
    build_training_batch,
    load_split,
    make_model,
    resolve_diseases_yaml,
)
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    load_track_r_assets,
    load_track_r_static_features,
)
from semantic_delphi_ukb.tte_targets import load_selected_disease_token_groups  # noqa: E402

STEPS = int(os.environ.get("BENCH_STEPS", "30"))
WARM = int(os.environ.get("BENCH_WARM", "10"))


def batch_hash(batch):
    h = hashlib.sha256()
    for t in batch:
        if t is None:
            h.update(b"<none>")
        else:
            h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:24]


def main() -> int:
    args = setup_args()
    print(f"[env] affinity={len(os.sched_getaffinity(0))} default_threads={torch.get_num_threads()}", flush=True)
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    diseases, _ = load_selected_disease_token_groups(diseases_yaml, args.data_dir)
    data, p2i, static = load_split(args.data_dir, "train", args.max_patients)
    prefix, anchor = load_track_r_assets(args.data_dir, "train", args.max_patients)
    static = load_track_r_static_features(args.data_dir, "train", args.static_conditioning, args.max_patients)
    model = make_model(args, args.data_dir, len(diseases)).to(args.device)
    opt = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.99), "cuda")
    model.train()

    def build():
        torch.manual_seed(1234)
        ix = torch.randint(len(p2i), (args.batch_size,))
        return build_training_batch(
            ix, data, p2i, static, args, padding="regular", select="random",
            prefix_ids=prefix, anchor_ages=anchor,
        )

    def fwd_bwd(batch):
        x, age, y, target_age, s, sm, bm, nm, tm, sfm = batch
        _, parts, *_ = model(
            x, age, s, y, target_age, static_token_mask=sm, next_event_mask=nm,
            time_loss_mask=tm, static_feature_mask=sfm, bos_token_mask=bm,
        )
        gap = parts.get("loss_gap", torch.zeros((), device=age.device))
        loss = parts["loss_ce"] + parts["loss_dt"] + model.config.time_gap_loss_weight * gap
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # ---- 1. bit-identity certificate ----
    print("\n[1] bit-identity across thread counts (batch assembly)", flush=True)
    hashes = {}
    for t in (1, 2, 4, 8, 16, 32, 104):
        torch.set_num_threads(t)
        digests = [batch_hash(build()) for _ in range(4)]
        hashes[t] = digests
        print(f"    threads={t:4d}  {digests}", flush=True)
    ref = hashes[1]
    identical = all(v == ref for v in hashes.values())
    print(f"    IDENTICAL_ACROSS_THREAD_COUNTS = {identical}", flush=True)

    # ---- 2. clean per-step timing ----
    print("\n[2] clean single-process timing (idle machine)", flush=True)
    rows = []
    for mode in ("build", "full"):
        for t in THREADS:
            torch.set_num_threads(t)
            for _ in range(WARM):
                b = build()
                if mode == "full":
                    fwd_bwd(b)
            torch.cuda.synchronize()
            c0, w0 = time.process_time(), time.perf_counter()
            for _ in range(STEPS):
                b = build()
                if mode == "full":
                    fwd_bwd(b)
            torch.cuda.synchronize()
            cpu, wall = (time.process_time() - c0) / STEPS, (time.perf_counter() - w0) / STEPS
            rows.append({"mode": mode, "threads": t, "cpu_ms": cpu * 1e3, "wall_ms": wall * 1e3})
            print(f"    [{mode:5s}] threads={t:4d} cpu={cpu*1e3:8.2f} ms  wall={wall*1e3:8.2f} ms", flush=True)

    # ---- 3. pure GPU time ----
    print("\n[3] pure GPU time per step (batches pre-built)", flush=True)
    torch.set_num_threads(2)
    pre = [build() for _ in range(64)]
    for b in pre[:8]:
        fwd_bwd(b)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for b in pre:
        fwd_bwd(b)
    ev1.record()
    torch.cuda.synchronize()
    gpu_ms = ev0.elapsed_time(ev1) / len(pre)
    print(f"    GPU-only fwd+bwd+opt = {gpu_ms:.2f} ms/step", flush=True)

    full1 = next(r for r in rows if r["mode"] == "full" and r["threads"] == 1)
    out = {
        "bit_identical_across_threads": identical,
        "rows": rows,
        "gpu_only_ms_per_step": gpu_ms,
        "wall_ms_threads1": full1["wall_ms"],
    }
    print("\n[4] derived", flush=True)
    print(f"    per-model wall @threads=1 : {full1['wall_ms']:.2f} ms/step")
    print(f"    GPU floor                 : {gpu_ms:.2f} ms/step")
    print(f"    max useful concurrency    : {full1['wall_ms']/gpu_ms:.2f} processes")
    print(f"    GPU-capped aggregate rate : {1000.0/gpu_ms:.1f} steps/s")
    print(f"    current aggregate rate    : {2/(91000/1000.0):.1f} steps/s (2 procs @~91 ms)", flush=True)
    Path(str(MIMIC_ROOT / "bench_clean.json")).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
