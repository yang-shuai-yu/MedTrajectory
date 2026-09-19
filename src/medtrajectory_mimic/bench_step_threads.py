"""Measure CPU-seconds and wall-seconds per training step vs torch thread count.

Runs one real MIMIC Track-R pretraining step (identical code path to
train_car_rope_pretraining.py) and decomposes the cost into
  (a) batch assembly (build_training_batch)
  (b) forward + backward + optimizer step
CPU time uses time.process_time(), which sums CPU time over ALL threads, so it is
independent of external CPU contention and reveals thread-pool spin overhead.
"""
from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(str(REPO_ROOT))
for _p in (REPO, REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA = Path(str(MIMIC_ROOT))

from semantic_delphi_ukb.train_car_rope_pretraining import build_parser  # noqa: E402
from semantic_delphi_ukb.train_car_rope import (  # noqa: E402
    build_training_batch,
    load_rope_wavelengths,
    load_split,
    make_model,
    parse_horizons,
    parse_positive_floats,
    resolve_diseases_yaml,
)
from semantic_delphi_ukb.track_r_batch import (  # noqa: E402
    load_track_r_assets,
    load_track_r_bos_token_id,
    load_track_r_static_features,
    validate_track_r_data_manifest,
)
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402
from semantic_delphi_ukb.tte_targets import load_selected_disease_token_groups  # noqa: E402

DDIR = DATA / os.environ.get("BENCH_DATA", "visit_B_m3_trackr")
PROTO = DATA / os.environ.get("BENCH_PROTO", "visit_B_m3_protocol.json")
THREADS = [int(t) for t in os.environ.get("BENCH_THREADS", "1,2,4,8,16,32").split(",")]
STEPS = int(os.environ.get("BENCH_STEPS", "20"))
WARM = int(os.environ.get("BENCH_WARM", "8"))


def setup_args():
    argv = [
        "--data-dir", str(DDIR), "--track-r-protocol", str(PROTO),
        "--include-static-prefix", "true", "--age-rope-variant", "legacy",
        "--use-age-encoding", "true", "--use-age-rope", "false",
        "--diseases-yaml", str(DATA / "multitype/mimic_diseases.yaml"),
        "--run-dir", "/tmp/bench_run", "--device", "cuda",
        "--seed", "44", "--no-tensorboard",
    ]
    args = build_parser().parse_args(argv)
    args.horizons = parse_horizons(args.horizons)
    args.rope_scales = parse_positive_floats(args.rope_scales, "--rope-scales")
    args.rope_wavelengths_days = ()
    args.dynamic_context_length = args.block_size
    protocol = load_track_r_protocol(args.track_r_protocol)
    args.track_r_protocol_sha256 = protocol.get("protocol_manifest_sha256")
    validate_track_r_data_manifest(args.data_dir, protocol)
    args.dynamic_context_length = int(protocol["dynamic_context_length"])
    args.track_r_bos_token_id = load_track_r_bos_token_id(args.data_dir)
    args.block_size = args.dynamic_context_length + 1 + int(protocol["static_prefix"]["fixed_length"])
    args.rope_wavelengths_days = load_rope_wavelengths(
        args.rope_wavelengths_manifest,
        protocol.get("wavelength_contract"),
        require_expected=False,
    )
    return args


def main() -> int:
    print(f"[env] sched_getaffinity={len(os.sched_getaffinity(0))} "
          f"torch.get_num_threads()={torch.get_num_threads()} "
          f"interop={torch.get_num_interop_threads()} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    args = setup_args()
    t0 = time.perf_counter()
    diseases_yaml = resolve_diseases_yaml(args.diseases_yaml)
    diseases, _groups = load_selected_disease_token_groups(diseases_yaml, args.data_dir)
    train_data, train_p2i, train_static = load_split(args.data_dir, "train", args.max_patients)
    train_prefix, train_anchor = load_track_r_assets(args.data_dir, "train", args.max_patients)
    train_static = load_track_r_static_features(
        args.data_dir, "train", args.static_conditioning, args.max_patients
    )
    load_s = time.perf_counter() - t0
    print(f"[load] {load_s:.1f}s data={type(train_data).__name__} shape={getattr(train_data, 'shape', None)} "
          f"dtype={getattr(train_data, 'dtype', None)} n_patients={len(train_p2i)} "
          f"prefix={train_prefix.shape} block_size={args.block_size}", flush=True)
    model = make_model(args, args.data_dir, len(diseases)).to(args.device)
    opt = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.99), "cuda")
    model.train()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_par/1e3:.1f}k device={args.device}", flush=True)

    def one_step(mode: str):
        ix = torch.randint(len(train_p2i), (args.batch_size,))
        batch = build_training_batch(
            ix, train_data, train_p2i, train_static, args,
            padding="regular", select="random",
            prefix_ids=train_prefix, anchor_ages=train_anchor,
        )
        if mode == "build":
            return
        x, age, y, target_age, s, static_mask, bos_mask, next_mask, time_mask, sfm = batch
        _, parts, *_ = model(
            x, age, s, y, target_age,
            static_token_mask=static_mask, next_event_mask=next_mask,
            time_loss_mask=time_mask, static_feature_mask=sfm, bos_token_mask=bos_mask,
        )
        gap = parts.get("loss_gap", torch.zeros((), device=age.device))
        loss = parts["loss_ce"] + parts["loss_dt"] + model.config.time_gap_loss_weight * gap
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    def measure(threads: int, mode: str):
        torch.set_num_threads(threads)
        for _ in range(WARM):
            one_step(mode)
        torch.cuda.synchronize()
        c0, w0 = time.process_time(), time.perf_counter()
        for _ in range(STEPS):
            one_step(mode)
        torch.cuda.synchronize()
        dc, dw = time.process_time() - c0, time.perf_counter() - w0
        return dc / STEPS, dw / STEPS

    rows = []
    for mode in ("build", "full"):
        for t in THREADS:
            cpu, wall = measure(t, mode)
            rows.append({"mode": mode, "threads": t, "cpu_s_per_step": cpu, "wall_s_per_step": wall})
            print(f"[{mode}] threads={t:3d} cpu={cpu*1000:8.2f} ms/step  wall={wall*1000:8.2f} ms/step  "
                  f"cpu/wall={cpu/wall:5.2f}x", flush=True)
    print("[summary]", flush=True)
    for r in rows:
        print(f"  {r['mode']:5s} threads={r['threads']:3d} cpu={r['cpu_s_per_step']*1000:8.2f}ms "
              f"wall={r['wall_s_per_step']*1000:8.2f}ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
