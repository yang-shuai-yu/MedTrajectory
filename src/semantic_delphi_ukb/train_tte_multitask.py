from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from semantic_delphi_ukb.multitype_batch import get_batch
from semantic_delphi_ukb.tte_model import TTEConfig, TTEMultitaskDelphi
from semantic_delphi_ukb.tte_targets import (
    build_patient_disease_ages,
    build_tte_batch,
    disease_specs_payload,
    load_selected_disease_token_groups,
)
from utils import get_p2i


out_dir = "ckpt/MedTrajectory_exp2_tte_multitask"
init_from_ckpt = "ckpt/MedTrajectory_exp2_modern_baseline/ckpt.pt"
eval_interval = 250
log_interval = 25
eval_iters = 25
eval_only = False
always_save_checkpoint = False
seed = 42

dataset = "ukb_semantic_multitype_explicit_split"
gradient_accumulation_steps = 1
batch_size = 96
block_size = 128
data_fraction = 1.0

learning_rate = 1e-4
max_iters = 20000
weight_decay = 2e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
decay_lr = True
warmup_iters = 500
lr_decay_iters = 20000
min_lr = 1e-5

device = "cpu"
dtype = "float32"
compile = False

tte_loss_weight = 0.05
tte_horizon_years = 10.0
diseases_yaml = "selected_diseases.yaml"
no_event_token_rate = 5
model_family = "medtrajectory_tte_multitask"


config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str, list))]
with open("configurator.py", encoding="utf-8") as f:
    exec(f.read())
config = {k: globals()[k] for k in config_keys}


def main() -> int:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    torch.set_default_dtype(ptdtype)

    checkpoint = torch.load(init_from_ckpt, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"].copy()
    diseases, token_groups = load_selected_disease_token_groups(Path(diseases_yaml))
    model_args.update(
        num_tte_tasks=len(diseases),
        tte_loss_weight=tte_loss_weight,
        tte_horizon_years=tte_horizon_years,
    )
    model = TTEMultitaskDelphi(TTEConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    print(f"initialized from {init_from_ckpt}; missing new keys: {missing}")
    model.to(device)

    data_dir = Path("data") / dataset
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    train_static = np.load(data_dir / "train_static.npy").astype(np.float32)
    val_static = np.load(data_dir / "val_static.npy").astype(np.float32)
    train_p2i = get_p2i(train_data)
    val_p2i = get_p2i(val_data)
    if data_fraction < 1.0:
        train_limit = max(1, int(data_fraction * len(train_p2i)))
        val_limit = max(1, int(data_fraction * len(val_p2i)))
        train_p2i = train_p2i[:train_limit]
        val_p2i = val_p2i[:val_limit]
        train_static = train_static[:train_limit]
        val_static = val_static[:val_limit]

    train_tte_ages, train_last_ages = build_patient_disease_ages(train_data, train_p2i, token_groups, int(model.config.vocab_size))
    val_tte_ages, val_last_ages = build_patient_disease_ages(val_data, val_p2i, token_groups, int(model.config.vocab_size))

    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if compile:
        model = torch.compile(model)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters, 4)
            data = train_data if split == "train" else val_data
            p2i = train_p2i if split == "train" else val_p2i
            static_mat = train_static if split == "train" else val_static
            tte_ages = train_tte_ages if split == "train" else val_tte_ages
            last_ages = train_last_ages if split == "train" else val_last_ages
            for step in range(eval_iters):
                ix = torch.randint(len(p2i), (batch_size,))
                x, a, y, b, s = get_batch(ix, data, p2i, static_mat, block_size=block_size, device=device, select="left", no_event_token_rate=no_event_token_rate, cut_batch=True)
                ev, dur, mask = build_tte_batch(ix, a, x > 1, tte_ages, last_ages, tte_horizon_years, device)
                with ctx:
                    _, loss, _, _ = model(x, a, s, y, b, validation_loss_mode=True, tte_event=ev, tte_duration=dur, tte_mask=mask)
                losses[step] = torch.stack([loss["loss_ce"], loss["loss_dt"], loss["loss_tte"], loss["loss"]]).detach().cpu()
            out[split] = losses.mean(0)
        model.train()
        return out

    def get_lr(it: int) -> float:
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    iter_num = 0
    best_val_loss = 1e9
    ix = torch.randint(len(train_p2i), (batch_size,))
    x, a, y, b, s = get_batch(ix, train_data, train_p2i, train_static, block_size=block_size, device=device, padding="random", select="left", no_event_token_rate=no_event_token_rate)
    t0 = time.time()
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss()
            val_loss = losses["val"][3].item()
            print(f"step {iter_num}: train loss {losses['train'][3].item():.4f}, val loss {val_loss:.4f}, val tte {losses['val'][2].item():.4f}")
            if always_save_checkpoint or best_val_loss > val_loss:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": val_loss,
                        "config": {**config, "diseases": disease_specs_payload(diseases)},
                    },
                    os.path.join(out_dir, "ckpt.pt"),
                )
            if best_val_loss > val_loss:
                best_val_loss = val_loss

        if iter_num == 0 and eval_only:
            break

        for _ in range(gradient_accumulation_steps):
            ev, dur, mask = build_tte_batch(ix, a, x > 1, train_tte_ages, train_last_ages, tte_horizon_years, device)
            with ctx:
                _, loss, _, _ = model(x, a, s, y, b, tte_event=ev, tte_duration=dur, tte_mask=mask)
            ix = torch.randint(len(train_p2i), (batch_size,))
            x, a, y, b, s = get_batch(ix, train_data, train_p2i, train_static, block_size=block_size, device=device, padding="random", select="left", no_event_token_rate=no_event_token_rate, cut_batch=True)
            scaler.scale(loss["loss"]).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {loss['loss'].item():.4f}, ce {loss['loss_ce'].item():.4f}, dt {loss['loss_dt'].item():.4f}, tte {loss['loss_tte'].item():.4f}, time {(t1 - t0) * 1000:.2f}ms")
        t0 = t1
        iter_num += 1
        if iter_num > max_iters:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
