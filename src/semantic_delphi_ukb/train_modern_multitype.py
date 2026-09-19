from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from utils import get_p2i
from semantic_delphi_ukb.modern_model import ModernMultitypeSemanticDelphi, ModernMultitypeSemanticDelphiConfig
from semantic_delphi_ukb.multitype_batch import get_batch


out_dir = "ckpt/MedTrajectory_exp2_modern_baseline"
eval_interval = 250
log_interval = 25
eval_iters = 25
eval_only = False
always_save_checkpoint = False
init_from = "scratch"
seed = 42

wandb_log = False
wandb_project = "delphi"
wandb_run_name = "medtrajectory_exp2_modern_" + str(time.time())

dataset = "ukb_semantic_multitype_explicit_split"
gradient_accumulation_steps = 1
batch_size = 96
block_size = 128

n_layer = 6
n_head = 8
n_embd = 64
dropout = 0.1
bias = False
vocab_size = 8424

learning_rate = 6e-4
max_iters = 100000
weight_decay = 2e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 1000
lr_decay_iters = 100000
min_lr = 6e-5

device = "cpu"
dtype = "float32"
compile = False

token_dropout = 0.0
t_min = 0.1
mask_ties = True
ignore_tokens = [0]
data_fraction = 1.0
no_event_token_rate = 5

token_embedding_path = "auto"
freeze_input_embeddings = False
tie_input_output_embeddings = False

use_age_rope = True
age_rope_scale = 1.0
swiglu_hidden_mult = 8.0 / 3.0

static_dim = 10
static_hidden_dim = 64
static_dropout = 0.05
fusion_mode = "residual"
fuse_static_to_logits = False

model_family = "medtrajectory_modern_baseline"


config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str, list))]
with open("configurator.py", encoding="utf-8") as f:
    exec(f.read())
config = {k: globals()[k] for k in config_keys}


def resolve_embedding_path(dataset_name: str, requested_path: str) -> str:
    if requested_path == "auto":
        return os.path.join("data", dataset_name, "semantic_input_embeddings_64d.npy")
    return requested_path


def main() -> int:
    global vocab_size, n_embd

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    torch.set_default_dtype(ptdtype)

    data_dir = Path("data") / dataset
    semantic_embedding_matrix = np.load(resolve_embedding_path(dataset, token_embedding_path)).astype(np.float32)
    vocab_size = int(semantic_embedding_matrix.shape[0])
    n_embd = int(semantic_embedding_matrix.shape[1])

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

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout,
        token_dropout=token_dropout,
        t_min=t_min,
        mask_ties=mask_ties,
        ignore_tokens=ignore_tokens,
        freeze_input_embeddings=freeze_input_embeddings,
        tie_input_output_embeddings=tie_input_output_embeddings,
        use_age_rope=use_age_rope,
        age_rope_scale=age_rope_scale,
        swiglu_hidden_mult=swiglu_hidden_mult,
        static_dim=static_dim,
        static_hidden_dim=static_hidden_dim,
        static_dropout=static_dropout,
        fusion_mode=fusion_mode,
        fuse_static_to_logits=fuse_static_to_logits,
    )

    iter_num = 0
    best_val_loss = 1e9
    checkpoint = None
    if init_from == "scratch":
        model = ModernMultitypeSemanticDelphi(ModernMultitypeSemanticDelphiConfig(**model_args), semantic_embedding_matrix)
    elif init_from == "resume":
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint["model_args"]
        for key in model_args:
            if key in checkpoint_model_args:
                model_args[key] = checkpoint_model_args[key]
        model = ModernMultitypeSemanticDelphi(ModernMultitypeSemanticDelphiConfig(**model_args))
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for key in list(state_dict.keys()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    else:
        raise ValueError(f"Unsupported init_from={init_from}")

    model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == "resume" and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if compile:
        model = torch.compile(model)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters, 2)
            data = train_data if split == "train" else val_data
            p2i = train_p2i if split == "train" else val_p2i
            static_mat = train_static if split == "train" else val_static
            for step in range(eval_iters):
                ix = torch.randint(len(p2i), (batch_size,))
                x, a, y, b, s = get_batch(
                    ix,
                    data,
                    p2i,
                    static_mat,
                    block_size=block_size,
                    device=device,
                    select="left",
                    no_event_token_rate=no_event_token_rate,
                    cut_batch=True,
                )
                with ctx:
                    _, loss, _ = model(x, a, s, y, b, validation_loss_mode=True)
                losses[step] = torch.stack([loss["loss_ce"], loss["loss_dt"]])
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

    if wandb_log:
        import wandb

        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    ix = torch.randint(len(train_p2i), (batch_size,))
    x, a, y, b, s = get_batch(
        ix,
        train_data,
        train_p2i,
        train_static,
        block_size=block_size,
        device=device,
        padding="random",
        select="left",
        no_event_token_rate=no_event_token_rate,
    )
    t0 = time.time()
    while True:
        metrics = {"iter": iter_num}
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss()
            val_loss = losses["val"].sum().item()
            print(f"step {iter_num}: train loss {losses['train'].sum().item():.4f}, val loss {val_loss:.4f}")
            metrics.update({"val/loss": val_loss, "val/loss_ce": losses["val"][0].item(), "val/loss_dt": losses["val"][1].item()})
            if always_save_checkpoint or best_val_loss > val_loss:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": val_loss,
                        "config": config,
                    },
                    os.path.join(out_dir, "ckpt.pt"),
                )
            if best_val_loss > val_loss:
                best_val_loss = val_loss

        if iter_num == 0 and eval_only:
            break

        for _ in range(gradient_accumulation_steps):
            with ctx:
                _, loss, _ = model(x, a, s, y, b)
            ix = torch.randint(len(train_p2i), (batch_size,))
            x, a, y, b, s = get_batch(
                ix,
                train_data,
                train_p2i,
                train_static,
                block_size=block_size,
                device=device,
                padding="random",
                select="left",
                no_event_token_rate=no_event_token_rate,
                cut_batch=True,
            )
            total_loss = loss["loss_ce"] + loss["loss_dt"]
            scaler.scale(total_loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {total_loss.item():.4f}, time {(t1 - t0) * 1000:.2f}ms")
            metrics.update({"train/loss": total_loss.item(), "lr": lr})
        t0 = t1
        if wandb_log and (iter_num % log_interval == 0 or "val/loss" in metrics):
            wandb.log(metrics)

        iter_num += 1
        if iter_num > max_iters:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
