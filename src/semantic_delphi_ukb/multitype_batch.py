from __future__ import annotations

import numpy as np
import torch


def get_batch(
    ix,
    data,
    p2i,
    static_matrix,
    select="left",
    padding="regular",
    block_size=48,
    device="cpu",
    no_event_token_rate=5,
    cut_batch=False,
    fixed_padding_ages=None,
):
    mask_time = -10000.0

    ix_list = [int(i) for i in ix.tolist()] if torch.is_tensor(ix) else [int(i) for i in ix]
    x = torch.tensor([p2i[i].tolist() for i in ix_list], dtype=torch.long)
    ix_tensor = torch.tensor(ix_list, dtype=torch.long)
    static_features = torch.tensor(static_matrix[np.asarray(ix_list, dtype=np.int64)], dtype=torch.float32)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(ix_tensor.sum().item()))

    if select == "left":
        traj_start_idx = x[:, 0]
    elif select == "right":
        traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
    elif select == "random":
        traj_start_idx = x[:, 0] + (torch.randint(2**31 - 1, (len(ix_list),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
        traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1).tolist()
    batch_idx = np.arange(block_size + 1)[None, :] + np.asarray(traj_start_idx, dtype=np.int64)[:, None]

    mask_src = data[:, 0][batch_idx].astype(np.int64)
    patient_src = data[p2i[np.asarray(ix_list, dtype=np.int64)][:, 0], 0].astype(np.int64)[:, None]
    mask = torch.tensor((mask_src == patient_src).tolist(), dtype=torch.bool)

    tokens = torch.tensor(data[:, 2][batch_idx].astype(np.int64).tolist(), dtype=torch.long)
    ages = torch.tensor(data[:, 1][batch_idx].astype(np.float32).tolist(), dtype=torch.float32)

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    fixed_pad_mask = None
    if fixed_padding_ages is not None:
        pad = torch.as_tensor(fixed_padding_ages, dtype=torch.float32)
        if pad.ndim != 2 or pad.shape[0] != len(ix_list):
            raise ValueError("fixed_padding_ages must have shape [len(ix), n_targets]")
        fixed_pad_mask = pad > mask_time / 2
    elif padding.lower() == "none" or padding is None or no_event_token_rate in (0, None):
        pad = torch.ones(len(ix_list), 0)
    elif padding == "regular":
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix_list), 1) + 1
    elif padding == "random":
        pad = torch.randint(1, 36525, (len(ix_list), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError

    m = ages.max(1, keepdim=True).values
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.long)])
    ages = torch.hstack([ages, pad])
    if fixed_pad_mask is not None:
        tokens[:, -pad.shape[1]:] = tokens[:, -pad.shape[1]:].masked_fill(~fixed_pad_mask, -1)
        ages[:, -pad.shape[1]:] = ages[:, -pad.shape[1]:].masked_fill(~fixed_pad_mask, mask_time)
    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)
    tokens = tokens + 1

    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]

    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)

    if device == "cuda":
        x, a, y, b, static_features = [
            tensor.pin_memory().to(device, non_blocking=True) for tensor in [x, a, y, b, static_features]
        ]
    else:
        x, a, y, b, static_features = x.to(device), a.to(device), y.to(device), b.to(device), static_features.to(device)
    return x, a, y, b, static_features
