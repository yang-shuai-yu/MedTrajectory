from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def load_protocol_config(common_path: Path, experiment_path: Path) -> dict[str, Any]:
    common = json.loads(common_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    return {"common": common, "experiment": experiment}


def resolve_path(repo_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_dir / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def restore_random_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def strip_compiled_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = dict(state_dict)
    for key in list(output):
        if key.startswith("_orig_mod."):
            output[key[len("_orig_mod.") :]] = output.pop(key)
    return output


def write_source_manifest(
    run_dir: Path,
    repo_dir: Path,
    entrypoint: Path,
    extra_paths: Sequence[Path] = (),
) -> None:
    tracked = [
        entrypoint,
        repo_dir / "src" / "semantic_delphi_ukb" / "modern_model.py",
        repo_dir / "src" / "semantic_delphi_ukb" / "tte_model.py",
        repo_dir / "src" / "semantic_delphi_ukb" / "paper_run.py",
        repo_dir / "configs" / "paper_protocol_v1" / "common.json",
        *extra_paths,
    ]
    files = []
    seen = set()
    for path in tracked:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        files.append(
            {
                "path": path.relative_to(repo_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {"manifest_version": "source_snapshot_v2", "version_control": "source_snapshot_without_git_metadata", "files": files}
    (run_dir / "source_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class TwoStageRiskScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        head_warmup_steps: int,
        total_steps: int,
        trunk_lr: float,
        head_lr: float,
        warmup_head_lr: float,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.head_warmup_steps = int(head_warmup_steps)
        self.total_steps = int(total_steps)
        self.trunk_lr = float(trunk_lr)
        self.head_lr = float(head_lr)
        self.warmup_head_lr = float(warmup_head_lr)
        self.min_lr_ratio = float(min_lr_ratio)
        self.last_step = -1

    def step(self, step: int) -> None:
        self.last_step = int(step)
        if step < self.head_warmup_steps:
            trunk_lr = 0.0
            head_lr = self.warmup_head_lr
        else:
            progress = min(
                1.0,
                (step - self.head_warmup_steps) / max(1, self.total_steps - self.head_warmup_steps),
            )
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
            trunk_lr = self.trunk_lr * factor
            head_lr = self.head_lr * factor
        self.optimizer.param_groups[0]["lr"] = float(trunk_lr)
        self.optimizer.param_groups[1]["lr"] = float(head_lr)

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.step(int(state["last_step"]))
