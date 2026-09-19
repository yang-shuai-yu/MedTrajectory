#!/usr/bin/env python3
"""Privacy-conscious JSONL, status, TensorBoard, and checkpoint monitoring."""

from __future__ import annotations

import json
import math
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (ValueError, TypeError, RuntimeError):
            pass
    return str(value)


def _numeric_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            result = float(item())
            return result if math.isfinite(result) else None
        except (ValueError, TypeError, RuntimeError):
            return None
    return None


class RunMonitor:
    def __init__(
        self,
        run_dir: str | Path,
        config: Optional[Mapping[str, Any]] = None,
        *,
        enable_tensorboard: bool = True,
        flush_secs: int = 30,
        heartbeat_secs: int = 60,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.enable_tensorboard = bool(enable_tensorboard)
        self.heartbeat_secs = int(heartbeat_secs)
        self._last_heartbeat = 0.0
        self._writer = None
        self.status_path = self.run_dir / "status.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            self._atomic_json_write(
                self.run_dir / "resolved_config.json",
                {"created_at": _utc_now(), "config": _json_safe(config)},
            )
        if self.enable_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(str(self.tensorboard_dir), flush_secs=flush_secs)

    def _atomic_json_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def _read_status(self) -> MutableMapping[str, Any]:
        if not self.status_path.exists():
            return {}
        try:
            with self.status_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def update_status(self, *, status: str, **fields: Any) -> None:
        payload = self._read_status()
        payload.update(_json_safe(fields))
        payload["status"] = status
        payload["updated_at"] = _utc_now()
        self._atomic_json_write(self.status_path, payload)

    def mark_running(self, **fields: Any) -> None:
        self.update_status(status="running", **fields)

    def mark_finished(self, **fields: Any) -> None:
        self.update_status(status="finished", **fields)
        self.flush()

    def mark_failed(self, exc: BaseException, **fields: Any) -> None:
        self.update_status(
            status="failed",
            error={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:4000],
                "traceback": "".join(traceback.format_exception(exc))[-12000:],
            },
            **fields,
        )
        self.flush()

    def log_step(self, global_step: int, metrics: Mapping[str, Any]) -> None:
        clean = {}
        for name, value in metrics.items():
            scalar = _numeric_scalar(value)
            if scalar is not None:
                clean[str(name)] = scalar
                if self._writer is not None:
                    self._writer.add_scalar(str(name), scalar, global_step)
        now = time.monotonic()
        if now - self._last_heartbeat >= self.heartbeat_secs:
            heartbeat_fields = {
                name.replace("/", "_"): value for name, value in clean.items()
            }
            heartbeat_fields.update(self.system_metrics())
            self.update_status(
                status="running",
                global_step=int(global_step),
                **heartbeat_fields,
            )
            self._last_heartbeat = now

    def log_epoch(self, metrics: Mapping[str, Any]) -> None:
        payload = {"timestamp": _utc_now(), **_json_safe(metrics)}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        step = int(metrics.get("global_step", metrics.get("epoch", 0)))
        if self._writer is not None:
            for name, value in metrics.items():
                scalar = _numeric_scalar(value)
                if scalar is not None and name not in {"epoch", "global_step"}:
                    tag = str(name) if "/" in str(name) else f"epoch/{name}"
                    self._writer.add_scalar(tag, scalar, step)
            self._writer.flush()

    def system_metrics(self) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        gib = 1024**3
        return {
            "gpu_device": int(device),
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / gib,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / gib,
            "gpu_memory_peak_gb": torch.cuda.max_memory_allocated(device) / gib,
        }

    def save_checkpoint(self, state: Mapping[str, Any], filename: str) -> Path:
        import torch

        if Path(filename).name != filename:
            raise ValueError("filename must not contain directory components")
        destination = self.checkpoint_dir / filename
        temp_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        torch.save(dict(state), temp_path)
        os.replace(temp_path, destination)
        return destination

    @staticmethod
    def capture_random_state() -> dict[str, Any]:
        import numpy as np
        import torch

        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
