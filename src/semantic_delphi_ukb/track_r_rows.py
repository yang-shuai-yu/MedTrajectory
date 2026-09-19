from __future__ import annotations

import gzip
import io
import json
from pathlib import Path


def resolve_rows_path(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    if path.suffix != ".gz":
        compressed = Path(f"{path}.gz")
        if compressed.exists():
            return compressed
    raise FileNotFoundError(path)


def read_json(path: Path):
    path = resolve_rows_path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_json_bytes(path: Path) -> bytes:
    path = resolve_rows_path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def write_json_gzip(path: Path, payload) -> Path:
    path = Path(path)
    if path.suffix != ".gz":
        raise ValueError("gzip JSON output must end in .gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
    return path
