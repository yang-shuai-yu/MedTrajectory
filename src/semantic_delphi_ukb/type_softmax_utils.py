from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


EVENT_TYPE_TO_ID: Dict[str, int] = {
    "padding": -1,
    "diagnosis": 0,
    "procedure": 1,
    "cancer": 2,
    "death": 3,
    "no_event": 4,
}


def resolve_vocab_csv(data_dir: Path) -> Path:
    manifest_path = data_dir / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Path(str(manifest["vocab_csv"]))


def load_token_type_ids(data_dir: Path, vocab_size: int) -> Tuple[np.ndarray, Dict[str, int]]:
    token_type_ids = np.full(vocab_size, -1, dtype=np.int64)
    vocab_csv = resolve_vocab_csv(data_dir)
    with vocab_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            token_id = int(row["token_id"])
            event_type = row["event_type"].strip()
            if 0 <= token_id < vocab_size:
                token_type_ids[token_id] = EVENT_TYPE_TO_ID.get(event_type, -1)
    return token_type_ids, dict(EVENT_TYPE_TO_ID)
