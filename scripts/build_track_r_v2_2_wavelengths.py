"""Build the frozen Track R v2.2 RoPE wavelength manifest from train only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.track_r_contract import load_track_r_protocol  # noqa: E402
from semantic_delphi_ukb.track_r_v2_2 import (  # noqa: E402
    build_wavelength_manifest,
    positive_consecutive_gaps,
    write_wavelength_manifest,
)
from utils import get_p2i  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--data-dir", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_r_protocol(args.protocol)
    if protocol["protocol_id"] != "track_r_v2_2":
        raise ValueError("wavelength builder requires Track R v2.2")
    data = np.memmap(args.data_dir / "train.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    gaps = positive_consecutive_gaps(data, get_p2i(data))
    architecture = protocol["architecture"]
    manifest = build_wavelength_manifest(
        gaps,
        n_heads=int(architecture["n_head"]),
        head_dim=int(architecture["n_embd"]) // int(architecture["n_head"]),
    )
    write_wavelength_manifest(args.output, manifest)
    print(json.dumps({
        "output": str(args.output),
        "sha256": manifest["sha256"],
        "positive_gap_count": int(gaps.size),
        "wavelength_lo_days": manifest["wavelength_lo_days"],
        "wavelength_hi_days": manifest["wavelength_hi_days"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
