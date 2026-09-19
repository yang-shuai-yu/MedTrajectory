from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.evaluate_calibration_auc import main as evaluate_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--split" not in arguments:
        arguments.extend(["--split", "val"])
    if "--age-groups" not in arguments:
        arguments.extend(["--age-groups", "50,55,60,65,70,75"])
    return evaluate_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
