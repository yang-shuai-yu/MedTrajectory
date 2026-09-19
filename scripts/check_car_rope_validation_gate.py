"""Check the preregistered A3-versus-A0 CARoPE validation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.comparison.read_text(encoding="utf-8"))
    models = payload["models"]
    contrast = payload["paired_bootstrap"]["A0 - A3"]
    checks = {
        "a3_auc_gt_a0": float(models["A3"]["auc"]) > float(models["A0"]["auc"]),
        "a3_auprc_gt_a0": float(models["A3"]["auprc"]) > float(models["A0"]["auprc"]),
        "paired_auc_ci_gt_zero": float(contrast["ci95_high"]) < 0.0,
        "paired_auc_holm_p_lt_0_05": float(contrast["p_holm"]) < 0.05,
    }
    result = {
        "gate": "CARoPE_validation_component_sweep_v1",
        "comparison": str(args.comparison.resolve()),
        "primary_contrast": "A3 - A0",
        "delta_auc": -float(contrast["delta_auc"]),
        "ci95_low": -float(contrast["ci95_high"]),
        "ci95_high": -float(contrast["ci95_low"]),
        "p_holm": float(contrast["p_holm"]),
        "checks": checks,
        "ok": all(checks.values()),
    }
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite validation gate: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
