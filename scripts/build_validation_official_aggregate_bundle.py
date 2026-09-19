"""Bundle model-specific official validation aggregates for the v2 consistency gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="model_name=calibration_auc_aggregates.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite official aggregate bundle: {args.out}")
    models = {}
    for spec in args.input:
        name, raw_path = spec.split("=", 1)
        if name in models:
            raise ValueError(f"duplicate model name: {name}")
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("horizon_risk_age_sex_macro", [])
        if not rows:
            raise ValueError(f"missing horizon_risk_age_sex_macro: {path}")
        models[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "horizon_risk_age_sex_macro": rows,
        }
    bundle = {
        "format": "paper_medical_control_official_bundle_v1",
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out), "models": sorted(models)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
