from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from semantic_delphi_ukb import patient_future_demo as base
from semantic_delphi_ukb.modern_selected_utils import load_modern_model, modern_model_spec


REPO_DIR = Path(__file__).resolve().parent.parent


def main(argv: Optional[Sequence[str]] = None) -> int:
    base.load_model = lambda _spec, split, device: load_modern_model(split=split, device=device)
    base.select_model_spec = lambda _model_id: modern_model_spec()
    parser = base.build_parser()
    parser.set_defaults(
        case_id="AF-FUTURE-MODERN-001",
        model_id="modern",
        output_dir=REPO_DIR / "tests" / "output" / "patient_future_demo_modern",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = base.build_payload(args)
    payload["model_id"] = "modern"
    payload["model"] = "MedTrajectory modern baseline"
    payload["visuals"] = base.write_visuals(args.output_dir, args.case_id, payload)

    json_path = args.output_dir / f"{args.case_id}_demo.json"
    md_path = args.output_dir / f"{args.case_id}_demo.md"
    json_path.write_text(base.json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    base.write_markdown(md_path, payload)
    print(
        base.json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "visuals": payload["visuals"],
                "split_patient_index": payload["split_patient_index"],
                "target_rank_among_diagnosis_tokens": payload["target_rank_among_diagnosis_tokens"],
                "target_in_top5_diagnosis": payload["target_in_top5_diagnosis"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
