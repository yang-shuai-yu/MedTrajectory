"""Select one validation-only sampler per Track G model family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.track_g_contract import assert_training_allowed, load_track_g_protocol  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, default=REPO_DIR / "configs/track_g_v1/TRACK_G_v1.json")
    value.add_argument("--grid-root", type=Path, default=REPO_DIR / "results/track_g_v1/sampler_grid")
    value.add_argument("--output", type=Path, default=REPO_DIR / "results/track_g_v1/manifests/sampler_selection.json")
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def config_key(sampler: dict) -> tuple[float, float, float]:
    return (
        float(sampler["temperature"]),
        float(sampler["top_p"]),
        float(sampler["death_logit_bias"]),
    )


def mean(values):
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else float("nan")


def select_candidate(candidates: list[dict]) -> dict:
    return max(
        candidates,
        key=lambda item: (
            item["diagnosis_jaccard"],
            item["first_event_top10"],
            -item["time_mae_days"],
            -item["death_brier"],
            -item["temperature"],
            -item["top_p"],
            -abs(item["death_logit_bias"]),
        ),
    )


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = load_track_g_protocol(args.protocol, REPO_DIR)
    assert_training_allowed(protocol)
    rows = []
    for path in sorted(args.grid_root.rglob("summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("protocol_manifest_sha256") != protocol["protocol_manifest_sha256"]:
            raise ValueError(f"grid summary protocol hash mismatch: {path}")
        if payload.get("split") != "val":
            raise ValueError(f"sampler selection may only consume validation summaries: {path}")
        rows.append(payload)

    expected = len(protocol["models"]) * len(protocol["seeds"])
    selection = protocol["sampler_selection"]
    expected *= len(selection["temperature_grid"])
    expected *= len(selection["top_p_grid"])
    expected *= len(selection["death_logit_bias_grid"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} sampler summaries, found {len(rows)}")

    family_for_model = {model["name"]: model["family"] for model in protocol["models"]}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(family_for_model[row["model"]], config_key(row["sampler"]))].append(row)

    expected_members = defaultdict(int)
    for model in protocol["models"]:
        expected_members[model["family"]] += len(protocol["seeds"])
    selected = {}
    audit = {}
    for (family, key), values in grouped.items():
        if len(values) != expected_members[family]:
            raise ValueError(f"incomplete seed/model coverage for family={family}, sampler={key}")
        metrics = {
            "diagnosis_jaccard": mean(value["metrics"]["diagnosis_jaccard"] for value in values),
            "first_event_top10": mean(value["metrics"]["hit_at_10"] for value in values),
            "time_mae_days": mean(value["metrics"]["first_event_time_mae_days"] for value in values),
            "death_brier": mean(value["metrics"]["death_brier"] for value in values),
        }
        audit.setdefault(family, []).append({
            "temperature": key[0],
            "top_p": key[1],
            "death_logit_bias": key[2],
            **metrics,
        })

    for family, candidates in audit.items():
        best = select_candidate(candidates)
        selected[family] = dict(best)

    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": protocol["protocol_manifest_sha256"],
        "source": "validation_only",
        "selected_by_family": selected,
        "candidate_audit": audit,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["selection_sha256"] = hashlib.sha256(encoded).hexdigest()
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
