"""Build a MIMIC-IV Track-R (v2.1 data contract) static-prefix overlay.

MIMIC only has sex as a static field, so bmi/smoking/alcohol are always 'missing'.
This keeps the 4-field static prefix and the frozen Track-R contract identical to UKB.
"""
from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT

import argparse
import csv
import gzip
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = str(REPO_ROOT)
import sys
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/scripts")

from semantic_delphi_ukb.track_r_contract import static_token_keys  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-data-dir", type=Path, required=True)
    p.add_argument("--patients-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


# Mirror the frozen v2.1 static fields (categories only; MIMIC fills sex, rest 'missing').
FIELDS = {
    "sex": {"categories": ["female", "male", "missing"]},
    "bmi": {"categories": ["missing", "lt18.5", "18.5_to_lt25", "25_to_lt30", "gte30"]},
    "smoking": {"categories": ["missing", "never", "previous", "current"]},
    "alcohol": {"categories": [
        "missing", "never", "special_occasions", "one_to_three_per_month",
        "once_or_twice_per_week", "three_or_four_times_per_week", "daily_or_almost_daily",
    ]},
}
LOGICAL_ORDER = ["sex", "bmi", "smoking", "alcohol"]

STATIC_PREFIX = {
    "enabled": True,
    "logical_order": LOGICAL_ORDER,
    "fixed_length": 4,
    "token_templates": {f: f"static:{f}:{{category}}" for f in LOGICAL_ORDER},
    "age_anchor": {
        "field_id": 21022,
        "instance": "0.0",
        "meaning": "age_at_recruitment",
        "fallback": "birth_anchor_sequence_start_age",
        "fallback_static_value_policy": "set_recruitment_measured_bmi_smoking_alcohol_to_missing",
        "allow_anchor_after_first_dynamic_event": True,
        "sequence_order_is_not_chronological": True,
        "attention_order": "causal_by_sequence_position",
        "temporal_visibility": "a_dynamic_query_may_attend_static_prefix_only_when_query_age_days_gte_static_anchor_age_days",
        "forbid_age_sort_after_prefix": True,
    },
}
DYNAMIC_BOS = {
    "required": True,
    "fixed_length": 1,
    "token_key": "dynamic:BOS",
    "sequence_position": "after_static_prefix_before_dynamic_window",
    "age_anchor": "first_dynamic_clinical_event_age_in_retained_window",
}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    source = args.source_data_dir
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    # load patient gender + anchor_age
    gender = {}
    anchor = {}
    with gzip.open(args.patients_csv, "rt") as f:
        for row in csv.DictReader(f):
            sid = row["subject_id"]
            gender[sid] = row.get("gender", "")
            try:
                anchor[sid] = float(row["anchor_age"]) * 365.25
            except (ValueError, TypeError):
                anchor[sid] = None

    source_manifest = json.loads((source / "prepare_manifest.json").read_text(encoding="utf-8"))
    source_embeddings = np.load(source / source_manifest["semantic_output"]).astype(np.float32)
    source_vocab_size = source_embeddings.shape[0]

    # token ids for static prefix (appended after dynamic vocab)
    all_keys = []
    for name in LOGICAL_ORDER:
        for category in FIELDS[name]["categories"]:
            key = f"static:{name}:{category}"
            if key not in all_keys:
                all_keys.append(key)
    token_id = {key: source_vocab_size + i for i, key in enumerate(all_keys)}
    dynamic_bos_token_id = source_vocab_size + len(all_keys)
    mask_token_id = dynamic_bos_token_id + 1

    # labels.csv: token_key per line, index = token_id
    labels = ["Padding"] * source_vocab_size
    vocab_csv = source / source_manifest["vocab_csv"]
    with vocab_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tid = int(row["token_id"])
            if 0 <= tid < source_vocab_size:
                labels[tid] = row["token_key"].strip()
    labels = labels + all_keys + ["dynamic:BOS", "dynamic:MASK"]

    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for split in ("train", "val", "test"):
            eids = []
            with (source / f"{split}_patient_index.csv").open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    eids.append(row["eid"])
            data = np.memmap(source / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
            first_age = {}
            for eid, age_days, _t in data:
                first_age.setdefault(int(eid), float(age_days))
            prefix_rows = []
            anchor_rows = []
            for eid in eids:
                g = gender.get(eid, "")
                sex_cat = "male" if g == "M" else ("female" if g == "F" else "missing")
                cats = {"sex": sex_cat, "bmi": "missing", "smoking": "missing", "alcohol": "missing"}
                prefix_rows.append([token_id[k] for k in static_token_keys(cats, LOGICAL_ORDER)])
                a = anchor.get(eid)
                if a is None or a <= 0:
                    a = max(1.0, first_age.get(int(eid), 1.0))
                anchor_rows.append(a)
            np.save(staging / f"{split}_static_prefix_token_ids.npy", np.asarray(prefix_rows, dtype=np.int64))
            np.save(staging / f"{split}_static_anchor_age_days.npy", np.asarray(anchor_rows, dtype=np.float32))
            for suffix in (".bin", "_patient_index.csv", "_followup_end_age_days.npy", "_static.npy"):
                shutil.copy(source / f"{split}{suffix}", staging / f"{split}{suffix}")
        shutil.copytree(source / "vocab", staging / "vocab", dirs_exist_ok=True)
        (staging / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")
        extended = np.vstack((source_embeddings, np.zeros((len(all_keys) + 2, source_embeddings.shape[1]), dtype=np.float32)))
        np.save(staging / "semantic_input_embeddings_64d.npy", extended)
        manifest = {
            **source_manifest,
            "semantic_output": "semantic_input_embeddings_64d.npy",
            "vocab_size": int(extended.shape[0]),
            "static_feature_order": [],
            "static_prefix": STATIC_PREFIX,
            "static_token_ids": token_id,
            "dynamic_bos_token_id": dynamic_bos_token_id,
            "dynamic_bos": DYNAMIC_BOS,
            "mask_token_id": mask_token_id,
            "dynamic_vocab_size": source_vocab_size,
            "source_dynamic_data_dir": str(source.resolve()),
        }
        (staging / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "vocab_size": manifest["vocab_size"], "n_static": len(all_keys)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
