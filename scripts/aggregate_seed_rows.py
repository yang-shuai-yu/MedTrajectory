"""Aggregate multi-seed baseline rows into a single ensembled rows.json.gz.

For stochastic baselines (MDRMF-Clinical-R, Med-BERT-*) trained on seeds
42/43/44, average the per-patient prediction score across seeds, keeping the
label/stratum fields from the first seed.  The output is a JSON array written
with the same helpers used by the rest of the pipeline, so it is directly
usable by compare_horizon_control_tasks.py.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from semantic_delphi_ukb.track_r_rows import read_json, write_json_gzip


def key(row: dict):
    age = row.get("age_start_years")
    return (
        int(row["patient_index"]),
        str(row["disease_id"]),
        float(row["horizon_years"]),
        str(row.get("sex", "")),
        None if age is None else float(age),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True,
                   help="rows.json.gz for one seed; repeat for each seed")
    p.add_argument("--output", type=Path, required=True,
                   help="output rows.json.gz")
    p.add_argument("--model", required=True, help="model display name")
    args = p.parse_args(argv)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for raw in args.input:
        for row in read_json(Path(raw)):
            grouped[key(row)].append(row)

    out_rows = []
    for item, rows in grouped.items():
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"label mismatch for key {item}: {labels}")
        base = dict(rows[0])
        base["score"] = sum(float(row["score"]) for row in rows) / len(rows)
        base["model"] = args.model
        out_rows.append(base)

    output = Path(args.output)
    write_json_gzip(output, out_rows)
    print(f"aggregated {len(grouped)} keys from {len(args.input)} seeds -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
