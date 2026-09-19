"""Inspect paired test-row label disagreements without running bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def key(row):
    age = row.get("age_start_years")
    return (int(row["patient_index"]), str(row["disease_id"]), float(row["horizon_years"]), str(row.get("sex", "")), None if age is None else float(age))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, help="name=rows.json")
    args = p.parse_args()
    models = {}
    for spec in args.input:
        name, raw = spec.split("=", 1)
        rows = json.loads(Path(raw).read_text(encoding="utf-8"))
        models[name] = {key(row): row for row in rows}
        print(name, "rows", len(rows), "keys", len(models[name]))
    names = list(models)
    for left, right in zip(names, names[1:]):
        common = sorted(set(models[left]) & set(models[right]))
        mismatches = [item for item in common if int(models[left][item]["label"]) != int(models[right][item]["label"])]
        age_delta = [item for item in common if float(models[left][item].get("prediction_age_days", -1)) != float(models[right][item].get("prediction_age_days", -1))]
        by_group = Counter((item[1], item[2]) for item in mismatches)
        print(left, right, "common", len(common), "label_mismatches", len(mismatches), "prediction_age_mismatches", len(age_delta), "groups", by_group.most_common(20))
        for item in mismatches[:5]:
            print("example", item, models[left][item].get("label"), models[right][item].get("label"))


if __name__ == "__main__":
    raise SystemExit(main())
