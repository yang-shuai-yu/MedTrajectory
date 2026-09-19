"""Build matched-patient cohorts for the visit-level multitype comparison (Design B M1 vs M2).

Both representations are evaluated on the SAME eligible patients (intersection of the
eligibility sets computed from each representation's own trajectory), so that the
M1-vs-M2 contrast is a paired comparison over an identical patient population.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import argparse, csv, json, math, sys
from pathlib import Path

import numpy as np

ROOT = str(REPO_ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from utils import get_p2i  # noqa: E402


def eligible_eids(data_dir, split, mh, mf, bf, fy, device_unused=None):
    data = np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    eids = []
    with (Path(data_dir) / f"{split}_patient_index.csv").open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            eids.append(str(row["eid"]))
    out = []
    for pi in range(len(p2i)):
        s, l = p2i[pi]
        rows = np.asarray(data[int(s):int(s) + int(l)])
        order = np.argsort(rows[:, 1], kind="stable")
        ages = np.asarray([float(r[1]) for r in rows[order]])
        if len(ages) < mh + mf:
            continue
        cut = int(math.floor((len(ages) - 1) * bf))
        cut = max(mh - 1, cut)
        cut = min(cut, len(ages) - mf - 1)
        base = ages[cut]
        future = ages[cut + 1:][ages[cut + 1:] <= base + fy * 365.25]
        if len(future) < mf:
            continue
        out.append(eids[pi])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-a", required=True)
    ap.add_argument("--data-b", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--baseline-fraction", type=float, default=0.65)
    ap.add_argument("--followup-years", type=float, default=10.0)
    args = ap.parse_args()

    for tag, mh, mf, cap in (("ukb", 8, 3, 0), ("relaxed", 3, 2, 2000)):
        a = set(eligible_eids(args.data_a, args.split, mh, mf, args.baseline_fraction, args.followup_years))
        b = set(eligible_eids(args.data_b, args.split, mh, mf, args.baseline_fraction, args.followup_years))
        inter = sorted(a & b)
        if cap and len(inter) > cap:
            inter = inter[:cap]
        out = Path(f"{args.out_prefix}_{tag}.txt")
        out.write_text("\n".join(inter) + "\n", encoding="utf-8")
        print(json.dumps({"cohort": tag, "eligible_a": len(a), "eligible_b": len(b),
                          "intersection": len(a & b), "written": len(inter), "file": str(out)}))


if __name__ == "__main__":
    main()
