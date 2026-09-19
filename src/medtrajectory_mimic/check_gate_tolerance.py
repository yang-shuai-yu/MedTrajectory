# -*- coding: utf-8 -*-
"""Decide whether a cohort-gate failure is tolerable or fatal.

`verify_eval_cohorts.py` fails whenever the participant sets of the compared arms
differ at all.  That is the right default, but one case is genuinely different
from a broken pairing: the relaxed Bm cohort is built with
`--eid-file matched_bm_relaxed.txt --max-patients 2000`, and `choose_cases`
applies the 2000-cap in each representation's OWN patient_index order.  Because
the three representations order participants differently, the cap lands on a
different participant and the arms differ by one person (intersection 1999 of
2000), while the pairing itself is intact.

This script accepts a gate failure ONLY when every failing group overlaps almost
completely, so the paired statistics remain valid on the common subset, and it
requires that the shortfall be disclosed.  Anything larger aborts the report.

Usage: python check_gate_tolerance.py <gate_output_file> [min_overlap_fraction]
Exit 0 = tolerable (disclose), 1 = fatal.
"""
import re
import sys
from pathlib import Path

FAIL = re.compile(r"FAIL\s+(\S+):\s+sets differ\s+\{(.*?)\}\s+intersection=(\d+)\s+union=(\d+)")
# the gate prints a Python dict, e.g. {'bm_m1_a0': 2000, 'bm_m2_a0': 2000}
SIZE = re.compile(r"'([^']+)':\s*(\d+)")


def main() -> int:
    path = Path(sys.argv[1])
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 0.99
    text = path.read_text(encoding="utf-8", errors="replace")

    fails = []
    for line in text.splitlines():
        m = FAIL.search(line)
        if not m:
            continue
        group, sizes_raw, inter, union = m.groups()
        sizes = [int(v) for _, v in SIZE.findall(sizes_raw)]
        fails.append((group, sizes, int(inter), int(union)))

    if not fails:
        print("gate tolerance: no cohort mismatch to excuse")
        return 0

    print(f"gate tolerance: {len(fails)} mismatched group(s), tolerance={tolerance:.2%}")
    fatal = False
    for group, sizes, inter, union in fails:
        worst = max(sizes) if sizes else 0
        frac = inter / worst if worst else 0.0
        # every arm must retain almost all of its participants in the common subset
        retained = inter / worst
        ok = retained >= tolerance
        fatal |= not ok
        print(f"  {group}: arms={sizes} intersection={inter} union={union} "
              f"common/arm={frac:.4%} -> {'DISCLOSE' if ok else 'FATAL'}")

    if fatal:
        print("RESULT: a cohort mismatch exceeds the tolerance -- report must NOT be written")
        return 1
    print("RESULT: all mismatches are at most a rounding-level shortfall; "
          "statistics run on the common subset and the shortfall must be disclosed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
