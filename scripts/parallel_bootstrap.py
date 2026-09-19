"""Parallel, execution-equivalent replacement for
compare_horizon_control_tasks._compact_bootstrap_macro_values.

STATUS: verification/analysis utility. It is NOT part of the frozen pipeline and
changes nothing under `src/`. It exists because the shipped function is a
single-process loop over 10,000 replicates, costing ~2.9 h per validation
population (measured 504.5 s / 10,000 replicates on the 262,200-row test data,
scaled by the 20.88x row ratio) while occupying 1 core of a 20-core quota.

EQUIVALENCE - both established by measurement, not assumed:
  * On the test data the concatenated chunk output is BIT-IDENTICAL to the matrix
    the shipped function produces, for BOTH the full and the anchor-filtered
    population.
  * On the validation full population it reproduces
    results/track_r_v2_2/final_assessment/assessment.json clinical_noninferiority
    with error exactly 0.000e+00 on both CI endpoints.
When a number is published, cite the procedure, not this script.

WHY: the shipped function is a single-process loop whose cost is per-replicate
memory traffic over every row. Replicates are statistically independent, so the
work partitions trivially.

EXACTNESS: every worker replays the identical RNG stream, drawing exactly one
`rng.integers(0, N, size=N)` per replicate in replicate order, and accumulates
over strata in the same order as the shipped loop (sequentially, NOT np.sum, so
the floating-point summation order is preserved). The replicate-invariant
pieces (argsort order, tie starts, sorted labels, mapped patient rows) are
precomputed once instead of once per replicate - they do not depend on the
multiplicities. Output rows are concatenated in replicate order, so the array
handed to seed_averaged_bootstrap_delta is the shipped function's output.

NOTE: hoisting the argsort alone buys only ~1.45x and introduces ~1e-16
differences by changing the summation order. The speedup comes from the process
fan-out, not from the hoist; the hoist is retained only because it is exact.

Subcommands:
  prep   --caches a=path,b=path --out DIR        build .npy inputs (slow: gzip JSON)
  chunk  --npy DIR --lo B --hi B --out F.npy     replicates [lo*8, hi*8)
  ci     --npy DIR --chunks a.npy,b.npy --expect mean,lo,hi
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import argparse, gzip, json, os, sys, time
import numpy as np

R = str(REPO_ROOT)
sys.path.insert(0, R + "/src")
from semantic_delphi_ukb.compare_horizon_control_tasks import _load_compact_inputs
from semantic_delphi_ukb.finalize_track_r_v2_2 import seed_averaged_bootstrap_delta

BATCH = 8
SEED = 20260815
TOTAL = 10000


def _groups_from_strata(stratum_ids):
    """Reproduce _load_compact_inputs' grouping byte-for-byte."""
    order = np.argsort(stratum_ids, kind="stable")
    boundaries = np.flatnonzero(np.diff(stratum_ids[order])) + 1
    return np.split(order, boundaries)


def cmd_prep(a):
    specs = a.caches.split(",")
    print("loading %d caches ..." % len(specs), flush=True)
    t = time.time()
    compact = _load_compact_inputs(specs)
    print("loaded in %.1f s | rows=%d | strata=%d" % (
        time.time() - t, len(compact["patient_ids"]), len(compact["groups"])), flush=True)

    os.makedirs(a.out, exist_ok=True)
    names = [str(n) for n in compact["names"]]
    patient_ids = np.asarray(compact["patient_ids"], dtype=np.int64)
    uniq, pinv = np.unique(patient_ids, return_inverse=True)
    np.save(a.out + "/pinv.npy", pinv.astype(np.int32))
    np.save(a.out + "/labels.npy", np.asarray(compact["labels"], dtype=np.int8))
    np.save(a.out + "/stratum_ids.npy", np.asarray(compact["stratum_ids"], dtype=np.int32))
    for n in names:
        np.save("%s/score_%s.npy" % (a.out, n), np.asarray(compact["scores"][n], dtype=np.float64))
    json.dump({"names": names, "n_pat": int(len(uniq)), "rows": int(len(patient_ids))},
              open(a.out + "/meta.json", "w"))
    print("saved %d scores, rows=%d, n_pat=%d -> %s" % (
        len(names), len(patient_ids), len(uniq), a.out), flush=True)
    # sanity: grouping reproduced from saved stratum_ids must match
    g1 = _groups_from_strata(np.asarray(compact["stratum_ids"], dtype=np.int32))
    g0 = compact["groups"]
    assert len(g1) == len(g0) and all(np.array_equal(x, y) for x, y in zip(g0, g1)), "group mismatch"
    print("grouping reproduced exactly (%d strata)" % len(g0), flush=True)


def _hoist(npy, names):
    """Replicate-invariant pieces per (name, stratum), computed once."""
    labels = np.load(npy + "/labels.npy")
    pinv = np.load(npy + "/pinv.npy")
    sid = np.load(npy + "/stratum_ids.npy")
    groups = _groups_from_strata(sid)
    pre = {}
    for name in names:
        scores = np.load("%s/score_%s.npy" % (npy, name))
        per = []
        for idx in groups:
            s = scores[idx]
            order = np.argsort(s, kind="mergesort")
            ss = s[order]
            starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1].astype(np.int64)
            lab = labels[idx][order].astype(np.float64)
            per.append((starts, lab, 1.0 - lab, pinv[idx][order]))
        pre[name] = per
        del scores
    return groups, pre


def _auc_hoisted(starts, pos_s, neg_s, rows, mult):
    w = mult[:, rows].astype(np.float64)
    positives = np.add.reduceat(w * pos_s, starts, axis=1)
    negatives = np.add.reduceat(w * neg_s, starts, axis=1)
    nbefore = np.cumsum(negatives, axis=1) - negatives
    num = np.sum(positives * (nbefore + 0.5 * negatives), axis=1)
    den = np.sum(positives, axis=1) * np.sum(negatives, axis=1)
    return np.divide(num, den, out=np.full(len(mult), np.nan), where=den > 0)


def cmd_chunk(a):
    meta = json.load(open(a.npy + "/meta.json"))
    names, n_pat = meta["names"], meta["n_pat"]
    lo, hi = int(a.lo), int(a.hi)          # batch indices
    r0, r1 = lo * BATCH, hi * BATCH
    print("chunk batches [%d,%d) = replicates [%d,%d)" % (lo, hi, r0, r1), flush=True)
    t = time.time()
    groups, pre = _hoist(a.npy, names)
    print("  hoisted in %.1f s (%d strata x %d names)" % (time.time() - t, len(groups), len(names)),
          flush=True)

    rng = np.random.default_rng(SEED)
    # replay the identical stream for replicates [0, r0) - cheap relative to the AUC work
    t = time.time()
    for _ in range(r0):
        rng.integers(0, n_pat, size=n_pat)
    print("  RNG replayed to replicate %d in %.1f s" % (r0, time.time() - t), flush=True)

    n_batches = hi - lo
    out = np.empty((n_batches * BATCH, len(names)), dtype=np.float64)
    t = time.time()
    for bi in range(n_batches):
        cnt = BATCH
        mult = np.empty((cnt, n_pat), dtype=np.uint16)
        for i in range(cnt):
            sel = rng.integers(0, n_pat, size=n_pat)
            mult[i] = np.bincount(sel, minlength=n_pat)
        macro_sum = np.zeros((cnt, len(names)), dtype=np.float64)
        valid_count = np.zeros(cnt, dtype=np.int32)
        for gi in range(len(groups)):
            aucs = np.column_stack([
                _auc_hoisted(*pre[name][gi], mult) for name in names
            ])
            valid = np.all(np.isfinite(aucs), axis=1)
            macro_sum[valid] += aucs[valid]
            valid_count[valid] += 1
        out[bi * BATCH:(bi + 1) * BATCH] = np.divide(
            macro_sum, valid_count[:, None], out=np.full_like(macro_sum, np.nan),
            where=valid_count[:, None] > 0)
        if (bi + 1) % 25 == 0 or bi + 1 == n_batches:
            el = time.time() - t
            print("    %d/%d batches  %.1f s  (%.1f s/batch)" % (
                bi + 1, n_batches, el, el / (bi + 1)), flush=True)
    np.save(a.out, out)
    print("wrote %s shape=%s in %.1f s" % (a.out, out.shape, time.time() - t), flush=True)


def cmd_ci(a):
    names = json.load(open(a.npy + "/meta.json"))["names"]
    parts = [np.load(p) for p in a.chunks.split(",")]
    bm = np.concatenate(parts, axis=0)
    assert bm.shape == (TOTAL, len(names)), bm.shape
    per_seed, ci = seed_averaged_bootstrap_delta(
        bm, names, (42, 43, 44), "A2", "A0", [2.5, 97.5])
    print("bootstrap matrix:", bm.shape)
    print("  ci95      : [%.15f, %.15f]" % (ci["ci95_low"], ci["ci95_high"]))
    print("  ci centre : %.15f   half-width %.15f" % (
        (ci["ci95_low"] + ci["ci95_high"]) / 2.0,
        (ci["ci95_high"] - ci["ci95_low"]) / 2.0))
    # NOTE: `per_seed` from seed_averaged_bootstrap_delta has shape
    # (bootstrap_replicates, training_seeds). Printing its rows as seeds - or its
    # mean as an observed difference - is the reported defect documented in
    # docs/CROSS_AGENT_VERIFICATION_HANDLING_20260916.md section 1.1. The observed
    # seed deltas come from the observed per-configuration summaries, not from here.
    print("  (observed per-seed deltas are NOT derivable from this matrix; see fb ci caller)")
    if a.expect:
        em, elo, ehi = [float(x) for x in a.expect.split(",")]
        print("  expected  : [%.15f, %.15f]  mean %.15f" % (elo, ehi, em))
        print("  err lo=%.3e  err hi=%.3e" % (abs(ci["ci95_low"] - elo), abs(ci["ci95_high"] - ehi)))
        ok = abs(ci["ci95_low"] - elo) < 1e-12 and abs(ci["ci95_high"] - ehi) < 1e-12
        print("  GATE:", "MATCH" if ok else "MISMATCH")
        if not ok:
            raise SystemExit("FAIL-FAST: CI did not reproduce")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--caches", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("chunk"); p.add_argument("--npy", required=True); p.add_argument("--lo", required=True)
    p.add_argument("--hi", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("ci"); p.add_argument("--npy", required=True); p.add_argument("--chunks", required=True)
    p.add_argument("--expect", default="")
    a = ap.parse_args()
    {"prep": cmd_prep, "chunk": cmd_chunk, "ci": cmd_ci}[a.cmd](a)
