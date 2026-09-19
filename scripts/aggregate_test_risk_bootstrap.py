# -*- coding: utf-8 -*-
"""Formal patient-level paired bootstrap for Track R v2.2 locked-TEST risk.

Reuses finalize_track_r_v2_2.py aggregation helpers, pointed at the TEST rows.
Reports A2 - A0 macro-AUROC delta, 95% CI (seed-averaged patient bootstrap,
10,000 replicates), and the non-inferiority verdict vs margin -0.005.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import sys
from pathlib import Path

ROOT = Path(str(REPO_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from semantic_delphi_ukb.finalize_track_r_v2_2 import (  # noqa: E402
    _load_compact_inputs,
    _compact_metrics,
    _compact_bootstrap_macro_values,
    seed_averaged_bootstrap_delta,
)

OUT = ROOT / "results/track_r_v2_2/test"
SEEDS = [42, 43, 44]
MODELS = ["A0", "A2"]
REPLICATES = 10000
BOOTSTRAP_SEED = 20260815
PERCENTILE = [2.5, 97.5]
MARGIN = -0.005


def main():
    specs = []
    for seed in SEEDS:
        for model in MODELS:
            name = f"seed{seed}:{model}"
            path = OUT / f"seed{seed}" / model / "rows.json.gz"
            specs.append(f"{name}={path}")

    compact = _load_compact_inputs(specs)
    clinical_summary, _ = _compact_metrics(compact, bins=10)
    bootstrap_macro = _compact_bootstrap_macro_values(
        compact, bootstrap=REPLICATES, seed=BOOTSTRAP_SEED
    )
    _, ci = seed_averaged_bootstrap_delta(
        bootstrap_macro, compact["names"], SEEDS, "A2", "A0", PERCENTILE
    )

    per_seed = [
        clinical_summary[f"seed{seed}:A2"]["auc"] - clinical_summary[f"seed{seed}:A0"]["auc"]
        for seed in SEEDS
    ]
    mean_delta = sum(per_seed) / len(per_seed)

    print("=== Track R v2.2 locked-TEST risk: A2 vs A0 (macro AUROC) ===")
    print("per-seed A2-A0 :", [round(x, 6) for x in per_seed])
    print(f"mean delta      : {mean_delta:+.6f}")
    print(f"95% CI (bootstrap): [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}]")
    print(f"non-inferiority margin : {MARGIN}")
    print(f"non-inferiority passed : {ci['ci95_low'] > MARGIN}")
    print(f"\nper-model macro AUROC (summary):")
    for m in ["A0", "A2"]:
        vals = [clinical_summary[f"seed{s}:{m}"]["auc"] for s in SEEDS]
        print(f"  {m}: " + " ".join(f"{v:.5f}" for v in vals) + f"  mean={sum(vals)/len(vals):.5f}")


if __name__ == "__main__":
    main()
