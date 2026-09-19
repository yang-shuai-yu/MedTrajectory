"""Full inventory: which models are trained, which are evaluated."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))
RUNS = DATA / "runs"

GROUPS = {
    "事件级 multitype (M1/M2/M3)": ["m1_abs", "m2_abs", "m3_abs", "m3_rel"],
    "事件级 A0/A2 (Track-R)": ["trackr_a0_abs", "trackr_a2_rel"],
    "Design A 主诊断": ["vA_a0", "vA_a2", "vA_a0_s43", "vA_a2_s43", "vA_a0_s44", "vA_a2_s44"],
    "Design B 主诊断+主操作": ["vB_m3_a0", "vB_m3_a2", "vB_m3_a0_s43", "vB_m3_a2_s43",
                              "vB_m3_a0_s44", "vB_m3_a2_s44"],
    "Design C 前2码合成": ["vC_a0", "vC_a2", "vC_a0_s43", "vC_a2_s43", "vC_a0_s44", "vC_a2_s44"],
    "Bm 配对 multitype": ["bm_m1_a0", "bm_m2_a0", "bm_m3_a0",
                          "bm_m1_a0_s43", "bm_m2_a0_s43", "bm_m3_a0_s43",
                          "bm_m1_a0_s44", "bm_m2_a0_s44", "bm_m3_a0_s44"],
    "ETHOS-Matched baseline": ["ethos_matched_a"],
}

print("=== TRAINING (100k iters each) ===")
total = done = 0
for label, names in GROUPS.items():
    marks = []
    for n in names:
        ck = RUNS / n / "checkpoints" / "last.pt"
        st = RUNS / n / "status.json"
        it = None
        if st.exists():
            try:
                it = json.loads(st.read_text()).get("iteration")
            except Exception:
                pass
        ok = ck.exists() and it == 100000
        total += 1
        done += int(ok)
        marks.append(f"{n}{'✓' if ok else f'({it})'}")
    print(f"  {label:26s} {sum(1 for n in names if (RUNS / n / 'checkpoints' / 'last.pt').exists())}/{len(names)}")
print(f"  TOTAL {done}/{total} trained")

print("\n=== EVALUATION (each model x 2 cohort definitions) ===")
EVAL = {
    "事件级": (["eval_quick/m1_abs", "eval_quick/m2_abs", "eval_quick/m3_abs", "eval_quick/m3_rel",
              "eval_trackr/trackr_a0_abs", "eval_trackr/trackr_a2_rel"], 6),
    "Design A": (["eval_visitA_seeds_ukb/vA_a0", "eval_visitA_seeds_ukb/vA_a2",
                  "eval_visitA_seeds_ukb/vA_a0_s43", "eval_visitA_seeds_ukb/vA_a2_s43",
                  "eval_visitA_seeds_ukb/vA_a0_s44", "eval_visitA_seeds_ukb/vA_a2_s44",
                  "eval_visitA_seeds_relaxed/vA_a0", "eval_visitA_seeds_relaxed/vA_a2",
                  "eval_visitA_seeds_relaxed/vA_a0_s43", "eval_visitA_seeds_relaxed/vA_a2_s43",
                  "eval_visitA_seeds_relaxed/vA_a0_s44", "eval_visitA_seeds_relaxed/vA_a2_s44"], 12),
    "Design B": (["eval_B3_ukb/vB_m3_a0", "eval_B3_ukb/vB_m3_a2"], 12),
    "Design C": (["eval_C3_ukb/vC_a0", "eval_C3_ukb/vC_a2",
                  "eval_C3_ukb/vC_a0_s43", "eval_C3_ukb/vC_a2_s43"], 12),
    "Bm 配对": (["eval_bm3_ukb/bm_m1_a0", "eval_bm3_ukb/bm_m2_a0", "eval_bm3_ukb/bm_m3_a0",
                "eval_bm3_ukb/bm_m1_a0_s43", "eval_bm3_ukb/bm_m2_a0_s43", "eval_bm3_ukb/bm_m3_a0_s43",
                "eval_bm3_ukb/bm_m1_a0_s44", "eval_bm3_ukb/bm_m2_a0_s44", "eval_bm3_ukb/bm_m3_a0_s44"], 18),
    "ETHOS": (["eval_ethos_ukb/ethos_matched_a", "eval_ethos_relaxed/ethos_matched_a"], 2),
}
tot = dn = 0
for label, (paths, expected) in EVAL.items():
    have = sum(1 for p in paths if (DATA / p / "summary.json").exists())
    tot += expected
    dn += have
    print(f"  {label:12s} {have:2d}/{expected}")
print(f"  TOTAL {dn}/{tot} evaluated")
