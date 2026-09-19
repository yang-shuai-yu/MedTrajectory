"""Compile MIMIC generation-eval summaries into a comparison table for Experiments 1 & 2."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

BASE = Path(str(MIMIC_ROOT / "eval"))
MODELS = [
    ("M1 (diagnosis only)", "m1_abs", "Exp1"),
    ("M2 (+procedure)", "m2_abs", "Exp1"),
    ("M3 (full)", "m3_abs", "Exp1"),
    ("M3 Relative", "m3_rel", "Exp2"),
]
KEYS = [
    ("hit_at_1", "Hit@1"),
    ("hit_at_10", "Hit@10"),
    ("diagnosis_jaccard", "Diag Jaccard"),
    ("diagnosis_recall", "Diag Recall"),
    ("first_event_time_mae_days", "Time MAE (d)"),
    ("event_count_mae", "Count MAE"),
    ("sequence_edit_distance", "Seq edit"),
    ("death_brier", "Death Brier"),
]

rows = []
for label, name, exp in MODELS:
    p = BASE / name / "summary.json"
    if not p.exists():
        rows.append({"model": label, "exp": exp, "status": "MISSING"})
        continue
    m = json.loads(p.read_text())["metrics"]
    row = {"model": label, "exp": exp, "n": m.get("patient_count"), "status": "ok"}
    for key, _ in KEYS:
        v = m.get(key)
        row[key] = None if v is None else float(v)
    rows.append(row)

print("| Model | Exp | n | " + " | ".join(lbl for _, lbl in KEYS) + " |")
print("|---|---|---:" + "|---:" * len(KEYS) + "|")
for r in rows:
    if r.get("status") != "ok":
        print(f"| {r['model']} | {r['exp']} | - | MISSING |")
        continue
    vals = []
    for key, _ in KEYS:
        v = r.get(key)
        vals.append("NA" if v is None else f"{v:.4f}")
    print(f"| {r['model']} | {r['exp']} | {r['n']} | " + " | ".join(vals) + " |")
