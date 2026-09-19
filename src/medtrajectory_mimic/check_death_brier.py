"""Check why Death Brier is exactly 0 in the Bm multitype comparison, and whether
death tokens exist in each Bm vocabulary."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))

for m in (1, 2, 3):
    s = json.loads((DATA / "eval_bm3_ukb" / f"bm_m{m}_a0" / "summary.json").read_text())
    met = s["metrics"]
    print(f"M{m}: death_brier={met.get('death_brier')} death_ece={met.get('death_ece')} "
          f"num_cases={s.get('num_cases')}")

print("\n=== vocab composition of each Bm representation ===")
for m in (1, 2, 3):
    ddir = DATA / f"visit_Bm_m{m}_trackr"
    man = json.loads((ddir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    vocab = man.get("vocab") or man.get("token_ids") or {}
    if isinstance(vocab, dict):
        keys = list(vocab)
        pref = {}
        for k in keys:
            p = k.split(":")[0]
            pref[p] = pref.get(p, 0) + 1
        print(f"  visit_Bm_m{m}_trackr: manifest keys={list(man)[:8]}")
        print(f"     vocab prefixes: {pref}")
    files = sorted(p.name for p in ddir.glob("*.json"))
    print(f"     json files: {files[:10]}")

print("\n=== death tokens actually present? (scan labels file if any) ===")
for m in (1, 2, 3):
    ddir = DATA / f"visit_Bm_m{m}_trackr"
    for cand in ("labels.json", "token_labels.json", "vocab.json"):
        p = ddir / cand
        if p.exists():
            blob = json.loads(p.read_text())
            if isinstance(blob, list):
                deaths = [x for x in blob if "death" in str(x)]
            else:
                deaths = [k for k in blob if "death" in str(k)]
            print(f"  {ddir.name}/{cand}: {len(deaths)} death entries {deaths[:3]}")
