"""Inspect the Bm vocabularies and the death-token situation."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import csv
import json
from collections import Counter
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))

for m in (1, 2, 3):
    ddir = DATA / f"visit_Bm_m{m}_trackr"
    man = json.loads((ddir / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
    vcsv = man.get("vocab_csv")
    print(f"=== visit_Bm_m{m}_trackr ===")
    print(f"   vocab_size={man.get('vocab_size')} vocab_csv={vcsv} "
          f"matched_universe={man.get('matched_universe')}")
    p = Path(vcsv) if vcsv and Path(vcsv).is_absolute() else (ddir / str(vcsv) if vcsv else None)
    if p is None or not p.exists():
        cand = list(DATA.glob(f"visit_Bm_m{m}*/**/*vocab*.csv")) + list(DATA.glob(f"**/visit_Bm_m{m}*vocab*.csv"))
        print(f"   vocab csv not found at {p}; candidates={cand[:3]}")
        continue
    with p.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"   {len(rows)} rows, columns={list(rows[0]) if rows else None}")
    tok_col = "token" if rows and "token" in rows[0] else list(rows[0])[0]
    pref = Counter(str(r[tok_col]).split(":")[0] for r in rows)
    print(f"   token prefixes: {dict(pref)}")
    deaths = [r[tok_col] for r in rows if str(r[tok_col]).startswith("death")]
    print(f"   death tokens: {len(deaths)} {deaths[:5]}")
