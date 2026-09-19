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
    csv_path = DATA / str(man["vocab_csv"]) if not Path(str(man["vocab_csv"])).is_absolute() else Path(str(man["vocab_csv"]))
    if not csv_path.exists():
        csv_path = ddir / "vocab" / "dynamic_token_vocab.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    et = Counter(r["event_type"] for r in rows)
    keys = Counter(r["token_key"].split(":")[0] for r in rows)
    print(f"visit_Bm_m{m}_trackr  vocab_size={man.get('vocab_size')} rows={len(rows)}")
    print(f"   event_type : {dict(et)}")
    print(f"   key prefix : {dict(keys)}")
