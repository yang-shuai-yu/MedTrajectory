try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from collections import Counter
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))
p = DATA / "eval_bm3_ukb" / "bm_m3_a0" / "patient_rows.jsonl"
dp = Counter()
ad = Counter()
n = 0
with p.open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        dp[round(float(r.get("death_probability", 0.0)), 3)] += 1
        ad[int(r.get("actual_death", 0))] += 1
        n += 1
print(f"bm_m3_a0  n={n}")
print(f"  actual_death counts      : {dict(ad)}")
print(f"  death_probability values : {dict(sorted(dp.items())[:10])}"
      + (" ..." if len(dp) > 10 else ""))
