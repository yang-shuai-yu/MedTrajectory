try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

DATA = Path(str(MIMIC_ROOT))
for p in [DATA / "eval_ethos_ukb" / "ethos_matched_a" / "summary.json",
          DATA / "eval_ethos_relaxed" / "ethos_matched_a" / "summary.json",
          DATA / "eval_visitA_seeds_ukb" / "vA_a2" / "summary.json",
          DATA / "eval_visitA_seeds_relaxed" / "vA_a2" / "summary.json"]:
    if not p.exists():
        print(f"{p}: missing")
        continue
    b = json.loads(p.read_text())
    m = b.get("metrics", {})
    print(f"{p.parent.parent.name}/{p.parent.name}:")
    print(f"   data_dir={b.get('data_dir')}")
    print(f"   num_cases={b.get('num_cases')} eid_file={b.get('eid_file')}")
    print(f"   jaccard={m.get('diagnosis_jaccard')} hit@10={m.get('hit_at_10')} "
          f"time_mae={m.get('first_event_time_mae_days')} count_mae={m.get('event_count_mae')}")
