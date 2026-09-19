"""Does the death token appear anywhere in the M3 training/validation data?

The draft claimed the death token "is never observed in training targets and never
generated".  Only the evaluation cohort's actual_death was verified (=0 for all 541
participants).  That does NOT establish anything about the training targets, so this
checks the raw token streams directly.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import csv
import json
from pathlib import Path

import numpy as np

DATA = Path(str(MIMIC_ROOT))
DDIR = DATA / "visit_Bm_m3_trackr"

man = json.loads((DDIR / "prepare_manifest.json").read_text(encoding="utf-8-sig"))
vcsv = DDIR / str(man["vocab_csv"])
rows = list(csv.DictReader(vcsv.open(encoding="utf-8", newline="")))
death_ids = [int(r["token_id"]) for r in rows if r.get("event_type") == "death"]
print("vocab_csv:", vcsv)
print("death token ids in vocab:", death_ids)

# the .bin stores a 1-based token index (+1 applied at build time); check both
for split in ("train", "val", "test"):
    f = DDIR / f"{split}.bin"
    if not f.exists():
        print(f"{split}: missing")
        continue
    arr = np.memmap(f, dtype=np.uint32, mode="r").reshape(-1, 3)
    tok = arr[:, 2]
    counts = {d: int((tok == d).sum()) for d in death_ids}
    counts_shift = {d + 1: int((tok == d + 1).sum()) for d in death_ids}
    # also look at the max token id actually present, to infer the encoding offset
    print(f"{split}: rows={arr.shape[0]} distinct_tokens={np.unique(tok).size} "
          f"token_min={tok.min()} token_max={tok.max()}")
    print(f"    exact death-id counts      : {counts}")
    print(f"    death-id+1 counts          : {counts_shift}")

print("\n=== same question for the event-level multitype vocabulary ===")
for ddir_name in ("multitype", "visit_B_m3_trackr", "visit_A_trackr", "visit_C_trackr"):
    d = DATA / ddir_name
    mp = d / "prepare_manifest.json"
    if not mp.exists():
        continue
    m = json.loads(mp.read_text(encoding="utf-8-sig"))
    vc = d / str(m["vocab_csv"])
    if not vc.exists():
        print(f"  {ddir_name}: no vocab csv")
        continue
    rr = list(csv.DictReader(vc.open(encoding="utf-8", newline="")))
    dids = [int(r["token_id"]) for r in rr if r.get("event_type") == "death"]
    for split in ("train", "test"):
        f = d / f"{split}.bin"
        if not f.exists():
            continue
        a = np.memmap(f, dtype=np.uint32, mode="r").reshape(-1, 3)
        t = a[:, 2]
        n = sum(int((t == x).sum()) for x in dids) + sum(int((t == x + 1).sum()) for x in dids)
        print(f"  {ddir_name:20s} {split:5s} death_ids={dids} occurrences={n}")
