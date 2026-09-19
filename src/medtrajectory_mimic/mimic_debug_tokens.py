try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import json, csv, sys
from pathlib import Path
import numpy as np, torch

REPO = str(REPO_ROOT)
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/src"); sys.path.insert(0, REPO + "/scripts")
from semantic_delphi_ukb.track_g_models import checkpoint_state
from utils import get_p2i

CKPT = str(MIMIC_ROOT / "runs" / "smoke_abs" / "checkpoints" / "last.pt")
DATA = str(MIMIC_ROOT / "multitype")

ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
model, family = checkpoint_state(ckpt, expected_family="carope")
model = model.cuda().eval()

manifest = json.loads((Path(DATA)/"prepare_manifest.json").read_text())
vocab_csv = Path(DATA)/manifest["vocab_csv"]
vocab_size = manifest["vocab_size"]
event_types = ["special"]*vocab_size
keys = ["special"]*vocab_size
for r in csv.DictReader(vocab_csv.open()):
    t = int(r["token_id"])
    if 0 <= t < vocab_size:
        event_types[t] = r["event_type"].strip()
        keys[t] = r["token_key"].strip()

data = np.memmap(Path(DATA)/"test.bin", dtype=np.uint32, mode="r").reshape(-1,3)
p2i = get_p2i(data)
static = np.load(Path(DATA)/"test_static.npy").astype(np.float32)

for pi in range(5):
    start, length = p2i[pi]
    rows = np.asarray(data[start:start+length])
    order = np.argsort(rows[:,1], kind="stable")
    rows = rows[order]
    toks = [int(r[2])+1 for r in rows]
    ages = [float(r[1]) for r in rows]
    if len(toks) < 3:
        continue
    cut = len(toks)//2
    hist_t = toks[:cut+1]; hist_a = ages[:cut+1]
    nxt = toks[cut+1]
    t = torch.tensor([hist_t], dtype=torch.long, device="cuda")
    a = torch.tensor([hist_a], dtype=torch.float32, device="cuda")
    s = torch.tensor(static[[pi]], dtype=torch.float32, device="cuda")
    logits, *_ = model(t, a, s, None, None)
    l = logits[0,-1]
    top = torch.argsort(l, descending=True)[:3].tolist()
    print(f"pi={pi} n_events={len(toks)} actual_next={nxt} ({event_types[nxt]}, {keys[nxt]}) top3={[(x,event_types[x],keys[x]) for x in top]}")
