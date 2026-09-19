try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import json, csv, sys, math
from pathlib import Path
import numpy as np, torch

REPO = str(REPO_ROOT)
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/src"); sys.path.insert(0, REPO + "/scripts")
from semantic_delphi_ukb.track_g_models import checkpoint_state
from semantic_delphi_ukb.track_g_generation import sample_event_and_wait_with_diagnostics
from utils import get_p2i

CKPT = str(MIMIC_ROOT / "runs" / "m3_abs" / "checkpoints" / "last.pt")
DATA = str(MIMIC_ROOT / "multitype")

ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
model, family = checkpoint_state(ckpt, expected_family="carope")
model = model.cuda().eval()
print("vocab_size", model.config.vocab_size, "block", model.config.block_size, "t_min", model.config.t_min, "ignore", model.config.ignore_tokens)

manifest = json.loads((Path(DATA)/"prepare_manifest.json").read_text())
vocab_csv = Path(DATA)/manifest["vocab_csv"]
vocab_size = manifest["vocab_size"]
event_types = ["special"]*vocab_size
keys = ["special"]*vocab_size
for r in csv.DictReader(vocab_csv.open()):
    t = int(r["token_id"])
    if 0 <= t < vocab_size:
        event_types[t] = r["event_type"].strip(); keys[t] = r["token_key"].strip()

data = np.memmap(Path(DATA)/"test.bin", dtype=np.uint32, mode="r").reshape(-1,3)
p2i = get_p2i(data)
static = np.load(Path(DATA)/"test_static.npy").astype(np.float32)

candidate_mask = torch.tensor([et in {"diagnosis","procedure","death"} for et in event_types], dtype=torch.bool, device="cuda")
death_mask = torch.tensor([et == "death" for et in event_types], dtype=torch.bool, device="cuda")

# pick a patient with >=11 events
for pi in range(200):
    start, length = p2i[pi]
    if length < 11:
        continue
    rows = np.asarray(data[start:start+length])
    order = np.argsort(rows[:,1], kind="stable")
    rows = rows[order]
    toks = [int(r[2])+1 for r in rows]
    ages = [float(r[1]) for r in rows]
    cut = max(7, int(math.floor((len(toks)-1)*0.65)))
    cut = min(cut, len(toks)-3-1)
    hist_t = toks[:cut+1]; hist_a = ages[:cut+1]
    actual = list(zip(toks[cut+1:], ages[cut+1:]))
    baseline = hist_a[-1]
    actual = [(t,a) for t,a in actual if a <= baseline + 10*365.25]

    t = torch.tensor([hist_t], dtype=torch.long, device="cuda")
    a = torch.tensor([hist_a], dtype=torch.float32, device="cuda")
    s = torch.tensor(static[[pi]], dtype=torch.float32, device="cuda")

    # single-step logits
    logits, *_ = model(t, a, s, None, None)
    ll = logits[0,-1]
    top5 = torch.argsort(ll, descending=True)[:5].tolist()
    nxt = toks[cut+1]
    print(f"\n=== patient {pi}: {len(toks)} events, cut={cut}, baseline_age={baseline:.0f}d ===")
    print("actual next token:", nxt, keys[nxt], event_types[nxt], f"age={ages[cut+1]:.0f}")
    print("single-step top5:", [(x, keys[x], event_types[x]) for x in top5])
    print("actual future (%d):" % len(actual), [(keys[t], round(a)) for t,a in actual[:8]])

    # full rollout (20 rollouts, max 30 tokens)
    generator = torch.Generator(device="cuda").manual_seed(20260819 + 1000003*pi)
    for ri in range(3):
        gen = []
        tt = t.clone(); aa = a.clone(); ss = s.clone()
        for step in range(30):
            lg, *_ = model(tt, aa, ss, None, None)
            lg = lg[0,-1][:vocab_size]
            u = torch.rand(1, generator=generator, device="cuda")
            ue = torch.rand(1, generator=generator, device="cuda")
            tok, wait, clamped = sample_event_and_wait_with_diagnostics(
                lg, candidate_mask=candidate_mask, ignore_tokens=model.config.ignore_tokens,
                t_min=model.config.t_min, temperature=0.8, top_p=0.9,
                death_token_mask=death_mask, death_logit_bias=0.0, minimum_wait_days=1.0,
                event_uniform=ue[0], wait_uniform=u[0])
            na = float(aa[0,-1]) + wait
            if na > baseline + 10*365.25:
                break
            gen.append((tok, na, event_types[tok], keys[tok]))
            tt = torch.cat([tt, torch.tensor([[tok]], device="cuda")], dim=1)
            aa = torch.cat([aa, torch.tensor([[na]], device="cuda")], dim=1)
            if event_types[tok] == "death":
                break
        print(f"rollout {ri}: {len(gen)} events", [(k, round(a)) for _,a,_,k in gen[:8]])
    if pi > 60:
        break
