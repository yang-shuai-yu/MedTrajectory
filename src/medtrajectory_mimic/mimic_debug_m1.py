try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import json, csv, sys, math
from pathlib import Path
import numpy as np, torch

REPO = str(REPO_ROOT)
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/src"); sys.path.insert(0, REPO + "/scripts")
from semantic_delphi_ukb.track_g_models import checkpoint_state
from semantic_delphi_ukb.track_g_generation import sample_event_and_wait_with_diagnostics, frozen_log_rate
from utils import get_p2i

CKPT = sys.argv[1]
DATA = sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "?"

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
        event_types[t] = r["event_type"].strip(); keys[t] = r["token_key"].strip()

data = np.memmap(Path(DATA)/"test.bin", dtype=np.uint32, mode="r").reshape(-1,3)
p2i = get_p2i(data)
static = np.load(Path(DATA)/"test_static.npy").astype(np.float32)
candidate_mask = torch.tensor([et in {"diagnosis","procedure","death"} for et in event_types], dtype=torch.bool, device="cuda")
death_mask = torch.tensor([et == "death" for et in event_types], dtype=torch.bool, device="cuda")

n_gen0 = 0; n_total = 0; waits = []
for pi in range(300):
    start, length = p2i[pi]
    if length < 11:
        continue
    rows = np.asarray(data[start:start+length])
    order = np.argsort(rows[:,1], kind="stable")
    rows = rows[order]
    toks = [int(r[2])+1 for r in rows]; ages = [float(r[1]) for r in rows]
    cut = max(7, int(math.floor((len(toks)-1)*0.65)))
    cut = min(cut, len(toks)-3-1)
    hist_t = toks[:cut+1]; hist_a = ages[:cut+1]
    baseline = hist_a[-1]
    t = torch.tensor([hist_t], dtype=torch.long, device="cuda")
    a = torch.tensor([hist_a], dtype=torch.float32, device="cuda")
    s = torch.tensor(static[[pi]], dtype=torch.float32, device="cuda")
    # first-step logits and rate
    lg, *_ = model(t, a, s, None, None)
    ll = lg[0,-1][:vocab_size]
    lr = frozen_log_rate(ll, model.config.ignore_tokens, model.config.t_min)
    rate = torch.exp(lr).item()
    waits.append(1.0/rate if rate > 0 else 1e9)
    # rollout
    gen = []
    tt = t.clone(); aa = a.clone(); ss = s.clone()
    generator = torch.Generator(device="cuda").manual_seed(20260819 + 1000003*pi)
    for step in range(30):
        lgn, *_ = model(tt, aa, ss, None, None)
        lgn = lgn[0,-1][:vocab_size]
        ue = torch.rand(1, generator=generator, device="cuda")
        uw = torch.rand(1, generator=generator, device="cuda")
        tok, wait, clamped = sample_event_and_wait_with_diagnostics(
            lgn, candidate_mask=candidate_mask, ignore_tokens=model.config.ignore_tokens,
            t_min=model.config.t_min, temperature=0.8, top_p=0.9,
            death_token_mask=death_mask, death_logit_bias=0.0, minimum_wait_days=1.0,
            event_uniform=ue[0], wait_uniform=uw[0])
        na = float(aa[0,-1]) + wait
        if na > baseline + 10*365.25:
            break
        gen.append(tok)
        tt = torch.cat([tt, torch.tensor([[tok]], device="cuda")], dim=1)
        aa = torch.cat([aa, torch.tensor([[na]], device="cuda")], dim=1)
        if event_types[tok] == "death":
            break
    n_total += 1
    if len(gen) == 0:
        n_gen0 += 1
    if pi < 5 and length >= 11:
        print(f"pi={pi} events={length} cut={cut} rate={rate:.4f} mean_wait_days={1.0/rate if rate>0 else -1:.1f} first_wait sample={waits[-1]:.1f}d gen={len(gen)}")

print(f"\n[{MODEL}] n={n_total} empty_rollouts={n_gen0} ({100*n_gen0/max(n_total,1):.1f}%) median_rate={np.median(waits):.4f} median_implied_wait_days={np.median([1/w if w>0 else 1e9 for w in waits]):.1f}")
