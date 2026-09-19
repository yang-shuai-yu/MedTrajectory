try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import numpy as np, sys
from pathlib import Path

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils import get_p2i

BASE = str(MIMIC_ROOT)

for name in ("multitype_m1", "multitype_m2", "multitype"):
    d = Path(BASE) / name
    for split in ("train", "val", "test"):
        data = np.memmap(d / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
        p2i = get_p2i(data)
        # death token_id = 4151 -> stored 4150 (only M3); count stored tokens == 4150
        death_rows = int((data[:, 2] == 4150).sum())
        n_pat = len(p2i)
        # count patients whose stored tokens include 4150 (death)
        starts = p2i[:, 0]; lens = p2i[:, 1]
        has_death = 0
        for s, l in zip(starts, lens):
            seg = data[s:s+l, 2]
            if int((seg == 4150).any()):
                has_death += 1
        print(f"{name} {split}: patients={n_pat} events={len(data)} death_rows={death_rows} patients_with_death={has_death}")
