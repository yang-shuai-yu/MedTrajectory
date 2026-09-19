try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import numpy as np, sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils import get_p2i

for name, path in (("MIMIC M3 train", str(MIMIC_ROOT / "multitype" / "train.bin")),
                   ("MIMIC M3 test", str(MIMIC_ROOT / "multitype" / "test.bin"))):
    data = np.memmap(path, dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    zero = pos = 0
    for s, l in zip(p2i[:, 0], p2i[:, 1]):
        ages = np.asarray(data[s:s+l, 1], dtype=np.int64)
        ages = np.sort(ages)
        if len(ages) < 2:
            continue
        d = np.diff(ages)
        zero += int((d == 0).sum())
        pos += int((d > 0).sum())
    tot = zero + pos
    print(f"{name}: events={len(data)} patients={len(p2i)} consecutive_pairs={tot} zero_gap={zero} ({100*zero/tot:.1f}%) positive_gap={pos} ({100*pos/tot:.1f}%)")
