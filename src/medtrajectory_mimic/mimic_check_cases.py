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

for name in ("multitype_m1", "multitype_m2", "multitype"):
    d = Path(str(MIMIC_ROOT)) / name
    data = np.memmap(d / "test.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    lens = p2i[:, 1]
    n = len(lens)
    ge11 = int((lens >= 11).sum())
    ge16 = int((lens >= 16).sum())
    print(f"{name} test: n={n} median_events={np.median(lens):.0f} mean={lens.mean():.1f} >=11={ge11} ({100*ge11/n:.1f}%) >=16={ge16} ({100*ge16/n:.1f}%)")
