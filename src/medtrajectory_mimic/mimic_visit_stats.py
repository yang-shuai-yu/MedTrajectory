try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import numpy as np, sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils import get_p2i

for name in ("visit_B_m1", "visit_B_m2", "visit_B_m3"):
    d = fstr(MIMIC_ROOT / "{name}")
    data = np.memmap(f"{d}/test.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    lens = p2i[:, 1]
    n = len(lens)
    # count distinct ages (visits) per patient
    visits = []
    for s, l in zip(p2i[:, 0], p2i[:, 1]):
        ages = np.asarray(data[s:s+l, 1])
        visits.append(len(np.unique(ages)))
    visits = np.asarray(visits)
    print(f"{name} test: patients={n} tokens/pat: median={np.median(lens):.0f} mean={lens.mean():.1f} | "
          f"distinct-visits/pat: median={np.median(visits):.0f} mean={visits.mean():.1f} max={visits.max()}")
    for th in (4, 5, 6, 8, 11):
        print(f"   tokens>={th}: {(lens >= th).sum()} ({100*(lens>=th).mean():.1f}%)   visits>={th}: {(visits >= th).sum()} ({100*(visits>=th).mean():.1f}%)")
