try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import sys
ROOT = str(REPO_ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol
p = load_track_r_protocol(str(MIMIC_ROOT / "mimic_track_r_v2_2.json"))
print("PROTOCOL_OK", p["protocol_id"], p["wavelength_contract"]["expected_sha256"][:16], "static_len", p["static_prefix"]["fixed_length"])
