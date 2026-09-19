"""Create a self-contained MIMIC Track-R v2.2 protocol by overriding the wavelength contract."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import json, sys
from pathlib import Path

ROOT = str(REPO_ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol, protocol_contract_sha256

UKB = Path(ROOT) / "configs/track_r_v2_2/TRACK_R_v2_2.json"
MIMIC_WAVELENGTH_SHA = "45cfb2d848c785184a864b70f597d05d1d436def5e19d7ff1b00e60051d360d3"
MIMIC_WAVELENGTH_PATH = str(MIMIC_ROOT / "mimic_rope_wavelengths.json")
OUT = str(MIMIC_ROOT / "mimic_track_r_v2_2.json")

proto = load_track_r_protocol(UKB)
proto.pop("extends", None)
proto["output_root"] = str(MIMIC_ROOT / "results" / "track_r_v2_2")
proto["wavelength_contract"]["manifest"] = MIMIC_WAVELENGTH_PATH
proto["wavelength_contract"]["expected_sha256"] = MIMIC_WAVELENGTH_SHA
proto["protocol_manifest_sha256"] = protocol_contract_sha256(proto)

Path(OUT).write_text(json.dumps(proto, indent=2, sort_keys=True), encoding="utf-8")
print("wrote", OUT)
print("protocol_manifest_sha256", proto["protocol_manifest_sha256"])
print("wavelength expected", proto["wavelength_contract"]["expected_sha256"][:16])
