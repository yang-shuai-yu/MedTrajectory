"""Create a self-contained MIMIC Track-R v2.2 protocol for a given wavelength manifest."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = str(REPO_ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol, protocol_contract_sha256

ap = argparse.ArgumentParser()
ap.add_argument("--wavelength", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--output-root", default="")
args = ap.parse_args()

UKB = Path(ROOT) / "configs/track_r_v2_2/TRACK_R_v2_2.json"
proto = load_track_r_protocol(UKB)
proto.pop("extends", None)
if args.output_root:
    proto["output_root"] = args.output_root
proto["wavelength_contract"]["manifest"] = args.wavelength
proto["wavelength_contract"]["expected_sha256"] = hashlib.sha256(Path(args.wavelength).read_bytes()).hexdigest()
# the builder writes a manifest whose recorded sha256 is over its canonical payload, not the file bytes;
# prefer the manifest's own recorded sha256 when present
try:
    recorded = json.loads(Path(args.wavelength).read_text()).get("sha256")
    if isinstance(recorded, str) and len(recorded) == 64:
        proto["wavelength_contract"]["expected_sha256"] = recorded
except Exception:
    pass
proto["protocol_manifest_sha256"] = protocol_contract_sha256(proto)
Path(args.out).write_text(json.dumps(proto, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"out": args.out, "wavelength_sha256": proto["wavelength_contract"]["expected_sha256"],
                  "protocol_sha256": proto["protocol_manifest_sha256"]}))
