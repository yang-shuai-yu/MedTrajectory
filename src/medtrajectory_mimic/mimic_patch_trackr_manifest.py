try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import json, sys
ROOT = str(REPO_ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/scripts")
from semantic_delphi_ukb.track_r_contract import load_track_r_protocol

PROTO = str(MIMIC_ROOT / "mimic_track_r_v2_2.json")
MANIFEST = str(MIMIC_ROOT / "multitype_trackr" / "prepare_manifest.json")

p = load_track_r_protocol(PROTO)
m = json.load(open(MANIFEST, encoding="utf-8"))
m["protocol_id"] = p["data_protocol_id"]  # track_r_v2_1
m["loss_contract"] = p["loss_contract"]
m["static_prefix"] = p["static_prefix"]
m["dynamic_bos"] = p["dynamic_bos"]
json.dump(m, open(MANIFEST, "w", encoding="utf-8"), indent=2)
print("patched manifest: protocol_id", m["protocol_id"], "static_len", len(m["static_prefix"]["logical_order"]), "bos fields", len(m["dynamic_bos"]))
