# -*- coding: utf-8 -*-
"""One-shot Track R v2.2 locked-test authorization.

1. Amends track_r_contract.py so test access is allowed ONLY with a valid
   `locked_test_authorization` record (previously it always rejected test flags).
2. Creates a standalone TEST protocol (configs/track_r_v2_2/TRACK_R_v2_2_TEST.json)
   with test_authorized=true, the frozen checkpoint/landmark/assessment hashes,
   and a freshly computed protocol_manifest_sha256.
3. Self-verifies the new protocol loads under the amended contract.

Run on server: <python interpreter, e.g. the project virtualenv>
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import REPO_ROOT
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(str(REPO_ROOT))
sys.path.insert(0, str(ROOT / "src"))

CONTRACT = ROOT / "src/semantic_delphi_ukb/track_r_contract.py"
BASE_PROTO = ROOT / "configs/track_r_v2_2/TRACK_R_v2_2.json"
TEST_PROTO = ROOT / "configs/track_r_v2_2/TRACK_R_v2_2_TEST.json"

OLD = (
    '        if protocol.get("locked_test_read") is not False or protocol.get("test_authorized") is not False:\n'
    '            raise ValueError("Track R v2.2 must keep locked test access disabled")'
)
NEW = (
    '        locked_test_read = protocol.get("locked_test_read")\n'
    '        test_authorized = protocol.get("test_authorized")\n'
    '        if locked_test_read or test_authorized:\n'
    '            if locked_test_read != test_authorized:\n'
    '                raise ValueError("Track R v2.2 locked-test flags must be enabled together")\n'
    '            authorization = protocol.get("locked_test_authorization")\n'
    '            if not isinstance(authorization, Mapping):\n'
    '                raise ValueError("Track R v2.2 test access requires a locked_test_authorization record")\n'
    '            if authorization.get("scope") != "one_shot_track_r_v2_2_locked_test":\n'
    '                raise ValueError("Track R v2.2 locked-test authorization scope is invalid")\n'
    '            if not isinstance(authorization.get("authorized_at"), str) or not authorization["authorized_at"]:\n'
    '                raise ValueError("Track R v2.2 locked-test authorization requires authorized_at")\n'
    '            for key in ("validation_protocol_sha256", "validation_assessment_sha256", "validation_landmarks_sha256"):\n'
    '                if not _is_sha256(authorization.get(key)):\n'
    '                    raise ValueError(f"Track R v2.2 locked-test authorization hash is invalid: {key}")\n'
    '            if not isinstance(authorization.get("checkpoints"), Mapping):\n'
    '                raise ValueError("Track R v2.2 locked-test authorization requires checkpoint hashes")'
)

CHECKPOINTS = {
    "seed42": {
        "A0": "7d2800a4dfa2cf76524a6df0d64b27a6b8bb1fcf1e0f91b00347faf5de53ca21",
        "A2": "6e3014eba678d563cfc8dbdd2ac5ce7f3b58e7beaae3c5e1505a176e8028f2b0",
        "A2-noAge": "37dcfc47847d5be14dd931988e75e6a242ac85cbf5fbbf3871d4b9353ffc6174",
    },
    "seed43": {
        "A0": "8c4ee441b068d3a1e9dd3d8899fa5fc2af0eab1a99be9247b2c22258dd31a59d",
        "A2": "30b9facda954a2dc2c05c57b4141c127cdf5b855db1350e52582925372d4177d",
        "A2-noAge": "48ad6903ea2216434b31dcc76fb4c2e4896b98e7f6e6183fea6fb5cd4d7dc089",
    },
    "seed44": {
        "A0": "4279c387a2f8f101ce338ddc5f384e16be0f72c33d66720a0ea4ad3a67010cc5",
        "A2": "84ed5fbf6967fe2997ef81c586288b8977295ec9d245d02f8c832fefe57fc3d7",
        "A2-noAge": "962e6ccd832257da6062bb2ad032b83d76c0e720c3116bebea4fffcbd9532fca",
    },
}


def main():
    # --- 1. amend contract ---
    text = CONTRACT.read_text(encoding="utf-8")
    assert text.count(OLD) == 1, "contract pattern not unique"
    text = text.replace(OLD, NEW)
    CONTRACT.write_text(text, encoding="utf-8")
    print("amended", CONTRACT)

    # --- 2. build standalone TEST protocol ---
    from semantic_delphi_ukb.track_r_contract import (
        load_track_r_protocol, protocol_contract_sha256,
    )
    base = load_track_r_protocol(BASE_PROTO)
    proto = copy.deepcopy(base)
    proto.pop("extends", None)
    proto["test_authorized"] = True
    proto["locked_test_read"] = True
    proto["locked_test_authorization"] = {
        "scope": "one_shot_track_r_v2_2_locked_test",
        "authorized_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validation_protocol_sha256": base["protocol_manifest_sha256"],
        "validation_assessment_sha256": "41a344a6666a0599b702fbae0bb0bfb1367a3fa30d2a104a27f660cb18789540",
        "validation_landmarks_sha256": "33e846f3e5a5d9ca3410648245b23cb2514d99bd706463daa4dfa17987a7e6d8",
        "checkpoints": CHECKPOINTS,
    }
    proto.pop("protocol_manifest_sha256", None)
    new_hash = protocol_contract_sha256(proto)
    proto["protocol_manifest_sha256"] = new_hash
    TEST_PROTO.write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", TEST_PROTO)
    print("new protocol_manifest_sha256 =", new_hash)

    # --- 3. self-verify ---
    loaded = load_track_r_protocol(TEST_PROTO)
    assert loaded["test_authorized"] is True and loaded["locked_test_read"] is True
    assert loaded["locked_test_authorization"]["checkpoints"]["seed42"]["A2"] == CHECKPOINTS["seed42"]["A2"]
    print("SELF_VERIFY_OK: TEST protocol loads under amended contract")


if __name__ == "__main__":
    main()
