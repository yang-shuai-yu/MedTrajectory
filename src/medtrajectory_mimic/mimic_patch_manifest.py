try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
for d in ('multitype_m1', 'multitype_m2'):
    p = fstr(MIMIC_ROOT / "{d}" / "prepare_manifest.json")
    with open(p) as fh:
        m = json.load(fh)
    m['vocab_csv'] = 'vocab/dynamic_token_vocab.csv'
    with open(p, 'w') as fh:
        json.dump(m, fh, indent=2)
    print(d, '->', json.dumps(m))
