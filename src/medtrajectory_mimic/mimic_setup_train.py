# -*- coding: utf-8 -*-
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import json
from pathlib import Path

D = Path(str(MIMIC_ROOT / "multitype"))

# 1) minimal diseases_yaml (1 dummy; risk head unused in generation pretraining)
yaml = """diseases:
  - id: "dummy"
    name: "dummy"
    category: ""
    icd10:
      - "A00"
"""
(D / 'mimic_diseases.yaml').write_text(yaml, encoding='utf-8')

# 2) prepare_manifest.json: add vocab_csv (for load_token_codes)
mf = json.loads((D / 'prepare_manifest.json').read_text(encoding='utf-8'))
mf['vocab_csv'] = 'vocab/dynamic_token_vocab.csv'
(D / 'prepare_manifest.json').write_text(json.dumps(mf, indent=2), encoding='utf-8')

print('manifest:', json.dumps(mf, indent=2))
print('diseases_yaml written')
