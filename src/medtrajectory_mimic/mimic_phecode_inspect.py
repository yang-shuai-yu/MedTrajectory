# -*- coding: utf-8 -*-
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import pyreadr

base = str(MIMIC_ROOT / "phewas_repo" / "data")

for name in ['phecode_map.rda', 'phecode_map_icd10.rda', 'pheinfo.rda']:
    print('==== %s ====' % name)
    try:
        res = pyreadr.read_r(base + '/' + name)
        for k, df in res.items():
            print('  key=%s shape=%s' % (k, df.shape))
            print('  columns:', list(df.columns))
            print(df.head(3).to_string())
    except Exception as e:
        print('  ERROR', e)
