# -*- coding: utf-8 -*-
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import gzip, csv, re
MIMIC = str(MIMIC_ROOT / "mimic-iv-3.1" / "hosp")

c10_mal = c10_in = c10_ben = 0
c9_neop = 0
with gzip.open(MIMIC + '/diagnoses_icd.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        c = row['icd_code']; v = row['icd_version']
        if v == '10':
            if re.match(r'^C', c): c10_mal += 1
            elif re.match(r'^D0[0-4]', c): c10_in += 1   # D00-D09 in situ (approx D00-D49)
            elif re.match(r'^D[1-4]', c): c10_ben += 1
        else:
            first3 = c[:3]
            if first3.isdigit() and 140 <= int(first3) <= 239:
                c9_neop += 1
print('ICD-10 malignant (C) :', c10_mal)
print('ICD-10 in-situ/benign (D00-D49 approx):', c10_in + c10_ben)
print('ICD-9 neoplasm (140-239):', c9_neop)
tot = c10_mal + c10_in + c10_ben + c9_neop
print('TOTAL neoplasm diagnosis rows:', tot)
