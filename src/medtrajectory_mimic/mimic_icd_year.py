# -*- coding: utf-8 -*-
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import gzip, csv
from collections import Counter, defaultdict
BASE = str(MIMIC_ROOT / "mimic-iv-3.1")

# admission year range
mn = None; mx = None
with gzip.open(BASE + '/hosp/admissions.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        a = row['admittime'][:4]
        if mn is None or a < mn: mn = a
        if mx is None or a > mx: mx = a
print('admission year range:', mn, '-', mx)

# ICD version of diagnoses by admission year
adm_year = {}
with gzip.open(BASE + '/hosp/admissions.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        adm_year[row['hadm_id']] = row['admittime'][:4]
adm_ver = defaultdict(Counter)
with gzip.open(BASE + '/hosp/diagnoses_icd.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        y = adm_year.get(row['hadm_id'])
        if y:
            adm_ver[y][row['icd_version']] += 1
print('diagnosis ICD version by admission year:')
for y in sorted(adm_ver):
    c = adm_ver[y]
    tot = sum(c.values())
    print('  %s: ICD9=%d (%.0f%%) ICD10=%d (%.0f%%)' % (
        y, c['9'], 100*c['9']/tot, c['10'], 100*c['10']/tot))

# procedures ICD version by year
adm_ver2 = defaultdict(Counter)
with gzip.open(BASE + '/hosp/procedures_icd.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        y = adm_year.get(row['hadm_id'])
        if y:
            adm_ver2[y][row['icd_version']] += 1
print('procedure ICD version by admission year:')
for y in sorted(adm_ver2):
    c = adm_ver2[y]
    tot = sum(c.values())
    print('  %s: ICD9=%d (%.0f%%) ICD10=%d (%.0f%%)' % (
        y, c['9'], 100*c['9']/tot, c['10'], 100*c['10']/tot))
