# -*- coding: utf-8 -*-
"""MIMIC-IV coding-systems analysis: procedures (ICD-PCS vs CPT/HCPCS), diagnoses (ICD-9/10),
cancer codes, services (surgery), death sources."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import gzip, csv, re
from collections import Counter, defaultdict

BASE = str(MIMIC_ROOT / "mimic-iv-3.1" / "hosp")

def headcols(p):
    with gzip.open(BASE + '/' + p, 'rt') as f:
        return next(csv.reader(f))

def sample(p, k=3):
    with gzip.open(BASE + '/' + p, 'rt') as f:
        r = csv.reader(f); h = next(r); rows = []
        for i, row in enumerate(r):
            if i >= k: break
            rows.append(row)
        return h, rows

def count(p):
    n = 0
    with gzip.open(BASE + '/' + p, 'rt') as f:
        for _ in f: n += 1
    return max(0, n - 1)

print('==== 1. procedures_icd (ICD procedure codes) ====')
h, rows = sample('procedures_icd.csv.gz', 3)
print('  cols:', h)
for r in rows: print('  sample:', r)
# classify code format
icd9 = icd10 = 0
with gzip.open(BASE + '/procedures_icd.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        v = row['icd_version']; c = row['icd_code']
        if v == '9': icd9 += 1
        else: icd10 += 1
print('  ICD9-PCS codes:', icd9, ' ICD10-PCS codes:', icd10)

print('\n==== 2. d_icd_procedures (dictionary) ====')
h, rows = sample('d_icd_procedures.csv.gz', 5)
print('  cols:', h)
for r in rows: print('  sample:', r)

print('\n==== 3. hcpcsevents (CPT/HCPCS codes) ====')
h, rows = sample('hcpcsevents.csv.gz', 3)
print('  cols:', h)
for r in rows: print('  sample:', r)
n = count('hcpcsevents.csv.gz')
print('  hcpcsevents rows:', n)
# unique hcpcs codes + category
codes = Counter()
with gzip.open(BASE + '/hcpcsevents.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        codes[row['hcpcs_cd']] += 1
print('  unique hcpcs codes:', len(codes))

print('\n==== 4. d_hcpcs (HCPCS dictionary) ====')
h, rows = sample('d_hcpcs.csv.gz', 5)
print('  cols:', h)
for r in rows: print('  sample:', r)
n = count('d_hcpcs.csv.gz')
print('  d_hcpcs rows:', n)

print('\n==== 5. services (curr_service: MED/SURG) ====')
h, rows = sample('services.csv.gz', 3)
print('  cols:', h)
svc = Counter()
with gzip.open(BASE + '/services.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        svc[row.get('curr_service','')] += 1
print('  curr_service distribution:', dict(svc))

print('\n==== 6. diagnoses_icd sample + cancer (C-codes / neoplasm) ====')
h, rows = sample('diagnoses_icd.csv.gz', 3)
print('  cols:', h)
for r in rows: print('  sample:', r)
# cancer: ICD10 C00-D49, ICD9 140-239
cancer10 = cancer9 = 0
with gzip.open(BASE + '/diagnoses_icd.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        c = row['icd_code']; v = row['icd_version']
        if v == '10' and re.match(r'^C\d', c):
            cancer10 += 1
        elif v == '9':
            cnum = c.split('.')[0]
            if cnum.isdigit() and 140 <= int(cnum) <= 239:
                cancer9 += 1
print('  cancer-ish diagnosis rows: ICD10 C=%d, ICD9 140-239=%d' % (cancer10, cancer9))

print('\n==== 7. death sources ====')
with gzip.open(BASE + '/patients.csv.gz', 'rt') as f:
    r = csv.DictReader(f); np_=0; nd=0
    for row in r:
        np_ += 1
        if row.get('dod',''): nd += 1
print('  patients.dod (overall death):', nd, '/', np_)
with gzip.open(BASE + '/admissions.csv.gz', 'rt') as f:
    r = csv.DictReader(f); na=0; ndt=0; nexp=0
    for row in r:
        na += 1
        if row.get('deathtime',''): ndt += 1
        if row.get('hospital_expire_flag','') == '1': nexp += 1
print('  admissions.deathtime:', ndt, '/', na, ' hospital_expire_flag=1:', nexp)
