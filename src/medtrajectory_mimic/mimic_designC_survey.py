# -*- coding: utf-8 -*-
"""Design C investigation: unique admission code-set counts and frequency thresholds.

For each admission build the ordered code list (diagnoses by seq_num, then procedures by seq_num,
each mapped to Phecode / CPT). Measure:
  * code-set size distribution
  * number of distinct full code sets
  * for N = 1..6: number of distinct prefixes of the first N codes, and how many survive
    various minimum-frequency thresholds
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import csv, gzip
from collections import Counter, defaultdict

HOSP = str(MIMIC_ROOT / "mimic-iv-3.1" / "hosp")
PHEWAS = str(MIMIC_ROOT / "phewas_repo" / "data")
ICD10CSV = str(REPO_ROOT / "outputs" / "icd_embedding_compare" / "icd10_ontology" / "phecode_icd10cm_map.csv")
import pyreadr


def canon(c):
    return (c or '').replace('.', '').strip().upper()


icd9 = {}
res = pyreadr.read_r(PHEWAS + '/phecode_map.rda')
for _, r in res['phecode_map'].iterrows():
    if str(r['vocabulary_id']).startswith('ICD9'):
        icd9[canon(str(r['code']))] = str(r['phecode'])
icd10 = {}
with open(ICD10CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        icd10[canon(row['ICD_id'])] = row['Phecode']

dx = defaultdict(list)
with gzip.open(HOSP + '/diagnoses_icd.csv.gz', 'rt') as f:
    for row in csv.DictReader(f):
        pc = icd9.get(canon(row['icd_code'])) if row['icd_version'] == '9' else icd10.get(canon(row['icd_code']))
        if pc:
            try:
                seq = int(row['seq_num'])
            except ValueError:
                seq = 999
            dx[row['hadm_id']].append((seq, 'phecode:' + pc))
px = defaultdict(list)
with gzip.open(HOSP + '/hcpcsevents.csv.gz', 'rt') as f:
    for row in csv.DictReader(f):
        code = (row.get('hcpcs_cd') or '').strip()
        if code:
            try:
                seq = int(row.get('seq_num') or 999)
            except ValueError:
                seq = 999
            px[row['hadm_id']].append((seq, 'cpt:' + code))

sets = []
sizes = Counter()
for hadm in set(dx) | set(px):
    codes = [c for _, c in sorted(dx.get(hadm, []))] + [c for _, c in sorted(px.get(hadm, []))]
    if not codes:
        continue
    sizes[len(codes)] += 1
    sets.append(tuple(codes))

print('admissions with >=1 mapped code:', len(sets))
print('code-set size distribution (size: admissions):', dict(sorted(sizes.items())[:15]))
full = Counter(sets)
print('distinct FULL code sets:', len(full), ' /  admissions:', len(sets))
print()
print('N | distinct prefixes | >=2 | >=5 | >=10 | >=20 | >=50 | >=100')
for N in range(1, 7):
    pre = Counter(tuple(s[:N]) for s in sets)
    row = [f'{N}', f'{len(pre)}']
    for th in (2, 5, 10, 20, 50, 100):
        row.append(str(sum(1 for v in pre.values() if v >= th)))
    print(' | '.join(row))
print()
# coverage: what fraction of admissions is covered by the retained vocabulary at N and threshold
print('coverage of admissions by retained prefix vocab:')
print('N | th | vocab | coverage')
for N in (1, 2, 3, 4):
    pre = Counter(tuple(s[:N]) for s in sets)
    for th in (2, 3, 5, 10, 20, 50, 100, 200):
        kept = {k for k, v in pre.items() if v >= th}
        cov = sum(v for k, v in pre.items() if k in kept)
        print(f'{N} | {th} | {len(kept)} | {100*cov/len(sets):.1f}%')
