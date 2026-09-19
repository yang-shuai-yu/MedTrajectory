# -*- coding: utf-8 -*-
"""Phecode coverage of MIMIC-IV diagnoses: what fraction of ICD-9 and ICD-10
diagnosis rows map to a Phecode (PheWAS v1.2), unifying the two coding systems."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import gzip, csv, re
from collections import Counter, defaultdict
import pyreadr

MIMIC = str(MIMIC_ROOT / "mimic-iv-3.1" / "hosp")
PHEWAS = str(MIMIC_ROOT / "phewas_repo" / "data")
ICD10CSV = str(REPO_ROOT / "outputs" / "icd_embedding_compare" / "icd10_ontology" / "phecode_icd10cm_map.csv")

def canon(code):
    """Strip dot -> canonical no-decimal form."""
    return (code or '').replace('.', '').strip().upper()

# 1) load ICD-9 map from RData
icd9_map = {}
res = pyreadr.read_r(PHEWAS + '/phecode_map.rda')
df9 = res['phecode_map']
for _, row in df9.iterrows():
    if str(row['vocabulary_id']).startswith('ICD9'):
        icd9_map[canon(str(row['code']))] = str(row['phecode'])
print('ICD-9 map entries:', len(icd9_map), '-> unique phecodes:', len(set(icd9_map.values())))

# 2) load ICD-10-CM map from CSV (already PheWAS v1.2)
icd10_map = {}
with open(ICD10CSV, 'r', newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        icd10_map[canon(row['ICD_id'])] = row['Phecode']
print('ICD-10-CM map entries:', len(icd10_map), '-> unique phecodes:', len(set(icd10_map.values())))

# 3) MIMIC diagnosis code frequencies
freq9 = Counter(); freq10 = Counter()
with gzip.open(MIMIC + '/diagnoses_icd.csv.gz', 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        v = row['icd_version']; c = canon(row['icd_code'])
        if v == '9':
            freq9[c] += 1
        else:
            freq10[c] += 1
tot9 = sum(freq9.values()); tot10 = sum(freq10.values())
print('\nMIMIC diagnosis rows: ICD9=%d ICD10=%d total=%d' % (tot9, tot10, tot9 + tot10))
print('MIMIC unique codes: ICD9=%d ICD10=%d' % (len(freq9), len(freq10)))

# 4) coverage (weighted by frequency)
def coverage(freq, mp, label):
    tot = sum(freq.values())
    mapped_unique = sum(1 for c in freq if c in mp)
    mapped_rows = sum(n for c, n in freq.items() if c in mp)
    unmapped_rows = Counter({c: n for c, n in freq.items() if c not in mp})
    print('\n[%s]' % label)
    print('  unique codes: %d (%.1f%% mapped)' % (len(freq), 100 * mapped_unique / len(freq)))
    print('  diagnosis rows: %d (%.2f%% mapped)' % (tot, 100 * mapped_rows / tot))
    # top unmapped by frequency
    top = unmapped_rows.most_common(15)
    print('  top unmapped codes (by rows):')
    for c, n in top:
        print('    %-10s %8d (%.2f%%)' % (c, n, 100 * n / tot))

coverage(freq9, icd9_map, 'ICD-9 -> Phecode')
coverage(freq10, icd10_map, 'ICD-10-CM -> Phecode')

# 5) combined + unique phecode count
allmapped = sum(n for c, n in list(freq9.items()) + list(freq10.items())
                if (c in icd9_map or c in icd10_map))
tot_all = tot9 + tot10
print('\n==== OVERALL ====')
print('  total diagnosis rows: %d' % tot_all)
print('  mapped to Phecode: %d (%.2f%%)' % (allmapped, 100 * allmapped / tot_all))
phecodes_used = set()
for c in freq9:
    if c in icd9_map: phecodes_used.add(icd9_map[c])
for c in freq10:
    if c in icd10_map: phecodes_used.add(icd10_map[c])
print('  unique phecodes used in MIMIC: %d' % len(phecodes_used))
