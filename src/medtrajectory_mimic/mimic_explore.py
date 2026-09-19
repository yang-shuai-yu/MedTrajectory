# -*- coding: utf-8 -*-
"""MIMIC-IV v3.1 exploration: hosp-module tables relevant to multitype trajectory modeling.
Prints schema, row counts, ICD version split, event-type proportions, mortality, admissions/patient."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT
import gzip, csv, glob, os
from collections import Counter, defaultdict

BASE = str(MIMIC_ROOT / "mimic-iv-3.1")

def path(rel):
    p = os.path.join(BASE, rel)
    return p if os.path.exists(p) else None

def count_rows(p):
    n = 0
    with gzip.open(p, 'rt') as f:
        for _ in f:
            n += 1
    return max(0, n - 1)  # header

def header(p):
    with gzip.open(p, 'rt') as f:
        return next(csv.reader(f))

def sample(p, k=3):
    with gzip.open(p, 'rt') as f:
        r = csv.reader(f)
        h = next(r)
        rows = []
        for i, row in enumerate(r):
            if i >= k:
                break
            rows.append(row)
        return h, rows

out = []

def emit(s):
    out.append(s)

emit('==== MIMIC-IV v3.1 database exploration ====')
emit('base: %s' % BASE)

# 1) top-level layout
emit('\n== 1. module layout ==')
for d in sorted(os.listdir(BASE)):
    full = os.path.join(BASE, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        emit('  %s/ : %d files' % (d, len(files)))

# 2) hosp module core tables
emit('\n== 2. hosp module: relevant tables ==')
tables = [
    'hosp/patients.csv.gz',
    'hosp/admissions.csv.gz',
    'hosp/diagnoses_icd.csv.gz',
    'hosp/procedures_icd.csv.gz',
    'hosp/d_icd_diagnoses.csv.gz',
    'hosp/d_icd_procedures.csv.gz',
    'hosp/drgcodes.csv.gz',
    'hosp/services.csv.gz',
    'hosp/transfers.csv.gz',
]
for t in tables:
    p = path(t)
    if p:
        n = count_rows(p)
        h = header(p)
        emit('  %-28s rows=%8d  cols=%s' % (t, n, ','.join(h)))
    else:
        emit('  %-28s MISSING' % t)

# 3) patients: demographics + mortality
emit('\n== 3. patients (demographics, mortality) ==')
p = path('hosp/patients.csv.gz')
h, rows = sample(p, 3)
emit('  header: %s' % ','.join(h))
emit('  sample: %s' % rows[0])
# full pass for stats
n_pat = 0; gender = Counter(); death = 0; age_sum = 0; age_n = 0
with gzip.open(p, 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        n_pat += 1
        gender[row.get('gender','')] += 1
        if row.get('dod',''):
            death += 1
        a = row.get('anchor_age','')
        if a and a.isdigit():
            age_sum += int(a); age_n += 1
emit('  patients: %d' % n_pat)
emit('  gender: %s' % dict(gender))
emit('  died (dod present): %d (%.2f%%)' % (death, 100*death/n_pat))
emit('  mean anchor_age: %.1f (n=%d)' % (age_sum/age_n if age_n else 0, age_n))

# 4) admissions: types, hospital expire, admissions per patient
emit('\n== 4. admissions ==')
p = path('hosp/admissions.csv.gz')
n_adm = 0; admit_type = Counter(); expire = 0; deathtime = 0; adm_per_pat = Counter()
with gzip.open(p, 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        n_adm += 1
        admit_type[row.get('admission_type','')] += 1
        if row.get('hospital_expire_flag','') == '1':
            expire += 1
        if row.get('deathtime',''):
            deathtime += 1
        adm_per_pat[row.get('subject_id','')] += 1
emit('  admissions: %d' % n_adm)
emit('  admission_type: %s' % dict(admit_type))
emit('  hospital_expire_flag=1: %d (%.2f%% of admissions)' % (expire, 100*expire/n_adm))
emit('  deathtime present: %d (%.2f%%)' % (deathtime, 100*deathtime/n_adm))
if adm_per_pat:
    vals = sorted(adm_per_pat.values())
    import statistics
    emit('  admissions/patient: mean=%.2f median=%d max=%d (n_pat=%d)' % (
        statistics.mean(vals), statistics.median(vals), max(vals), len(vals)))

# 5) diagnoses_icd: ICD version, unique codes, seq
emit('\n== 5. diagnoses_icd ==')
p = path('hosp/diagnoses_icd.csv.gz')
n_diag = 0; ver = Counter(); codes = Counter(); seq1 = 0
with gzip.open(p, 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        n_diag += 1
        ver[row.get('icd_version','')] += 1
        codes[row.get('icd_code','')] += 1
        if row.get('seq_num','') == '1':
            seq1 += 1
emit('  diagnoses rows: %d' % n_diag)
emit('  icd_version: %s' % dict(ver))
emit('  unique icd codes: %d' % len(codes))
emit('  seq_num=1 (primary) rows: %d (%.2f%%)' % (seq1, 100*seq1/n_diag))

# 6) procedures_icd
emit('\n== 6. procedures_icd ==')
p = path('hosp/procedures_icd.csv.gz')
n_proc = 0; ver = Counter(); codes = Counter()
with gzip.open(p, 'rt') as f:
    r = csv.DictReader(f)
    for row in r:
        n_proc += 1
        ver[row.get('icd_version','')] += 1
        codes[row.get('icd_code','')] += 1
emit('  procedures rows: %d' % n_proc)
emit('  icd_version: %s' % dict(ver))
emit('  unique icd codes: %d' % len(codes))

# 7) dictionaries
emit('\n== 7. dictionaries ==')
for t in ['hosp/d_icd_diagnoses.csv.gz', 'hosp/d_icd_procedures.csv.gz']:
    p = path(t)
    if p:
        n = count_rows(p)
        h = header(p)
        emit('  %-28s rows=%d cols=%s' % (t, n, ','.join(h)))

# 8) event-type proportions (diagnosis vs procedure vs death-event)
emit('\n== 8. event-type proportions (for multitype framing) ==')
n_diag_ev = n_diag if 'n_diag' in dir() else 0
emit('  diagnosis events: %d' % n_diag)
emit('  procedure events: %d' % n_proc)
emit('  death events (deathtime in admissions): %d' % deathtime)
tot = n_diag + n_proc + deathtime
emit('  proportions: diag %.1f%%, proc %.1f%%, death %.1f%%' % (
    100*n_diag/tot, 100*n_proc/tot, 100*deathtime/tot))

open(str(MIMIC_ROOT / "exploration_report.txt"), 'w').write('\n'.join(out))
print('\n'.join(out))
