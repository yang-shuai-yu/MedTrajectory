# -*- coding: utf-8 -*-
"""Build MIMIC-IV VISIT-LEVEL (admission-level) event sequences — design B.

Design B: each admission (hadm_id) contributes at most TWO tokens at the admission
timestamp: the principal diagnosis (diagnoses_icd seq_num=1 -> Phecode) and the
principal procedure (hcpcsevents seq_num=1 -> CPT). This collapses the 93% same-day
event runs of the raw event-level build into one time point per visit, so the
continuous relative-time RoPE operates on inter-admission gaps.

Token convention (unchanged): 0=Padding, 1=No event, 2..N+1=dynamic; .bin stores token_id-1.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import gzip, csv, json, argparse
from array import array
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyreadr

HOSP = str(MIMIC_ROOT / "mimic-iv-3.1" / "hosp")
PHEWAS = str(MIMIC_ROOT / "phewas_repo" / "data")
ICD10CSV = str(REPO_ROOT / "outputs" / "icd_embedding_compare" / "icd10_ontology" / "phecode_icd10cm_map.csv")
SEM = str(MIMIC_ROOT / "semantic_init" / "mimic_semantic_embeddings_64d.npy")
ALIGN = str(MIMIC_ROOT / "semantic_init" / "mimic_semantic_token_alignment.csv")


def canon(code):
    return (code or '').replace('.', '').strip().upper()


def load_phecode_maps():
    icd9 = {}
    res = pyreadr.read_r(PHEWAS + '/phecode_map.rda')
    for _, r in res['phecode_map'].iterrows():
        if str(r['vocabulary_id']).startswith('ICD9'):
            icd9[canon(str(r['code']))] = str(r['phecode'])
    icd10 = {}
    with open(ICD10CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            icd10[canon(row['ICD_id'])] = row['Phecode']
    return icd9, icd10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--event-types', default='diagnosis,procedure,death')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()
    event_types = {x.strip() for x in args.event_types.split(',') if x.strip()}

    icd9, icd10 = load_phecode_maps()

    print('loading patients / admissions ...')
    pat = {}
    with gzip.open(HOSP + '/patients.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            pat[row['subject_id']] = row
    adm_time = {}
    adm_subj = {}
    with gzip.open(HOSP + '/admissions.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            hadm = row['hadm_id']
            adm_subj[hadm] = row['subject_id']
            try:
                adm_time[hadm] = datetime.strptime(row['admittime'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                adm_time[hadm] = None

    print('principal diagnosis per admission ...')
    dx_by_hadm = {}
    with gzip.open(HOSP + '/diagnoses_icd.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            if row.get('seq_num') != '1':
                continue
            pc = (icd9.get(canon(row['icd_code'])) if row['icd_version'] == '9'
                  else icd10.get(canon(row['icd_code'])))
            if pc:
                dx_by_hadm[row['hadm_id']] = 'phecode:' + pc

    print('principal procedure per admission ...')
    px_by_hadm = {}
    with gzip.open(HOSP + '/hcpcsevents.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            if row.get('seq_num') != '1':
                continue
            code = (row.get('hcpcs_cd') or '').strip()
            if code:
                px_by_hadm[row['hadm_id']] = 'cpt:' + code

    print('building visit events ...')
    # per subject: list of (admittime, token_key)
    visits = defaultdict(list)
    for hadm, subj in adm_subj.items():
        t = adm_time.get(hadm)
        if t is None:
            continue
        role_order = []
        if 'diagnosis' in event_types and hadm in dx_by_hadm:
            role_order.append(dx_by_hadm[hadm])
        if 'procedure' in event_types and hadm in px_by_hadm:
            role_order.append(px_by_hadm[hadm])
        for tk in role_order:
            visits[subj].append((t, tk))
    if 'death' in event_types:
        for subj, r in pat.items():
            dod = r.get('dod', '')
            if dod:
                try:
                    visits[subj].append((datetime.strptime(dod, '%Y-%m-%d'), 'death:event'))
                except ValueError:
                    pass

    type_to_kind = {'phecode:': 'diagnosis', 'cpt:': 'procedure', 'death:event': 'death'}
    dynamic_keys = []
    with open(ALIGN, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['event_type'] in event_types:
                dynamic_keys.append(row['token_key'])
    token_id = {tk: i + 2 for i, tk in enumerate(dynamic_keys)}
    n_vocab = len(dynamic_keys) + 2
    print('dynamic tokens=%d vocab_size=%d' % (len(dynamic_keys), n_vocab))

    patient_ids = list(visits.keys())
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(patient_ids))
    n = len(patient_ids); n_train = int(n * 0.8); n_val = int(n * 0.1)
    split_of = {}
    for i, idx in enumerate(perm):
        pid = patient_ids[idx]
        split_of[pid] = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')

    buffers = {s: {'payload': array('I'), 'rows': []} for s in ('train', 'val', 'test')}
    for pid in patient_ids:
        r = pat[pid]
        try:
            anchor_age = float(r['anchor_age']); anchor_year = int(r['anchor_year'])
        except (ValueError, TypeError):
            continue
        anchor_date = datetime(anchor_year, 1, 1)
        ev = []
        for t, tk in visits[pid]:
            if tk not in token_id:
                continue
            age_days = anchor_age * 365.25 + (t - anchor_date).days
            ev.append((age_days, tk))
        ev.sort(key=lambda x: (x[0], x[1]))
        ev = [(a, k) for i, (a, k) in enumerate(ev) if i == 0 or (a, k) != ev[i - 1]]
        if len(ev) < 2:
            continue
        s = split_of[pid]
        row_index = len(buffers[s]['rows'])
        for age_days, tk in ev:
            buffers[s]['payload'].extend((int(pid), max(0, int(round(age_days))), token_id[tk] - 1))
        sex = 1.0 if r.get('gender') == 'M' else 0.0
        buffers[s]['rows'].append({'row_index': row_index, 'eid': pid, 'num_events': len(ev),
                                   'sex': sex, 'anchor_age_days': anchor_age * 365.25})

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for s in ('train', 'val', 'test'):
        b = buffers[s]
        arr = np.asarray(b['payload'], dtype=np.uint32)
        with (out / f'{s}.bin').open('wb') as fh:
            arr.tofile(fh)
        static = np.asarray([[r['sex']] for r in b['rows']], dtype=np.float32)
        np.save(out / f'{s}_static.npy', static)
        followup = np.asarray([r['anchor_age_days'] for r in b['rows']], dtype=np.float32)
        np.save(out / f'{s}_followup_end_age_days.npy', followup)
        with (out / f'{s}_patient_index.csv').open('w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['row_index', 'eid', 'num_events', 'sex'])
            for r in b['rows']:
                w.writerow([r['row_index'], r['eid'], r['num_events'], r['sex']])
        lens = [r['num_events'] for r in b['rows']]
        print('split %s: patients=%d events=%d median_visits=%.0f' % (
            s, len(b['rows']), len(b['payload']) // 3, float(np.median(lens)) if lens else 0))

    vocab_dir = out / 'vocab'; vocab_dir.mkdir(parents=True, exist_ok=True)
    align_index = {}
    with open(ALIGN, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            align_index[row['token_key']] = i
    sem = np.load(SEM).astype(np.float32)
    sem_sub = sem[np.asarray([align_index[tk] for tk in dynamic_keys], dtype=np.int64)]
    sem_full = np.vstack([np.zeros((2, 64), dtype=np.float32), sem_sub])
    with (vocab_dir / 'dynamic_token_vocab.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['token_id', 'token_key', 'event_type'])
        w.writerow([0, 'Padding', 'special'])
        w.writerow([1, 'No event', 'special'])
        for i, tk in enumerate(dynamic_keys):
            et = 'death' if tk == 'death:event' else ('procedure' if tk.startswith('cpt:') else 'diagnosis')
            w.writerow([i + 2, tk, et])
    np.save(vocab_dir / 'semantic_input_embeddings_64d.npy', sem_full)
    manifest = {
        'semantic_output': 'vocab/semantic_input_embeddings_64d.npy',
        'static_feature_order': ['sex'],
        'vocab_size': n_vocab,
        'vocab_csv': 'vocab/dynamic_token_vocab.csv',
        'visit_level': 'design_B_principal_dx_plus_principal_procedure',
    }
    with (out / 'prepare_manifest.json').open('w') as fh:
        json.dump(manifest, fh, indent=2)
    print('done. vocab_size=%d semantic=%s' % (n_vocab, sem_full.shape))


if __name__ == '__main__':
    main()
