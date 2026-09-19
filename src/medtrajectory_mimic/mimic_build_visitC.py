# -*- coding: utf-8 -*-
"""Design C: one token per admission = composite of the first N codes of the admission code set.

Ordering: diagnoses by seq_num, then procedures by seq_num.
Composite frequency is counted on the TRAIN split only; composites below the threshold
collapse to '<OTHER>'. Semantic init for a composite = mean of its component Qwen embeddings.
Death remains a separate terminal event.
"""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import gzip, csv, json, argparse
from array import array
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyreadr

HOSP = str(MIMIC_ROOT / "mimic-iv-3.1" / "hosp")
PHEWAS = str(MIMIC_ROOT / "phewas_repo" / "data")
ICD10CSV = str(REPO_ROOT / "outputs" / "icd_embedding_compare" / "icd10_ontology" / "phecode_icd10cm_map.csv")
SEM = str(MIMIC_ROOT / "semantic_init" / "mimic_semantic_embeddings_64d.npy")
ALIGN = str(MIMIC_ROOT / "semantic_init" / "mimic_semantic_token_alignment.csv")


def canon(c):
    return (c or '').replace('.', '').strip().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-codes', type=int, default=2)
    ap.add_argument('--min-freq', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    icd9 = {}
    res = pyreadr.read_r(PHEWAS + '/phecode_map.rda')
    for _, r in res['phecode_map'].iterrows():
        if str(r['vocabulary_id']).startswith('ICD9'):
            icd9[canon(str(r['code']))] = str(r['phecode'])
    icd10 = {}
    with open(ICD10CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            icd10[canon(row['ICD_id'])] = row['Phecode']

    pat = {}
    with gzip.open(HOSP + '/patients.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            pat[row['subject_id']] = row
    adm_time, adm_subj = {}, {}
    with gzip.open(HOSP + '/admissions.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            adm_subj[row['hadm_id']] = row['subject_id']
            try:
                adm_time[row['hadm_id']] = datetime.strptime(row['admittime'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                adm_time[row['hadm_id']] = None

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

    N = args.n_codes
    admission_key = {}
    for hadm in set(dx) | set(px):
        codes = [c for _, c in sorted(dx.get(hadm, []))] + [c for _, c in sorted(px.get(hadm, []))]
        if codes:
            admission_key[hadm] = '|'.join(codes[:N])

    # patient split (same rule as the other builds)
    patient_ids = sorted(pat)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(patient_ids))
    n = len(patient_ids); n_train = int(n * 0.8); n_val = int(n * 0.1)
    split_of = {}
    for i, idx in enumerate(perm):
        pid = patient_ids[idx]
        split_of[pid] = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')

    # count composite frequency on TRAIN only
    train_freq = Counter()
    for hadm, subj in adm_subj.items():
        if split_of.get(subj) == 'train' and hadm in admission_key:
            train_freq[admission_key[hadm]] += 1
    retained = sorted(k for k, v in train_freq.items() if v >= args.min_freq)
    retained_set = set(retained)
    kept_visits = sum(v for k, v in train_freq.items() if k in retained_set)
    total_visits = sum(train_freq.values())
    print('distinct composites=%d retained=%d train_coverage=%.1f%% (min_freq=%d, N=%d)' % (
        len(train_freq), len(retained), 100 * kept_visits / max(total_visits, 1), args.min_freq, N))

    # vocabulary: retained composites + OTHER (+ death + special)
    dynamic_keys = ['composite:' + k for k in retained] + ['other:visit', 'death:event']
    token_id = {tk: i + 2 for i, tk in enumerate(dynamic_keys)}
    n_vocab = len(dynamic_keys) + 2

    # per-patient visit events
    visits = defaultdict(list)
    for hadm, subj in adm_subj.items():
        t = adm_time.get(hadm)
        if t is None or hadm not in admission_key:
            continue
        key = admission_key[hadm]
        tk = 'composite:' + key if key in retained_set else 'other:visit'
        visits[subj].append((t, tk))
    for subj, r in pat.items():
        dod = r.get('dod', '')
        if dod:
            try:
                visits[subj].append((datetime.strptime(dod, '%Y-%m-%d'), 'death:event'))
            except ValueError:
                pass

    buffers = {s: {'payload': array('I'), 'rows': []} for s in ('train', 'val', 'test')}
    for pid, evs in visits.items():
        r = pat[pid]
        try:
            anchor_age = float(r['anchor_age']); anchor_year = int(r['anchor_year'])
        except (ValueError, TypeError):
            continue
        anchor_date = datetime(anchor_year, 1, 1)
        ev = [(anchor_age * 365.25 + (t - anchor_date).days, tk) for t, tk in evs]
        ev.sort(key=lambda x: (x[0], x[1]))
        ev = [(a, k) for i, (a, k) in enumerate(ev) if i == 0 or (a, k) != ev[i - 1]]
        if len(ev) < 2:
            continue
        s = split_of[pid]
        ridx = len(buffers[s]['rows'])
        for age_days, tk in ev:
            buffers[s]['payload'].extend((int(pid), max(0, int(round(age_days))), token_id[tk] - 1))
        buffers[s]['rows'].append({'row_index': ridx, 'eid': pid, 'num_events': len(ev),
                                   'sex': 1.0 if r.get('gender') == 'M' else 0.0,
                                   'anchor_age_days': anchor_age * 365.25})

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for s in ('train', 'val', 'test'):
        b = buffers[s]
        np.asarray(b['payload'], dtype=np.uint32).tofile(out / f'{s}.bin')
        np.save(out / f'{s}_static.npy', np.asarray([[r['sex']] for r in b['rows']], dtype=np.float32))
        np.save(out / f'{s}_followup_end_age_days.npy', np.asarray([r['anchor_age_days'] for r in b['rows']], dtype=np.float32))
        with (out / f'{s}_patient_index.csv').open('w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['row_index', 'eid', 'num_events', 'sex'])
            for r in b['rows']:
                w.writerow([r['row_index'], r['eid'], r['num_events'], r['sex']])
        print('split %s: patients=%d events=%d' % (s, len(b['rows']), len(b['payload']) // 3))

    # semantic embeddings: composite = mean of component code embeddings; OTHER/death = zeros
    align_index = {}
    with open(ALIGN, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            align_index[row['token_key']] = i
    sem = np.load(SEM).astype(np.float32)
    rows = [np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32)]
    for tk in dynamic_keys:
        if tk.startswith('composite:'):
            parts = tk[len('composite:'):].split('|')
            vecs = [sem[align_index[p]] for p in parts if p in align_index]
            rows.append(np.mean(np.vstack(vecs), axis=0) if vecs else np.zeros(64, dtype=np.float32))
        else:
            rows.append(np.zeros(64, dtype=np.float32))
    sem_full = np.vstack(rows).astype(np.float32)
    vocab_dir = out / 'vocab'; vocab_dir.mkdir(parents=True, exist_ok=True)
    with (vocab_dir / 'dynamic_token_vocab.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['token_id', 'token_key', 'event_type'])
        w.writerow([0, 'Padding', 'special']); w.writerow([1, 'No event', 'special'])
        for i, tk in enumerate(dynamic_keys):
            et = 'death' if tk == 'death:event' else 'diagnosis'
            w.writerow([i + 2, tk, et])
    np.save(vocab_dir / 'semantic_input_embeddings_64d.npy', sem_full)
    (out / 'prepare_manifest.json').write_text(json.dumps({
        'semantic_output': 'vocab/semantic_input_embeddings_64d.npy',
        'static_feature_order': ['sex'],
        'vocab_size': n_vocab,
        'vocab_csv': 'vocab/dynamic_token_vocab.csv',
        'visit_level': 'design_C_composite_first%d_minfreq%d' % (N, args.min_freq),
    }, indent=2), encoding='utf-8')
    print('done. vocab_size=%d semantic=%s' % (n_vocab, sem_full.shape))


if __name__ == '__main__':
    main()
