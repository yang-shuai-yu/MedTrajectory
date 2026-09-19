# -*- coding: utf-8 -*-
"""Design B-matched: visit-level multitype ablation on a FIXED patient universe and a FIXED split.

Problem being fixed: the original visit builds each computed their own random permutation over
their own patient list, so M1/M2/M3 had effectively independent train/val/test assignments and
the test-set intersection collapsed (44 patients at UKB thresholds). Here:

  1. the patient universe is fixed to participants with >= 2 diagnosis-bearing visits (the most
     restrictive setting, i.e. M1-eligible), so every representation is built over the SAME patients;
  2. the split is computed once over that universe (seed 1337, 80/10/10) and reused across the
     three event-type builds, so M1 test subset M2 test subset M3 test.

Write the split map on the first invocation with --split-map-out, then reuse with --split-map-in.
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


def canon(c):
    return (c or '').replace('.', '').strip().upper()


def load_all_visits():
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
    dx_by_hadm = {}
    with gzip.open(HOSP + '/diagnoses_icd.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            if row.get('seq_num') != '1':
                continue
            pc = (icd9.get(canon(row['icd_code'])) if row['icd_version'] == '9'
                  else icd10.get(canon(row['icd_code'])))
            if pc:
                dx_by_hadm[row['hadm_id']] = 'phecode:' + pc
    px_by_hadm = {}
    with gzip.open(HOSP + '/hcpcsevents.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            if row.get('seq_num') != '1':
                continue
            code = (row.get('hcpcs_cd') or '').strip()
            if code:
                px_by_hadm[row['hadm_id']] = 'cpt:' + code

    visits = defaultdict(list)
    for hadm, subj in adm_subj.items():
        t = adm_time.get(hadm)
        if t is None:
            continue
        if hadm in dx_by_hadm:
            visits[subj].append((t, dx_by_hadm[hadm]))
        if hadm in px_by_hadm:
            visits[subj].append((t, px_by_hadm[hadm]))
    return pat, visits, dx_by_hadm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--event-types', default='diagnosis')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--split-map-out', default=None)
    ap.add_argument('--split-map-in', default=None)
    args = ap.parse_args()
    event_types = {x.strip() for x in args.event_types.split(',') if x.strip()}

    pat, visits, _ = load_all_visits()

    # universe = participants with >= 2 diagnosis-bearing visits (M1-eligible)
    universe = sorted(
        pid for pid, evs in visits.items()
        if len({t for t, tk in evs if tk.startswith('phecode:')}) >= 2
    )
    print('universe (>=2 diagnosis-bearing visits): %d' % len(universe))

    if args.split_map_in:
        split_of = {}
        with open(args.split_map_in, newline='') as fh:
            for row in csv.DictReader(fh):
                split_of[row['eid']] = row['split']
        print('loaded split map from %s (%d patients)' % (args.split_map_in, len(split_of)))
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(universe))
        n = len(universe); n_train = int(n * 0.8); n_val = int(n * 0.1)
        split_of = {}
        for i, idx in enumerate(perm):
            pid = universe[idx]
            split_of[pid] = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')
        if args.split_map_out:
            with open(args.split_map_out, 'w', newline='') as fh:
                w = csv.writer(fh); w.writerow(['eid', 'split'])
                for pid in universe:
                    w.writerow([pid, split_of[pid]])
            print('wrote split map to %s' % args.split_map_out)

    dynamic_keys = []
    with open(ALIGN, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['event_type'] in event_types:
                dynamic_keys.append(row['token_key'])
    token_id = {tk: i + 2 for i, tk in enumerate(dynamic_keys)}
    n_vocab = len(dynamic_keys) + 2
    print('dynamic tokens=%d vocab_size=%d' % (len(dynamic_keys), n_vocab))

    buffers = {s: {'payload': array('I'), 'rows': []} for s in ('train', 'val', 'test')}
    for pid in universe:
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
            if tk == 'death:event' and 'death' not in event_types:
                continue
            ev.append((anchor_age * 365.25 + (t - anchor_date).days, tk))
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

    align_index = {}
    with open(ALIGN, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            align_index[row['token_key']] = i
    sem = np.load(SEM).astype(np.float32)
    sem_sub = sem[np.asarray([align_index[tk] for tk in dynamic_keys], dtype=np.int64)]
    sem_full = np.vstack([np.zeros((2, 64), dtype=np.float32), sem_sub])
    vocab_dir = out / 'vocab'; vocab_dir.mkdir(parents=True, exist_ok=True)
    with (vocab_dir / 'dynamic_token_vocab.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['token_id', 'token_key', 'event_type'])
        w.writerow([0, 'Padding', 'special']); w.writerow([1, 'No event', 'special'])
        for i, tk in enumerate(dynamic_keys):
            et = 'death' if tk == 'death:event' else ('procedure' if tk.startswith('cpt:') else 'diagnosis')
            w.writerow([i + 2, tk, et])
    np.save(vocab_dir / 'semantic_input_embeddings_64d.npy', sem_full)
    (out / 'prepare_manifest.json').write_text(json.dumps({
        'semantic_output': 'vocab/semantic_input_embeddings_64d.npy',
        'static_feature_order': ['sex'],
        'vocab_size': n_vocab,
        'vocab_csv': 'vocab/dynamic_token_vocab.csv',
        'visit_level': 'design_B_matched_fixed_universe_fixed_split',
        'matched_universe': 'ge2_diagnosis_bearing_visits',
        'split_seed': args.seed,
    }, indent=2), encoding='utf-8')
    print('done. vocab_size=%d semantic=%s' % (n_vocab, sem_full.shape))


if __name__ == '__main__':
    main()
