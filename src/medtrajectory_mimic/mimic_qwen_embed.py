# -*- coding: utf-8 -*-
"""Generate Qwen text-embedding-v4 semantic initialization for MIMIC-IV vocabulary:
Phecode diagnoses + CPT/HCPCS procedures + death -> 1024-dim -> PCA 64-dim."""
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import MIMIC_ROOT, REPO_ROOT
except ImportError:  # executed from inside the source tree
    from paths import MIMIC_ROOT, REPO_ROOT
import json, time, csv, gzip
import urllib.request
import numpy as np
import pyreadr
from sklearn.decomposition import PCA

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'
MIMIC = str(MIMIC_ROOT / "mimic-iv-3.1" / "hosp")
PHEWAS = str(MIMIC_ROOT / "phewas_repo" / "data")
ICD10CSV = str(REPO_ROOT / "outputs" / "icd_embedding_compare" / "icd10_ontology" / "phecode_icd10cm_map.csv")
OUT = str(MIMIC_ROOT / "semantic_init")

import os
os.makedirs(OUT, exist_ok=True)

def canon(code):
    return (code or '').replace('.', '').strip().upper()

# ---- 1. load maps and phecode descriptions ----
icd9_map = {}
res = pyreadr.read_r(PHEWAS + '/phecode_map.rda')
for _, row in res['phecode_map'].iterrows():
    if str(row['vocabulary_id']).startswith('ICD9'):
        icd9_map[canon(str(row['code']))] = str(row['phecode'])
icd10_map = {}
with open(ICD10CSV, 'r', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        icd10_map[canon(row['ICD_id'])] = row['Phecode']

pheinfo = pyreadr.read_r(PHEWAS + '/pheinfo.rda')['pheinfo']
phe_desc = {str(r['phecode']): str(r['description']) for _, r in pheinfo.iterrows()}

# ---- 2. determine used phecodes (from MIMIC diagnoses) ----
used_phecodes = set()
with gzip.open(MIMIC + '/diagnoses_icd.csv.gz', 'rt') as f:
    for row in csv.DictReader(f):
        c = canon(row['icd_code']); v = row['icd_version']
        pc = icd9_map.get(c) if v == '9' else icd10_map.get(c)
        if pc:
            used_phecodes.add(pc)

# ---- 3. determine used CPT codes (from hcpcsevents) ----
used_cpt = {}
with gzip.open(MIMIC + '/hcpcsevents.csv.gz', 'rt') as f:
    for row in csv.DictReader(f):
        cd = row['hcpcs_cd']; desc = row.get('short_description', '')
        if cd and cd not in used_cpt:
            used_cpt[cd] = desc or cd

# ---- 4. build token list: (token_key, event_type, description) ----
tokens = []
for pc in sorted(used_phecodes):
    tokens.append((f'phecode:{pc}', 'diagnosis', phe_desc.get(pc, pc)))
for cd in sorted(used_cpt):
    tokens.append((f'cpt:{cd}', 'procedure', used_cpt[cd]))
tokens.append(('death:event', 'death', 'death'))

print('tokens to embed:', len(tokens))
print('  phecodes:', len(used_phecodes), ' cpt:', len(used_cpt), ' death: 1')

# ---- 5. batch-embed with Qwen ----
def embed_batch(texts):
    body = json.dumps({"model": "text-embedding-v4", "input": texts, "dimensions": 1024}).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={
        'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read().decode('utf-8'))
            embs = [d['embedding'] for d in sorted(out['data'], key=lambda x: x['index'])]
            return embs
        except Exception as e:
            print('  retry', attempt, 'err', str(e)[:120])
            time.sleep(3)
    raise RuntimeError('embedding failed')

descs = [t[2] for t in tokens]
BATCH = 10
all_embs = []
for i in range(0, len(descs), BATCH):
    batch = descs[i:i + BATCH]
    embs = embed_batch(batch)
    all_embs.extend(embs)
    if (i // BATCH) % 20 == 0:
        print('  embedded', i + len(batch), '/', len(descs))

X = np.asarray(all_embs, dtype=np.float64)
print('embedding matrix:', X.shape)

# ---- 6. PCA to 64 dim ----
pca = PCA(n_components=64)
X64 = pca.fit_transform(X)
print('PCA explained variance (64d):', round(float(pca.explained_variance_ratio_.sum()), 4))

# ---- 7. save ----
np.save(OUT + '/mimic_semantic_embeddings_64d.npy', X64.astype(np.float32))
with open(OUT + '/mimic_semantic_token_alignment.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['token_key', 'event_type', 'description', 'embedding_index'])
    for idx, (tk, et, desc) in enumerate(tokens):
        w.writerow([tk, et, desc, idx])
with open(OUT + '/pca_meta.json', 'w') as f:
    json.dump({'n_tokens': len(tokens), 'input_dim': 1024, 'output_dim': 64,
               'explained_variance_ratio': float(pca.explained_variance_ratio_.sum()),
               'model': 'text-embedding-v4'}, f, indent=2)
print('saved to', OUT)
print('done. npy shape:', X64.shape)
