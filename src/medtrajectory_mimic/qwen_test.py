# -*- coding: utf-8 -*-
import os, json, urllib.request

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'

def embed(text, dimensions=1024):
    body = json.dumps({"model": "text-embedding-v4", "input": text, "dimensions": dimensions}).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={
        'Authorization': 'Bearer ' + API_KEY,
        'Content-Type': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode('utf-8')[:500]}

# test 1: dimensions param
res = embed('Intestinal infection')
print('=== test with dimensions=1024 ===')
print(json.dumps(res, ensure_ascii=False)[:600])

# test 2: without dimensions (default 1024 for text-embedding-v4)
body2 = json.dumps({"model": "text-embedding-v4", "input": "Intestinal infection"}).encode('utf-8')
req2 = urllib.request.Request(URL, data=body2, headers={'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req2, timeout=60) as r:
        res2 = json.loads(r.read().decode('utf-8'))
    emb = res2['data'][0]['embedding']
    print('\n=== test without dimensions ===')
    print('dimension:', len(emb))
    print('first 5:', emb[:5])
except Exception as e:
    print('\ntest2 error:', e)
