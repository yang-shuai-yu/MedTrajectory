# -*- coding: utf-8 -*-
import os, json, urllib.request, urllib.error

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'

def call(inputs, dimensions=1024):
    body = json.dumps({"model": "text-embedding-v4", "input": inputs, "dimensions": dimensions}).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={
        'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode('utf-8'))
            return 'OK', len(d['data'])
    except urllib.error.HTTPError as e:
        return 'HTTP %d' % e.code, e.read().decode('utf-8')[:300]

for n in [1, 2, 5, 10, 25]:
    texts = ['Intestinal infection'] * n
    code, body = call(texts)
    print('batch %2d -> %s | %s' % (n, code, body))

# also test without dimensions param
body = json.dumps({"model": "text-embedding-v4", "input": ['a', 'b']}).encode('utf-8')
req = urllib.request.Request(URL, data=body, headers={'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode('utf-8'))
        print('no-dim batch2 -> OK, data len', len(d['data']))
except urllib.error.HTTPError as e:
    print('no-dim batch2 -> HTTP %d' % e.code, e.read().decode('utf-8')[:300])
