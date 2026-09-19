import hashlib
import sys

import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ckpt["model"]
h = hashlib.sha256()
for k in sorted(sd):
    h.update(k.encode())
    h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
print(f"{sys.argv[1]}\titer={ckpt['iteration']}\tsha256={h.hexdigest()}")
