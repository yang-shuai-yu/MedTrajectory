"""Read the accepted model's terminal hidden state without changing its weights."""
import bisect
import contextlib
import io
from pathlib import Path
import sys
import time

import numpy as np
import torch

import core
from assets import CONFIG, INPUT, P, module, read, require, sha, verify_sources


def setup():
    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only worker required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)


class Adapter:
    def __init__(self, method):
        setup()
        self.method = method
        item = CONFIG["models"][method]
        root = Path(item["release"])
        sys.path.insert(0, str(root))
        runtime = module("runtime", root / "runtime.py")
        state = torch.load(item["checkpoint"]["path"], map_location="cpu", weights_only=False)
        require(state["identity"] == read(item["training_identity_path"]), "Checkpoint identity differs")
        require(state["step"] == item["checkpoint"]["step"], "Checkpoint step differs")
        labels = [v for v in (INPUT / "labels.csv").read_text().splitlines() if v.startswith("diag:")]
        if method in ("gpt", "bert"):
            native = state["diagnosis_labels"]
            first = 2 if method == "gpt" else 3
            require(native[first:] == labels, "Native diagnosis ordering differs")
            source = module("time_backbone_" + method, root / "model.py")
            constructor = source.DiagnosisGPT if method == "gpt" else source.DiagnosisBERT
            self.model = constructor(len(native))
            if method == "bert":
                import transformers
                from transformers.models.bert import modeling_bert
                require(state["identity"]["libraries"] == dict(torch=torch.__version__,
                    transformers=transformers.__version__, modeling_bert_sha256=sha(modeling_bert.__file__)),
                    "BERT runtime implementation changed")
        else:
            native = read(Path(item["training_identity_path"]).parents[2] / "input/vocabulary.json")["labels"]
            require(native[2:] == labels, "Adapted diagnosis ordering differs")
            source = module("models", root / "models.py")
            # Some pinned constructors print model size; keep stdout machine-readable.
            with contextlib.redirect_stdout(io.StringIO()):
                self.model = source.build(method, native)
        self.model.load_state_dict(state["model"], strict=True)
        del state
        self.model.eval().requires_grad_(False)
        self.protocol = runtime.PROTOCOL
        self.metadata = dict(method=method, checkpoint=item["checkpoint"],
                             frozen_parameters=sum(p.numel() for p in self.model.parameters()),
                             torch_version=torch.__version__, feature="native terminal hidden state + six prefix covariates")

    def feature(self, row):
        require(set(row) == {"history", "history_ages", "cut_age"}, "Non-prefix inputs supplied")
        method, m = self.method, self.model
        history, ages = row["history"], row["history_ages"]
        require(len(history) == len(ages) and all(0 <= t < 1123 for t in history),
                "Invalid diagnosis prefix")
        require(all(a <= b for a, b in zip(ages, ages[1:])), "Unordered diagnosis prefix")
        with torch.inference_mode():
            if method == "gpt":
                x = torch.tensor([[1, *[t+2 for t in history[-127:]]]], dtype=torch.long)
                pos = torch.arange(x.shape[1])[None]
                hidden = m.dropout(m.norm(m.token(x) + m.position(pos)))
                mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).triu(1)
                hidden = m.encoder(hidden, mask=mask, src_key_padding_mask=x == 0)
            elif method == "bert":
                x = torch.tensor([[1, *[t+3 for t in history[-126:]], 2]], dtype=torch.long)
                hidden = m.bert(input_ids=x, attention_mask=x.ne(0), return_dict=True).last_hidden_state
            elif method == "medbert":
                x = torch.tensor([[*[t+2 for t in history[-127:]], 0]], dtype=torch.long)
                hidden = m.core.hidden(dict(x=x, valid=torch.ones_like(x, dtype=torch.bool)))
            elif method == "trajgpt":
                x = torch.tensor([[1, *[t+2 for t in history[-127:]]]], dtype=torch.long)
                observed = torch.tensor([[0., *ages[-127:]]], dtype=torch.float64) / 365.25
                valid = torch.ones_like(x, dtype=torch.bool)
                hidden = m.embedding(x)
                for block in m.blocks:
                    hidden, _ = block(hidden, observed, retention_mask=valid, forward_impl="parallel")
                hidden = m.norm(hidden)
            else:
                require(method == "ethos", "Unknown method")
                history, ages = history[-127:], ages[-127:]
                sequence = [1]
                thresholds = self.protocol["models"]["ethos"]["gap_bin_minutes"]
                for i, token in enumerate(history):
                    if i:
                        bucket = bisect.bisect_right(thresholds, (ages[i]-ages[i-1])*1440.)
                        if bucket:
                            sequence.append(1125 + bucket - 1)
                    sequence.append(token+2)
                captured = []
                hook = m.core.lm_head.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
                try:
                    m.core(torch.tensor([sequence], dtype=torch.long))
                    require(len(captured) == 1, "ETHOS terminal state not captured")
                    hidden = captured[0]
                finally:
                    hook.remove()
            result = np.r_[hidden[0, -1].float().numpy(), np.array(core.covariates(row), dtype=np.float32)]
        require(np.isfinite(result).all() and not torch.cuda.is_initialized(), "Invalid CPU features")
        return result.astype(np.float32)

    def gate(self):
        started = time.time()
        shapes = []
        for n in (0, 3, 127):
            row = dict(history=[i % 1123 for i in range(n)],
                       history_ages=[15000.+i*10 for i in range(n)], cut_age=17000.)
            first = self.feature(row)
            second = self.feature(row)
            require(np.array_equal(first, second), "Frozen feature replay differs")
            shapes.append(len(first))
        require(len(set(shapes)) == 1 and not any(p.requires_grad for p in self.model.parameters()),
                "Backbone not frozen or inconsistent feature dimensions")
        return dict(passed=True, method=self.method, feature_width=shapes[0],
                    empty_and_long_prefix_passed=True, exact_feature_replay=True,
                    frozen_backbone=True, cuda_initialized=torch.cuda.is_initialized(),
                    seconds=time.time()-started, metadata=self.metadata)


def smoke(method):
    require(method in P["methods"], "Smoke requires a method")
    verify_sources(method)
    return Adapter(method).gate()
