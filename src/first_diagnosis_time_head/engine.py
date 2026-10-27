"""Resumable CPU feature extraction, independent time fitting and gated evaluation."""
import argparse
import copy
import fcntl
import io
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

import core
from assets import OPS, P, ROOT, RUN, examples_for, read, require, save, sha, verify_sources
from head import TimeHead, nll
from model_adapter import Adapter


def progress(method, stage, **values):
    save(OPS / (method + ".json"), dict(stage=stage, updated_unix=time.time(), **values))


def binary_save(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".part")
    with temporary.open("wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def features(adapter, split, rows):
    folder = RUN / adapter.method / "features" / split
    chunk = P["resources"]["feature_chunk_size"]
    result, hashes = [], []
    started = time.time()
    newly_done = 0
    for start in range(0, len(rows), chunk):
        subset = rows[start:start+chunk]
        path = folder / f"chunk_{start:06d}.npy"
        stamp = dict(source_sha256=core.digest(subset), checkpoint_sha256=adapter.metadata["checkpoint"]["sha256"],
                     manifest_sha256=sha(ROOT / "manifest.json"))
        receipt = read(path.with_suffix(".receipt.json"))
        if receipt:
            require(all(receipt[k] == v for k, v in stamp.items()) and sha(path) == receipt["sha256"],
                    "Feature chunk identity differs")
            values = np.load(path, allow_pickle=False)
        else:
            values = np.stack([adapter.feature(core.model_input(row)) for row in subset])
            binary_save(path, lambda stream: np.save(stream, values, allow_pickle=False))
            save(path.with_suffix(".receipt.json"), dict(stamp, sha256=sha(path)), immutable=True)
            newly_done += len(subset)
        require(values.shape[0] == len(subset) and np.isfinite(values).all(), "Feature chunk invalid")
        result.append(values)
        hashes.append(sha(path))
        progress(adapter.method, "features_" + split, completed=start+len(subset), total=len(rows),
                 elapsed_seconds=time.time()-started,
                 new_patients_per_second=newly_done / max(time.time()-started, .001))
    array = np.concatenate(result)
    save(folder / "READY.json", dict(chunks=hashes, examples_sha256=core.digest(rows),
         shape=list(array.shape), manifest_sha256=sha(ROOT / "manifest.json")), immutable=True)
    return torch.from_numpy(array)


def labels(rows):
    return (torch.tensor([r["duration"] for r in rows], dtype=torch.float64),
            torch.tensor([r["event"] for r in rows], dtype=torch.bool),
            torch.tensor([r["loss_eligible"] for r in rows], dtype=torch.bool))


class Trainer:
    def __init__(self, method, train_x, train_rows, selection_x, selection_rows, metadata):
        self.method, self.x, self.y = method, train_x, labels(train_rows)
        self.vx, self.vy = selection_x, labels(selection_rows)
        self.directory = RUN / method
        self.rng = np.random.default_rng(P["seed"])
        random.seed(P["seed"])
        np.random.seed(P["seed"])
        torch.manual_seed(P["seed"])
        self.model = TimeHead(self.x, self.y[0], self.y[1])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=P["training"]["learning_rate"],
                                          weight_decay=P["training"]["weight_decay"])
        self.identity = dict(method=method, manifest_sha256=sha(ROOT / "manifest.json"),
            train_sha256=core.digest(train_rows), selection_sha256=core.digest(selection_rows),
            train_features_sha256=sha(self.directory / "features/train/READY.json"),
            selection_features_sha256=sha(self.directory / "features/selection/READY.json"),
            backbone=metadata, torch_version=torch.__version__)
        save(self.directory / "identity.json", self.identity, immutable=True)

    def update(self, step):
        ids = self.rng.integers(len(self.x), size=P["training"]["batch_size"])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        mu, sigma = self.model(self.x[ids])
        loss = nll(mu, sigma, *(t[ids] for t in self.y))
        require(bool(torch.isfinite(loss)), "Nonfinite training likelihood")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), P["training"]["gradient_clip"],
                                       error_if_nonfinite=True)
        self.optimizer.step()
        return float(loss)

    def validate(self):
        self.model.eval()
        with torch.inference_mode():
            value = float(nll(*self.model(self.vx), *self.vy))
        require(math.isfinite(value), "Nonfinite selection likelihood")
        return value

    def state(self, step, validation):
        return dict(identity=self.identity, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    step=step, validation_nll=validation, data_rng=self.rng.bit_generator.state,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state())

    def load(self, state):
        require(state["identity"] == self.identity, "Time checkpoint identity changed")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.bit_generator.state = state["data_rng"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        return state["step"]

    def commit(self, step, val):
        path = self.directory / "checkpoints" / f"step_{step:06d}.pt"
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=False)
            require(old["identity"] == self.identity and old["step"] == step,
                    "Orphan checkpoint identity mismatch")
            require(all(torch.equal(v, old["model"][k]) for k, v in self.model.state_dict().items()),
                    "Replayed checkpoint differs")
        else:
            binary_save(path, lambda stream: torch.save(self.state(step, val), stream))
        receipt = dict(path=str(path), sha256=sha(path), step=step, validation_nll=val)
        save(path.with_suffix(".receipt.json"), receipt, immutable=True)
        self.pointers()

    def pointers(self):
        receipts = [read(p) for p in sorted((self.directory / "checkpoints").glob("step_*.receipt.json"))]
        for r in receipts:
            require(sha(r["path"]) == r["sha256"], "Time checkpoint changed")
        if receipts:
            save(self.directory / "last.json", receipts[-1])
        candidates = [r for r in receipts if r["step"] >= P["training"]["checkpoint_interval"]]
        if candidates:
            save(self.directory / "best.json", min(candidates, key=lambda r: (r["validation_nll"], r["step"])))
        return receipts[-1] if receipts else None

    def fit(self):
        receipt = self.pointers()
        if receipt:
            start = self.load(torch.load(receipt["path"], map_location="cpu", weights_only=False))
            require(read(self.directory / "recovery_gate.json").get("passed"), "Missing resume gate")
        else:
            gate(self)
            start = 0
            self.commit(0, self.validate())
        for step in range(start+1, P["training"]["steps"]+1):
            loss = self.update(step)
            if step == 1 or step % P["training"]["checkpoint_interval"] == 0:
                val = self.validate()
                self.commit(step, val)
                progress(self.method, "training_head", step=step, total=P["training"]["steps"],
                         training_nll=loss, selection_nll=val, best=read(self.directory / "best.json"))
        best = read(self.directory / "best.json")
        self.load(torch.load(best["path"], map_location="cpu", weights_only=False))
        return best


def gate(trainer):
    buffer = io.BytesIO()
    torch.save(trainer.state(0, trainer.validate()), buffer)
    first = trainer.update(1)
    expected = copy.deepcopy(trainer.model.state_dict())
    data_rng = copy.deepcopy(trainer.rng.bit_generator.state)
    buffer.seek(0)
    trainer.load(torch.load(buffer, map_location="cpu", weights_only=False))
    second = trainer.update(1)
    require(first == second and trainer.rng.bit_generator.state == data_rng and
            all(torch.equal(v, expected[k]) for k, v in trainer.model.state_dict().items()),
            "Full-state update replay differs")
    buffer.seek(0)
    trainer.load(torch.load(buffer, map_location="cpu", weights_only=False))
    save(trainer.directory / "recovery_gate.json", dict(passed=True, exact_update_replay=True,
        manifest_sha256=sha(ROOT / "manifest.json"), restored_to_step=0), immutable=True)


def evaluate(trainer, adapter, split, baseline):
    rows = examples_for(split, trainer.method)
    x = features(adapter, split, rows)
    best = read(trainer.directory / "best.json")
    trainer.model.eval()
    with torch.inference_mode():
        mu, sigma = trainer.model(x)
    outputs = []
    for ordinal, row in enumerate(rows):
        mean, scale = float(mu[ordinal]), float(sigma[ordinal])
        normals = core.normal_draws(P["evaluation"]["sampling_seed"], split,
                                   row["patient_index"], row["cut_index"], P["evaluation"]["rollouts"])
        output = dict(ordinal=ordinal, patient_index=row["patient_index"], cut_index=row["cut_index"],
            source_sha256=core.digest(row), event=row["event"], duration=row["duration"],
            mu=mean, sigma=scale, samples=core.predictions(mean, scale, normals, P["horizon_days"]),
            median=core.median_time(mean), checkpoint_sha256=best["sha256"])
        path = trainer.directory / split / f"case_{ordinal:06d}.json"
        save(path, output, immutable=True)
        save(path.with_suffix(".receipt.json"), dict(sha256=sha(path)), immutable=True)
        outputs.append(output)
    summary = dict(method=trainer.method, split=split, checkpoint=best,
        metrics=core.summarize(outputs, baseline, P["evaluation"]["rollouts"]),
        manifest_sha256=sha(ROOT / "manifest.json"), native_clock=False,
        independent_time_adaptation=True, shared_trained_parameters=False, frozen_backbone=True,
        feature_sha256=sha(trainer.directory / "features" / split / "READY.json"),
        patient_content_exported=False)
    save(trainer.directory / (split + "_summary.json"), summary, immutable=True)
    progress(trainer.method, "verifying_" + split)
    import verify
    verify.check(trainer.method, split)


def run(method):
    os.umask(0o077)
    os.nice(max(0, P["resources"]["nice_increment"] - os.getpriority(os.PRIO_PROCESS, 0)))
    verify_sources(method)
    directory = RUN / method
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress(method, "loading_backbone")
        adapter = Adapter(method)
        model_gate = adapter.gate()
        # Elapsed timing is diagnostic, outside the immutable computational gate.
        model_gate.pop("seconds")
        save(directory / "backbone_gate.json", model_gate, immutable=True)
        train = examples_for("train")
        selection = examples_for("selection")
        train_x = features(adapter, "train", train)
        selection_x = features(adapter, "selection", selection)
        trainer = Trainer(method, train_x, train, selection_x, selection, adapter.metadata)
        trainer.fit()
        baseline = float(np.median([r["duration"] for r in train if r["event"]]))
        for split in ("val", "test"):
            evaluate(trainer, adapter, split, baseline)
        verify_sources(method)
        progress(method, "complete", selected_checkpoint=read(directory / "best.json"),
                 validation_accepted=True, test_accepted=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=P["methods"])
    args = parser.parse_args()
    try:
        run(args.method)
    except BaseException as error:
        save(OPS / (args.method + ".failure.json"),
             dict(error=type(error).__name__ + ": " + str(error), failed_unix=time.time()))
        traceback.print_exc()
        raise
