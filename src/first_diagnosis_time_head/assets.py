"""Pinned private data and model identities, with an explicit test gate."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import core

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / "protocol.json").read_text())
CONFIG = json.loads((ROOT / "config.json").read_text()) if (ROOT / "config.json").exists() else {}
CODE = Path(os.environ.get("MEDTRAJECTORY_CODE_ROOT", str(ROOT)))
DATA = Path(os.environ.get("MEDTRAJECTORY_DATA_ROOT", str(ROOT / "data")))
RUN, OPS = DATA / "runs" / P["id"], CODE / "ops" / P["id"]
INPUT = DATA / "snapshots" / P["input_release"]
BASE = CODE / "releases/frozen_baseline_20261014_r1"
HASHES = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if stamp not in HASHES:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4194304), b""):
                h.update(block)
        HASHES[stamp] = h.hexdigest()
    return HASHES[stamp]


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def save(path, value, immutable=False):
    path = Path(path)
    require(path.resolve().is_relative_to(RUN) or path.resolve().is_relative_to(OPS),
            "Output outside this release")
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(read(path) == value, "Immutable output differs")
        return
    temp = path.with_name(path.name + "." + str(os.getpid()) + ".part")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_sources(method=None):
    manifest = read(ROOT / "manifest.json")
    for name, expected in manifest["files"].items():
        require(sha(ROOT / name) == expected, "Time release source changed: " + name)
    for path, expected in CONFIG["sources"].items():
        require(sha(path) == expected, "Accepted dependency changed")
    for key in ([method] if method else P["methods"]):
        item = CONFIG["models"][key]
        require(sha(item["checkpoint"]["path"]) == item["checkpoint"]["sha256"],
                "Accepted checkpoint changed")
        require(sha(item["training_identity_path"]) == item["training_identity_sha256"],
                "Accepted training identity changed")
    return sha(ROOT / "manifest.json")


def test_gate(method):
    require(method in P["methods"], "Test access needs a method")
    acceptance = read(RUN / method / "val_acceptance.json")
    best = read(RUN / method / "best.json")
    require(acceptance.get("accepted") and acceptance["checkpoint_sha256"] == best["sha256"],
            "Test requires accepted validation and a frozen selected checkpoint")
    require(acceptance["manifest_sha256"] == sha(ROOT / "manifest.json") and
            acceptance["summary_sha256"] == sha(RUN / method / "val_summary.json"),
            "Validation acceptance changed")
    return acceptance


def cohort(split, method=None):
    require(split in ("val", "test"), "Invalid evaluation split")
    if split == "test":
        test_gate(method)
    path = BASE / "results/track_g_v1/manifests" / (split + "_generation_cohort.json")
    require(sha(path) == CONFIG["cohorts"][split], "Frozen cohort changed")
    cases = read(path)["cases"]
    require(len(cases) == (1000 if split == "val" else 1669), "Cohort size changed")
    return cases


def records(split, method=None):
    import numpy as np
    require(split in ("train", "val", "test"), "Invalid split")
    if split == "test":
        test_gate(method)
    require(sha(INPUT / "READY.json") == P["snapshot_sha256"], "Snapshot changed")
    ready = read(INPUT / "READY.json")
    names = ["labels.csv", split + ".bin", split + "_patient_index.csv",
             split + "_followup_end_age_days.npy"]
    for name in names:
        require(sha(INPUT / name) == ready["files"][name]["sha256"], "Snapshot data changed")
    labels = (INPUT / "labels.csv").read_text().splitlines()
    diagnostic = {i: j for j, i in enumerate(i for i, v in enumerate(labels) if v.startswith("diag:"))}
    require(len(diagnostic) == 1123, "Diagnosis vocabulary changed")
    clinical = np.array([v.startswith(("diag:", "proc:", "cancer:", "death:")) for v in labels])
    raw = np.memmap(INPUT / (split + ".bin"), dtype="<u4", mode="r").reshape(-1, 3)
    starts = np.r_[0, np.flatnonzero(np.diff(raw[:, 0])) + 1]
    ends = np.r_[starts[1:], len(raw)]
    followup = np.load(INPUT / (split + "_followup_end_age_days.npy"), allow_pickle=False)
    with (INPUT / (split + "_patient_index.csv")).open() as stream:
        index = list(csv.DictReader(stream))
    require(len(index) == len(starts) == len(followup), "Patient alignment differs")
    for patient, (item, start, end) in enumerate(zip(index, starts, ends)):
        require(int(raw[start, 0]) == int(item["eid"]) and int(item["num_events"]) == end-start,
                "Patient source row differs")
        if int(item["model_eligible"]) != 1 or int(item["anchor_only"]) != 0:
            continue
        block = raw[start:end]
        block = block[np.argsort(block[:, 1], kind="stable")]
        block = block[clinical[block[:, 2]+1]]
        yield patient, [int(t)+1 for t in block[:, 2]], [float(a) for a in block[:, 1]], float(followup[patient]), diagnostic


def examples_for(split, method=None):
    path = RUN / "input" / (split + ".json")
    receipt = read(path.with_suffix(".receipt.json"))
    if split == "test":
        test_gate(method)
    if receipt:
        require(receipt["manifest_sha256"] == sha(ROOT / "manifest.json") and
                sha(path) == receipt["sha256"], "Prepared data identity differs")
        return read(path)
    evaluation = split in ("val", "test")
    source_split = "val" if split == "selection" else split
    cases = cohort(split, method) if evaluation else []
    excluded = {c["patient_index"] for c in cohort("val")} if split == "selection" else set()
    selected = {c["patient_index"]: c for c in cases}
    if evaluation:
        require(len(selected) == len(cases), "Repeated cohort patient")
    result, eligible = [], 0
    for patient, tokens, ages, followup, diagnostic in records(source_split, method):
        if evaluation and patient not in selected or patient in excluded:
            continue
        cut = selected[patient]["cut_index"] if evaluation else core.cut_position(ages, P["cut"], P["horizon_days"])
        if cut is None:
            continue
        row = core.example(tokens, ages, cut, followup, diagnostic, P["horizon_days"])
        if not evaluation and not row["loss_eligible"]:
            continue
        eligible += 1
        result.append(dict(row, patient_index=patient, cut_index=cut))
    if evaluation:
        positions = {c["patient_index"]: i for i, c in enumerate(cases)}
        result.sort(key=lambda r: positions[r["patient_index"]])
        require(len(result) == len(cases), "Missing evaluation patients")
    else:
        result.sort(key=lambda r: core.digest([P["seed"], split, r["patient_index"]]))
        result = result[:P["partition"][split + "_max"]]
        require(len(result) >= 1000, "Insufficient fitting patients")
    save(path, result, immutable=True)
    receipt = dict(sha256=sha(path), manifest_sha256=sha(ROOT / "manifest.json"),
                   eligible=eligible, selected=len(result), observed=sum(r["event"] for r in result),
                   zero_day=sum(r["event"] and r["duration"] == 0 for r in result),
                   censored=sum(not r["event"] for r in result),
                   no_exposure=sum(not r["loss_eligible"] for r in result),
                   selected_patient_keys_sha256=core.digest([[r["patient_index"], r["cut_index"]] for r in result]))
    save(path.with_suffix(".receipt.json"), receipt, immutable=True)
    return result
