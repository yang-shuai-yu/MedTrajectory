"""Independently reconstruct source targets and saved prediction arithmetic."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from assets import OPS, P, ROOT, RUN, cohort, examples_for, read, records, require, save, sha
from head import TimeHead


def close(a, b):
    return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def feature_array(method, split):
    folder = RUN / method / "features" / split
    manifest = read(folder / "READY.json")
    require(manifest["manifest_sha256"] == sha(ROOT / "manifest.json"), "Feature release differs")
    files = sorted(folder.glob("chunk_*.npy"))
    require([sha(p) for p in files] == manifest["chunks"], "Feature inventory changed")
    return torch.from_numpy(np.concatenate([np.load(p, allow_pickle=False) for p in files]))


def check(method, split):
    directory = RUN / method
    summary = read(directory / (split + "_summary.json"))
    checkpoint = read(directory / "best.json")
    require(checkpoint == summary["checkpoint"] and sha(checkpoint["path"]) == checkpoint["sha256"],
            "Evaluation checkpoint changed")
    require(read(directory / "recovery_gate.json").get("passed") and
            read(directory / "backbone_gate.json").get("passed"), "Missing acceptance gates")
    train = examples_for("train")
    selection = examples_for("selection")
    require(not {r["patient_index"] for r in selection} &
            {c["patient_index"] for c in cohort("val")}, "Selection leaks evaluation patients")
    baseline = statistics.median([r["duration"] for r in train if r["event"]])
    train_x = feature_array(method, "train")
    head = TimeHead(train_x, torch.tensor([r["duration"] for r in train]),
                    torch.tensor([r["event"] for r in train]))
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    require(state["identity"] == read(directory / "identity.json"), "Head identity differs")
    head.load_state_dict(state["model"], strict=True)
    head.eval()
    with torch.inference_mode():
        predicted_mu, predicted_sigma = head(feature_array(method, split))
    cases = cohort(split, method)
    prepared = examples_for(split, method)
    positions = {c["patient_index"]: (i, c["cut_index"]) for i, c in enumerate(cases)}
    truth = {}
    for patient, tokens, ages, followup, diagnostic in records(split, method):
        if patient not in positions:
            continue
        ordinal, cut = positions[patient]
        first = next((ages[k]-ages[cut] for k in range(cut+1, len(tokens))
                      if tokens[k] in diagnostic and ages[k] <= ages[cut]+P["horizon_days"]), None)
        exposure = max(0., min(P["horizon_days"], followup-ages[cut]))
        source = prepared[ordinal]
        require(source["history"] == [diagnostic[t] for t in tokens[:cut+1] if t in diagnostic] and
                source["history_ages"] == [a for t, a in zip(tokens[:cut+1], ages[:cut+1]) if t in diagnostic] and
                source["cut_age"] == ages[cut], "Prepared model prefix differs")
        require(source["event"] == (first is not None) and
                source["duration"] == (first if first is not None else exposure) and
                source["administrative_exposure"] == exposure, "Prepared censor/target differs")
        truth[ordinal] = (first, exposure)
    require(len(truth) == len(cases), "Independent truth incomplete")
    mae, deterministic, baseline_paired, baseline_all, available = [], [], [], [], []
    chain = hashlib.sha256()
    normal = statistics.NormalDist()
    for ordinal, case in enumerate(cases):
        path = directory / split / f"case_{ordinal:06d}.json"
        require(sha(path) == read(path.with_suffix(".receipt.json"))["sha256"], "Saved prediction changed")
        row = read(path)
        require((row["ordinal"], row["patient_index"], row["cut_index"]) ==
                (ordinal, case["patient_index"], case["cut_index"]), "Prediction patient differs")
        source_digest = hashlib.sha256(json.dumps(prepared[ordinal], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(row["source_sha256"] == source_digest and row["checkpoint_sha256"] == checkpoint["sha256"],
                "Prediction source differs")
        target, exposure = truth[ordinal]
        require(row["event"] == (target is not None) and row["duration"] == (target if target is not None else exposure),
                "Prediction target differs")
        require(row["mu"] == float(predicted_mu[ordinal]) and
                row["sigma"] == float(predicted_sigma[ordinal]), "Saved output differs from learned head")
        require(close(row["median"], max(0., math.expm1(row["mu"]))), "Median output differs")
        require(len(row["samples"]) == P["evaluation"]["rollouts"], "Incomplete prediction draws")
        valid = []
        for roll, saved in enumerate(row["samples"]):
            payload = [P["evaluation"]["sampling_seed"], split, case["patient_index"], case["cut_index"],
                       roll, "first_diagnosis_time"]
            raw = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).digest()
            bits = int.from_bytes(raw[:8], "big") >> 11
            uniform = min(1-2**-53, (bits+.5)/2**53)
            z = row["mu"]+row["sigma"]*normal.inv_cdf(uniform)
            expected = None if z > math.log1p(P["horizon_days"]) else max(0., math.expm1(z))
            require(close(saved, expected), "Predicted draw differs from learned distribution")
            if saved is not None and target is not None:
                valid.append(abs(saved-target))
        available.append(len(valid) / P["evaluation"]["rollouts"])
        if valid:
            mae.append(statistics.mean(valid))
            baseline_paired.append(abs(baseline-target))
        if target is not None:
            deterministic.append(abs(row["median"]-target))
            baseline_all.append(abs(baseline-target))
        chain.update((path.name + ":" + sha(path) + "\n").encode())
    mean = lambda values: statistics.mean(values) if values else None
    expected = dict(patients=len(cases), observed_target_patients=len(deterministic),
        paired_patients=len(mae), sampled_time_mae_days=mean(mae),
        paired_rollout_availability=mean(available), deterministic_median_mae_days=mean(deterministic),
        training_observed_median_days=baseline, baseline_mae_same_paired_patients=mean(baseline_paired),
        baseline_mae_all_observed_patients=mean(baseline_all))
    require(all(close(v, summary["metrics"][k]) for k, v in expected.items()), "Time aggregate differs")
    require(len(list((directory / split).glob("case_*.json"))) == 2*len(cases), "Unexpected prediction inventory")
    accepted = dict(accepted=True, scope="Time adaptation integration and arithmetic, not superiority",
        method=method, split=split, checkpoint_sha256=checkpoint["sha256"],
        summary_sha256=sha(directory / (split + "_summary.json")),
        private_cases_sha256=chain.hexdigest(), manifest_sha256=sha(ROOT / "manifest.json"),
        independent_target_censor_reconstruction=True, learned_head_outputs_rerun=True,
        original_backbone_features_independently_rerun=False,
        patient_content_exported=False, metrics=expected)
    save(directory / (split + "_acceptance.json"), accepted, immutable=True)
    return accepted
