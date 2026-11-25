"""Offline CPU inference for the released MedTrajectory A2 generation model."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def exact_keys(value, required, optional=(), label="object"):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ValueError(f"{label} requires fields: {', '.join(required)}")
    unknown = value.keys() - set(required) - set(optional)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")


def number(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return float(value)


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def validate_input(value, vocab):
    exact_keys(value, ("schema_version", "static", "history"), ("generation",), "input")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    static = value["static"]
    fields = ["anchor_age_days", *vocab["static_order"]]
    exact_keys(static, fields, label="static")
    anchor = number(static["anchor_age_days"], "static.anchor_age_days", 1, 150 * 365.25)
    prefix = []
    for field in vocab["static_order"]:
        if not isinstance(static[field], str):
            raise ValueError(f"static.{field} must be a category string")
        key = f"static:{field}:{static[field]}"
        if key not in vocab["static_token_ids"]:
            raise ValueError(f"Unknown static category: {key}")
        prefix.append(vocab["static_token_ids"][key])
    history = value["history"]
    if not isinstance(history, list) or not 1 <= len(history) <= 10000:
        raise ValueError("history must contain 1 to 10000 events")
    lookup = {code: index for index, code in enumerate(vocab["dynamic_labels"])}
    tokens, ages = [], []
    for i, event in enumerate(history):
        exact_keys(event, ("code", "age_days"), label=f"history[{i}]")
        code = event["code"]
        if not isinstance(code, str) or code not in lookup or lookup[code] < 2:
            raise ValueError(f"Unknown clinical code in history[{i}]; use vocabulary.json exactly")
        if code.startswith("death:"):
            raise ValueError("History contains a death event; forward generation is not applicable")
        age = number(event["age_days"], f"history[{i}].age_days", 0, 150 * 365.25)
        if ages and age < ages[-1]:
            raise ValueError("history must be ordered by nondecreasing age_days")
        tokens.append(lookup[code])
        ages.append(age)
    if anchor > ages[-1]:
        raise ValueError("Static measurement anchor cannot be after the prediction origin")
    options = value.get("generation", {})
    exact_keys(options, (), ("seed", "num_trajectories", "max_new_events", "horizon_days"), "generation")
    settings = {
        "seed": integer(options.get("seed", 20260819), "seed", 0, 2**63 - 1),
        "num_trajectories": integer(options.get("num_trajectories", 1), "num_trajectories", 1, 20),
        "max_new_events": integer(options.get("max_new_events", 30), "max_new_events", 1, 256),
        "horizon_days": number(options.get("horizon_days", 3652.5), "horizon_days", 0.000001, 3652.5),
    }
    return prefix, anchor, tokens, ages, settings


def load_bundle(directory=ROOT / "model"):
    import torch
    from semantic_delphi_ukb.car_rope_model import CARoPEConfig, CARoPEHorizonMedTrajectory

    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["format_version"] != 1 or set(manifest["files"]) != {"model.pt", "config.json", "vocabulary.json"}:
        raise ValueError("Unsupported model bundle manifest")
    for name, identity in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != identity["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError(f"Model bundle integrity check failed: {name}")
    config = read_json(directory / "config.json")
    vocab = read_json(directory / "vocabulary.json")
    if (len(vocab["dynamic_labels"]) != 8424 or config["vocab_size"] != 8445
            or vocab["bos_token_id"] != 8443 or vocab["mask_token_id"] != 8444
            or config["block_size"] != manifest["dynamic_context_length"] + 5):
        raise ValueError("Incompatible model/vocabulary/context contract")
    with contextlib.redirect_stdout(sys.stderr):
        model = CARoPEHorizonMedTrajectory(CARoPEConfig(**config))
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, vocab, manifest


def run(value, bundle):
    import torch
    from semantic_delphi_ukb.track_g_generation import (
        clinical_candidate_mask, clinical_event_type, sample_event_and_wait_with_diagnostics,
    )

    model, vocab, manifest = bundle
    prefix, anchor, input_tokens, input_ages, settings = validate_input(value, vocab)
    labels = vocab["dynamic_labels"]
    dynamic_length = manifest["dynamic_context_length"]
    history_tokens, history_ages = input_tokens[-dynamic_length:], input_ages[-dynamic_length:]
    origin = history_ages[-1]
    limit = origin + settings["horizon_days"]
    start_tokens = prefix + [vocab["bos_token_id"]] + history_tokens
    start_ages = [anchor] * len(prefix) + [history_ages[0]] + history_ages
    candidate_mask = clinical_candidate_mask(labels, "cpu")
    death_mask = torch.tensor([label.startswith("death:") for label in labels])
    generator = torch.Generator(device="cpu").manual_seed(settings["seed"])
    uniforms = torch.rand((settings["num_trajectories"], settings["max_new_events"], 2), generator=generator)
    sampler = manifest["sampler"]
    trajectories = []
    with torch.inference_mode():
        for rollout in range(settings["num_trajectories"]):
            tokens = torch.tensor([start_tokens], dtype=torch.long)
            ages = torch.tensor([start_ages], dtype=torch.float32)
            events = []
            reason = "event_limit"
            for step in range(settings["max_new_events"]):
                static_mask = torch.zeros_like(tokens, dtype=torch.bool)
                static_mask[:, :len(prefix)] = True
                bos_mask = torch.zeros_like(tokens, dtype=torch.bool)
                bos_mask[:, len(prefix)] = True
                logits = model(tokens, ages, None, static_token_mask=static_mask, bos_token_mask=bos_mask)[0]
                logits = logits[0, -1, :len(labels)]
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Model returned nonfinite logits")
                token, wait, clamped = sample_event_and_wait_with_diagnostics(
                    logits, candidate_mask=candidate_mask, ignore_tokens=model.config.ignore_tokens,
                    t_min=model.config.t_min, temperature=sampler["temperature"], top_p=sampler["top_p"],
                    death_token_mask=death_mask, death_logit_bias=sampler["death_logit_bias"],
                    minimum_wait_days=manifest["minimum_wait_days"],
                    event_uniform=uniforms[rollout, step, 0], wait_uniform=uniforms[rollout, step, 1],
                )
                previous_age = float(ages[0, -1])
                next_age = previous_age + wait
                if next_age > limit:
                    reason = "horizon"
                    break
                event_type = clinical_event_type(labels[token])
                events.append({"code": labels[token], "token_id": token, "event_type": event_type,
                               "age_days": next_age, "age_years": next_age / 365.25,
                               "days_from_origin": next_age - origin, "waiting_days": wait,
                               "minimum_wait_applied": clamped})
                if event_type == "death":
                    reason = "death"
                    break
                tokens = torch.cat((tokens, torch.tensor([[token]], dtype=torch.long)), dim=1)
                ages = torch.cat((ages, torch.tensor([[next_age]], dtype=torch.float32)), dim=1)
                if tokens.size(1) > model.config.block_size:
                    keep = len(prefix) + 1
                    tokens = torch.cat((tokens[:, :keep], tokens[:, -dynamic_length:]), dim=1)
                    ages = torch.cat((ages[:, :keep], ages[:, -dynamic_length:]), dim=1)
            trajectories.append({"trajectory_index": rollout, "events": events,
                                 "event_count": len(events), "stop_reason": reason})
    notes = ["Generated samples are research outputs, not calibrated individual disease or mortality risks."]
    if len(input_tokens) < 8:
        notes.append("History is shorter than the eight-event minimum in the paper generation cohort.")
    return {
        "schema_version": 1, "task": "multitype_trajectory_generation", "trained_model": True,
        "model_id": manifest["model_id"], "weights_sha256": manifest["files"]["model.pt"]["sha256"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "runtime": {"torch": torch.__version__, "device": "cpu"},
        "prediction_origin_age_days": origin, "horizon_end_age_days": limit,
        "history_events_received": len(input_tokens), "history_events_used": len(history_tokens),
        "history_events_dropped": len(input_tokens) - len(history_tokens),
        "generation": settings, "sampler": sampler,
        "trajectories": trajectories, "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; choose a new file")
    value = read_json(args.input)
    vocab = read_json(args.model_dir / "vocabulary.json")
    validate_input(value, vocab)
    import torch
    torch.set_num_threads(1)
    result = run(value, load_bundle(args.model_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "model_id": result["model_id"],
                      "trajectories": len(result["trajectories"])}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
