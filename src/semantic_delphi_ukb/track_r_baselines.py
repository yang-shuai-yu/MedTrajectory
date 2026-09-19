from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from semantic_delphi_ukb.evaluate_medical_control_tasks import _sanitize_json, _summary
from semantic_delphi_ukb.track_r_rows import write_json_gzip


MODEL_DISPLAY_NAMES = {
    "logistic": "Logistic-R",
    "cox": "Cox-R",
    "mdrmf-clinical": "MDRMF-Clinical-R",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit Track R Logistic, Cox, or MDRMF-Clinical-R baselines.")
    p.add_argument("--model", choices=("logistic", "cox", "mdrmf-clinical"), required=True)
    p.add_argument("--train-features", type=Path, required=True)
    p.add_argument("--val-features", type=Path, required=True)
    p.add_argument("--eval-features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cpu")
    p.add_argument("--calibration-bins", type=int, default=10)
    return p


def atomic_write_json(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("lifelines", "numpy", "pandas", "scikit-learn", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def standardize(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(0)
    std = train.std(0)
    std[std == 0] = 1.0
    return (train - mean) / std, *((value - mean) / std for value in others), mean, std


def standardize_for_cox(
    train: np.ndarray,
    *others: np.ndarray,
    minimum_scale: float = 0.05,
    clip: float = 20.0,
):
    train64 = np.asarray(train, dtype=np.float64)
    mean = train64.mean(0)
    std = np.maximum(train64.std(0), minimum_scale)

    def transform(value: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(value, dtype=np.float64) - mean) / std
        return np.clip(scaled, -clip, clip)

    return transform(train64), *(transform(value) for value in others), mean, std


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from semantic_delphi_ukb.horizon_control_metrics import binary_auc
    return binary_auc(scores, labels)


def fit_logistic(train, val, evaluate):
    from sklearn.linear_model import LogisticRegression

    train_x, val_x, eval_x, mean, std = standardize(train["x"], val["x"], evaluate["x"])
    shape = train["labels"].shape[1:]
    predictions = np.zeros((len(eval_x), *shape), dtype=np.float32)
    selected = {}
    fitted = {}
    for horizon in range(shape[0]):
        for disease in range(shape[1]):
            train_mask = train["label_mask"][:, horizon, disease].astype(bool)
            val_mask = val["label_mask"][:, horizon, disease].astype(bool)
            y = train["labels"][train_mask, horizon, disease].astype(np.int8)
            if len(y) == 0 or np.unique(y).size < 2:
                prior = float(y.mean()) if len(y) else 0.0
                predictions[:, horizon, disease] = prior
                selected[f"{horizon}:{disease}"] = {"constant_prior": prior, "val_auc": None}
                fitted[f"{horizon}:{disease}"] = {"constant_prior": prior}
                continue
            best = (-np.inf, None, None)
            for c in (0.01, 0.1, 1.0, 10.0):
                model = LogisticRegression(C=c, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42)
                model.fit(train_x[train_mask], y)
                score = model.predict_proba(val_x[val_mask])[:, 1]
                metric = auc(score, val["labels"][val_mask, horizon, disease])
                if np.isfinite(metric) and metric > best[0]:
                    best = (metric, c, model)
            if best[2] is None:
                predictions[:, horizon, disease] = float(y.mean()) if len(y) else 0.0
            else:
                predictions[:, horizon, disease] = best[2].predict_proba(eval_x)[:, 1]
                fitted[f"{horizon}:{disease}"] = best[2]
            selected[f"{horizon}:{disease}"] = {"C": best[1], "val_auc": best[0]}
    checkpoint = {"models": fitted, "train_mean": mean, "train_std": std, "shape": shape}
    return predictions, {"selected": selected, "train_mean": mean.tolist(), "train_std": std.tolist()}, checkpoint


def fit_cox(train, val, evaluate):
    try:
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceError
        from lifelines.utils import concordance_index
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Cox-R requires lifelines and pandas in the remote training environment") from exc
    train_x, val_x, eval_x, mean, std = standardize_for_cox(
        train["x"], val["x"], evaluate["x"]
    )
    diseases = train["survival_duration"].shape[1]
    horizons = train["labels"].shape[1]
    predictions = np.zeros((len(eval_x), horizons, diseases), dtype=np.float32)
    selected = {}
    fitted = {}
    columns = [f"x{index}" for index in range(train_x.shape[1])]
    val_frame = pd.DataFrame(val_x, columns=columns)
    eval_frame = pd.DataFrame(eval_x, columns=columns)
    for disease in range(diseases):
        best = (-np.inf, None, None)
        failures = []
        frame = pd.DataFrame(train_x, columns=columns)
        frame["duration"] = train["survival_duration"][:, disease]
        frame["event"] = train["survival_event"][:, disease]
        for penalizer in (0.1, 1.0, 10.0):
            model = CoxPHFitter(penalizer=penalizer)
            try:
                model.fit(
                    frame,
                    duration_col="duration",
                    event_col="event",
                    show_progress=False,
                    batch_mode=True,
                    fit_options={"step_size": 0.5, "max_steps": 100},
                )
            except (ConvergenceError, np.linalg.LinAlgError, ValueError) as exc:
                failures.append({
                    "penalizer": penalizer,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                continue
            risk = model.predict_partial_hazard(val_frame).to_numpy()
            metric = concordance_index(
                val["survival_duration"][:, disease],
                -risk,
                event_observed=val["survival_event"][:, disease],
            )
            if metric > best[0]:
                best = (metric, penalizer, model)
        model = best[2]
        if model is None:
            raise RuntimeError(f"Cox-R failed every penalizer for disease {disease}: {failures}")
        survival = model.predict_survival_function(eval_frame, times=[1.0, 5.0, 10.0]).to_numpy().T
        predictions[:, :, disease] = 1.0 - survival
        selected[str(disease)] = {
            "penalizer": best[1],
            "val_concordance": best[0],
            "failed_candidates": failures,
        }
        fitted[str(disease)] = model
    cox_transform = {"dtype": "float64", "minimum_scale": 0.05, "clip": 20.0}
    checkpoint = {
        "models": fitted,
        "train_mean": mean,
        "train_std": std,
        "horizons": [1.0, 5.0, 10.0],
        "standardization": cox_transform,
    }
    return predictions, {
        "selected": selected,
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "standardization": cox_transform,
    }, checkpoint


def fit_mdrmf(
    train,
    val,
    evaluate,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    metric_callback=None,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_x, val_x, eval_x, mean, std = standardize(train["x"], val["x"], evaluate["x"])
    output_dim = int(np.prod(train["labels"].shape[1:]))
    tx = torch.from_numpy(train_x).float().to(device)
    ty = torch.from_numpy(train["labels"].reshape(len(train_x), -1)).float().to(device)
    tm = torch.from_numpy(train["label_mask"].reshape(len(train_x), -1)).float().to(device)
    vx = torch.from_numpy(val_x).float().to(device)
    vy = torch.from_numpy(val["labels"].reshape(len(val_x), -1)).float().to(device)
    vm = torch.from_numpy(val["label_mask"].reshape(len(val_x), -1)).float().to(device)
    candidates = ((1, 50), (2, 100), (3, 100), (3, 200))
    best_loss, best_state, best_spec, best_history = float("inf"), None, None, None
    for depth, width in candidates:
        layers = []
        input_dim = train_x.shape[1]
        for _ in range(depth):
            layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, output_dim))
        candidate = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-4, weight_decay=0.0)
        candidate_best, candidate_state, remaining = float("inf"), None, patience
        history = []
        for epoch in range(max_epochs):
            candidate.train()
            epoch_started = time.monotonic()
            order = torch.randperm(len(tx), device=device)
            train_numerator = train_denominator = 0.0
            for start in range(0, len(tx), batch_size):
                indices = order[start : start + batch_size]
                optimizer.zero_grad()
                raw = F.binary_cross_entropy_with_logits(candidate(tx[indices]), ty[indices], reduction="none")
                denominator = tm[indices].sum().clamp_min(1.0)
                loss = (raw * tm[indices]).sum() / denominator
                loss.backward()
                optimizer.step()
                train_numerator += float((raw.detach() * tm[indices]).sum().item())
                train_denominator += float(denominator.item())
            candidate.eval()
            with torch.no_grad():
                val_numerator = val_denominator = 0.0
                for start in range(0, len(vx), batch_size):
                    raw = F.binary_cross_entropy_with_logits(
                        candidate(vx[start : start + batch_size]), vy[start : start + batch_size], reduction="none"
                    )
                    current_mask = vm[start : start + batch_size]
                    val_numerator += float((raw * current_mask).sum().item())
                    val_denominator += float(current_mask.sum().clamp_min(1.0).item())
                val_loss = val_numerator / max(val_denominator, 1.0)
            history.append({"epoch": epoch, "train_loss": train_numerator / max(train_denominator, 1.0), "val_loss": val_loss})
            if metric_callback is not None:
                metric_callback({
                    "candidate_depth": depth,
                    "candidate_width": width,
                    "epoch": epoch,
                    "train_loss": history[-1]["train_loss"],
                    "val_loss": val_loss,
                    "epoch_seconds": time.monotonic() - epoch_started,
                })
            if val_loss < candidate_best:
                candidate_best = val_loss
                candidate_state = {name: value.detach().cpu().clone() for name, value in candidate.state_dict().items()}
                remaining = patience
            else:
                remaining -= 1
                if remaining == 0:
                    break
        if candidate_best < best_loss:
            best_loss, best_state, best_spec, best_history = candidate_best, candidate_state, (depth, width), history
    depth, width = best_spec
    layers = []
    input_dim = train_x.shape[1]
    for _ in range(depth):
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    model = nn.Sequential(*layers).to(device)
    model.load_state_dict(best_state)
    with torch.no_grad():
        prediction = np.concatenate([
            torch.sigmoid(model(torch.from_numpy(eval_x[start : start + batch_size]).float().to(device))).cpu().numpy()
            for start in range(0, len(eval_x), batch_size)
        ])
    prediction = prediction.reshape(len(eval_x), *train["labels"].shape[1:])
    checkpoint = {
        "state_dict": best_state, "depth": depth, "width": width,
        "input_dim": train_x.shape[1], "output_dim": output_dim,
        "train_mean": mean, "train_std": std,
        "output_shape": train["labels"].shape[1:],
    }
    details = {"best_val_loss": best_loss, "selected_depth": depth, "selected_width": width, "history": best_history, "train_mean": mean.tolist(), "train_std": std.tolist()}
    return prediction, details, checkpoint


def output_rows(prediction, evaluate, schema, model_name):
    rows = []
    for row in range(len(prediction)):
        for horizon, horizon_years in enumerate(schema["horizons_years"]):
            for disease, disease_id in enumerate(schema["disease_ids"]):
                if not evaluate["label_mask"][row, horizon, disease]:
                    continue
                rows.append({
                    "patient_index": int(evaluate["patient_index"][row]),
                    "disease_id": disease_id,
                    "horizon_years": float(horizon_years),
                    "sex": str(evaluate["sex"][row]),
                    "age_start_years": float(evaluate["age_start_years"][row]),
                    "prediction_age_days": float(evaluate["prediction_age_days"][row]),
                    "label": int(evaluate["labels"][row, horizon, disease]),
                    "score": float(prediction[row, horizon, disease]),
                    "score_is_probability": True,
                    "model": model_name,
                })
    return rows


def predict_from_checkpoint(model_name: str, checkpoint_path: Path, evaluate: dict[str, np.ndarray]) -> np.ndarray:
    if model_name == "mdrmf-clinical":
        import torch
        import torch.nn as nn

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        layers = []
        input_dim = int(checkpoint["input_dim"])
        for _ in range(int(checkpoint["depth"])):
            layers.extend((nn.Linear(input_dim, int(checkpoint["width"])), nn.ReLU()))
            input_dim = int(checkpoint["width"])
        layers.append(nn.Linear(input_dim, int(checkpoint["output_dim"])))
        model = nn.Sequential(*layers)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        x = (evaluate["x"] - checkpoint["train_mean"]) / checkpoint["train_std"]
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.from_numpy(x).float())).numpy()
        return scores.reshape(len(x), *checkpoint["output_shape"])
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    x = (evaluate["x"] - checkpoint["train_mean"]) / checkpoint["train_std"]
    if model_name == "cox":
        x = np.clip(
            np.asarray(x, dtype=np.float64),
            -float(checkpoint.get("standardization", {}).get("clip", 20.0)),
            float(checkpoint.get("standardization", {}).get("clip", 20.0)),
        )
    shape = evaluate["labels"].shape[1:]
    prediction = np.zeros((len(x), *shape), dtype=np.float32)
    if model_name == "logistic":
        for horizon in range(shape[0]):
            for disease in range(shape[1]):
                model = checkpoint["models"].get(f"{horizon}:{disease}")
                if isinstance(model, dict) and "constant_prior" in model:
                    prediction[:, horizon, disease] = float(model["constant_prior"])
                else:
                    prediction[:, horizon, disease] = model.predict_proba(x)[:, 1] if model is not None else 0.0
    else:
        import pandas as pd

        columns = [f"x{index}" for index in range(x.shape[1])]
        frame = pd.DataFrame(x, columns=columns)
        for disease in range(shape[1]):
            survival = checkpoint["models"][str(disease)].predict_survival_function(
                frame, times=checkpoint["horizons"]
            ).to_numpy().T
            prediction[:, :, disease] = 1.0 - survival
    return prediction


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir / "status.json"
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.time()
    resolved_args = {
        name: str(value.resolve()) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    atomic_write_json(args.output_dir / "run_config.json", {
        "arguments": resolved_args,
        "package_versions": package_versions(),
    })
    atomic_write_json(status_path, {
        "status": "running",
        "model": MODEL_DISPLAY_NAMES[args.model],
        "started_at_unix": started,
    })

    def record_metric(metric):
        append_jsonl(metrics_path, metric)
        atomic_write_json(status_path, {
            "status": "running",
            "model": MODEL_DISPLAY_NAMES[args.model],
            "started_at_unix": started,
            "latest_metric": metric,
        })

    try:
        evaluate = load(args.eval_features)
        schema = json.loads(args.eval_features.with_suffix(".schema.json").read_text(encoding="utf-8-sig"))
        if args.predict_only:
            if args.checkpoint is None:
                raise ValueError("--predict-only requires --checkpoint")
            prediction = predict_from_checkpoint(args.model, args.checkpoint, evaluate)
            details = {"predict_only": True, "checkpoint": str(args.checkpoint.resolve())}
            checkpoint = None
        else:
            train, val = load(args.train_features), load(args.val_features)
            if args.model == "logistic":
                prediction, details, checkpoint = fit_logistic(train, val, evaluate)
            elif args.model == "cox":
                prediction, details, checkpoint = fit_cox(train, val, evaluate)
            else:
                prediction, details, checkpoint = fit_mdrmf(
                    train,
                    val,
                    evaluate,
                    args.seed,
                    args.max_epochs,
                    args.patience,
                    args.batch_size,
                    args.device,
                    record_metric,
                )
            details["validation_reporting_policy"] = {
                "validation_is_used_for_selection_and_descriptive_reporting": True,
                "validation_metrics_are_selection_biased_descriptive": True,
                "test_is_independent_of_selection": True,
            }
        details["package_versions"] = package_versions()
        rows = output_rows(prediction, evaluate, schema, MODEL_DISPLAY_NAMES[args.model])
        rows_path = write_json_gzip(args.output_dir / "rows.json.gz", rows)
        summary = _summary(rows, MODEL_DISPLAY_NAMES[args.model], args.calibration_bins)
        summary_payload = {
            "protocol": schema["protocol_id"],
            "split": schema["split"],
            "family": "track_r_baseline",
            "model": MODEL_DISPLAY_NAMES[args.model],
            "validation_rows": str(rows_path.resolve()),
            "validation_rows_format": "json.gz",
            "patient_rows": len(rows),
            "summary": summary,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(_sanitize_json(summary_payload), indent=2, allow_nan=False), encoding="utf-8"
        )
        (args.output_dir / "training_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        if not args.predict_only:
            if args.model == "mdrmf-clinical":
                import torch

                torch.save(checkpoint, args.output_dir / "checkpoint.pt")
            else:
                with (args.output_dir / "checkpoint.pkl").open("wb") as handle:
                    pickle.dump(checkpoint, handle)
        finished = {
            "status": "finished",
            "model": MODEL_DISPLAY_NAMES[args.model],
            "rows": len(rows),
            "rows_path": str(rows_path.resolve()),
            "elapsed_seconds": time.time() - started,
            "macro": _sanitize_json(summary["macro"]),
        }
        atomic_write_json(status_path, finished)
        print(json.dumps(_sanitize_json(finished)))
    except Exception as exc:
        atomic_write_json(status_path, {
            "status": "failed",
            "model": MODEL_DISPLAY_NAMES[args.model],
            "elapsed_seconds": time.time() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
