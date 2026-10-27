"""Pure endpoint, prefix and metric contracts."""
import hashlib
import json
import math
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def cut_position(ages, settings, horizon):
    n = len(ages)
    if n < settings["min_history_events"] + settings["min_future_events"]:
        return None
    cut = min(max(settings["min_history_events"] - 1,
                  math.floor((n - 1) * settings["baseline_fraction"])),
              n - settings["min_future_events"] - 1)
    if sum(a <= ages[cut] + horizon for a in ages[cut+1:]) < settings["min_future_events"]:
        return None
    return cut


def example(tokens, ages, cut, followup, diagnostic, horizon):
    if len(tokens) != len(ages) or not 0 <= cut < len(tokens):
        raise ValueError("Invalid clinical cut")
    if not all(math.isfinite(a) and a >= 0 for a in ages) or not math.isfinite(followup):
        raise ValueError("Invalid source time")
    if any(a > b for a, b in zip(ages, ages[1:])):
        raise ValueError("Unordered source time")
    now = float(ages[cut])
    history = [diagnostic[t] for t in tokens[:cut+1] if t in diagnostic]
    history_ages = [float(a) for t, a in zip(tokens[:cut+1], ages[:cut+1]) if t in diagnostic]
    targets = [float(a - now) for t, a in zip(tokens[cut+1:], ages[cut+1:])
               if t in diagnostic and a <= now + horizon]
    observed = bool(targets)
    exposure = min(horizon, max(0., followup - now))
    return dict(history=history, history_ages=history_ages, cut_age=now,
                event=observed, duration=targets[0] if observed else exposure,
                loss_eligible=observed or exposure > 0,
                administrative_exposure=exposure)


def model_input(row):
    # Explicit allow-list prevents labels or censoring metadata reaching the backbone/head.
    return {key: row[key] for key in ("history", "history_ages", "cut_age")}


def covariates(row):
    ages = row["history_ages"]
    if len(ages) != len(row["history"]) or any(a > row["cut_age"] for a in ages):
        raise ValueError("Invalid observed prefix")
    gaps = [b-a for a, b in zip(ages, ages[1:])]
    return [math.log1p(len(ages)), row["cut_age"] / 36525.,
            math.log1p(row["cut_age"] - ages[-1]) if ages else 0.,
            math.log1p(ages[-1] - ages[0]) if ages else 0.,
            math.log1p(statistics.median(gaps)) if gaps else 0., float(not ages)]


def normal_draws(seed, split, patient, cut, count):
    normal = statistics.NormalDist()
    result = []
    for roll in range(count):
        bits = int(digest([seed, split, patient, cut, roll, "first_diagnosis_time"])[:16], 16) >> 11
        uniform = min(1 - 2**-53, (bits + .5) / 2**53)
        result.append(normal.inv_cdf(uniform))
    return result


def predictions(mu, sigma, normals, horizon):
    if not math.isfinite(mu) or not .05 <= sigma <= 5.000001:
        raise ValueError("Invalid learned time distribution")
    boundary = math.log1p(horizon)
    return [None if mu + sigma*z > boundary else max(0., math.expm1(mu + sigma*z))
            for z in normals]


def median_time(mu):
    # Report an uncapped deterministic median, independently from sampled MAE.
    if not math.isfinite(mu) or mu > 700:
        raise ValueError("Nonfinite deterministic prediction")
    return max(0., math.expm1(mu))


def summarize(rows, baseline, rollouts):
    errors, median_errors, baseline_all, baseline_paired, availability = [], [], [], [], []
    truth_count = 0
    for row in rows:
        if len(row["samples"]) != rollouts:
            raise ValueError("Draw count differs")
        good = [s for s in row["samples"] if s is not None] if row["event"] else []
        availability.append(len(good) / rollouts)
        if row["event"]:
            truth_count += 1
            median_errors.append(abs(row["median"] - row["duration"]))
            baseline_all.append(abs(baseline - row["duration"]))
        if good:
            errors.append(statistics.mean(abs(s-row["duration"]) for s in good))
            baseline_paired.append(abs(baseline-row["duration"]))
    mean = lambda values: statistics.mean(values) if values else None
    return dict(patients=len(rows), observed_target_patients=truth_count,
                paired_patients=len(errors), sampled_time_mae_days=mean(errors),
                paired_rollout_availability=mean(availability),
                deterministic_median_mae_days=mean(median_errors),
                training_observed_median_days=baseline,
                baseline_mae_same_paired_patients=mean(baseline_paired),
                baseline_mae_all_observed_patients=mean(baseline_all))
