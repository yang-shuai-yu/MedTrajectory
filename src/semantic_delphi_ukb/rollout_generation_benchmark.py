from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
for path in [REPO_DIR, REPO_DIR / "src", REPO_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from semantic_delphi_ukb.architecture_baselines import (  # noqa: E402
    ArchitectureBaselineConfig,
    HistoryMaskBERTBaseline,
    MambaHistoryBaseline,
)
from semantic_delphi_ukb.selected_disease_demo import (  # noqa: E402
    default_model_specs,
    load_labels,
    load_model as load_exp2_model,
)
from semantic_delphi_ukb.tte_model import HorizonRiskConfig, HorizonRiskMedTrajectory  # noqa: E402
from utils import get_p2i  # noqa: E402


MASK_TIME = -10000.0


DEFAULT_SPECS = [
    (
        "MedTrajectory Gated RoPE",
        "medtrajectory",
        "gated",
        REPO_DIR / "results" / "gated_rope_ablation" / "gate100" / "ckpt.pt",
        "native autoregressive hazard rollout",
    ),
    (
        "Exp2 multitype internal anchor",
        "exp2",
        "exp2_anchor",
        REPO_DIR / "ckpt" / "Delphi_semantic_icd64_multitype_explicit_split" / "ckpt.pt",
        "native Exp2/MedTrajectory-style autoregressive hazard rollout",
    ),
    (
        "BERT pseudo-rollout",
        "bert",
        "bert",
        REPO_DIR / "results" / "architecture_baselines" / "bert_full_noleak" / "ckpt.pt",
        "pseudo autoregressive hazard rollout from a no-leak encoder",
    ),
    (
        "Mamba pseudo-rollout",
        "mamba",
        "mamba",
        REPO_DIR / "results" / "architecture_baselines" / "mamba_full_noleak" / "ckpt.pt",
        "pseudo autoregressive hazard rollout from a no-leak SSM",
    ),
    (
        "Old TTE+RoPE main",
        "medtrajectory",
        "old_tte",
        REPO_DIR / "results" / "high_priority_ablation" / "tte_trunk_aux02" / "ckpt.pt",
        "native autoregressive hazard rollout",
    ),
]


@dataclass
class Event:
    age_days: float
    age_years: float
    token_id: int
    token: str
    event_type: str


class RolloutModel:
    def __init__(
        self,
        name: str,
        kind: str,
        alias: str,
        protocol: str,
        model: torch.nn.Module,
        block_size: int,
        vocab_size: int,
        labels: list[str],
        ignore_tokens: Sequence[int],
        device: str,
    ):
        self.name = name
        self.kind = kind
        self.alias = alias
        self.protocol = protocol
        self.model = model
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.labels = labels
        self.ignore_tokens = tuple(int(x) for x in ignore_tokens)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def forward_last(self, tokens: torch.Tensor, ages: torch.Tensor, static_features: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = tokens[:, -self.block_size :]
        ages = ages[:, -self.block_size :]
        if self.kind == "medtrajectory":
            logits, _, _, _, _ = self.model(tokens, ages, static_features)
        elif self.kind == "exp2":
            logits, _, _ = self.model(tokens, ages, static_features)
        else:
            logits = self.model(tokens, ages, static_features)
        return logits[:, -1, :]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-patient autoregressive rollout generation benchmark.")
    parser.add_argument("--dataset", type=str, default="ukb_semantic_multitype_explicit_split")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "rollout_generation_benchmark")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--models", type=str, default="gated,exp2_anchor,bert,mamba")
    parser.add_argument("--max-patients", type=int, default=100)
    parser.add_argument("--num-rollouts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--followup-years", type=float, default=10.0)
    parser.add_argument("--baseline-fraction", type=float, default=0.65)
    parser.add_argument("--min-history-events", type=int, default=8)
    parser.add_argument("--min-future-events", type=int, default=3)
    parser.add_argument("--scope", choices=["clinical", "diagnosis"], default="clinical")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument(
        "--death-logit-bias",
        type=float,
        default=0.0,
        help="Additive bias for death:* logits during rollout; negative values implement a survival-gated death sampler.",
    )
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--write-sample-json", action="store_true")
    return parser


def resolve_data_dir(args: argparse.Namespace) -> Path:
    return args.data_dir or (REPO_DIR / "data" / args.dataset)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_state_dict(state_dict: dict) -> dict:
    state_dict = dict(state_dict)
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    return state_dict


def load_split(data_dir: Path, split: str):
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    static = np.load(data_dir / f"{split}_static.npy").astype(np.float32)
    labels = load_labels(data_dir / "labels.csv")
    return data, p2i, static, labels


def label_for(labels: Sequence[str], token_id: int) -> str:
    if 0 <= int(token_id) < len(labels):
        return labels[int(token_id)]
    return str(token_id)


def event_type(label: str) -> str:
    if ":" in label:
        return label.split(":", 1)[0]
    return label


def patient_events(data: np.ndarray, p2i: np.ndarray, labels: Sequence[str], patient_index: int) -> list[Event]:
    start, length = p2i[int(patient_index)]
    rows = data[int(start) : int(start) + int(length)]
    events: list[Event] = []
    for _, age_days, raw_token in rows:
        token_id = int(raw_token) + 1
        label = label_for(labels, token_id)
        events.append(
            Event(
                age_days=float(age_days),
                age_years=float(age_days) / 365.25,
                token_id=token_id,
                token=label,
                event_type=event_type(label),
            )
        )
    return events


def choose_cases(
    data: np.ndarray,
    p2i: np.ndarray,
    labels: Sequence[str],
    max_patients: int,
    baseline_fraction: float,
    min_history_events: int,
    min_future_events: int,
    followup_years: float,
) -> list[dict]:
    cases = []
    for patient_index in range(len(p2i)):
        events = patient_events(data, p2i, labels, patient_index)
        if len(events) < min_history_events + min_future_events:
            continue
        cut = int(math.floor((len(events) - 1) * baseline_fraction))
        cut = max(min_history_events - 1, min(cut, len(events) - min_future_events - 1))
        baseline_age = events[cut].age_years
        max_future_age = baseline_age + followup_years
        future = [event for event in events[cut + 1 :] if event.age_years <= max_future_age]
        if len(future) < min_future_events:
            continue
        if not any(event.event_type == "diag" for event in future):
            continue
        cases.append({"patient_index": patient_index, "cut_index": cut, "baseline_age_years": baseline_age})
        if max_patients > 0 and len(cases) >= max_patients:
            break
    return cases


def selected_specs(model_spec: str):
    aliases = {item.strip().lower() for item in model_spec.split(",") if item.strip()}
    if "all" in aliases:
        aliases.update(item[2] for item in DEFAULT_SPECS)
    out = []
    for spec in DEFAULT_SPECS:
        if spec[2] in aliases or spec[0].lower() in aliases:
            out.append(spec)
    if not out:
        raise ValueError(f"No model selected from {model_spec!r}")
    return out


def load_medtrajectory_wrapper(name: str, kind: str, alias: str, ckpt_path: Path, protocol: str, labels: list[str], device: str):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HorizonRiskMedTrajectory(HorizonRiskConfig(**checkpoint["model_args"]))
    model.load_state_dict(normalize_state_dict(checkpoint["model"]), strict=True)
    model = model.to(device)
    config = model.config
    return RolloutModel(name, kind, alias, protocol, model, config.block_size, config.vocab_size, labels, config.ignore_tokens, device)


def load_architecture_wrapper(name: str, kind: str, alias: str, ckpt_path: Path, protocol: str, labels: list[str], device: str):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint["model_args"]
    config = ArchitectureBaselineConfig(
        block_size=int(ckpt_args.get("block_size", checkpoint.get("block_size", 128))),
        vocab_size=int(checkpoint["vocab_size"]),
        n_layer=int(ckpt_args["n_layer"]),
        n_head=int(ckpt_args["n_head"]),
        n_embd=int(ckpt_args["n_embd"]),
        dropout=float(ckpt_args["dropout"]),
        static_dim=int(checkpoint["static_dim"]),
    )
    model = HistoryMaskBERTBaseline(config) if kind == "bert" else MambaHistoryBaseline(config)
    model.load_state_dict(normalize_state_dict(checkpoint["model"]), strict=True)
    model = model.to(device)
    return RolloutModel(name, kind, alias, protocol, model, config.block_size, config.vocab_size, labels, (0, 1), device)


def load_exp2_wrapper(name: str, kind: str, alias: str, protocol: str, split: str, device: str):
    exp2_spec = next(spec for spec in default_model_specs() if spec.model_id == "exp2")
    loaded = load_exp2_model(exp2_spec, split=split, device=device)
    return RolloutModel(
        name=name,
        kind=kind,
        alias=alias,
        protocol=protocol,
        model=loaded.model,
        block_size=loaded.block_size,
        vocab_size=len(loaded.labels),
        labels=loaded.labels,
        ignore_tokens=loaded.model.config.ignore_tokens,
        device=device,
    )


def load_rollout_models(args: argparse.Namespace, labels: list[str]) -> list[RolloutModel]:
    wrappers = []
    for name, kind, alias, ckpt_path, protocol in selected_specs(args.models):
        if kind == "exp2":
            wrappers.append(load_exp2_wrapper(name, kind, alias, protocol, args.split, args.device))
        elif not ckpt_path.exists():
            print(f"[skip missing] {name}: {ckpt_path}", file=sys.stderr)
        elif kind == "medtrajectory":
            wrappers.append(load_medtrajectory_wrapper(name, kind, alias, ckpt_path, protocol, labels, args.device))
        elif kind in {"bert", "mamba"}:
            wrappers.append(load_architecture_wrapper(name, kind, alias, ckpt_path, protocol, labels, args.device))
        else:
            raise ValueError(f"Unknown model kind: {kind}")
    if not wrappers:
        raise RuntimeError("No rollout models could be loaded.")
    return wrappers


def allowed_mask(labels: Sequence[str], scope: str, device: str, ignore_tokens: Sequence[int]) -> torch.Tensor:
    allowed = torch.zeros(len(labels), dtype=torch.bool, device=device)
    prefixes = ("diag:",) if scope == "diagnosis" else ("diag:", "proc:", "cancer:", "death:")
    for idx, label in enumerate(labels):
        if label.startswith(prefixes):
            allowed[idx] = True
    for idx, label in enumerate(labels):
        if idx <= 1 or label in {"", "Padding", "No event"}:
            allowed[idx] = False
    for token_id in ignore_tokens:
        if 0 <= int(token_id) < len(labels):
            allowed[int(token_id)] = False
    return allowed


def diagnosis_ids(labels: Sequence[str]) -> list[int]:
    return [idx for idx, label in enumerate(labels) if label.startswith("diag:")]


def death_ids(labels: Sequence[str]) -> list[int]:
    return [idx for idx, label in enumerate(labels) if label.startswith("death:")]


def context_for_case(events: Sequence[Event], cut_index: int, block_size: int, device: str):
    context = events[: cut_index + 1]
    if len(context) > block_size:
        context = context[-block_size:]
    tokens = torch.tensor([[event.token_id for event in context]], dtype=torch.long, device=device)
    ages = torch.tensor([[event.age_days for event in context]], dtype=torch.float32, device=device)
    return tokens, ages


def topk_first_diag_hits(
    model: RolloutModel,
    context_tokens: torch.Tensor,
    context_ages: torch.Tensor,
    static_features: torch.Tensor,
    first_diag_token: Optional[int],
) -> dict[str, int]:
    if first_diag_token is None:
        return {"top1_first_diag_hit": 0, "top5_first_diag_hit": 0, "top10_first_diag_hit": 0}
    with torch.no_grad():
        logits = model.forward_last(context_tokens, context_ages, static_features)[0]
    ids = [idx for idx in diagnosis_ids(model.labels) if idx < logits.numel()]
    if not ids:
        return {"top1_first_diag_hit": 0, "top5_first_diag_hit": 0, "top10_first_diag_hit": 0}
    candidate = torch.tensor(ids, dtype=torch.long, device=logits.device)
    top = candidate[torch.topk(logits[candidate], k=min(10, len(ids))).indices].detach().cpu().tolist()
    return {
        "top1_first_diag_hit": int(first_diag_token in top[:1]),
        "top5_first_diag_hit": int(first_diag_token in top[:5]),
        "top10_first_diag_hit": int(first_diag_token in top[:10]),
    }


@torch.no_grad()
def rollout_many(
    model: RolloutModel,
    context_tokens: torch.Tensor,
    context_ages: torch.Tensor,
    static_features: torch.Tensor,
    allowed: torch.Tensor,
    num_rollouts: int,
    max_new_tokens: int,
    max_age_years: float,
    seed: int,
    temperature: float,
    no_repeat: bool,
    death_logit_bias: float = 0.0,
) -> list[list[Event]]:
    n = int(num_rollouts)
    tokens = context_tokens.repeat(n, 1)
    ages = context_ages.repeat(n, 1)
    static_batch = static_features.repeat(n, 1)
    active = torch.ones(n, dtype=torch.bool, device=model.device)
    generated: list[list[Event]] = [[] for _ in range(n)]
    generator_device = "cuda" if str(model.device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    max_age_days = float(max_age_years) * 365.25
    max_wait_days = 365.25 * 80.0
    temp = max(float(temperature), 1e-6)
    death_token_ids = [
        idx
        for idx in death_ids(model.labels)
        if 0 <= int(idx) < int(allowed.numel()) and bool(allowed[int(idx)].detach().cpu().item())
    ]
    death_index = torch.tensor(death_token_ids, dtype=torch.long, device=model.device) if death_token_ids else None

    for _ in range(int(max_new_tokens)):
        active_ids = torch.where(active)[0]
        if int(active_ids.numel()) == 0:
            break
        logits = model.forward_last(tokens[active_ids], ages[active_ids], static_batch[active_ids]) / temp
        logits[:, ~allowed[: logits.size(1)]] = -torch.inf
        if death_index is not None and float(death_logit_bias) != 0.0:
            valid_death = death_index[death_index < logits.size(1)]
            if int(valid_death.numel()) > 0:
                logits[:, valid_death] = logits[:, valid_death] + float(death_logit_bias)
        if no_repeat:
            seen = tokens[active_ids].clone()
            seen[seen < 2] = 0
            seen = seen.clamp(min=0, max=logits.size(1) - 1)
            logits.scatter_(1, seen, -torch.inf)
        uniform = torch.rand(logits.shape, generator=generator, device=logits.device).clamp_min(1e-12)
        waiting = -torch.log(uniform) * torch.exp(-logits)
        waiting = torch.clamp(waiting, min=0.0, max=max_wait_days)
        waiting = waiting.masked_fill(~torch.isfinite(logits), torch.inf)
        delta_days, next_tokens = waiting.min(dim=1)
        next_ages = ages[active_ids, -1] + delta_days

        append_tokens = torch.zeros((n, 1), dtype=torch.long, device=model.device)
        append_ages = torch.full((n, 1), MASK_TIME, dtype=torch.float32, device=model.device)
        append_tokens[active_ids, 0] = next_tokens
        append_ages[active_ids, 0] = next_ages
        tokens = torch.cat([tokens, append_tokens], dim=1)
        ages = torch.cat([ages, append_ages], dim=1)

        for j, rollout_id in enumerate(active_ids.detach().cpu().tolist()):
            age_days = float(next_ages[j].detach().cpu().item())
            if age_days > max_age_days:
                active[rollout_id] = False
                continue
            token_id = int(next_tokens[j].detach().cpu().item())
            label = label_for(model.labels, token_id)
            item = Event(
                age_days=age_days,
                age_years=age_days / 365.25,
                token_id=token_id,
                token=label,
                event_type=event_type(label),
            )
            generated[rollout_id].append(item)
            if item.event_type == "death":
                active[rollout_id] = False
    return generated


def first_event(events: Sequence[Event], kind: Optional[str] = None) -> Optional[Event]:
    for event in events:
        if kind is None or event.event_type == kind:
            return event
    return None


def token_set(events: Sequence[Event], kind: Optional[str] = None) -> set[int]:
    return {event.token_id for event in events if kind is None or event.event_type == kind}


def token_list(events: Sequence[Event], kind: Optional[str] = None) -> list[int]:
    return [event.token_id for event in events if kind is None or event.event_type == kind]


def safe_mean(values: Sequence[float]) -> float:
    clean = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def set_recall(actual: set[int], predicted: set[int]) -> float:
    if not actual:
        return float("nan")
    return len(actual & predicted) / len(actual)


def jaccard(actual: set[int], predicted: set[int]) -> float:
    union = actual | predicted
    if not union:
        return float("nan")
    return len(actual & predicted) / len(union)


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + int(ca != cb)))
        prev = cur
    return prev[-1]


def norm_edit_distance(a: Sequence[int], b: Sequence[int]) -> float:
    denom = max(len(a), len(b), 1)
    return levenshtein(a, b) / denom


def first_age_by_token(events: Sequence[Event], kind: Optional[str] = None) -> dict[int, float]:
    out: dict[int, float] = {}
    for event in events:
        if kind is not None and event.event_type != kind:
            continue
        out.setdefault(event.token_id, event.age_years)
    return out


def matched_time_mae(actual: Sequence[Event], generated: Sequence[Event], kind: str = "diag") -> float:
    actual_age = first_age_by_token(actual, kind=kind)
    generated_age = first_age_by_token(generated, kind=kind)
    errors = [abs(age - generated_age[token]) for token, age in actual_age.items() if token in generated_age]
    return safe_mean(errors)


def event_counter(events: Sequence[Event], kind: Optional[str] = None) -> Counter:
    return Counter(token_list(events, kind=kind))


def event_type_counter(events: Sequence[Event]) -> Counter:
    return Counter(event.event_type for event in events)


def js_divergence(counts_a: Counter, counts_b: Counter) -> float:
    keys = sorted(set(counts_a) | set(counts_b))
    if not keys:
        return float("nan")
    pa = np.asarray([float(counts_a.get(k, 0.0)) for k in keys], dtype=np.float64)
    pb = np.asarray([float(counts_b.get(k, 0.0)) for k in keys], dtype=np.float64)
    if pa.sum() <= 0 or pb.sum() <= 0:
        return float("nan")
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pa + pb)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def evaluate_case(
    model: RolloutModel,
    events: Sequence[Event],
    case: dict,
    static_features: torch.Tensor,
    args: argparse.Namespace,
    model_index: int,
) -> tuple[dict, Counter, Counter, Counter, Counter, Optional[dict]]:
    cut_index = int(case["cut_index"])
    baseline_age = float(case["baseline_age_years"])
    future_limit = baseline_age + float(args.followup_years)
    actual_future = [event for event in events[cut_index + 1 :] if event.age_years <= future_limit]
    actual_diag = [event for event in actual_future if event.event_type == "diag"]
    actual_death = first_event(actual_future, kind="death")
    first_diag = first_event(actual_future, kind="diag")
    context_tokens, context_ages = context_for_case(events, cut_index, model.block_size, args.device)
    allowed = allowed_mask(model.labels, args.scope, args.device, model.ignore_tokens)
    top_hits = topk_first_diag_hits(
        model,
        context_tokens,
        context_ages,
        static_features,
        first_diag.token_id if first_diag is not None else None,
    )
    seed = int(args.seed + 100_003 * int(case["patient_index"]) + 7_919 * model_index)
    rollouts = rollout_many(
        model=model,
        context_tokens=context_tokens,
        context_ages=context_ages,
        static_features=static_features,
        allowed=allowed,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        max_age_years=future_limit,
        seed=seed,
        temperature=args.temperature,
        no_repeat=not args.allow_repeat,
        death_logit_bias=args.death_logit_bias,
    )

    actual_diag_set = token_set(actual_future, kind="diag")
    actual_seq = token_list(actual_future)
    actual_diag_seq = token_list(actual_future, kind="diag")
    union_diag: set[int] = set()
    union_all: set[int] = set()
    rollout_recalls = []
    rollout_jaccards = []
    rollout_diag_jaccards = []
    rollout_edit = []
    rollout_diag_edit = []
    rollout_time_mae = []
    rollout_first_diag_hits = []
    rollout_lengths = []
    death_hits = 0
    death_age_errors = []
    generated_counts = Counter()
    generated_diag_counts = Counter()
    generated_type_counts = Counter()
    sample_rollout = None

    for ridx, generated in enumerate(rollouts):
        gen_diag_set = token_set(generated, kind="diag")
        gen_all_set = token_set(generated)
        union_diag |= gen_diag_set
        union_all |= gen_all_set
        rollout_recalls.append(set_recall(actual_diag_set, gen_diag_set))
        rollout_jaccards.append(jaccard(token_set(actual_future), gen_all_set))
        rollout_diag_jaccards.append(jaccard(actual_diag_set, gen_diag_set))
        rollout_edit.append(norm_edit_distance(actual_seq, token_list(generated)))
        rollout_diag_edit.append(norm_edit_distance(actual_diag_seq, token_list(generated, kind="diag")))
        rollout_time_mae.append(matched_time_mae(actual_future, generated, kind="diag"))
        rollout_first_diag_hits.append(int(first_diag is not None and first_diag.token_id in gen_diag_set))
        rollout_lengths.append(len(generated))
        generated_counts.update(event_counter(generated))
        generated_diag_counts.update(event_counter(generated, kind="diag"))
        generated_type_counts.update(event_type_counter(generated))
        gen_death = first_event(generated, kind="death")
        if gen_death is not None:
            death_hits += 1
            if actual_death is not None:
                death_age_errors.append(abs(gen_death.age_years - actual_death.age_years))
        if ridx == 0:
            sample_rollout = [
                {"age_years": event.age_years, "token_id": event.token_id, "token": event.token, "event_type": event.event_type}
                for event in generated
            ]

    death_prob = death_hits / max(1, len(rollouts))
    actual_death_flag = int(actual_death is not None)
    row = {
        "model": model.name,
        "alias": model.alias,
        "protocol": model.protocol,
        "patient_index": int(case["patient_index"]),
        "cut_index": cut_index,
        "baseline_age_years": baseline_age,
        "actual_future_events": len(actual_future),
        "actual_future_diag_count": len(actual_diag),
        "actual_death": actual_death_flag,
        "actual_death_age_years": actual_death.age_years if actual_death else "",
        "first_future_diag_token": first_diag.token if first_diag else "",
        "num_rollouts": len(rollouts),
        "generated_length_mean": safe_mean(rollout_lengths),
        "diag_recall_union": set_recall(actual_diag_set, union_diag),
        "diag_recall_mean_rollout": safe_mean(rollout_recalls),
        "diag_jaccard_union": jaccard(actual_diag_set, union_diag),
        "diag_jaccard_mean_rollout": safe_mean(rollout_diag_jaccards),
        "event_jaccard_union": jaccard(token_set(actual_future), union_all),
        "event_jaccard_mean_rollout": safe_mean(rollout_jaccards),
        "sequence_edit_norm_mean": safe_mean(rollout_edit),
        "diagnosis_sequence_edit_norm_mean": safe_mean(rollout_diag_edit),
        "matched_diag_time_mae_years": safe_mean(rollout_time_mae),
        "first_diag_rollout_hit_rate": safe_mean(rollout_first_diag_hits),
        "predicted_death_prob": death_prob,
        "death_brier": (death_prob - actual_death_flag) ** 2,
        "death_age_mae_years": safe_mean(death_age_errors),
        **top_hits,
    }
    actual_counts = event_counter(actual_future)
    actual_diag_counts = event_counter(actual_future, kind="diag")
    actual_type_counts = event_type_counter(actual_future)
    sample = None
    if sample_rollout is not None:
        sample = {
            "model": model.name,
            "patient_index": int(case["patient_index"]),
            "baseline_age_years": baseline_age,
            "actual_future": [
                {"age_years": event.age_years, "token_id": event.token_id, "token": event.token, "event_type": event.event_type}
                for event in actual_future
            ],
            "sample_rollout": sample_rollout,
        }
    return row, actual_counts, generated_counts, actual_diag_counts, generated_diag_counts, {"actual_type": actual_type_counts, "generated_type": generated_type_counts, "sample": sample}


def death_calibration_rows(patient_rows: Sequence[dict], bins: int = 5) -> list[dict]:
    rows = []
    by_model = defaultdict(list)
    for row in patient_rows:
        by_model[row["model"]].append(row)
    for model, model_rows in by_model.items():
        for bin_idx in range(bins):
            lo = bin_idx / bins
            hi = (bin_idx + 1) / bins
            if bin_idx == bins - 1:
                selected = [row for row in model_rows if lo <= float(row["predicted_death_prob"]) <= hi]
            else:
                selected = [row for row in model_rows if lo <= float(row["predicted_death_prob"]) < hi]
            if not selected:
                continue
            mean_pred = safe_mean([float(row["predicted_death_prob"]) for row in selected])
            observed = safe_mean([int(row["actual_death"]) for row in selected])
            rows.append(
                {
                    "model": model,
                    "bin": bin_idx,
                    "prob_low": lo,
                    "prob_high": hi,
                    "n": len(selected),
                    "mean_predicted_death_prob": mean_pred,
                    "observed_death_rate": observed,
                    "abs_calibration_error": abs(mean_pred - observed),
                }
            )
    return rows


def summarize(patient_rows: Sequence[dict], distribution_rows: Sequence[dict], calibration_rows: Sequence[dict]) -> list[dict]:
    by_model = defaultdict(list)
    for row in patient_rows:
        by_model[row["model"]].append(row)
    dist_by_model = {row["model"]: row for row in distribution_rows}
    cal_by_model = defaultdict(list)
    for row in calibration_rows:
        cal_by_model[row["model"]].append(row)
    out = []
    for model, rows in by_model.items():
        total = len(rows)
        ece = 0.0
        for row in cal_by_model.get(model, []):
            ece += (int(row["n"]) / max(1, total)) * float(row["abs_calibration_error"])
        dist = dist_by_model.get(model, {})
        out.append(
            {
                "model": model,
                "patients": total,
                "rollouts_per_patient": int(rows[0]["num_rollouts"]) if rows else 0,
                "protocol": rows[0]["protocol"] if rows else "",
                "diag_recall_union_mean": safe_mean([float(row["diag_recall_union"]) for row in rows]),
                "diag_recall_mean_rollout": safe_mean([float(row["diag_recall_mean_rollout"]) for row in rows]),
                "diag_jaccard_union_mean": safe_mean([float(row["diag_jaccard_union"]) for row in rows]),
                "diag_jaccard_mean_rollout": safe_mean([float(row["diag_jaccard_mean_rollout"]) for row in rows]),
                "first_diag_rollout_hit_rate": safe_mean([float(row["first_diag_rollout_hit_rate"]) for row in rows]),
                "top1_first_diag_hit_rate": safe_mean([int(row["top1_first_diag_hit"]) for row in rows]),
                "top5_first_diag_hit_rate": safe_mean([int(row["top5_first_diag_hit"]) for row in rows]),
                "top10_first_diag_hit_rate": safe_mean([int(row["top10_first_diag_hit"]) for row in rows]),
                "event_jaccard_union_mean": safe_mean([float(row["event_jaccard_union"]) for row in rows]),
                "event_jaccard_mean_rollout": safe_mean([float(row["event_jaccard_mean_rollout"]) for row in rows]),
                "sequence_edit_norm_mean": safe_mean([float(row["sequence_edit_norm_mean"]) for row in rows]),
                "diagnosis_sequence_edit_norm_mean": safe_mean([float(row["diagnosis_sequence_edit_norm_mean"]) for row in rows]),
                "matched_diag_time_mae_years": safe_mean([float(row["matched_diag_time_mae_years"]) for row in rows]),
                "predicted_death_prob_mean": safe_mean([float(row["predicted_death_prob"]) for row in rows]),
                "actual_death_rate": safe_mean([int(row["actual_death"]) for row in rows]),
                "death_brier_mean": safe_mean([float(row["death_brier"]) for row in rows]),
                "death_ece": ece,
                "death_age_mae_years": safe_mean([float(row["death_age_mae_years"]) for row in rows]),
                "generated_length_mean": safe_mean([float(row["generated_length_mean"]) for row in rows]),
                "token_js_divergence": dist.get("token_js_divergence", ""),
                "diagnosis_token_js_divergence": dist.get("diagnosis_token_js_divergence", ""),
                "event_type_js_divergence": dist.get("event_type_js_divergence", ""),
            }
        )
    return out


def draw_charts(out_dir: Path, summary_rows: Sequence[dict], calibration_rows: Sequence[dict]) -> dict[str, str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"[warn] Pillow unavailable, skip charts: {exc}", file=sys.stderr)
        return {}

    def font(size: int, bold: bool = False):
        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    def text_size(draw, text, fnt):
        box = draw.textbbox((0, 0), str(text), font=fnt)
        return box[2] - box[0], box[3] - box[1]

    def grouped_bar(path: Path, title: str, metrics: list[tuple[str, str]], higher_is_better: bool = True):
        models = [row["model"] for row in summary_rows]
        width, height = 1500, 860
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        title_font = font(34, True)
        label_font = font(18)
        value_font = font(16, True)
        draw.text((50, 30), title, fill=(11, 37, 69), font=title_font)
        left, right, top, bottom = 95, width - 55, 135, height - 190
        draw.line((left, bottom, right, bottom), fill=(80, 90, 100), width=2)
        draw.line((left, top, left, bottom), fill=(80, 90, 100), width=2)
        all_values = []
        for row in summary_rows:
            for key, _ in metrics:
                try:
                    value = float(row[key])
                except Exception:
                    continue
                if math.isfinite(value):
                    all_values.append(value)
        max_v = max(all_values + [1.0])
        min_v = 0.0
        if not higher_is_better:
            max_v = max(all_values + [0.1]) * 1.10
        else:
            max_v = min(1.0, max(max_v, 0.8))
        for i in range(6):
            val = min_v + (max_v - min_v) * i / 5
            y = bottom - (bottom - top) * ((val - min_v) / max(max_v - min_v, 1e-9))
            draw.line((left, y, right, y), fill=(230, 235, 241), width=1)
            draw.text((left - 72, y - 10), f"{val:.2f}", fill=(91, 103, 112), font=label_font)
        colors = [(46, 116, 181), (61, 153, 142), (235, 154, 76), (188, 88, 88)]
        group_w = (right - left) / max(1, len(models))
        bar_w = min(42, (group_w - 28) / max(1, len(metrics)))
        for mi, row in enumerate(summary_rows):
            gx = left + mi * group_w + 16
            for ki, (key, _) in enumerate(metrics):
                value = float(row.get(key) or 0.0)
                if not math.isfinite(value):
                    value = 0.0
                x0 = gx + ki * (bar_w + 8)
                x1 = x0 + bar_w
                y0 = bottom - (bottom - top) * ((value - min_v) / max(max_v - min_v, 1e-9))
                draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=colors[ki % len(colors)])
                txt = f"{value:.3f}"
                tw, th = text_size(draw, txt, value_font)
                draw.text((x0 + (bar_w - tw) / 2, y0 - th - 5), txt, fill=(11, 37, 69), font=value_font)
            short = (
                row["model"]
                .replace("MedTrajectory ", "MedTraj\n")
                .replace("Exp2 multitype internal anchor", "Exp2\ninternal")
                .replace(" pseudo-rollout", "\npseudo")
            )
            for li, line in enumerate(short.split("\n")):
                tw, _ = text_size(draw, line, label_font)
                draw.text((gx + (bar_w * len(metrics) + 8 * (len(metrics) - 1) - tw) / 2, bottom + 18 + li * 22), line, fill=(40, 48, 58), font=label_font)
        legend_x, legend_y = 80, height - 65
        for ki, (_, label) in enumerate(metrics):
            x = legend_x + ki * 330
            draw.rounded_rectangle((x, legend_y, x + 28, legend_y + 18), radius=4, fill=colors[ki % len(colors)])
            draw.text((x + 38, legend_y - 2), label, fill=(40, 48, 58), font=label_font)
        img.save(path)

    def death_chart(path: Path):
        models = [row["model"] for row in summary_rows]
        width, height = 1350, 760
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        title_font = font(32, True)
        label_font = font(18)
        value_font = font(17, True)
        draw.text((50, 30), "Death calibration in multi-patient rollout benchmark", fill=(11, 37, 69), font=title_font)
        left, right, top, bottom = 95, width - 55, 125, height - 155
        draw.line((left, bottom, right, bottom), fill=(80, 90, 100), width=2)
        draw.line((left, top, left, bottom), fill=(80, 90, 100), width=2)
        for i in range(6):
            val = i / 5
            y = bottom - (bottom - top) * val
            draw.line((left, y, right, y), fill=(230, 235, 241), width=1)
            draw.text((left - 68, y - 10), f"{val:.1f}", fill=(91, 103, 112), font=label_font)
        group_w = (right - left) / max(1, len(models))
        colors = [(46, 116, 181), (188, 88, 88)]
        for mi, row in enumerate(summary_rows):
            gx = left + mi * group_w + 36
            vals = [float(row["predicted_death_prob_mean"]), float(row["actual_death_rate"])]
            for ki, val in enumerate(vals):
                x0 = gx + ki * 54
                x1 = x0 + 42
                y0 = bottom - (bottom - top) * val
                draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=colors[ki])
                txt = f"{val:.3f}"
                tw, th = text_size(draw, txt, value_font)
                draw.text((x0 + (42 - tw) / 2, y0 - th - 5), txt, fill=(11, 37, 69), font=value_font)
            short = (
                row["model"]
                .replace("MedTrajectory ", "MedTraj\n")
                .replace("Exp2 multitype internal anchor", "Exp2\ninternal")
                .replace(" pseudo-rollout", "\npseudo")
            )
            for li, line in enumerate(short.split("\n")):
                tw, _ = text_size(draw, line, label_font)
                draw.text((gx + (96 - tw) / 2, bottom + 18 + li * 22), line, fill=(40, 48, 58), font=label_font)
        draw.rounded_rectangle((80, height - 68, 108, height - 50), radius=4, fill=colors[0])
        draw.text((118, height - 70), "mean generated death probability", fill=(40, 48, 58), font=label_font)
        draw.rounded_rectangle((450, height - 68, 478, height - 50), radius=4, fill=colors[1])
        draw.text((488, height - 70), "observed death rate", fill=(40, 48, 58), font=label_font)
        img.save(path)

    out = {
        "metric_chart": str(out_dir / "rollout_generation_metric_comparison.png"),
        "error_chart": str(out_dir / "rollout_generation_error_comparison.png"),
        "death_chart": str(out_dir / "rollout_death_calibration.png"),
    }
    grouped_bar(
        Path(out["metric_chart"]),
        "Higher-is-better rollout generation metrics",
        [
            ("diag_recall_union_mean", "future disease recall"),
            ("diag_jaccard_mean_rollout", "diagnosis Jaccard"),
            ("top10_first_diag_hit_rate", "Top-10 first disease"),
            ("first_diag_rollout_hit_rate", "rollout first-disease hit"),
        ],
        higher_is_better=True,
    )
    grouped_bar(
        Path(out["error_chart"]),
        "Lower-is-better rollout generation errors",
        [
            ("sequence_edit_norm_mean", "sequence edit"),
            ("matched_diag_time_mae_years", "time MAE years"),
            ("death_brier_mean", "death Brier"),
            ("diagnosis_token_js_divergence", "diagnosis JS"),
        ],
        higher_is_better=False,
    )
    death_chart(Path(out["death_chart"]))
    _ = calibration_rows
    return out


def write_markdown(path: Path, summary_rows: Sequence[dict], chart_paths: dict[str, str], args: argparse.Namespace) -> None:
    lines = [
        "# Multi-patient rollout generation benchmark",
        "",
        "## Protocol",
        "",
        (
            f"Split={args.split}; max_patients={args.max_patients}; rollouts_per_patient={args.num_rollouts}; "
            f"max_new_tokens={args.max_new_tokens}; followup_years={args.followup_years}; baseline_fraction={args.baseline_fraction}."
        ),
        "",
        "MedTrajectory and Exp2 are native autoregressive hazard rollouts on the same explicit-split multitype competition data. BERT and Mamba are pseudo-rollouts: no-leak history encoders applied step-by-step to generated histories.",
        "",
        "Important boundary: Exp2 is an internal MedTrajectory/Delphi_rec-derived competition branch, not the official Delphi-2M demo. The official Delphi demo is listed separately as a reference-only row because it uses ukb_simulated_data val and a different protocol.",
        "",
        "## Summary",
        "",
        "| Model | Disease recall | Diagnosis Jaccard | Top-10 first disease | Time MAE | Edit distance | Death Brier | Diagnosis JS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {recall:.4f} | {jac:.4f} | {top10:.4f} | {mae:.4f} | {edit:.4f} | {brier:.4f} | {js:.4f} |".format(
                model=row["model"],
                recall=float(row["diag_recall_union_mean"]),
                jac=float(row["diag_jaccard_mean_rollout"]),
                top10=float(row["top10_first_diag_hit_rate"]),
                mae=float(row["matched_diag_time_mae_years"]) if str(row["matched_diag_time_mae_years"]) else float("nan"),
                edit=float(row["sequence_edit_norm_mean"]),
                brier=float(row["death_brier_mean"]),
                js=float(row["diagnosis_token_js_divergence"]) if str(row["diagnosis_token_js_divergence"]) else float("nan"),
            )
        )
    lines += [
        "",
        "## Original Delphi Demo Reference Only",
        "",
        "| Reference | Dataset | Split | Patients | Eval targets | Top1 | Top5 | Top10 | Notebook mean AUC | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        "| Delphi-2M original demo | ukb_simulated_data | val | 7144 | 171903 | 0.0448 | 0.1449 | 0.2293 | 0.7565 | reference only, not same-split rollout |",
        "",
        "A copied patient-level Delphi demo artifact is kept at demo/original_delphi_patient_future_demo/ for inspection. It is a one-case reference demo from the read-only Delphi_rec workspace, not the Exp2 internal rollout row above.",
    ]
    if chart_paths:
        lines.extend(
            [
                "",
                "## Figures",
                "",
                f"![Higher-is-better metrics]({Path(chart_paths['metric_chart']).name})",
                "",
                f"![Lower-is-better errors]({Path(chart_paths['error_chart']).name})",
                "",
                f"![Death calibration]({Path(chart_paths['death_chart']).name})",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args)
    data, p2i, static, labels = load_split(data_dir, args.split)
    cases = choose_cases(
        data=data,
        p2i=p2i,
        labels=labels,
        max_patients=args.max_patients,
        baseline_fraction=args.baseline_fraction,
        min_history_events=args.min_history_events,
        min_future_events=args.min_future_events,
        followup_years=args.followup_years,
    )
    models = load_rollout_models(args, labels)

    patient_rows = []
    distribution_rows = []
    sample_payload = []
    for model_index, model in enumerate(models):
        actual_counts_total = Counter()
        generated_counts_total = Counter()
        actual_diag_total = Counter()
        generated_diag_total = Counter()
        actual_type_total = Counter()
        generated_type_total = Counter()
        for case in cases:
            events = patient_events(data, p2i, model.labels, int(case["patient_index"]))
            static_features = torch.tensor(static[[int(case["patient_index"])]], dtype=torch.float32, device=args.device)
            row, actual_counts, generated_counts, actual_diag, generated_diag, extras = evaluate_case(
                model=model,
                events=events,
                case=case,
                static_features=static_features,
                args=args,
                model_index=model_index,
            )
            patient_rows.append(row)
            actual_counts_total.update(actual_counts)
            generated_counts_total.update(generated_counts)
            actual_diag_total.update(actual_diag)
            generated_diag_total.update(generated_diag)
            actual_type_total.update(extras["actual_type"])
            generated_type_total.update(extras["generated_type"])
            if args.write_sample_json and len(sample_payload) < 16 and extras.get("sample"):
                sample_payload.append(extras["sample"])
        distribution_rows.append(
            {
                "model": model.name,
                "token_js_divergence": js_divergence(actual_counts_total, generated_counts_total),
                "diagnosis_token_js_divergence": js_divergence(actual_diag_total, generated_diag_total),
                "event_type_js_divergence": js_divergence(actual_type_total, generated_type_total),
                "actual_event_tokens": int(sum(actual_counts_total.values())),
                "generated_event_tokens": int(sum(generated_counts_total.values())),
                "actual_diag_tokens": int(sum(actual_diag_total.values())),
                "generated_diag_tokens": int(sum(generated_diag_total.values())),
            }
        )

    calibration = death_calibration_rows(patient_rows)
    summary_rows = summarize(patient_rows, distribution_rows, calibration)
    chart_paths = draw_charts(args.out_dir, summary_rows, calibration)

    write_csv(args.out_dir / "rollout_patient_metrics.csv", patient_rows)
    write_csv(args.out_dir / "rollout_summary.csv", summary_rows)
    write_csv(args.out_dir / "rollout_distribution_distance.csv", distribution_rows)
    write_csv(args.out_dir / "rollout_death_calibration_bins.csv", calibration)
    write_markdown(args.out_dir / "rollout_generation_benchmark.md", summary_rows, chart_paths, args)
    if args.write_sample_json:
        (args.out_dir / "rollout_sample_cases.json").write_text(json.dumps(sample_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "rollout_run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "data_dir": str(data_dir),
                "models": [{"name": model.name, "alias": model.alias, "kind": model.kind, "protocol": model.protocol} for model in models],
                "num_cases": len(cases),
                "charts": chart_paths,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "patient_rows": len(patient_rows), "summary_rows": len(summary_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
