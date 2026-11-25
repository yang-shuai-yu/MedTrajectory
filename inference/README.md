# Minimal trained inference package

This package runs **MedTrajectory Ours(Relative), A2, UK Biobank, training seed 42**
on new, locally supplied histories. It includes a 5.6 MB trained state dictionary,
the exact model configuration and vocabulary, and an offline CPU command line.
The model was trained for 100,000 updates. No training or embedding API is needed
for inference. The example history is entirely synthetic.

## Run

From the repository root, with Python 3.10 or later:

```bash
python -m venv .venv-inference
# Linux: source .venv-inference/bin/activate
# Windows PowerShell: .venv-inference\Scripts\Activate.ps1
python -m pip install -r inference/requirements.txt
python inference/predict.py --input inference/examples/synthetic_history.json --output generated.json
```

Use a new output filename: existing files are never overwritten. After installation
and download, prediction is offline and does not send input or output anywhere.
The CPU package was checked with Python 3.12 and PyTorch 2.7.1+cpu on Windows.
Other platforms have not been validated in this release.

## Input contract

See [input.schema.json](input.schema.json) and the [synthetic example](examples/synthetic_history.json).
One JSON file describes one person's history; it contains no name or patient ID.

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1` |
| `static.anchor_age_days` | Age in **days since birth** when the baseline static features were measured; no later than the final historical event |
| `static.sex` | `female`, `male`, or `missing` |
| `static.bmi` | `missing`, `lt18.5`, `18.5_to_lt25`, `25_to_lt30`, or `gte30`; categorical BMI, not a raw numeric value |
| `static.smoking` | `missing`, `never`, `previous`, or `current` |
| `static.alcohol` | `missing`, `never`, `special_occasions`, `one_to_three_per_month`, `once_or_twice_per_week`, `three_or_four_times_per_week`, or `daily_or_almost_daily` |
| `history` | 1–10,000 events, ordered by nondecreasing `age_days`; ties retain the supplied order |
| `history[].code` | Exact vocabulary key, e.g. `diag:I10`; use [model/vocabulary.json](model/vocabulary.json), with no inferred code mapping |
| `history[].age_days` | Age at that event in days; finite and nonnegative |
| `generation.seed` | Integer RNG seed, default `20260819` |
| `generation.num_trajectories` | 1–20 sampled continuations, default 1 |
| `generation.max_new_events` | 1–256 generated clinical events per continuation, default 30 |
| `generation.horizon_days` | Positive follow-up duration, at most 3652.5 days (10 years); default 3652.5 |

The prediction origin is the **last historical event**, not the present date. If
predicting after a known event-free interval is required, this release does not
implement that conditioning. Do not fabricate an event to move the origin.
The age conversion is `age_years * 365.25`; ages are processed in float32 as in the
original evaluator. Input/output ages are therefore subject to float32 rounding.

Only the latest 128 clinical events are retained, with four static tokens and a
dynamic BOS token. Truncation is reported in the output. The static prefix stays
in its trained order; its information is gated by the measurement age. Missing
values must be explicitly marked `missing`. If the measurement age is unknown,
the original preparation convention is to set the anchor to
`max(1, first_event_age_days)` and BMI/smoking/alcohol to `missing`.

Codes are model-specific: `diag:` means ICD-10 diagnoses, `proc:` OPCS-4
procedures, `cancer:` cancer registration codes, and `death:` cause-coded death
events. Unknown codes are rejected. A history already containing death is
rejected. MIMIC codes or free text must not be passed as if they used this UKB
vocabulary. Histories shorter than the paper's eight-event eligibility minimum
can run, but are explicitly flagged.

## Output contract

The output is one JSON object; see [output.schema.json](output.schema.json).

| Field | Meaning |
|---|---|
| `task`, `trained_model`, `model_id` | Identify real trained multitype generation |
| `weights_sha256`, `source_checkpoint_sha256` | Exact exported-weight and original-checkpoint identities |
| `prediction_origin_age_days`, `horizon_end_age_days` | Prediction interval |
| `history_events_received/used/dropped` | Context truncation accounting |
| `generation`, `sampler`, `runtime` | Effective controls, frozen sampler, CPU/PyTorch version |
| `trajectories[].events[]` | `code`, `token_id`, `event_type`, `age_days`, `age_years`, `days_from_origin`, `waiting_days`, `minimum_wait_applied` |
| `trajectories[].stop_reason` | `death`, `horizon` (the next sampled event exceeded the horizon), or `event_limit` |
| `trajectories[].event_count` | Number of returned events |
| `notes` | Scope and short-history qualification |

An empty trajectory is valid if the next event is beyond the horizon. Death is
included if sampled inside the horizon, then generation stops immediately.
`event_limit` indicates truncation, not a disease-free remainder of follow-up.
`waiting_days` is the sampled gap from the preceding float32 model age.

The sampler reuses the public Track G implementation: temperature 0.8, top-p 0.9,
death logit bias 0; identity and waiting-time draws use separate uniforms. Waiting
times use the original **untempered, unbiased clinical-event log-rate** with
`t_min=0.1`, excluding padding/no-event/static/BOS/mask tokens, and a minimum
one-day gap. The existing auxiliary time-gap head is not used. Repeated codes are
possible. Multiple samples are stochastic continuations, not calibrated risks.

## Model provenance and scope

This is one fixed trained seed, chosen as the first registered seed, not the best
test seed. It is the A2 **pretraining** checkpoint used by the multitype generation
protocol, not the separately fine-tuned disease risk checkpoint. The exported
state dictionary retains the architecture's auxiliary heads for strict loading,
but the interface exposes only event generation; it does not expose their risk
outputs. It is not a calibrated mortality estimator or a clinical decision tool.

[model/manifest.json](model/manifest.json) records the original checkpoint hash,
training iteration, vocabulary and protocol hashes, and per-file SHA-256 checks.
The export removes optimizer/scheduler/RNG state and private training paths. It
also removes four never-used `static_encoder` tensors from the historical
`static_dim=0` branch to match the current public source; active weights are
unchanged. The vocabulary includes code-to-token mappings only, without patient
records, identifiers, counts, or individual embeddings. The trained token embedding
matrix is already inside the model weights; no separate semantic service is needed.

The code and this exported model are released under this repository's Apache-2.0
license. This does not grant access to UK Biobank or MIMIC-IV source data. Training
data and patient-level examples are not added by this package. No privacy guarantee
or full reproduction of the manuscript's multi-seed performance is implied by a
successful synthetic inference run.

## Training and re-export

For the prepared-data pipeline and training commands, use
[docs/REPRODUCE.md](../docs/REPRODUCE.md),
[Track R v2.2](../configs/track_r_v2_2/TRACK_R_v2_2.json), and
[Track G v1](../configs/track_g_v1/TRACK_G_v1.json). The main architecture is six
layers, eight heads, width 64. A2 uses absolute age encoding and
`age_rope_variant=additive_v2_2`; use the recorded wavelengths and input contract.
Training requires independently authorized cohort access and prepared assets;
the inference example does not train or recreate the paper's cohort results.

The narrowly scoped exporter accepts the registered seed42 A2 checkpoint, its
trusted SHA-256, matching `labels.csv`, resolved Track R v2.1 static protocol, and
Track G sampler-selection JSON:

```bash
python inference/export_checkpoint.py --checkpoint PATH --checkpoint-sha256 SHA256 --labels PATH --protocol PATH --sampler PATH --output NEW_MODEL_DIRECTORY
```

It uses restricted `weights_only=True` deserialization with a small explicit
allowlist for the historical NumPy RNG/path metadata, which is not exported. The
inference loader needs no such allowlist. Changing the vocabulary or checkpoint
requires a new bundle and compatibility validation, not swapping `model.pt` alone.

```bash
python -m unittest discover -s inference -p test_inference.py -v
```
