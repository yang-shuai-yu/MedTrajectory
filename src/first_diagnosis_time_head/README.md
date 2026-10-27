# First-diagnosis time head

Training and evaluation code for the **independent first-diagnosis time head** used for the
five methods whose native output carries no operable continuous time — GPT, BERT, ETHOS,
Med-BERT and TrajGPT. These are the `Time MAE` values reported in Table III of the
manuscript and, with their rollout-pair availability, in Supplementary Table S13e.

The five backbones are frozen; only the time head is trained. The head therefore measures
how much timing information is recoverable from a frozen representation, not how well a
model generates time natively. `Time MAE` in the manuscript is conditional on an available
predicted–reference pair.

## Files

| File | Contents |
|---|---|
| `head.py` | The head itself: network, censored likelihood and the `nll` helper |
| `core.py` | Example construction (history, cut, first-diagnosis target) and metric helpers |
| `engine.py` | Training, model selection and evaluation driver |
| `model_adapter.py` | Feature extraction from the five frozen backbones into each head |
| `assets.py` | Pinned data/model identity checks and the input/output path contract |
| `protocol.json` | Frozen experimental settings: methods, horizon, cut rule, target, censoring, partitions, training and evaluation schedule |
| `verify.py` | Re-derives the exported summaries from the prepared inputs and checks them against the saved receipts |
| `test_core.py` | Unit tests for `core.py` (`python -m unittest test_core`) |

## What is not included

- **Frozen checkpoints and prepared inputs.** They are derived from access-controlled UK
  Biobank data; see `docs/DATA.md`.
- **The remote job orchestration.** The run copy of this experiment also contained
  `deploy.py` and `supervisor.py`, which drive detached workers over an institutional
  cluster (host aliases, `fcntl` locks, per-method handsets). They are omitted here because
  they are site-specific and not needed to understand or re-implement the method.
- **`config.json`**, the machine-local path overrides. `assets.py` tolerates its absence.

## Path contract

The published copy differs from the run copy **only** in how two roots are resolved. The run
copy pinned them to an institutional filesystem; here they come from environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `MEDTRAJECTORY_CODE_ROOT` | tree holding `releases/` and `ops/` | this directory |
| `MEDTRAJECTORY_DATA_ROOT` | tree holding `runs/`, `snapshots/` and `data/` | `./data` under this directory |

All other identities are checked from `protocol.json` and from the snapshot receipts, so a
wrong input is rejected rather than silently scored.

## Running

```bash
cd src/first_diagnosis_time_head
python -m unittest test_core          # unit tests, no data required
python engine.py --method gpt         # requires the prepared inputs and frozen checkpoints
```

`engine.py --method` accepts the five method names in `protocol.json`. Evaluation writes
only inside the run and ops trees, by construction (`assets.save` refuses any other target).
