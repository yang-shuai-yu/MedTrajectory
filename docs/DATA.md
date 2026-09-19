# =============================================================================
# Data access statement
#
# This repository contains SOURCE CODE ONLY. No participant-level data, model
# checkpoints, or prediction dumps are distributed here.
# =============================================================================

## 1. Why no data is included

Both cohorts used in this work are access-controlled and **cannot be
redistributed** by us. Their terms do not transfer with the code, and the
Apache-2.0 license of this repository does not grant any rights over them.

| Dataset | Access route | Notes |
|---|---|---|
| **UK Biobank** | Application to UK Biobank (https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access). Access is granted per approved project and per approved researcher. | All UKB-derived analyses in this work were performed under the project's approved application. |
| **MIMIC-IV** | Credentialed access via PhysioNet (https://physionet.org/content/mimiciv/). Requires completion of the CITI "Data or Specimens Only Research" training and signing of the data use agreement. | Version used: **MIMIC-IV 3.1**. |

Users must obtain their own access and reconstruct the input files locally. The
scripts in this repository expect the datasets to be placed under paths given by
environment variables (see §4).

## 2. Reproducing the input layout

The pipeline consumes preprocessed, tokenised event sequences rather than raw
source tables. The preparation scripts are included:

| Cohort | Preparation entry point |
|---|---|
| UK Biobank | `src/semantic_delphi_ukb/prepare_paper_protocol_dataset.py`, `src/semantic_delphi_ukb/prepare_multitype_dataset.py`, `src/semantic_delphi_ukb/build_multitype.py` |
| MIMIC-IV | `src/medtrajectory_mimic/mimic_build_dataset.py` (event level), `mimic_build_visit.py` (visit level), `mimic_build_track_r.py` (Track-R static prefix and anchor) |

## 3. Third-party components

Some artefacts depend on third-party services or models whose own terms apply:

- **Qwen `text-embedding-v4`** (Alibaba Cloud DashScope) is used to initialise
  the MIMIC-IV semantic token embeddings (1024-d, PCA-reduced to 64-d). The
  generator is `src/medtrajectory_mimic/mimic_qwen_embed.py`. **It reads its
  credential from the `DASHSCOPE_API_KEY` environment variable**; no key is
  stored in this repository. Running it requires your own DashScope account and
  is subject to Alibaba Cloud's terms of service.
- **PubMedBERT / BiomedCLIP** are used only in the encoding-method comparison
  (`scripts/compare_icd_embeddings.py`) and are subject to their own licenses.
- **Phecode maps** are derived from the PheWAS/phecode resources
  (https://phewascatalog.org/); see `src/medtrajectory_mimic/mimic_qwen_embed.py`
  for how they are consumed.

Precomputed embedding matrices are **not** distributed here, because they are
derived from the access-controlled vocabularies. They can be regenerated with
the scripts above once you have the underlying data.

## 4. Expected environment variables

After the path-configuration refactor (see `UPLOAD_MANIFEST.md` §5.1) the code
resolves paths through:

| Variable | Meaning | Default |
|---|---|---|
| `MEDTRAJECTORY_ROOT` | repository root | resolved from the module location |
| `MEDTRAJECTORY_DATA_ROOT` | prepared dataset root | `<root>/data` |
| `MEDTRAJECTORY_RUNS_ROOT` | run output root | `<root>/results` |
| `DASHSCOPE_API_KEY` | credential for the optional embedding regeneration | unset |

## 5. Ethics

UK Biobank and MIMIC-IV are de-identified resources governed by their own
ethics and data-governance frameworks. **The exact ethics approval identifiers
and data-application identifiers for this project must be supplied by the
authors before publication** — see the placeholders noted in the manuscript
package (Supplementary S1.4).
