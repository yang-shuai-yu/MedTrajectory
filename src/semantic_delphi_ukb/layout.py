from __future__ import annotations
try:  # path decoupling: see src/semantic_delphi_ukb/paths.py
    from semantic_delphi_ukb.paths import EXTERNAL_ROOT
except ImportError:  # executed from inside the source tree
    from paths import EXTERNAL_ROOT

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = REPO_DIR.parent

REMOTE_CODE_ROOT = Path(str(EXTERNAL_ROOT))
REMOTE_DATASET_ROOT = Path(str(EXTERNAL_ROOT))


@dataclass(frozen=True)
class ProjectLayout:
    environment: str
    repo_dir: Path
    workspace_dir: Path
    dataset_root: Path
    repo_data_root: Path
    outputs_root: Path
    tokens_jsonl: Path
    tokens_noevents_jsonl: Path
    ukb_extract_root: Path
    cohort_root: Path


@dataclass(frozen=True)
class MultitypePaths:
    data_version: str
    experiment: str
    dataset_root: Path
    multitype_root: Path
    canonical_dir: Path
    vocab_dir: Path
    model_input_dir: Path
    patient_records_core_jsonl: Path
    patient_records_extended_jsonl: Path
    dynamic_token_vocab_csv: Path
    dynamic_token_types_csv: Path
    static_schema_json: Path
    tokens_exp0_diag_jsonl: Path
    tokens_exp1_diag_proc_jsonl: Path
    tokens_exp2_diag_proc_cancer_death_jsonl: Path
    static_features_v1_csv: Path
    experiment_dir: Path
    train_bin: Path
    val_bin: Path
    test_bin: Path
    labels_csv: Path
    patient_index_csv: Path
    train_static_npy: Path
    val_static_npy: Path
    test_static_npy: Path
    semantic_input_embeddings_npy: Path
    prepare_manifest_json: Path


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _normalize_data_version(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("data_version cannot be empty.")
    return value if value.startswith("v") else f"v{value}"


def _normalize_experiment(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("experiment cannot be empty.")
    return value if value.startswith("exp") else f"exp{value}"


def resolve_project_layout(dataset_root: Optional[Path] = None) -> ProjectLayout:
    repo_dir = _normalize_path(REPO_DIR)
    workspace_dir = _normalize_path(WORKSPACE_DIR)

    if dataset_root is not None:
        resolved_dataset_root = _normalize_path(dataset_root)
        environment = "custom_dataset_root"
    else:
        local_dataset_root = workspace_dir / "dataset"
        if local_dataset_root.exists():
            resolved_dataset_root = _normalize_path(local_dataset_root)
            environment = "local_workspace"
        elif REMOTE_DATASET_ROOT.exists():
            resolved_dataset_root = REMOTE_DATASET_ROOT
            environment = "remote_server"
        else:
            resolved_dataset_root = _normalize_path(local_dataset_root)
            environment = "fallback_workspace"

    return ProjectLayout(
        environment=environment,
        repo_dir=repo_dir,
        workspace_dir=workspace_dir,
        dataset_root=resolved_dataset_root,
        repo_data_root=repo_dir / "data",
        outputs_root=repo_dir / "outputs",
        tokens_jsonl=resolved_dataset_root / "tokens.jsonl",
        tokens_noevents_jsonl=resolved_dataset_root / "tokens_noevents.jsonl",
        ukb_extract_root=resolved_dataset_root / "UKB_extract",
        cohort_root=resolved_dataset_root / "cohort_stratification",
    )


def resolve_multitype_paths(
    project_layout: ProjectLayout,
    data_version: str = "v1",
    experiment: str = "exp2",
) -> MultitypePaths:
    data_version = _normalize_data_version(data_version)
    experiment = _normalize_experiment(experiment)

    multitype_root = project_layout.dataset_root / f"ukb_multitype_{data_version}"
    canonical_dir = multitype_root / "canonical"
    vocab_dir = multitype_root / "vocab"
    model_input_dir = multitype_root / "model_input"

    experiment_dir = project_layout.repo_data_root / f"ukb_semantic_multitype_{experiment}"

    return MultitypePaths(
        data_version=data_version,
        experiment=experiment,
        dataset_root=project_layout.dataset_root,
        multitype_root=multitype_root,
        canonical_dir=canonical_dir,
        vocab_dir=vocab_dir,
        model_input_dir=model_input_dir,
        patient_records_core_jsonl=canonical_dir / "patient_records_core.jsonl",
        patient_records_extended_jsonl=canonical_dir / "patient_records_extended.jsonl",
        dynamic_token_vocab_csv=vocab_dir / "dynamic_token_vocab.csv",
        dynamic_token_types_csv=vocab_dir / "dynamic_token_types.csv",
        static_schema_json=vocab_dir / "static_schema.json",
        tokens_exp0_diag_jsonl=model_input_dir / "tokens_exp0_diag.jsonl",
        tokens_exp1_diag_proc_jsonl=model_input_dir / "tokens_exp1_diag_proc.jsonl",
        tokens_exp2_diag_proc_cancer_death_jsonl=model_input_dir / "tokens_exp2_diag_proc_cancer_death.jsonl",
        static_features_v1_csv=model_input_dir / "static_features_v1.csv",
        experiment_dir=experiment_dir,
        train_bin=experiment_dir / "train.bin",
        val_bin=experiment_dir / "val.bin",
        test_bin=experiment_dir / "test.bin",
        labels_csv=experiment_dir / "labels.csv",
        patient_index_csv=experiment_dir / "patient_index.csv",
        train_static_npy=experiment_dir / "train_static.npy",
        val_static_npy=experiment_dir / "val_static.npy",
        test_static_npy=experiment_dir / "test_static.npy",
        semantic_input_embeddings_npy=experiment_dir / "semantic_input_embeddings_64d.npy",
        prepare_manifest_json=experiment_dir / "prepare_manifest.json",
    )
