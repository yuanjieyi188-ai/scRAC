from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PreprocessingConfig:
    normalization_target_sum: float
    genes_per_protein: int
    gene_cap: int


@dataclass(frozen=True)
class Stage1Config:
    hidden_dim: int
    latent_dim: int
    projection_dim: int
    depth: int
    dropout: float
    contrastive_weight: float
    rna_zinb_weight: float
    protein_zinb_weight: float
    temperature_init: float
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_clip: float


@dataclass(frozen=True)
class RetrievalConfig:
    index: str
    neighbors: int
    temperature: float
    exact_cosine: bool


@dataclass(frozen=True)
class Stage3Config:
    condition_mode: str
    hidden_dim: int
    depth: int
    time_dim: int
    dropout: float
    mask_probability: float
    endpoint_weight: float
    source_noise_std: float
    finetune_rna_encoder: bool
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_clip: float
    validation_interval: int
    solver: str
    nfe: int


@dataclass(frozen=True)
class ExperimentConfig:
    folds: int
    seed: int
    validation_fraction_of_reference: float
    protein_obsm_key: str


@dataclass(frozen=True)
class PaperConfig:
    method: str
    preprocessing: PreprocessingConfig
    stage1: Stage1Config
    retrieval: RetrievalConfig
    stage3: Stage3Config
    experiment: ExperimentConfig


_EXPECTED: dict[str, dict[str, Any]] = {
    "preprocessing": {
        "normalization_target_sum": 10000.0,
        "genes_per_protein": 100,
        "gene_cap": 1000,
    },
    "stage1": {
        "hidden_dim": 256,
        "latent_dim": 128,
        "projection_dim": 64,
        "depth": 2,
        "dropout": 0.1,
        "contrastive_weight": 10.0,
        "rna_zinb_weight": 0.5,
        "protein_zinb_weight": 0.5,
        "temperature_init": 0.07,
        "epochs": 20,
        "learning_rate": 4e-4,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "gradient_clip": 1.0,
    },
    "retrieval": {
        "index": "IndexFlatIP",
        "neighbors": 5,
        "temperature": 1.0,
        "exact_cosine": True,
    },
    "stage3": {
        "condition_mode": "query_similarity",
        "hidden_dim": 512,
        "depth": 4,
        "time_dim": 64,
        "dropout": 0.1,
        "mask_probability": 0.2,
        "endpoint_weight": 1.0,
        "source_noise_std": 0.0,
        "finetune_rna_encoder": True,
        "epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 1024,
        "gradient_clip": 1.0,
        "validation_interval": 1,
        "solver": "euler",
        "nfe": 1,
    },
    "experiment": {"folds": 5},
}


def _validate_paper_values(raw: dict[str, Any]) -> None:
    if raw.get("method") != "scRAC":
        raise ValueError("The paper configuration must use method=scRAC.")
    for section, expected_values in _EXPECTED.items():
        actual = raw.get(section, {})
        for key, expected in expected_values.items():
            value = actual.get(key)
            if value != expected:
                raise ValueError(
                    f"Paper configuration mismatch: {section}.{key}={value!r}; "
                    f"expected {expected!r}."
                )


def load_paper_config(path: str | Path) -> PaperConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _validate_paper_values(raw)
    return PaperConfig(
        method=raw["method"],
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        stage1=Stage1Config(**raw["stage1"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        stage3=Stage3Config(**raw["stage3"]),
        experiment=ExperimentConfig(**raw["experiment"]),
    )
