"""Unified configuration for scRAG-Flow pipeline stages."""
from dataclasses import dataclass, field


@dataclass
class Stage1Config:
    # Architecture
    hidden_dim: int = 256
    latent_dim: int = 128
    proj_dim: int = 64
    depth: int = 2
    dropout: float = 0.1

    # Loss weights
    alpha: float = 10.0
    beta: float = 0.5
    gamma: float = 0.5

    # Training
    epochs: int = 40
    lr: float = 4e-4
    weight_decay: float = 1e-4
    batch_size: int = 256

    # Gene selection
    n_top_genes: int = 3000
    use_per_protein_hvg: bool = True
    n_per_protein: int = 100
    use_protein_guided_hvg: bool = False

    # Data split
    train_ratio: float = 0.8
    seed: int = 42


@dataclass
class Stage3Config:
    # Architecture
    model: str = "film_gate"       # "film" | "film_gate"
    hidden_dim: int = 512
    depth: int = 2
    dropout: float = 0.1
    use_adaln_zero: bool = True    # True=AdaLN-Zero, False=Standard FiLM

    # Training
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    finetune_stage1: bool = True

    # Retrieval
    k_retrieval: int = 5
    val_ratio: float = 0.10        # fraction of total cells for validation

    # Input augmentation
    mask_prob: float = 0.0
    retrieval_noise: float = 0.0

    # Evaluation
    eval_every: int = 5
    seed: int = 42


# Preset configs for each dataset
def get_eccite_config():
    s1 = Stage1Config(n_top_genes=3000, use_per_protein_hvg=True, n_per_protein=100)
    s3 = Stage3Config(use_adaln_zero=True, mask_prob=0.0, model="film_gate")
    return s1, s3


def get_gse3p_config():
    s1 = Stage1Config(n_top_genes=1000, use_per_protein_hvg=True, n_per_protein=100)
    s3 = Stage3Config(use_adaln_zero=True, mask_prob=0.0, model="film_gate")
    return s1, s3


def get_cbmc_config():
    s1 = Stage1Config(n_top_genes=1000, use_protein_guided_hvg=True,
                       use_per_protein_hvg=False)
    s3 = Stage3Config(use_adaln_zero=False, mask_prob=0.2, model="film_gate")
    return s1, s3
