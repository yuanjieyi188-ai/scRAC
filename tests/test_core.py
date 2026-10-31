from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scrac.config import load_paper_config
from scrac.data import protein_targeted_gene_indices
from scrac.metrics import rbf_mmd
from scrac.losses import flow_matching_objective
from scrac.models import RetrievalConditionedFlow


def test_paper_config_values() -> None:
    config = load_paper_config(ROOT / "configs" / "paper.yaml")
    assert config.stage1.epochs == 20
    assert config.stage1.batch_size == 256
    assert config.retrieval.neighbors == 5
    assert config.stage3.depth == 4
    assert config.stage3.mask_probability == 0.2
    assert config.stage3.nfe == 1


def test_flow_architecture_and_zero_initialization() -> None:
    model = RetrievalConditionedFlow(protein_dim=13)
    assert len(model.blocks) == 4
    x = torch.randn(5, 13)
    latent = torch.randn(5, 128)
    similarity = torch.randn(5, 1)
    velocity = model(x, torch.zeros(5), latent, similarity)
    assert torch.allclose(velocity, torch.zeros_like(velocity))


def test_endpoint_uses_unmasked_source() -> None:
    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: list[torch.Tensor] = []

        def forward(self, x, t, latent, similarity):
            self.inputs.append(x.detach().clone())
            return torch.zeros_like(x)

    torch.manual_seed(7)
    model = RecordingModel()
    source = torch.ones(4, 3)
    target = torch.zeros(4, 3)
    latent = torch.zeros(4, 128)
    similarity = torch.zeros(4, 1)
    flow_matching_objective(
        model,
        source,
        target,
        latent,
        similarity,
        mask_probability=1.0,
        endpoint_weight=1.0,
    )
    assert torch.allclose(model.inputs[0], torch.zeros_like(source))
    assert torch.allclose(model.inputs[1], source)


def test_readme_scope_notice() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "core scRAC model and benchmark workflow" in text
    assert "complete project code will be provided after the peer-review process" in text


def test_protein_targeted_gene_screening() -> None:
    rna = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    protein = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    selected = protein_targeted_gene_indices(
        rna, protein, genes_per_protein=1, gene_cap=1
    )
    assert selected.tolist() == [0]


def test_mmd_identity() -> None:
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    assert abs(rbf_mmd(values, values)) < 1e-12


def test_retrieval_self_exclusion(monkeypatch) -> None:
    import scrac.retrieval as retrieval

    class FakeIndexFlatIP:
        def __init__(self, dimension: int) -> None:
            self.dimension = dimension
            self.features = None

        def add(self, features: np.ndarray) -> None:
            self.features = features.copy()

        def search(self, query: np.ndarray, k: int):
            similarities = query @ self.features.T
            indices = np.argsort(-similarities, axis=1)[:, :k]
            ordered = np.take_along_axis(similarities, indices, axis=1)
            return ordered.astype(np.float32), indices.astype(np.int64)

    class FakeFaiss:
        IndexFlatIP = FakeIndexFlatIP

    monkeypatch.setattr(retrieval, "faiss", FakeFaiss)
    features = np.eye(4, dtype=np.float32)
    proteins = np.arange(8, dtype=np.float32).reshape(4, 2)
    memory = retrieval.ExactProteinMemory(temperature=1.0)
    memory.fit(features, proteins)
    result = memory.retrieve(
        features,
        neighbors=2,
        self_indices=np.arange(4, dtype=np.int64),
    )
    assert np.all(result.indices != np.arange(4)[:, None])
    assert np.allclose(result.weights.sum(axis=1), 1.0)


def test_dataset_release_files() -> None:
    dataset_readme = (ROOT / "datasets" / "README.md").read_text(encoding="utf-8")
    assert "GSE100866" in dataset_readme
    assert "GSE164378" in dataset_readme
    assert "GSM5008740" in dataset_readme
    assert "GSM5008737" in dataset_readme
    assert (ROOT / "datasets" / "download_geo.sh").is_file()
    assert (ROOT / "datasets" / "prepare_h5ad.py").is_file()
