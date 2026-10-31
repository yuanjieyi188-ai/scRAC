from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


def l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, eps)


@dataclass(frozen=True)
class RetrievalResult:
    indices: np.ndarray
    similarities: np.ndarray
    weights: np.ndarray
    protein_source: np.ndarray
    mean_similarity: np.ndarray


class ExactProteinMemory:
    def __init__(self, temperature: float = 1.0) -> None:
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required for the paper retrieval implementation."
            ) from _FAISS_IMPORT_ERROR
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)
        self.features: np.ndarray | None = None
        self.proteins: np.ndarray | None = None
        self.index = None

    def fit(self, rna_latents: np.ndarray, proteins: np.ndarray) -> None:
        features = l2_normalize(rna_latents)
        proteins = np.asarray(proteins, dtype=np.float32)
        if features.shape[0] != proteins.shape[0]:
            raise ValueError("RNA latents and proteins require the same row count.")
        self.features = np.ascontiguousarray(features)
        self.proteins = np.ascontiguousarray(proteins)
        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(self.features)

    def retrieve(
        self,
        query_latents: np.ndarray,
        neighbors: int,
        self_indices: np.ndarray | None = None,
    ) -> RetrievalResult:
        if self.index is None or self.features is None or self.proteins is None:
            raise RuntimeError("Call fit before retrieve.")
        query = np.ascontiguousarray(l2_normalize(query_latents))
        n_bank = self.features.shape[0]
        if self_indices is None:
            if not 1 <= neighbors <= n_bank:
                raise ValueError("neighbors exceeds the memory size.")
            similarities, indices = self.index.search(query, neighbors)
        else:
            if not 1 <= neighbors < n_bank:
                raise ValueError("Self-excluding retrieval requires neighbors < memory size.")
            raw_similarities, raw_indices = self.index.search(query, neighbors + 1)
            self_indices = np.asarray(self_indices, dtype=np.int64)
            indices = np.empty((query.shape[0], neighbors), dtype=np.int64)
            similarities = np.empty((query.shape[0], neighbors), dtype=np.float32)
            for row in range(query.shape[0]):
                keep = raw_indices[row] != self_indices[row]
                indices[row] = raw_indices[row][keep][:neighbors]
                similarities[row] = raw_similarities[row][keep][:neighbors]

        logits = similarities / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        protein_source = np.einsum(
            "nk,nkp->np", weights, self.proteins[indices]
        ).astype(np.float32)
        mean_similarity = np.sum(weights * similarities, axis=1, keepdims=True).astype(
            np.float32
        )
        return RetrievalResult(
            indices=indices.astype(np.int64),
            similarities=similarities.astype(np.float32),
            weights=weights,
            protein_source=protein_source,
            mean_similarity=mean_similarity,
        )
