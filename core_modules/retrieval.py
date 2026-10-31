"""Exact retrieval backend with FAISS-first and sklearn fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None


@dataclass
class RetrievalResult:
    indices: np.ndarray
    scores: np.ndarray


class MemoryBank:
    def __init__(self, use_faiss: bool = True) -> None:
        self.use_faiss = use_faiss and faiss is not None
        self.index = None
        self.features: Optional[np.ndarray] = None
        self.payload: Optional[np.ndarray] = None
        self.nn: Optional[NearestNeighbors] = None

    def fit(self, features: np.ndarray, payload: np.ndarray) -> None:
        feats = features.astype(np.float32, copy=False)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        self.features = feats
        self.payload = payload.astype(np.float32, copy=False)
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(feats.shape[1])
            self.index.add(feats)
        else:
            self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
            self.nn.fit(feats)

    def query(self, features: np.ndarray, k: int, exclude_self: bool = False) -> RetrievalResult:
        feats = features.astype(np.float32, copy=False)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        query_k = k + 1 if exclude_self else k
        if self.use_faiss:
            scores, indices = self.index.search(feats, query_k)
        else:
            distances, indices = self.nn.kneighbors(feats, n_neighbors=query_k)
            scores = 1.0 - distances
        if exclude_self:
            indices = indices[:, 1:]
            scores = scores[:, 1:]
        return RetrievalResult(indices=indices, scores=scores)

    def aggregate(self, features: np.ndarray, k: int, exclude_self: bool = False, query_batch: int = 10000) -> np.ndarray:
        if len(features) <= query_batch:
            result = self.query(features, k=k, exclude_self=exclude_self)
            weights = np.exp(result.scores - result.scores.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True) + 1e-8
            neighbors = self.payload[result.indices]
            return np.einsum("bk,bkd->bd", weights, neighbors).astype(np.float32)
        # Batch queries to avoid GPU OOM
        results = []
        for i in range(0, len(features), query_batch):
            batch = features[i:i + query_batch]
            result = self.query(batch, k=k, exclude_self=exclude_self)
            weights = np.exp(result.scores - result.scores.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True) + 1e-8
            neighbors = self.payload[result.indices]
            results.append(np.einsum("bk,bkd->bd", weights, neighbors).astype(np.float32))
        return np.concatenate(results, axis=0)
