from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold


def _as_float32_dense(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def library_log_normalize(matrix, target_sum: float):
    if sp.issparse(matrix):
        result = matrix.tocsr(copy=True).astype(np.float32)
        row_sums = np.asarray(result.sum(axis=1)).reshape(-1)
        scale = target_sum / np.maximum(row_sums, 1e-8)
        result = sp.diags(scale) @ result
        result.data = np.log1p(result.data)
        return result.tocsr()
    array = np.asarray(matrix, dtype=np.float32)
    row_sums = array.sum(axis=1, keepdims=True)
    scaled = array * (target_sum / np.maximum(row_sums, 1e-8))
    return np.log1p(scaled).astype(np.float32)


@dataclass(frozen=True)
class LoadedDataset:
    rna_counts: object
    protein_counts: np.ndarray
    gene_names: np.ndarray
    protein_names: list[str]


@dataclass(frozen=True)
class PreparedFold:
    rna_counts: np.ndarray
    rna_normalized: np.ndarray
    protein_counts: np.ndarray
    protein_normalized: np.ndarray
    selected_genes: list[str]
    protein_names: list[str]


def load_h5ad_dataset(path: str | Path, protein_obsm_key: str) -> LoadedDataset:
    import anndata as ad

    adata = ad.read_h5ad(path)
    if protein_obsm_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{protein_obsm_key!r}].")
    protein_object = adata.obsm[protein_obsm_key]
    if isinstance(protein_object, pd.DataFrame):
        protein_names = [str(name) for name in protein_object.columns]
        protein_counts = protein_object.to_numpy(dtype=np.float32)
    else:
        protein_counts = np.asarray(protein_object, dtype=np.float32)
        protein_names = [f"protein_{index}" for index in range(protein_counts.shape[1])]
    if np.min(protein_counts) < 0:
        raise ValueError("Protein count input must be nonnegative.")
    return LoadedDataset(
        rna_counts=adata.X,
        protein_counts=protein_counts,
        gene_names=np.asarray(adata.var_names, dtype=str),
        protein_names=protein_names,
    )


def protein_targeted_gene_indices(
    rna_normalized_fit,
    protein_normalized_fit: np.ndarray,
    genes_per_protein: int,
    gene_cap: int,
) -> np.ndarray:
    n_cells, n_genes = rna_normalized_fit.shape
    n_proteins = protein_normalized_fit.shape[1]
    if n_cells != protein_normalized_fit.shape[0]:
        raise ValueError("RNA and protein matrices require aligned rows.")

    protein = np.asarray(protein_normalized_fit, dtype=np.float64)
    protein_centered = protein - protein.mean(axis=0, keepdims=True)
    protein_scale = np.sqrt(np.square(protein_centered).sum(axis=0))

    if sp.issparse(rna_normalized_fit):
        rna = rna_normalized_fit.tocsr().astype(np.float64)
        gene_sum = np.asarray(rna.sum(axis=0)).reshape(-1)
        gene_square_sum = np.asarray(rna.multiply(rna).sum(axis=0)).reshape(-1)
        cross = np.asarray(rna.T @ protein_centered)
    else:
        rna = np.asarray(rna_normalized_fit, dtype=np.float64)
        gene_sum = rna.sum(axis=0)
        gene_square_sum = np.square(rna).sum(axis=0)
        cross = rna.T @ protein_centered

    gene_scale = np.sqrt(
        np.maximum(gene_square_sum - np.square(gene_sum) / n_cells, 0.0)
    )
    denominator = gene_scale[:, None] * protein_scale[None, :]
    correlations = np.divide(
        np.abs(cross),
        denominator,
        out=np.zeros((n_genes, n_proteins), dtype=np.float64),
        where=denominator > 1e-12,
    )

    union: set[int] = set()
    per_protein = min(genes_per_protein, n_genes)
    for protein_index in range(n_proteins):
        scores = correlations[:, protein_index]
        top = np.argpartition(scores, -per_protein)[-per_protein:]
        union.update(int(index) for index in top)

    candidates = np.fromiter(union, dtype=np.int64)
    candidate_scores = correlations[candidates].max(axis=1)
    order = np.lexsort((candidates, -candidate_scores))
    return candidates[order[: min(gene_cap, candidates.size)]]


def prepare_fold_data(
    dataset: LoadedDataset,
    fit_indices: np.ndarray,
    normalization_target_sum: float,
    genes_per_protein: int,
    gene_cap: int,
) -> PreparedFold:
    rna_normalized_all = library_log_normalize(
        dataset.rna_counts, normalization_target_sum
    )
    protein_normalized_all = library_log_normalize(
        dataset.protein_counts, normalization_target_sum
    )
    selected_indices = protein_targeted_gene_indices(
        rna_normalized_all[fit_indices],
        protein_normalized_all[fit_indices],
        genes_per_protein,
        gene_cap,
    )
    rna_counts = _as_float32_dense(dataset.rna_counts[:, selected_indices])
    rna_normalized = _as_float32_dense(rna_normalized_all[:, selected_indices])
    return PreparedFold(
        rna_counts=rna_counts,
        rna_normalized=rna_normalized,
        protein_counts=np.asarray(dataset.protein_counts, dtype=np.float32),
        protein_normalized=np.asarray(protein_normalized_all, dtype=np.float32),
        selected_genes=dataset.gene_names[selected_indices].tolist(),
        protein_names=dataset.protein_names,
    )


def generate_five_fold_splits(
    n_cells: int,
    folds: int,
    seed: int,
    validation_fraction_of_reference: float,
) -> list[dict[str, list[int]]]:
    outer = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    splits: list[dict[str, list[int]]] = []
    all_indices = np.arange(n_cells)
    for reference_indices, test_indices in outer.split(all_indices):
        shuffled = rng.permutation(reference_indices)
        n_validation = max(
            1, int(round(len(shuffled) * validation_fraction_of_reference))
        )
        validation_indices = shuffled[:n_validation]
        fit_indices = shuffled[n_validation:]
        splits.append(
            {
                "fit": fit_indices.astype(int).tolist(),
                "validation": validation_indices.astype(int).tolist(),
                "test": test_indices.astype(int).tolist(),
            }
        )
    return splits


def validate_splits(
    splits: Iterable[dict[str, list[int]]], n_cells: int, folds: int
) -> list[dict[str, list[int]]]:
    splits = list(splits)
    if len(splits) != folds:
        raise ValueError(f"Expected {folds} folds, received {len(splits)}.")
    test_coverage: list[int] = []
    for fold_id, split in enumerate(splits):
        parts = {name: np.asarray(split[name], dtype=np.int64) for name in ("fit", "validation", "test")}
        joined = np.concatenate(list(parts.values()))
        if joined.size != np.unique(joined).size:
            raise ValueError(f"Fold {fold_id} contains overlapping partitions.")
        if joined.min(initial=0) < 0 or joined.max(initial=0) >= n_cells:
            raise ValueError(f"Fold {fold_id} contains an out-of-range index.")
        test_coverage.extend(parts["test"].tolist())
    if sorted(test_coverage) != list(range(n_cells)):
        raise ValueError("Test folds must cover every cell exactly once.")
    return splits


def load_or_create_splits(
    path: str | Path | None,
    n_cells: int,
    folds: int,
    seed: int,
    validation_fraction_of_reference: float,
) -> list[dict[str, list[int]]]:
    if path is None:
        splits = generate_five_fold_splits(
            n_cells, folds, seed, validation_fraction_of_reference
        )
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            splits = json.load(handle)
    return validate_splits(splits, n_cells, folds)
