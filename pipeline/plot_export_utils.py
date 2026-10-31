"""Utilities for exporting plot-ready intermediate files.

Drop this file into your project, for example:
    pipeline/plot_export_utils.py

It is designed to be used by run_cbmc_5cv_auto_export.py.

Exported contents include:
- Stage 1 z_rna / z_protein / p_rna / p_protein
- train / val / test metadata with optional UMAP coordinates
- FAISS top-k indices / cosine scores / softmax weights
- retrieval aggregate protein values
- Stage 3 delta / correction magnitude
- per-cell and per-protein metrics
- training curves
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import torch


CELL_TYPE_CANDIDATES = [
    "cell_type", "celltype", "celltypes", "CellType", "Cell_Type",
    "annotation", "annotations", "cell_type_major", "celltype_major",
    "major_cell_type", "cell_subtype", "subtype", "label", "labels",
    "cluster", "clusters", "seurat_clusters", "leiden", "louvain",
]

OPTIONAL_OBS_COLUMNS = [
    "batch", "donor", "condition", "sample", "sample_id", "patient",
    "dataset", "split", "stim", "treatment",
]


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape[0], dtype=np.float32)
    for i in range(a.shape[0]):
        out[i] = _safe_corr(a[i], b[i])
    return out


def colwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape[1], dtype=np.float32)
    for j in range(a.shape[1]):
        out[j] = _safe_corr(a[:, j], b[:, j])
    return out


def compute_per_cell_and_protein_metrics(
    truth: np.ndarray,
    retrieval: np.ndarray,
    prediction: np.ndarray,
    protein_names: Iterable[str],
    out_prefix: str | Path,
) -> None:
    """Save per-cell and per-protein metrics for plotting."""
    out_prefix = Path(out_prefix)
    _ensure_dir(out_prefix.parent)

    truth = np.asarray(truth, dtype=np.float32)
    retrieval = np.asarray(retrieval, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    protein_names = list(map(str, protein_names))

    ret_cell_pcc = rowwise_corr(truth, retrieval)
    pred_cell_pcc = rowwise_corr(truth, prediction)
    ret_cell_rmse = np.sqrt(np.mean((retrieval - truth) ** 2, axis=1))
    pred_cell_rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=1))
    ret_cell_mae = np.mean(np.abs(retrieval - truth), axis=1)
    pred_cell_mae = np.mean(np.abs(prediction - truth), axis=1)

    cell_df = pd.DataFrame({
        "cell_local_index": np.arange(truth.shape[0]),
        "retrieval_pcc": ret_cell_pcc,
        "stage3_pcc": pred_cell_pcc,
        "delta_pcc": pred_cell_pcc - ret_cell_pcc,
        "retrieval_rmse": ret_cell_rmse,
        "stage3_rmse": pred_cell_rmse,
        "rmse_reduction": ret_cell_rmse - pred_cell_rmse,
        "retrieval_mae": ret_cell_mae,
        "stage3_mae": pred_cell_mae,
        "mae_reduction": ret_cell_mae - pred_cell_mae,
    })
    cell_df.to_csv(f"{out_prefix}_per_cell_metrics.csv", index=False)

    ret_prot_pcc = colwise_corr(truth, retrieval)
    pred_prot_pcc = colwise_corr(truth, prediction)
    ret_prot_rmse = np.sqrt(np.mean((retrieval - truth) ** 2, axis=0))
    pred_prot_rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=0))
    ret_prot_mae = np.mean(np.abs(retrieval - truth), axis=0)
    pred_prot_mae = np.mean(np.abs(prediction - truth), axis=0)
    mean_true = truth.mean(axis=0)
    var_true = truth.var(axis=0)

    prot_df = pd.DataFrame({
        "protein": protein_names,
        "mean_true_expression": mean_true,
        "true_variance": var_true,
        "retrieval_pcc": ret_prot_pcc,
        "stage3_pcc": pred_prot_pcc,
        "delta_pcc": pred_prot_pcc - ret_prot_pcc,
        "retrieval_rmse": ret_prot_rmse,
        "stage3_rmse": pred_prot_rmse,
        "rmse_reduction": ret_prot_rmse - pred_prot_rmse,
        "retrieval_mae": ret_prot_mae,
        "stage3_mae": pred_prot_mae,
        "mae_reduction": ret_prot_mae - pred_prot_mae,
    })
    prot_df.to_csv(f"{out_prefix}_per_protein_metrics.csv", index=False)


def export_stage1_embeddings(
    model,
    rna_np: np.ndarray,
    protein_np: np.ndarray,
    device: str,
    out_prefix: str | Path,
    batch_size: int = 1024,
) -> dict[str, tuple[int, ...]]:
    """Export z_rna, z_protein, p_rna, p_protein for one split.

    The model must return these keys in forward(), as ContrastiveAutoencoder does.
    """
    out_prefix = Path(out_prefix)
    _ensure_dir(out_prefix.parent)

    model.eval()
    rna_np = _as_float32(rna_np)
    protein_np = _as_float32(protein_np)

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(rna_np),
        torch.from_numpy(protein_np),
    )

    z_rna_list, z_protein_list, p_rna_list, p_protein_list = [], [], [], []
    with torch.no_grad():
        for rb, pb in torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False):
            rb = rb.to(device)
            pb = pb.to(device)
            out = model(rb, pb)
            z_rna_list.append(out["z_rna"].detach().cpu().numpy())
            z_protein_list.append(out["z_protein"].detach().cpu().numpy())
            p_rna_list.append(out["p_rna"].detach().cpu().numpy())
            p_protein_list.append(out["p_protein"].detach().cpu().numpy())

    arrays = {
        "z_rna": np.concatenate(z_rna_list, axis=0).astype(np.float32),
        "z_protein": np.concatenate(z_protein_list, axis=0).astype(np.float32),
        "p_rna": np.concatenate(p_rna_list, axis=0).astype(np.float32),
        "p_protein": np.concatenate(p_protein_list, axis=0).astype(np.float32),
    }

    for name, arr in arrays.items():
        np.save(f"{out_prefix}_{name}.npy", arr)

    return {name: arr.shape for name, arr in arrays.items()}


def detect_cell_type_column(adata) -> Optional[str]:
    obs_cols = list(map(str, adata.obs.columns))
    for cand in CELL_TYPE_CANDIDATES:
        if cand in obs_cols:
            return cand
    return None


def export_metadata_umap(
    adata,
    idx: np.ndarray,
    split_name: str,
    out_csv: str | Path,
    cell_type_col: Optional[str] = None,
) -> pd.DataFrame:
    """Export metadata aligned with a split array order.

    The output always has:
    - cell_local_index
    - cell_original_index
    - split

    If available, it also exports:
    - cell_id from adata.obs_names
    - cell_type
    - UMAP1 / UMAP2 from adata.obsm["X_umap"]
    - common optional obs columns, e.g. batch / donor / condition
    """
    out_csv = Path(out_csv)
    _ensure_dir(out_csv.parent)

    idx = np.asarray(idx)
    obs = adata.obs.iloc[idx].copy()
    if cell_type_col is None:
        cell_type_col = detect_cell_type_column(adata)

    meta = pd.DataFrame({
        "cell_local_index": np.arange(len(idx)),
        "cell_original_index": idx,
        "cell_id": adata.obs_names[idx].astype(str),
        "split": split_name,
    })

    if cell_type_col is not None and cell_type_col in obs.columns:
        meta["cell_type"] = obs[cell_type_col].astype(str).values
    else:
        meta["cell_type"] = "unknown"

    for col in OPTIONAL_OBS_COLUMNS:
        if col in obs.columns and col not in meta.columns:
            meta[col] = obs[col].astype(str).values

    if "X_umap" in adata.obsm:
        um = np.asarray(adata.obsm["X_umap"])[idx]
        if um.ndim == 2 and um.shape[1] >= 2:
            meta["UMAP1"] = um[:, 0]
            meta["UMAP2"] = um[:, 1]

    meta.to_csv(out_csv, index=False)
    return meta


def export_pca2_coordinates(
    features: np.ndarray,
    out_csv: str | Path,
    prefix: str = "PC",
) -> None:
    """Export 2D PCA coordinates for embeddings when UMAP is absent."""
    out_csv = Path(out_csv)
    _ensure_dir(out_csv.parent)
    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=0).fit_transform(features)
        pd.DataFrame({
            f"{prefix}1": coords[:, 0],
            f"{prefix}2": coords[:, 1],
        }).to_csv(out_csv, index=False)
    except Exception as exc:
        pd.DataFrame({"error": [str(exc)]}).to_csv(out_csv, index=False)


def _softmax_weights(scores: np.ndarray) -> np.ndarray:
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    return weights.astype(np.float32)


def aggregate_from_query_result(bank, indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
    weights = _softmax_weights(scores)
    neighbors = bank.payload[indices]
    return np.einsum("bk,bkd->bd", weights, neighbors).astype(np.float32)


def export_retrieval_query(
    bank,
    features: np.ndarray,
    k: int,
    exclude_self: bool,
    out_prefix: str | Path,
    query_batch: int = 10000,
) -> dict[str, tuple[int, ...]]:
    """Save FAISS/sklearn top-k indices, scores, weights, and aggregate.

    This uses MemoryBank.query(), so it works with both FAISS and sklearn fallback.
    """
    out_prefix = Path(out_prefix)
    _ensure_dir(out_prefix.parent)

    features = _as_float32(features)
    all_indices, all_scores, all_agg = [], [], []

    for start in range(0, len(features), query_batch):
        batch = features[start:start + query_batch]
        result = bank.query(batch, k=k, exclude_self=exclude_self)
        agg = aggregate_from_query_result(bank, result.indices, result.scores)
        all_indices.append(result.indices.astype(np.int64))
        all_scores.append(result.scores.astype(np.float32))
        all_agg.append(agg)

    indices = np.concatenate(all_indices, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    weights = _softmax_weights(scores)
    aggregate = np.concatenate(all_agg, axis=0)

    np.save(f"{out_prefix}_topk_indices.npy", indices)
    np.save(f"{out_prefix}_topk_scores.npy", scores)
    np.save(f"{out_prefix}_topk_weights.npy", weights)
    np.save(f"{out_prefix}_retrieval_aggregate.npy", aggregate)

    return {
        "topk_indices": indices.shape,
        "topk_scores": scores.shape,
        "topk_weights": weights.shape,
        "retrieval_aggregate": aggregate.shape,
    }


def save_training_curve_csv(records: list[dict], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    _ensure_dir(out_csv.parent)
    pd.DataFrame(records).to_csv(out_csv, index=False)


def save_manifest(path: str | Path, manifest: dict) -> None:
    path = Path(path)
    _ensure_dir(path.parent)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(manifest), f, indent=2, ensure_ascii=False)
