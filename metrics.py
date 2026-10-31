# -*- coding: utf-8 -*-
"""
scChord Evaluation Metrics

This module provides evaluation metrics including:
- PCC (Pearson Correlation Coefficient): protein-protein, cell-cell
- RMSE (Root Mean Square Error)
- CMD (Correlation Matrix Distance)
- MMD (Maximum Mean Discrepancy)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from typing import Tuple, Optional
import torch


def compute_pcc(
    pred: np.ndarray,
    true: np.ndarray,
    protein_names: Optional[list] = None,
    cell_ids: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    n_cells, n_proteins = pred.shape
    pcc_protein = []
    for j in range(n_proteins):
        x = pred[:, j]
        y = true[:, j]
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            pcc, _ = pearsonr(x, y)
        else:
            pcc = 0.0
        pcc_protein.append(pcc)
    pcc_protein = np.array(pcc_protein)

    pcc_cell = []
    for i in range(n_cells):
        x = pred[i, :]
        y = true[i, :]
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            pcc, _ = pearsonr(x, y)
        else:
            pcc = 0.0
        pcc_cell.append(pcc)
    pcc_cell = np.array(pcc_cell)

    pcc_protein_mean = np.nanmean(pcc_protein)
    pcc_cell_mean = np.nanmean(pcc_cell)

    return pcc_protein, pcc_cell, pcc_protein_mean, pcc_cell_mean


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    import scanpy as sc
    import anndata

    pred_adata = anndata.AnnData(pred.copy())
    true_adata = anndata.AnnData(true.copy())

    sc.pp.scale(pred_adata, zero_center=True, copy=False)
    sc.pp.scale(true_adata, zero_center=True, copy=False)

    pred_scaled = pred_adata.X
    true_scaled = true_adata.X

    rmse = np.sqrt(metrics.mean_squared_error(
        true_scaled.flatten(),
        pred_scaled.flatten()
    ))
    return rmse


def cmd_dist(A: np.ndarray, B: np.ndarray) -> float:
    a = np.multiply(A, B).sum()
    b = np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')
    return 1 - a / (b + 1e-8)


def compute_cmd(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    pred_df = pd.DataFrame(pred)
    true_df = pd.DataFrame(true)

    pred_cell_corr = pred_df.T.corr()
    true_cell_corr = true_df.T.corr()
    pred_cell_corr = pred_cell_corr.fillna(0)
    true_cell_corr = true_cell_corr.fillna(0)
    cmd_cell = cmd_dist(pred_cell_corr.values, true_cell_corr.values)

    pred_prot_corr = pred_df.corr()
    true_prot_corr = true_df.corr()
    pred_prot_corr = pred_prot_corr.fillna(0)
    true_prot_corr = true_prot_corr.fillna(0)
    cmd_protein = cmd_dist(pred_prot_corr.values, true_prot_corr.values)

    return cmd_cell, cmd_protein


def compute_mmd(x: np.ndarray, y: np.ndarray, kernel='rbf', max_samples: int = 2000) -> float:
    # Subsample to avoid O(N^2) memory blowup on large test sets
    rng = np.random.default_rng(42)
    if x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], max_samples, replace=False)
        x = x[idx]
    if y.shape[0] > max_samples:
        idx = rng.choice(y.shape[0], max_samples, replace=False)
        y = y[idx]
    x_torch = torch.from_numpy(x).float()
    y_torch = torch.from_numpy(y).float()
    x_kernel = compute_kernel(x_torch, x_torch)
    y_kernel = compute_kernel(y_torch, y_torch)
    xy_kernel = compute_kernel(x_torch, y_torch)
    mmd_value = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd_value.item()


def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / 1.0)


def evaluate_predictions(
    pred: np.ndarray,
    true: np.ndarray,
    protein_names: Optional[list] = None,
    verbose: bool = True,
    max_eval_cells: int = 5000,
) -> dict:
    # Subsample large test sets for fast evaluation (PCC stabilizes ~5000 cells)
    if pred.shape[0] > max_eval_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(pred.shape[0], max_eval_cells, replace=False)
        pred = pred[idx]
        true = true[idx]
    pcc_protein, pcc_cell, pcc_protein_mean, pcc_cell_mean = compute_pcc(
        pred, true, protein_names
    )
    rmse = compute_rmse(pred, true)
    cmd_cell, cmd_protein = compute_cmd(pred, true)
    mmd = compute_mmd(pred, true)
    results = {
        'pcc_protein': pcc_protein,
        'pcc_cell': pcc_cell,
        'pcc_protein_mean': pcc_protein_mean,
        'pcc_protein_median': np.nanmedian(pcc_protein),
        'pcc_cell_mean': pcc_cell_mean,
        'pcc_cell_median': np.nanmedian(pcc_cell),
        'rmse': rmse,
        'cmd_cell': cmd_cell,
        'cmd_protein': cmd_protein,
        'mmd': mmd
    }

    if verbose:
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"PCC (Protein-protein): mean = {pcc_protein_mean:.4f}, median = {np.nanmedian(pcc_protein):.4f}")
        print(f"PCC (Cell-cell):       mean = {pcc_cell_mean:.4f}, median = {np.nanmedian(pcc_cell):.4f}")
        print(f"CMD (Cell-cell):       {cmd_cell:.4f}")
        print(f"CMD (Protein-protein): {cmd_protein:.4f}")
        print(f"RMSE:                  {rmse:.4f}")
        print(f"MMD:                   {mmd:.4f}")
        print("=" * 50 + "\n")

    return results
