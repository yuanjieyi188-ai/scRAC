"""Shared utilities for pipeline training scripts."""
import numpy as np
import torch
from scipy.sparse import issparse


def select_protein_guided_genes(train_adata, protein, n_top_genes=1000):
    """Protein-guided HVG: top genes by max |corr| with any protein."""
    X = train_adata.X
    if issparse(X):
        X = X.toarray()
    rna = X.astype(np.float64)
    prot = protein.astype(np.float64)
    rna_c = rna - rna.mean(axis=0, keepdims=True)
    prot_c = prot - prot.mean(axis=0, keepdims=True)
    cov = rna_c.T @ prot_c
    std_rna = np.sqrt((rna_c ** 2).sum(axis=0)) + 1e-8
    std_prot = np.sqrt((prot_c ** 2).sum(axis=0)) + 1e-8
    corr = cov / np.outer(std_rna, std_prot)
    max_corr = np.max(np.abs(corr), axis=1)
    top_idx = np.argsort(max_corr)[-n_top_genes:]
    return list(np.array(train_adata.var_names)[top_idx])


@torch.no_grad()
def encode_rna(model, array, device, bs=4096):
    """Batch-encode RNA through Stage1 encoder."""
    outs = []
    for i in range(0, len(array), bs):
        e = min(i + bs, len(array))
        b = torch.from_numpy(array[i:e]).to(device)
        outs.append(model.encode_rna(b).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)
