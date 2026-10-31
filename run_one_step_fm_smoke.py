"""Dependency-light synthetic smoke test for one-step FM code paths."""
from __future__ import annotations

import numpy as np
import torch

from core_modules.flow_retrieval import FlowMemoryBank, l2_normalize_np
from models.one_step_fm import OneStepRetrievalFM, integrate_flow, one_step_flow_matching_loss


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    n, rna_dim, protein_dim = 256, 16, 8
    z = l2_normalize_np(np.random.randn(n, rna_dim).astype(np.float32))
    projection = np.random.randn(rna_dim, protein_dim).astype(np.float32) * 0.3
    y = (z @ projection + 0.03 * np.random.randn(n, protein_dim)).astype(np.float32)

    bank = FlowMemoryBank(use_faiss=False)
    bank.fit(z, y)
    retrieved = bank.retrieve(z, k=5, query_bank_indices=np.arange(n))

    model = OneStepRetrievalFM(
        protein_dim=protein_dim,
        rna_dim=rna_dim,
        hidden_dim=64,
        depth=2,
        condition_mode="query_delta",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x0 = torch.from_numpy(retrieved.protein_prior)
    x1 = torch.from_numpy(y)
    q = torch.from_numpy(z)
    r = torch.from_numpy(retrieved.reference_feature)
    s = torch.from_numpy(retrieved.mean_similarity)

    first = None
    for _ in range(80):
        out = one_step_flow_matching_loss(
            model, x0, x1, q, r, s, endpoint_weight=1.0
        )
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()
        if first is None:
            first = float(out.loss.detach())
    pred1 = integrate_flow(model, x0, q, r, s, nfe=1)
    pred4 = integrate_flow(model, x0, q, r, s, nfe=4)
    mse_prior = torch.mean((x0 - x1) ** 2).item()
    mse_one = torch.mean((pred1 - x1) ** 2).item()
    integration_gap = torch.mean((pred1 - pred4) ** 2).sqrt().item()
    assert np.isfinite(mse_one)
    assert mse_one < mse_prior, (mse_one, mse_prior)
    print({
        "initial_training_loss": first,
        "final_training_loss": float(out.loss.detach()),
        "retrieval_mse": mse_prior,
        "one_step_mse": mse_one,
        "one_vs_four_step_rmse": integration_gap,
        "status": "PASS",
    })


if __name__ == "__main__":
    main()
