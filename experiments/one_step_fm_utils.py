"""Shared training/evaluation helpers for one-step FM experiments."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics import evaluate_predictions
from models.one_step_fm import (
    OneStepRetrievalFM,
    flow_diagnostics,
    integrate_flow,
    one_step_flow_matching_loss,
)


@dataclass
class FlowTrainResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_val_pcc: float
    history: list[dict]


def _to_loader(*arrays: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in arrays]
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_in_batches(
    model: OneStepRetrievalFM,
    x0: np.ndarray,
    query_rna: np.ndarray,
    reference_rna: np.ndarray,
    similarity: np.ndarray,
    batch_size: int,
    device: torch.device,
    nfe: int = 1,
    solver: str = "euler",
) -> np.ndarray:
    model.eval()
    loader = _to_loader(
        x0, query_rna, reference_rna, similarity,
        batch_size=batch_size, shuffle=False,
    )
    outputs = []
    for x0_b, q_b, r_b, s_b in loader:
        pred = integrate_flow(
            model,
            x0_b.to(device),
            q_b.to(device),
            r_b.to(device),
            s_b.to(device),
            nfe=nfe,
            solver=solver,
        )
        outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def train_one_step_fm(
    model: OneStepRetrievalFM,
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    protein_names: list[str],
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    endpoint_weight: float = 1.0,
    source_noise_std: float = 0.0,
    eval_every: int = 2,
    log_fn=print,
) -> FlowTrainResult:
    x0_tr, q_tr, r_tr, sim_tr, y_tr = train_arrays
    x0_va, q_va, r_va, sim_va, y_va = val_arrays
    loader = _to_loader(
        x0_tr, q_tr, r_tr, sim_tr, y_tr,
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_pcc = -float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "fm": 0.0, "endpoint": 0.0}
        n_batches = 0
        start_time = time.time()
        for x0_b, q_b, r_b, sim_b, y_b in loader:
            x0_b = x0_b.to(device)
            q_b = q_b.to(device)
            r_b = r_b.to(device)
            sim_b = sim_b.to(device)
            y_b = y_b.to(device)
            loss_out = one_step_flow_matching_loss(
                model,
                x0_b,
                y_b,
                q_b,
                r_b,
                sim_b,
                endpoint_weight=endpoint_weight,
                source_noise_std=source_noise_std,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss_out.loss.detach())
            totals["fm"] += float(loss_out.fm_loss.detach())
            totals["endpoint"] += float(loss_out.endpoint_loss.detach())
            n_batches += 1

        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(n_batches, 1),
            "train_fm": totals["fm"] / max(n_batches, 1),
            "train_endpoint": totals["endpoint"] / max(n_batches, 1),
            "seconds": time.time() - start_time,
        }

        if epoch % eval_every == 0 or epoch <= 2 or epoch == epochs:
            val_pred = predict_in_batches(
                model, x0_va, q_va, r_va, sim_va,
                batch_size=batch_size, device=device, nfe=1,
            )
            val_metrics = evaluate_predictions(
                val_pred, y_va, protein_names, verbose=False
            )
            val_pcc = float(val_metrics["pcc_protein_mean"])
            row["val_pcc_nfe1"] = val_pcc
            row["val_mse_nfe1"] = float(np.mean((val_pred - y_va) ** 2))
            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
            log_fn(
                f"epoch={epoch:03d} loss={row['train_loss']:.5f} "
                f"fm={row['train_fm']:.5f} endpoint={row['train_endpoint']:.5f} "
                f"val_PCC-P@1={val_pcc:.4f}"
            )
        else:
            log_fn(f"epoch={epoch:03d} loss={row['train_loss']:.5f}")
        history.append(row)

    return FlowTrainResult(
        best_state=best_state,
        best_epoch=best_epoch,
        best_val_pcc=best_val_pcc,
        history=history,
    )


@torch.no_grad()
def export_diagnostics(
    model: OneStepRetrievalFM,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_dir: str | Path,
    device: torch.device,
    max_cells: int = 4096,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x0, q, r, sim, y = arrays
    n = min(len(x0), max_cells)
    x0_t = torch.from_numpy(x0[:n]).to(device)
    q_t = torch.from_numpy(q[:n]).to(device)
    r_t = torch.from_numpy(r[:n]).to(device)
    sim_t = torch.from_numpy(sim[:n]).to(device)
    y_t = torch.from_numpy(y[:n]).to(device)
    diag = flow_diagnostics(model, x0_t, y_t, q_t, r_t, sim_t)
    alignment = diag["alignment"].cpu().numpy()
    path_variation = diag["path_variation"].cpu().numpy()
    times = diag["times"].cpu().numpy()
    np.savez_compressed(
        output_dir / "flow_diagnostics.npz",
        times=times,
        alignment=alignment,
        path_variation=path_variation,
    )
    return {
        "mean_alignment_t0": float(alignment[:, 0].mean()),
        "mean_alignment_all_t": float(alignment.mean()),
        "median_path_variation": float(np.median(path_variation)),
    }
