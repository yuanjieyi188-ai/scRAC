from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import PaperConfig
from .losses import flow_matching_objective
from .metrics import protein_pearson_mean
from .models import ContrastiveZINBAutoencoder, RetrievalConditionedFlow
from .retrieval import ExactProteinMemory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(*arrays: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(array, dtype=np.float32)) for array in arrays]
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def encode_rna(
    model: ContrastiveZINBAutoencoder,
    rna: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(rna), batch_size):
        batch = torch.from_numpy(rna[start : start + batch_size]).to(device)
        output.append(model.encode_rna(batch).cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def train_stage1(
    model: ContrastiveZINBAutoencoder,
    rna_normalized: np.ndarray,
    rna_counts: np.ndarray,
    protein_normalized: np.ndarray,
    protein_counts: np.ndarray,
    config: PaperConfig,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> list[dict[str, float]]:
    stage = config.stage1
    loader = _loader(
        rna_normalized,
        rna_counts,
        protein_normalized,
        protein_counts,
        batch_size=stage.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=stage.learning_rate, weight_decay=stage.weight_decay
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, stage.epochs + 1):
        model.train()
        sums = {"total": 0.0, "contrastive": 0.0, "rna_zinb": 0.0, "protein_zinb": 0.0}
        batches = 0
        for rna_norm, rna_raw, protein_norm, protein_raw in loader:
            losses = model.objective(
                rna_norm.to(device),
                rna_raw.to(device),
                protein_norm.to(device),
                protein_raw.to(device),
                stage.contrastive_weight,
                stage.rna_zinb_weight,
                stage.protein_zinb_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), stage.gradient_clip)
            optimizer.step()
            for key in sums:
                sums[key] += float(losses[key].detach())
            batches += 1
        row = {"epoch": float(epoch), **{key: value / batches for key, value in sums.items()}}
        history.append(row)
        log(
            f"stage1 epoch={epoch:02d}/{stage.epochs} "
            f"loss={row['total']:.5f} contrastive={row['contrastive']:.5f}"
        )
    return history


@torch.no_grad()
def predict_one_step(
    flow: RetrievalConditionedFlow,
    encoder: ContrastiveZINBAutoencoder,
    rna_normalized: np.ndarray,
    source: np.ndarray,
    similarity: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    flow.eval()
    encoder.eval()
    loader = _loader(
        rna_normalized, source, similarity, batch_size=batch_size, shuffle=False
    )
    predictions: list[np.ndarray] = []
    for rna, source_batch, similarity_batch in loader:
        rna = rna.to(device)
        source_batch = source_batch.to(device)
        similarity_batch = similarity_batch.to(device)
        latent = encoder.encode_rna(rna)
        prediction = flow.predict_one_step(source_batch, latent, similarity_batch)
        predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def train_stage3(
    stage1_model: ContrastiveZINBAutoencoder,
    flow: RetrievalConditionedFlow,
    fit_rna: np.ndarray,
    fit_protein: np.ndarray,
    validation_rna: np.ndarray,
    validation_protein: np.ndarray,
    initial_fit_source: np.ndarray,
    initial_fit_similarity: np.ndarray,
    config: PaperConfig,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    stage = config.stage3
    for parameter in stage1_model.parameters():
        parameter.requires_grad = False
    for parameter in stage1_model.rna_encoder.parameters():
        parameter.requires_grad = True

    trainable_parameters = list(flow.parameters()) + list(
        stage1_model.rna_encoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=stage.learning_rate,
        weight_decay=stage.weight_decay,
    )
    loader = _loader(
        fit_rna,
        initial_fit_source,
        initial_fit_similarity,
        fit_protein,
        batch_size=stage.batch_size,
        shuffle=True,
    )

    best_pcc = -np.inf
    best_epoch = 0
    best_flow_state = deepcopy(flow.state_dict())
    best_encoder_state = deepcopy(stage1_model.rna_encoder.state_dict())
    history: list[dict[str, float]] = []

    for epoch in range(1, stage.epochs + 1):
        flow.train()
        stage1_model.rna_encoder.train()
        totals = {"total": 0.0, "fm": 0.0, "endpoint": 0.0}
        batches = 0
        for rna, source, similarity, target in loader:
            rna = rna.to(device)
            source = source.to(device)
            similarity = similarity.to(device)
            target = target.to(device)
            query_latent = stage1_model.encode_rna(rna)
            losses = flow_matching_objective(
                flow,
                source,
                target,
                query_latent,
                similarity,
                stage.mask_probability,
                stage.endpoint_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, stage.gradient_clip)
            optimizer.step()
            totals["total"] += float(losses.total.detach())
            totals["fm"] += float(losses.flow_matching.detach())
            totals["endpoint"] += float(losses.endpoint.detach())
            batches += 1

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_total": totals["total"] / batches,
            "train_fm": totals["fm"] / batches,
            "train_endpoint": totals["endpoint"] / batches,
        }

        if epoch % stage.validation_interval == 0:
            fit_latent = encode_rna(stage1_model, fit_rna, device)
            validation_latent = encode_rna(stage1_model, validation_rna, device)
            validation_memory = ExactProteinMemory(config.retrieval.temperature)
            validation_memory.fit(fit_latent, fit_protein)
            validation_retrieval = validation_memory.retrieve(
                validation_latent, config.retrieval.neighbors
            )
            validation_prediction = predict_one_step(
                flow,
                stage1_model,
                validation_rna,
                validation_retrieval.protein_source,
                validation_retrieval.mean_similarity,
                device,
                stage.batch_size,
            )
            validation_pcc = protein_pearson_mean(
                validation_prediction, validation_protein
            )
            row["validation_pcc_protein"] = validation_pcc
            if validation_pcc > best_pcc:
                best_pcc = validation_pcc
                best_epoch = epoch
                best_flow_state = deepcopy(flow.state_dict())
                best_encoder_state = deepcopy(stage1_model.rna_encoder.state_dict())
            log(
                f"stage3 epoch={epoch:02d}/{stage.epochs} "
                f"loss={row['train_total']:.5f} val_PCC-P={validation_pcc:.4f}"
            )
        history.append(row)

    flow.load_state_dict(best_flow_state)
    stage1_model.rna_encoder.load_state_dict(best_encoder_state)
    return history, {
        "best_epoch": best_epoch,
        "best_validation_pcc_protein": float(best_pcc),
        "flow_state": best_flow_state,
        "rna_encoder_state": best_encoder_state,
    }


def save_json(path: str | Path, payload: object) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2, ensure_ascii=False)


def config_to_dict(config: PaperConfig) -> dict[str, object]:
    return asdict(config)
