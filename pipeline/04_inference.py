"""Inference and evaluation for scRAG-Flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core_modules.retrieval import MemoryBank
from metrics import evaluate_predictions
from models.clip_vae import ContrastiveAutoencoder
from models.rag_flow import ResidualCorrectionModel, apply_residual_correction
from utils import preprocess_cross_dataset, preprocess_single_dataset, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, default="")
    parser.add_argument("--stage1_path", type=str, required=True)
    parser.add_argument("--flow_path", type=str, required=True)
    parser.add_argument("--bank_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_top_genes", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_bank(path: str) -> MemoryBank:
    data = np.load(path, allow_pickle=True)
    bank = MemoryBank()
    bank.fit(data["features"], data["payload"])
    return bank


@torch.no_grad()
def predict(stage1, flow, bank, rna_norm, batch_size, device, k, ode_steps):
    preds = []
    for start in range(0, len(rna_norm), batch_size):
        end = min(start + batch_size, len(rna_norm))
        rna_batch = torch.from_numpy(rna_norm[start:end]).to(device)
        cond = stage1.encode_rna(rna_batch)
        retrieved = bank.aggregate(cond.cpu().numpy(), k=k, exclude_self=False)
        pred = apply_residual_correction(flow, torch.from_numpy(retrieved).to(device), cond)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    stage1_ckpt = torch.load(args.stage1_path, map_location=device, weights_only=False)
    stage1 = ContrastiveAutoencoder(**stage1_ckpt["config"]).to(device)
    stage1.load_state_dict(stage1_ckpt["model_state_dict"])
    stage1.eval()

    flow_ckpt = torch.load(args.flow_path, map_location=device, weights_only=False)
    flow = ResidualCorrectionModel(**flow_ckpt["config"]).to(device)
    flow.load_state_dict(flow_ckpt["model_state_dict"])
    flow.eval()

    if args.test_data_path:
        bundle = preprocess_cross_dataset(args.data_path, args.test_data_path, n_top_genes=args.n_top_genes)
    else:
        bundle = preprocess_single_dataset(
            args.data_path, n_top_genes=args.n_top_genes, train_ratio=args.train_ratio, seed=args.seed
        )

    bank = load_bank(args.bank_path)
    predictions = predict(
        stage1, flow, bank, bundle.rna_norm_test, args.batch_size, device, args.k, args.ode_steps
    )
    results = evaluate_predictions(predictions, bundle.protein_norm_test, bundle.protein_names, verbose=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "ground_truth.npy", bundle.protein_norm_test)
    save_json(
        output_dir / "metrics.json",
        {k: float(v) if np.isscalar(v) else None for k, v in results.items() if "pcc_" not in k or k.endswith(("mean", "median"))},
    )


if __name__ == "__main__":
    main()
