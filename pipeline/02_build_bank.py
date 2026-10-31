"""Stage 2: Build FAISS retrieval bank. Exports build_bank() for experiments."""
from __future__ import annotations

import argparse, sys, json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_modules.retrieval import MemoryBank
from models.clip_vae import ContrastiveAutoencoder
from utils import preprocess_single_dataset


def build_bank(stage1_model, bundle, output_dir, k=5, device="cuda"):
    """Build retrieval bank and compute retrieval starts for train/val/test.

    Args:
        stage1_model: trained ContrastiveAutoencoder (eval mode)
        bundle: DataBundle with train/val/test arrays
        output_dir: Path to save memory_bank.npz
        k: number of retrieval neighbors
        device: torch device

    Returns:
        dict with keys: train_start, val_start, test_start, bank
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    from pipeline.utils_train import encode_rna

    # Encode all splits
    rna_train = bundle.rna_norm_train
    rna_val = getattr(bundle, 'rna_norm_val', None)
    rna_test = bundle.rna_norm_test

    train_feat = encode_rna(stage1_model, rna_train, device)

    # Build bank on train (or train+val)
    if rna_val is not None:
        val_feat = encode_rna(stage1_model, rna_val, device)
        bank_features = np.concatenate([train_feat, val_feat], axis=0)
        bank_payload = np.concatenate([bundle.protein_norm_train, bundle.protein_norm_val], axis=0)
    else:
        val_feat = None
        bank_features = train_feat
        bank_payload = bundle.protein_norm_train

    test_feat = encode_rna(stage1_model, rna_test, device)

    bank = MemoryBank()
    bank.fit(bank_features, bank_payload)

    # Retrieval starts
    train_start = bank.aggregate(train_feat, k=k, exclude_self=True)
    val_start = bank.aggregate(val_feat, k=k, exclude_self=True) if val_feat is not None else None
    test_start = bank.aggregate(test_feat, k=k, exclude_self=False)

    # Save bank
    np.savez_compressed(
        output_dir / "memory_bank.npz",
        features=bank.features,
        payload=bank_payload,
    )

    return {
        "train_start": train_start,
        "val_start": val_start,
        "test_start": test_start,
        "train_feat": train_feat,
        "val_feat": val_feat,
        "test_feat": test_feat,
        "bank": bank,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--stage1_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--per_protein_hvg", action="store_true")
    p.add_argument("--protein_guided_hvg", action="store_true")
    p.add_argument("--n_per_protein", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.stage1_path, map_location=device, weights_only=False)
    model = ContrastiveAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Reuse gene_names from Stage1 to avoid recomputing HVG
    gene_names = ckpt.get("gene_names", None)

    bundle = preprocess_single_dataset(
        args.data_path, n_top_genes=args.n_top_genes,
        use_per_protein_hvg=args.per_protein_hvg,
        use_protein_guided_hvg=args.protein_guided_hvg,
        n_per_protein=args.n_per_protein, seed=args.seed,
        gene_names=gene_names,
    )

    build_bank(model, bundle, args.output_dir, k=args.k, device=str(device))


if __name__ == "__main__":
    main()
