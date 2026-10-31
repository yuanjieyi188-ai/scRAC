"""Stage 1: Train contrastive autoencoder. Exports train_stage1() for experiments."""
from __future__ import annotations

import argparse, sys, json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.clip_vae import ContrastiveAutoencoder, Stage1LossWeights
from utils import (
    create_dataloaders, preprocess_cross_dataset, preprocess_single_dataset,
    save_json, set_seed,
)


def train_stage1(bundle, config, output_dir, device="cuda"):
    """Train Stage1 contrastive VAE. Returns trained model.

    Args:
        bundle: DataBundle from preprocess_single_dataset
        config: Stage1Config dataclass
        output_dir: Path to save checkpoints and logs
        device: torch device string

    Returns:
        ContrastiveAutoencoder with best checkpoint loaded
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data metadata (gene/protein names for downstream reuse)
    save_json(output_dir / "data_info.json", {
        "gene_names": bundle.gene_names,
        "protein_names": bundle.protein_names,
        "n_train": len(bundle.train_idx),
        "n_test": len(bundle.test_idx),
    })

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader, test_loader = create_dataloaders(
        bundle, batch_size=config.batch_size, num_workers=0, pin_memory=pin_memory)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    set_seed(config.seed)
    model = ContrastiveAutoencoder(
        rna_dim=len(bundle.gene_names), protein_dim=len(bundle.protein_names),
        hidden_dim=config.hidden_dim, latent_dim=config.latent_dim,
        proj_dim=config.proj_dim, depth=config.depth, dropout=config.dropout,
    ).to(device)

    weights = Stage1LossWeights(config.alpha, config.beta, config.gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            losses = model.compute_loss(batch, weights)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses["loss"].item()
            n += 1
        train_loss /= max(n, 1)

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                val_loss += model.compute_loss(batch, weights)["loss"].item()
                nv += 1
        val_loss /= max(nv, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best and save
    model.load_state_dict(best_state)
    checkpoint = {
        "epoch": len(history),
        "model_state_dict": best_state,
        "config": {
            "rna_dim": len(bundle.gene_names), "protein_dim": len(bundle.protein_names),
            "hidden_dim": config.hidden_dim, "latent_dim": config.latent_dim,
            "proj_dim": config.proj_dim, "depth": config.depth, "dropout": config.dropout,
        },
        "gene_names": bundle.gene_names,
        "protein_names": bundle.protein_names,
        "train_idx": bundle.train_idx,
        "test_idx": bundle.test_idx,
    }
    torch.save(checkpoint, output_dir / "stage1_best.pt")
    torch.save(checkpoint, output_dir / "stage1_last.pt")
    save_json(output_dir / "history.json", {"best_val_loss": best_val, "history": history})
    save_json(output_dir / "best_metrics.json", {"best_val_loss": best_val})

    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--per_protein_hvg", action="store_true")
    p.add_argument("--n_per_protein", type=int, default=100)
    p.add_argument("--protein_guided_hvg", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    from pipeline.config import Stage1Config
    config = Stage1Config(
        n_top_genes=args.n_top_genes, epochs=args.epochs, lr=args.lr,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, depth=args.depth,
        dropout=args.dropout, alpha=args.alpha, weight_decay=args.weight_decay,
        batch_size=args.batch_size, use_per_protein_hvg=args.per_protein_hvg,
        use_protein_guided_hvg=args.protein_guided_hvg,
        n_per_protein=args.n_per_protein, seed=args.seed,
    )
    bundle = preprocess_single_dataset(
        args.data_path, n_top_genes=config.n_top_genes,
        use_per_protein_hvg=config.use_per_protein_hvg,
        use_protein_guided_hvg=config.use_protein_guided_hvg,
        n_per_protein=config.n_per_protein, seed=config.seed)
    train_stage1(bundle, config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
