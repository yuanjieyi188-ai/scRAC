"""Stage 3: Train FiLM residual model. Exports train_stage3() for experiments."""
from __future__ import annotations

import argparse, sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_modules.retrieval import MemoryBank
from metrics import evaluate_predictions
from models.clip_vae import ContrastiveAutoencoder
from models.rag_flow import FiLMResidualMLP, FiLMGateResidualMLP
from utils import preprocess_single_dataset, save_json, set_seed


def build_s3_model(config, protein_dim, cond_dim=128):
    """Build Stage3 model based on config.model choice."""
    if config.model == "film_gate":
        return FiLMGateResidualMLP(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=config.hidden_dim, depth=config.depth,
            dropout=config.dropout, use_adaln_zero=config.use_adaln_zero,
        )
    else:
        return FiLMResidualMLP(
            protein_dim=protein_dim, cond_dim=cond_dim,
            hidden_dim=config.hidden_dim, depth=config.depth,
            dropout=config.dropout, use_adaln_zero=config.use_adaln_zero,
        )


def train_stage3(stage1_model, starts, bundle, config, output_dir, device="cuda"):
    """Train Stage3 FiLM residual model.

    Args:
        stage1_model: trained ContrastiveAutoencoder (will be finetuned)
        starts: dict with train_start, val_start, test_start, train_feat, val_feat, test_feat
        bundle: DataBundle (must have rna_norm_train/val/test and protein_norm_train/val/test)
        config: Stage3Config dataclass
        output_dir: Path to save checkpoints
        device: torch device

    Returns:
        tuple of (model_best_state, history_list)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    from pipeline.utils_train import encode_rna

    rna_train = bundle.rna_norm_train
    rna_val = getattr(bundle, 'rna_norm_val', bundle.rna_norm_test)
    pro_train = bundle.protein_norm_train
    pro_val = getattr(bundle, 'protein_norm_val', bundle.protein_norm_test)
    protein_names = bundle.protein_names

    train_start = starts["train_start"]
    val_start = starts["val_start"]
    val_feat = starts.get("val_feat")
    test_start = starts.get("test_start")
    test_feat = starts.get("test_feat")

    # Build model
    model = build_s3_model(config, len(protein_names)).to(device)

    # Finetune Stage1
    if config.finetune_stage1:
        stage1_model.train()
        for p in stage1_model.parameters():
            p.requires_grad = True

    params = list(model.parameters())
    if config.finetune_stage1:
        params += list(stage1_model.parameters())

    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    ds_train = TensorDataset(
        torch.from_numpy(rna_train.astype(np.float32)),
        torch.from_numpy(train_start),
        torch.from_numpy(pro_train.astype(np.float32)),
    )
    dl = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True,
                    pin_memory=device.type == "cuda")

    # Determine output semantics
    is_gate_model = (config.model == "film_gate")

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        if config.finetune_stage1:
            stage1_model.train()

        total_loss = 0.0
        nb = 0
        for rna_b, st_b, tg_b in dl:
            rna_b = rna_b.to(device)
            st_b = st_b.to(device)
            tg_b = tg_b.to(device)

            if config.finetune_stage1:
                cond = stage1_model.encode_rna(rna_b)
            else:
                with torch.no_grad():
                    cond = stage1_model.encode_rna(rna_b)

            # Apply mask (only during training, only if mask_prob > 0)
            if config.mask_prob > 0:
                mask = torch.rand_like(st_b) < config.mask_prob
                st_b = torch.where(mask, torch.zeros_like(st_b), st_b)

            out = model(st_b, cond)
            if is_gate_model:
                # Gate model returns velocity v = gate*delta
                loss = torch.mean((out - (tg_b - st_b)) ** 2)
            else:
                # FiLMResidualMLP returns delta
                loss = torch.mean(((st_b + out) - tg_b) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            nb += 1

        train_loss = total_loss / max(nb, 1)

        # Validation
        if epoch % config.eval_every == 0 or epoch <= 2 or epoch == config.epochs:
            model.eval()
            if config.finetune_stage1:
                stage1_model.eval()

            with torch.no_grad():
                # Recompute val features if Stage1 was finetuned
                if config.finetune_stage1 and rna_val is not None:
                    val_feat_cur = encode_rna(stage1_model, rna_val, device)
                else:
                    val_feat_cur = val_feat

                sv = torch.from_numpy(val_start).to(device)
                cv = torch.from_numpy(val_feat_cur).to(device)
                out_v = model(sv, cv)
                if is_gate_model:
                    val_preds = (sv + out_v).cpu().numpy().astype(np.float32)
                else:
                    val_preds = (sv + out_v).cpu().numpy().astype(np.float32)

                val_loss = np.mean((val_preds - pro_val) ** 2)

                # Also compute val PCC for reference
                res = evaluate_predictions(val_preds, pro_val, protein_names, verbose=False)
                val_pcc = res["pcc_protein_mean"]

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "val_pcc": float(val_pcc),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                # Save best model and val predictions
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "stage1_state_dict": stage1_model.state_dict() if config.finetune_stage1 else None,
                }, output_dir / "stage3_best.pt")
                np.save(output_dir / "val_predictions.npy", val_preds)
                np.save(output_dir / "val_ground_truth.npy", pro_val)

    # Save history
    save_json(output_dir / "history.json", {"history": history, "best_val_loss": float(best_val_loss)})

    return best_state, history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--stage1_path", type=str, required=True)
    p.add_argument("--bank_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="film_gate")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mask_prob", type=float, default=0.0)
    p.add_argument("--use_adaln_zero", type=bool, default=True)
    p.add_argument("--finetune_stage1", type=bool, default=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--per_protein_hvg", action="store_true")
    p.add_argument("--protein_guided_hvg", action="store_true")
    p.add_argument("--n_per_protein", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from pipeline.config import Stage3Config
    from pipeline.utils_train import encode_rna

    config = Stage3Config(
        model=args.model, hidden_dim=args.hidden_dim, depth=args.depth,
        dropout=args.dropout, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, mask_prob=args.mask_prob,
        use_adaln_zero=args.use_adaln_zero, finetune_stage1=args.finetune_stage1,
        k_retrieval=args.k, seed=args.seed,
    )

    # Load Stage1
    s1_ckpt = torch.load(args.stage1_path, map_location=device, weights_only=False)
    s1 = ContrastiveAutoencoder(**s1_ckpt["config"]).to(device)
    s1.load_state_dict(s1_ckpt["model_state_dict"])
    gene_names = s1_ckpt.get("gene_names", None)

    # Load data with same genes
    bundle = preprocess_single_dataset(
        args.data_path, n_top_genes=args.n_top_genes,
        use_per_protein_hvg=args.per_protein_hvg,
        use_protein_guided_hvg=args.protein_guided_hvg,
        n_per_protein=args.n_per_protein, seed=args.seed,
        gene_names=gene_names,
    )

    # Load bank
    bank_data = np.load(args.bank_path, allow_pickle=True)
    bank = MemoryBank()
    bank.fit(bank_data["features"], bank_data["payload"])

    # Compute starts
    s1.eval()
    train_feat = encode_rna(s1, bundle.rna_norm_train, device)
    test_feat = encode_rna(s1, bundle.rna_norm_test, device)
    train_start = bank.aggregate(train_feat, k=args.k, exclude_self=True)
    test_start = bank.aggregate(test_feat, k=args.k, exclude_self=False)

    starts = {
        "train_start": train_start,
        "test_start": test_start,
        "train_feat": train_feat,
        "test_feat": test_feat,
    }

    train_stage3(s1, starts, bundle, config, args.output_dir, str(device))


if __name__ == "__main__":
    main()
