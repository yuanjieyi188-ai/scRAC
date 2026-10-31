"""Stage 4: Final evaluation. Exports evaluate_test() for experiments."""
from __future__ import annotations

import argparse, sys, json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_modules.retrieval import MemoryBank
from metrics import evaluate_predictions
from models.clip_vae import ContrastiveAutoencoder
from models.rag_flow import FiLMResidualMLP, FiLMGateResidualMLP
from utils import preprocess_single_dataset, save_json


def evaluate_test(stage1_model, s3_best_state, s3_config, starts, bundle,
                  output_dir, device="cuda"):
    """Final evaluation on held-out test set.

    Args:
        stage1_model: finetuned Stage1 model
        s3_best_state: best Stage3 model state_dict
        s3_config: Stage3Config for building the model
        starts: dict with test_start, test_feat (or train_feat for bank rebuild)
        bundle: DataBundle with test arrays
        output_dir: Path to save predictions and metrics
        device: torch device

    Returns:
        dict of metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    from pipeline.utils_train import encode_rna

    # Build model and load best state
    from pipeline.config import Stage3Config
    model = FiLMGateResidualMLP(
        protein_dim=len(bundle.protein_names), cond_dim=128,
        hidden_dim=s3_config.hidden_dim, depth=s3_config.depth,
        dropout=s3_config.dropout, use_adaln_zero=s3_config.use_adaln_zero,
    ).to(device)
    model.load_state_dict(s3_best_state)
    model.eval()
    stage1_model.eval()

    rna_train = bundle.rna_norm_train
    rna_test = bundle.rna_norm_test
    pro_test = bundle.protein_norm_test
    protein_names = bundle.protein_names

    # Rebuild bank with finetuned Stage1 for final test evaluation
    train_feat_final = encode_rna(stage1_model, rna_train, device)
    test_feat_final = encode_rna(stage1_model, rna_test, device)

    bank_final = MemoryBank()
    bank_final.fit(train_feat_final, bundle.protein_norm_train)
    test_start_final = bank_final.aggregate(test_feat_final, k=5, exclude_self=False)

    # Stage3 prediction
    with torch.no_grad():
        tt = torch.from_numpy(test_start_final).to(device)
        tc = torch.from_numpy(test_feat_final).to(device)
        out = model(tt, tc)
        test_preds = (tt + out).cpu().numpy().astype(np.float32)

    # Retrieval-only baseline (same bank, no Stage3)
    retrieval_preds = test_start_final

    # Full metrics (MMD sampled at 2000 cells)
    results = evaluate_predictions(test_preds, pro_test, protein_names, verbose=True)

    # Retrieval metrics
    ret_results = evaluate_predictions(retrieval_preds, pro_test, protein_names, verbose=False)

    # Save predictions and ground truth
    np.save(output_dir / "predictions.npy", test_preds)
    np.save(output_dir / "ground_truth.npy", pro_test)
    np.save(output_dir / "retrieval_predictions.npy", retrieval_preds)

    # Save per-protein and per-cell PCC for downstream plotting
    np.save(output_dir / "pcc_protein.npy", results["pcc_protein"])
    np.save(output_dir / "pcc_cell.npy", results["pcc_cell"])
    np.save(output_dir / "protein_names.npy", np.array(protein_names))

    metrics = {
        "pcc_protein_mean": float(results["pcc_protein_mean"]),
        "pcc_protein_median": float(results.get("pcc_protein_median", 0)),
        "pcc_cell_mean": float(results["pcc_cell_mean"]),
        "pcc_cell_median": float(results.get("pcc_cell_median", 0)),
        "rmse": float(results.get("rmse", 0)),
        "cmd_cell": float(results.get("cmd_cell", 0)),
        "cmd_protein": float(results.get("cmd_protein", 0)),
        "mmd": float(results.get("mmd", 0)),
        "retrieval_pcc": float(ret_results["pcc_protein_mean"]),
        "stage3_delta": float(results["pcc_protein_mean"] - ret_results["pcc_protein_mean"]),
        "protein_names": protein_names,
    }
    save_json(output_dir / "metrics.json", metrics)

    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--stage1_path", type=str, required=True)
    p.add_argument("--stage3_path", type=str, required=True)
    p.add_argument("--bank_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="film_gate")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_adaln_zero", type=bool, default=True)
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

    # Load Stage1
    s1_ckpt = torch.load(args.stage1_path, map_location=device, weights_only=False)
    s1 = ContrastiveAutoencoder(**s1_ckpt["config"]).to(device)
    s1.load_state_dict(s1_ckpt["model_state_dict"])
    s1.eval()
    gene_names = s1_ckpt.get("gene_names", None)

    # Load Stage3
    s3_ckpt = torch.load(args.stage3_path, map_location=device, weights_only=False)
    s3_state = s3_ckpt["model_state_dict"]
    s3_config = Stage3Config(
        model=args.model, hidden_dim=args.hidden_dim, depth=args.depth,
        dropout=args.dropout, use_adaln_zero=args.use_adaln_zero,
    )

    # Load data with same genes
    bundle = preprocess_single_dataset(
        args.data_path, n_top_genes=args.n_top_genes,
        use_per_protein_hvg=args.per_protein_hvg,
        use_protein_guided_hvg=args.protein_guided_hvg,
        n_per_protein=args.n_per_protein, seed=args.seed,
        gene_names=gene_names,
    )

    # Dummy starts (evaluate_test rebuilds bank internally)
    starts = {}

    evaluate_test(s1, s3_state, s3_config, starts, bundle, args.output_dir, str(device))


if __name__ == "__main__":
    main()
