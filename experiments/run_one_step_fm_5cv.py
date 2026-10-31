"""Strict 5-fold evaluation of retrieval-initialized one-step flow matching.

Example:
  python -u experiments/run_one_step_fm_5cv.py \
      --dataset cbmc --data_path data/example.h5ad --device cuda

This script intentionally freezes Stage 1 during Stage 3. Rebuilding a memory
bank after every encoder update would otherwise make the retrieved source state
and the RNA displacement condition stale.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_modules.flow_retrieval import FlowMemoryBank, l2_normalize_np
from experiments.one_step_fm_utils import (
    export_diagnostics,
    predict_in_batches,
    train_one_step_fm,
)
from metrics import evaluate_predictions
from models.clip_vae import ContrastiveAutoencoder, Stage1LossWeights
from models.one_step_fm import OneStepRetrievalFM
from pipeline.config import get_cbmc_config, get_eccite_config, get_gse3p_config
from utils import preprocess_single_dataset, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cbmc", "eccite", "gse3p"], required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only_fold", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--stage1_epochs", type=int, default=-1)
    p.add_argument("--flow_epochs", type=int, default=40)
    p.add_argument("--flow_hidden_dim", type=int, default=512)
    p.add_argument("--flow_depth", type=int, default=3)
    p.add_argument("--flow_dropout", type=float, default=0.1)
    p.add_argument("--flow_lr", type=float, default=1e-3)
    p.add_argument("--flow_weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--endpoint_weight", type=float, default=1.0)
    p.add_argument("--source_noise_std", type=float, default=0.0)
    p.add_argument(
        "--condition_mode",
        choices=["query", "delta", "query_delta", "query_ref_delta"],
        default="query_delta",
    )
    p.add_argument("--retrieval_temperature", type=float, default=1.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--eval_every", type=int, default=2)
    return p.parse_args()


def dataset_config(name: str):
    if name == "cbmc":
        return get_cbmc_config()
    if name == "eccite":
        return get_eccite_config()
    return get_gse3p_config()


def train_stage1(model, bundle, config, device, log_fn):
    model.train()
    weights = Stage1LossWeights(config.alpha, config.beta, config.gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    dataset = TensorDataset(
        torch.from_numpy(bundle.rna_norm_train),
        torch.from_numpy(bundle.rna_raw_train),
        torch.from_numpy(bundle.protein_norm_train),
        torch.from_numpy(bundle.protein_raw_train),
    )
    loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=0)
    for epoch in range(1, config.epochs + 1):
        running = 0.0
        batches = 0
        for rna_norm, rna_raw, pro_norm, pro_raw in loader:
            batch = {
                "rna_norm": rna_norm.to(device),
                "rna_raw": rna_raw.to(device),
                "protein_norm": pro_norm.to(device),
                "protein_raw": pro_raw.to(device),
            }
            out = model.compute_loss(batch, weights)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(out["loss"].detach())
            batches += 1
        if epoch == 1 or epoch == config.epochs or epoch % 10 == 0:
            log_fn(f"Stage1 epoch={epoch:03d} loss={running/max(batches,1):.5f}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


@torch.no_grad()
def encode_rna(model, array: np.ndarray, device: torch.device, batch_size: int = 4096):
    outputs = []
    for start in range(0, len(array), batch_size):
        batch = torch.from_numpy(array[start:start + batch_size]).to(device)
        outputs.append(model.encode_rna(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    s1c, s3c = dataset_config(args.dataset)
    s1c.seed = args.seed
    if args.stage1_epochs > 0:
        s1c.epochs = args.stage1_epochs
    output_root = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / f"{args.dataset}_one_step_fm_5cv"
    output_root.mkdir(parents=True, exist_ok=True)

    import scanpy as sc
    n_cells = sc.read_h5ad(args.data_path, backed="r").n_obs
    rng = np.random.default_rng(args.seed)
    folds = np.array_split(rng.permutation(n_cells), args.folds)
    fold_ids = [args.only_fold] if args.only_fold >= 0 else list(range(args.folds))
    all_results = []

    for fold_id in fold_ids:
        fold_start = time.time()
        fold_dir = output_root / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(fold_dir / "run.log", "w", encoding="utf-8")

        def log(message: str) -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        test_idx = folds[fold_id]
        train_pool = np.concatenate([folds[i] for i in range(args.folds) if i != fold_id])
        inner_rng = np.random.default_rng(args.seed + fold_id)
        perm = inner_rng.permutation(train_pool)
        n_val = max(1, int(len(train_pool) * s3c.val_ratio / 0.8))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        eval_idx = np.concatenate([val_idx, test_idx])
        log(f"fold={fold_id} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        # Strict preprocessing: gene selection and all supervised Stage-1 losses use train only.
        bundle = preprocess_single_dataset(
            args.data_path,
            n_top_genes=s1c.n_top_genes,
            seed=args.seed,
            use_protein_guided_hvg=s1c.use_protein_guided_hvg,
            use_per_protein_hvg=s1c.use_per_protein_hvg,
            n_per_protein=s1c.n_per_protein,
            fold_train_idx=train_idx,
            fold_test_idx=eval_idx,
        )
        n_val_local = len(val_idx)
        rna_tr = bundle.rna_norm_train
        rna_va = bundle.rna_norm_test[:n_val_local]
        rna_te = bundle.rna_norm_test[n_val_local:]
        pro_tr = bundle.protein_norm_train
        pro_va = bundle.protein_norm_test[:n_val_local]
        pro_te = bundle.protein_norm_test[n_val_local:]

        set_seed(args.seed + fold_id)
        stage1 = ContrastiveAutoencoder(
            rna_dim=len(bundle.gene_names),
            protein_dim=len(bundle.protein_names),
            hidden_dim=s1c.hidden_dim,
            latent_dim=s1c.latent_dim,
            proj_dim=s1c.proj_dim,
            depth=s1c.depth,
            dropout=s1c.dropout,
        ).to(device)
        train_stage1(stage1, bundle, s1c, device, log)

        z_tr = l2_normalize_np(encode_rna(stage1, rna_tr, device))
        z_va = l2_normalize_np(encode_rna(stage1, rna_va, device))
        z_te = l2_normalize_np(encode_rna(stage1, rna_te, device))

        # Train-only bank during model selection: no validation-label leakage.
        bank = FlowMemoryBank(temperature=args.retrieval_temperature)
        bank.fit(z_tr, pro_tr)
        ret_tr = bank.retrieve(z_tr, k=args.k, query_bank_indices=np.arange(len(z_tr)))
        ret_va = bank.retrieve(z_va, k=args.k)
        ret_te = bank.retrieve(z_te, k=args.k)

        retrieval_metrics = evaluate_predictions(
            ret_te.protein_prior, pro_te, bundle.protein_names, verbose=False
        )
        log(f"retrieval test PCC-P={retrieval_metrics['pcc_protein_mean']:.4f}")

        flow = OneStepRetrievalFM(
            protein_dim=len(bundle.protein_names),
            rna_dim=s1c.latent_dim,
            hidden_dim=args.flow_hidden_dim,
            depth=args.flow_depth,
            dropout=args.flow_dropout,
            condition_mode=args.condition_mode,
            use_similarity=True,
        ).to(device)
        train_arrays = (
            ret_tr.protein_prior, z_tr, ret_tr.reference_feature,
            ret_tr.mean_similarity, pro_tr,
        )
        val_arrays = (
            ret_va.protein_prior, z_va, ret_va.reference_feature,
            ret_va.mean_similarity, pro_va,
        )
        test_arrays = (
            ret_te.protein_prior, z_te, ret_te.reference_feature,
            ret_te.mean_similarity, pro_te,
        )
        train_result = train_one_step_fm(
            flow,
            train_arrays,
            val_arrays,
            bundle.protein_names,
            device=device,
            epochs=args.flow_epochs,
            batch_size=args.batch_size,
            lr=args.flow_lr,
            weight_decay=args.flow_weight_decay,
            endpoint_weight=args.endpoint_weight,
            source_noise_std=args.source_noise_std,
            eval_every=args.eval_every,
            log_fn=log,
        )
        flow.load_state_dict(train_result.best_state)
        flow.eval()

        nfe_metrics = {}
        for nfe in sorted(set(args.nfe)):
            pred = predict_in_batches(
                flow,
                ret_te.protein_prior,
                z_te,
                ret_te.reference_feature,
                ret_te.mean_similarity,
                batch_size=args.batch_size,
                device=device,
                nfe=nfe,
            )
            np.save(fold_dir / f"predictions_nfe{nfe}.npy", pred)
            metrics = evaluate_predictions(pred, pro_te, bundle.protein_names, verbose=False)
            nfe_metrics[str(nfe)] = {
                key: float(value)
                for key, value in metrics.items()
                if np.isscalar(value)
            }
            log(
                f"NFE={nfe} PCC-P={metrics['pcc_protein_mean']:.4f} "
                f"PCC-C={metrics['pcc_cell_mean']:.4f} MMD={metrics.get('mmd', float('nan')):.4f}"
            )

        diagnostics = export_diagnostics(flow, test_arrays, fold_dir, device)
        np.save(fold_dir / "ground_truth.npy", pro_te)
        np.save(fold_dir / "retrieval_prior.npy", ret_te.protein_prior)
        np.save(fold_dir / "query_rna_latent.npy", z_te)
        np.save(fold_dir / "retrieved_rna_latent.npy", ret_te.reference_feature)
        np.save(fold_dir / "rna_displacement.npy", z_te - ret_te.reference_feature)
        np.savez_compressed(
            fold_dir / "retrieval_details.npz",
            indices=ret_te.indices,
            scores=ret_te.scores,
            weights=ret_te.weights,
            mean_similarity=ret_te.mean_similarity,
        )
        torch.save(
            {
                "model_state_dict": train_result.best_state,
                "model_config": {
                    "protein_dim": len(bundle.protein_names),
                    "rna_dim": s1c.latent_dim,
                    "hidden_dim": args.flow_hidden_dim,
                    "depth": args.flow_depth,
                    "dropout": args.flow_dropout,
                    "condition_mode": args.condition_mode,
                    "use_similarity": True,
                },
                "best_epoch": train_result.best_epoch,
                "best_val_pcc_nfe1": train_result.best_val_pcc,
            },
            fold_dir / "one_step_fm_best.pt",
        )
        save_json(fold_dir / "history.json", {"history": train_result.history})
        fold_summary = {
            "fold": fold_id,
            "best_epoch": train_result.best_epoch,
            "best_val_pcc_nfe1": train_result.best_val_pcc,
            "retrieval_pcc_protein": float(retrieval_metrics["pcc_protein_mean"]),
            "nfe_metrics": nfe_metrics,
            "diagnostics": diagnostics,
            "seconds": time.time() - fold_start,
        }
        save_json(fold_dir / "summary.json", fold_summary)
        all_results.append(fold_summary)
        log_file.close()

    pcc_nfe1 = [r["nfe_metrics"]["1"]["pcc_protein_mean"] for r in all_results]
    overall = {
        "dataset": args.dataset,
        "folds": all_results,
        "mean_pcc_protein_nfe1": float(np.mean(pcc_nfe1)),
        "std_pcc_protein_nfe1": float(np.std(pcc_nfe1)),
        "config": vars(args),
    }
    save_json(output_root / "summary.json", overall)
    print(json.dumps({k: v for k, v in overall.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
