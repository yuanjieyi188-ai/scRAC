from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scragflow.config import load_paper_config
from scragflow.data import load_h5ad_dataset, load_or_create_splits, prepare_fold_data
from scragflow.metrics import evaluate
from scragflow.models import ContrastiveZINBAutoencoder, RetrievalConditionedFlow
from scragflow.retrieval import ExactProteinMemory
from scragflow.training import (
    config_to_dict,
    encode_rna,
    predict_one_step,
    save_json,
    set_seed,
    train_stage1,
    train_stage3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-configured scRAGFlow five-fold benchmark.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paper_config(args.config)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = load_h5ad_dataset(args.data, config.experiment.protein_obsm_key)
    n_cells = dataset.protein_counts.shape[0]
    splits = load_or_create_splits(
        args.splits,
        n_cells,
        config.experiment.folds,
        config.experiment.seed,
        config.experiment.validation_fraction_of_reference,
    )
    save_json(args.output / "splits.json", splits)
    save_json(args.output / "paper_config.json", config_to_dict(config))

    pooled_prediction = np.empty_like(dataset.protein_counts, dtype=np.float32)
    pooled_target = np.empty_like(dataset.protein_counts, dtype=np.float32)
    fold_summaries: list[dict[str, object]] = []

    for fold_id, split in enumerate(splits):
        print(f"\n===== {args.dataset_name}: fold {fold_id + 1}/{config.experiment.folds} =====", flush=True)
        set_seed(config.experiment.seed + fold_id)
        fit_indices = np.asarray(split["fit"], dtype=np.int64)
        validation_indices = np.asarray(split["validation"], dtype=np.int64)
        test_indices = np.asarray(split["test"], dtype=np.int64)

        prepared = prepare_fold_data(
            dataset,
            fit_indices,
            config.preprocessing.normalization_target_sum,
            config.preprocessing.genes_per_protein,
            config.preprocessing.gene_cap,
        )
        fold_dir = args.output / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(fold_dir / "indices.json", split)
        save_json(fold_dir / "selected_genes.json", prepared.selected_genes)

        stage1 = ContrastiveZINBAutoencoder(
            rna_dim=prepared.rna_normalized.shape[1],
            protein_dim=prepared.protein_normalized.shape[1],
            hidden_dim=config.stage1.hidden_dim,
            latent_dim=config.stage1.latent_dim,
            projection_dim=config.stage1.projection_dim,
            depth=config.stage1.depth,
            dropout=config.stage1.dropout,
            temperature_init=config.stage1.temperature_init,
        ).to(device)
        stage1_history = train_stage1(
            stage1,
            prepared.rna_normalized[fit_indices],
            prepared.rna_counts[fit_indices],
            prepared.protein_normalized[fit_indices],
            prepared.protein_counts[fit_indices],
            config,
            device,
        )
        torch.save(stage1.state_dict(), fold_dir / "stage1_checkpoint.pt")
        save_json(fold_dir / "stage1_history.json", stage1_history)

        fit_latent = encode_rna(stage1, prepared.rna_normalized[fit_indices], device)
        initial_memory = ExactProteinMemory(config.retrieval.temperature)
        initial_memory.fit(fit_latent, prepared.protein_normalized[fit_indices])
        fit_retrieval = initial_memory.retrieve(
            fit_latent,
            config.retrieval.neighbors,
            self_indices=np.arange(len(fit_indices), dtype=np.int64),
        )

        flow = RetrievalConditionedFlow(
            protein_dim=prepared.protein_normalized.shape[1],
            rna_latent_dim=config.stage1.latent_dim,
            hidden_dim=config.stage3.hidden_dim,
            depth=config.stage3.depth,
            time_dim=config.stage3.time_dim,
            dropout=config.stage3.dropout,
        ).to(device)
        stage3_history, best_checkpoint = train_stage3(
            stage1,
            flow,
            prepared.rna_normalized[fit_indices],
            prepared.protein_normalized[fit_indices],
            prepared.rna_normalized[validation_indices],
            prepared.protein_normalized[validation_indices],
            fit_retrieval.protein_source,
            fit_retrieval.mean_similarity,
            config,
            device,
        )
        torch.save(
            {
                "flow": best_checkpoint["flow_state"],
                "rna_encoder": best_checkpoint["rna_encoder_state"],
                "best_epoch": best_checkpoint["best_epoch"],
                "best_validation_pcc_protein": best_checkpoint["best_validation_pcc_protein"],
            },
            fold_dir / "stage3_joint_checkpoint.pt",
        )
        save_json(fold_dir / "stage3_history.json", stage3_history)

        reference_indices = np.concatenate([fit_indices, validation_indices])
        reference_latent = encode_rna(
            stage1, prepared.rna_normalized[reference_indices], device
        )
        test_latent = encode_rna(stage1, prepared.rna_normalized[test_indices], device)
        final_memory = ExactProteinMemory(config.retrieval.temperature)
        reference_protein = prepared.protein_normalized[reference_indices]
        final_memory.fit(reference_latent, reference_protein)
        np.savez_compressed(
            fold_dir / "reference_bank.npz",
            reference_indices=reference_indices,
            rna_latents=reference_latent,
            protein_profiles=reference_protein,
        )
        test_retrieval = final_memory.retrieve(
            test_latent, config.retrieval.neighbors
        )
        test_prediction = predict_one_step(
            flow,
            stage1,
            prepared.rna_normalized[test_indices],
            test_retrieval.protein_source,
            test_retrieval.mean_similarity,
            device,
            config.stage3.batch_size,
        )
        test_target = prepared.protein_normalized[test_indices]
        test_metrics = evaluate(test_prediction, test_target)
        retrieval_metrics = evaluate(test_retrieval.protein_source, test_target)

        np.savez_compressed(
            fold_dir / "outputs.npz",
            prediction=test_prediction,
            target=test_target,
            retrieval_source=test_retrieval.protein_source,
            retrieval_similarity=test_retrieval.mean_similarity,
            retrieval_indices=test_retrieval.indices,
            retrieval_weights=test_retrieval.weights,
            protein_names=np.asarray(prepared.protein_names, dtype=object),
            test_indices=test_indices,
        )
        save_json(fold_dir / "test_metrics.json", test_metrics)
        save_json(fold_dir / "retrieval_metrics.json", retrieval_metrics)

        pooled_prediction[test_indices] = test_prediction
        pooled_target[test_indices] = test_target
        fold_summary = {
            "fold": fold_id,
            "selected_gene_count": len(prepared.selected_genes),
            "best_epoch": best_checkpoint["best_epoch"],
            "best_validation_pcc_protein": best_checkpoint["best_validation_pcc_protein"],
            "test_pcc_protein": test_metrics["pcc_protein_mean"],
            "test_pcc_cell": test_metrics["pcc_cell_mean"],
            "retrieval_pcc_protein": retrieval_metrics["pcc_protein_mean"],
        }
        fold_summaries.append(fold_summary)
        save_json(args.output / "fold_summary.json", fold_summaries)
        print(
            f"fold={fold_id} test_PCC-P={test_metrics['pcc_protein_mean']:.4f} "
            f"test_PCC-C={test_metrics['pcc_cell_mean']:.4f}",
            flush=True,
        )

    pooled_metrics = evaluate(pooled_prediction, pooled_target)
    np.savez_compressed(
        args.output / "pooled_out_of_fold.npz",
        prediction=pooled_prediction,
        target=pooled_target,
        protein_names=np.asarray(dataset.protein_names, dtype=object),
    )
    save_json(args.output / "pooled_metrics.json", pooled_metrics)
    save_json(
        args.output / "summary.json",
        {
            "dataset": args.dataset_name,
            "folds": fold_summaries,
            "pooled_metrics": pooled_metrics,
        },
    )
    print(
        f"\npooled PCC-P={pooled_metrics['pcc_protein_mean']:.4f} "
        f"PCC-C={pooled_metrics['pcc_cell_mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
