"""Shared output saving for all experiments. Avoids code duplication across datasets."""
import numpy as np, torch, os, json
from utils import save_json


def save_fold_outputs(fold_dir, *,
    train_idx, val_idx, test_idx,
    s1_state_dict, s1_config, gene_names, protein_names,
    s3_best_state, s3_config, s1_final_state_dict,
    bank_features, bank_payload,
    test_preds, test_truth,
    retrieval_preds,
    val_preds, val_truth,
    pcc_protein, pcc_cell,
    test_pcc, ret_pcc, cell_pcc,
    results_dict,
):
    """Save all outputs for one CV fold. Called by all experiment scripts."""

    os.makedirs(fold_dir, exist_ok=True)

    # Split indices
    np.savez(f'{fold_dir}/split.npz', train=train_idx, val=val_idx, test=test_idx)

    # Stage1 best model
    torch.save({
        'model_state_dict': s1_state_dict,
        'config': s1_config,
        'gene_names': gene_names,
        'protein_names': protein_names,
    }, f'{fold_dir}/stage1_best.pt')

    # Stage1 finetuned model (post-Stage3)
    torch.save({
        'model_state_dict': s1_final_state_dict,
        'config': s1_config,
        'gene_names': gene_names,
        'protein_names': protein_names,
    }, f'{fold_dir}/stage1_best_ft.pt')

    # Stage3 best model
    torch.save({
        'model_state_dict': s3_best_state,
        'stage1_state_dict': s1_final_state_dict,
        'config': s3_config,
    }, f'{fold_dir}/stage3_best.pt')

    # Bank
    np.savez_compressed(f'{fold_dir}/memory_bank.npz',
                        features=bank_features, payload=bank_payload)

    # Predictions
    np.save(f'{fold_dir}/predictions.npy', test_preds)
    np.save(f'{fold_dir}/ground_truth.npy', test_truth)
    np.save(f'{fold_dir}/retrieval_predictions.npy', retrieval_preds)
    np.save(f'{fold_dir}/val_predictions.npy', val_preds)
    np.save(f'{fold_dir}/val_ground_truth.npy', val_truth)

    # Per-protein/cell PCC for downstream plotting
    np.save(f'{fold_dir}/pcc_protein.npy', pcc_protein)
    np.save(f'{fold_dir}/pcc_cell.npy', pcc_cell)
    np.save(f'{fold_dir}/protein_names.npy', np.array(protein_names))

    # Full metrics JSON
    save_json(f'{fold_dir}/metrics.json', {
        'pcc_protein_mean': float(results_dict.get('pcc_protein_mean', test_pcc)),
        'pcc_cell_mean': float(results_dict.get('pcc_cell_mean', cell_pcc)),
        'cmd_protein': float(results_dict.get('cmd_protein', 0)),
        'cmd_cell': float(results_dict.get('cmd_cell', 0)),
        'rmse': float(results_dict.get('rmse', 0)),
        'mmd': float(results_dict.get('mmd', 0)),
        'retrieval_pcc': float(ret_pcc),
        'stage3_delta': float(test_pcc - ret_pcc),
    })


def log_fold_summary(fold_results, log_fn):
    """Print and log 5CV summary table."""
    pccs = [r['test_pcc'] for r in fold_results]
    rets = [r['retrieval'] for r in fold_results]
    cells = [r['test_cell'] for r in fold_results]
    a, ra, ca = np.array(pccs), np.array(rets), np.array(cells)

    log_fn(f'{"Fold":<8} {"Retrieval":>10} {"Stage3 PCC":>10} {"Cell PCC":>10}')
    log_fn('-' * 42)
    for r in fold_results:
        log_fn(f'{r["fold"]:<8} {r["retrieval"]:>10.4f} {r["test_pcc"]:>10.4f} {r["test_cell"]:>10.4f}')
    log_fn(f'{"Mean±Std":<8} {ra.mean():>10.4f} {a.mean():>10.4f} {ca.mean():>10.4f}')
    log_fn(f'{"":<8} {"±"+str(ra.std())[:6]:>10} {"±"+str(a.std())[:6]:>10} {"±"+str(ca.std())[:6]:>10}')

    return a.mean(), a.std()
