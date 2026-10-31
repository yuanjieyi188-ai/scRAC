"""Cross-dataset: GSE3P→ECCITE with encoder FT + bank rebuild. Full pipeline."""
import sys, time, numpy as np, torch, os
ROOT = '/data3/liuyinbo/fastpaper/cell_gene_model/scClip/scClip_old3_3P5P'
sys.path.insert(0, ROOT)

from pipeline.config import get_gse3p_config
from pipeline.utils_train import encode_rna
from pipeline.save_utils import save_fold_outputs
from utils import preprocess_single_dataset, set_seed, save_json, log_normalize
from core_modules.retrieval import MemoryBank
from models.clip_vae import ContrastiveAutoencoder, Stage1LossWeights
from models.rag_flow import FiLMGateResidualMLP
from metrics import evaluate_predictions
import torch.nn as nn
import scanpy as sc

s1c, s3c = get_gse3p_config()
device = 'cuda'; OUT = ROOT + '/outputs/gse3p_to_eccite'
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

def L(msg): t = time.time()-T0; lf.write(f'[{t:6.0f}s] {msg}\n'); lf.flush(); print(f'[{t:6.0f}s] {msg}', flush=True)
lf = open(OUT + '/log.txt', 'w')
L('GSE3P->ECCITE Cross-Dataset (FT+rebuild)')

# ═══════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════
gse = sc.read_h5ad(ROOT + '/data/GSE164378_3P.h5ad')
ecc = sc.read_h5ad(ROOT + '/data/ECCITE-seq-5P.h5ad')
gse_proteins = list(gse.obsm['protein_expression'].columns)
ecc_proteins = list(ecc.obsm['protein_expression'].columns)
shared = sorted(set(gse_proteins) & set(ecc_proteins))
L(f'GSE3P: {gse.n_obs}c {len(gse_proteins)}p | ECCITE: {ecc.n_obs}c {len(ecc_proteins)}p | Shared: {len(shared)}')

# ═══════════════════════════════════════════
# 2. GSE3P split & Stage1 training
# ═══════════════════════════════════════════
n_gse = gse.n_obs; rng = np.random.default_rng(s1c.seed); shuf = rng.permutation(n_gse)
n_val = int(n_gse * 0.01); n_tst = int(n_gse * 0.01)  # minimal val/test for cross-dataset
tst_idx = shuf[:n_tst]; val_idx = shuf[n_tst:n_tst+n_val]; tr_idx = shuf[n_tst+n_val:]
L(f'GSE3P split: train={len(tr_idx)} val={len(val_idx)} test={len(tst_idx)}')

bundle = preprocess_single_dataset(ROOT + '/data/GSE164378_3P.h5ad',
    n_top_genes=s1c.n_top_genes, use_per_protein_hvg=s1c.use_per_protein_hvg,
    n_per_protein=s1c.n_per_protein, seed=s1c.seed,
    fold_train_idx=np.concatenate([tr_idx, val_idx]), fold_test_idx=tst_idx)
P = len(bundle.protein_names); gse_genes = bundle.gene_names
L(f'HVG: {len(gse_genes)} genes, {P} proteins')

pool_loc = {o: i for i, o in enumerate(np.concatenate([tr_idx, val_idx]))}
tr_loc = np.array([pool_loc[i] for i in tr_idx]); va_loc = np.array([pool_loc[i] for i in val_idx])
rna_tr = bundle.rna_norm_train[tr_loc]; rna_va = bundle.rna_norm_train[va_loc]; rna_te = bundle.rna_norm_test
prot_tr = bundle.protein_norm_train[tr_loc]; prot_va = bundle.protein_norm_train[va_loc]; prot_te = bundle.protein_norm_test
prot_raw_tr = bundle.protein_raw_train[tr_loc]; prot_raw_va = bundle.protein_raw_train[va_loc]

L('=== Phase 1: Train GSE3P Stage1 ===')
t1 = time.time(); set_seed(s1c.seed)
s1 = ContrastiveAutoencoder(len(gse_genes), P, s1c.hidden_dim, s1c.latent_dim, s1c.proj_dim,
    s1c.depth, s1c.dropout).to(device)
w = Stage1LossWeights(s1c.alpha, s1c.beta, s1c.gamma); o1 = torch.optim.AdamW(s1.parameters(), lr=s1c.lr, weight_decay=s1c.weight_decay)
ds1 = torch.utils.data.TensorDataset(
    torch.from_numpy(np.concatenate([rna_tr, rna_va], 0)),
    torch.from_numpy(np.concatenate([bundle.rna_raw_train[tr_loc], bundle.rna_raw_train[va_loc]], 0)),
    torch.from_numpy(np.concatenate([prot_tr, prot_va], 0)),
    torch.from_numpy(np.concatenate([prot_raw_tr, prot_raw_va], 0)))
for ep in range(1, s1c.epochs+1):
    s1.train()
    for batch in torch.utils.data.DataLoader(ds1, s1c.batch_size, True):
        b = {'rna_norm': batch[0].to(device), 'rna_raw': batch[1].to(device),
             'protein_norm': batch[2].to(device), 'protein_raw': batch[3].to(device)}
        o1.zero_grad(); lo = s1.compute_loss(b, w); lo['loss'].backward()
        torch.nn.utils.clip_grad_norm_(s1.parameters(), 1.0); o1.step()
s1.eval(); L(f'Stage1: {time.time()-t1:.0f}s')

# ═══════════════════════════════════════════
# 3. Bank + Stage3
# ═══════════════════════════════════════════
tc_tr = encode_rna(s1, rna_tr, device); tc_va = encode_rna(s1, rna_va, device); tc_te = encode_rna(s1, rna_te, device)
bank = MemoryBank(); bank.fit(np.concatenate([tc_tr, tc_va], 0), np.concatenate([prot_tr, prot_va], 0))
ts_tr = bank.aggregate(tc_tr, s3c.k_retrieval, True); ts_va = bank.aggregate(tc_va, s3c.k_retrieval, True)
ts_te = bank.aggregate(tc_te, s3c.k_retrieval, False)
ret_gse = evaluate_predictions(ts_te, prot_te, bundle.protein_names, verbose=False)['pcc_protein_mean']
L(f'GSE3P Retrieval: {ret_gse:.4f}')

L('=== Phase 2: Train GSE3P Stage3 ===')
t3 = time.time(); s1.train()
for p in s1.parameters(): p.requires_grad = True
m = FiLMGateResidualMLP(P, 128, s3c.hidden_dim, s3c.depth, s3c.dropout,
    use_adaln_zero=s3c.use_adaln_zero).to(device)
nn.init.zeros_(m.delta_head.weight); nn.init.zeros_(m.delta_head.bias)
o3 = torch.optim.AdamW(list(m.parameters())+list(s1.parameters()), lr=s3c.lr, weight_decay=s3c.weight_decay)
ds3 = torch.utils.data.TensorDataset(torch.from_numpy(rna_tr.astype(np.float32)),
    torch.from_numpy(ts_tr), torch.from_numpy(prot_tr.astype(np.float32)))
bvl = float('inf'); bs = None
for ep in range(1, s3c.epochs+1):
    m.train(); s1.train()
    for rb, sb, tb in torch.utils.data.DataLoader(ds3, s3c.batch_size, True):
        rb, sb, tb = rb.to(device), sb.to(device), tb.to(device); cd = s1.encode_rna(rb)
        vp = m(sb, cd); loss = torch.mean((vp - (tb - sb))**2)
        o3.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters())+list(s1.parameters()), 1.0); o3.step()
    if ep % s3c.eval_every == 0 or ep <= 2 or ep == s3c.epochs:
        m.eval(); s1.eval()
        with torch.no_grad():
            vp = np.concatenate([(torch.from_numpy(ts_va[i:i+1024]).to(device) +
                m(torch.from_numpy(ts_va[i:i+1024]).to(device), torch.from_numpy(tc_va[i:i+1024]).to(device))).detach().cpu().numpy()
                for i in range(0, len(tc_va), 1024)], 0)
            if np.mean((vp - prot_va)**2) < bvl: bvl = np.mean((vp - prot_va)**2); bs = {k: v.clone() for k, v in m.state_dict().items()}

m.load_state_dict(bs); m.eval(); s1.eval()
with torch.no_grad():
    pr_gse = np.concatenate([(torch.from_numpy(ts_te[i:i+1024]).to(device) +
        m(torch.from_numpy(ts_te[i:i+1024]).to(device), torch.from_numpy(tc_te[i:i+1024]).to(device))).detach().cpu().numpy()
        for i in range(0, len(tc_te), 1024)], 0).astype(np.float32)
res_gse = evaluate_predictions(pr_gse, prot_te, bundle.protein_names, verbose=False)
s3_gse = res_gse['pcc_protein_mean']
L(f'GSE3P Stage3: {s3_gse:.4f}  Delta={s3_gse-ret_gse:+.4f} ({time.time()-t3:.0f}s)')

# Save GSE3P training artifacts
val_pr = np.concatenate([(torch.from_numpy(ts_va[i:i+1024]).to(device) +
    m(torch.from_numpy(ts_va[i:i+1024]).to(device), torch.from_numpy(tc_va[i:i+1024]).to(device))).detach().cpu().numpy()
    for i in range(0, len(tc_va), 1024)], 0)
save_fold_outputs(OUT + '/gse3p_training',
    train_idx=tr_idx, val_idx=val_idx, test_idx=tst_idx,
    s1_state_dict=s1.state_dict(),
    s1_config={'rna_dim': len(gse_genes), 'protein_dim': P, 'hidden_dim': s1c.hidden_dim,
               'latent_dim': s1c.latent_dim, 'proj_dim': s1c.proj_dim, 'depth': s1c.depth, 'dropout': s1c.dropout},
    gene_names=gse_genes, protein_names=bundle.protein_names,
    s3_best_state=bs, s3_config={'use_adaln_zero': s3c.use_adaln_zero, 'mask_prob': s3c.mask_prob},
    s1_final_state_dict=s1.state_dict(),
    bank_features=np.concatenate([tc_tr, tc_va], 0), bank_payload=np.concatenate([prot_tr, prot_va], 0),
    test_preds=pr_gse, test_truth=prot_te, retrieval_preds=ts_te,
    val_preds=val_pr, val_truth=prot_va,
    pcc_protein=res_gse['pcc_protein'], pcc_cell=res_gse['pcc_cell'],
    test_pcc=s3_gse, ret_pcc=ret_gse, cell_pcc=res_gse['pcc_cell_mean'], results_dict=res_gse)

# ═══════════════════════════════════════════
# 4. Align ECCITE RNA to GSE3P genes
# ═══════════════════════════════════════════
L('=== Phase 3: Align ECCITE RNA & Fine-tune Encoder ===')
gene_to_idx = {g: i for i, g in enumerate(gse_genes)}
ecc_gene_set = set(ecc.var_names)
common_genes = [g for g in gse_genes if g in ecc_gene_set]
L(f'Common genes: {len(common_genes)}/{len(gse_genes)}')

ecc_rna = np.zeros((ecc.n_obs, len(gse_genes)), dtype=np.float32)
for g in common_genes:
    gi_ecc = list(ecc.var_names).index(g); gi_gse = gene_to_idx[g]
    col = ecc.X[:, gi_ecc]; ecc_rna[:, gi_gse] = col.toarray().flatten() if hasattr(col, 'toarray') else col
ecc_rna_norm = log_normalize(ecc_rna)
np.save(OUT + '/eccite_aligned_rna_norm.npy', ecc_rna_norm.astype(np.float32))

# GSE3P per-gene mean for batch correction
gse_train_rna = np.zeros((len(tr_idx) + len(val_idx), len(gse_genes)), dtype=np.float32)
for g in common_genes:
    gi_gse = list(gse.var_names).index(g); gi_out = gene_to_idx[g]
    col = gse.X[np.concatenate([tr_idx, val_idx]), gi_gse]
    gse_train_rna[:, gi_out] = col.toarray().flatten() if hasattr(col, 'toarray') else col
gse_train_norm = log_normalize(np.clip(gse_train_rna, 0, None))
gse_gene_mean = gse_train_norm.mean(0)
np.save(OUT + '/gse_gene_means.npy', gse_gene_mean.astype(np.float32))

# ═══════════════════════════════════════════
# 5. Fine-tune encoder on ECCITE RNA (reconstruction only)
# ═══════════════════════════════════════════
for n, p in s1.named_parameters(): p.requires_grad = 'rna_encoder' in n
opt_ft = torch.optim.AdamW(filter(lambda p: p.requires_grad, s1.parameters()), lr=5e-5)
loss_fn = nn.MSELoss(); N_FT = 10

for ep in range(1, N_FT + 1):
    s1.train(); tl = 0.0; nb = 0
    for i in range(0, len(ecc_rna_norm), 256):
        rna_b = torch.from_numpy(ecc_rna_norm[i:i+256]).to(device)
        z = s1.rna_encoder(rna_b); rna_mu, _, _ = s1.rna_decoder(z)
        loss = loss_fn(rna_mu, rna_b)
        opt_ft.zero_grad(); loss.backward(); opt_ft.step()
        tl += loss.item(); nb += 1
    if ep % 5 == 0 or ep <= 2: L(f'  FT epoch {ep}: recon_loss={tl/max(nb,1):.4f}')

s1.eval()
for p in s1.parameters(): p.requires_grad = True
torch.save({'model_state_dict': s1.state_dict(), 'config': {'rna_dim': len(gse_genes), 'protein_dim': P,
    'hidden_dim': s1c.hidden_dim, 'latent_dim': s1c.latent_dim, 'proj_dim': s1c.proj_dim,
    'depth': s1c.depth, 'dropout': s1c.dropout}, 'gene_names': gse_genes}, OUT + '/stage1_finetuned.pt')

# ═══════════════════════════════════════════
# 6. Rebuild bank + ECCITE inference
# ═══════════════════════════════════════════
L('=== Phase 4: Rebuild bank & ECCITE inference ===')
# Re-encode GSE3P bank with FT encoder
gse_rna_full = np.zeros((len(tr_idx) + len(val_idx), len(gse_genes)), dtype=np.float32)
for g in common_genes:
    gi_gse = list(gse.var_names).index(g); gi_out = gene_to_idx[g]
    col = gse.X[np.concatenate([tr_idx, val_idx]), gi_gse]
    gse_rna_full[:, gi_out] = col.toarray().flatten() if hasattr(col, 'toarray') else col
gse_rna_full_norm = log_normalize(np.clip(gse_rna_full, 0, None))
gse_feat_ft = encode_rna(s1, gse_rna_full_norm, device)

bank_ft = MemoryBank(); bank_ft.fit(gse_feat_ft, np.concatenate([prot_tr, prot_va], 0))
np.savez_compressed(OUT + '/memory_bank_ft.npz', features=gse_feat_ft,
    payload=np.concatenate([prot_tr, prot_va], 0))

# Batch-corrected ECCITE RNA
ecc_input = ecc_rna_norm - ecc_rna_norm.mean(0) + gse_gene_mean
np.save(OUT + '/eccite_aligned_rna_bc.npy', ecc_input.astype(np.float32))

ecc_feat_ft = encode_rna(s1, ecc_input, device)
ecc_start_ft = bank_ft.aggregate(ecc_feat_ft, s3c.k_retrieval, False)
with torch.no_grad():
    ecc_preds_ft = np.concatenate([(torch.from_numpy(ecc_start_ft[i:i+1024]).to(device) +
        m(torch.from_numpy(ecc_start_ft[i:i+1024]).to(device), torch.from_numpy(ecc_feat_ft[i:i+1024]).to(device))).detach().cpu().numpy()
        for i in range(0, len(ecc_feat_ft), 1024)], 0).astype(np.float32)
L(f'ECCITE predictions: {ecc_preds_ft.shape}')

# ═══════════════════════════════════════════
# 7. Evaluate shared proteins
# ═══════════════════════════════════════════
L('=== Phase 5: Evaluate ===')
gse_p2i = {p: i for i, p in enumerate(gse_proteins)}; ecc_p2i = {p: i for i, p in enumerate(ecc_proteins)}
sh_gse = [gse_p2i[p] for p in shared]; sh_ecc = [ecc_p2i[p] for p in shared]

ecc_pred_sh = ecc_preds_ft[:, sh_gse]
ecc_start_sh = ecc_start_ft[:, sh_gse]
ecc_truth_raw = np.column_stack([ecc.obsm['protein_expression'].values[:, i] for i in sh_ecc])
ecc_truth_norm = log_normalize(np.clip(ecc_truth_raw, 0, None))

# ── Protein batch correction: align ECCITE proteins to GSE3P space ──
gse_tr_prot_raw = gse.obsm['protein_expression'].values[np.concatenate([tr_idx, val_idx])]
gse_tr_prot_norm = log_normalize(np.clip(gse_tr_prot_raw, 0, None))
gse_prot_mean_bc = np.array([gse_tr_prot_norm[:, gse_p2i[p]].mean() for p in shared])
ecc_prot_mean_bc = ecc_truth_norm.mean(0)
ecc_truth_bc = ecc_truth_norm - ecc_prot_mean_bc + gse_prot_mean_bc
L(f'Protein BC: ECCITE mean {ecc_prot_mean_bc.mean():.3f}->{ecc_truth_bc.mean(0).mean():.3f} (GSE3P: {gse_prot_mean_bc.mean():.3f})')
np.save(OUT + '/gse_prot_means.npy', gse_prot_mean_bc.astype(np.float32))

L('\n--- Retrieval (shared proteins, protein-BC) ---')
ret_ev = evaluate_predictions(ecc_start_sh, ecc_truth_bc, shared, verbose=True)
ret_rmse_raw = float(np.sqrt(np.mean((ecc_start_sh - ecc_truth_bc)**2)))
L(f'PCC-P={ret_ev["pcc_protein_mean"]:.4f}  CMD-P={ret_ev.get("cmd_protein",0):.4f}  CMD-C={ret_ev.get("cmd_cell",0):.4f}  RMSE(z)={ret_ev.get("rmse",0):.4f}  RMSE(raw)={ret_rmse_raw:.4f}  MMD={ret_ev.get("mmd",0):.4f}')

L('\n--- Stage3 (shared proteins, protein-BC) ---')
s3_ev = evaluate_predictions(ecc_pred_sh, ecc_truth_bc, shared, verbose=True)
s3_rmse_raw = float(np.sqrt(np.mean((ecc_pred_sh - ecc_truth_bc)**2)))
L(f'PCC-P={s3_ev["pcc_protein_mean"]:.4f}  CMD-P={s3_ev.get("cmd_protein",0):.4f}  CMD-C={s3_ev.get("cmd_cell",0):.4f}  RMSE(z)={s3_ev.get("rmse",0):.4f}  RMSE(raw)={s3_rmse_raw:.4f}  MMD={s3_ev.get("mmd",0):.4f}')
L(f'Stage3 Delta: PCC-P {s3_ev["pcc_protein_mean"]-ret_ev["pcc_protein_mean"]:+.4f}  RMSE(raw) {s3_rmse_raw-ret_rmse_raw:+.4f}')

# ═══════════════════════════════════════════
# 8. Save all cross-dataset outputs
# ═══════════════════════════════════════════
np.save(OUT + '/eccite_all_228_predictions.npy', ecc_preds_ft)
np.save(OUT + '/eccite_features_ft.npy', ecc_feat_ft)
np.save(OUT + '/eccite_retrieval_start_ft.npy', ecc_start_ft)
np.save(OUT + '/eccite_shared_truth.npy', ecc_truth_bc)
np.save(OUT + '/eccite_shared_truth_raw.npy', ecc_truth_norm)
np.save(OUT + '/eccite_shared_retrieval.npy', ecc_start_sh)
np.save(OUT + '/eccite_shared_predictions.npy', ecc_pred_sh)
np.save(OUT + '/shared_pcc_protein_retrieval.npy', ret_ev['pcc_protein'])
np.save(OUT + '/shared_pcc_protein_stage3.npy', s3_ev['pcc_protein'])
np.save(OUT + '/gse_proteins.npy', np.array(gse_proteins))
np.save(OUT + '/eccite_proteins.npy', np.array(ecc_proteins))
np.save(OUT + '/shared_proteins.npy', np.array(shared))
np.save(OUT + '/gse_genes.npy', np.array(gse_genes))
np.save(OUT + '/common_genes.npy', np.array(common_genes))

save_json(OUT + '/cross_dataset_metrics.json', {
    'gse3p_retrieval_pcc': float(ret_gse), 'gse3p_stage3_pcc': float(s3_gse),
    'gse3p_delta': float(s3_gse - ret_gse),
    'eccite_retrieval_pcc_protein': float(ret_ev['pcc_protein_mean']),
    'eccite_retrieval_cmd_protein': float(ret_ev.get('cmd_protein', 0)),
    'eccite_retrieval_cmd_cell': float(ret_ev.get('cmd_cell', 0)),
    'eccite_retrieval_rmse_z': float(ret_ev.get('rmse', 0)),
    'eccite_retrieval_rmse_raw': float(ret_rmse_raw),
    'eccite_retrieval_mmd': float(ret_ev.get('mmd', 0)),
    'eccite_stage3_pcc_protein': float(s3_ev['pcc_protein_mean']),
    'eccite_stage3_cmd_protein': float(s3_ev.get('cmd_protein', 0)),
    'eccite_stage3_cmd_cell': float(s3_ev.get('cmd_cell', 0)),
    'eccite_stage3_rmse_z': float(s3_ev.get('rmse', 0)),
    'eccite_stage3_rmse_raw': float(s3_rmse_raw),
    'eccite_stage3_mmd': float(s3_ev.get('mmd', 0)),
    'stage3_delta_pcc': float(s3_ev['pcc_protein_mean'] - ret_ev['pcc_protein_mean']),
    'n_shared_proteins': len(shared),
    'method': 'FT 10ep encoder on ECCITE + bank rebuild',
})

L(f'\nTOTAL: {(time.time()-T0)/60:.0f} min  Output: {OUT}')
lf.close()
