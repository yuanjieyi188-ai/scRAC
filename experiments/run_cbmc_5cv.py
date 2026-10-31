"""CBMC(example) 5-Fold CV: Standard FiLM+Gate + mask=0.2."""
import sys, time, numpy as np, torch, os
ROOT = '/data3/liuyinbo/fastpaper/cell_gene_model/scClip/scClip_old3_3P5P'
sys.path.insert(0, ROOT)

from pipeline.config import get_cbmc_config
from pipeline.utils_train import encode_rna
from pipeline.save_utils import save_fold_outputs, log_fold_summary
from utils import preprocess_single_dataset, set_seed, save_json
from core_modules.retrieval import MemoryBank
from models.clip_vae import ContrastiveAutoencoder, Stage1LossWeights
from models.rag_flow import FiLMGateResidualMLP
from metrics import evaluate_predictions
import torch.nn as nn
import scanpy as sc

s1c, s3c = get_cbmc_config()
device = 'cuda'; OUT = ROOT + '/outputs/cbmc_5cv'
os.makedirs(OUT, exist_ok=True)
DATA = ROOT + '/data/example.h5ad'
T0 = time.time()

def L(msg): t = time.time()-T0; lf.write(f'[{t:6.0f}s] {msg}\n'); lf.flush(); print(f'[{t:6.0f}s] {msg}', flush=True)

lf = open(OUT + '/5cv.log', 'w')
L('CBMC 5-Fold CV: Standard FiLM+Gate + mask=0.2')

adata = sc.read_h5ad(DATA); n = adata.n_obs
rng = np.random.default_rng(s1c.seed); shuffled = rng.permutation(n)
folds = np.array_split(shuffled, 5)
L(f'Folds: {[len(f) for f in folds]}')

fold_results = []
for fid in range(5):
    L(f'\nFOLD {fid}:')
    test_idx = folds[fid]; train_pool = np.concatenate([folds[i] for i in range(5) if i != fid])
    val_frac = s3c.val_ratio / 0.8; n_pool = len(train_pool); n_val = int(n_pool * val_frac)
    perm = rng.permutation(n_pool)
    val_idx = train_pool[perm[:n_val]]; train_idx = train_pool[perm[n_val:]]
    L(f'  Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

    # Stage1
    t1 = time.time()
    bundle = preprocess_single_dataset(DATA, n_top_genes=s1c.n_top_genes,
        use_protein_guided_hvg=s1c.use_protein_guided_hvg, seed=s1c.seed,
        fold_train_idx=train_pool, fold_test_idx=test_idx)
    P = len(bundle.protein_names)
    pool_loc = {orig: i for i, orig in enumerate(train_pool)}
    tl_ = np.array([pool_loc[i] for i in train_idx]); vl_ = np.array([pool_loc[i] for i in val_idx])
    rna_tr = bundle.rna_norm_train[tl_]; rna_va = bundle.rna_norm_train[vl_]; rna_te = bundle.rna_norm_test
    prot_tr = bundle.protein_norm_train[tl_]; prot_va = bundle.protein_norm_train[vl_]; prot_te = bundle.protein_norm_test

    set_seed(s1c.seed); s1 = ContrastiveAutoencoder(
        rna_dim=len(bundle.gene_names), protein_dim=P, hidden_dim=s1c.hidden_dim,
        latent_dim=s1c.latent_dim, proj_dim=s1c.proj_dim, depth=s1c.depth, dropout=s1c.dropout).to(device)
    w = Stage1LossWeights(s1c.alpha, s1c.beta, s1c.gamma)
    o1 = torch.optim.AdamW(s1.parameters(), lr=s1c.lr, weight_decay=s1c.weight_decay)
    ds1 = torch.utils.data.TensorDataset(torch.from_numpy(bundle.rna_norm_train), torch.from_numpy(bundle.rna_raw_train),
        torch.from_numpy(bundle.protein_norm_train), torch.from_numpy(bundle.protein_raw_train))
    for ep in range(1, s1c.epochs+1):
        s1.train()
        for batch in torch.utils.data.DataLoader(ds1, s1c.batch_size, True):
            b = {'rna_norm': batch[0].to(device), 'rna_raw': batch[1].to(device),
                 'protein_norm': batch[2].to(device), 'protein_raw': batch[3].to(device)}
            o1.zero_grad(); lo = s1.compute_loss(b, w); lo['loss'].backward()
            torch.nn.utils.clip_grad_norm_(s1.parameters(), 1.0); o1.step()
    s1.eval(); L(f'  S1: {time.time()-t1:.0f}s')

    # Bank: train only (CBMC specific)
    tc_tr = encode_rna(s1, rna_tr, device); tc_va = encode_rna(s1, rna_va, device); tc_te = encode_rna(s1, rna_te, device)
    bank = MemoryBank(); bank.fit(tc_tr, prot_tr)
    ts_tr = bank.aggregate(tc_tr, k=s3c.k_retrieval, exclude_self=True)
    ts_va = bank.aggregate(tc_va, k=s3c.k_retrieval, exclude_self=False)
    ts_te = bank.aggregate(tc_te, k=s3c.k_retrieval, exclude_self=False)
    ret_pcc = evaluate_predictions(ts_te, prot_te, bundle.protein_names, verbose=False)['pcc_protein_mean']
    L(f'  Retrieval: {ret_pcc:.4f}')

    # Stage3: Standard FiLM+Gate + mask=0.2
    s1_pre_state = {k: v.clone() for k, v in s1.state_dict().items()}
    t3 = time.time(); s1.train()
    for p in s1.parameters(): p.requires_grad = True
    m = FiLMGateResidualMLP(P, 128, s3c.hidden_dim, s3c.depth, s3c.dropout, use_adaln_zero=s3c.use_adaln_zero).to(device)
    o3 = torch.optim.AdamW(list(m.parameters())+list(s1.parameters()), lr=s3c.lr, weight_decay=s3c.weight_decay)
    ds3 = torch.utils.data.TensorDataset(torch.from_numpy(rna_tr.astype(np.float32)), torch.from_numpy(ts_tr),
                                          torch.from_numpy(prot_tr.astype(np.float32)))
    dl3 = torch.utils.data.DataLoader(ds3, s3c.batch_size, True)
    bvp = -1.0; bs = None
    for ep in range(1, s3c.epochs+1):
        m.train(); s1.train()
        for rb, sb, tb in dl3:
            rb, sb, tb = rb.to(device), sb.to(device), tb.to(device); cd = s1.encode_rna(rb)
            mask = torch.rand_like(sb) < s3c.mask_prob; sb = torch.where(mask, torch.zeros_like(sb), sb)
            vp = m(sb, cd); loss = torch.mean((vp - (tb - sb))**2)
            o3.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o3.step()
        if ep % s3c.eval_every == 0 or ep <= 2 or ep == s3c.epochs:
            m.eval(); s1.eval()
            with torch.no_grad():
                tc_va_cur = encode_rna(s1, rna_va, device)
                sv = torch.from_numpy(ts_va).to(device); cv = torch.from_numpy(tc_va_cur).to(device)
                vpc = evaluate_predictions((sv+m(sv,cv)).cpu().numpy(), prot_va, bundle.protein_names, verbose=False)['pcc_protein_mean']
                if vpc > bvp: bvp = vpc; bs = {k: v.clone() for k, v in m.state_dict().items()}

    m.load_state_dict(bs); m.eval(); s1.eval()
    with torch.no_grad():
        tc_tr2 = encode_rna(s1, rna_tr, device); tc_te2 = encode_rna(s1, rna_te, device)
        bk2 = MemoryBank(); bk2.fit(tc_tr2, prot_tr)
        tse2 = bk2.aggregate(tc_te2, s3c.k_retrieval, False)
        st_ = torch.from_numpy(tse2).to(device); ct_ = torch.from_numpy(tc_te2).to(device)
        pr = (st_ + m(st_, ct_)).cpu().numpy().astype(np.float32)
    res = evaluate_predictions(pr, prot_te, bundle.protein_names, verbose=False)
    tp = res['pcc_protein_mean']; tc_cell = res['pcc_cell_mean']
    L(f'  S3: {time.time()-t3:.0f}s  Test PCC={tp:.4f}  Cell={tc_cell:.4f}')

    tc_va_final = encode_rna(s1, rna_va, device)
    sv = torch.from_numpy(ts_va).to(device); cv = torch.from_numpy(tc_va_final).to(device)
    val_pr = (sv + m(sv, cv)).detach().cpu().numpy()

    save_fold_outputs(OUT + f'/fold{fid}',
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        s1_state_dict=s1_pre_state,
        s1_config={'rna_dim': len(bundle.gene_names), 'protein_dim': P, 'hidden_dim': s1c.hidden_dim,
                   'latent_dim': s1c.latent_dim, 'proj_dim': s1c.proj_dim, 'depth': s1c.depth, 'dropout': s1c.dropout},
        gene_names=bundle.gene_names, protein_names=bundle.protein_names,
        s3_best_state=bs, s3_config={'use_adaln_zero': s3c.use_adaln_zero, 'mask_prob': s3c.mask_prob},
        s1_final_state_dict=s1.state_dict(),
        bank_features=encode_rna(s1, rna_tr, device), bank_payload=prot_tr,
        test_preds=pr, test_truth=prot_te, retrieval_preds=ts_te,
        val_preds=val_pr, val_truth=prot_va,
        pcc_protein=res['pcc_protein'], pcc_cell=res['pcc_cell'],
        test_pcc=tp, ret_pcc=ret_pcc, cell_pcc=tc_cell, results_dict=res)
    fold_results.append({'fold': fid, 'retrieval': float(ret_pcc), 'test_pcc': float(tp), 'test_cell': float(tc_cell)})

L(f'\n{"="*60}\n5CV SUMMARY\n{"="*60}')
mean_pcc, std_pcc = log_fold_summary(fold_results, L)
L(f'\nTOTAL: {(time.time()-T0)/60:.0f} min')
save_json(OUT + '/summary.json', {'folds': fold_results, 'mean': float(mean_pcc), 'std': float(std_pcc)})
lf.close()
