# scRAG-Flow: ECCITE-5P Pipeline (May 2026)

## Pipeline

```
RNA ──► Stage1: Contrastive VAE ──► RNA Embedding ──┐
                                                     │
                                                     ▼
                                              Memory Bank (FAISS)
                                               top-5 retrieval
                                                     │
                                                     ▼
RNA ──► Stage1 (finetuned) ──► cond ──┐              │
                                      │              │
Retrieved Protein ────────────────────┤              │
                                      ▼              │
                      Stage3: AdaLN-Zero FiLM+Gate   │
                        v = gate * delta             │
                        pred = x0 + v                │
                                      │              │
                                      ▼              │
                               Predicted Protein     │
```

---

## Stage 1: Per-Protein Contrastive VAE

### Gene Selection — Per-Protein HVG
- For each protein, select top-100 genes by Pearson |r|
- Union across all proteins, capped at 3000

### Architecture
```
RNA (n_genes) ──► GRNEncoder(h=256, d=128) ──► z_rna (128)
Protein(P) ────► GRNEncoder(h=256, d=128) ──► z_prot (128)
                    │
                    ├── Projection + InfoNCE (alpha=10.0)
                    ├── ZINBDecoder → RNA (beta=0.5)
                    └── ZINBDecoder → Protein (gamma=0.5)
```
- lr=4e-4, epochs=40, batch=256

---

## Stage 2: FAISS Memory Bank
- L2-normalize RNA embeddings → cosine similarity search
- Train+Val bank, K=5 softmax-weighted average
- Train & Val: exclude_self=True
- Test: exclude_self=False

---

## Stage 3: AdaLN-Zero FiLM+Gate

### Architecture
```
x0(P) ──► Linear(P→512)
               │
    ┌── FiLM Block × 2 ──┐
    │  Linear→LN          │
    │  h = h*(1+γ) + β    │  ← cond(128) → cond_proj (zero-init)
    │  GELU→Dropout        │
    │  h = h_in + h*α     │  ← residual (zero-init alpha)
    └─────────────────────┘
               │
    ┌──────────┴──────────┐
    delta_head(512→P)     gate_head(512→P)→Sigmoid
    (zero-init)           (default init)
    └──────────┬──────────┘
               │
         v = gate * delta  (velocity)
         pred = x0 + v
```

### Loss (Flow Matching)
```
x0 = start_b                  (retrieved, NO mask)
x1 = target_b                 (ground truth)
v_pred = model(x0, cond)      (gate * delta)
v_true = x1 - x0              (residual)
loss = MSE(v_pred, v_true)
```

### Config
- hidden=512, depth=2, dropout=0.1
- lr=1e-3, batch=1024, epochs=20
- mask_prob=0.0, finetune_stage1=True
- delta_head zero-initialized (identity at start)
- Best model by val_loss

---

## Results: ECCITE-seq-5P (54 proteins, 49,147 cells)

### Standard 5-Fold CV (70/10/20 split)

| Fold | Retrieval | Stage3 PCC | Cell PCC |
|------|:---------:|:----------:|:--------:|
| 0 | 0.420 | 0.457 | 0.927 |
| 1 | 0.429 | 0.469 | 0.928 |
| 2 | 0.425 | 0.462 | 0.928 |
| 3 | 0.427 | 0.462 | 0.928 |
| 4 | 0.425 | 0.462 | 0.927 |
| **Mean ± Std** | **0.425 ± 0.003** | **0.462 ± 0.004** | **0.928 ± 0.000** |

Δ vs Retrieval: **+0.037**

---

### GSE164378-3P (228 proteins, 161,764 cells) — 80/20 Split

| Stage | PCC-Protein | PCC-Cell |
|-------|:-----------:|:--------:|
| Stage2 retrieval | 0.499 | 0.887 |
| Stage3 FiLM+Gate | **0.545** | **0.896** |
| Δ | **+0.046** | **+0.009** |

---

## Run

```bash
python experiments/run_eccite_5cv.py   # ECCITE 5-fold CV
python experiments/run_gse3p.py        # GSE3P 80/20 split
```

## File Structure
```
scClip_old3_3P5P/
├── models/           # Stage1 (clip_vae) + Stage3 (rag_flow)
├── core_modules/     # networks, losses, retrieval
├── pipeline/         # train, build bank, inference
├── experiments/      # run_eccite_5cv.py
├── data/             # ECCITE-seq-5P.h5ad
├── utils.py
├── metrics.py
└── best_model_architecture.md
```
