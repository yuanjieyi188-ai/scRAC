# scRAC

Retrieval-augmented one-step flow matching for single-cell RNA-to-protein
translation. Given a scRNA-seq profile, scRAC predicts the paired continuous
protein expression (CITE-seq / ADT) with a three-stage pipeline: a
cross-modal encoder, retrieval of a protein prior, and a one-step flow-matching
correction network.

## Method

### Stage 1 — Cross-modal contrastive autoencoder

`ContrastiveAutoencoder` (`models/clip_vae.py`) maps RNA and protein into a
shared latent space using gated-residual (GRN) encoders. It is trained with an
InfoNCE contrastive loss between the two modalities plus ZINB reconstruction
losses on each decoder, so the embedding retains biological structure while
aligning the modalities.

### Stage 2 — Retrieval-initialized protein prior

`FlowMemoryBank` (`core_modules/flow_retrieval.py`) stores L2-normalized RNA
latents and their true protein profiles. For a query cell it retrieves the
top-`k` nearest neighbors by cosine similarity and applies a softmax-weighted
average to synthesize:

- `x0` — the source protein prior, and
- the retrieved RNA reference used to form the transcriptomic displacement
  condition `normalize(z_query) - normalize(z_reference)`.

### Stage 3 — One-step conditional flow matching

`OneStepRetrievalFM` (`models/one_step_fm.py`) is a time-conditioned vector
field (AdaLN-Zero MLP) that transports the retrieval prior `x0` toward the true
protein `x1` along the straight coupling `x_t = (1-t) x_0 + t x_1`. It is
trained with a random-time flow-matching loss plus an endpoint-anchoring loss at
`t = 0`, so a **single Euler step** is accurate at inference:

```
x_hat = x0 + v_theta(x0, t=0, condition)
```

Supported condition modes: `query`, `delta`, `query_delta`, `query_ref_delta`.

## Datasets

| Name   | Description                      |
|--------|----------------------------------|
| `cbmc` | CBMC CITE-seq                    |
| `eccite`| ECCITE-seq-5P                   |
| `gse3p`| GSE164378 3' CITE-seq           |

Dataset presets live in `pipeline/config.py`.

## Installation

```bash
pip install torch scanpy anndata numpy scipy scikit-learn
# optional, for exact cosine retrieval
pip install faiss-cpu
```

## Usage

Main entry point — strict 5-fold cross-validation of the full pipeline:

```bash
python -u experiments/run_one_step_fm_5cv.py \
  --dataset eccite \
  --data_path data/ECCITE-seq-5P.annotated.h5ad \
  --device cuda
```

A quick one-fold smoke run:

```bash
python -u experiments/run_one_step_fm_5cv.py \
  --dataset eccite \
  --data_path data/ECCITE-seq-5P.annotated.h5ad \
  --only_fold 0 \
  --stage1_epochs 5 \
  --flow_epochs 10 \
  --device cuda
```

Key options:

- `--dataset` / `--data_path` — dataset and `.h5ad` input.
- `--folds` / `--only_fold` — number of CV folds, or run a single fold.
- `--condition_mode` — flow conditioning: `query`, `delta`, `query_delta`, `query_ref_delta`.
- `--finetune_stage1` — jointly fine-tune the Stage-1 encoder during flow training (default on).
- `--endpoint_weight` — weight of the one-step endpoint-anchoring loss.
- `--mask_prob` — source-masking probability for robustness augmentation.
- `--nfe` — number of function evaluations to report (`1 2 4 8`).

Dataset-specific scripts (`experiments/run_{cbmc,eccite,gse3p}_5cv.py`) and the
cross-dataset transfer script (`experiments/run_gse3p_to_eccite.py`) are also
provided.

## Evaluation

`metrics.py` reports per-protein Pearson correlation (`pcc_protein_mean`),
per-cell correlation (`pcc_cell_mean`), root mean squared error, and MMD. Per
fold, the run writes `predictions_nfe{1,2,4,8}.npy`, `retrieval_prior.npy`,
`ground_truth.npy`, `flow_diagnostics.npz`, and `summary.json`.

## Repository structure

```
core_modules/    # GRN networks, losses, retrieval bank
models/          # Stage-1 ContrastiveAutoencoder, Stage-3 OneStepRetrievalFM
pipeline/        # train stage 1, build bank, train/evaluate stage 3
experiments/     # 5-fold CV and cross-dataset scripts
metrics.py       # PCC / RMSE / MMD evaluation
utils.py         # data loading and preprocessing
```
