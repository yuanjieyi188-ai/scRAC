# scRAGFlow Core Implementation

> **Code scope:** This repository presents the core scRAGFlow model and benchmark workflow. The complete project code will be provided after the peer-review process.

scRAGFlow translates single-cell RNA profiles into surface-protein abundance through three stages:

1. protein-targeted gene screening and a contrastive ZINB autoencoder;
2. exact cosine retrieval of five reference cells and similarity-weighted protein-source synthesis;
3. retrieval-initialized, gene-conditioned one-step flow matching with AdaLN-Zero blocks.

The configuration in `configs/paper.yaml` follows the architecture, objectives, optimization settings, retrieval protocol, and NFE=1 inference rule reported in the paper. The experiment entry point validates these fixed values before training.

## Repository layout

```text
configs/paper.yaml          Paper configuration
datasets/README.md          Dataset sources and preparation
datasets/download_geo.sh    GEO processed-matrix downloader
datasets/prepare_h5ad.py    H5AD conversion and dimension checks
scragflow/data.py           H5AD loading, normalization, gene screening, folds
scragflow/models.py         Stage 1 encoders/decoders and Stage 3 vector field
scragflow/losses.py         InfoNCE, ZINB, and flow-matching objectives
scragflow/retrieval.py      FAISS IndexFlatIP retrieval and source synthesis
scragflow/training.py       Three-stage training and checkpoint selection
scragflow/metrics.py        Benchmark metrics
experiments/run_5fold.py    Five-fold benchmark runner
scripts/audit_release.py    Release consistency checks
tests/test_core.py          Core implementation tests
```

## Dataset preparation

Download and convert the CBMC, ECCITE-5′, and GSE3P-3′ matrices using the instructions in [`datasets/README.md`](datasets/README.md):

```bash
bash datasets/download_geo.sh all
python datasets/prepare_h5ad.py --dataset cbmc --raw-dir datasets/raw/cbmc --output datasets/processed/cbmc.h5ad
```

Each generated `.h5ad` file stores RNA UMI counts in `adata.X`, protein UMI counts in `adata.obsm["protein_expression"]`, gene names in `adata.var_names`, and protein names in the protein DataFrame columns.

RNA and protein counts are library-size normalized to 10,000 and transformed with `log1p`. Gene screening is fitted on the optimization partition of each fold: the top 100 genes for every protein are unioned, ranked by their maximum absolute Pearson correlation, and capped at 1,000 genes.

## Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate scragflow
```

A pip specification is also provided:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA execution, install the PyTorch build matching the local CUDA driver before installing the remaining requirements.

## Run the five-fold benchmark

```bash
python experiments/run_5fold.py \
  --data datasets/processed/cbmc.h5ad \
  --dataset-name CBMC \
  --output outputs/cbmc \
  --device cuda
```

The runner uses the fixed paper configuration and writes fold indices, selected genes, checkpoints, retrieval priors, predictions, targets, per-fold metrics, pooled out-of-fold metrics, and the resolved configuration.

A prepared split file can be supplied to reproduce established fold assignments:

```bash
python experiments/run_5fold.py \
  --data datasets/processed/cbmc.h5ad \
  --dataset-name CBMC \
  --output outputs/cbmc \
  --device cuda \
  --splits /path/to/five_fold_splits.json
```

The split JSON contains five objects with `fit`, `validation`, and `test` integer-index arrays.

## Paper configuration

- Gene screening: top 100 genes per protein; union capped at 1,000.
- Stage 1: hidden 256, latent 128, projection 64, depth 2, dropout 0.1.
- Stage 1 objective: `10 * InfoNCE + 0.5 * RNA-ZINB + 0.5 * protein-ZINB`.
- Stage 1 optimization: AdamW, 20 epochs, learning rate `4e-4`, weight decay `1e-4`, batch size 256, gradient clipping 1.0.
- Retrieval: L2-normalized 128-dimensional RNA latents, FAISS `IndexFlatIP`, `K=5`, temperature 1.0.
- Stage 3 condition: normalized query RNA latent plus weighted mean retrieval similarity.
- Stage 3 vector field: hidden 512, four AdaLN-Zero blocks, time dimension 64, dropout 0.1.
- Stage 3 objective: source-mask probability 0.2, unmasked endpoint weight 1.0, source noise 0.
- Stage 3 optimization: AdamW, 50 epochs, learning rate `1e-3`, weight decay `1e-4`, batch size 1,024.
- Inference: one Euler update, NFE=1.

## Consistency check

```bash
python scripts/audit_release.py
pytest -q
```
