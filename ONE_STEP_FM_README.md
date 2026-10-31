# Retrieval-initialized One-Step Flow Matching

## Recommended first run

Use one fold and shortened epochs to verify the full data pipeline:

```bash
python -u experiments/run_one_step_fm_5cv.py \
  --dataset cbmc \
  --data_path data/example.h5ad \
  --only_fold 0 \
  --stage1_epochs 5 \
  --flow_epochs 10 \
  --device cuda
```

Then run the complete experiment:

```bash
python -u experiments/run_one_step_fm_5cv.py \
  --dataset cbmc \
  --data_path data/example.h5ad \
  --flow_epochs 40 \
  --endpoint_weight 1.0 \
  --condition_mode query_delta \
  --nfe 1 2 4 8 \
  --device cuda
```

Replace the dataset name/path with `eccite` or `gse3p` as needed.

## Main model

- Source protein state: softmax-weighted top-k retrieved protein prior.
- Retrieved RNA reference: the same weights applied to neighbor RNA latents.
- Transcriptomic displacement: normalized query latent minus retrieved RNA latent.
- Flow path: `x_t=(1-t)x_0+t*x_1`.
- Flow target: `x_1-x_0`.
- Inference: one Euler step `x_hat=x_0+v_theta(x_0,0,c)`.
- Training: random-time flow-matching loss plus an endpoint anchoring loss.

## Why Stage 1 is frozen

The retrieved source state depends on the Stage-1 latent geometry. Updating the
encoder while keeping precomputed nearest neighbors and priors fixed makes the
retrieval condition stale. The supplied first-pass implementation freezes Stage
1, which is the safest setting for demonstrating that one-step FM works.

## Files produced per fold

- `predictions_nfe{1,2,4,8}.npy`
- `retrieval_prior.npy`
- `rna_displacement.npy`
- `flow_diagnostics.npz`
- `one_step_fm_best.pt`
- `summary.json`
