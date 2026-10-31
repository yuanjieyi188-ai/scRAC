# Code audit and one-step FM migration notes

## What the current experiments actually train

The four scripts under `experiments/` instantiate `FiLMGateResidualMLP` and
optimize a source-endpoint residual target:

```python
vp = model(retrieved_protein, query_rna_latent)
loss = mse(vp, target_protein - retrieved_protein)
pred = retrieved_protein + vp
```

This is a valid retrieval-conditioned residual predictor, but it is not flow
matching because no random time, interpolated state, or time-conditioned vector
field is used.

## Existing flow code that was close but unsafe

`models/rag_flow.py` already contains `FiLMRectifiedFlow` and Euler integration.
The legacy training branch in `pipeline/03_train_torch_residual.py` has three
important issues:

1. It samples two different Gaussian perturbations for the state used in `x_t`
   and the state used to define the target velocity. The network is therefore
   asked to predict an inconsistent target.
2. It has no endpoint objective at `t=0`, so one Euler step is not explicitly
   optimized.
3. Evaluation hard-codes one step and does not report the NFE=1/2/4/8 evidence
   needed to justify the one-step claim.

## Recommended model

For query RNA latent `z_q`, retrieve the top-k reference cells and apply the
same softmax weights to both modalities:

- `x_0 = sum_k w_k y_k`: source protein prior.
- `z_0 = sum_k w_k z_k`: retrieved RNA reference.
- `delta_z = normalize(z_q) - normalize(z_0)`: transcriptomic displacement.

The vector field receives the current protein state, time, query RNA latent,
and transcriptomic displacement. The default condition is `[z_q, delta_z]`.

The term *displacement* is preferred to *perturbation*: nearest-neighbor cell
differences are observational and are not automatically causal interventions.

## Training objective

```text
x_t = (1-t) x_0 + t x_1,       t ~ Uniform(0,1)
u_t = x_1 - x_0
L_FM = ||v_theta(x_t,t,c)-u_t||^2
L_1step = ||x_0 + v_theta(x_0,0,c)-x_1||^2
L = L_FM + lambda_1step L_1step
```

Inference uses one Euler step:

```text
x_hat = x_0 + v_theta(x_0, 0, c)
```

## Validation protocol corrections

The supplied `run_one_step_fm_5cv.py` uses a train-only memory bank during
inner validation and trains Stage 1 only on the inner training split. This
avoids using validation protein labels through either Stage-1 supervision or
retrieval.

Stage 1 is frozen during flow training. If Stage 1 is fine-tuned, the memory
bank and all retrieved priors should be rebuilt as the embedding geometry
changes; otherwise the retrieval source becomes stale.

## First experiments to run

1. Run one fold with 5 Stage-1 epochs and 10 flow epochs to verify the pipeline.
2. Run CBMC with NFE 1, 2, 4, and 8.
3. Compare `condition_mode=query` and `condition_mode=query_delta`.
4. Only after CBMC is stable, launch full 5-fold ECCITE and GSE3P runs.
