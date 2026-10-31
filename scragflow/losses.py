from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    def __init__(self, initial_temperature: float = 0.07) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(initial_temperature)))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-4, max=100.0)

    def forward(self, rna_projection: torch.Tensor, protein_projection: torch.Tensor) -> torch.Tensor:
        rna_projection = F.normalize(rna_projection, dim=-1)
        protein_projection = F.normalize(protein_projection, dim=-1)
        logits = rna_projection @ protein_projection.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )


def zinb_negative_log_likelihood(
    counts: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
    dropout_logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if dispersion.ndim == 1:
        dispersion = dispersion.unsqueeze(0)
    softplus_neg_pi = F.softplus(-dropout_logits)
    log_theta = torch.log(dispersion + eps)
    log_theta_mu = torch.log(dispersion + mean + eps)
    pi_theta_log = -dropout_logits + dispersion * (log_theta - log_theta_mu)

    zero_case = F.softplus(pi_theta_log) - softplus_neg_pi
    nonzero_case = (
        -softplus_neg_pi
        + pi_theta_log
        + counts * (torch.log(mean + eps) - log_theta_mu)
        + torch.lgamma(counts + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(counts + 1.0)
    )
    log_likelihood = torch.where(counts < eps, zero_case, nonzero_case)
    return -log_likelihood.mean()


@dataclass
class FlowLoss:
    total: torch.Tensor
    flow_matching: torch.Tensor
    endpoint: torch.Tensor


def flow_matching_objective(
    model: nn.Module,
    retrieved_source: torch.Tensor,
    target: torch.Tensor,
    query_latent: torch.Tensor,
    similarity: torch.Tensor,
    mask_probability: float,
    endpoint_weight: float,
) -> FlowLoss:
    """Paper objective with masked FM paths and an unmasked endpoint anchor."""
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be in [0, 1].")

    keep_mask = torch.rand_like(retrieved_source) >= mask_probability
    masked_source = retrieved_source * keep_mask.to(retrieved_source.dtype)

    t = torch.rand(
        retrieved_source.shape[0],
        device=retrieved_source.device,
        dtype=retrieved_source.dtype,
    )
    x_t = (1.0 - t[:, None]) * masked_source + t[:, None] * target
    target_velocity = target - masked_source
    predicted_velocity = model(x_t, t, query_latent, similarity)
    fm_loss = F.mse_loss(predicted_velocity, target_velocity)

    t0 = torch.zeros_like(t)
    endpoint_velocity = model(retrieved_source, t0, query_latent, similarity)
    endpoint_prediction = retrieved_source + endpoint_velocity
    endpoint_loss = F.mse_loss(endpoint_prediction, target)

    return FlowLoss(
        total=fm_loss + endpoint_weight * endpoint_loss,
        flow_matching=fm_loss,
        endpoint=endpoint_loss,
    )
