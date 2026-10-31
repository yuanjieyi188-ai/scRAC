"""Retrieval-initialized one-step conditional flow matching.

The source state is a retrieval-synthesized protein prior x0. The target state
is the paired protein measurement x1. A time-conditioned vector field is
trained on the straight coupling x_t=(1-t)x0+t*x1 and is additionally anchored
at t=0 so that a single Euler step is accurate at inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim != 1:
        t = t.reshape(-1)
    half = dim // 2
    if half == 0:
        return t[:, None]
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FlowAdaLNZeroBlock(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Linear(context_dim, 3 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(context).chunk(3, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        h = self.mlp(h)
        return x + torch.tanh(gate) * h


class OneStepRetrievalFM(nn.Module):
    """Time-conditioned vector field for local RNA-to-protein transport.

    condition_mode:
      - "query": query RNA latent only.
      - "delta": transcriptomic displacement z_query - z_reference only.
      - "query_delta": concatenates query latent and displacement (recommended).
      - "query_ref_delta": concatenates query, retrieved reference and displacement.

    Both query and reference representations are L2-normalized inside forward,
    making the displacement compatible with cosine retrieval.
    """

    def __init__(
        self,
        protein_dim: int,
        rna_dim: int = 128,
        hidden_dim: int = 512,
        depth: int = 3,
        time_dim: int = 64,
        dropout: float = 0.1,
        condition_mode: str = "query_delta",
        use_similarity: bool = True,
    ) -> None:
        super().__init__()
        valid = {"query", "delta", "query_delta", "query_ref_delta"}
        if condition_mode not in valid:
            raise ValueError(f"condition_mode must be one of {sorted(valid)}")
        self.protein_dim = protein_dim
        self.rna_dim = rna_dim
        self.time_dim = time_dim
        self.condition_mode = condition_mode
        self.use_similarity = use_similarity

        multiplier = {
            "query": 1,
            "delta": 1,
            "query_delta": 2,
            "query_ref_delta": 3,
        }[condition_mode]
        cond_in_dim = multiplier * rna_dim + (1 if use_similarity else 0)

        self.input_proj = nn.Linear(protein_dim, hidden_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FlowAdaLNZeroBlock(hidden_dim, hidden_dim, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, protein_dim)

        # Stable initialization: the initial vector field is exactly zero.
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def build_condition(
        self,
        query_rna: torch.Tensor,
        reference_rna: torch.Tensor,
        similarity: torch.Tensor | None,
    ) -> torch.Tensor:
        query = F.normalize(query_rna, dim=-1)
        reference = F.normalize(reference_rna, dim=-1)
        displacement = query - reference
        if self.condition_mode == "query":
            pieces = [query]
        elif self.condition_mode == "delta":
            pieces = [displacement]
        elif self.condition_mode == "query_delta":
            pieces = [query, displacement]
        else:
            pieces = [query, reference, displacement]
        if self.use_similarity:
            if similarity is None:
                similarity = torch.zeros(
                    query.shape[0], 1, device=query.device, dtype=query.dtype
                )
            if similarity.ndim == 1:
                similarity = similarity[:, None]
            pieces.append(similarity)
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        query_rna: torch.Tensor,
        reference_rna: torch.Tensor,
        similarity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t.ndim != 1:
            t = t.reshape(-1)
        cond = self.build_condition(query_rna, reference_rna, similarity)
        context = self.condition_encoder(cond) + self.time_encoder(
            timestep_embedding(t.to(x_t.dtype), self.time_dim)
        )
        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, context)
        return self.velocity_head(self.final_norm(h))


@dataclass
class FlowLossOutput:
    loss: torch.Tensor
    fm_loss: torch.Tensor
    endpoint_loss: torch.Tensor
    velocity_mse_at_zero: torch.Tensor


def one_step_flow_matching_loss(
    model: OneStepRetrievalFM,
    x0: torch.Tensor,
    x1: torch.Tensor,
    query_rna: torch.Tensor,
    reference_rna: torch.Tensor,
    similarity: torch.Tensor | None = None,
    endpoint_weight: float = 1.0,
    source_noise_std: float = 0.0,
) -> FlowLossOutput:
    """Standard conditional FM plus explicit one-step endpoint anchoring.

    If source noise is enabled, exactly the same perturbed source is used in
    x_t and the target velocity. This avoids the inconsistent-noise bug in the
    legacy rectified-flow branch.
    """
    if endpoint_weight < 0:
        raise ValueError("endpoint_weight must be non-negative")
    source = x0
    if source_noise_std > 0:
        source = source + source_noise_std * torch.randn_like(source)

    target_velocity = x1 - source
    t = torch.rand(source.shape[0], device=source.device, dtype=source.dtype)
    x_t = (1.0 - t[:, None]) * source + t[:, None] * x1
    pred_velocity = model(x_t, t, query_rna, reference_rna, similarity)
    fm_loss = F.mse_loss(pred_velocity, target_velocity)

    t0 = torch.zeros_like(t)
    pred_velocity_0 = model(source, t0, query_rna, reference_rna, similarity)
    endpoint_loss = F.mse_loss(source + pred_velocity_0, x1)
    loss = fm_loss + endpoint_weight * endpoint_loss
    return FlowLossOutput(
        loss=loss,
        fm_loss=fm_loss,
        endpoint_loss=endpoint_loss,
        velocity_mse_at_zero=F.mse_loss(pred_velocity_0, target_velocity),
    )


@torch.no_grad()
def integrate_flow(
    model: OneStepRetrievalFM,
    x0: torch.Tensor,
    query_rna: torch.Tensor,
    reference_rna: torch.Tensor,
    similarity: torch.Tensor | None = None,
    nfe: int = 1,
    solver: str = "euler",
) -> torch.Tensor:
    """Integrate from t=0 to 1. nfe=1 is the proposed inference rule."""
    if nfe < 1:
        raise ValueError("nfe must be >= 1")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")

    x = x0
    dt = 1.0 / nfe
    for step in range(nfe):
        t0 = torch.full(
            (x.shape[0],), step * dt, device=x.device, dtype=x.dtype
        )
        v0 = model(x, t0, query_rna, reference_rna, similarity)
        if solver == "euler":
            x = x + dt * v0
        else:
            x_euler = x + dt * v0
            t1 = torch.full(
                (x.shape[0],), (step + 1) * dt, device=x.device, dtype=x.dtype
            )
            v1 = model(x_euler, t1, query_rna, reference_rna, similarity)
            x = x + 0.5 * dt * (v0 + v1)
    return x


@torch.no_grad()
def flow_diagnostics(
    model: OneStepRetrievalFM,
    x0: torch.Tensor,
    x1: torch.Tensor,
    query_rna: torch.Tensor,
    reference_rna: torch.Tensor,
    similarity: torch.Tensor | None = None,
    n_time_points: int = 11,
) -> dict[str, torch.Tensor]:
    """Compute manuscript-ready velocity alignment and path variation."""
    target_v = x1 - x0
    times = torch.linspace(0.0, 1.0, n_time_points, device=x0.device, dtype=x0.dtype)
    velocities = []
    alignments = []
    for t_scalar in times:
        t = torch.full((x0.shape[0],), t_scalar, device=x0.device, dtype=x0.dtype)
        x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        v = model(x_t, t, query_rna, reference_rna, similarity)
        velocities.append(v)
        alignments.append(F.cosine_similarity(v, target_v, dim=-1))
    velocity_stack = torch.stack(velocities, dim=1)
    alignment_stack = torch.stack(alignments, dim=1)
    variation = (velocity_stack[:, 1:] - velocity_stack[:, :-1]).norm(dim=-1)
    variation = variation.mean(dim=1) / (target_v.norm(dim=-1) + 1e-8)
    return {
        "times": times,
        "alignment": alignment_stack,
        "path_variation": variation,
    }
