from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import SymmetricInfoNCE, zinb_negative_log_likelihood


class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.gate(self.norm(x)).chunk(2, dim=-1)
        h = F.gelu(value * torch.sigmoid(gate))
        return x + self.output(self.dropout(h))


class GatedResidualEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        for block in self.blocks:
            h = block(h)
        return self.output_projection(self.output_norm(h))


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.dropout_head = nn.Linear(hidden_dim, output_dim)
        self.dispersion = nn.Parameter(torch.zeros(output_dim))

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(latent)
        mean = F.softplus(self.mean_head(hidden)) + 1e-4
        dispersion = F.softplus(self.dispersion) + 1e-4
        dropout_logits = self.dropout_head(hidden)
        return mean, dispersion, dropout_logits


class ContrastiveZINBAutoencoder(nn.Module):
    def __init__(
        self,
        rna_dim: int,
        protein_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        projection_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
        temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        self.rna_encoder = GatedResidualEncoder(
            rna_dim, hidden_dim, latent_dim, depth, dropout
        )
        self.protein_encoder = GatedResidualEncoder(
            protein_dim, hidden_dim, latent_dim, depth, dropout
        )
        self.rna_projection = ProjectionHead(latent_dim, projection_dim)
        self.protein_projection = ProjectionHead(latent_dim, projection_dim)
        self.rna_decoder = ZINBDecoder(latent_dim, hidden_dim, rna_dim)
        self.protein_decoder = ZINBDecoder(latent_dim, hidden_dim, protein_dim)
        self.info_nce = SymmetricInfoNCE(temperature_init)

    def encode_rna(self, rna_normalized: torch.Tensor) -> torch.Tensor:
        return self.rna_encoder(rna_normalized)

    def forward(
        self, rna_normalized: torch.Tensor, protein_normalized: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        rna_latent = self.rna_encoder(rna_normalized)
        protein_latent = self.protein_encoder(protein_normalized)
        rna_mean, rna_dispersion, rna_dropout = self.rna_decoder(rna_latent)
        protein_mean, protein_dispersion, protein_dropout = self.protein_decoder(
            protein_latent
        )
        return {
            "rna_latent": rna_latent,
            "protein_latent": protein_latent,
            "rna_projection": self.rna_projection(rna_latent),
            "protein_projection": self.protein_projection(protein_latent),
            "rna_mean": rna_mean,
            "rna_dispersion": rna_dispersion,
            "rna_dropout": rna_dropout,
            "protein_mean": protein_mean,
            "protein_dispersion": protein_dispersion,
            "protein_dropout": protein_dropout,
        }

    def objective(
        self,
        rna_normalized: torch.Tensor,
        rna_counts: torch.Tensor,
        protein_normalized: torch.Tensor,
        protein_counts: torch.Tensor,
        contrastive_weight: float,
        rna_zinb_weight: float,
        protein_zinb_weight: float,
    ) -> dict[str, torch.Tensor]:
        output = self(rna_normalized, protein_normalized)
        contrastive = self.info_nce(
            output["rna_projection"], output["protein_projection"]
        )
        rna_zinb = zinb_negative_log_likelihood(
            rna_counts,
            output["rna_mean"],
            output["rna_dispersion"],
            output["rna_dropout"],
        )
        protein_zinb = zinb_negative_log_likelihood(
            protein_counts,
            output["protein_mean"],
            output["protein_dispersion"],
            output["protein_dropout"],
        )
        total = (
            contrastive_weight * contrastive
            + rna_zinb_weight * rna_zinb
            + protein_zinb_weight * protein_zinb
        )
        return {
            "total": total,
            "contrastive": contrastive,
            "rna_zinb": rna_zinb,
            "protein_zinb": protein_zinb,
        }


def sinusoidal_time_embedding(t: torch.Tensor, dimension: int) -> torch.Tensor:
    t = t.reshape(-1)
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    angles = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class AdaLNZeroBlock(nn.Module):
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

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(context).chunk(3, dim=-1)
        transformed = self.norm(hidden) * (1.0 + scale) + shift
        transformed = self.mlp(transformed)
        return hidden + torch.tanh(gate) * transformed


class RetrievalConditionedFlow(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        rna_latent_dim: int = 128,
        hidden_dim: int = 512,
        depth: int = 4,
        time_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.input_projection = nn.Linear(protein_dim, hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(rna_latent_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim, hidden_dim, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, protein_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(
        self,
        protein_state: torch.Tensor,
        time: torch.Tensor,
        query_rna_latent: torch.Tensor,
        retrieval_similarity: torch.Tensor,
    ) -> torch.Tensor:
        query_rna_latent = F.normalize(query_rna_latent, dim=-1)
        if retrieval_similarity.ndim == 1:
            retrieval_similarity = retrieval_similarity[:, None]
        condition = torch.cat([query_rna_latent, retrieval_similarity], dim=-1)
        context = self.condition_projection(condition) + self.time_projection(
            sinusoidal_time_embedding(time.to(protein_state.dtype), self.time_dim)
        )
        hidden = self.input_projection(protein_state)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.velocity_head(self.final_norm(hidden))

    @torch.no_grad()
    def predict_one_step(
        self,
        retrieved_source: torch.Tensor,
        query_rna_latent: torch.Tensor,
        retrieval_similarity: torch.Tensor,
    ) -> torch.Tensor:
        t0 = torch.zeros(
            retrieved_source.shape[0],
            device=retrieved_source.device,
            dtype=retrieved_source.dtype,
        )
        return retrieved_source + self(
            retrieved_source, t0, query_rna_latent, retrieval_similarity
        )
