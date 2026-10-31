"""Stage-3 retrieval augmented residual & flow models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_modules.networks import AdaLNZeroMLP


class RetrievalFlowModel(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vector_field = AdaLNZeroMLP(
            input_dim=protein_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.vector_field(x_t, t, cond)


class ResidualMLP(nn.Module):
    """Simple MLP predicting residual. Replaces ResidualCorrectionModel with configurable depth."""

    def __init__(
        self,
        protein_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        with_raw_rna: bool = False,
        raw_rna_dim: int = 0,
    ) -> None:
        super().__init__()
        input_dim = protein_dim + cond_dim + (raw_rna_dim if with_raw_rna else 0)
        self.with_raw_rna = with_raw_rna
        layers = []
        dims = [input_dim] + [hidden_dim] * depth + [protein_dim]
        for idx in range(len(dims) - 2):
            layers.extend([
                nn.Linear(dims[idx], dims[idx + 1]),
                nn.LayerNorm(dims[idx + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, start: torch.Tensor, cond: torch.Tensor, raw_rna: torch.Tensor | None = None) -> torch.Tensor:
        inputs = [start, cond]
        if self.with_raw_rna and raw_rna is not None:
            inputs.append(raw_rna)
        return self.net(torch.cat(inputs, dim=-1))


class GateResidualMLP(nn.Module):
    """Residual with gating: pred = start + gate * delta."""

    def __init__(
        self,
        protein_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        with_raw_rna: bool = False,
        raw_rna_dim: int = 0,
    ) -> None:
        super().__init__()
        input_dim = protein_dim + cond_dim + (raw_rna_dim if with_raw_rna else 0)
        self.with_raw_rna = with_raw_rna
        layers = []
        dims = [input_dim] + [hidden_dim] * depth + [hidden_dim]
        for idx in range(len(dims) - 2):
            layers.extend([
                nn.Linear(dims[idx], dims[idx + 1]),
                nn.LayerNorm(dims[idx + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.backbone = nn.Sequential(*layers)
        self.delta_head = nn.Linear(hidden_dim, protein_dim)
        self.gate_head = nn.Linear(hidden_dim, protein_dim)

    def forward(self, start: torch.Tensor, cond: torch.Tensor, raw_rna: torch.Tensor | None = None) -> torch.Tensor:
        inputs = [start, cond]
        if self.with_raw_rna and raw_rna is not None:
            inputs.append(raw_rna)
        h = self.backbone(torch.cat(inputs, dim=-1))
        delta = self.delta_head(h)
        gate = torch.sigmoid(self.gate_head(h))
        return start + gate * delta


class ZINBResidualMLP(nn.Module):
    """Residual MLP with 3 output heads for ZINB loss (mu, theta, pi)."""

    def __init__(
        self,
        protein_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = protein_dim + cond_dim
        layers = []
        dims = [input_dim] + [hidden_dim] * depth + [hidden_dim]
        for idx in range(len(dims) - 2):
            layers.extend([
                nn.Linear(dims[idx], dims[idx + 1]),
                nn.LayerNorm(dims[idx + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, protein_dim)
        self.theta = nn.Parameter(torch.zeros(protein_dim))
        self.pi_head = nn.Linear(hidden_dim, protein_dim)

    def forward(self, start: torch.Tensor, cond: torch.Tensor, raw_rna: torch.Tensor | None = None) -> dict:
        h = self.backbone(torch.cat([start, cond], dim=-1))
        mu = F.softplus(self.mu_head(h)) + 1e-4
        theta = F.softplus(self.theta) + 1e-4
        pi = self.pi_head(h)
        return {"mu": mu, "theta": theta, "pi": pi, "pred": mu}


class GateAttentionResidual(nn.Module):
    """Per-protein self-attention with multi-k retrieval inputs."""

    def __init__(
        self,
        cond_dim: int,
        protein_dim: int,
        num_k: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.protein_dim = protein_dim
        self.num_k = num_k
        self.protein_embed = nn.Parameter(torch.randn(protein_dim, hidden_dim) * 0.02)
        self.context = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.start_proj = nn.Linear(num_k, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, starts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        start_tokens = starts.transpose(1, 2)
        h = self.start_proj(start_tokens)
        h = h + self.protein_embed.unsqueeze(0) + self.context(cond).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        base = starts[:, starts.size(1) // 2, :].unsqueeze(-1)
        delta = self.delta_head(h)
        gate = torch.sigmoid(self.gate_head(h))
        pred = base + gate * delta
        return pred.squeeze(-1)


class AlphaBlendResidual(nn.Module):
    """Exp-B: Learnable alpha blending between retrieval and RNA-only prediction.
    pred = alpha * start + (1 - alpha) * pro_from_rna
    """

    def __init__(
        self,
        protein_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.protein_dim = protein_dim
        # RNA → protein direct path
        rna_dims = [cond_dim] + [hidden_dim] * depth + [protein_dim]
        rna_layers = []
        for i in range(len(rna_dims) - 2):
            rna_layers.extend([nn.Linear(rna_dims[i], rna_dims[i+1]), nn.LayerNorm(rna_dims[i+1]), nn.GELU(), nn.Dropout(dropout)])
        rna_layers.append(nn.Linear(rna_dims[-2], rna_dims[-1]))
        self.rna_to_pro = nn.Sequential(*rna_layers)
        # Alpha weight predictor (from cond)
        alpha_dims = [cond_dim] + [hidden_dim] * max(depth - 1, 1) + [protein_dim]
        alpha_layers = []
        for i in range(len(alpha_dims) - 2):
            alpha_layers.extend([nn.Linear(alpha_dims[i], alpha_dims[i+1]), nn.LayerNorm(alpha_dims[i+1]), nn.GELU(), nn.Dropout(dropout)])
        alpha_layers.append(nn.Linear(alpha_dims[-2], alpha_dims[-1]))
        self.alpha_net = nn.Sequential(*alpha_layers)

    def forward(self, start: torch.Tensor, cond: torch.Tensor, raw_rna: torch.Tensor | None = None) -> torch.Tensor:
        pro_from_rna = self.rna_to_pro(cond)
        alpha = torch.sigmoid(self.alpha_net(cond))
        return alpha * start + (1 - alpha) * pro_from_rna


class FiLMResidualMLP(nn.Module):
    """FiLM-conditioned residual. Supports two modes via use_adaln_zero flag.

    use_adaln_zero=True (default): AdaLN-Zero with identity-preserving init.
        Best for datasets with larger residuals (ECCITE, GSE3P). No mask needed.

    use_adaln_zero=False: Standard FiLM with default init (gamma*h+beta).
        Best for datasets with small residuals (CBMC). Requires mask_prob=0.2.
    """

    def __init__(
        self,
        protein_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        use_adaln_zero: bool = True,
    ) -> None:
        super().__init__()
        self.use_adaln_zero = use_adaln_zero
        self.input_proj = nn.Linear(protein_dim, hidden_dim)
        self.blocks = nn.ModuleList()

        if use_adaln_zero:
            # AdaLN-Zero: zero-init cond_proj(→gamma,beta,alpha) + skip connection
            for _ in range(depth):
                block = nn.ModuleDict({
                    "linear": nn.Linear(hidden_dim, hidden_dim),
                    "norm": nn.LayerNorm(hidden_dim, elementwise_affine=False),
                    "cond_proj": nn.Linear(cond_dim, hidden_dim * 3),
                    "dropout": nn.Dropout(dropout),
                })
                nn.init.zeros_(block["cond_proj"].weight)
                nn.init.zeros_(block["cond_proj"].bias)
                self.blocks.append(block)
            self.output = nn.Linear(hidden_dim, protein_dim)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            # Standard FiLM: gamma*h+beta, no zero-init (matches old3_2)
            for _ in range(depth):
                block = nn.ModuleDict({
                    "linear": nn.Linear(hidden_dim, hidden_dim),
                    "norm": nn.LayerNorm(hidden_dim),
                    "gamma": nn.Linear(cond_dim, hidden_dim),
                    "beta": nn.Linear(cond_dim, hidden_dim),
                    "dropout": nn.Dropout(dropout),
                })
                self.blocks.append(block)
            self.output = nn.Linear(hidden_dim, protein_dim)

    def forward(self, start: torch.Tensor, cond: torch.Tensor,
                raw_rna: torch.Tensor | None = None, t: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(start)
        if self.use_adaln_zero:
            for block in self.blocks:
                h_in = h
                h = block["linear"](h)
                h = block["norm"](h)
                gamma, beta, alpha = block["cond_proj"](cond).chunk(3, dim=-1)
                h = h * (1 + gamma) + beta
                h = F.gelu(h)
                h = block["dropout"](h)
                h = h_in + h * alpha
        else:
            for block in self.blocks:
                h = block["linear"](h)
                h = block["norm"](h)
                gamma = block["gamma"](cond)
                beta = block["beta"](cond)
                h = gamma * h + beta
                h = F.gelu(h)
                h = block["dropout"](h)
        return self.output(h)


class FiLMGateResidualMLP(nn.Module):
    """FiLM + Gate heads. Supports AdaLN-Zero (default) or Standard FiLM mode."""

    def __init__(
        self,
        protein_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        use_zinb: bool = False,
        use_adaln_zero: bool = True,
    ) -> None:
        super().__init__()
        self.use_zinb = use_zinb
        self.use_adaln_zero = use_adaln_zero
        self.input_proj = nn.Linear(protein_dim, hidden_dim)
        self.blocks = nn.ModuleList()

        if use_adaln_zero:
            for _ in range(depth):
                block = nn.ModuleDict({
                    "linear": nn.Linear(hidden_dim, hidden_dim),
                    "norm": nn.LayerNorm(hidden_dim, elementwise_affine=False),
                    "cond_proj": nn.Linear(cond_dim, hidden_dim * 3),
                    "dropout": nn.Dropout(dropout),
                })
                nn.init.zeros_(block["cond_proj"].weight)
                nn.init.zeros_(block["cond_proj"].bias)
                self.blocks.append(block)
            self.delta_head = nn.Linear(hidden_dim, protein_dim)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
        else:
            for _ in range(depth):
                block = nn.ModuleDict({
                    "linear": nn.Linear(hidden_dim, hidden_dim),
                    "norm": nn.LayerNorm(hidden_dim),
                    "gamma": nn.Linear(cond_dim, hidden_dim),
                    "beta": nn.Linear(cond_dim, hidden_dim),
                    "dropout": nn.Dropout(dropout),
                })
                self.blocks.append(block)
            self.delta_head = nn.Linear(hidden_dim, protein_dim)

        self.gate_head = nn.Linear(hidden_dim, protein_dim)
        if use_zinb:
            self.theta = nn.Parameter(torch.zeros(protein_dim))
            self.pi_head = nn.Linear(hidden_dim, protein_dim)

    def forward(self, start: torch.Tensor, cond: torch.Tensor, raw_rna: torch.Tensor | None = None) -> torch.Tensor | dict:
        h = self.input_proj(start)
        if self.use_adaln_zero:
            for block in self.blocks:
                h_in = h
                h = block["linear"](h)
                h = block["norm"](h)
                gamma, beta, alpha = block["cond_proj"](cond).chunk(3, dim=-1)
                h = h * (1 + gamma) + beta
                h = F.gelu(h)
                h = block["dropout"](h)
                h = h_in + h * alpha
        else:
            for block in self.blocks:
                h = block["linear"](h)
                h = block["norm"](h)
                gamma = block["gamma"](cond)
                beta = block["beta"](cond)
                h = gamma * h + beta
                h = F.gelu(h)
                h = block["dropout"](h)
        delta = self.delta_head(h)
        gate = torch.sigmoid(self.gate_head(h))
        v = gate * delta
        if self.use_zinb:
            theta = F.softplus(self.theta) + 1e-4
            pi = self.pi_head(h)
            return {"mu": start + v, "theta": theta, "pi": pi, "pred": start + v}
        return v


@torch.no_grad()
# ─── Step 1: FiLMRectifiedFlow ────────────────────────────────────────

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half = embedding_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
    angles = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMFlowBlock(nn.Module):
    """AdaLN-Zero block: per-layer FiLM modulation + residual with alpha gating."""

    def __init__(self, hidden_dim: int, film_cond_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Each block generates its own gamma, beta, alpha (zero-init → identity)
        self.cond_proj = nn.Linear(film_cond_dim, hidden_dim * 3)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, film_cond: torch.Tensor) -> torch.Tensor:
        gamma, beta, alpha = self.cond_proj(film_cond).chunk(3, dim=-1)
        h = self.norm(x)
        h = h * (1 + gamma) + beta  # FiLM modulation
        h = self.mlp(h)
        return x + h * alpha  # Residual with zero-init alpha → identity at init


class FiLMRectifiedFlow(nn.Module):
    """Data-to-Data Rectified Flow with DiT-style AdaLN-Zero blocks."""

    def __init__(
        self,
        protein_dim: int = 13,
        cond_dim: int = 128,
        time_dim: int = 64,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.input_proj = nn.Linear(protein_dim, hidden_dim)

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # film_cond = concat(cond, t_emb), shared globally
        film_cond_dim = cond_dim + hidden_dim

        # DiT-style AdaLN-Zero blocks, each with own cond_proj
        self.blocks = nn.ModuleList([
            FiLMFlowBlock(hidden_dim, film_cond_dim, dropout) for _ in range(depth)
        ])

        self.velocity_head = nn.Linear(hidden_dim, protein_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(get_timestep_embedding(t, self.time_dim))
        film_cond = torch.cat([cond, t_emb], dim=-1)  # [B, cond_dim + hidden_dim]

        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, film_cond)
        return self.velocity_head(h)


@torch.no_grad()
def euler_integrate(
    model: FiLMRectifiedFlow,
    x_0: torch.Tensor,
    cond: torch.Tensor,
    steps: int = 1,
) -> torch.Tensor:
    x = x_0
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((x.shape[0],), step * dt, device=x.device)
        v = model(x, t, cond)
        x = x + v * dt
    return x


def apply_residual_correction(
    model: nn.Module,
    start: torch.Tensor,
    cond: torch.Tensor,
    raw_rna: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(model, (ZINBResidualMLP,)):
        out = model(start, cond, raw_rna)
        return out["pred"]
    if isinstance(model, FiLMGateResidualMLP):
        out = model(start, cond, raw_rna)
        if isinstance(out, dict):
            return out["pred"]  # ZINB dict: "pred" is absolute
        return start + out  # velocity → absolute
    if isinstance(model, (AlphaBlendResidual,)):
        return model(start, cond, raw_rna)
    if isinstance(model, FiLMRectifiedFlow):
        return euler_integrate(model, start, cond, steps=1)
    if isinstance(model, (FiLMResidualMLP, ResidualMLP, GateResidualMLP)):
        return start + model(start, cond, raw_rna)
    # Fallback for generic models (GateAttention etc.)
    out = model(start, cond, raw_rna)
    return out if isinstance(out, torch.Tensor) and out.shape == start.shape else start + out
