import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Simple MLP block used inside transformer-style modules."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Learnable attention pooling across a set dimension (permutation-invariant)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, set_len, dim)
        q = self.query.unsqueeze(0).unsqueeze(1)  # (1,1,dim)
        attn_logits = (x * q).sum(dim=-1)  # (batch, set_len)
        attn = attn_logits.softmax(dim=-1).unsqueeze(-1)  # (batch, set_len, 1)
        return (x * attn).sum(dim=1)  # (batch, dim)


class PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding for sequences (time axis)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, dim)
        time_len = x.size(1)
        return x + self.pe[:time_len].unsqueeze(0)


class EgoSetEncoder(nn.Module):
    """
    Encodes a set of agents at each timestep into a single vector that is
    permutation-invariant w.r.t. non-ego agents. The 0th agent is treated as
    the ego and always kept in index 0.
    """

    def __init__(
        self,
        feature_dim: int = 13,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Per-agent feature embedding
        self.agent_mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Role embedding to distinguish ego vs others (keeps permutation invariance across non-ego)
        self.role_embedding = nn.Embedding(2, d_model)  # 0: other, 1: ego

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.agent_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = AttentionPool(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch*time, agents, feature_dim)

        Returns:
            torch.Tensor: (batch*time, d_model) fused representation for the timestep
        """
        batch_time, num_agents, _ = x.shape

        # Per-agent embedding
        h = self.agent_mlp(x)  # (B*T, A, d)

        # Add role embedding (ego at index 0 -> role 1, others 0)
        role_ids = torch.zeros(batch_time, num_agents, dtype=torch.long, device=x.device)
        role_ids[:, 0] = 1
        h = h + self.role_embedding(role_ids)

        # Self-attention across agents (equivariant to permutation of non-ego)
        h = self.agent_encoder(h)  # (B*T, A, d)

        # Ego token and pooled context from others
        ego = h[:, 0, :]  # (B*T, d)
        others = h[:, 1:, :] if num_agents > 1 else h[:, 0:1, :]
        context = self.pool(others)  # (B*T, d)
        fused = torch.cat([ego, context], dim=-1)
        return self.fuse(fused)  # (B*T, d)


class ImitationTransformer(nn.Module):
    """
    Full model: per-timestep set encoder (agents) + temporal transformer over time.
    Predicts next movement for ego agent as 3 values in range [-3, 3].
    """

    def __init__(
        self,
        feature_dim: int = 13,
        d_model: int = 128,
        agent_heads: int = 8,
        agent_layers: int = 2,
        time_heads: int = 8,
        time_layers: int = 2,
        dropout: float = 0.1,
        max_time: int = 64,
    ):
        super().__init__()
        self.set_encoder = EgoSetEncoder(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=agent_heads,
            num_layers=agent_layers,
            dropout=dropout,
        )

        self.time_pos = PositionalEncoding(d_model, max_len=max_time)
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=time_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.time_encoder = nn.TransformerEncoder(time_encoder_layer, num_layers=time_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, agents, features)

        Returns:
            torch.Tensor: (batch, 3) prediction clamped to [-3, 3].
        """
        b, t, a, f = x.shape
        x = x.reshape(b * t, a, f)
        step_repr = self.set_encoder(x)  # (B*T, d)
        step_repr = step_repr.view(b, t, -1)  # (B, T, d)

        h = self.time_pos(step_repr)
        # Use a causal mask so we only attend to past and current timesteps
        attn_mask = torch.triu(torch.ones(t, t, device=h.device), diagonal=1).bool()
        h = self.time_encoder(h, mask=attn_mask)
        last = h[:, -1, :]  # (B, d) 
        out = self.head(last)  # (B, 3) 
        # Scale each output dimension to its natural range: [3, 3, 30]
        scales = torch.tensor([3.0, 3.0, 30.0], device=out.device, dtype=out.dtype).unsqueeze(0)
        return torch.tanh(out) * scales


def build_model(
    feature_dim: int = 13,
    d_model: int = 128,
    agent_heads: int = 8,
    agent_layers: int = 2,
    time_heads: int = 8,
    time_layers: int = 2,
    dropout: float = 0.1,
    max_time: int = 64,
) -> ImitationTransformer:
    return ImitationTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        agent_heads=agent_heads,
        agent_layers=agent_layers,
        time_heads=time_heads,
        time_layers=time_layers,
        dropout=dropout,
        max_time=max_time,
    )


