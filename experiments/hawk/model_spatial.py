import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.size(1)
        return x + self.pe[:t].unsqueeze(0)


class FeedForward(nn.Module):
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





class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim: int = 6, d_model: int = 256, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1, max_agents: int = 10):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.pos = PositionalEncoding(d_model, max_len=max_agents)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu")
        # MPS workaround: disable nested tensor path to avoid unsupported op
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.pos(h)
        # Bidirectional self-attention over all timesteps (no causal mask)
        h = self.encoder(h)
        return h[:, -1, :]  # (B, D)


class SpatialTransformer(nn.Module):
    def __init__(self, feature_dim: int = 6, d_model: int = 256, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1, max_agents: int = 10):
        super().__init__()
        self.temporal = TemporalEncoder(feature_dim=feature_dim, d_model=d_model, nhead=num_heads, num_layers=num_layers, dropout=dropout, max_agents=max_agents)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 10))

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        # temporal: (B, T, F)
        h = self.temporal(temporal)
        out = self.head(h)  # (B,10)
        # scale: 5 pairs of (dx,dy) -> [-1,1] each
        return torch.tanh(out)


def build_model(
    feature_dim: int = 6,
    d_model: int = 256,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    max_agents: int = 10,
) -> SpatialTransformer:
    return SpatialTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_agents=max_agents,
    )




if __name__ == "__main__":
    # Simple smoke test with random inputs
    torch.manual_seed(0)
    batch_size = 4
    A = 10  # number of agents
    F = 6  # feature length

    model = build_model(feature_dim=F, d_model=256, num_layers=2, num_heads=8, dropout=0.1, max_agents=10)
    model.eval()

    # Random inputs roughly in [-1, 1]
    spatial = torch.randn(batch_size, A, F).clamp(min=-1.5, max=1.5)

    with torch.no_grad():
        out = model(spatial)
    print("spatial:", spatial.shape, "out:", out.shape)
    print("pred sample:", out[0].tolist())