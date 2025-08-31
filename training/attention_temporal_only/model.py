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
    def __init__(self, feature_dim: int = 13, d_model: int = 256, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1, max_time: int = 20):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.pos = PositionalEncoding(d_model, max_len=max_time)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu")
        # MPS workaround: disable nested tensor path to avoid unsupported op
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = sanitize_tensor(h, "temp_embed")
        h = self.pos(h)
        h = sanitize_tensor(h, "temp_pos")
        # Bidirectional self-attention over all timesteps (no causal mask)
        h = self.encoder(h)
        h = sanitize_tensor(h, "temp_encoder")
        return h[:, -1, :]  # (B, D)


class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim: int = 5, d_model: int = 256, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1, max_time: int = 20):
        super().__init__()
        self.temporal = TemporalEncoder(feature_dim=feature_dim, d_model=d_model, nhead=num_heads, num_layers=num_layers, dropout=dropout, max_time=max_time)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        # temporal: (B, T, F)
        h = self.temporal(temporal)
        out = self.head(h)  # (B,2)
        # scale: dx,dy -> [-1,1] 
        scales = torch.tensor([1.0, 1.0], device=out.device, dtype=out.dtype).unsqueeze(0)
        return torch.tanh(out) * scales


def build_model(
    feature_dim: int = 5,
    d_model: int = 256,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    max_time: int = 20,
) -> TemporalTransformer:
    return TemporalTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_time=max_time,
    )


def sanitize_tensor(x: torch.Tensor, name: str = "") -> torch.Tensor:
    if not torch.isfinite(x).all():
        num_nan = torch.isnan(x).sum().item()
        num_inf = torch.isinf(x).sum().item()
        print(f"Warning: non-finite in {name}: nan={num_nan} inf={num_inf}. Replacing with 0.")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


if __name__ == "__main__":
    # Simple smoke test with random inputs
    torch.manual_seed(0)
    batch_size = 4
    T = 20  # temporal length
    F = 5  # feature length

    model = build_model(feature_dim=F, d_model=256, num_layers=2, num_heads=8, dropout=0.1, max_time=20)
    model.eval()

    # Random inputs roughly in [-1, 1]
    temporal = torch.randn(batch_size, T, F).clamp(min=-1.5, max=1.5)

    with torch.no_grad():
        out = model(temporal)
    print("temporal:", temporal.shape, "out:", out.shape)
    print("pred sample:", out[0].tolist())