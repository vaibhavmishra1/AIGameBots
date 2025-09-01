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


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, D); mask: (B, N) True for PAD/invalid
        logits = (x * self.query).sum(dim=-1)  # (B, N)
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        # Handle all-masked: return zeros for those batches
        all_masked = mask is not None and mask.all(dim=-1)
        if all_masked.any():
            logits[all_masked] = 0.0  # or some null value
        # Sanitize logits: replace remaining -inf with large negative
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e9)
        attn = logits.softmax(dim=-1).unsqueeze(-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # just in case
        return (x * attn).sum(dim=1)  # (B, D)


class SpatialSetEncoder(nn.Module):
    """Ego-aware set encoder for spatial snapshot at a timestep.
    - Distinguishes agent 0 with a role embedding
    - Permutation-invariant to other agents via self-attention + attention pooling
    """

    def __init__(self, feature_dim: int = 6, d_model: int = 256, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.role_embedding = nn.Embedding(2, d_model)  # 0: other, 1: ego
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu")
        # MPS workaround: disable nested tensor path to avoid unsupported op
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pool = AttentionPool(d_model)
        self.fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, A, F)
        b, a, f = x.shape
        
        h = self.embed(x)  # (B, A, D)
        h = sanitize_tensor(h, "spat_embed")
        role_ids = torch.zeros(b, a, dtype=torch.long, device=x.device)
        role_ids[:, 0] = 1
        # print("SpatialSetEncoder- role_ids= ", role_ids.shape)
        # print("SpatialSetEncoder- role_ids= ", role_ids)
        h = h + self.role_embedding(role_ids)
        h = sanitize_tensor(h, "spat_role")
        # Mask rows that are all zeros
        pad_mask = (x.abs().sum(dim=-1) == 0)  # (B, A) True=pad
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = sanitize_tensor(h, "spat_encoder")
        ego = h[:, 0, :]  # (B, D)
        others = h[:, 1:, :]
        others_mask = pad_mask[:, 1:] if a > 1 else pad_mask  # (B, A-1)
        context = self.pool(others if a > 1 else h[:, 0:1, :], others_mask if a > 1 else pad_mask[:, 0:1])
        context = sanitize_tensor(context, "spat_pool")
        
        return self.fuse(torch.cat([ego, context], dim=-1))  # (B, D)


class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim: int = 6, d_model: int = 256, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1, max_time: int = 16):
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


class TemporalSpatialTransformer(nn.Module):
    def __init__(self, feature_dim: int = 6, d_model: int = 256, temp_layers: int = 2, temp_heads: int = 8, spat_layers: int = 2, spat_heads: int = 8, dropout: float = 0.1, max_time: int = 16, max_agents: int = 10):
        super().__init__()
        self.temporal = TemporalEncoder(feature_dim=feature_dim, d_model=d_model, nhead=temp_heads, num_layers=temp_layers, dropout=dropout, max_time=max_time)
        self.spatial = TemporalEncoder(feature_dim=feature_dim, d_model=d_model, nhead=spat_heads, num_layers=spat_layers, dropout=dropout, max_time=max_agents)
        self.fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 10))

    def forward(self, temporal: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        # temporal: (B, T, F) ; spatial: (B, A, F)
        ht = self.temporal(temporal)
        hs = self.spatial(spatial)
        h = self.fuse(torch.cat([ht, hs], dim=-1))
        out = self.head(h)  # (B,10)
        # scale: 5 pairs of (dx,dy) -> [-1,1] each
        return torch.tanh(out)


def build_model(
    feature_dim: int = 6,
    d_model: int = 256,
    temp_layers: int = 2,
    temp_heads: int = 8,
    spat_layers: int = 2,
    spat_heads: int = 8,
    dropout: float = 0.1,
    max_time: int = 16,
) -> TemporalSpatialTransformer:
    return TemporalSpatialTransformer(
        feature_dim=feature_dim,
        d_model=d_model,
        temp_layers=temp_layers,
        temp_heads=temp_heads,
        spat_layers=spat_layers,
        spat_heads=spat_heads,
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
    T = 16  # temporal length
    A = 10  # number of agents in spatial snapshot
    F = 6

    model = build_model(feature_dim=F, d_model=256, temp_layers=2, spat_layers=2, temp_heads=8, spat_heads=8, dropout=0.1)
    model.eval()

    # Random inputs roughly in [-1, 1]
    temporal = torch.randn(batch_size, T, F).clamp(min=-1.5, max=1.5)
    spatial = torch.randn(batch_size, A, F).clamp(min=-1.5, max=1.5)
    # Simulate zero-padded agents in spatial
    spatial[:, 5:, :] = 0.0

    with torch.no_grad():
        out = model(temporal, spatial)
    print("temporal:", temporal.shape, "spatial:", spatial.shape, "out:", out.shape)
    print("pred sample:", out[0].tolist())