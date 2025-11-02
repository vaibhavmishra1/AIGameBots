import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights

# Config values (from config.py)
input_shape = (96, 150, 280, 3)  # (timesteps, height, width, channels)
n_keys = 11      # keyboard outputs
n_clicks = 2     # mouse buttons
n_mouse_x = 23   # mouse x positions
n_mouse_y = 15   # mouse y positions


class TimeDistributed(nn.Module):
    """
    Applies a module to each timestep of a sequence.
    Input shape: (batch, timesteps, ...)
    Output shape follows underlying module per timestep.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps = x.size(0), x.size(1)
        feature_dims = x.size()[2:]
        x = x.contiguous().view(batch_size * timesteps, *feature_dims)
        y = self.module(x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("TimeDistributed module must return a Tensor")
        y = y.contiguous()
        out_dims = y.size()[1:]
        y = y.view(batch_size, timesteps, *out_dims)
        return y


def _build_sincos_1d_position_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Create 1D sine-cosine positional embeddings of shape (1, length, dim).
    """
    assert dim % 2 == 0, "Embedding dim must be divisible by 2 for 1D sincos"
    positions = torch.arange(length, device=device).float().unsqueeze(1)  # (L, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0)  # (1, L, D)


class ViTFeatureExtractor(nn.Module):
    """
    Wrapper around torchvision ViT to extract features instead of classification.
    Uses the [CLS] token output from the encoder as frame-level features.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained ViT-B/16 (or None for no weights during testing)
        try:
            self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        except Exception:
            # Fallback to random weights if download fails
            print("Warning: Could not load pretrained ViT weights, using random initialization")
            self.vit = models.vit_b_16(weights=None)

        # Remove the classification head to get features
        self.vit.heads = nn.Identity()

        # ViT expects (B, C, H, W) with H=W=224
        self.expected_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W) - batch of frames
        Returns:
            features: (B*T, 768) - [CLS] token features from ViT
        """
        # Resize to expected input size (224x224)
        if x.shape[-1] != self.expected_size or x.shape[-2] != self.expected_size:
            x = F.interpolate(x, size=(self.expected_size, self.expected_size), mode='bicubic', align_corners=False)

        # Forward through ViT (output is [CLS] token: (B*T, 768))
        return self.vit(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer supporting per-sample 3D attention masks."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # attn_mask can be (B, T, T) float additive mask, or (T, T)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Broadcast across batch and heads is handled by MHA for 2D masks
                expanded_mask = attn_mask
            else:
                # Expand to (B * num_heads, T, T)
                num_heads = self.self_attn.num_heads
                B, T, _ = attn_mask.shape
                expanded_mask = attn_mask.repeat_interleave(num_heads, dim=0)
        else:
            expanded_mask = None

        x_attn, _ = self.self_attn(x, x, x, attn_mask=expanded_mask, need_weights=False)
        return x_attn

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Pre-norm (norm_first=True)
        x = x + self.dropout1(self._sa_block(self.norm1(x), attn_mask))
        x = x + self.dropout2(self._ff_block(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of encoder layers with final LayerNorm; supports per-sample masks."""

    def __init__(self, d_model: int, nhead: int, depth: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                activation='gelu',
            ) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class ViViTCSGOModel(nn.Module):
    """
    Simplified ViViT model using pretrained PyTorch ViT for spatial encoding.

    Pipeline:
      - Use pretrained ViT-B/16 to extract features from each frame
      - Temporal Transformer encoder over sequence of frame features
      - Shared MLP and per-timestep heads (keys, clicks, mouse_x, mouse_y, value)
    """

    def __init__(
        self,
        model_name: str = 'vivit_vitb16',
        temporal_depth: int = 8,
        aux_input_length: int = 17,
        aux_input_on: bool = True,
        freeze_backbone: bool = False,
        use_dynamic_k: bool = True,
        k_tau: float = 0.5,
    ):
        super().__init__()

        self.model_name = model_name
        self.temporal_depth = temporal_depth
        self.aux_input_length = aux_input_length
        self.aux_input_on = aux_input_on
        self.freeze_backbone = freeze_backbone
        self.use_dynamic_k = use_dynamic_k
        self.k_tau = k_tau

        # Pretrained ViT-B/16 for spatial feature extraction (768-dim features)
        self.spatial_encoder = ViTFeatureExtractor()

        if self.freeze_backbone:
            for param in self.spatial_encoder.parameters():
                param.requires_grad = False

        # Temporal transformer (768-dim input from ViT)
        self.temporal_encoder = TransformerEncoder(
            d_model=768, nhead=12, depth=temporal_depth, mlp_ratio=4.0, dropout=0.1
        )
        # Normalize temporal inputs (features + positional encoding) before the transformer
        self.temporal_input_norm = nn.LayerNorm(768)

        # Per-timestep dynamic-k predictor from current-frame spatial feature
        self.k_proj = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Heads and shared layers
        if self.aux_input_on:
            self.aux_dense = nn.Linear(aux_input_length, 256)
            shared_in = 768 + 256  # temporal D (768) + aux 256
        else:
            self.aux_dense = None
            shared_in = 768

        self.shared_dense = nn.Linear(shared_in, 256)

        # Per-timestep heads
        self.keys_output = TimeDistributed(nn.Linear(256, n_keys))
        self.clicks_output = TimeDistributed(nn.Linear(256, n_clicks))
        self.mouse_x_output = TimeDistributed(nn.Linear(256, n_mouse_x))
        self.mouse_y_output = TimeDistributed(nn.Linear(256, n_mouse_y))
        self.value_output = TimeDistributed(nn.Linear(256, 1))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights properly for transformers
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization for better convergence.
        Preserves pretrained ViT weights and only initializes newly added layers."""
        for name, module in self.named_modules():
            # Skip ViT spatial encoder to preserve pretrained weights
            if 'spatial_encoder' in name:
                continue

            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                # Standard initialization for layer norm
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, aux_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: (batch, timesteps, height, width, channels)
            aux_input: Optional [(batch, timesteps, aux_dim) or (batch, aux_dim)] previous action vectors

        Returns:
            Tuple of (keys, clicks, mouse_x, mouse_y, value)
            where each is (batch, timesteps, ...)
        """
        b, t, h, w, c = x.shape

        # Convert to (B, T, C, H, W)
        if c == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, 3, H, W)

        # Spatial encoding: extract features from each frame using ViT
        x_btchw = x.view(b * t, 3, h, w)  # (B*T, 3, H, W)
        frame_features = self.spatial_encoder(x_btchw)  # (B*T, 768)
        frame_seq = frame_features.view(b, t, 768)  # (B, T, 768)

        # Predict per-timestep dynamic lookback proportion p in [0,1] from current frame features
        # k_t (frames) = 1 + p_t * (T - 1)
        p = torch.sigmoid(self.k_proj(frame_seq)).squeeze(-1)  # (B, T)

        # Temporal encoding with positional embeddings and dynamic-k gating mask
        pos_t = _build_sincos_1d_position_embedding(t, 768, frame_seq.device)
        temporal_in = self.temporal_input_norm(frame_seq + pos_t)  # (B, T, 768)

        attn_mask = None
        if self.use_dynamic_k and t > 0:
            # Per-sample gates: G[b, t, s] ~ 1 if (t - s) <= k_{b,t} else ~0 (causal)
            k_frames = 1.0 + p * float(t - 1)  # (B, T)
            idx = torch.arange(t, device=frame_seq.device)
            dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0).float()  # (T, T)
            gates = torch.sigmoid((k_frames.unsqueeze(-1) - dist.unsqueeze(0)) / max(self.k_tau, 1e-6))  # (B, T, T)
            tril = torch.tril(torch.ones((t, t), device=frame_seq.device))
            gates = gates * tril  # zero out future positions
            attn_mask = torch.log(gates.clamp(min=1e-6))  # (B, T, T)

            self.last_k_penalty = p.mean()
        else:
            self.last_k_penalty = torch.zeros((), device=frame_seq.device)

        temporal_out = self.temporal_encoder(temporal_in, mask=attn_mask)  # (B, T, 768)

        # Handle auxiliary input
        if self.aux_input_on and aux_input is not None:
            if aux_input.dim() == 2:
                aux_features = self.aux_dense(aux_input)  # (B, 256)
                aux_features = aux_features.unsqueeze(1).repeat(1, t, 1)  # (B, T, 256)
            elif aux_input.dim() == 3:
                b2, t2, d2 = aux_input.size()
                aux_flat = aux_input.contiguous().view(b2 * t2, d2)
                aux_proj = self.aux_dense(aux_flat)
                aux_features = aux_proj.view(b2, t2, 256)
            else:
                raise ValueError("aux_input must have shape (batch, aux_dim) or (batch, timesteps, aux_dim)")
            x_features = torch.cat([temporal_out, aux_features], dim=-1)  # (B, T, 768+256)
        else:
            x_features = temporal_out  # (B, T, 768)

        # Shared dense and heads
        shared = self.shared_dense(x_features)  # (B, T, 256)

        keys_out = self.sigmoid(self.keys_output(shared))
        clicks_out = self.sigmoid(self.clicks_output(shared))
        mouse_x_out = self.softmax(self.mouse_x_output(shared))
        mouse_y_out = self.softmax(self.mouse_y_output(shared))
        value_out = self.value_output(shared)

        return keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out

    def get_k_penalty(self) -> torch.Tensor:
        return getattr(self, 'last_k_penalty', torch.zeros((), device=next(self.parameters()).device))

    def get_output_concatenated(self, x: torch.Tensor, aux_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = self.forward(x, aux_input)
        return torch.cat([keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out], dim=-1)


def create_vivit_model(
    model_name: str = 'vivit_vitb16',
    aux_input_on: bool = True,
    temporal_depth: int = 8,
    freeze_backbone: bool = False,
    use_dynamic_k: bool = True,
    k_tau: float = 0.5,
) -> ViViTCSGOModel:
    """
    Factory to create ViViTCSGOModel with pretrained ViT-B/16 spatial encoder.
    """
    # Determine aux_input_length: if aux is enabled, use previous action dimension
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    aux_len = action_dim if aux_input_on else 17  # fallback to config default length

    model = ViViTCSGOModel(
        model_name=model_name,
        temporal_depth=temporal_depth,
        aux_input_length=aux_len,
        aux_input_on=aux_input_on,
        freeze_backbone=freeze_backbone,
        use_dynamic_k=use_dynamic_k,
        k_tau=k_tau,
    )

    return model


if __name__ == "__main__":
    # Quick shape test
    torch.manual_seed(0)
    model = create_vivit_model(aux_input_on=True)
    b, t, h, w, c = 1, input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    x = torch.randn(b, t, h, w, c)

    # Aux inputs: previous actions per timestep
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    aux = torch.randn(b, t, action_dim)

    y = model.get_output_concatenated(x, aux)
    print("✓ Forward pass successful, output shape:", tuple(y.shape))

