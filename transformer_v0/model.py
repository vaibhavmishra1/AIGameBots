import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

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


class CausalSelfAttentionKV(nn.Module):
    """
    Multi-head self-attention with optional KV caching for streaming inference.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, H, L, Hd)
        B, L, D = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, L, Hd) -> (B, L, D)
        B, H, L, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Hd)

    def forward_full(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Full-sequence causal self-attention.
        x: (B, T, D); attn_mask: Optional (B, T, T) additive mask (log-space).
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scale = (self.head_dim) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        T = x.size(1)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal, float('-inf'))
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_scores = attn_scores + attn_mask.unsqueeze(1)
            else:
                attn_scores = attn_scores + attn_mask.view(1, 1, T, T)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, v)
        out = self._merge_heads(ctx)
        return self.out_proj(out)

    def forward_step(
        self,
        x_t: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        max_len: int,
        gating_log_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Streaming step with KV cache.
        x_t: (B, 1, D); kv_cache: (K_prev, V_prev) with (B, H, L, Hd)
        gating_log_mask: Optional (B, 1, L') additive mask.
        """
        q = self._split_heads(self.q_proj(x_t))  # (B, H, 1, Hd)
        k_t = self._split_heads(self.k_proj(x_t))  # (B, H, 1, Hd)
        v_t = self._split_heads(self.v_proj(x_t))  # (B, H, 1, Hd)
        if kv_cache is None:
            K = k_t
            V = v_t
        else:
            K_prev, V_prev = kv_cache
            K = torch.cat([K_prev, k_t], dim=2)
            V = torch.cat([V_prev, v_t], dim=2)
            if K.size(2) > max_len:
                K = K[:, :, -max_len:, :]
                V = V[:, :, -max_len:, :]
        scale = (self.head_dim) ** -0.5
        attn_scores = torch.matmul(q, K.transpose(-2, -1)) * scale  # (B, H, 1, L')
        if gating_log_mask is not None:
            attn_scores = attn_scores + gating_log_mask.unsqueeze(1)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, V)  # (B, H, 1, Hd)
        out = self._merge_heads(ctx)  # (B, 1, D)
        out = self.out_proj(out)
        return out, (K, V)

class CausalTransformerLayer(nn.Module):
    """Transformer encoder layer with causal self-attention and optional KV cache."""
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.self_attn = CausalSelfAttentionKV(d_model=d_model, nhead=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, int(d_model * mlp_ratio))
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(int(d_model * mlp_ratio), d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def forward_full(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Pre-norm
        x = x + self.dropout1(self.self_attn.forward_full(self.norm1(x), attn_mask))
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
        return x

    def forward_step(
        self,
        x_t: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        max_len: int,
        gating_log_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x_t: (B, 1, D)
        h = self.norm1(x_t)
        y_t, new_cache = self.self_attn.forward_step(h, kv_cache, max_len, gating_log_mask=gating_log_mask)
        x_t = x_t + self.dropout1(y_t)
        h2 = self.norm2(x_t)
        ff = self.linear2(self.dropout(self.activation(self.linear1(h2))))
        x_t = x_t + self.dropout2(ff)
        return x_t, new_cache


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


class EfficientNetFeatureExtractor(nn.Module):
    """
    Lightweight spatial feature extractor using EfficientNet-B0.
    Returns global pooled features per frame.
    """

    def __init__(self):
        super().__init__()
        try:
            self.net = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            print("Warning: Could not load pretrained EfficientNet-B0 weights, using random initialization")
            self.net = models.efficientnet_b0(weights=None)

        # Expected input size (kept at 224 to match ImageNet pretraining)
        self.expected_size = 224
        # Output embedding dimension after global pooling
        self.output_dim = 1280

        # Register ImageNet normalization buffers (not trainable)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W) - batch of frames in [0,1]
        Returns:
            features: (B*T, 1280) - global pooled EfficientNet features
        """
        if x.shape[-1] != self.expected_size or x.shape[-2] != self.expected_size:
            x = F.interpolate(x, size=(self.expected_size, self.expected_size), mode='bicubic', align_corners=False)

        # Normalize to ImageNet stats
        mean = self.imagenet_mean.to(dtype=x.dtype, device=x.device)
        std = self.imagenet_std.to(dtype=x.dtype, device=x.device)
        x = (x - mean) / std

        # Forward through EfficientNet features + global pooling
        feats = self.net.features(x)                 # (B*T, C, H', W')
        pooled = self.net.avgpool(feats)            # (B*T, C, 1, 1)
        vec = torch.flatten(pooled, 1)              # (B*T, C)
        return vec


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
      - Use pretrained EfficientNet-B0 to extract features from each frame
      - Temporal Transformer encoder over sequence of frame features
      - Shared MLP and per-timestep heads (keys, clicks, mouse_x, mouse_y, value)
    """

    def __init__(
        self,
        model_name: str = 'vivit_vitb16',
        temporal_depth: int = 8,
        temporal_d_model: int = 512,
        spatial_input_size: int = 192,
        aux_input_length: int = 17,
        aux_input_on: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.temporal_depth = temporal_depth
        self.aux_input_length = aux_input_length
        self.aux_input_on = aux_input_on
        self.freeze_backbone = freeze_backbone

        # Pretrained EfficientNet-B0 for spatial feature extraction (1280-dim features)
        self.spatial_encoder = EfficientNetFeatureExtractor()
        # Reduce input resize for faster inference if specified
        try:
            if isinstance(spatial_input_size, int) and spatial_input_size > 0:
                self.spatial_encoder.expected_size = int(spatial_input_size)
        except Exception:
            pass
        self.spatial_dim = getattr(self.spatial_encoder, 'output_dim', 1280)
        self.temporal_d_model = temporal_d_model

        if self.freeze_backbone:
            for param in self.spatial_encoder.parameters():
                param.requires_grad = False

        # Project high-dim spatial features to a smaller temporal model dimension
        self.spatial_proj = nn.Linear(self.spatial_dim, self.temporal_d_model)

        # Choose a valid number of heads for MultiheadAttention given spatial_dim
        def _pick_nhead(d: int) -> int:
            for h in (16, 10, 8, 5, 4, 2, 1):
                if d % h == 0:
                    return h
            for h in range(1, 33):
                if d % h == 0:
                    return h
            return 8
        nhead = _pick_nhead(self.temporal_d_model)

        # Temporal transformer with causal attention stack (supports KV cache)
        self.temporal_layers = nn.ModuleList([
            CausalTransformerLayer(
                d_model=self.temporal_d_model,
                nhead=nhead,
                mlp_ratio=2.0,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(temporal_depth)
        ])
        self.temporal_final_norm = nn.LayerNorm(self.temporal_d_model)
        # Normalize temporal inputs (features + positional encoding) before the transformer
        self.temporal_input_norm = nn.LayerNorm(self.temporal_d_model)

        # No dynamic-k predictor (removed)

        # Heads and shared layers
        if self.aux_input_on:
            self.aux_dense = nn.Linear(aux_input_length, 256)
            shared_in = self.temporal_d_model + 256  # temporal D + aux 256
        else:
            self.aux_dense = None
            shared_in = self.temporal_d_model

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

        # Spatial encoding: extract features from each frame using EfficientNet
        x_btchw = x.view(b * t, 3, h, w)  # (B*T, 3, H, W)
        frame_features = self.spatial_encoder(x_btchw)  # (B*T, spatial_dim)
        frame_seq_spatial = frame_features.view(b, t, self.spatial_dim)  # (B, T, spatial_dim)
        frame_seq = self.spatial_proj(frame_seq_spatial)  # (B, T, temporal_d_model)

        # Temporal encoding with positional embeddings
        pos_t = _build_sincos_1d_position_embedding(t, self.temporal_d_model, frame_seq.device)
        temporal_in = self.temporal_input_norm(frame_seq + pos_t)  # (B, T, D)

        if getattr(self, '_stateful', False) and t == 1:
            # Streaming path with KV cache
            if not hasattr(self, '_kv_caches') or self._kv_caches is None:
                self._kv_caches = [None for _ in range(len(self.temporal_layers))]
                self._max_seq_len = input_shape[0]
            x_step = temporal_in  # (B, 1, D)
            new_caches = []
            for li, layer in enumerate(self.temporal_layers):
                cache_li = self._kv_caches[li]
                x_step, cache_new = layer.forward_step(x_step, cache_li, max_len=self._max_seq_len, gating_log_mask=None)
                new_caches.append(cache_new)
            self._kv_caches = new_caches
            temporal_out = self.temporal_final_norm(x_step)  # (B, 1, D)
        else:
            x_seq = temporal_in
            for layer in self.temporal_layers:
                x_seq = layer.forward_full(x_seq, attn_mask=None)
            temporal_out = self.temporal_final_norm(x_seq)  # (B, T, D)

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
            x_features = torch.cat([temporal_out, aux_features], dim=-1)  # (B, T, D+256)
        else:
            x_features = temporal_out  # (B, T, D)

        # Shared dense and heads
        shared = self.shared_dense(x_features)  # (B, T, 256)

        keys_out = self.sigmoid(self.keys_output(shared))
        clicks_out = self.sigmoid(self.clicks_output(shared))
        mouse_x_out = self.softmax(self.mouse_x_output(shared))
        mouse_y_out = self.softmax(self.mouse_y_output(shared))
        value_out = self.value_output(shared)

        return keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out

    def get_output_concatenated(self, x: torch.Tensor, aux_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = self.forward(x, aux_input)
        return torch.cat([keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out], dim=-1)

    # Stateful streaming API
    def set_stateful(self, enabled: bool = True) -> None:
        self._stateful = bool(enabled)
        if not self._stateful:
            self.reset_states()

    def reset_states(self) -> None:
        self._kv_caches = None

def create_vivit_model(
    model_name: str = 'vivit_vitb16',
    aux_input_on: bool = True,
    temporal_depth: int = 8,
    temporal_d_model: int = 512,
    spatial_input_size: int = 192,
    freeze_backbone: bool = False,
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
        temporal_d_model=temporal_d_model,
        spatial_input_size=spatial_input_size,
        aux_input_length=aux_len,
        aux_input_on=aux_input_on,
        freeze_backbone=freeze_backbone,
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

