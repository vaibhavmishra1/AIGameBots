import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from typing import Optional, Tuple, Dict, Any

# Optional timm import for student backbones
try:
    import timm  # type: ignore
    _has_timm = True
except Exception:
    _has_timm = False
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

        # ViT expects (B, C, H, W) with H=W=384 for SWAG weights
        self.expected_size = 224

        # Normalization params from PyTorch docs
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W) - batch of frames (already preprocessed: resized, cropped, scaled [0,1], normalized)
        Returns:
            features: (B*T, 768) - [CLS] token features from ViT
        """
        # Forward through ViT (output is [CLS] token: (B*T, 768))
        return self.vit(x)


class DeiTFeatureExtractor(nn.Module):
    """
    Wrapper around timm DeiT-Tiny to extract [CLS] features (embed_dim=192).
    Falls back to random init if pretrained weights unavailable.
    """
    def __init__(self):
        super().__init__()
        if not _has_timm:
            raise RuntimeError("timm not available for DeiTFeatureExtractor")
        try:
            # num_classes=0 makes forward() return features for many timm models
            self.deit = timm.create_model(
                'deit_tiny_patch16_224',
                pretrained=True,
                num_classes=0
            )
        except Exception:
            self.deit = timm.create_model(
                'deit_tiny_patch16_224',
                pretrained=False,
                num_classes=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.deit(x)
        # timm usually returns (B, D) when num_classes=0; handle (B, tokens, D) just in case
        if feats.dim() == 3:
            return feats[:, 0, :]
        return feats


class MobileNetSmallFeatureExtractor(nn.Module):
    """
    Wrapper around torchvision MobileNetV3-Small to produce a compact feature vector (dim=192).
    """
    def __init__(self, out_dim: int = 192):
        super().__init__()
        try:
            backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features  # convolutional body
        # Final conv channels of MobileNetV3-Small are typically 576
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(576, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # (B, C, H, W)
        x = self.pool(x)      # (B, C, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)
        return self.proj(x)   # (B, out_dim)


class TransformerEncoder(nn.Module):
    """Wrapper around nn.TransformerEncoder with batch_first=True."""

    def __init__(self, d_model: int, nhead: int, depth: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        # Apply a final LayerNorm after the stack for stability
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


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
    ):
        super().__init__()

        self.model_name = model_name
        self.temporal_depth = temporal_depth
        self.aux_input_length = aux_input_length
        self.aux_input_on = aux_input_on
        self.freeze_backbone = freeze_backbone

        # Select spatial encoder and set feature dimension
        self.spatial_dim = 768
        if self.model_name in ('vivit_vitb16', 'vit_b_16', 'vitb16'):
            # Pretrained ViT-B/16 for spatial feature extraction (768-dim features)
            self.spatial_encoder = ViTFeatureExtractor()
            self.spatial_dim = 768
        elif self.model_name in ('deit_tiny', 'vivit_deit_tiny'):
            if not _has_timm:
                # Fallback to MobileNet if timm unavailable
                self.spatial_encoder = MobileNetSmallFeatureExtractor(out_dim=192)
            else:
                self.spatial_encoder = DeiTFeatureExtractor()
            self.spatial_dim = 192
        elif self.model_name in ('mobilenet_small', 'vivit_mobilenet_small'):
            self.spatial_encoder = MobileNetSmallFeatureExtractor(out_dim=192)
            self.spatial_dim = 192
        else:
            # Default to ViT-B/16
            self.spatial_encoder = ViTFeatureExtractor()
            self.spatial_dim = 768

        if self.freeze_backbone:
            for param in self.spatial_encoder.parameters():
                param.requires_grad = False

        # Temporal transformer (d_model matches spatial feature dimension)
        # Choose attention heads based on width
        nhead = 12 if self.spatial_dim >= 768 else max(1, self.spatial_dim // 64)
        self.temporal_encoder = TransformerEncoder(
            d_model=self.spatial_dim, nhead=nhead, depth=temporal_depth, mlp_ratio=4.0, dropout=0.1
        )
        # Normalize temporal inputs (features + positional encoding) before the transformer
        self.temporal_input_norm = nn.LayerNorm(self.spatial_dim)

        # Heads and shared layers
        if self.aux_input_on:
            self.aux_dense = nn.Linear(aux_input_length, 256)
            shared_in = self.spatial_dim + 256  # temporal D + aux 256
        else:
            self.aux_dense = None
            shared_in = self.spatial_dim

        self.shared_dense = nn.Linear(shared_in, 256)

        # Per-timestep heads
        self.keys_output = TimeDistributed(nn.Linear(256, n_keys))
        self.clicks_output = TimeDistributed(nn.Linear(256, n_clicks))
        self.mouse_x_output = TimeDistributed(nn.Linear(256, n_mouse_x))
        self.mouse_y_output = TimeDistributed(nn.Linear(256, n_mouse_y))
        self.value_output = TimeDistributed(nn.Linear(256, 1))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # Projections for KD (student-to-teacher 192/other -> 768)
        # Present for all models for simplicity; no overhead if unused
        self.proj_s2t_frame = nn.Linear(self.spatial_dim, 768) if self.spatial_dim != 768 else nn.Identity()
        self.proj_s2t_temp = nn.Linear(self.spatial_dim, 768) if self.spatial_dim != 768 else nn.Identity()

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

    def forward(
        self,
        x: torch.Tensor,
        aux_input: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: (batch, timesteps, channels, height, width) - already preprocessed
            aux_input: Optional [(batch, timesteps, aux_dim) or (batch, aux_dim)] previous action vectors
            return_features: If True, also return intermediate features for KD

        Returns:
            Tuple of (keys, clicks, mouse_x, mouse_y, value)
            where each is (batch, timesteps, ...)
        """
        b, t, c, h, w = x.shape

        # Spatial encoding: extract features from each frame using ViT
        x_btchw = x.view(b * t, c, h, w)  # (B*T, C, H, W)
        frame_features = self.spatial_encoder(x_btchw)  # (B*T, D)
        frame_seq = frame_features.view(b, t, self.spatial_dim)  # (B, T, D)

        # Temporal encoding with positional embeddings
        pos_t = _build_sincos_1d_position_embedding(t, self.spatial_dim, frame_seq.device)
        temporal_in = self.temporal_input_norm(frame_seq + pos_t)  # (B, T, D)
        temporal_out = self.temporal_encoder(temporal_in)  # (B, T, D)

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

        if return_features:
            features: Dict[str, torch.Tensor] = {
                'frame_seq': frame_seq,          # (B, T, D_student_or_teacher)
                'temporal_out': temporal_out,    # (B, T, D_student_or_teacher)
            }
            return keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out, features
        else:
            return keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out

    def get_output_concatenated(self, x: torch.Tensor, aux_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        outs = self.forward(x, aux_input)
        if isinstance(outs, tuple) and len(outs) == 6:
            keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out, _ = outs
        else:
            keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = outs
        return torch.cat([keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out], dim=-1)


def create_vivit_model(
    model_name: str = 'vivit_vitb16',
    aux_input_on: bool = True,
    temporal_depth: int = 8,
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

