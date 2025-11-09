import os
import sys
from typing import Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the stateful temporal transformer with KV cache
from model_mobilevit import StatefulTemporalTransformer

# Try to import timm for pretrained models
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not installed. Install with: pip install timm")

# Config values (from config.py)
input_shape = (96, 150, 280, 3)  # (timesteps, height, width, channels)
n_keys = 11      # keyboard outputs
n_clicks = 2     # mouse buttons
n_mouse_x = 23   # mouse x positions
n_mouse_y = 15   # mouse y positions


class PretrainedMobileViTEncoder(nn.Module):
    """
    Uses actual pretrained MobileViT from timm library.
    Extracts spatial features while preserving spatial dimensions.
    """
    
    def __init__(self, model_name='mobilevit_s', freeze_early_layers=False):
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm library required. Install with: pip install timm")
        
        # Load pretrained MobileViT
        # Options: mobilevit_s (~5.6M params), mobilevit_xs (~2.3M params), mobilevit_xxs (~1.3M params)
        self.base_model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,  # Extract features, not classification
            out_indices=(2, 3, 4),  # Get multiple feature scales
        )
        
        # Get feature dimensions from the model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            features = self.base_model(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
            self.feature_scales = [f.shape[2] for f in features]  # Spatial dimensions
        
        print(f"MobileViT feature dims: {self.feature_dims} at scales {self.feature_scales}")
        
        # Feature pyramid fusion
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(dim, 256, 1) for dim in self.feature_dims
        ])
        
        # Final projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Optionally freeze early layers for faster training
        if freeze_early_layers:
            # Freeze first 60% of the network
            total_params = len(list(self.base_model.parameters()))
            freeze_until = int(total_params * 0.6)
            for i, param in enumerate(self.base_model.parameters()):
                if i < freeze_until:
                    param.requires_grad = False
            print(f"Froze {freeze_until}/{total_params} early MobileViT parameters")
        
        self.output_channels = 256
        
    def forward(self, x):
        """
        Args:
            x: (B*T, 3, H, W) - batch of frames
        Returns:
            features: (B*T, 256, H', W') - spatial features
        """
        B_T, C, H, W = x.shape
        
        # Pad to square preserving aspect ratio
        target_size = 256
        aspect = W / H
        if aspect > 1:  # Wider than tall
            new_w = target_size
            new_h = int(target_size / aspect)
        else:
            new_h = target_size
            new_w = int(target_size * aspect)
        
        # Resize to new_h x new_w
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Pad to target_size x target_size
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Extract multi-scale features
        features = self.base_model(x)
        
        # Feature pyramid network - combine multi-scale features
        # Upsample all to the same spatial size (use middle scale)
        target_size = features[1].shape[2:]  # Middle scale spatial size
        
        fpn_features = []
        for i, (feat, conv) in enumerate(zip(features, self.fpn_convs)):
            feat_proj = conv(feat)
            if feat_proj.shape[2:] != target_size:
                feat_proj = F.interpolate(feat_proj, size=target_size, mode='bilinear', align_corners=False)
            fpn_features.append(feat_proj)
        
        # Combine features (could use attention here, but mean is simple and effective)
        combined = torch.stack(fpn_features, dim=0).mean(dim=0)
        
        # Final processing
        output = self.output_conv(combined)
        
        return output


class MobileNetV3Alternative(nn.Module):
    """
    Alternative: Use MobileNetV3 if MobileViT is not available.
    MobileNetV3 is more widely available and well-tested.
    """
    
    def __init__(self, freeze_early_layers=False):
        super().__init__()
        
        # Use torchvision's pretrained MobileNetV3
        from torchvision import models
        
        # Load pretrained MobileNetV3-Small (lighter and faster)
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        
        # Extract feature layers (before classifier)
        # We'll use features from multiple stages
        features = mobilenet.features
        
        # Split into stages for multi-scale extraction
        self.stage1 = nn.Sequential(*features[:4])   # 1/4 scale
        self.stage2 = nn.Sequential(*features[4:9])  # 1/8 scale  
        self.stage3 = nn.Sequential(*features[9:])   # 1/16 scale
        
        # Get output dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            s1 = self.stage1(dummy)
            s2 = self.stage2(s1)
            s3 = self.stage3(s2)
            
        # Feature adaptation (just final stage for now)
        self.adapt3 = nn.Conv2d(s3.shape[1], 256, 1)
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        if freeze_early_layers:
            # Freeze stage1 and part of stage2
            for param in self.stage1.parameters():
                param.requires_grad = False
            for param in list(self.stage2.parameters())[:len(list(self.stage2.parameters()))//2]:
                param.requires_grad = False
            print("Froze early MobileNetV3 layers")
        
        self.output_channels = 256
        
    def forward(self, x):
        """
        Args:
            x: (B*T, 3, H, W) - batch of frames
        Returns:
            features: (B*T, 256, H', W') - spatial features
        """
        # Resize to MobileNet expected size
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Multi-scale features
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        
        # Adapt channels and combine
        # We'll use s3 as main features and add context from s1, s2
        feat3 = self.adapt3(s3)
        
        # Use feat3 as main features (already 256 channels)
        # Skip multi-scale fusion for simplicity
        combined = feat3
        
        output = self.output_conv(combined)
        return output


class MobileViTTemporalModelPretrained(nn.Module):
    """
    Complete model with pretrained spatial encoder + Temporal Transformer.
    """
    
    def __init__(
        self,
        spatial_encoder='mobilevit_s',  # or 'mobilenet_v3'
        freeze_early_layers=False,
        temporal_layers=6,
        temporal_heads=8,
        temporal_dim=256,
        max_seq_len=96,
        aux_input_length=17,
        aux_input_on=True,
        dropout=0.1,
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        # Choose spatial encoder
        if spatial_encoder.startswith('mobilevit'):
            try:
                self.spatial_encoder = PretrainedMobileViTEncoder(
                    model_name=spatial_encoder,
                    freeze_early_layers=freeze_early_layers
                )
                print(f"Using pretrained {spatial_encoder}")
            except:
                print(f"Failed to load {spatial_encoder}, falling back to MobileNetV3")
                self.spatial_encoder = MobileNetV3Alternative(
                    freeze_early_layers=freeze_early_layers
                )
        else:
            self.spatial_encoder = MobileNetV3Alternative(
                freeze_early_layers=freeze_early_layers
            )
            print("Using pretrained MobileNetV3-Small")
        
        spatial_channels = self.spatial_encoder.output_channels
        
        # Spatial pooling with attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(spatial_channels, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Project to temporal dimension
        self.spatial_proj = nn.Linear(spatial_channels, temporal_dim)
        
        # Temporal transformer with KV caching for stateful inference
        self.temporal_transformer = StatefulTemporalTransformer(
            d_model=temporal_dim,
            nhead=temporal_heads,
            num_layers=temporal_layers,
            max_seq_len=max_seq_len,
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, temporal_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Auxiliary input
        self.aux_input_on = aux_input_on
        if aux_input_on:
            self.aux_dense = nn.Sequential(
                nn.Linear(aux_input_length, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 256)
            )
            feature_dim = temporal_dim + 256
        else:
            feature_dim = temporal_dim
        
        # Output heads
        self.shared_dense = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        
        self.keys_output = nn.Linear(256, n_keys)
        self.clicks_output = nn.Linear(256, n_clicks)
        self.mouse_x_output = nn.Linear(256, n_mouse_x)
        self.mouse_y_output = nn.Linear(256, n_mouse_y)
        self.value_output = nn.Linear(256, 1)
        
    def forward(self, x, aux_input=None, stateful: bool = False):
        """
        Args:
            x: (B, T, H, W, C) - video frames
            aux_input: Optional (B, T, aux_dim) - auxiliary features
            stateful: If True, use KV caching for streaming inference
        Returns:
            Tuple of (keys, clicks, mouse_x, mouse_y, value) predictions
        """
        B, T, H, W, C = x.shape
        
        # Rearrange for spatial encoding
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Extract spatial features with pretrained encoder
        spatial_features = self.spatial_encoder(x)  # (B*T, C, H', W')
        
        # Attention pooling
        B_T, C_s, H_s, W_s = spatial_features.shape
        attn_weights = self.spatial_attention(spatial_features)  # (B*T, 1, H', W')
        attn_weights = attn_weights.view(B_T, 1, -1)  # (B*T, 1, H'*W')
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        spatial_flat = spatial_features.view(B_T, C_s, -1)  # (B*T, C, H'*W')
        pooled = (spatial_flat * attn_weights).sum(dim=-1)  # (B*T, C)
        
        # Project and reshape for temporal
        temporal_input = self.spatial_proj(pooled)  # (B*T, temporal_dim)
        temporal_input = temporal_input.view(B, T, -1)  # (B, T, temporal_dim)
        
        # Add positional encoding
        temporal_input = temporal_input + self.pos_encoding[:, :T]
        
        # Temporal transformer (stateful enables KV caching)
        temporal_features = self.temporal_transformer(
            temporal_input,
            use_cache=bool(stateful),
        )
        
        # Add auxiliary input
        if self.aux_input_on and aux_input is not None:
            if aux_input.dim() == 2:
                aux_features = self.aux_dense(aux_input).unsqueeze(1).expand(-1, T, -1)
            else:
                B2, T2, D2 = aux_input.shape
                aux_flat = aux_input.view(B2 * T2, D2)
                aux_features = self.aux_dense(aux_flat).view(B2, T2, 256)
            features = torch.cat([temporal_features, aux_features], dim=-1)
        else:
            features = temporal_features
        
        # Output heads
        shared = self.shared_dense(features)
        
        keys_out = torch.sigmoid(self.keys_output(shared))
        clicks_out = torch.sigmoid(self.clicks_output(shared))
        mouse_x_out = F.softmax(self.mouse_x_output(shared), dim=-1)
        mouse_y_out = F.softmax(self.mouse_y_output(shared), dim=-1)
        value_out = self.value_output(shared)
        
        return keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out
    
    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    # Stateful streaming control
    def set_stateful(self, enabled: bool = True) -> None:
        if not enabled:
            self.reset_states()
        # No flag needed here; temporal_transformer handles cache internally

    def reset_states(self) -> None:
        self.temporal_transformer.reset_cache()


def create_mobilevit_pretrained(
    spatial_encoder='mobilenet_v3',  # 'mobilevit_s' requires timm
    freeze_early_layers=False,
    temporal_layers=6,
    temporal_heads=8,
    temporal_dim=256,
    aux_input_on=True,
    dropout=0.1,
):
    """
    Factory function to create model with pretrained spatial encoder.
    
    Args:
        spatial_encoder: Choose from:
            - 'mobilenet_v3': MobileNetV3 (always available, ~3M params)
            - 'mobilevit_s': MobileViT-Small (~5.6M params, requires timm)
            - 'mobilevit_xs': MobileViT-XS (~2.3M params, requires timm)
            - 'mobilevit_xxs': MobileViT-XXS (~1.3M params, requires timm)
        freeze_early_layers: Whether to freeze early layers of pretrained model
    """
    
    # Calculate auxiliary input dimension
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    aux_len = action_dim if aux_input_on else 17
    
    model = MobileViTTemporalModelPretrained(
        spatial_encoder=spatial_encoder,
        freeze_early_layers=freeze_early_layers,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        temporal_dim=temporal_dim,
        max_seq_len=input_shape[0],
        aux_input_length=aux_len,
        aux_input_on=aux_input_on,
        dropout=dropout,
    )
    
    return model


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Testing Pretrained MobileViT/MobileNet Model")
    print("=" * 60)
    
    # Test with MobileNetV3 (always available)
    print("\n1. Testing with MobileNetV3 (always available):")
    model = create_mobilevit_pretrained(
        spatial_encoder='mobilevit_s',
        freeze_early_layers=True,  # Freeze early layers for faster training
        temporal_layers=4,
        aux_input_on=True
    )
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test forward pass
    B, T, H, W, C = 1, 4, 150, 280, 3
    x = torch.randn(B, T, H, W, C)
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    aux = torch.randn(B, T, action_dim)
    
    with torch.no_grad():
        outputs = model(x, aux)
    
    print("\n✓ Forward pass successful")
    for i, name in enumerate(['keys', 'clicks', 'mouse_x', 'mouse_y', 'value']):
        print(f"  {name}: {outputs[i].shape}")
    
    # Benchmark
    print("\n2. Benchmarking inference speed:")
    model.eval()
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(x, aux)
    
    # Time it
    num_iterations = 10
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x, aux)
    elapsed = time.time() - start
    
    ms_per_batch = (elapsed / num_iterations) * 1000
    fps = (B * T * num_iterations) / elapsed
    
    print(f"  Inference: {ms_per_batch:.1f} ms/batch")
    print(f"  Throughput: {fps:.1f} frames/second")
    
    print("\n" + "=" * 60)
    print("TIP: To use actual MobileViT, install timm:")
    print("  pip install timm")
    print("Then use: spatial_encoder='mobilevit_s'")
    print("=" * 60)
