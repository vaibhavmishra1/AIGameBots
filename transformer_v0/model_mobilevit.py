import os
import sys
from typing import Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Config values (from config.py)
input_shape = (96, 150, 280, 3)  # (timesteps, height, width, channels)
n_keys = 11      # keyboard outputs
n_clicks = 2     # mouse buttons
n_mouse_x = 23   # mouse x positions
n_mouse_y = 15   # mouse y positions


class MobileViTBlock(nn.Module):
    """MobileViT block combining local and global processing."""
    
    def __init__(self, in_channels, out_channels, d_model, layers=2, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Local representation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        
        # Linear projection to transformer dimension
        self.conv2 = nn.Conv2d(in_channels, d_model, 1)
        
        # Global representation (transformer)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for stability
            ),
            num_layers=layers
        )
        
        # Project back
        self.conv3 = nn.Conv2d(d_model, out_channels, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Local processing
        x = self.conv1(x)
        
        # Unfold into patches and project
        x_proj = self.conv2(x)  # (B, d_model, H, W)
        
        # Create patches
        ph, pw = H // self.patch_size, W // self.patch_size
        # Reshape: (B, C, H, W) -> (B, C, ph, patch_size, pw, patch_size)
        x_patches = x_proj.view(B, self.d_model, ph, self.patch_size, pw, self.patch_size)
        # Transpose: -> (B, ph, pw, patch_size, patch_size, C)
        x_patches = x_patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        # Flatten patches: -> (B, ph*pw, patch_size*patch_size*C)
        patches = x_patches.view(B, ph * pw, self.patch_size * self.patch_size * self.d_model)
        
        # Global processing (transformer on patches)
        patches = self.transformer(patches)
        
        # Fold back to spatial
        # Reshape: (B, ph*pw, patch_size*patch_size*C) -> (B, ph, pw, patch_size, patch_size, C)
        x_global = patches.view(B, ph, pw, self.patch_size, self.patch_size, self.d_model)
        # Transpose: -> (B, C, ph, patch_size, pw, patch_size)
        x_global = x_global.permute(0, 5, 1, 3, 2, 4).contiguous()
        # Reshape: -> (B, C, H, W)
        x_global = x_global.view(B, self.d_model, H, W)
        
        # Project back and fusion
        x = self.conv3(x_global)
        x = self.conv4(x)
        
        return x


class MobileViTSpatialEncoder(nn.Module):
    """
    Lightweight MobileViT-inspired spatial encoder.
    ~5M parameters, efficient for real-time inference.
    """
    
    def __init__(self, img_size=(150, 280), output_dim=256):
        super().__init__()
        
        # Simplified architecture without MobileViT blocks due to dimension constraints
        # We'll use efficient conv blocks instead
        
        # Initial downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 75x140
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        
        # Stage 1: Add spatial attention
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
        )
        self.stage1_attn = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 1),
            nn.Sigmoid()
        )
        self.stage1_down = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=2, padding=1),  # 38x70
            nn.BatchNorm2d(96),
            nn.SiLU(),
        )
        
        # Stage 2: Standard convolution
        self.stage2 = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 19x35
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        
        # Stage 3: Final feature extraction
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.Conv2d(192, output_dim, 3, stride=2, padding=1),  # 10x18
            nn.BatchNorm2d(output_dim),
            nn.SiLU(),
        )
        
        # Global context via adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, output_dim),
            nn.Sigmoid()
        )
        
        # Final spatial dimensions
        self.final_height = 10
        self.final_width = 18
        self.output_channels = output_dim
        
    def forward(self, x):
        """
        Args:
            x: (B*T, 3, H, W) - batch of frames
        Returns:
            features: (B*T, C, H', W') - spatial features with global context
        """
        x = self.stem(x)      # (B*T, 64, 75, 140)
        
        # Stage 1 with spatial attention
        x = self.stage1_conv(x)
        attn = self.stage1_attn(x)
        x = x * attn
        x = self.stage1_down(x)  # (B*T, 96, 38, 70)
        
        x = self.stage2(x)    # (B*T, 128, 19, 35)
        x = self.stage3(x)    # (B*T, 256, 10, 18)
        
        # Add global context gating
        B, C, H, W = x.shape
        global_feat = self.global_pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        gate = self.global_gate(global_feat).view(B, C, 1, 1)  # (B, C, 1, 1)
        x = x * gate  # Modulate features with global context
        
        return x


class StatefulTemporalTransformer(nn.Module):
    """
    Temporal transformer with KV caching for stateful inference.
    Processes sequences of spatial features from MobileViT.
    """
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, max_seq_len=96):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_seq_len = max_seq_len
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Transformer layers with KV caching support
        self.layers = nn.ModuleList([
            StatefulTransformerLayer(d_model, nhead, dim_feedforward=d_model*4)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # KV cache for stateful inference
        self._kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
        self._cache_seq_len = 0
        
    def forward(self, x, use_cache=False):
        """
        Args:
            x: (B, T, C) - temporal features
            use_cache: Whether to use KV caching (for inference)
        Returns:
            output: (B, T, C)
        """
        B, T, C = x.shape
        
        # Add positional encoding
        if use_cache and self._kv_cache is not None:
            # For cached inference, only add position for new tokens
            start_pos = self._cache_seq_len
            pos = self.pos_encoding[:, start_pos:start_pos+T]
        else:
            pos = self.pos_encoding[:, :T]
        x = x + pos
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            if use_cache:
                # Initialize cache dict if needed
                if self._kv_cache is None:
                    self._kv_cache = {}
                    
                # Get cache for this layer or use 'init' flag for first time
                cache_for_layer = self._kv_cache.get(i, 'init')
                result = layer(x, kv_cache=cache_for_layer)
                
                # Result will be tuple when caching is enabled
                if isinstance(result, tuple):
                    x, new_kv = result
                    self._kv_cache[i] = new_kv
                else:
                    # Shouldn't happen with current logic
                    x = result
            else:
                # When no cache, layer returns just output
                x = layer(x, kv_cache=None)
                
        x = self.norm(x)
        
        if use_cache:
            self._cache_seq_len += T
            # Trim cache if it exceeds max length
            if self._cache_seq_len > self.max_seq_len:
                self._trim_cache()
                
        return x
    
    def reset_cache(self):
        """Clear the KV cache."""
        self._kv_cache = None
        self._cache_seq_len = 0
        
    def _trim_cache(self):
        """Trim cache to keep only recent tokens."""
        if self._kv_cache is None:
            return
        trim_len = self._cache_seq_len - self.max_seq_len + 16  # Keep some buffer
        for i in self._kv_cache:
            k, v = self._kv_cache[i]
            self._kv_cache[i] = (k[:, :, trim_len:], v[:, :, trim_len:])
        self._cache_seq_len -= trim_len


class StatefulTransformerLayer(nn.Module):
    """Single transformer layer with KV caching support."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu
        
    def forward(self, x, kv_cache=None):
        """
        Args:
            x: (B, T, D) - input sequence
            kv_cache: Optional tuple of (K_prev, V_prev) for caching
        Returns:
            If kv_cache is provided: (output, new_kv) tuple
            Otherwise: just output
        """
        # Self-attention with potential KV caching
        return_cache = kv_cache is not None or kv_cache == 'init'
        
        if kv_cache is not None and kv_cache != 'init':
            # For cached inference with existing cache
            q = x
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, x], dim=1)
            v = torch.cat([v_prev, x], dim=1)
            
            # Apply attention
            x2 = self.norm1(q)
            attn_out, _ = self.self_attn(x2, k, v, need_weights=False, is_causal=False)
            x = x + self.dropout1(attn_out)
            
            new_kv = (k, v)
        else:
            # Standard forward pass or initial cache
            x2 = self.norm1(x)
            # Create causal mask
            seq_len = x2.size(1)
            mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x2.device)
            attn_out, _ = self.self_attn(x2, x2, x2, attn_mask=mask, need_weights=False)
            x = x + self.dropout1(attn_out)
            
            if kv_cache == 'init':
                # Initialize cache
                new_kv = (x2.clone(), x2.clone())
            else:
                new_kv = None
            
        # FFN
        x2 = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x2)))))
        
        if return_cache and new_kv is not None:
            return x, new_kv
        return x


class MobileViTTemporalModel(nn.Module):
    """
    Complete model: MobileViT spatial encoder + Temporal Transformer with KV caching.
    Designed for efficient inference with stateful processing.
    """
    
    def __init__(
        self,
        spatial_dim=256,
        temporal_layers=6,
        temporal_heads=8,
        max_seq_len=96,
        aux_input_length=17,
        aux_input_on=True,
        dropout=0.1,
    ):
        super().__init__()
        
        # Spatial encoder (MobileViT-based)
        self.spatial_encoder = MobileViTSpatialEncoder(output_dim=spatial_dim)
        
        # Spatial pooling options
        self.spatial_pool_type = 'attention'  # 'attention', 'avg', or 'max'
        
        if self.spatial_pool_type == 'attention':
            # Learnable spatial attention pooling
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(spatial_dim, 64, 1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1),
            )
        
        # Project spatial features for temporal model
        spatial_feat_dim = spatial_dim
        if self.spatial_pool_type != 'flatten':
            # After pooling, we have spatial_dim features
            self.spatial_proj = nn.Linear(spatial_dim, spatial_dim)
        else:
            # If flattening, we have spatial_dim * H * W features
            self.spatial_proj = nn.Linear(
                spatial_dim * self.spatial_encoder.final_height * self.spatial_encoder.final_width,
                spatial_dim
            )
        
        # Temporal transformer with KV caching
        self.temporal_transformer = StatefulTemporalTransformer(
            d_model=spatial_dim,
            nhead=temporal_heads,
            num_layers=temporal_layers,
            max_seq_len=max_seq_len
        )
        
        # Auxiliary input processing
        self.aux_input_on = aux_input_on
        if aux_input_on:
            self.aux_dense = nn.Sequential(
                nn.Linear(aux_input_length, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 256)
            )
            feature_dim = spatial_dim + 256
        else:
            feature_dim = spatial_dim
            
        # Shared output processing
        self.shared_dense = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        
        # Task-specific heads
        self.keys_output = nn.Linear(256, n_keys)
        self.clicks_output = nn.Linear(256, n_clicks)
        self.mouse_x_output = nn.Linear(256, n_mouse_x)
        self.mouse_y_output = nn.Linear(256, n_mouse_y)
        self.value_output = nn.Linear(256, 1)
        
        # State for stateful inference
        self._stateful_mode = False
        self._spatial_cache = None
        
    def forward(self, x, aux_input=None, stateful=False):
        """
        Args:
            x: (B, T, H, W, C) - video frames
            aux_input: Optional (B, T, aux_dim) - auxiliary features
            stateful: Whether to use stateful inference with caching
        Returns:
            Tuple of (keys, clicks, mouse_x, mouse_y, value) predictions
        """
        B, T, H, W, C = x.shape
        
        # Rearrange to (B, T, C, H, W) then (B*T, C, H, W)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B * T, C, H, W)
        
        # Spatial encoding with MobileViT
        if stateful and self._stateful_mode and T == 1:
            # For stateful inference, we can cache spatial features
            # But for simplicity, we'll recompute them (they're fast with MobileViT)
            spatial_features = self.spatial_encoder(x)  # (B*T, C, H', W')
        else:
            spatial_features = self.spatial_encoder(x)  # (B*T, C, H', W')
            
        # Pool spatial features
        if self.spatial_pool_type == 'attention':
            # Attention-weighted pooling
            B_T, C_s, H_s, W_s = spatial_features.shape
            attention_weights = self.spatial_attention(spatial_features)  # (B*T, 1, H', W')
            attention_weights = attention_weights.view(B_T, 1, -1)  # (B*T, 1, H'*W')
            attention_weights = F.softmax(attention_weights, dim=-1)
            spatial_flat = spatial_features.view(B_T, C_s, -1)  # (B*T, C, H'*W')
            pooled_features = (spatial_flat * attention_weights).sum(dim=-1)  # (B*T, C)
        elif self.spatial_pool_type == 'avg':
            pooled_features = F.adaptive_avg_pool2d(spatial_features, 1).squeeze(-1).squeeze(-1)
        elif self.spatial_pool_type == 'max':
            pooled_features = F.adaptive_max_pool2d(spatial_features, 1).squeeze(-1).squeeze(-1)
        else:  # flatten
            pooled_features = spatial_features.flatten(1)
            
        # Project spatial features
        temporal_input = self.spatial_proj(pooled_features)  # (B*T, spatial_dim)
        temporal_input = temporal_input.view(B, T, -1)  # (B, T, spatial_dim)
        
        # Temporal transformer with optional KV caching
        if stateful and self._stateful_mode:
            temporal_features = self.temporal_transformer(temporal_input, use_cache=True)
        else:
            temporal_features = self.temporal_transformer(temporal_input, use_cache=False)
            
        # Add auxiliary input if provided
        if self.aux_input_on and aux_input is not None:
            if aux_input.dim() == 2:
                # (B, aux_dim) -> (B, T, 256)
                aux_features = self.aux_dense(aux_input).unsqueeze(1).expand(-1, T, -1)
            else:
                # (B, T, aux_dim) -> (B, T, 256)
                B2, T2, D2 = aux_input.shape
                aux_flat = aux_input.view(B2 * T2, D2)
                aux_features = self.aux_dense(aux_flat).view(B2, T2, 256)
            features = torch.cat([temporal_features, aux_features], dim=-1)
        else:
            features = temporal_features
            
        # Shared processing and output heads
        shared = self.shared_dense(features)  # (B, T, 256)
        
        # Task-specific predictions
        keys_out = torch.sigmoid(self.keys_output(shared))
        clicks_out = torch.sigmoid(self.clicks_output(shared))
        mouse_x_out = F.softmax(self.mouse_x_output(shared), dim=-1)
        mouse_y_out = F.softmax(self.mouse_y_output(shared), dim=-1)
        value_out = self.value_output(shared)
        
        return keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out
    
    def set_stateful(self, enabled=True):
        """Enable/disable stateful inference mode."""
        self._stateful_mode = enabled
        if not enabled:
            self.reset_states()
            
    def reset_states(self):
        """Reset all cached states."""
        self.temporal_transformer.reset_cache()
        self._spatial_cache = None
        
    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_mobilevit_temporal_model(
    spatial_dim=256,
    temporal_layers=6,
    temporal_heads=8,
    max_seq_len=96,
    aux_input_on=True,
    dropout=0.1,
):
    """Factory function to create MobileViT + Temporal Transformer model."""
    
    # Calculate auxiliary input dimension
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    aux_len = action_dim if aux_input_on else 17
    
    model = MobileViTTemporalModel(
        spatial_dim=spatial_dim,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        max_seq_len=max_seq_len,
        aux_input_length=aux_len,
        aux_input_on=aux_input_on,
        dropout=dropout,
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    torch.manual_seed(0)
    
    print("Creating MobileViT + Temporal Transformer model...")
    model = create_mobilevit_temporal_model(
        spatial_dim=256,
        temporal_layers=4,  # Fewer layers for testing
        temporal_heads=8,
        aux_input_on=True
    )
    
    # Count parameters
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test forward pass
    print("\n--- Testing batch inference ---")
    B, T, H, W, C = 1, 4, 150, 280, 3  # Smaller batch for testing
    x = torch.randn(B, T, H, W, C)
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    aux = torch.randn(B, T, action_dim)
    
    with torch.no_grad():
        outputs = model(x, aux, stateful=False)
    
    print(f"✓ Batch forward pass successful")
    for i, name in enumerate(['keys', 'clicks', 'mouse_x', 'mouse_y', 'value']):
        print(f"  {name}: {outputs[i].shape}")
    
    # Test stateful inference
    print("\n--- Testing stateful inference ---")
    model.set_stateful(True)
    model.reset_states()
    
    # Process sequence frame by frame (simulating real-time inference)
    all_outputs = []
    for t in range(2):
        x_t = torch.randn(1, 1, H, W, C)  # Single frame
        aux_t = torch.randn(1, 1, action_dim)
        
        with torch.no_grad():
            outputs_t = model(x_t, aux_t, stateful=True)
        all_outputs.append([o.squeeze(1) for o in outputs_t])
    
    print(f"✓ Stateful inference successful")
    print(f"  Processed 2 frames sequentially with KV caching")
    
    print("\nModel is ready for training!")
