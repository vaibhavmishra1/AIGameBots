import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared config values (shapes, counts)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))
from config import (  # noqa: E402
    N_TIMESTEPS,
    input_shape,
    n_keys,
    n_clicks,
    n_mouse_x,
    n_mouse_y,
)


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


class PatchEmbed2D(nn.Module):
    """
    2D Patch embedding using Conv2d.
    Splits HxW image into (H/ps_h)*(W/ps_w) patches and projects to embed_dim.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: Tuple[int, int] = (10, 10)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B*T, C, H, W)
        x = self.proj(x)  # (B*T, D, H', W')
        b, d, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B*T, N_patches, D)
        return x, hp, wp


def _build_sincos_2d_position_embedding(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Create 2D sine-cosine positional embeddings of shape (1, height*width, dim).
    """
    assert dim % 4 == 0, "Embedding dim must be divisible by 4 for 2D sincos"
    half_dim = dim // 2
    dim_h = dim_w = half_dim

    y = torch.arange(height, device=device).float()
    x = torch.arange(width, device=device).float()
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # (H', W')

    omega_h = torch.arange(dim_h // 2, device=device).float()
    omega_h = 1.0 / (10000 ** (2 * omega_h / dim_h))
    omega_w = torch.arange(dim_w // 2, device=device).float()
    omega_w = 1.0 / (10000 ** (2 * omega_w / dim_w))

    out_y = torch.einsum('hw,d->hwd', grid_y, omega_h)
    out_x = torch.einsum('hw,d->hwd', grid_x, omega_w)

    pos_y = torch.cat([out_y.sin(), out_y.cos()], dim=-1)  # (H', W', dim_h)
    pos_x = torch.cat([out_x.sin(), out_x.cos()], dim=-1)  # (H', W', dim_w)

    pos = torch.cat([pos_y, pos_x], dim=-1)  # (H', W', dim)
    pos = pos.view(1, height * width, dim)
    return pos


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
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ViViTCSGOModel(nn.Module):
    """
    ViViT-style video transformer for CSGO action recognition.

    Pipeline:
      - Per-frame patch embedding (Conv2d with patch_size = 10x10)
      - Per-frame spatial Transformer encoder over patch tokens
      - Mean-pool tokens to obtain a frame embedding
      - Temporal Transformer encoder over sequence of frame embeddings
      - Shared MLP and per-timestep heads (keys, clicks, mouse_x, mouse_y, value)
    """

    def __init__(
        self,
        model_name: str = 'vivit_default',
        embed_dim: int = 512,
        spatial_depth: int = 4,
        temporal_depth: int = 4,
        nhead: int = 8,
        patch_size: Tuple[int, int] = (10, 10),
        aux_input_length: int = 17,
        aux_input_on: bool = True,
        pretrained: bool = False,
        pretrained_model_name: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.spatial_depth = spatial_depth
        self.temporal_depth = temporal_depth
        self.nhead = nhead
        self.patch_size = patch_size
        self.aux_input_length = aux_input_length
        self.aux_input_on = aux_input_on
        self._use_pretrained = bool(pretrained)
        self._pretrained_model_name = pretrained_model_name or 'vit_small_patch16_224'

        # Stateful controls
        self._stateful: bool = False
        self._temporal_cache: Optional[torch.Tensor] = None  # (B, t_cached, D)

        # Normalization buffers (ImageNet) matching EfficientNet path
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

        # Modules
        self.patch_embed = PatchEmbed2D(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)
        self.spatial_encoder = TransformerEncoder(d_model=embed_dim, nhead=nhead, depth=spatial_depth, mlp_ratio=4.0, dropout=0.1)
        self.temporal_encoder = TransformerEncoder(d_model=embed_dim, nhead=nhead, depth=temporal_depth, mlp_ratio=4.0, dropout=0.1)

        # Heads and shared layers
        if self.aux_input_on:
            self.aux_dense = nn.Linear(aux_input_length, 256)
            shared_in = 512 + 256  # temporal D + aux 256
        else:
            self.aux_dense = None
            shared_in = 512

        # If embed_dim != 512, project to 512 before heads to match downstream sizes
        self.proj_to_head = nn.Linear(embed_dim, 512) if embed_dim != 512 else nn.Identity()
        self.shared_dense = nn.Linear(shared_in, 256)

        # Per-timestep heads
        self.keys_output = TimeDistributed(nn.Linear(256, n_keys))
        self.clicks_output = TimeDistributed(nn.Linear(256, n_clicks))
        self.mouse_x_output = TimeDistributed(nn.Linear(256, n_mouse_x))
        self.mouse_y_output = TimeDistributed(nn.Linear(256, n_mouse_y))
        self.value_output = TimeDistributed(nn.Linear(256, 1))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # Optionally initialize spatial vision backbone from pretrained ViT
        if self._use_pretrained:
            self._init_from_vit_pretrained(self._pretrained_model_name)

    def _encode_frames_spatial(self, x_btchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_btchw: (B*T, C, H, W)
        Returns:
            frame_embeds: (B*T, D)
        """
        tokens, hp, wp = self.patch_embed(x_btchw)  # (B*T, Np, D)
        pos = _build_sincos_2d_position_embedding(hp, wp, self.embed_dim, tokens.device)  # (1, Np, D)
        tokens = tokens + pos
        tokens = self.spatial_encoder(tokens)  # (B*T, Np, D)
        frame_embeds = tokens.mean(dim=1)  # mean pool tokens -> (B*T, D)
        return frame_embeds

    def _encode_temporal(self, frame_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_seq: (B, T, D)
        Returns:
            temporal_out: (B, T, D)
        """
        b, t, d = frame_seq.shape
        pos_t = _build_sincos_1d_position_embedding(t, d, frame_seq.device)  # (1, T, D)
        x = frame_seq + pos_t
        x = self.temporal_encoder(x)  # (B, T, D)
        return x

    def set_stateful(self, enabled: bool = True) -> None:
        self._stateful = bool(enabled)
        if not self._stateful:
            self.reset_states()

    def reset_states(self) -> None:
        self._temporal_cache = None

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

        # Convert to (B, T, C, H, W) and normalize with ImageNet stats
        if c == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = (x - self.imagenet_mean) / self.imagenet_std  # (B, T, 3, H, W)

        # Spatial encoding per frame
        x_btchw = x.view(b * t, c, h, w)
        frame_bt = self._encode_frames_spatial(x_btchw)  # (B*T, D)
        frame_seq = frame_bt.view(b, t, self.embed_dim)  # (B, T, D)

        # Stateful cache management
        if self._stateful:
            if self._temporal_cache is None:
                self._temporal_cache = frame_seq
            else:
                self._temporal_cache = torch.cat([self._temporal_cache, frame_seq], dim=1)
                if self._temporal_cache.size(1) > N_TIMESTEPS:
                    self._temporal_cache = self._temporal_cache[:, -N_TIMESTEPS:, :]
            temporal_in = self._temporal_cache
        else:
            temporal_in = frame_seq

        # Temporal encoding
        temporal_out = self._encode_temporal(temporal_in)  # (B, T_eff, D)

        # Project to head dimension and slice to the last t steps if stateful
        temporal_out = self.proj_to_head(temporal_out)  # (B, T_eff, 512)
        if temporal_out.size(1) != t:
            # keep only the most recent t tokens to align heads per input length
            temporal_out = temporal_out[:, -t:, :]

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
            x_features = torch.cat([temporal_out, aux_features], dim=-1)
        else:
            x_features = temporal_out

        # Shared dense and heads
        shared = self.shared_dense(x_features)  # (B, T, 256)

        keys_out = self.keys_output(shared)
        clicks_out = self.clicks_output(shared)
        mouse_x_out = self.softmax(self.mouse_x_output(shared))
        mouse_y_out = self.softmax(self.mouse_y_output(shared))
        value_out = self.value_output(shared)

        return keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out

    def get_output_concatenated(self, x: torch.Tensor, aux_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = self.forward(x, aux_input)
        return torch.cat([keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out], dim=-1)

    def _init_from_vit_pretrained(self, timm_model_name: str) -> None:
        """
        Initialize patch embedding and spatial Transformer encoder from a pretrained ViT.
        Uses timm to load weights and maps them into nn.TransformerEncoder layers.
        Falls back gracefully if timm is unavailable or shapes do not match.
        """
        try:
            import timm  # type: ignore
        except Exception:
            print("[ViViT] timm not installed; skipping pretrained ViT initialization", file=sys.stderr)
            return

        try:
            vit = timm.create_model(timm_model_name, pretrained=True)
        except Exception as e:
            print(f"[ViViT] Failed to load timm model '{timm_model_name}': {e}", file=sys.stderr)
            return

        vit.eval()

        with torch.no_grad():
            # 1) Patch embedding projection
            if hasattr(vit, 'patch_embed') and hasattr(vit.patch_embed, 'proj'):
                src_proj = vit.patch_embed.proj
                dst_proj = self.patch_embed.proj

                src_w = src_proj.weight  # (out, in, kh, kw)
                src_b = src_proj.bias
                dst_w = dst_proj.weight

                if src_w.shape[1] != dst_w.shape[1]:
                    print("[ViViT] In-channels mismatch for patch embed; skipping patch init", file=sys.stderr)
                else:
                    # Resize kernel if patch sizes differ (e.g., 16 -> 10)
                    if src_w.shape[2:] != dst_w.shape[2:]:
                        try:
                            resized = F.interpolate(src_w, size=dst_w.shape[2:], mode='bilinear', align_corners=False)
                            src_w = resized
                        except Exception as e:
                            print(f"[ViViT] Patch kernel resize failed: {e}; skipping patch init", file=sys.stderr)
                            src_w = None

                    if src_w is not None and src_w.shape[0] == dst_w.shape[0]:
                        dst_proj.weight.copy_(src_w)
                        if (src_b is not None) and (dst_proj.bias is not None) and (src_b.shape == dst_proj.bias.shape):
                            dst_proj.bias.copy_(src_b)
                    else:
                        print("[ViViT] Embed dim mismatch for patch embed; skipping patch init", file=sys.stderr)

            # 2) Spatial encoder blocks (map first N ViT blocks)
            blocks_src = getattr(vit, 'blocks', None)
            if blocks_src is None:
                print("[ViViT] timm ViT has no blocks; skipping spatial encoder init", file=sys.stderr)
                return

            layers_dst = self.spatial_encoder.encoder.layers
            num_to_copy = min(len(blocks_src), len(layers_dst))
            if num_to_copy == 0:
                return

            # Validate dims and heads via first block
            try:
                embed_dim_src = blocks_src[0].attn.qkv.weight.shape[1]
                heads_src = int(blocks_src[0].attn.num_heads)
            except Exception:
                embed_dim_src = None
                heads_src = None

            if (embed_dim_src is not None and embed_dim_src != self.embed_dim) or (heads_src is not None and heads_src != self.nhead):
                print("[ViViT] Encoder dim/heads mismatch; skipping spatial encoder init", file=sys.stderr)
                return

            for i in range(num_to_copy):
                blk = blocks_src[i]
                layer = layers_dst[i]

                # Norms
                try:
                    layer.norm1.weight.copy_(blk.norm1.weight)
                    layer.norm1.bias.copy_(blk.norm1.bias)
                    layer.norm2.weight.copy_(blk.norm2.weight)
                    layer.norm2.bias.copy_(blk.norm2.bias)
                except Exception:
                    pass

                # Attention: map qkv and proj
                try:
                    qkv_w = blk.attn.qkv.weight  # (3D, D)
                    qkv_b = blk.attn.qkv.bias    # (3D)
                    layer.self_attn.in_proj_weight.copy_(qkv_w)
                    layer.self_attn.in_proj_bias.copy_(qkv_b)
                    layer.self_attn.out_proj.weight.copy_(blk.attn.proj.weight)
                    layer.self_attn.out_proj.bias.copy_(blk.attn.proj.bias)
                except Exception:
                    pass

                # MLP: fc1/fc2 -> linear1/linear2
                try:
                    layer.linear1.weight.copy_(blk.mlp.fc1.weight)
                    layer.linear1.bias.copy_(blk.mlp.fc1.bias)
                    layer.linear2.weight.copy_(blk.mlp.fc2.weight)
                    layer.linear2.bias.copy_(blk.mlp.fc2.bias)
                except Exception:
                    pass

        print(f"[ViViT] Loaded pretrained ViT weights from '{timm_model_name}' into patch embed and {num_to_copy} spatial blocks")


def create_vivit_model(
    model_name: str = 'vivit_default',
    pretrained: bool = False,
    aux_input_on: bool = True,
    freeze_backbone: bool = False,  # kept for API parity; not used here
) -> ViViTCSGOModel:
    """
    Factory to create ViViTCSGOModel with config-driven I/O sizes.
    """
    # Determine aux_input_length: if aux is enabled, prefer previous action dimension
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    aux_len = action_dim if aux_input_on else 17  # fallback to config default length

    # Choose ViT pretrained backbone config when requested (defaults to ViT-S/16)
    if pretrained:
        vit_name = 'vit_small_patch16_224'
        embed_dim = 384
        nhead = 6
    else:
        vit_name = None
        embed_dim = 512
        nhead = 8

    model = ViViTCSGOModel(
        model_name=model_name,
        embed_dim=embed_dim,
        spatial_depth=4,
        temporal_depth=4,
        nhead=nhead,
        patch_size=(10, 10),
        aux_input_length=aux_len,
        aux_input_on=aux_input_on,
        pretrained=pretrained,
        pretrained_model_name=vit_name,
    )

    # No backbone params to freeze; kept for API compatibility
    for p in model.parameters():
        p.requires_grad = True

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

