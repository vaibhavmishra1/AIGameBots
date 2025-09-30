import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import math
import sys
import os
import ssl
from typing import Optional, Tuple

# Add the parent directory to the path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))

# Fix SSL certificate issues on some systems
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


class ConvLSTM2D(nn.Module):
    """
    Convolutional LSTM implementation for PyTorch.
    Based on the TensorFlow ConvLSTM2D used in the original model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        return_sequences: bool = True,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0
    ):
        super(ConvLSTM2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        # ConvLSTM cell dimensions
        self.hidden_dim = out_channels

        # Convolution layers for gates
        self.conv_xi = nn.Conv2d(in_channels, out_channels * 4, kernel_size, stride, padding, dilation)
        self.conv_hi = nn.Conv2d(out_channels, out_channels * 4, kernel_size, stride, padding, dilation, bias=False)

        # Dropout layers
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout2d(dropout)
        if self.recurrent_dropout > 0:
            self.recurrent_dropout_layer = nn.Dropout2d(recurrent_dropout)

    def forward(self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass through ConvLSTM2D.

        Args:
            x: Input tensor of shape (batch, timesteps, channels, height, width)
            states: Optional tuple of (hidden_state, cell_state)

        Returns:
            Output tensor and new states
        """
        batch_size, timesteps, channels, height, width = x.size()

        if states is None:
            h_t = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
            c_t = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
        else:
            h_t, c_t = states

        outputs = []

        for t in range(timesteps):
            x_t = x[:, t, :, :, :]  # (batch, channels, height, width)

            # Apply dropout to input if specified
            if self.dropout > 0:
                x_t = self.dropout_layer(x_t)

            # Compute gates
            gates = self.conv_xi(x_t) + self.conv_hi(h_t)

            # Split gates
            i_t, f_t, g_t, o_t = torch.chunk(gates, 4, dim=1)

            # Apply activations
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            # Apply recurrent dropout to hidden state if specified
            if self.recurrent_dropout > 0 and t > 0:
                h_t = self.recurrent_dropout_layer(h_t)

            # Update cell and hidden states
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            if self.return_sequences:
                outputs.append(h_t)

        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)  # (batch, timesteps, channels, height, width)
        else:
            outputs = h_t  # (batch, channels, height, width)

        return outputs, (h_t, c_t)


class TimeDistributed(nn.Module):
    """
    Applies a module to each timestep of a sequence.
    Similar to TensorFlow's TimeDistributed layer.
    """

    def __init__(self, module: nn.Module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, timesteps, channels, height, width)

        Returns:
            Output tensor of shape (batch, timesteps, output_channels, output_height, output_width)
        """
        batch_size, timesteps = x.size(0), x.size(1)
        feature_dims = x.size()[2:]

        x = x.contiguous().view(batch_size * timesteps, *feature_dims)
        y = self.module(x)

        if isinstance(y, torch.Tensor):
            y = y.contiguous()
            output_dims = y.size()[1:]
            y = y.view(batch_size, timesteps, *output_dims)
        else:
            raise TypeError("TimeDistributed module must return a Tensor")
        return y


class CSGOModel(nn.Module):
    """
    PyTorch implementation of the CSGO behavioral cloning model.
    Based on the TensorFlow model from dm_train_model.py.
    """

    def __init__(
        self,
        model_name: str = 'default',
        input_shape: Tuple[int, int, int, int] = (96, 150, 280, 3),
        n_keys: int = 11,
        n_clicks: int = 2,
        n_mouse_x: int = 23,
        n_mouse_y: int = 15,
        aux_input_length: int = 17,
        aux_input_on: bool = False,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the CSGO model.

        Args:
            model_name: Name of the model configuration (affects architecture)
            input_shape: Shape of input tensor (timesteps, height, width, channels)
            n_keys: Number of keyboard outputs
            n_clicks: Number of mouse button outputs
            n_mouse_x: Number of discrete mouse x positions
            n_mouse_y: Number of discrete mouse y positions
            aux_input_length: Length of auxiliary input
            aux_input_on: Whether to use auxiliary input
            pretrained: Whether to use pretrained EfficientNet weights
        """
        super(CSGOModel, self).__init__()

        self.model_name = model_name
        self.input_shape = input_shape
        self.n_keys = n_keys
        self.n_clicks = n_clicks
        self.n_mouse_x = n_mouse_x
        self.n_mouse_y = n_mouse_y
        self.aux_input_length = aux_input_length
        self.aux_input_on = aux_input_on

        # Stateful controls
        self._stateful: bool = False
        self._convlstm1_state = None  # Tuple[Tensor, Tensor] or None
        self._convlstm_extra_state = None  # Tuple[Tensor, Tensor] or None
        self._lstm_state = None  # Tuple[Tensor, Tensor] or None

        # Load EfficientNetB0 as base model
        if pretrained and 'randinit' not in model_name:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.base_model = efficientnet_b0(weights=weights)
        else:
            self.base_model = efficientnet_b0(weights=None)

        # Freeze or unfreeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = not freeze_backbone

        # Use only the feature extractor portion of EfficientNetB0
        self.intermediate_model = self.base_model.features

        # TimeDistributed wrapper for intermediate model
        self.time_distributed = TimeDistributed(self.intermediate_model)

        # Determine ConvLSTM2D parameters based on model name
        convlstm_filters = 512 if 'big' in model_name else 256
        dropout = 0.5 if 'drop' in model_name else 0.0
        recurrent_dropout = 0.5 if 'drop' in model_name else 0.0

        # First ConvLSTM2D layer
        self.convlstm1 = ConvLSTM2D(
            in_channels=1280,  # EfficientNetB0 feature size
            out_channels=convlstm_filters,
            kernel_size=(3, 3),
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )

        # Extra ConvLSTM2D layer if specified
        if 'extra' in model_name:
            self.convlstm_extra = ConvLSTM2D(
                in_channels=convlstm_filters,
                out_channels=256,
                kernel_size=(3, 3),
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            )

        # Flatten layer
        self.flatten = TimeDistributed(nn.Flatten())

        # LSTM layer if specified
        if 'LSTM' in model_name:
            self.lstm_dropout = nn.Dropout(0.5) if 'drop' in model_name else None
            self.lstm = nn.LSTM(
                input_size=convlstm_filters * 5 * 9,  # 5x9 feature map size
                hidden_size=256,
                batch_first=True,
                dropout=0.0,  # Internal dropout handled separately
                bidirectional=False
            )
            self.lstm_output_dropout = nn.Dropout(0.5) if 'drop' in model_name else None

        # Auxiliary input layer if needed
        if self.aux_input_on:
            self.aux_dense = nn.Linear(aux_input_length, 256)

        # Shared dense layers
        if 'LSTM' in model_name:
            shared_input_size = 256 + (256 if self.aux_input_on else 0)
        else:
            shared_input_size = convlstm_filters * 5 * 9 + (256 if self.aux_input_on else 0)

        self.shared_dense = nn.Linear(shared_input_size, 256)

        # Output heads
        self.keys_output = TimeDistributed(nn.Linear(256, n_keys))
        self.clicks_output = TimeDistributed(nn.Linear(256, n_clicks))
        self.mouse_x_output = TimeDistributed(nn.Linear(256, n_mouse_x))
        self.mouse_y_output = TimeDistributed(nn.Linear(256, n_mouse_y))
        self.value_output = TimeDistributed(nn.Linear(256, 1))

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # ImageNet normalization buffers for EfficientNet inputs
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor, aux_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, timesteps, height, width, channels)
            aux_input: Optional auxiliary input tensor

        Returns:
            Tuple of output tensors for keys, clicks, mouse_x, mouse_y, and value
        """
        batch_size, timesteps, height, width, channels = x.size()

        # Ensure input is in the right format (batch, timesteps, channels, height, width)
        if channels == 3:
            x = x.permute(0, 1, 4, 2, 3)  # (batch, timesteps, 3, height, width)

        # Normalize to ImageNet statistics expected by EfficientNet
        x = (x - self.imagenet_mean) / self.imagenet_std

        # Apply intermediate model with TimeDistributed
        x = self.time_distributed(x)  # (batch, timesteps, 1280, 5, 9)

        # Apply ConvLSTM2D layers
        # ConvLSTM 1 (optionally stateful)
        convlstm1_states_in = self._convlstm1_state if self._stateful else None
        x, convlstm1_states_out = self.convlstm1(x, states=convlstm1_states_in)
        if self._stateful:
            self._convlstm1_state = convlstm1_states_out

        if 'extra' in self.model_name:
            convlstm_extra_states_in = self._convlstm_extra_state if self._stateful else None
            x, convlstm_extra_states_out = self.convlstm_extra(x, states=convlstm_extra_states_in)
            if self._stateful:
                self._convlstm_extra_state = convlstm_extra_states_out

        # Flatten
        x = self.flatten(x)  # (batch, timesteps, convlstm_filters * 5 * 9)

        # Apply LSTM if specified
        if 'LSTM' in self.model_name:
            if self.lstm_dropout is not None:
                x = self.lstm_dropout(x)
            # Pack LSTM state if stateful; expects (h_0, c_0)
            lstm_state_in = self._lstm_state if self._stateful else None
            x, lstm_state_out = self.lstm(x, lstm_state_in)  # (batch, timesteps, 256)
            if self._stateful:
                self._lstm_state = lstm_state_out
            if self.lstm_output_dropout is not None:
                x = self.lstm_output_dropout(x)

        # Handle auxiliary input
        if self.aux_input_on and aux_input is not None:
            # Support either (batch, aux_dim) or (batch, timesteps, aux_dim)
            if aux_input.dim() == 2:
                aux_features = self.aux_dense(aux_input)  # (batch, 256)
                aux_features = aux_features.unsqueeze(1).repeat(1, timesteps, 1)  # (batch, timesteps, 256)
            elif aux_input.dim() == 3:
                b, t, d = aux_input.size()
                aux_flat = aux_input.contiguous().view(b * t, d)
                aux_proj = self.aux_dense(aux_flat)  # (b*t, 256)
                aux_features = aux_proj.view(b, t, 256)
            else:
                raise ValueError("aux_input must have shape (batch, aux_dim) or (batch, timesteps, aux_dim)")
            x = torch.cat([x, aux_features], dim=-1)  # (batch, timesteps, 256 + 256)

        # Apply shared dense layer
        x = self.shared_dense(x)  # (batch, timesteps, 256)

        # Generate outputs
        keys_out = self.sigmoid(self.keys_output(x))  # (batch, timesteps, n_keys)
        clicks_out = self.sigmoid(self.clicks_output(x))  # (batch, timesteps, n_clicks)
        mouse_x_out = self.softmax(self.mouse_x_output(x))  # (batch, timesteps, n_mouse_x)
        mouse_y_out = self.softmax(self.mouse_y_output(x))  # (batch, timesteps, n_mouse_y)
        value_out = self.value_output(x)  # (batch, timesteps, 1)

        return keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out

    # Stateful API
    def set_stateful(self, enabled: bool = True) -> None:
        self._stateful = bool(enabled)
        if not self._stateful:
            self.reset_states()

    def reset_states(self) -> None:
        self._convlstm1_state = None
        self._convlstm_extra_state = None
        self._lstm_state = None

    def get_output_concatenated(self, x: torch.Tensor, aux_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get concatenated output similar to the original TensorFlow implementation.

        Args:
            x: Input tensor
            aux_input: Optional auxiliary input

        Returns:
            Concatenated output tensor
        """
        keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = self.forward(x, aux_input)

        # Concatenate all outputs
        output = torch.cat([keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out], dim=-1)

        return output


# Factory function to create model based on configuration
def create_model(
    model_name: str = 'default',
    pretrained: bool = True,
    aux_input_on: bool = False,
    freeze_backbone: bool = False
) -> CSGOModel:
    """
    Create a CSGO model with the specified configuration.

    Args:
        model_name: Name of the model configuration
        pretrained: Whether to use pretrained weights
        aux_input_on: Whether to use auxiliary input

    Returns:
        CSGOModel instance
    """
    from config import (
        input_shape, n_keys, n_clicks, n_mouse_x, n_mouse_y,
        aux_input_length as cfg_aux_len, AUX_INPUT_ON
    )

    # If aux is enabled, use per-timestep previous action vector length
    use_aux = aux_input_on or AUX_INPUT_ON
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    aux_len = action_dim if use_aux else cfg_aux_len

    return CSGOModel(
        model_name=model_name,
        input_shape=input_shape,
        n_keys=n_keys,
        n_clicks=n_clicks,
        n_mouse_x=n_mouse_x,
        n_mouse_y=n_mouse_y,
        aux_input_length=aux_len,
        aux_input_on=use_aux,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )

if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model(model_name='default', pretrained=False, aux_input_on=True)
    print("✓ Model created successfully")

    # Test input shape: (batch, timesteps, height, width, channels)
    test_input = torch.randn(1, 96, 150, 280, 3)
    aux_dim = model.aux_input_length if model.aux_input_on else 0
    aux_input = torch.randn(1, aux_dim) if aux_dim > 0 else None

    # Forward pass
    output = model.get_output_concatenated(test_input, aux_input)
    print(f"✓ Forward pass successful, output shape: {output.shape}")

    # Check output dimensions match expected (batch, timesteps, 53)
    expected_features = 11 + 2 + 23 + 15 + 1  # keys + clicks + mouse_x + mouse_y + value
    assert output.shape == (1, 96, expected_features), f"Expected shape (1, 96, {expected_features}), got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")

    print("✓ All tests passed!")
    print(f"Model architecture:\n{model}")