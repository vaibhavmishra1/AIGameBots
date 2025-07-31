import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for better temporal understanding."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and reshape
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and final linear
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_linear(context)
        
        return output


class AgentLSTM(nn.Module):
    """Advanced LSTM model for agent imitation learning."""
    
    def __init__(
        self,
        feature_dim: int = 20,
        action_dim: int = 4,
        hidden_dim: int = 256,
        num_lstm_layers: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True,
        use_residual: bool = True,
        activation: str = "gelu"
    ):
        """
        Initialize the LSTM model.
        
        Args:
            feature_dim: Dimension of input features
            action_dim: Dimension of actions (4 for movement and rotation)
            hidden_dim: Hidden dimension for LSTM and other layers
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            use_residual: Whether to use residual connections
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Feature processing
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation
        )
        
        # Action embedding (for previous actions)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            self.activation,
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation
        )
        
        # Combined input dimension (features + previous actions)
        self.combined_dim = hidden_dim * 2
        
        # Input projection
        self.input_projection = nn.Linear(self.combined_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = MultiHeadAttention(hidden_dim, num_heads=8, dropout=dropout)
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        ])
        
        # Action-specific heads for better specialization
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(action_dim, 32),
                self.activation,
                nn.Linear(32, 1)
            ) for _ in range(action_dim)
        ])
        
        # Normalization layers
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        features: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            features: Input features [batch_size, seq_len, feature_dim]
            prev_actions: Previous actions [batch_size, seq_len-1, action_dim]
            hidden_state: Previous LSTM hidden state (optional)
            
        Returns:
            actions: Predicted actions [batch_size, action_dim]
            hidden_state: Updated LSTM hidden state
        """
        batch_size, seq_len, _ = features.shape
        
        # Encode features
        encoded_features = self.feature_encoder(features)  # [batch, seq_len, hidden_dim]
        
        # Process previous actions
        if prev_actions.shape[1] > 0:
            # Pad previous actions to match sequence length
            padded_actions = torch.zeros(
                batch_size, seq_len, self.action_dim,
                dtype=prev_actions.dtype, device=prev_actions.device
            )
            padded_actions[:, :prev_actions.shape[1]] = prev_actions
            
            encoded_actions = self.action_encoder(padded_actions)  # [batch, seq_len, hidden_dim]
        else:
            # If no previous actions, use zeros
            encoded_actions = torch.zeros_like(encoded_features)
        
        # Combine features and actions
        combined = torch.cat([encoded_features, encoded_actions], dim=-1)  # [batch, seq_len, hidden_dim*2]
        combined = self.input_projection(combined)  # [batch, seq_len, hidden_dim]
        
        # Apply residual connection if enabled
        if self.use_residual:
            residual = combined
        
        # LSTM processing
        lstm_out, hidden_state = self.lstm(combined, hidden_state)  # [batch, seq_len, hidden_dim]
        lstm_out = self.lstm_norm(lstm_out)
        
        # Add residual connection
        if self.use_residual:
            lstm_out = lstm_out + residual
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out = self.attention(lstm_out)
            attn_out = self.attention_norm(attn_out + lstm_out)  # Residual connection
            sequence_output = attn_out
        else:
            sequence_output = lstm_out
        
        # Use the last timestep output for action prediction
        final_hidden = sequence_output[:, -1, :]  # [batch, hidden_dim]
        
        # Generate base actions
        base_actions = self.output_layers[0](final_hidden)  # [batch, action_dim]
        
        # Apply action-specific heads for refinement
        refined_actions = []
        for i, head in enumerate(self.action_heads):
            action_input = base_actions.clone()
            refined_action = head(action_input).squeeze(-1)
            refined_actions.append(refined_action)
        
        actions = torch.stack(refined_actions, dim=1)  # [batch, action_dim]
        
        # Apply tanh to ensure output is in [-1, 1] range (will be denormalized later)
        actions = torch.tanh(actions)
        
        return actions, hidden_state


class AgentLSTMWithUncertainty(AgentLSTM):
    """LSTM model with uncertainty estimation for better decision making."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add uncertainty estimation heads
        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                self.activation,
                nn.Linear(64, 1),
                nn.Softplus()  # Ensure positive uncertainty
            ) for _ in range(self.action_dim)
        ])
    
    def forward(
        self,
        features: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.
        
        Returns:
            actions: Predicted actions [batch_size, action_dim]
            uncertainties: Uncertainty estimates [batch_size, action_dim]
            hidden_state: Updated LSTM hidden state
        """
        # Get base predictions
        actions, hidden_state = super().forward(features, prev_actions, hidden_state)
        
        # Get final hidden state for uncertainty estimation
        batch_size, seq_len, _ = features.shape
        
        # Re-encode features for uncertainty path
        encoded_features = self.feature_encoder(features)
        if prev_actions.shape[1] > 0:
            padded_actions = torch.zeros(
                batch_size, seq_len, self.action_dim,
                dtype=prev_actions.dtype, device=prev_actions.device
            )
            padded_actions[:, :prev_actions.shape[1]] = prev_actions
            encoded_actions = self.action_encoder(padded_actions)
        else:
            encoded_actions = torch.zeros_like(encoded_features)
        
        combined = torch.cat([encoded_features, encoded_actions], dim=-1)
        combined = self.input_projection(combined)
        lstm_out, _ = self.lstm(combined, hidden_state)
        
        final_hidden = lstm_out[:, -1, :]
        
        # Estimate uncertainties
        uncertainties = []
        for i, head in enumerate(self.uncertainty_heads):
            uncertainty = head(final_hidden).squeeze(-1)
            uncertainties.append(uncertainty)
        
        uncertainties = torch.stack(uncertainties, dim=1)  # [batch, action_dim]
        
        return actions, uncertainties, hidden_state


def create_model(
    feature_dim: int = 20,
    action_dim: int = 4,
    model_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Create a model instance.
    
    Args:
        feature_dim: Dimension of input features
        action_dim: Dimension of actions
        model_type: Type of model ('standard' or 'uncertainty')
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    if model_type == "standard":
        return AgentLSTM(feature_dim, action_dim, **kwargs)
    elif model_type == "uncertainty":
        return AgentLSTMWithUncertainty(feature_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    model = create_model(
        feature_dim=20,
        action_dim=4,
        hidden_dim=256,
        num_lstm_layers=3,
        use_attention=True,
        model_type="uncertainty"
    )
    
    # Test input
    batch_size = 32
    seq_len = 5
    features = torch.randn(batch_size, seq_len, 20)
    prev_actions = torch.randn(batch_size, seq_len - 1, 4)
    
    # Forward pass
    if isinstance(model, AgentLSTMWithUncertainty):
        actions, uncertainties, hidden = model(features, prev_actions)
        print(f"Actions shape: {actions.shape}")
        print(f"Uncertainties shape: {uncertainties.shape}")
    else:
        actions, hidden = model(features, prev_actions)
        print(f"Actions shape: {actions.shape}")
    
    print(f"Hidden state shapes: {hidden[0].shape}, {hidden[1].shape}")
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
