import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FeedforwardNetwork(nn.Module):
    """
    A feedforward neural network for game action prediction.
    
    Input: 500-dimensional feature vector
    Output: Single scalar value for binary classification (0 or 1) or regression
    """
    
    def __init__(self, 
                 input_dim: int = 500,
                 hidden_dims: list = [512, 256, 128, 64],
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 task_type: str = 'classification'):
        """
        Initialize the feedforward network.
        
        Args:
            input_dim: Dimension of input features (default: 500)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            task_type: Type of task ('classification' or 'regression')
        """
        super(FeedforwardNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.task_type = task_type
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer - single neuron for both classification and regression
        layers.append(nn.Linear(prev_dim, 1))
        
        # For classification, we'll apply sigmoid in the forward pass
        # For regression, we'll return raw output
        if task_type == 'classification':
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function based on string name."""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions (only for classification tasks).
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        with torch.no_grad():
            return self.forward(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get predictions.
        
        Args:
            x: Input tensor
            threshold: Classification threshold (only for classification tasks)
            
        Returns:
            Predictions tensor of shape (batch_size, 1)
        """
        with torch.no_grad():
            if self.task_type == 'classification':
                probs = self.predict_proba(x)
                return (probs >= threshold).float()
            else:
                return self.forward(x)


class FeedforwardNetworkWithSkip(nn.Module):
    """
    A feedforward network with skip connections for better gradient flow.
    """
    
    def __init__(self, 
                 input_dim: int = 500,
                 hidden_dims: list = [512, 256, 128, 64],
                 dropout_rate: float = 0.3,
                 task_type: str = 'classification'):
        super(FeedforwardNetworkWithSkip, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.task_type = task_type
        
        # Build layers with skip connections
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # For classification, add sigmoid
        if task_type == 'classification':
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
        
        # Skip connection layers (if input_dim matches any hidden_dim)
        self.skip_connections = nn.ModuleList()
        for hidden_dim in hidden_dims:
            if hidden_dim == input_dim:
                self.skip_connections.append(nn.Linear(input_dim, hidden_dim))
            else:
                # Use a dummy layer that does nothing for non-matching dimensions
                self.skip_connections.append(nn.Identity())
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        current = x
        
        for i, layer in enumerate(self.layers):
            # Apply skip connection if available
            if isinstance(self.skip_connections[i], nn.Linear):
                skip = self.skip_connections[i](x)
                current = layer(current) + skip
            else:
                current = layer(current)
        
        output = self.output_layer(current)
        
        # Apply sigmoid for classification tasks
        if self.sigmoid is not None:
            output = self.sigmoid(output)
        
        return output


def create_model(model_type: str = 'standard', 
                input_dim: int = 500,
                hidden_dims: Optional[list] = None,
                dropout_rate: float = 0.3,
                task_type: str = 'classification') -> nn.Module:
    """
    Factory function to create different types of feedforward networks.
    
    Args:
        model_type: Type of model ('standard' or 'skip')
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Initialized model
    """
    if hidden_dims is None:
        hidden_dims = [512, 256, 128, 64]
    
    if model_type == 'standard':
        return FeedforwardNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            task_type=task_type
        )
    elif model_type == 'skip':
        return FeedforwardNetworkWithSkip(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            task_type=task_type
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_loss_function(task_type: str):
    """
    Get appropriate loss function based on task type.
    
    Args:
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Loss function
    """
    if task_type == 'classification':
        return nn.BCELoss()  # Binary Cross-Entropy for classification
    elif task_type == 'regression':
        return nn.MSELoss()  # Mean Squared Error for regression
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test classification model
    print("Testing Classification Model:")
    model_cls = create_model('standard', input_dim=500, task_type='classification')
    
    # Create dummy input
    batch_size = 32
    x = torch.randn(batch_size, 500)
    
    # Forward pass
    output = model_cls(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test predictions
    probs = model_cls.predict_proba(x)
    predictions = model_cls.predict(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Test regression model
    print("\nTesting Regression Model:")
    model_reg = create_model('standard', input_dim=500, task_type='regression')
    output_reg = model_reg(x)
    print(f"Regression output shape: {output_reg.shape}")
    
    # Test skip connection model
    print("\nTesting Skip Connection Model:")
    skip_model = create_model('skip', input_dim=500, task_type='classification')
    skip_output = skip_model(x)
    print(f"Skip model output shape: {skip_output.shape}")
