import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DualRegressionLoss(nn.Module):
    """
    Custom loss function for dual regression tasks.
    
    The model outputs 2 values, and we compute separate MSE losses for each output.
    Final loss can be a weighted combination of both losses.
    """
    
    def __init__(self, 
                 weight1: float = 1.0, 
                 weight2: float = 1.0,
                 reduction: str = 'mean'):
        """
        Initialize the dual regression loss.
        
        Args:
            weight1: Weight for the first regression loss
            weight2: Weight for the second regression loss
            reduction: Specifies the reduction to apply ('mean', 'sum', or 'none')
        """
        super(DualRegressionLoss, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the dual regression loss.
        
        Args:
            predictions: Model predictions of shape (batch_size, 2)
            targets: Target values of shape (batch_size, 2)
            
        Returns:
            Combined loss value
        """
        # Ensure inputs have correct shape
        assert predictions.shape[-1] == 2, f"Predictions must have 2 outputs, got {predictions.shape[-1]}"
        assert targets.shape[-1] == 2, f"Targets must have 2 outputs, got {targets.shape[-1]}"
        
        # Split predictions and targets
        pred1, pred2 = predictions[:, 0], predictions[:, 1]
        target1, target2 = targets[:, 0], targets[:, 1]
        
        # Compute individual MSE losses
        loss1 = F.mse_loss(pred1, target1, reduction=self.reduction)
        loss2 = F.mse_loss(pred2, target2, reduction=self.reduction)
        
        # Combine losses with weights
        total_loss = self.weight1 * loss1 + self.weight2 * loss2
        
        return total_loss
    
    def compute_individual_losses(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute and return individual losses along with the total loss.
        
        Args:
            predictions: Model predictions of shape (batch_size, 2)
            targets: Target values of shape (batch_size, 2)
            
        Returns:
            Tuple of (total_loss, loss1, loss2)
        """
        # Split predictions and targets
        pred1, pred2 = predictions[:, 0], predictions[:, 1]
        target1, target2 = targets[:, 0], targets[:, 1]
        
        # Compute individual MSE losses
        loss1 = F.mse_loss(pred1, target1, reduction=self.reduction)
        loss2 = F.mse_loss(pred2, target2, reduction=self.reduction)
        
        # Combine losses with weights
        total_loss = self.weight1 * loss1 + self.weight2 * loss2
        
        return total_loss, loss1, loss2


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
        layers.append(nn.Linear(prev_dim, 2))
        
        
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
            
            return self.forward(x)


def create_model(input_dim: int = 500,
                hidden_dims: Optional[list] = None,
                dropout_rate: float = 0.3):
    
    
    return FeedforwardNetwork(
            input_dim=input_dim,
            dropout_rate=dropout_rate,
            
        )


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

import os
import numpy as np
# Example usage and testing
if __name__ == "__main__":
    # Test classification model
    print("Testing Classification Model:")
    file_path = os.path.join("/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/clash_squad_agent_trajectories_features","trajectory_29381_features_48820.npy")
    features = np.load(file_path)
    print(features.shape)
    features = features.reshape(1, -1)
    print(features.shape)
    # Broadcast features to match batch size
    features = np.tile(features, (32, 1))
    print(f"Broadcasted features shape: {features.shape}")
    features = torch.tensor(features,  dtype=torch.float32)
    # features = np.reshape(features, (1, 160))
    
    model_cls = create_model( input_dim=360)
    
    # Forward pass
    predictions = model_cls(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Create dummy targets for testing (batch_size, 2)
    batch_size = features.shape[0]
    targets = torch.randn(batch_size, 2, dtype=torch.float32)
    print(f"Targets shape: {targets.shape}")
    
    # Test custom loss function
    print("\n=== Testing Custom Dual Regression Loss ===")
    
    # Initialize custom loss with equal weights
    dual_loss = DualRegressionLoss(weight1=1.0, weight2=1.0)
    
    # Compute loss
    total_loss = dual_loss(predictions, targets)
    print(f"Total combined loss: {total_loss.item():.4f}")
    
    # Get individual losses for monitoring
    total_loss_detailed, loss1, loss2 = dual_loss.compute_individual_losses(predictions, targets)
    print(f"Loss 1 (first output): {loss1.item():.4f}")
    print(f"Loss 2 (second output): {loss2.item():.4f}")
    print(f"Combined loss (detailed): {total_loss_detailed.item():.4f}")
    
    # Test with different weights
    print("\n=== Testing with Different Weights ===")
    weighted_loss = DualRegressionLoss(weight1=2.0, weight2=0.5)
    weighted_total = weighted_loss(predictions, targets)
    print(f"Weighted loss (2.0, 0.5): {weighted_total.item():.4f}")
    
    print("\n=== Usage Example for Training ===")
    print("# In your training loop:")
    print("# loss_fn = DualRegressionLoss(weight1=1.0, weight2=1.0)")
    print("# predictions = model(batch_features)  # shape: (batch_size, 2)")
    print("# targets = batch_targets  # shape: (batch_size, 2)")
    print("# loss = loss_fn(predictions, targets)")
    print("# loss.backward()")
    
