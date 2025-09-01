import torch
import torch.nn as nn

class SimpleFeedForward(nn.Module):
    """
    Simple feedforward model for regression.
    Input: (batch_size, 6)
    Output: (batch_size, 2), each in [-1, 1]
    """
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # ensures output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# Example usage:
if __name__ == "__main__":
    model = SimpleFeedForward()
    x = torch.randn(8, 6)  # batch of 8
    y_pred = model(x)
    print("Output shape:", y_pred.shape)
    print("Output sample:", y_pred[0])
    # Example target and loss
    y_true = torch.randn(8, 2).clamp(-1, 1)
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y_true)
    print("Sample loss:", loss.item())
