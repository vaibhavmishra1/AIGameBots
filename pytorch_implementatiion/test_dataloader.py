#!/usr/bin/env python3
"""
Test script for the PyTorch CSGO DataLoader.
Demonstrates how to use the dataloader with PyTorch models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))
from dataloader import create_data_loaders, CSGODataLoader
import torch
import torch.nn as nn
import time


class SimpleCSGOModel(nn.Module):
    """
    Simple PyTorch model for CSGO behavioral cloning.
    This is a simplified version for demonstration.
    """

    def __init__(self, input_shape, output_size):
        super(SimpleCSGOModel, self).__init__()

        # CNN layers for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 7))  # Adjust to get consistent feature size
        )

        # Calculate CNN output size
        with torch.no_grad():
            # CNN expects (batch, channels, height, width), but our input is (batch, seq_len, height, width, channels)
            dummy_frame = torch.zeros(1, 3, input_shape[2], input_shape[3])  # (batch, channels, height, width)
            cnn_output = self.cnn(dummy_frame)
            cnn_output_size = cnn_output.numel()

        # LSTM for temporal processing
        self.lstm = nn.LSTM(cnn_output_size, 256, batch_first=True)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, height, width, channels = x.shape

        # Reshape to process all frames at once: (batch*seq_len, channels, height, width)
        x_reshaped = x.view(-1, channels, height, width)
        x_reshaped = x_reshaped.permute(0, 1, 2, 3)  # (batch*seq_len, channels, height, width)

        # Process all frames through CNN
        features = self.cnn(x_reshaped)  # (batch*seq_len, cnn_features)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq_len, cnn_features)

        # LSTM processing
        lstm_out, _ = self.lstm(features)

        # Take the last output of LSTM sequence
        lstm_last = lstm_out[:, -1, :]

        # Final output
        output = self.output_layers(lstm_last)

        return output


def test_dataloader():
    """Test the dataloader functionality."""
    print("Testing CSGO DataLoader...")

    # Create small data loaders for testing
    data_loaders = create_data_loaders(
        batch_size=2,
        starting_num=10,
        highest_num=12,  # Small range for testing
        shuffle=False,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=False
    )

    training_loader_full = data_loaders[8]  # training_loader_full
    validation_loader_full = data_loaders[9]  # validation_loader_full

    print(f"Training loader length: {len(training_loader_full)}")
    print(f"Validation loader length: {len(validation_loader_full)}")

    # Test loading batches
    print("\nTesting batch loading...")
    start_time = time.time()

    for i, (batch_x, batch_y) in enumerate(training_loader_full):
        print(f"Batch {i}: X shape: {batch_x.shape}, Y shape: {batch_y.shape}")
        print(f"  X range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
        print(f"  Y range: [{batch_y.min():.3f}, {batch_y.max():.3f}]")

        if i >= 2:  # Just test a few batches
            break

    print(f"Loading time: {time.time() - start_time:.3f} seconds")


def test_model_forward():
    """Test forward pass with a simple model."""
    print("\nTesting model forward pass...")

    from config import N_TIMESTEPS, csgo_img_dimension

    # Create a small dataset for testing
    data_list = [f"10-{i}" for i in range(0, 100, 10)][:5]  # Just 5 samples
    dataset = CSGODataLoader(
        data_list=data_list,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Get one batch
    for batch_x, batch_y in dataset:
        print(f"Input shape: {batch_x.shape}")
        print(f"Target shape: {batch_y.shape}")

        # Create model
        input_shape = batch_x.shape
        output_size = batch_y.shape[-1]

        model = SimpleCSGOModel(input_shape, output_size)
        model.eval()

        # Forward pass
        with torch.no_grad():
            output = model(batch_x)
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

        break


def test_training_loop():
    """Test a simple training loop."""
    print("\nTesting training loop...")

    # Import config values
    from config import N_TIMESTEPS, csgo_img_dimension, n_keys, n_clicks, n_mouse_x, n_mouse_y

    # Create small data loaders for testing
    data_loaders = create_data_loaders(
        batch_size=1,
        starting_num=10,
        highest_num=10,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    training_loader_full = data_loaders[8]

    # Create model
    input_shape = (1, N_TIMESTEPS, *csgo_img_dimension, 3)  # batch_size=1 for testing
    output_size = n_keys + n_clicks + n_mouse_x + n_mouse_y + 2  # +2 for reward and advantage

    model = SimpleCSGOModel(input_shape, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    num_epochs = 1
    num_batches = 3  # Just test a few batches

    print("Running training loop...")
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(training_loader_full):
            if batch_idx >= num_batches:
                break

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, -1, :])  # Use last timestep target

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    print("Training loop completed successfully!")


if __name__ == "__main__":
    print("CSGO DataLoader Test Script")
    print("=" * 40)

    try:
        test_dataloader()
        test_model_forward()
        test_training_loop()
        print("\nAll tests passed successfully! ✅")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
