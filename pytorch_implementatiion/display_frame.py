#!/usr/bin/env python3
"""
Simple script to display the first frame from the first batch.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))
from dataloader import create_data_loaders
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def display_first_frame():
    """Display the first frame from the first batch."""
    print("Loading CSGO dataset...")

    # Create a small data loader for testing
    data_loaders = create_data_loaders(
        batch_size=16,
        starting_num=10,
        highest_num=10,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    training_loader_full = data_loaders[0]  # training_loader_full

    print("Loading first batch...")

    # Get the first batch
    for batch_x, batch_y in training_loader_full:
        print(f"✓ Batch X shape: {batch_x.shape}")
        print(f"✓ Batch Y shape: {batch_y.shape}")

        # Extract first batch, first frame: (16, 96, 150, 280, 3) -> (150, 280, 3)
        first_frame = batch_x[1, 0]  # Shape: (150, 280, 3)
        print(f"✓ First frame shape: {first_frame.shape}")
        print(f"✓ First frame range: [{first_frame.min():.3f}, {first_frame.max():.3f}]")

        # Convert to numpy for display
        frame_np = first_frame.numpy()

        # Convert from BGR to RGB for correct colors (OpenCV stores as BGR)
        frame_np = frame_np[:, :, [2, 1, 0]]  # BGR -> RGB

        # Normalize to [0, 1] for display
        frame_np = frame_np / 255.0

        print("Creating visualization...")

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Display original frame
        ax1.imshow(frame_np)
        ax1.set_title(f'First Frame (Shape: {frame_np.shape})')
        ax1.axis('off')

        # Display frame info
        ax2.text(0.1, 0.9, f'Shape: {frame_np.shape}', fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.8, f'Min: {frame_np.min():.3f}', fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.7, f'Max: {frame_np.max():.3f}', fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.6, f'Mean: {frame_np.mean():.3f}', fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.5, 'Data Type: Game Screenshot', fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.4, 'Format: RGB (150×280×3)', fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.3, 'Batch: 0/16', fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.2, 'Frame: 0/96', fontsize=12, transform=ax2.transAxes)
        ax2.axis('off')
        ax2.set_title('Frame Information')

        plt.tight_layout()
        plt.savefig('first_frame_display.png', dpi=150, bbox_inches='tight')

        print("✅ Frame saved as 'first_frame_display.png'")
        print("📁 To view the image, open 'first_frame_display.png' in your image viewer")

        # Print some additional information
        print("\n📊 Frame Statistics:")
        print(f"   Shape: {frame_np.shape}")
        print(f"   Data Type: {frame_np.dtype}")
        print(f"   Range: [{frame_np.min():.3f}, {frame_np.max():.3f}]")
        print(f"   Mean: {frame_np.mean():.3f}")
        print(f"   Std: {frame_np.std():.3f}")

        break

if __name__ == "__main__":
    print("🎮 CSGO DataLoader - First Frame Display")
    print("=" * 45)

    try:
        display_first_frame()
        print("✅ Success!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
