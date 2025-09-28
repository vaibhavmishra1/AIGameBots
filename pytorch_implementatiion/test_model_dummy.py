#!/usr/bin/env python3
"""
Test script to run the PyTorch CSGO model with dummy input data.
This allows testing the model without needing Counter-Strike running.
"""

import os
import sys
import numpy as np
import torch
import time

# Add Counter-Strike_Behavioural_Cloning to path for config imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))
from config import *

# Import the model
from model import create_model

# Import PyTorch
import torch

# We'll define our own simplified TorchAgent and onehot_to_actions to avoid Windows dependencies

def get_device():
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TorchAgent:
    """
    Simplified wrapper around the PyTorch model for testing.
    """

    def __init__(self, checkpoint_path: str, model_name: str = 'default', aux_input_on: bool = False, pretrained: bool = False, stateful: bool = True):
        self.device = get_device()
        self.model = create_model(model_name=model_name, pretrained=pretrained, aux_input_on=aux_input_on)
        self.model.to(self.device)
        self.model.eval()
        # enable stateful sequence processing if requested
        try:
            self.model.set_stateful(stateful)
        except Exception:
            pass
        self._load_weights(checkpoint_path)

    def _load_weights(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt  # assume plain state_dict
        # Load with strict=False to tolerate minor key mismatches
        self.model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def predict_on_batch(self, x_input_main_np: np.ndarray) -> np.ndarray:
        # Expect shape (1, 1, H, W, 3)
        x = torch.from_numpy(x_input_main_np).float()
        if x.max() > 1.0:
            x = x / 255.0
        x = x.to(self.device)

        keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = self.model(x)

        # Use last timestep (we provide T=1)
        keys = keys_out[:, -1, :]
        clicks = clicks_out[:, -1, :]
        mouse_x = mouse_x_out[:, -1, :]
        mouse_y = mouse_y_out[:, -1, :]
        value = value_out[:, -1, :]

        y = torch.cat([keys, clicks, mouse_x, mouse_y, value], dim=-1)  # shape (1, F)
        return y.detach().cpu().numpy()

    def reset_states(self) -> None:
        try:
            self.model.reset_states()
        except Exception:
            pass


def onehot_to_actions(y_preds):
    """
    Convert model predictions to actions.
    Returns: [keys_pressed, mouse_x, mouse_y, Lclicks, Rclicks, val_pred]
    """
    y_preds = y_preds.squeeze()

    keys_pred = y_preds[0:n_keys]
    Lclicks_pred = y_preds[n_keys:n_keys+1]
    Rclicks_pred = y_preds[n_keys+1:n_keys+n_clicks]
    mouse_x_pred = y_preds[n_keys+n_clicks:n_keys+n_clicks+n_mouse_x]
    mouse_y_pred = y_preds[n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y]
    val_pred = 0.0
    if y_preds.shape[0] >= (n_keys + n_clicks + n_mouse_x + n_mouse_y + 1):
        val_pred = y_preds[n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1][0]

    keys_pressed = []
    keys_pressed_onehot = np.round(keys_pred)
    if keys_pressed_onehot[0] == 1:
        keys_pressed.append('w')
    if keys_pressed_onehot[1] == 1:
        keys_pressed.append('a')
    if keys_pressed_onehot[2] == 1:
        keys_pressed.append('s')
    if keys_pressed_onehot[3] == 1:
        keys_pressed.append('d')
    if keys_pressed_onehot[4] == 1:
        keys_pressed.append('space')
    if keys_pressed_onehot[5] == 1:
        keys_pressed.append('ctrl')
    if keys_pressed_onehot[6] == 1:
        keys_pressed.append('shift')
    if keys_pressed_onehot[7] == 1:
        keys_pressed.append('1')
    if keys_pressed_onehot[8] == 1:
        keys_pressed.append('2')
    if keys_pressed_onehot[9] == 1:
        keys_pressed.append('3')
    if keys_pressed_onehot[10] == 1:
        keys_pressed.append('r')

    Lclicks = int(np.round(Lclicks_pred[0]))
    Rclicks = int(np.round(Rclicks_pred[0]))

    id_x = int(np.argmax(mouse_x_pred))
    mouse_x = mouse_x_possibles[id_x]
    id_y = int(np.argmax(mouse_y_pred))
    mouse_y = mouse_y_possibles[id_y]

    return [keys_pressed, mouse_x, mouse_y, Lclicks, Rclicks, float(val_pred)]


def create_dummy_image(height=150, width=280, channels=3):
    """
    Create a dummy CSGO-like image with some realistic patterns.
    
    Args:
        height: Image height (default 150 from config)
        width: Image width (default 280 from config)  
        channels: Number of channels (default 3 for RGB)
    
    Returns:
        numpy array of shape (height, width, channels) with values 0-255
    """
    # Create base image with gradient background (sky-like)
    img = np.zeros((height, width, channels), dtype=np.uint8)
    
    # Sky gradient (top part)
    for y in range(height // 3):
        intensity = int(135 + (y * 50) / (height // 3))  # Sky blue gradient
        img[y, :, 0] = intensity - 30  # R
        img[y, :, 1] = intensity - 10  # G  
        img[y, :, 2] = intensity       # B
    
    # Ground/walls (middle and bottom)
    for y in range(height // 3, height):
        base_intensity = int(80 + np.random.normal(0, 10))
        base_intensity = np.clip(base_intensity, 50, 120)
        
        # Add some horizontal variation (walls/structures)
        for x in range(width):
            variation = int(np.random.normal(0, 15))
            intensity = np.clip(base_intensity + variation, 20, 200)
            
            img[y, x, 0] = intensity + np.random.randint(-10, 10)  # R
            img[y, x, 1] = intensity + np.random.randint(-10, 10)  # G
            img[y, x, 2] = intensity + np.random.randint(-10, 10)  # B
    
    # Add some rectangular "structures" (buildings, walls)
    num_rects = np.random.randint(2, 6)
    for _ in range(num_rects):
        x1 = np.random.randint(0, width - 50)
        y1 = np.random.randint(height // 2, height - 30)
        w = np.random.randint(20, 80)
        h = np.random.randint(15, 40)
        x2 = min(x1 + w, width)
        y2 = min(y1 + h, height)
        
        color = np.random.randint(60, 140, 3)
        img[y1:y2, x1:x2] = color
    
    # Add crosshair-like center mark
    center_x, center_y = width // 2, height // 2
    img[center_y-2:center_y+3, center_x-10:center_x+11] = [255, 255, 255]
    img[center_y-10:center_y+11, center_x-2:center_x+3] = [255, 255, 255]
    
    return img


def test_model_with_dummy_data():
    """
    Test the model with dummy input data.
    """
    print("🎮 Testing CSGO PyTorch Model with Dummy Data")
    print("=" * 50)
    
    # Check if checkpoint exists
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'default_best.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        else:
            print("  No checkpoint directory found")
        return
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # Create dummy image data
    print(f"\n📸 Creating dummy image data...")
    print(f"Expected input shape: {input_shape_lstm_pred}")  # From config
    
    # Create dummy image matching expected dimensions
    dummy_img = create_dummy_image(
        height=csgo_img_dimension[0],  # 150
        width=csgo_img_dimension[1],   # 280
        channels=3
    )
    
    # Prepare input in expected format: (batch=1, timestep=1, H, W, C)
    x_input_main = np.expand_dims(dummy_img, axis=0)  # Add batch dim: (1, H, W, C)
    x_input_main = np.expand_dims(x_input_main, axis=0)  # Add timestep dim: (1, 1, H, W, C)
    
    print(f"Dummy input shape: {x_input_main.shape}")
    print(f"Dummy input range: [{x_input_main.min()}, {x_input_main.max()}]")
    
    # Initialize the model
    print(f"\n🤖 Loading PyTorch model...")
    try:
        model_agent = TorchAgent(
            checkpoint_path=checkpoint_path,
            model_name='default',
            aux_input_on=False,
            pretrained=False,
            stateful=True
        )
        print("✅ Model loaded successfully!")
        
        # Get device info
        device = model_agent.device
        print(f"🔧 Using device: {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run prediction
    print(f"\n🔮 Running model prediction...")
    try:
        start_time = time.time()
        y_preds = model_agent.predict_on_batch(x_input_main)
        inference_time = time.time() - start_time
        
        print(f"✅ Prediction successful!")
        print(f"⏱️  Inference time: {inference_time:.4f} seconds")
        print(f"📊 Raw output shape: {y_preds.shape}")
        print(f"📊 Raw output range: [{y_preds.min():.4f}, {y_preds.max():.4f}]")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return
    
    # Decode predictions to actions
    print(f"\n🎯 Decoding predictions to actions...")
    try:
        actions = onehot_to_actions(y_preds)
        keys_pressed, mouse_x, mouse_y, left_click, right_click, value_pred = actions
        
        print("✅ Actions decoded successfully!")
        print("\n🎮 PREDICTED ACTIONS:")
        print("-" * 30)
        print(f"🔤 Keys pressed: {keys_pressed if keys_pressed else 'None'}")
        print(f"🖱️  Mouse X: {mouse_x}")
        print(f"🖱️  Mouse Y: {mouse_y}")
        print(f"🔫 Left click: {'Yes' if left_click else 'No'}")
        print(f"🔫 Right click: {'Yes' if right_click else 'No'}")
        print(f"💰 Value prediction: {value_pred:.4f}")
        
    except Exception as e:
        print(f"❌ Error decoding actions: {e}")
        return
    
    # Show detailed breakdown of raw predictions
    print(f"\n📈 DETAILED PREDICTION BREAKDOWN:")
    print("-" * 40)
    
    y_flat = y_preds.squeeze()
    
    # Keys (11 outputs)
    keys_pred = y_flat[0:n_keys]
    print(f"🔤 Key probabilities:")
    key_names = ['W', 'A', 'S', 'D', 'Space', 'Ctrl', 'Shift', '1', '2', '3', 'R']
    for i, (key, prob) in enumerate(zip(key_names, keys_pred)):
        status = "✓" if prob > 0.5 else " "
        print(f"   {status} {key:5}: {prob:.3f}")
    
    # Clicks (2 outputs)
    clicks_pred = y_flat[n_keys:n_keys+n_clicks]
    print(f"\n🔫 Click probabilities:")
    print(f"   {'✓' if clicks_pred[0] > 0.5 else ' '} Left : {clicks_pred[0]:.3f}")
    print(f"   {'✓' if clicks_pred[1] > 0.5 else ' '} Right: {clicks_pred[1]:.3f}")
    
    # Mouse movement
    mouse_x_pred = y_flat[n_keys+n_clicks:n_keys+n_clicks+n_mouse_x]
    mouse_y_pred = y_flat[n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y]
    
    print(f"\n🖱️  Mouse X distribution (top 5):")
    top_x_indices = np.argsort(mouse_x_pred)[-5:][::-1]
    for idx in top_x_indices:
        print(f"   {mouse_x_possibles[idx]:6.1f}: {mouse_x_pred[idx]:.3f}")
    
    print(f"\n🖱️  Mouse Y distribution (top 5):")
    top_y_indices = np.argsort(mouse_y_pred)[-5:][::-1]
    for idx in top_y_indices:
        print(f"   {mouse_y_possibles[idx]:6.1f}: {mouse_y_pred[idx]:.3f}")
    
    # Value prediction (if available)
    if len(y_flat) > n_keys + n_clicks + n_mouse_x + n_mouse_y:
        value_raw = y_flat[n_keys+n_clicks+n_mouse_x+n_mouse_y]
        print(f"\n💰 Value function: {value_raw:.4f}")
    
    print(f"\n🎉 Test completed successfully!")
    print(f"🔧 Model appears to be working correctly with dummy data.")
    

def test_multiple_predictions():
    """
    Test the model with multiple different dummy images to see variety in predictions.
    """
    print("\n" + "="*60)
    print("🔄 Testing Multiple Predictions")
    print("="*60)
    
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'default_best.pt')
    
    if not os.path.exists(checkpoint_path):
        print("❌ Checkpoint not found, skipping multiple prediction test")
        return
    
    # Load model
    model_agent = TorchAgent(
        checkpoint_path=checkpoint_path,
        model_name='default',
        aux_input_on=False,
        pretrained=False,
        stateful=True
    )
    
    print("Testing 5 different dummy images...")
    
    for i in range(5):
        print(f"\n🖼️  Test Image {i+1}:")
        print("-" * 20)
        
        # Create different dummy image each time
        dummy_img = create_dummy_image()
        x_input = np.expand_dims(np.expand_dims(dummy_img, axis=0), axis=0)
        
        # Predict
        y_preds = model_agent.predict_on_batch(x_input)
        actions = onehot_to_actions(y_preds)
        keys_pressed, mouse_x, mouse_y, left_click, right_click, value_pred = actions
        
        # Show summary
        keys_str = ', '.join(keys_pressed) if keys_pressed else 'None'
        print(f"Keys: {keys_str}")
        print(f"Mouse: ({mouse_x}, {mouse_y})")
        print(f"Clicks: L={left_click}, R={right_click}")
        print(f"Value: {value_pred:.3f}")
        
        # Reset states between predictions to see variety
        if i % 2 == 1:
            model_agent.reset_states()


if __name__ == "__main__":
    print("🚀 Starting CSGO Model Test Suite")
    
    # Test basic functionality
    test_model_with_dummy_data()
    
    # Test multiple predictions
    test_multiple_predictions()
    
    print("\n✨ All tests completed!")
