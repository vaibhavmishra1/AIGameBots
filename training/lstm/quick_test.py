#!/usr/bin/env python3
"""
Quick test script to verify LSTM training setup works correctly.
"""

import torch
import numpy as np
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from dataset import AgentDataset, create_dataloaders
        from model import create_model, AgentLSTM, AgentLSTMWithUncertainty
        from train import Trainer, ActionLoss
        from inference import AgentInference
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dataset(data_dir: str):
    """Test dataset loading."""
    print("\nTesting dataset...")
    try:
        from dataset import AgentDataset
        
        # Try to load a small subset
        dataset = AgentDataset(
            data_dir=data_dir,
            max_samples=10,
            normalize=True
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test loading a sample
        (features, prev_actions), target = dataset[0]
        print(f"✓ Sample shapes - Features: {features.shape}, Prev actions: {prev_actions.shape}, Target: {target.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def test_model():
    """Test model creation and forward pass."""
    print("\nTesting model...")
    try:
        from model import create_model
        
        # Test standard model
        model = create_model(
            feature_dim=20,
            action_dim=4,
            hidden_dim=128,
            num_lstm_layers=2,
            model_type="standard"
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 5
        features = torch.randn(batch_size, seq_len, 20)
        prev_actions = torch.randn(batch_size, seq_len - 1, 4)
        
        actions, hidden = model(features, prev_actions)
        print(f"✓ Standard model output shape: {actions.shape}")
        
        # Test uncertainty model
        model_unc = create_model(
            feature_dim=20,
            action_dim=4,
            hidden_dim=128,
            num_lstm_layers=2,
            model_type="uncertainty"
        )
        
        actions, uncertainties, hidden = model_unc(features, prev_actions)
        print(f"✓ Uncertainty model output shapes - Actions: {actions.shape}, Uncertainties: {uncertainties.shape}")
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_training_step(data_dir: str):
    """Test a single training step."""
    print("\nTesting training step...")
    try:
        from dataset import create_dataloaders
        from model import create_model
        from train import ActionLoss
        
        # Create small dataloader
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            max_samples=10,
            num_workers=0
        )
        
        # Create model and loss
        model = create_model(
            feature_dim=20,
            action_dim=4,
            hidden_dim=64,
            num_lstm_layers=2
        )
        
        criterion = ActionLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Get one batch
        (features, prev_actions), targets = next(iter(train_loader))
        
        # Forward pass
        predictions, _ = model(features, prev_actions)
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful - Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False

def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) available")
        return True
    elif torch.cuda.is_available():
        print(f"✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("✗ No GPU available - will use CPU (training will be slower)")
        return False

def main():
    """Run all tests."""
    print("LSTM Agent Setup Test")
    print("=" * 50)
    
    # Get data directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/clash_squad_partitioned_features_chunked"
    
    print(f"Data directory: {data_dir}")
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"\n✗ Data directory not found: {data_dir}")
        print("Please provide the correct path as an argument:")
        print("  python quick_test.py /path/to/dataset")
        return
    
    # Run tests
    tests = [
        ("Imports", test_imports, []),
        ("GPU", check_gpu, []),
        ("Dataset", test_dataset, [data_dir]),
        ("Model", test_model, []),
        ("Training", test_training_step, [data_dir])
    ]
    
    results = []
    for name, test_func, args in tests:
        try:
            success = test_func(*args)
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("-" * 50)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:<20} {status}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Start training with default settings:")
        print(f"   python train.py --data_dir {data_dir} --max_samples 10000")
        print("\n2. Monitor training progress:")
        print("   tensorboard --logdir runs/lstm_agent")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("Common issues:")
        print("- Missing dependencies: pip install -r requirements.txt")
        print("- Wrong data path: Check that features/ and actions/ folders exist")
        print("- Corrupted data files: Try loading them manually with numpy")

if __name__ == "__main__":
    main() 