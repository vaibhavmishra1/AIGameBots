import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import time

from model import create_model, AgentLSTMWithUncertainty
from dataset import AgentDataset


class AgentInference:
    """Inference class for LSTM agent model."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        # Device selection with MPS support
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # Create model
        self.model = create_model(
            feature_dim=self.config['feature_dim'],
            action_dim=self.config['action_dim'],
            model_type=self.config['model_type'],
            hidden_dim=self.config['hidden_dim'],
            num_lstm_layers=self.config['num_lstm_layers'],
            dropout=0.0,  # No dropout during inference
            use_attention=self.config['use_attention'],
            use_residual=self.config['use_residual'],
            activation=self.config['activation']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize normalization stats (you might need to save these during training)
        self._load_normalization_stats()
        
        # Hidden state for sequential prediction
        self.hidden_state = None
        
        # Action history buffer
        self.action_history = []
        self.max_history = self.config['use_history'] - 1
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model type: {self.config['model_type']}")
        print(f"Hidden dim: {self.config['hidden_dim']}")
        print(f"LSTM layers: {self.config['num_lstm_layers']}")
    
    def _load_normalization_stats(self):
        """Load normalization statistics."""
        # Try to load saved stats
        stats_path = Path(self.config['checkpoint_dir']) / 'normalization_stats.npz'
        
        if stats_path.exists():
            stats = np.load(stats_path)
            self.feature_mean = stats['feature_mean']
            self.feature_std = stats['feature_std']
            self.action_ranges = stats['action_ranges']
        else:
            # Default values if stats not found
            print("Warning: Normalization stats not found, using defaults")
            self.feature_mean = np.zeros(self.config['feature_dim'])
            self.feature_std = np.ones(self.config['feature_dim'])
            self.action_ranges = np.array([
                [-1, 1],   # movedirection_x
                [-1, 1],   # movedirection_z
                [-10, 10], # lookrotation_x
                [-10, 10]  # lookrotation_z
            ])
    
    def reset(self):
        """Reset the model state for a new episode."""
        self.hidden_state = None
        self.action_history = []
    
    def predict(
        self,
        features: np.ndarray,
        return_uncertainty: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict next action given current features.
        
        Args:
            features: Current features [feature_dim] or [seq_len, feature_dim]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [action_dim]
                - 'uncertainties': Uncertainty estimates (if requested and available)
        """
        with torch.no_grad():
            # Prepare features
            if features.ndim == 1:
                # Single frame, need to create sequence
                seq_len = self.config['use_history']
                feature_seq = np.zeros((seq_len, self.config['feature_dim']))
                feature_seq[-1] = features  # Put current features at the end
            else:
                feature_seq = features
            
            # Normalize features
            if self.config.get('normalize', True):
                feature_seq = (feature_seq - self.feature_mean) / self.feature_std
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(feature_seq).unsqueeze(0).to(self.device)
            
            # Prepare previous actions
            if len(self.action_history) > 0:
                prev_actions = np.array(self.action_history[-self.max_history:])
                # Normalize previous actions
                if self.config.get('normalize', True):
                    for i in range(self.config['action_dim']):
                        action_range = self.action_ranges[i]
                        prev_actions[:, i] = 2 * (prev_actions[:, i] - action_range[0]) / (action_range[1] - action_range[0]) - 1
                
                # Pad if needed
                if len(prev_actions) < self.max_history:
                    padding = np.zeros((self.max_history - len(prev_actions), self.config['action_dim']))
                    prev_actions = np.vstack([padding, prev_actions])
            else:
                prev_actions = np.zeros((self.max_history, self.config['action_dim']))
            
            prev_actions_tensor = torch.FloatTensor(prev_actions).unsqueeze(0).to(self.device)
            
            # Forward pass
            if isinstance(self.model, AgentLSTMWithUncertainty) and return_uncertainty:
                actions, uncertainties, self.hidden_state = self.model(
                    feature_tensor, prev_actions_tensor, self.hidden_state
                )
                uncertainties = uncertainties.cpu().numpy()[0]
            else:
                actions, self.hidden_state = self.model(
                    feature_tensor, prev_actions_tensor, self.hidden_state
                )
                uncertainties = None
            
            # Convert actions to numpy
            actions = actions.cpu().numpy()[0]
            
            # Denormalize actions
            if self.config.get('normalize', True):
                for i in range(self.config['action_dim']):
                    action_range = self.action_ranges[i]
                    actions[i] = (actions[i] + 1) * (action_range[1] - action_range[0]) / 2 + action_range[0]
            
            # Clip actions to valid ranges
            actions[0] = np.clip(actions[0], -1, 1)    # movedirection_x
            actions[1] = np.clip(actions[1], -1, 1)    # movedirection_z
            actions[2] = np.clip(actions[2], -10, 10)  # lookrotation_x
            actions[3] = np.clip(actions[3], -10, 10)  # lookrotation_z
            
            # Update action history
            self.action_history.append(actions.copy())
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)
            
            result = {'actions': actions}
            if uncertainties is not None:
                result['uncertainties'] = uncertainties
            
            return result
    
    def predict_batch(
        self,
        features_batch: np.ndarray,
        prev_actions_batch: Optional[np.ndarray] = None,
        return_uncertainty: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict actions for a batch of inputs.
        
        Args:
            features_batch: Batch of features [batch_size, seq_len, feature_dim]
            prev_actions_batch: Batch of previous actions [batch_size, seq_len-1, action_dim]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                - 'uncertainties': Uncertainty estimates (if requested and available)
        """
        with torch.no_grad():
            batch_size = features_batch.shape[0]
            
            # Normalize features
            if self.config.get('normalize', True):
                features_batch = (features_batch - self.feature_mean) / self.feature_std
            
            # Convert to tensors
            feature_tensor = torch.FloatTensor(features_batch).to(self.device)
            
            if prev_actions_batch is None:
                prev_actions_batch = np.zeros((batch_size, self.max_history, self.config['action_dim']))
            
            # Normalize previous actions
            if self.config.get('normalize', True):
                for i in range(self.config['action_dim']):
                    action_range = self.action_ranges[i]
                    prev_actions_batch[:, :, i] = 2 * (prev_actions_batch[:, :, i] - action_range[0]) / (action_range[1] - action_range[0]) - 1
            
            prev_actions_tensor = torch.FloatTensor(prev_actions_batch).to(self.device)
            
            # Forward pass
            if isinstance(self.model, AgentLSTMWithUncertainty) and return_uncertainty:
                actions, uncertainties, _ = self.model(feature_tensor, prev_actions_tensor)
                uncertainties = uncertainties.cpu().numpy()
            else:
                actions, _ = self.model(feature_tensor, prev_actions_tensor)
                uncertainties = None
            
            # Convert actions to numpy
            actions = actions.cpu().numpy()
            
            # Denormalize actions
            if self.config.get('normalize', True):
                for i in range(self.config['action_dim']):
                    action_range = self.action_ranges[i]
                    actions[:, i] = (actions[:, i] + 1) * (action_range[1] - action_range[0]) / 2 + action_range[0]
            
            # Clip actions to valid ranges
            actions[:, 0] = np.clip(actions[:, 0], -1, 1)    # movedirection_x
            actions[:, 1] = np.clip(actions[:, 1], -1, 1)    # movedirection_z
            actions[:, 2] = np.clip(actions[:, 2], -10, 10)  # lookrotation_x
            actions[:, 3] = np.clip(actions[:, 3], -10, 10)  # lookrotation_z
            
            result = {'actions': actions}
            if uncertainties is not None:
                result['uncertainties'] = uncertainties
            
            return result
    
    def benchmark(self, num_iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with timing statistics
        """
        # Warm up
        dummy_features = np.random.randn(self.config['feature_dim'])
        for _ in range(10):
            self.predict(dummy_features)
        
        # Time single predictions
        self.reset()
        single_times = []
        for _ in range(num_iterations):
            features = np.random.randn(self.config['feature_dim'])
            start = time.time()
            self.predict(features)
            single_times.append(time.time() - start)
        
        # Time batch predictions
        batch_sizes = [1, 16, 32, 64, 128]
        batch_times = {}
        
        for batch_size in batch_sizes:
            features_batch = np.random.randn(batch_size, self.config['use_history'], self.config['feature_dim'])
            
            times = []
            for _ in range(num_iterations // 10):
                start = time.time()
                self.predict_batch(features_batch)
                times.append(time.time() - start)
            
            batch_times[batch_size] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'per_sample': np.mean(times) / batch_size
            }
        
        return {
            'single_prediction': {
                'mean_ms': np.mean(single_times) * 1000,
                'std_ms': np.std(single_times) * 1000,
                'fps': 1.0 / np.mean(single_times)
            },
            'batch_predictions': batch_times
        }


def test_on_dataset(checkpoint_path: str, data_dir: str, num_samples: int = 100):
    """Test model on dataset samples."""
    # Create inference engine
    engine = AgentInference(checkpoint_path)
    
    # Create dataset
    dataset = AgentDataset(
        data_dir=data_dir,
        sequence_length=20,
        use_history=engine.config['use_history'],
        normalize=engine.config['normalize'],
        max_samples=num_samples,
        is_train=True,  # Use training split for testing
        train_split=1.0  # Use all available samples
    )
    
    # Test predictions
    errors = []
    uncertainties = []
    
    for i in range(min(num_samples, len(dataset))):
        (features, prev_actions), target = dataset[i]
        
        # Predict
        result = engine.predict_batch(
            features.unsqueeze(0).numpy(),
            prev_actions.unsqueeze(0).numpy(),
            return_uncertainty=True
        )
        
        predicted = result['actions'][0]
        
        # Denormalize target for comparison
        target_denorm = dataset.denormalize_actions(target.unsqueeze(0)).numpy()[0]
        
        # Calculate error
        error = np.abs(predicted - target_denorm)
        errors.append(error)
        
        if 'uncertainties' in result:
            uncertainties.append(result['uncertainties'][0])
    
    errors = np.array(errors)
    
    # Print statistics
    print("\nPrediction Error Statistics:")
    print(f"{'Action':<20} {'Mean Error':<15} {'Std Error':<15} {'Max Error':<15}")
    print("-" * 65)
    
    action_names = ['movedirection_x', 'movedirection_z', 'lookrotation_x', 'lookrotation_z']
    for i, name in enumerate(action_names):
        print(f"{name:<20} {errors[:, i].mean():<15.6f} {errors[:, i].std():<15.6f} {errors[:, i].max():<15.6f}")
    
    if uncertainties:
        uncertainties = np.array(uncertainties)
        print("\nUncertainty Statistics:")
        for i, name in enumerate(action_names):
            print(f"{name:<20} {uncertainties[:, i].mean():<15.6f} {uncertainties[:, i].std():<15.6f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM Agent Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['test', 'benchmark', 'interactive'], 
                        default='test', help='Inference mode')
    parser.add_argument('--data_dir', type=str, 
                        default='/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/clash_squad_partitioned_features_chunked',
                        help='Path to dataset (for test mode)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'mps', 'cuda', 'cpu'], 
                        help='Device to use (auto will choose MPS > CUDA > CPU)')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_on_dataset(args.checkpoint, args.data_dir, args.num_samples)
    
    elif args.mode == 'benchmark':
        engine = AgentInference(args.checkpoint, device=args.device)
        results = engine.benchmark()
        
        print("\nBenchmark Results:")
        print(f"Single prediction: {results['single_prediction']['mean_ms']:.2f} ± "
              f"{results['single_prediction']['std_ms']:.2f} ms "
              f"({results['single_prediction']['fps']:.1f} FPS)")
        
        print("\nBatch predictions:")
        for batch_size, stats in results['batch_predictions'].items():
            print(f"  Batch size {batch_size}: {stats['mean']*1000:.2f} ms total, "
                  f"{stats['per_sample']*1000:.2f} ms per sample")
    
    elif args.mode == 'interactive':
        engine = AgentInference(args.checkpoint, device=args.device)
        print("\nInteractive mode. Enter 'quit' to exit.")
        print("Enter features as comma-separated values (20 values)")
        
        while True:
            try:
                input_str = input("\nEnter features: ")
                if input_str.lower() == 'quit':
                    break
                
                features = np.array([float(x) for x in input_str.split(',')])
                if len(features) != 20:
                    print(f"Error: Expected 20 features, got {len(features)}")
                    continue
                
                result = engine.predict(features, return_uncertainty=True)
                
                print("\nPredicted actions:")
                action_names = ['movedirection_x', 'movedirection_z', 'lookrotation_x', 'lookrotation_z']
                for i, name in enumerate(action_names):
                    if 'uncertainties' in result:
                        print(f"  {name}: {result['actions'][i]:.4f} (± {result['uncertainties'][i]:.4f})")
                    else:
                        print(f"  {name}: {result['actions'][i]:.4f}")
                
            except Exception as e:
                print(f"Error: {e}")


if __name__ == '__main__':
    main() 