import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from dataset import create_dataloaders, AgentDataset
from model import create_model, AgentLSTMWithUncertainty


class ActionLoss(nn.Module):
    """Custom loss function for action prediction."""
    
    def __init__(
        self,
        loss_type: str = "smooth_l1",
        action_weights: Optional[List[float]] = None,
        uncertainty_weight: float = 0.1
    ):
        super().__init__()
        self.loss_type = loss_type
        self.action_weights = action_weights or [1.0, 1.0, 1.0, 1.0]  # Default equal weights
        self.uncertainty_weight = uncertainty_weight
        
        if loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == "smooth_l1":
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        elif loss_type == "huber":
            self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate loss with optional uncertainty weighting.
        
        Args:
            predictions: Predicted actions [batch, action_dim]
            targets: Target actions [batch, action_dim]
            uncertainties: Optional uncertainty estimates [batch, action_dim]
            
        Returns:
            Total loss and dictionary of loss components
        """
        # Base loss per action
        base_losses = self.base_loss(predictions, targets)  # [batch, action_dim]
        
        # Apply action weights
        weights = torch.tensor(self.action_weights, device=predictions.device)
        weighted_losses = base_losses * weights
        
        # If uncertainties provided, use them to weight the losses
        if uncertainties is not None:
            # Higher uncertainty -> lower weight
            uncertainty_weights = 1.0 / (1.0 + uncertainties)
            weighted_losses = weighted_losses * uncertainty_weights
            
            # Add uncertainty regularization (encourage lower uncertainty)
            uncertainty_reg = self.uncertainty_weight * uncertainties.mean()
        else:
            uncertainty_reg = 0.0
        
        # Total loss
        action_loss = weighted_losses.mean()
        total_loss = action_loss + uncertainty_reg
        
        # Loss components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'uncertainty_reg': uncertainty_reg.item() if uncertainties is not None else 0.0
        }
        
        # Per-action losses
        for i in range(base_losses.shape[1]):
            loss_dict[f'action_{i}_loss'] = base_losses[:, i].mean().item()
        
        return total_loss, loss_dict


class Trainer:
    """Trainer class for LSTM model."""
    
    def __init__(self, config: Dict):
        self.config = config
        # Device selection with MPS support
        if config['device'] == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif config['device'] == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif config['device'] == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        # Create model
        self.model = create_model(
            feature_dim=config['feature_dim'],
            action_dim=config['action_dim'],
            model_type=config['model_type'],
            hidden_dim=config['hidden_dim'],
            num_lstm_layers=config['num_lstm_layers'],
            dropout=config['dropout'],
            use_attention=config['use_attention'],
            use_residual=config['use_residual'],
            activation=config['activation']
        ).to(self.device)
        
        # Loss function
        self.criterion = ActionLoss(
            loss_type=config['loss_type'],
            action_weights=config.get('action_weights'),
            uncertainty_weight=config.get('uncertainty_weight', 0.1)
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training (only for CUDA, not MPS)
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available() and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Create data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            sequence_length=config['sequence_length'],
            use_history=config['use_history'],
            normalize=config['normalize'],
            max_samples=config.get('max_samples'),
            train_split=config['train_split'],
            random_seed=config['random_seed']
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save normalization statistics if normalized
        if config['normalize'] and hasattr(self.train_loader.dataset, 'save_normalization_stats'):
            stats_path = self.checkpoint_dir / 'normalization_stats.npz'
            self.train_loader.dataset.save_normalization_stats(str(stats_path))
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Save config
        with open(self.checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        opt_type = self.config['optimizer']
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif opt_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif opt_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_type = self.config.get('scheduler', 'cosine')
        
        if sched_type == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif sched_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config.get('min_lr', 1e-6),
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, ((features, prev_actions), targets) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            prev_actions = prev_actions.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                if isinstance(self.model, AgentLSTMWithUncertainty):
                    predictions, uncertainties, _ = self.model(features, prev_actions)
                else:
                    predictions, _ = self.model(features, prev_actions)
                    uncertainties = None
                
                loss, loss_dict = self.criterion(predictions, targets, uncertainties)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('grad_clip', 1.0)
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('grad_clip', 1.0)
                )
                self.optimizer.step()
            
            # Update metrics
            epoch_losses.append(loss_dict['total_loss'])
            for k, v in loss_dict.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'action_loss': f"{loss_dict['action_loss']:.4f}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config.get('log_interval', 10) == 0:
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                
                # Log learning rate
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/lr', lr, self.global_step)
            
            self.global_step += 1
        
        # Calculate epoch statistics
        epoch_stats = {k: np.mean(v) for k, v in epoch_metrics.items()}
        epoch_stats['epoch'] = self.epoch
        
        return epoch_stats
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = []
        val_metrics = {}
        
        with torch.no_grad():
            for (features, prev_actions), targets in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                features = features.to(self.device)
                prev_actions = prev_actions.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if isinstance(self.model, AgentLSTMWithUncertainty):
                    predictions, uncertainties, _ = self.model(features, prev_actions)
                else:
                    predictions, _ = self.model(features, prev_actions)
                    uncertainties = None
                
                loss, loss_dict = self.criterion(predictions, targets, uncertainties)
                
                # Update metrics
                val_losses.append(loss_dict['total_loss'])
                for k, v in loss_dict.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v)
        
        # Calculate validation statistics
        val_stats = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # Log to tensorboard
        for k, v in val_stats.items():
            self.writer.add_scalar(f'val/{k}', v, self.epoch)
        
        return val_stats
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.6f}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for self.epoch in range(self.epoch, self.config['num_epochs']):
            # Train epoch
            epoch_start = time.time()
            train_stats = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            # Validate
            val_stats = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_stats['total_loss'])
                else:
                    self.scheduler.step()
            
            # Check for improvement
            if val_stats['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_stats['total_loss']
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.save_checkpoint(is_best=False)
            
            # Print epoch summary
            print(f"\nEpoch {self.epoch} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_stats['total_loss']:.6f}")
            print(f"  Val Loss: {val_stats['total_loss']:.6f}")
            print(f"  Best Val Loss: {self.best_val_loss:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 20):
                print(f"\nEarly stopping triggered after {self.epoch} epochs")
                break
        
        print("\nTraining completed!")
        self.writer.close()


def create_config(args) -> Dict:
    """Create configuration dictionary from arguments."""
    config = {
        # Data
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'sequence_length': args.sequence_length,
        'use_history': args.use_history,
        'normalize': args.normalize,
        'max_samples': args.max_samples,
        'train_split': args.train_split,
        'random_seed': args.random_seed,
        
        # Model
        'feature_dim': args.feature_dim,
        'action_dim': args.action_dim,
        'model_type': args.model_type,
        'hidden_dim': args.hidden_dim,
        'num_lstm_layers': args.num_lstm_layers,
        'dropout': args.dropout,
        'use_attention': args.use_attention,
        'use_residual': args.use_residual,
        'activation': args.activation,
        
        # Training
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'loss_type': args.loss_type,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'use_amp': args.use_amp,
        'patience': args.patience,
        
        # Loss weights
        'action_weights': args.action_weights,
        'uncertainty_weight': args.uncertainty_weight,
        
        # Logging
        'log_dir': args.log_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'log_interval': args.log_interval,
        
        # Device
        'device': args.device
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train LSTM for agent imitation learning')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default='/Users/vaibhavmishra/Desktop/clash_squad_partitioned_features_chunked',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length')
    parser.add_argument('--use_history', type=int, default=5, help='Number of history frames')
    parser.add_argument('--normalize', action='store_true', default=True, help='Normalize data')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train split ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    
    # Model arguments
    parser.add_argument('--feature_dim', type=int, default=20, help='Feature dimension')
    parser.add_argument('--action_dim', type=int, default=4, help='Action dimension')
    parser.add_argument('--model_type', type=str, default='uncertainty', 
                        choices=['standard', 'uncertainty'], help='Model type')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_lstm_layers', type=int, default=4, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use_attention', action='store_true', default=True, help='Use attention')
    parser.add_argument('--use_residual', action='store_true', default=True, help='Use residual connections')
    parser.add_argument('--activation', type=str, default='gelu', 
                        choices=['relu', 'gelu', 'tanh'], help='Activation function')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'plateau', 'none'], help='LR scheduler')
    parser.add_argument('--loss_type', type=str, default='smooth_l1', 
                        choices=['mse', 'smooth_l1', 'huber'], help='Loss function')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    # Loss weights
    parser.add_argument('--action_weights', type=float, nargs=4, 
                        default=[1.0, 1.0, 2.0, 2.0], 
                        help='Weights for each action dimension')
    parser.add_argument('--uncertainty_weight', type=float, default=0.1, 
                        help='Weight for uncertainty regularization')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='runs/lstm_agent', help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/lstm_agent', help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    
    # Device
    parser.add_argument('--device', type=str, default='mps', 
                        choices=['auto', 'mps', 'cuda', 'cpu'], 
                        help='Device to use (auto will choose MPS > CUDA > CPU)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create config
    config = create_config(args)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
