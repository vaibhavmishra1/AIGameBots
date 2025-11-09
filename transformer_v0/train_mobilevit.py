import os
import sys
import time
import math
import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Ensure we can import shared CSGO modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))

from dataloader import create_data_loaders
from model_mobilevit import create_mobilevit_temporal_model
from model_mobilevit_pretrained import create_mobilevit_pretrained
from config import (
    N_TIMESTEPS,
    n_keys,
    n_clicks,
    n_mouse_x,
    n_mouse_y,
    GAMMA,
)


def bce_from_probs(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Stable BCE on probabilities under AMP by temporarily disabling autocast
    and computing loss in float32.
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        pred_f32 = pred.float()
        target_f32 = target.float()
        return F.binary_cross_entropy(pred_f32, target_f32, reduction='mean')


def categorical_ce_from_probs(pred_probs: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy when the model outputs probabilities (already softmaxed)
    and targets are one-hot. This mirrors the helper used in the ViViT trainer.
    """
    eps = 1e-8
    pred_probs = pred_probs.clamp(min=eps, max=1.0)
    loss = -(target_onehot * pred_probs.log()).sum(dim=-1)
    return loss.mean()


def create_optimizer_and_scheduler(model, args, steps_per_epoch):
    """Create optimizer with proper learning rate scheduling."""
    
    # Separate parameters for different learning rates
    spatial_params = []
    temporal_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'spatial_encoder' in name:
            spatial_params.append(param)
        elif 'temporal_transformer' in name:
            temporal_params.append(param)
        else:
            other_params.append(param)
    
    # Use different learning rates for different parts
    param_groups = [
        {'params': spatial_params, 'lr': args.lr * 0.5},  # Increased from 0.1
        {'params': temporal_params, 'lr': args.lr},
        {'params': other_params, 'lr': args.lr},  # Increased from 0.5
    ]
    
    # AdamW optimizer with better settings for transformers
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),  # Lower beta2 for transformers
        eps=1e-8
    )
    print(f"Optimizer LR groups: {[g['lr'] for g in optimizer.param_groups]}")  # Debug print
    
    # Create warmup + cosine decay schedule
    num_training_steps = args.epochs * steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    # Linear warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # Increased from 0.001 for faster warmup
        end_factor=1.0,
        total_iters=num_warmup_steps
    )
    
    # Cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=args.lr * 0.001  # Decreased from 0.01 for less aggressive decay
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps]
    )
    
    return optimizer, scheduler


def compute_loss(
    outputs: Tuple[torch.Tensor, ...],
    y_true: torch.Tensor,
    loss_weights: Dict[str, float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss with configurable weights for different components."""
    
    if loss_weights is None:
        loss_weights = {
            'keys': 1.0,
            'clicks': 1.0,
            'mouse_x': 1.0,  # Emphasize mouse accuracy
            'mouse_y': 1.0,
            'value': 0.5,
        }
    
    keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = outputs
    
    # Split ground truth
    idx_keys_end = n_keys
    idx_clicks_end = idx_keys_end + n_clicks
    idx_mouse_x_end = idx_clicks_end + n_mouse_x
    idx_mouse_y_end = idx_mouse_x_end + n_mouse_y
    idx_reward = idx_mouse_y_end
    
    keys_true = y_true[:, :, 0:idx_keys_end]
    clicks_true = y_true[:, :, idx_keys_end:idx_clicks_end]
    mouse_x_true = y_true[:, :, idx_clicks_end:idx_mouse_x_end]
    mouse_y_true = y_true[:, :, idx_mouse_x_end:idx_mouse_y_end]
    reward_true = y_true[:, :, idx_reward:idx_reward + 1]
    
    # Compute individual losses
    # Keys: separate important keys
    loss_wasd = bce_from_probs(keys_out[:, :, 0:4], keys_true[:, :, 0:4])
    loss_space = bce_from_probs(keys_out[:, :, 4:5], keys_true[:, :, 4:5])
    loss_other_keys = bce_from_probs(keys_out[:, :, 5:], keys_true[:, :, 5:])
    loss_keys = (loss_wasd + loss_space * 0.5 + loss_other_keys * 0.3) / 2.8
    
    # Clicks: emphasize left click (fire)
    loss_lclick = bce_from_probs(clicks_out[:, :, 0:1], clicks_true[:, :, 0:1])
    loss_rclick = bce_from_probs(clicks_out[:, :, 1:2], clicks_true[:, :, 1:2])
    loss_clicks = loss_lclick + loss_rclick * 0.5
    
    # Mouse movements: use cross-entropy
    # IMPORTANT: Our heads already output probabilities via softmax.
    # Use CE on probabilities (optionally mixed with EMD) instead of CE on logits.
    # Flatten (B, T, C) -> (B*T, C)
    mx_pred = mouse_x_out.reshape(-1, n_mouse_x)
    mx_true = mouse_x_true.reshape(-1, n_mouse_x)
    my_pred = mouse_y_out.reshape(-1, n_mouse_y)
    my_true = mouse_y_true.reshape(-1, n_mouse_y)
    # CE component
    ce_x = categorical_ce_from_probs(mx_pred, mx_true)
    ce_y = categorical_ce_from_probs(my_pred, my_true)
    # EMD (Wasserstein-1) component across class CDFs (stabilizes small shifts)
    cdf_pred_x = torch.cumsum(mx_pred, dim=-1)
    cdf_true_x = torch.cumsum(mx_true, dim=-1)
    cdf_pred_y = torch.cumsum(my_pred, dim=-1)
    cdf_true_y = torch.cumsum(my_true, dim=-1)
    emd_x = (cdf_pred_x - cdf_true_x).abs().mean()
    emd_y = (cdf_pred_y - cdf_true_y).abs().mean()
    alpha = 0.2
    loss_mouse_x = alpha * emd_x + (1.0 - alpha) * ce_x
    loss_mouse_y = alpha * emd_y + (1.0 - alpha) * ce_y
    
    # Value function: TD loss
    T = value_out.shape[1]
    if T > 1:
        v_t = value_out[:, :-1, :]
        v_tp1 = value_out[:, 1:, :]
        r_t = reward_true[:, :-1, :]
        td_target = r_t + GAMMA * v_tp1.detach()
        loss_value = F.mse_loss(v_t, td_target)
    else:
        # Single frame, just predict reward
        loss_value = F.mse_loss(value_out, reward_true)
    
    # Combine losses with weights
    total_loss = (
        loss_weights['keys'] * loss_keys +
        loss_weights['clicks'] * loss_clicks +
        loss_weights['mouse_x'] * loss_mouse_x +
        loss_weights['mouse_y'] * loss_mouse_y +
        loss_weights['value'] * loss_value
    )
    
    # Store individual losses for logging
    parts = {
        'loss_keys': float(loss_keys.item()),
        'loss_clicks': float(loss_clicks.item()),
        'loss_mouse_x': float(loss_mouse_x.item()),
        'loss_mouse_y': float(loss_mouse_y.item()),
        'loss_value': float(loss_value.item()),
        'loss_total': float(total_loss.item()),
    }
    
    return total_loss, parts


def compute_metrics(
    outputs: Tuple[torch.Tensor, ...],
    y_true: torch.Tensor,
) -> Dict[str, float]:
    """Compute accuracy metrics."""
    
    keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = outputs
    
    # Split ground truth
    idx_keys_end = n_keys
    idx_clicks_end = idx_keys_end + n_clicks
    idx_mouse_x_end = idx_clicks_end + n_mouse_x
    idx_mouse_y_end = idx_mouse_x_end + n_mouse_y
    
    keys_true = y_true[:, :, 0:idx_keys_end]
    clicks_true = y_true[:, :, idx_keys_end:idx_clicks_end]
    mouse_x_true = y_true[:, :, idx_clicks_end:idx_mouse_x_end]
    mouse_y_true = y_true[:, :, idx_mouse_x_end:idx_mouse_y_end]
    
    # Compute accuracies
    wasd_acc = ((keys_out[:, :, 0:4] > 0.5) == keys_true[:, :, 0:4]).float().mean()
    lclk_acc = ((clicks_out[:, :, 0:1] > 0.5) == clicks_true[:, :, 0:1]).float().mean()
    
    # Mouse accuracy (argmax)
    mx_pred = mouse_x_out.argmax(dim=-1)
    mx_true = mouse_x_true.argmax(dim=-1)
    my_pred = mouse_y_out.argmax(dim=-1)
    my_true = mouse_y_true.argmax(dim=-1)
    m_x_acc = (mx_pred == mx_true).float().mean()
    m_y_acc = (my_pred == my_true).float().mean()
    
    # Additional metrics
    no_fire = 1.0 - clicks_true[:, :, 0].mean()  # Percentage of no fire
    
    return {
        'Lclk_acc': float(lclk_acc.item()),
        'no_fire': float(no_fire.item()),
        'm_x_acc': float(m_x_acc.item()),
        'm_y_acc': float(m_y_acc.item()),
        'wasd_acc': float(wasd_acc.item()),
    }


def validate(model, val_loader, device):
    """Run validation."""
    model.eval()
    val_loss = 0.0
    val_loss_parts = {}
    val_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_aux in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_aux = batch_aux.to(device)
            
            outputs = model(batch_x, aux_input=batch_aux, stateful=False)
            loss, loss_parts = compute_loss(outputs, batch_y)
            
            val_loss += float(loss.item())
            
            # Accumulate loss parts
            for key, value in loss_parts.items():
                if key not in val_loss_parts:
                    val_loss_parts[key] = 0.0
                val_loss_parts[key] += value
            
            # Accumulate metrics
            batch_metrics = compute_metrics(outputs, batch_y)
            for key, value in batch_metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = 0.0
                val_metrics[key] += value
            
            num_batches += 1
            
            # Early stop for validation to save time
            if num_batches >= 50:
                break
    
    # Average
    val_loss /= max(num_batches, 1)
    for key in val_loss_parts:
        val_loss_parts[key] /= max(num_batches, 1)
    for key in val_metrics:
        val_metrics[key] /= max(num_batches, 1)
    
    model.train()
    return val_loss, val_loss_parts, val_metrics


def train(args):
    """Main training loop."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        batch_size=args.batch_size,
        starting_num=args.starting_num,
        highest_num=args.highest_num,
        folder_name=args.data_dir,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        n_jitter=args.n_jitter,
        is_mirror=args.is_mirror,
        transform=True,
    )
    
    steps_per_epoch = len(train_loader)
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Create model (pretrained or scratch)
    if args.spatial_encoder and args.spatial_encoder != 'scratch':
        # Use pretrained spatial backbone (mobilevit_s/xs/xxs via timm, or mobilenet_v3)
        print(f"Using pretrained spatial encoder: {args.spatial_encoder} "
              f"(freeze_early_layers={args.freeze_early_layers})")
        model = create_mobilevit_pretrained(
            spatial_encoder=args.spatial_encoder,
            freeze_early_layers=bool(args.freeze_early_layers),
            temporal_layers=args.temporal_layers,
            temporal_heads=args.temporal_heads,
            temporal_dim=args.spatial_dim,
            aux_input_on=args.use_prev_actions,
            dropout=args.dropout,
        )
    else:
        # Use lightweight scratch MobileViT-like encoder
        print("Using scratch MobileViT-like spatial encoder")
        model = create_mobilevit_temporal_model(
            spatial_dim=args.spatial_dim,
            temporal_layers=args.temporal_layers,
            temporal_heads=args.temporal_heads,
            max_seq_len=N_TIMESTEPS,
            aux_input_on=args.use_prev_actions,
            dropout=args.dropout,
        )
    model.to(device)
    
    # Print model info
    total_params, trainable_params = model.count_parameters()
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, steps_per_epoch)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_mouse_acc = 0.0
    global_step = 0
    
    # Load checkpoint if resuming
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        if 'best_mouse_acc' in checkpoint:
            best_mouse_acc = checkpoint['best_mouse_acc']
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_metrics = {}
        num_batches = 0
        
        model.train()
        for batch_idx, (batch_x, batch_y, batch_aux) in enumerate(train_loader, 1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_aux = batch_aux.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(batch_x, aux_input=batch_aux if args.use_prev_actions else None, stateful=False)
                loss, loss_parts = compute_loss(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Accumulate metrics
            running_loss += float(loss.item())
            metrics = compute_metrics(outputs, batch_y)
            for key, value in metrics.items():
                if key not in running_metrics:
                    running_metrics[key] = 0.0
                running_metrics[key] += value
            num_batches += 1
            global_step += 1
            
            # Log progress
            if batch_idx % args.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = running_loss / num_batches
                avg_metrics = {k: v/num_batches for k, v in running_metrics.items()}
                print(
                    f"epoch {epoch} step {batch_idx} | "
                    f"loss {avg_loss:.4f} | "
                    f"Lclk_acc {avg_metrics['Lclk_acc']:.3f} "
                    f"m_x_acc {avg_metrics['m_x_acc']:.3f} "
                    f"m_y_acc {avg_metrics['m_y_acc']:.3f} "
                    f"wasd_acc {avg_metrics['wasd_acc']:.3f} | "
                    f"lr {current_lr:.6f}"
                )
            
            # Early stopping per epoch for debugging
            if args.debug and batch_idx >= 100:
                break
        
        # Validation
        val_loss, val_loss_parts, val_metrics = validate(model, val_loader, device)
        
        # Calculate averages
        avg_train_loss = running_loss / num_batches
        avg_train_metrics = {k: v/num_batches for k, v in running_metrics.items()}
        
        # Print epoch summary
        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch} done in {elapsed:.1f}s | "
            f"train_loss {avg_train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"Lclk_acc {val_metrics['Lclk_acc']:.3f} "
            f"m_x_acc {val_metrics['m_x_acc']:.3f} "
            f"m_y_acc {val_metrics['m_y_acc']:.3f} "
            f"wasd_acc {val_metrics['wasd_acc']:.3f}"
        )
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_mouse_acc': best_mouse_acc,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'args': vars(args),
        }
        
        # Save last checkpoint
        last_path = os.path.join(args.save_dir, 'mobilevit_last.pt')
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, 'mobilevit_best_loss.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model (loss) with val_loss={val_loss:.4f}")
        
        # Save best checkpoint based on mouse accuracy
        avg_mouse_acc = (val_metrics['m_x_acc'] + val_metrics['m_y_acc']) / 2
        if avg_mouse_acc > best_mouse_acc:
            best_mouse_acc = avg_mouse_acc
            best_path = os.path.join(args.save_dir, 'mobilevit_best_mouse.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model (mouse) with avg_mouse_acc={avg_mouse_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileViT + Temporal Transformer for CSGO")
    
    # Model architecture
    parser.add_argument('--spatial_encoder', type=str, default='mobilevit_s',
                        help="Spatial backbone: 'scratch', 'mobilenet_v3', 'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs'")
    parser.add_argument('--freeze_early_layers', action='store_true',
                        help="Freeze early layers of pretrained spatial encoder")
    parser.add_argument('--spatial_dim', type=int, default=256,
                        help="Dimension of spatial features from MobileViT")
    parser.add_argument('--temporal_layers', type=int, default=6,
                        help="Number of temporal transformer layers")
    parser.add_argument('--temporal_heads', type=int, default=8,
                        help="Number of attention heads in temporal transformer")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate")
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size (adjust based on GPU memory)")
    parser.add_argument('--epochs', type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help="Weight decay for AdamW")
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help="Ratio of steps for warmup")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # Data
    parser.add_argument('--starting_num', type=int, default=2,
                        help="First file number in dataset")
    parser.add_argument('--highest_num', type=int, default=190,
                        help="Last file number in dataset")
    parser.add_argument('--n_jitter', type=int, default=0,
                        help="Temporal jitter for data augmentation")
    parser.add_argument('--is_mirror', action='store_true',
                        help="Enable mirror augmentation")
    parser.add_argument('--data_dir', type=str,
                        default='/root/AIGameBots/Counter-Strike_Behavioural_Cloning/dataset_dm_expert_dust2/',
                        help="Path to dataset directory")
    
    # Other
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints',
                        help="Directory to save checkpoints")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument('--log_every', type=int, default=10,
                        help="Log frequency (in steps)")
    parser.add_argument('--use_prev_actions', action='store_true', default=True,
                        help="Use previous actions as auxiliary input")
    parser.add_argument('--resume', type=str, default='',
                        help="Path to checkpoint to resume from")
    parser.add_argument('--debug', action='store_true',
                        help="Debug mode with limited steps")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
