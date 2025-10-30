import os
import sys
import time
import math
import json
import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Add the parent directory to the path to import config and shared modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))
from config import (
    N_TIMESTEPS,
    n_keys,
    n_clicks,
    n_mouse_x,
    n_mouse_y,
    GAMMA,
)

from model import create_model
from dataloader import create_data_loaders


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def bce_from_probs(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred, target, reduction='mean')


def categorical_ce_from_probs(pred_probs: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    pred_probs = pred_probs.clamp(min=eps, max=1.0)
    loss = -(target_onehot * pred_probs.log()).sum(dim=-1)
    return loss.mean()


def create_gaussian_targets(target_onehot: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    Convert one-hot targets to Gaussian soft targets.
    
    Args:
        target_onehot: One-hot encoded targets [batch, timesteps, num_classes]
        sigma: Standard deviation of the Gaussian distribution
        
    Returns:
        Gaussian soft targets with same shape as input
    """
    device = target_onehot.device
    batch_size, timesteps, num_classes = target_onehot.shape
    
    # Get the index of the true class for each sample
    true_indices = target_onehot.argmax(dim=-1)  # [batch, timesteps]
    
    # Create position indices for each class
    class_indices = torch.arange(num_classes, device=device).float()  # [num_classes]
    
    # Expand dimensions for broadcasting
    true_indices_expanded = true_indices.unsqueeze(-1).float()  # [batch, timesteps, 1]
    class_indices_expanded = class_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, num_classes]
    
    # Calculate Gaussian weights: exp(-0.5 * ((x - mu) / sigma)^2)
    distances = class_indices_expanded - true_indices_expanded  # [batch, timesteps, num_classes]
    gaussian_weights = torch.exp(-0.5 * (distances / sigma) ** 2)
    
    # Normalize so that the peak (correct class) has value 1
    # We do this by dividing by the maximum value in each sample
    max_weights = gaussian_weights.max(dim=-1, keepdim=True)[0]  # [batch, timesteps, 1]
    gaussian_targets = gaussian_weights / max_weights
    
    return gaussian_targets


def gaussian_categorical_ce_loss(pred_probs: torch.Tensor, target_onehot: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    Categorical cross-entropy loss with Gaussian-weighted error.
    This version gives ~0 loss when predicting the correct class with high confidence.
    
    Args:
        pred_probs: Predicted probabilities [batch, timesteps, num_classes]
        target_onehot: One-hot encoded targets [batch, timesteps, num_classes]
        sigma: Standard deviation for Gaussian weighting
        
    Returns:
        Loss value
    """
    # Get the true class indices
    true_indices = target_onehot.argmax(dim=-1)  # [batch, timesteps]
    
    # Standard cross-entropy loss for the correct class
    eps = 1e-8
    pred_probs = pred_probs.clamp(min=eps, max=1.0)
    correct_class_loss = -torch.gather(pred_probs.log(), dim=-1, index=true_indices.unsqueeze(-1)).squeeze(-1)
    
    # Add Gaussian-weighted penalty for incorrect predictions
    device = target_onehot.device
    batch_size, timesteps, num_classes = target_onehot.shape
    
    # Create position indices for each class
    class_indices = torch.arange(num_classes, device=device).float()
    
    # Expand dimensions for broadcasting
    true_indices_expanded = true_indices.unsqueeze(-1).float()  # [batch, timesteps, 1]
    class_indices_expanded = class_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, num_classes]
    
    # Calculate distances and Gaussian weights
    distances = class_indices_expanded - true_indices_expanded  # [batch, timesteps, num_classes]
    gaussian_weights = torch.exp(-0.5 * (distances / sigma) ** 2)
    
    # Set weight to 0 for the correct class (no penalty for being right)
    correct_mask = (class_indices_expanded == true_indices_expanded).float()
    gaussian_weights = gaussian_weights * (1 - correct_mask)
    
    # Penalty for putting probability mass on wrong classes, weighted by distance
    wrong_class_penalty = (gaussian_weights * pred_probs).sum(dim=-1)
    
    # Total loss: negative log likelihood of correct class + Gaussian-weighted penalty
    total_loss = correct_class_loss + wrong_class_penalty
    
    return total_loss.mean()


def compute_custom_loss(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y_true: torch.Tensor,
    mouse_sigma_x: float = 1.5,
    mouse_sigma_y: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Replicates the TensorFlow custom loss components from dm_train_model.py.

    y_true shape: [batch, timesteps, n_keys+n_clicks+n_mouse_x+n_mouse_y + 2]
    outputs: (keys, clicks, mouse_x, mouse_y, value)
    """
    keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = outputs

    # Indices into y_true
    idx_keys_end = n_keys
    idx_clicks_end = idx_keys_end + n_clicks
    idx_mouse_x_end = idx_clicks_end + n_mouse_x
    idx_mouse_y_end = idx_mouse_x_end + n_mouse_y
    idx_reward = idx_mouse_y_end  # last two slots: reward, advantage

    keys_true = y_true[:, :, 0:idx_keys_end]
    clicks_true = y_true[:, :, idx_keys_end:idx_clicks_end]
    mouse_x_true = y_true[:, :, idx_clicks_end:idx_mouse_x_end]
    mouse_y_true = y_true[:, :, idx_mouse_x_end:idx_mouse_y_end]
    reward_true = y_true[:, :, idx_reward:idx_reward + 1]

    # 1) Keys BCE (match TF segments used in custom_loss)
    loss1a = bce_from_probs(keys_out[:, :, 0:4], keys_true[:, :, 0:4])  # wasd
    loss1b = bce_from_probs(keys_out[:, :, 4:5], keys_true[:, :, 4:5])  # space
    loss1c = bce_from_probs(keys_out[:, :, n_keys - 1:n_keys], keys_true[:, :, n_keys - 1:n_keys])  # reload 'r'
    loss1d = bce_from_probs(keys_out[:, :, n_keys - 4:n_keys - 1], keys_true[:, :, n_keys - 4:n_keys - 1])  # 1,2,3

    # 2) Clicks BCE
    loss2a = bce_from_probs(clicks_out[:, :, 0:1], clicks_true[:, :, 0:1])  # left click
    loss2b = bce_from_probs(clicks_out[:, :, 1:2], clicks_true[:, :, 1:2])  # right click

    # 3) Mouse X mixed loss: CE + normalized EMD (Wasserstein-1) with uniform bins
    # Cross-entropy component
    ce_x = categorical_ce_from_probs(mouse_x_out, mouse_x_true)
    # EMD component (normalize across classes to stabilize scale)
    cdf_pred_x = torch.cumsum(mouse_x_out, dim=-1)
    cdf_true_x = torch.cumsum(mouse_x_true, dim=-1)
    emd_x = (cdf_pred_x - cdf_true_x).abs().mean(dim=-1).mean()
    # Blend: keep CE dominant initially
    alpha = 0.2
    loss3 = alpha * emd_x + (1.0 - alpha) * ce_x
    # Mouse Y mixed loss: CE + normalized EMD (Wasserstein-1) with uniform bins
    ce_y = categorical_ce_from_probs(mouse_y_out, mouse_y_true)
    cdf_pred_y = torch.cumsum(mouse_y_out, dim=-1)
    cdf_true_y = torch.cumsum(mouse_y_true, dim=-1)
    emd_y = (cdf_pred_y - cdf_true_y).abs().mean(dim=-1).mean()
    loss4 = alpha * emd_y + (1.0 - alpha) * ce_y

    # 4) Critic loss: 10 * MSE( reward_t + gamma * v_{t+1} - v_t ) over consecutive timesteps
    v_t = value_out[:, :-1, :]  # [B, T-1, 1]
    v_tp1 = value_out[:, 1:, :]  # [B, T-1, 1]
    r_t = reward_true[:, :-1, :]  # [B, T-1, 1]
    td_target = r_t + GAMMA * v_tp1
    loss_crit = 10.0 * F.mse_loss(v_t, td_target, reduction='mean')

    total_loss = loss1a + loss1b + loss1c + loss1d + loss2a + loss2b + loss3 + loss4 #+ loss_crit
    #total_loss = loss3 

    parts = {
        'loss_keys_wasd': float(loss1a.detach().cpu().item()),
        'loss_keys_space': float(loss1b.detach().cpu().item()),
        'loss_keys_reload': float(loss1c.detach().cpu().item()),
        'loss_keys_123': float(loss1d.detach().cpu().item()),
        'loss_click_L': float(loss2a.detach().cpu().item()),
        'loss_click_R': float(loss2b.detach().cpu().item()),
        'loss_mouse_x': float(loss3.detach().cpu().item()),
        'loss_mouse_y': float(loss4.detach().cpu().item()),
        'loss_critic': float(loss_crit.detach().cpu().item()),
    }
    
    return total_loss, parts


def compute_metrics(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y_true: torch.Tensor,
) -> Dict[str, float]:
    keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = outputs

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

    # Lclk accuracy (threshold 0.5)
    lclk_pred = (clicks_out[:, :, 0:1] >= 0.5).float()
    lclk_true = clicks_true[:, :, 0:1]
    lclk_acc = (lclk_pred == lclk_true).float().mean()

    # no_fire = 1 - mean(lclk_true)
    no_fire = 1.0 - lclk_true.mean()

    # Mouse accuracies (categorical)
    mx_pred = mouse_x_out.argmax(dim=-1)
    mx_true = mouse_x_true.argmax(dim=-1)
    my_pred = mouse_y_out.argmax(dim=-1)
    my_true = mouse_y_true.argmax(dim=-1)
    m_x_acc = (mx_pred == mx_true).float().mean()
    m_y_acc = (my_pred == my_true).float().mean()

    # WASD accuracy: element-wise binary accuracy over first 4 keys
    wasd_pred = (keys_out[:, :, 0:4] >= 0.5).float()
    wasd_true = keys_true[:, :, 0:4]
    wasd_acc = (wasd_pred == wasd_true).float().mean()

    # Critic MSE metric scaled x100 (match TF metric)
    v_t = value_out[:, :-1, :]
    v_tp1 = value_out[:, 1:, :]
    r_t = reward_true[:, :-1, :]
    td_target = r_t + GAMMA * v_tp1
    crit_mse = 100.0 * F.mse_loss(v_t, td_target, reduction='mean')

    return {
        'Lclk_acc': float(lclk_acc.detach().cpu().item()),
        'no_fire': float(no_fire.detach().cpu().item()),
        'm_x_acc': float(m_x_acc.detach().cpu().item()),
        'm_y_acc': float(m_y_acc.detach().cpu().item()),
        'wasd_acc': float(wasd_acc.detach().cpu().item()),
        'crit_mse': float(crit_mse.detach().cpu().item()),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    mouse_sigma_x: float = 1.5,
    mouse_sigma_y: float = 1.0,
    use_prev_actions: bool = True,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    agg_metrics: Dict[str, float] = {}

    for batch_x, batch_y, batch_aux in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_aux = batch_aux.to(device)

        aux = batch_aux if use_prev_actions else None
        outputs = model(batch_x, aux_input=aux)
        loss, _ = compute_custom_loss(outputs, batch_y)
        metrics = compute_metrics(outputs, batch_y)

        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

        for k, v in metrics.items():
            agg_metrics[k] = agg_metrics.get(k, 0.0) + v

    avg_loss = total_loss / max(total_batches, 1)
    avg_metrics = {k: v / max(total_batches, 1) for k, v in agg_metrics.items()}
    return avg_loss, avg_metrics


def train(
    args: argparse.Namespace,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    # Data
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

    # Model
    model = create_model(model_name=args.model_name, pretrained=args.pretrained, aux_input_on=args.use_prev_actions, freeze_backbone=args.freeze_backbone)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = math.inf

    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        
        epoch_start = time.time()
        running_loss = 0.0
        running_batches = 0

        for batch_idx, (batch_x, batch_y, batch_aux) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_aux = batch_aux.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(batch_x, aux_input=(batch_aux if args.use_prev_actions else None))
                loss, loss_parts = compute_custom_loss(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item())
            running_batches += 1
            global_step += 1

            if batch_idx % args.log_every == 0:
                metrics = compute_metrics(outputs, batch_y)
                msg = (
                    f"epoch {epoch} step {batch_idx} | loss {loss.item():.4f} "
                    f"| Lclk_acc {metrics['Lclk_acc']:.3f} m_x_acc {metrics['m_x_acc']:.3f} "
                    f"m_y_acc {metrics['m_y_acc']:.3f} wasd_acc {metrics['wasd_acc']:.3f}"
                )
                print(msg)

        avg_train_loss = running_loss / max(running_batches, 1)

        # # Validation
        # val_loss, val_metrics = validate(model, val_loader, device)

        # elapsed = time.time() - epoch_start
        # print(
        #     f"epoch {epoch} done in {elapsed:.1f}s | train_loss {avg_train_loss:.4f} | "
        #     f"val_loss {val_loss:.4f} | Lclk_acc {val_metrics.get('Lclk_acc', 0.0):.3f} "
        #     f"m_x_acc {val_metrics.get('m_x_acc', 0.0):.3f} m_y_acc {val_metrics.get('m_y_acc', 0.0):.3f} "
        #     f"wasd_acc {val_metrics.get('wasd_acc', 0.0):.3f} crit_mse {val_metrics.get('crit_mse', 0.0):.3f}"
        # )

        # # # Save checkpoint each epoch
        # ckpt_path = os.path.join(args.save_dir, f"{args.model_name}_epoch{epoch}.pt")
        # torch.save(
        #     {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'args': vars(args),
        #     },
        #     ckpt_path,
        # )
        # print(f"saved checkpoint: {ckpt_path}")

        # Track and save best
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_path = os.path.join(args.save_dir, f"{args.model_name}_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"updated best model: {best_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CSGO behavioral cloning model (PyTorch)")
    parser.add_argument('--model_name', type=str, default='default', help="Model configuration name (affects architecture)")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--start_epoch', type=int, default=1, help="Start epoch (for resume)")
    parser.add_argument('--starting_num', type=int, default=2, help="Lowest file number to use")
    parser.add_argument('--highest_num', type=int, default=190, help="Highest file number to use")
    parser.add_argument('--n_jitter', type=int, default=1, help="Temporal jitter frames")
    parser.add_argument('--is_mirror', action='store_true', help="Enable mirror augmentation")
    parser.add_argument('--data_dir', type=str, default='/Users/vaibhav/Desktop/AIGameBots/Counter-Strike_Behavioural_Cloning/dataset_dm_expert_dust2/', help="Dataset directory")
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'checkpoints'), help="Checkpoint directory")
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained EfficientNet weights")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze EfficientNet backbone params")
    parser.set_defaults(pretrained=True)
    parser.set_defaults(freeze_backbone=False)
    parser.add_argument('--num_workers', type=int, default=1, help="DataLoader workers")
    parser.add_argument('--log_every', type=int, default=5, help="Steps between logs")

    # Toggle feeding previous actions as auxiliary input
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_prev_actions', dest='use_prev_actions', action='store_true', help="Feed previous actions as auxiliary input")
    group.add_argument('--no_prev_actions', dest='use_prev_actions', action='store_false', help="Disable feeding previous actions as auxiliary input")
    parser.set_defaults(use_prev_actions=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)

