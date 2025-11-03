import os
import sys
import time
import math
import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Ensure we can import shared CSGO modules (config, etc.) used by reused helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))

# Reuse dataloader from the PyTorch implementation via local re-export
from dataloader import create_data_loaders  # noqa: E402

from model import create_vivit_model  # noqa: E402

# Local copies of loss/metrics (copied from pytorch_implementatiion/train.py to avoid cross-package deps)
from config import (
    N_TIMESTEPS,
    n_keys,
    n_clicks,
    n_mouse_x,
    n_mouse_y,
    GAMMA,
)

def bce_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, reduction='mean')

def bce_from_probs(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Compute BCE on probabilities in float32 with autocast disabled to avoid AMP runtime errors
    # PyTorch recommends BCEWithLogits for autocast; however, the model outputs probabilities.
    # Running this op in FP32 outside autocast keeps numerical stability and avoids the warning.
    with torch.amp.autocast(device_type='cuda', enabled=False):
        pred_f32 = pred.float()
        target_f32 = target.float()
        return F.binary_cross_entropy(pred_f32, target_f32, reduction='mean')

def categorical_ce_from_probs(pred_probs: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    pred_probs = pred_probs.clamp(min=eps, max=1.0)
    loss = -(target_onehot * pred_probs.log()).sum(dim=-1)
    return loss.mean()


def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler,
                    epoch: int,
                    global_step: int,
                    best_loss: float,
                    args: argparse.Namespace) -> None:
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_loss': best_loss,
        'args': vars(args),
    }
    torch.save(state, path)


def try_load_checkpoint(path: str,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scaler: torch.cuda.amp.GradScaler,
                        device: torch.device) -> Tuple[int, int, float]:
    """
    Load checkpoint (supports model-only .pt or full training state).
    Returns (start_epoch, global_step, best_loss).
    """
    if not path or not os.path.isfile(path):
        print(f"No checkpoint found at {path}, starting fresh")
        return 1, 0, math.inf

    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    # Determine model state dict
    if isinstance(ckpt, dict) and ('model_state_dict' in ckpt):
        model_state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and ('state_dict' in ckpt):
        model_state = ckpt['state_dict']
    else:
        # Assume raw state_dict saved via torch.save(model.state_dict(), ...)
        model_state = ckpt

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[resume] Missing keys in state_dict: {len(missing)}")
    if unexpected:
        print(f"[resume] Unexpected keys in state_dict: {len(unexpected)}")

    start_epoch = 1
    global_step = 0
    best_loss = math.inf

    if isinstance(ckpt, dict):
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f"[resume] Optimizer load failed: {e}")
        if 'scaler_state_dict' in ckpt:
            try:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception as e:
                print(f"[resume] GradScaler load failed: {e}")
        if 'epoch' in ckpt and isinstance(ckpt['epoch'], int):
            start_epoch = int(ckpt['epoch']) + 1
        if 'global_step' in ckpt and isinstance(ckpt['global_step'], int):
            global_step = int(ckpt['global_step'])
        if 'best_loss' in ckpt:
            try:
                best_loss = float(ckpt['best_loss'])
            except Exception:
                best_loss = math.inf

    print(f"Resumed. Next epoch: {start_epoch}, global_step: {global_step}, best_loss: {best_loss}")
    return start_epoch, global_step, best_loss

def compute_custom_loss(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y_true: torch.Tensor,
    mouse_sigma_x: float = 1.5,
    mouse_sigma_y: float = 1.0,
    loss_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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

    loss1a = bce_from_probs(keys_out[:, :, 0:4], keys_true[:, :, 0:4])
    loss1b = bce_from_probs(keys_out[:, :, 4:5], keys_true[:, :, 4:5])
    loss1c = bce_from_probs(keys_out[:, :, n_keys - 1:n_keys], keys_true[:, :, n_keys - 1:n_keys])
    loss1d = bce_from_probs(keys_out[:, :, n_keys - 4:n_keys - 1], keys_true[:, :, n_keys - 4:n_keys - 1])

    loss2a = bce_from_probs(clicks_out[:, :, 0:1], clicks_true[:, :, 0:1])
    loss2b = bce_from_probs(clicks_out[:, :, 1:2], clicks_true[:, :, 1:2])

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

    total_loss = loss1a + loss1b + loss1c + loss1d + loss2a + loss2b + loss3 + loss4  # + loss_crit

    # Apply loss scaling if specified (helps with training stability)
    if loss_scale != 1.0:
        total_loss = total_loss * loss_scale
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

def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Run validation on the model.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device to run validation on

    Returns:
        Tuple of (validation_loss, validation_metrics)
    """
    model.eval()
    val_loss = 0.0
    val_metrics = {
        'Lclk_acc': 0.0,
        'no_fire': 0.0,
        'm_x_acc': 0.0,
        'm_y_acc': 0.0,
        'wasd_acc': 0.0,
        'crit_mse': 0.0,
    }

    num_batches = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_aux in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_aux = batch_aux.to(device)

            # Forward pass
            outputs = model(batch_x, aux_input=batch_aux)
            loss, _ = compute_custom_loss(outputs, batch_y)

            # Accumulate loss
            val_loss += float(loss.detach().cpu().item())

            # Accumulate metrics
            batch_metrics = compute_metrics(outputs, batch_y)
            for key in val_metrics:
                val_metrics[key] += batch_metrics[key]

            num_batches += 1

    # Average over all validation batches
    val_loss /= max(num_batches, 1)
    for key in val_metrics:
        val_metrics[key] /= max(num_batches, 1)

    model.train()
    return val_loss, val_metrics


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

    # clicks_out is already probabilities in [0,1]; threshold directly
    lclk_pred = (clicks_out[:, :, 0:1] >= 0.5).float()
    lclk_true = clicks_true[:, :, 0:1]
    lclk_acc = (lclk_pred == lclk_true).float().mean()

    no_fire = 1.0 - lclk_true.mean()

    mx_pred = mouse_x_out.argmax(dim=-1)
    mx_true = mouse_x_true.argmax(dim=-1)
    my_pred = mouse_y_out.argmax(dim=-1)
    my_true = mouse_y_true.argmax(dim=-1)
    m_x_acc = (mx_pred == mx_true).float().mean()
    m_y_acc = (my_pred == my_true).float().mean()

    # keys_out is already probabilities in [0,1]; threshold directly
    wasd_pred = (keys_out[:, :, 0:4] >= 0.5).float()
    wasd_true = keys_true[:, :, 0:4]
    wasd_acc = (wasd_pred == wasd_true).float().mean()

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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(args: argparse.Namespace) -> None:
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
    model = create_vivit_model(
        model_name=args.model_name,
        aux_input_on=args.use_prev_actions,
        temporal_depth=args.temporal_depth,
        freeze_backbone=args.freeze_backbone,
        use_dynamic_k=(not getattr(args, 'no_dynamic_k', False)),
        k_tau=args.k_tau,
    )
    model.to(device)

    # Simple AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    # No scheduler - keep learning rate constant

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = math.inf

    global_step = 0
    start_epoch = 1

    # Optionally resume
    if getattr(args, 'resume', None):
        start_epoch, global_step, best_loss = try_load_checkpoint(args.resume, model, optimizer, scaler, device)
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_batches = 0

        for batch_idx, (batch_x, batch_y, batch_aux) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_aux = batch_aux.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(batch_x, aux_input=(batch_aux if args.use_prev_actions else None))
                loss, loss_parts = compute_custom_loss(outputs, batch_y, loss_scale=args.loss_scale)

                # Dynamic-k sparsity penalty (encourage smaller k)
                if hasattr(model, 'get_k_penalty') and args.k_penalty_weight > 0.0:
                    loss = loss + args.k_penalty_weight * model.get_k_penalty()

            scaler.scale(loss).backward()

            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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
                # Optionally log mean k frames used
                if hasattr(model, 'get_k_penalty'):
                    p_mean = float(model.get_k_penalty().detach().cpu().item())
                    k_mean = 1.0 + p_mean * (N_TIMESTEPS - 1)
                    msg += f" k_mean {k_mean:.1f}"
                print(msg)

        avg_train_loss = running_loss / max(running_batches, 1)

        # Validation
        val_loss, val_metrics = validate(model, val_loader, device)
        # Include sparsity penalty in reported val loss for fairness
        if hasattr(model, 'get_k_penalty') and args.k_penalty_weight > 0.0:
            val_loss = float(val_loss) + float(args.k_penalty_weight) * float(model.get_k_penalty().detach().cpu().item())
        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch} done in {elapsed:.1f}s | train_loss {avg_train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | Lclk_acc {val_metrics['Lclk_acc']:.3f} "
            f"m_x_acc {val_metrics['m_x_acc']:.3f} m_y_acc {val_metrics['m_y_acc']:.3f} "
            f"wasd_acc {val_metrics['wasd_acc']:.3f} crit_mse {val_metrics['crit_mse']:.3f}"
        )

        # Save last checkpoint every epoch
        last_path = os.path.join(args.save_dir, f"{args.model_name}_last_iter2.pt")
        save_checkpoint(last_path, model, optimizer, scaler, epoch, global_step, best_loss, args)

        # Track and save best
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_path = os.path.join(args.save_dir, f"{args.model_name}_best_iter2.pt")
            save_checkpoint(best_path, model, optimizer, scaler, epoch, global_step, best_loss, args)
            print(f"updated best model: {best_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViViT CSGO behavioral cloning model")
    parser.add_argument('--model_name', type=str, default='vivit_vitb16', help="Model configuration name")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size")
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay for AdamW")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument('--loss_scale', type=float, default=1.0, help="Loss scaling factor for training stability")
    parser.add_argument('--starting_num', type=int, default=2, help="Lowest file number to use")
    parser.add_argument('--highest_num', type=int, default=190, help="Highest file number to use")
    parser.add_argument('--n_jitter', type=int, default=1, help="Temporal jitter frames")
    parser.add_argument('--is_mirror', action='store_true', help="Enable mirror augmentation")
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/AIGameBots/Counter-Strike_Behavioural_Cloning/dataset_dm_expert_dust2/', help="Dataset directory")
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'checkpoints'), help="Checkpoint directory")
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained ViT weights for spatial encoder")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze pretrained spatial encoder weights during training")
    parser.set_defaults(pretrained=True)
    parser.set_defaults(freeze_backbone=False)
    parser.add_argument('--num_workers', type=int, default=8, help="DataLoader workers")
    parser.add_argument('--log_every', type=int, default=5, help="Steps between logs")
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument('--temporal_depth', type=int, default=4, help="Temporal depth")
    # Dynamic-k controls
    parser.add_argument('--no_dynamic_k', action='store_true', help='Disable dynamic-k attention gating')
    parser.add_argument('--k_tau', type=float, default=0.5, help='Temperature for k gating sharpness (lower = harder)')
    parser.add_argument('--k_penalty_weight', type=float, default=0.01, help='Weight for dynamic-k sparsity penalty')
    # Toggle feeding previous actions as auxiliary input
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_prev_actions', dest='use_prev_actions', action='store_true', help="Feed previous actions as auxiliary input")
    group.add_argument('--no_prev_actions', dest='use_prev_actions', action='store_false', help="Disable feeding previous actions as auxiliary input")
    parser.set_defaults(use_prev_actions=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)