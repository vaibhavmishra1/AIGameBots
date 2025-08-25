import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import build_model
from processed_dataloader import ProcessedH5Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn(batch):
    # Default collate is fine, but be explicit for clarity
    features, actions = zip(*batch)
    x = torch.stack(features, dim=0)  # (B, T, A, F)
    y = torch.stack(actions, dim=0)   # (B, 3)
    return x, y


def create_loaders(
    h5_path: str,
    group: str = "processed",
    batch_size: int = 8,
    overfit_n: int = 0,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset = ProcessedH5Dataset(h5_path, group_name=group, return_numpy=False)
    total = len(dataset)
    if total == 0:
        raise RuntimeError("Dataset appears to be empty.")

    if overfit_n > 0:
        n = min(overfit_n, total)
        indices = list(range(n))
        dataset = Subset(dataset, indices)
        train_loader = DataLoader(
            dataset,
            batch_size=min(batch_size, n),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        # For overfit, use the same data as validation to observe memorization
        val_loader = DataLoader(
            dataset,
            batch_size=min(batch_size, n),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader, n

    # Standard split if not overfitting
    indices = list(range(total))
    split = int(0.9 * total)
    train_idx, val_idx = indices[:split], indices[split:]
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, total


def train_one_epoch(model, loader, optimizer, device, loss_type: str = "mse", huber_delta: float = 0.1, weight_alpha: float = 0.0, weight_beta: float = 0.3) -> float:
    model.train()
    criterion = nn.MSELoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none", beta=huber_delta)
    total_loss = 0.0
    total_count = 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        # Exclude rotation from loss for now (use only first two components)
        if loss_type == "mse":
            per_elem = criterion(pred[:, :2], y[:, :2])  # (B,2)
        elif loss_type == "l1":
            per_elem = (pred[:, :2] - y[:, :2]).abs()
        else:  # huber / smooth L1
            per_elem = huber(pred[:, :2], y[:, :2])

        per_sample = per_elem.mean(dim=1)  # (B,)

        # Base weights = 1.0
        weights = torch.ones_like(per_sample)

        # Downweight cases where delta_x or delta_y is exactly/near zero
        eps = 1e-8
        zero_mask = (y[:, 0].abs() <= eps) | (y[:, 1].abs() <= eps)
        weights = torch.where(zero_mask, torch.full_like(per_sample, 0.001), weights)

        # Movement magnitude weighting to combat mean-collapse
        if weight_alpha > 0.0:
            mag = torch.linalg.vector_norm(y[:, :2], ord=2, dim=1)  # (B,)
            weights = weights * (1.0 + weight_alpha * torch.clamp(mag / max(weight_beta, 1e-6), max=1.0))

        loss = (per_sample * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, device, loss_type: str = "mse", huber_delta: float = 0.1, weight_alpha: float = 0.0, weight_beta: float = 0.3) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    huber = nn.SmoothL1Loss(reduction="none", beta=huber_delta)
    total_loss = 0.0
    total_count = 0
    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        # Evaluate using only translation components (exclude rotation)
        if loss_type == "mse":
            per_elem = criterion(pred[:, :2], y[:, :2])
        elif loss_type == "l1":
            per_elem = (pred[:, :2] - y[:, :2]).abs()
        else:
            per_elem = huber(pred[:, :2], y[:, :2])

        per_sample = per_elem.mean(dim=1)
        # Base weights = 1.0
        weights = torch.ones_like(per_sample)
        eps = 1e-8
        zero_mask = (y[:, 0].abs() <= eps) | (y[:, 1].abs() <= eps)
        weights = torch.where(zero_mask, torch.full_like(per_sample, 0.001), weights)
        if weight_alpha > 0.0:
            mag = torch.linalg.vector_norm(y[:, :2], ord=2, dim=1)
            weights = weights * (1.0 + weight_alpha * torch.clamp(mag / max(weight_beta, 1e-6), max=1.0))
        loss = (per_sample * weights).mean()
        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz
    return total_loss / max(1, total_count)


@torch.no_grad()
def print_sample_predictions(model, loader, device, num_samples: int = 10) -> None:
    model.eval()
    shown = 0
    for x, y in tqdm(loader, desc="Samples", leave=False):
        x = x.to(device)
        pred = model(x)
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        pred_cpu = pred.cpu()
        for i in range(x_cpu.size(0)):
            print(f"sample {shown:02d} | target={y_cpu[i].tolist()} pred={pred_cpu[i].tolist()}")
            shown += 1
            if shown >= num_samples:
                return


@torch.no_grad()
def eval_with_breakdown(model, loader, device, tag: str) -> None:
    model.eval()
    total = 0
    mse_xy = 0.0
    mse_rot = 0.0
    for x, y in tqdm(loader, desc=f"{tag} breakdown", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        diff_xy = (pred[:, :2] - y[:, :2]) ** 2
        mse_xy += diff_xy.mean(dim=1).sum().item()
        diff_rot = (pred[:, 2] - y[:, 2]) ** 2
        mse_rot += diff_rot.mean().item() * x.size(0)
        total += x.size(0)
    if total > 0:
        print(f"{tag} MSE(xy)={mse_xy/total:.6f}  MSE(rot)={mse_rot/total:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train imitation model on processed H5 data")
    parser.add_argument("--h5_path", type=str, default="/Users/vaibhav/Desktop/processed_game_logs.h5")
    parser.add_argument("--group", type=str, default="processed")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--overfit", type=int, default=16, help="Use N samples to overfit; 0 disables")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--agent_layers", type=int, default=2)
    parser.add_argument("--time_layers", type=int, default=2)
    parser.add_argument("--agent_heads", type=int, default=8)
    parser.add_argument("--time_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join("training", "ann3", "checkpoints"))
    parser.add_argument("--ckpt_path", type=str, default="", help="Optional explicit checkpoint path to resume from")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1", "huber"])  # to combat mean-collapse
    parser.add_argument("--huber_delta", type=float, default=0.1)
    parser.add_argument("--weight_alpha", type=float, default=0.2, help="movement magnitude weighting strength (0 disables)")
    parser.add_argument("--weight_beta", type=float, default=0.3, help="normalization for weighting")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, total = create_loaders(
        h5_path=args.h5_path,
        group=args.group,
        batch_size=args.batch_size,
        overfit_n=args.overfit,
        num_workers=args.num_workers,
    )
    print(f"Dataset samples: {total} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(
        feature_dim=13,
        d_model=args.d_model,
        agent_heads=args.agent_heads,
        agent_layers=args.agent_layers,
        time_heads=args.time_heads,
        time_layers=args.time_layers,
        dropout=args.dropout,
        max_time=64,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Checkpoints
    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_ckpt_path = args.ckpt_path if args.ckpt_path else os.path.join(args.ckpt_dir, "imitation_transformer_last.pt")
    best_ckpt_path = os.path.join(args.ckpt_dir, "imitation_transformer_best.pt")

    # Optionally resume
    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        ckpt_to_load = last_ckpt_path if os.path.isfile(last_ckpt_path) else (args.ckpt_path if args.ckpt_path else "")
        if ckpt_to_load and os.path.isfile(ckpt_to_load):
            print(f"Resuming from checkpoint: {ckpt_to_load}")
            state = torch.load(ckpt_to_load, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            if "optimizer_state_dict" in state:
                try:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                except Exception:
                    print("Warning: optimizer state could not be loaded; continuing without it.")
            start_epoch = int(state.get("epoch", 0)) + 1
            best_val = float(state.get("best_val", float("inf")))
        else:
            print("--resume specified but no checkpoint found; starting fresh.")

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_type=args.loss_type, huber_delta=args.huber_delta,
            weight_alpha=args.weight_alpha, weight_beta=args.weight_beta,
        )
        val_loss = evaluate(
            model, val_loader, device,
            loss_type=args.loss_type, huber_delta=args.huber_delta,
            weight_alpha=args.weight_alpha, weight_beta=args.weight_beta,
        )
        best_val = min(best_val, val_loss)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_val={best_val:.6f}")

        # Save last checkpoint each epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
        }, last_ckpt_path)

        # Save best checkpoint when improved
        if val_loss <= best_val:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "args": vars(args),
            }, best_ckpt_path)

    # Show a few predictions to verify overfitting behavior
    # Breakdown metrics on train/val to verify we're matching the intended targets
    eval_with_breakdown(model, train_loader, device, tag="Train")
    eval_with_breakdown(model, val_loader, device, tag="Val")

    print("\nPredictions on a few train samples:")
    print_sample_predictions(model, train_loader, device, num_samples=min(10, args.overfit if args.overfit > 0 else 10))
    print("\nPredictions on a few val samples:")
    print_sample_predictions(model, val_loader, device, num_samples=min(10, args.overfit if args.overfit > 0 else 10))

    # Save a final copy of best as a stable name for convenience
    final_best_path = os.path.join(args.ckpt_dir, "imitation_transformer.pt")
    if os.path.isfile(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location="cpu")
        torch.save(best_state, final_best_path)
        print(f"Saved best checkpoint to {final_best_path}")
    else:
        # Fallback: save last
        last_state = torch.load(last_ckpt_path, map_location="cpu") if os.path.isfile(last_ckpt_path) else {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 0,
            "best_val": best_val,
            "args": vars(args),
        }
        torch.save(last_state, final_best_path)
        print(f"Saved checkpoint to {final_best_path}")


if __name__ == "__main__":
    main()


"""
train 1:
python3 train_processed.py \
  --h5_path "/Users/vaibhav/Desktop/processed_game_logs.h5" \
  --overfit 0 --epochs 5 --batch_size 128 \
  --d_model 512 --agent_layers 3 --time_layers 2 \
  --agent_heads 8 --time_heads 8 \
  --lr 2e-4 --weight_decay 0.01 --num_workers 8 --dropout 0.1


 training on 512 samples to check for overfitting:

 python3 train_processed.py \
  --h5_path "/Users/vaibhav/Desktop/processed_game_logs.h5" \
  --overfit 10000 --epochs 100 --batch_size 64 \
  --d_model 256 --agent_layers 4 --time_layers 4 \
  --agent_heads 8 --time_heads 8 \
  --lr 1e-4 --num_workers 8
"""