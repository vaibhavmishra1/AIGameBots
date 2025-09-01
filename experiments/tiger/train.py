import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from processed_dataloader import ProcessedH5Dataset
from model_temporal_and_spatial import TemporalSpatialTransformer
from model_temporal import TemporalTransformer
from model_spatial import SpatialTransformer
from model_ann import SimpleFeedForward
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
    if len(batch[0]) == 3:  # temporal_and_spatial
        temporals, spatials, actions = zip(*batch)
        t = torch.stack(temporals, dim=0)
        s = torch.stack(spatials, dim=0)
        a = torch.stack(actions, dim=0)
        return t, s, a
    elif len(batch[0]) == 2:  # temporal_only or spatial_only
        x, actions = zip(*batch)
        x = torch.stack(x, dim=0)
        a = torch.stack(actions, dim=0)
        return x, a


def create_loaders(h5_path: str, group: str, batch_size: int, overfit_n: int, num_workers: int, experiment_type: str) -> Tuple[DataLoader, DataLoader, int]:
    ds = ProcessedH5Dataset(h5_path, group_name=group, return_numpy=False, experiment_type=experiment_type)
    total = len(ds)
    if total == 0:
        raise RuntimeError("Dataset is empty")

    if overfit_n > 0:
        n = min(overfit_n, total)
        sub = Subset(ds, list(range(n)))
        train_loader = DataLoader(sub, batch_size=min(batch_size, n), shuffle=True, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)
        val_loader = DataLoader(sub, batch_size=min(batch_size, n), shuffle=False, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)
        return train_loader, val_loader, n

    idx = list(range(total))
    split = int(0.9 * total)
    train_idx, val_idx = idx[:split], idx[split:]
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)
    return train_loader, val_loader, total


def normalize_targets(actions: torch.Tensor) -> torch.Tensor:
    # Model outputs dx,dy in [-1,1], rot in [-30,30].
    # Dataset already provides dx,dy in [-1,1], so pass through unchanged.
    return actions


def sanitize_tensor(x: torch.Tensor, name: str = "") -> torch.Tensor:
    if not torch.isfinite(x).all():
        num_nan = torch.isnan(x).sum().item()
        num_inf = torch.isinf(x).sum().item()
        print(f"Warning: non-finite values in {name}: nan={num_nan} inf={num_inf}. Replacing with 0.")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def train_one_epoch(model, loader, optimizer, device, experiment_type, huber: bool = False) -> float:
    model.train()
    criterion = nn.SmoothL1Loss() if huber else nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    if experiment_type == "tiger_temporal_and_spatial":
        for temporal, spatial, actions in tqdm(loader, desc="Train", leave=False):
            
            temporal = temporal.to(device)
            spatial = spatial.to(device)
            actions = actions.to(device)
            y = normalize_targets(actions)

            optimizer.zero_grad(set_to_none=True)
            pred = model(temporal, spatial)
            target = y
            
            loss = criterion(pred, target)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = temporal.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    elif experiment_type == "tiger_temporal_only":
        for temporal, actions in tqdm(loader, desc="Train", leave=False):
            temporal = temporal.to(device)
            actions = actions.to(device)
            y = normalize_targets(actions)
            
            optimizer.zero_grad(set_to_none=True)
            pred = model(temporal)
            target = y
            
            loss = criterion(pred, target)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = temporal.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    elif experiment_type == "tiger_spatial_only":
        for spatial, actions in tqdm(loader, desc="Train", leave=False):
            spatial = spatial.to(device)
            actions = actions.to(device)
            y = normalize_targets(actions)
            
            optimizer.zero_grad(set_to_none=True)
            pred = model(spatial)
            target = y
            
            loss = criterion(pred, target)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = spatial.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    elif experiment_type == "tiger_only_main_agent_current_features":
        for temporal, actions in tqdm(loader, desc="Train", leave=False):
            temporal = temporal.to(device)
            actions = actions.to(device)
            y = normalize_targets(actions)
            
            optimizer.zero_grad(set_to_none=True)
            pred = model(temporal)
            target = y
            
            loss = criterion(pred, target)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = temporal.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, device, experiment_type, huber: bool = False) -> float:
    model.eval()
    criterion = nn.SmoothL1Loss() if huber else nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    if experiment_type == "tiger_temporal_and_spatial":
        for temporal, spatial, actions in tqdm(loader, desc="Val", leave=False):
            temporal = temporal.to(device)
            spatial = spatial.to(device)
            actions = actions.to(device)
            y = normalize_targets(actions)
            pred = model(temporal, spatial)
            
            target = y
            loss = criterion(pred, target)
            bs = temporal.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    elif experiment_type == "tiger_temporal_only":
        for temporal, actions in tqdm(loader, desc="Val", leave=False):
            #temporal = sanitize_tensor(temporal.to(device), name="temporal")
            temporal = temporal.to(device)
            #actions = sanitize_tensor(actions.to(device), name="actions")
            actions = actions.to(device)
            y = normalize_targets(actions)
            #pred = sanitize_tensor(model(temporal), name="pred")
            pred = model(temporal)
            #target = sanitize_tensor(y[:, :2], name="target_xy")
            target = y[:, :2]
            loss = criterion(pred, target)
            bs = temporal.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    elif experiment_type == "tiger_spatial_only":
        for spatial, actions in loader:
            spatial = spatial.to(device)
            actions = actions.to(device)
            y = normalize_targets(actions)
            pred = model(spatial)
            target = y
            loss = criterion(pred, target)
            bs = spatial.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    elif experiment_type == "tiger_only_main_agent_current_features":
        for temporal, actions in loader:
            temporal = temporal.to(device)
            actions = actions.to(device)
            y = normalize_targets(actions)
            pred = model(temporal)
            target = y
            loss = criterion(pred, target)
            bs = temporal.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    return total_loss / max(1, total_count)


@torch.no_grad()
def print_samples(model, loader, experiment_type, device, k: int = 10) -> None:
    model.eval()
    shown = 0
    if experiment_type == "tiger_temporal_and_spatial":
        for temporal, spatial, actions in loader:
            temporal = temporal.to(device)
            spatial = spatial.to(device)
            pred = model(temporal, spatial).cpu()
            for i in range(temporal.size(0)):
                px, py, prot = pred[i].tolist()
                ax, ay, arot = actions[i].tolist()
                # Remember: targets are unnormalized for dx/dy
                print(f"[{shown:02d}] tgt=({ax:.4f},{ay:.4f},{arot:.4f}) pred=({px:.4f},{py:.4f},{prot:.4f})")
                shown += 1
                if shown >= k:
                    return
    elif experiment_type == "tiger_temporal_only":
        for temporal, actions in loader:
            temporal = temporal.to(device)
            pred = model(temporal).cpu()
            for i in range(temporal.size(0)):
                px, py = pred[i].tolist()
                ax, ay = actions[i].tolist()
                print(f"[{shown:02d}] tgt=({ax:.4f},{ay:.4f}) pred=({px:.4f},{py:.4f})")
                shown += 1
                if shown >= k:
                    return
    elif experiment_type == "tiger_spatial_only":
        for spatial, actions in loader:
            spatial = spatial.to(device)
            pred = model(spatial).cpu()
            for i in range(spatial.size(0)):
                px, py = pred[i].tolist()
                ax, ay = actions[i].tolist()
                print(f"[{shown:02d}] tgt=({ax:.4f},{ay:.4f}) pred=({px:.4f},{py:.4f})")
                shown += 1
                if shown >= k:
                    return
    elif experiment_type == "tiger_only_main_agent_current_features":
        for temporal, actions in loader:
            temporal = temporal.to(device)
            pred = model(temporal).cpu()
            for i in range(temporal.size(0)):
                px, py = pred[i].tolist()
                ax, ay = actions[i].tolist()
                print(f"[{shown:02d}] tgt=({ax:.4f},{ay:.4f}) pred=({px:.4f},{py:.4f})")
                shown += 1
                if shown >= k:
                    return
def select_model(experiment_type: str, d_model: int, temp_layers: int, temp_heads: int, spat_layers: int, spat_heads: int, dropout: int, max_time: int, max_agents: int):
    if experiment_type == "tiger_temporal_and_spatial":
        return TemporalSpatialTransformer(
        feature_dim=6,
        d_model=d_model,
        temp_layers=temp_layers,
        temp_heads=temp_heads,
        spat_layers=spat_layers,
        spat_heads=spat_heads,
        dropout=dropout,
        max_time=max_time,
        max_agents=max_agents
        )
    elif experiment_type == "tiger_temporal_only":
        return TemporalTransformer(
        feature_dim=6,
        d_model=d_model,
        num_layers=temp_layers,
        num_heads=temp_heads,
        dropout=dropout,
        max_time=max_time,
    )
    elif experiment_type == "tiger_spatial_only":
        return SpatialTransformer(
        feature_dim=6,
        d_model=d_model,
        num_layers=spat_layers,
        num_heads=spat_heads,
        dropout=dropout,
        max_agents=max_agents
    )
    elif experiment_type == "tiger_only_main_agent_current_features":
        return SimpleFeedForward(
        input_dim=6,
        hidden_dim=32,
        output_dim=2
    )

def main():
    parser = argparse.ArgumentParser(description="Train attention_v1 Transformer on processed H5 data")
    parser.add_argument("--h5_path", type=str, default="/Users/vaibhav/Desktop/processed_game_logs_attention_1.h5")
    parser.add_argument("--group", type=str, default="processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--temp_layers", type=int, default=2)
    parser.add_argument("--spat_layers", type=int, default=2)
    parser.add_argument("--temp_heads", type=int, default=8)
    parser.add_argument("--spat_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--include_rot", action="store_true")
    parser.add_argument("--huber", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join("training", "attention_v1", "checkpoints"))
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--experiment_type", type=str, default="tiger_temporal_and_spatial")
    parser.add_argument("--max_time", type=int, default=20)
    parser.add_argument("--max_agents", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"args: {args}")
    train_loader, val_loader, total = create_loaders(args.h5_path, args.group, args.batch_size, args.overfit, args.num_workers, args.experiment_type)
    print(f"Dataset samples: {total} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


    model = select_model(args.experiment_type, args.d_model, args.temp_layers, args.temp_heads, args.spat_layers, args.spat_heads, args.dropout, args.max_time, args.max_agents)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_ckpt = args.ckpt_path if args.ckpt_path else os.path.join(args.ckpt_dir, f"{args.experiment_type}_last.pt")
    best_ckpt = os.path.join(args.ckpt_dir, f"{args.experiment_type}_best.pt")
    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        ckpt_to_load = last_ckpt if os.path.isfile(last_ckpt) else args.ckpt_path
        if ckpt_to_load and os.path.isfile(ckpt_to_load):
            print(f"Resuming from {ckpt_to_load}")
            state = torch.load(ckpt_to_load, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            if "optimizer_state_dict" in state:
                try:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                except Exception:
                    print("Warning: optimizer state could not be loaded; continuing without it.")
            start_epoch = int(state.get("epoch", 0)) + 1
            best_val = float(state.get("best_val", best_val))
        else:
            print("--resume specified but no checkpoint found; starting fresh.")

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.experiment_type, huber=args.huber)
        val_loss = evaluate(model, val_loader, device, args.experiment_type, huber=args.huber)
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_val={best_val:.6f}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
        }, last_ckpt)
        if improved:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "args": vars(args),
            }, best_ckpt)

    print("\nSample predictions (val):")
    print_samples(model, val_loader, experiment_type=args.experiment_type, device=device, k=10)


if __name__ == "__main__":
    main()



"""
python3 train.py \
  --h5_path "dataset_exp_tiger_0p02_0p3_100000.h5" \
  --epochs 20 --batch_size 128 \
  --lr 2e-4 --num_workers 4 --huber --experiment_type tiger_temporal_and_spatial > output_temporal_and_spatial.txt 


python3 train.py \
  --h5_path "dataset_exp_tiger_0p02_0p3_100000.h5" \
  --epochs 20 --batch_size 128 \
  --lr 2e-4 --num_workers 4 --huber --experiment_type tiger_temporal_only > output_temporal_only.txt 


python3 train.py \
  --h5_path "dataset_exp_tiger_0p02_0p3_100000.h5" \
  --epochs 20 --batch_size 128 \
  --lr 2e-4 --num_workers 4 --huber --experiment_type tiger_spatial_only > output_spatial_only.txt 


"""
