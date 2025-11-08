import os
import sys
import time
import argparse
from typing import Tuple

import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity

# Ensure shared config (sizes) is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))
from config import (  # noqa: E402
    N_TIMESTEPS,
    csgo_img_dimension,
    n_keys,
    n_clicks,
    n_mouse_x,
    n_mouse_y,
)

from model import create_vivit_model  # noqa: E402


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def make_dummy_batch(batch_size: int, timesteps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W = csgo_img_dimension
    C = 3
    x = torch.rand(batch_size, timesteps, H, W, C)
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    prev_actions = torch.zeros(batch_size, timesteps, action_dim)
    return x, prev_actions


@torch.no_grad()
def time_forward(model: nn.Module, x: torch.Tensor, aux: torch.Tensor, device: torch.device, iters: int, warmup: int) -> float:
    # Warmup
    for _ in range(max(warmup, 0)):
        _ = model(x, aux_input=aux)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    # Timed
    t0 = time.time()
    for _ in range(max(iters, 1)):
        _ = model(x, aux_input=aux)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / max(iters, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ViViTCSGOModel forward pass")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=N_TIMESTEPS)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--temporal_depth', type=int, default=8)
    parser.add_argument('--temporal_d_model', type=int, default=512)
    parser.add_argument('--use_prev_actions', action='store_true', default=True)
    parser.add_argument('--no_prev_actions', dest='use_prev_actions', action='store_false')
    parser.add_argument('--trace', action='store_true', help='Run torch.profiler and print top ops')
    parser.add_argument('--streaming', action='store_true', help='Profile streaming T=1 with KV cache across timesteps')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model = create_vivit_model(
        aux_input_on=args.use_prev_actions,
        temporal_depth=args.temporal_depth,
        temporal_d_model=args.temporal_d_model,
        freeze_backbone=False,
    )
    model.to(device)
    model.eval()

    params = count_parameters(model)
    print(f"Parameters: {params/1e6:.2f} M")

    # Inputs
    x, aux = make_dummy_batch(args.batch_size, args.timesteps)
    x = x.to(device).float()
    aux = aux.to(device).float() if args.use_prev_actions else None

    # Latency/throughput for full context
    avg_s = time_forward(model, x, aux, device, iters=args.iters, warmup=args.warmup)
    fps = args.timesteps / avg_s
    print(f"[Batch {args.batch_size}, T={args.timesteps}] avg forward: {avg_s*1000:.2f} ms  |  seq FPS: {fps:.1f}")

    # Streaming T=1
    x1, aux1 = make_dummy_batch(args.batch_size, 1)
    x1 = x1.to(device).float()
    aux1 = aux1.to(device).float() if args.use_prev_actions else None
    # Non-stateful single step
    avg_s_1 = time_forward(model, x1, aux1, device, iters=max(args.iters*2, 20), warmup=args.warmup)
    print(f"[Batch {args.batch_size}, T=1 no-cache] avg forward: {avg_s_1*1000:.2f} ms  |  FPS: {1.0/avg_s_1:.1f}")

    if args.streaming:
        # Stateful streaming across N steps (simulate online rollout)
        steps = max(args.timesteps, 96)
        if hasattr(model, 'set_stateful'):
            model.set_stateful(True)
        # Warmup a few steps
        for _ in range(5):
            _ = model(x1, aux_input=aux1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            _ = model(x1, aux_input=aux1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        t1 = time.time()
        avg_step = (t1 - t0) / steps
        print(f"[Streaming stateful] avg step: {avg_step*1000:.2f} ms  |  step FPS: {1.0/avg_step:.1f}")
        if hasattr(model, 'set_stateful'):
            model.set_stateful(False)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        _ = model(x, aux_input=aux)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak CUDA memory (approx, one forward): {peak_mb:.1f} MB")

    if args.trace:
        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            _ = model(x, aux_input=aux)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        sort_key = 'cuda_time_total' if device.type == 'cuda' else 'cpu_time_total'
        print(prof.key_averages().table(sort_by=sort_key, row_limit=25))


if __name__ == '__main__':
    main()


