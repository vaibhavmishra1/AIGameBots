import os
import sys
from typing import Tuple

import torch

# Import shared config for shapes/counts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Counter-Strike_Behavioural_Cloning'))
from config import (  # noqa: E402
    N_TIMESTEPS,
    csgo_img_dimension,
    n_keys,
    n_clicks,
    n_mouse_x,
    n_mouse_y,
)

# Use local training helpers to avoid cross-package dependencies
from train import compute_custom_loss, compute_metrics  # noqa: E402

from model import create_vivit_model  # noqa: E402


def make_dummy_batch(batch_size: int = 1, timesteps: int = N_TIMESTEPS) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, W = csgo_img_dimension
    C = 3
    # Inputs in range [0,1] to mimic dataloader normalization; model will re-normalize to ImageNet
    x = torch.rand(batch_size, timesteps, H, W, C)

    # Targets: [keys(11), clicks(2), mouse_x(23 one-hot), mouse_y(15 one-hot), reward, advantage]
    action_dim = n_keys + n_clicks + n_mouse_x + n_mouse_y
    y = torch.zeros(batch_size, timesteps, action_dim + 2)

    # Random binary keys/clicks and categorical mouse
    keys = (torch.rand(batch_size, timesteps, n_keys) > 0.7).float()
    clicks = (torch.rand(batch_size, timesteps, n_clicks) > 0.8).float()

    mx_idx = torch.randint(0, n_mouse_x, (batch_size, timesteps))
    my_idx = torch.randint(0, n_mouse_y, (batch_size, timesteps))
    mx = torch.zeros(batch_size, timesteps, n_mouse_x)
    my = torch.zeros(batch_size, timesteps, n_mouse_y)
    mx.scatter_(2, mx_idx.unsqueeze(-1), 1.0)
    my.scatter_(2, my_idx.unsqueeze(-1), 1.0)

    # Assemble y
    y[:, :, 0:n_keys] = keys
    y[:, :, n_keys:n_keys + n_clicks] = clicks
    y[:, :, n_keys + n_clicks:n_keys + n_clicks + n_mouse_x] = mx
    y[:, :, n_keys + n_clicks + n_mouse_x:n_keys + n_clicks + n_mouse_x + n_mouse_y] = my
    # reward and advantage = 0 by default

    # Aux input: previous actions (without reward/advantage); shift by 1 timestep
    prev_actions = torch.zeros(batch_size, timesteps, action_dim)
    prev_actions[:, 1:, :] = y[:, :-1, :action_dim]

    return x, y, prev_actions


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    model = create_vivit_model(aux_input_on=True)
    model.to(device)
    model.train()

    x, y, aux = make_dummy_batch(batch_size=1, timesteps=N_TIMESTEPS)
    x, y, aux = x.to(device), y.to(device), aux.to(device)

    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        outputs = model(x, aux_input=aux)
        loss, parts = compute_custom_loss(outputs, y)

    print("Forward OK. Loss:", float(loss.detach().cpu().item()))
    metrics = compute_metrics(outputs, y)
    print("Metrics:", {k: round(v, 3) for k, v in metrics.items()})


if __name__ == '__main__':
    main()


