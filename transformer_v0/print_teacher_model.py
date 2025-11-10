import os
import argparse
import torch

from model import create_vivit_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_teacher(ckpt_path: str, temporal_depth: int = 8, use_prev_actions: bool = True) -> torch.nn.Module:
    """
    Create the ViT-B/16 teacher model and load weights from a checkpoint.
    Handles both full training checkpoints and raw state_dict files.
    """
    model = create_vivit_model(
        model_name='vivit_vitb16',
        aux_input_on=use_prev_actions,
        temporal_depth=temporal_depth,
        freeze_backbone=True,
    )
    device = get_device()
    model.to(device)
    model.eval()

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[info] Loading teacher checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and ('model_state_dict' in ckpt):
            state = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and ('state_dict' in ckpt):
            state = ckpt['state_dict']
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] Missing keys in teacher state_dict: {len(missing)}")
        if unexpected:
            print(f"[warn] Unexpected keys in teacher state_dict: {len(unexpected)}")
    else:
        if ckpt_path:
            print(f"[warn] Checkpoint not found at: {ckpt_path}")
        else:
            print("[warn] No checkpoint path provided; using randomly initialized (or pretrained backbone) weights.")

    return model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print ViViT teacher model architecture and stats")
    parser.add_argument(
        '--teacher_ckpt',
        type=str,
        default='/root/AIGameBots/transformer_v0/checkpoints/vivit_vitb16_best_vit_teacher_2.pt',
        help='Path to teacher checkpoint (.pt)'
    )
    parser.add_argument('--temporal_depth', type=int, default=8, help='Temporal transformer depth')
    parser.add_argument('--use_prev_actions', action='store_true', help='Teacher expects prev actions as aux input')
    parser.add_argument('--no_prev_actions', dest='use_prev_actions', action='store_false', help='Disable aux input')
    parser.set_defaults(use_prev_actions=True)
    args = parser.parse_args()

    teacher = load_teacher(args.teacher_ckpt, temporal_depth=args.temporal_depth, use_prev_actions=args.use_prev_actions)

    # Print architecture
    print("\n===== Teacher Model (ViT-B/16) =====")
    print(teacher)

    # Print parameter stats
    total_params = count_parameters(teacher)
    trainable_params = count_trainable_parameters(teacher)
    print("\n===== Parameter Counts =====")
    print(f"Total parameters       : {total_params:,}")
    print(f"Trainable parameters   : {trainable_params:,}")
    print(f"Frozen parameters      : {total_params - trainable_params:,}")


if __name__ == '__main__':
    main()


