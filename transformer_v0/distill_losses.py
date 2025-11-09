import torch
import torch.nn.functional as F
from typing import Tuple


def bce_probs(student_probs: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    """
    Binary cross entropy between two probability tensors in [0,1].
    Shapes are broadcast-compatible; reduction is mean.
    """
    # Disable AMP autocast to avoid BCE autocast safety error on probability inputs
    with torch.amp.autocast(device_type='cuda', enabled=False):
        student = student_probs.float()
        teacher = teacher_probs.float()
        return F.binary_cross_entropy(student, teacher, reduction='mean')


def _soften_distribution(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature to a categorical probability distribution using log-softmax scaling.
    """
    eps = 1e-8
    probs = probs.clamp(min=eps, max=1.0)
    log_probs = torch.log(probs)
    softened = torch.softmax(log_probs / temperature, dim=-1)
    return softened


def kl_divergence_probs(
    teacher_probs: torch.Tensor,
    student_probs: torch.Tensor,
    temperature: float = 2.0
) -> torch.Tensor:
    """
    KL( teacher || student ) for categorical distributions provided as probabilities.
    Applies temperature scaling in log space and multiplies by T^2 as in Hinton et al.
    """
    t_soft = _soften_distribution(teacher_probs, temperature)
    s_soft = _soften_distribution(student_probs, temperature)
    eps = 1e-8
    kl = (t_soft * (torch.log(t_soft + eps) - torch.log(s_soft + eps))).sum(dim=-1).mean()
    return (temperature ** 2) * kl


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error with mean reduction."""
    return F.mse_loss(a, b, reduction='mean')


def warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """
    Linear warmup factor in [0,1] across warmup epochs.
    """
    if warmup_epochs <= 0:
        return 1.0
    return float(max(0, min(epoch, warmup_epochs)) / max(1, warmup_epochs))


