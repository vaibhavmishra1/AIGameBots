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


def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance loss for feature alignment.
    Expects shape (..., D); computes 1 - cosine_similarity along last dim and averages.
    """
    # Flatten any leading dims into one for similarity calculation per vector
    if a.dim() != b.dim():
        raise ValueError("cosine_loss tensors must have same number of dimensions")
    # Compute per-vector cosine similarity along last dim, then average
    cos_sim = F.cosine_similarity(a, b, dim=-1)
    return (1.0 - cos_sim).mean()

def warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """
    Linear warmup factor in [0,1] across warmup epochs.
    """
    if warmup_epochs <= 0:
        return 1.0
    return float(max(0, min(epoch, warmup_epochs)) / max(1, warmup_epochs))


def rkd_distance(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Relational Knowledge Distillation (distance) on token sequences.
    Preserves pairwise geometry among tokens by matching normalized
    pairwise distance matrices (per-sample).

    Args:
        student_tokens: (B, N, D_s) student token embeddings
        teacher_tokens: (B, N, D_t) teacher token embeddings
        eps: numerical stability constant

    Returns:
        Scalar loss (Smooth L1 between normalized pairwise distances)
    """
    if student_tokens.dim() != 3 or teacher_tokens.dim() != 3:
        raise ValueError("rkd_distance expects (B, N, D) tensors for both student and teacher")
    if student_tokens.size(0) != teacher_tokens.size(0) or student_tokens.size(1) != teacher_tokens.size(1):
        raise ValueError("Batch size and token count (B, N) must match between student and teacher")

    # Compute batched pairwise Euclidean distances across tokens: (B, N, N)
    with torch.no_grad():
        # Teacher distances (stopgrad by design)
        d_t = torch.cdist(teacher_tokens, teacher_tokens, p=2)  # (B, N, N)
        d_t = d_t / (d_t.mean(dim=(1, 2), keepdim=True) + eps)
    d_s = torch.cdist(student_tokens, student_tokens, p=2)  # (B, N, N)
    d_s = d_s / (d_s.mean(dim=(1, 2), keepdim=True) + eps)

    # Smooth L1 between full matrices (diagonals are zero and match)
    return F.smooth_l1_loss(d_s, d_t, reduction='mean')

