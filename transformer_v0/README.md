# ViViT-based CSGO Action Model

This module provides a ViViT-style video transformer for Counter-Strike behavioral cloning. It mirrors the existing ConvLSTM pipeline interfaces (inputs, heads, loss, metrics, dataloader) so you can swap models without changing the rest of the training stack.

## Key Points
- Input shape: `(batch, 96, 150, 280, 3)` from `Counter-Strike_Behavioural_Cloning/config.py`.
- Backbone: Spatial Transformer per-frame → Temporal Transformer across frames.
- Heads per timestep: keys(11, sigmoid), clicks(2, sigmoid), mouse_x(23, softmax), mouse_y(15, softmax), value(1, linear).
- Aux input: previous action vector per timestep (optional) projected and concatenated before the shared MLP.
- Normalization: ImageNet mean/std in-model (no dataloader changes needed).

## Train
```bash
cd transformer_v0
python3 train.py \
  --model_name vivit_default \
  --batch_size 1 \
  --epochs 10 \
  --lr 1e-4 \
  --starting_num 2 \
  --highest_num 190 \
  --data_dir /Users/vaibhav/Desktop/AIGameBots/Counter-Strike_Behavioural_Cloning/dataset_dm_expert_dust2/ \
  --use_prev_actions \
  --num_workers 1
```
- Checkpoints: saved to `transformer_v0/checkpoints/{model_name}_best.pt`.
- Flags:
  - `--use_prev_actions` or `--no_prev_actions`
  - `--is_mirror` (data augmentation)
  - `--n_jitter` (temporal jitter)
  - `--num_workers` (loader workers)

## Quick Forward Test
```python
import torch
from model import create_vivit_model

model = create_vivit_model(aux_input_on=True)
x = torch.randn(1, 96, 150, 280, 3)
action_dim = 11 + 2 + 23 + 15
aux = torch.randn(1, 96, action_dim)

out = model.get_output_concatenated(x, aux)
print(out.shape)  # (1, 96, 52 or 53 depending on value head)
```

## Dataloader Reuse
This package re-exports the existing PyTorch dataloader:
```python
from dataloader import create_data_loaders
train_loader, val_loader = create_data_loaders(batch_size=1, starting_num=2, highest_num=190)
```

## Stateful Inference
Enable temporal caching for streaming/timestep-by-timestep inference:
```python
model.set_stateful(True)
# call model(...) with T=1 repeatedly
```

## Notes
- Loss/metrics/validate are reused from the ConvLSTM pipeline to keep parity.
- No extra dependencies required. Optionally, you can add `timm` later for pretrained ViT weights.

## Knowledge Distillation (Teacher → Student)
- Teacher checkpoint path (default): `/root/AIGameBots/transformer_v0/checkpoints/vivit_vitb16_best_vit_teacher_2.pt`
- Run KD with a compact student (baseline KD):
```bash
cd transformer_v0
python3 train.py --kd --student_model deit_tiny --teacher_ckpt /root/AIGameBots/transformer_v0/checkpoints/vivit_vitb16_best_vit_teacher_2.pt --lr 5e-5  --use_prev_actions
```
- New: Relational Token Distillation (RTD)
  - Preserves the temporal token geometry by matching teacher and student pairwise token distance matrices (RKD), on top of logits KD and feature cosine alignment.
  - Use:
```bash
cd transformer_v0
python3 train.py --kd --student_model deit_tiny \
  --teacher_ckpt /root/AIGameBots/transformer_v0/checkpoints/vivit_vitb16_best_vit_teacher_2.pt \
  --lr 5e-5 --use_prev_actions \
  --distill_method rkd --alpha_kd 0.1 --beta_kd 0.05 --gamma_kd_rkd 0.5 --temp_kd 4.0
```
- KD hyperparameters (defaults in code):
  - `--alpha_kd` (response KD, default 0.1)
  - `--beta_kd` (feature cosine KD, default 0.05)
  - `--gamma_kd_rkd` (relational KD weight, default 0.5)
  - `--temp_kd` (KL temperature, default 4.0)
  - `--kd_warmup_epochs` (default 5)
- Outputs:
  - Last: `checkpoints/student_last_kd.pt`
  - Best: `checkpoints/student_best_kd.pt`
