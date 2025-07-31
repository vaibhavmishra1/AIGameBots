# LSTM Agent for Imitation Learning

This is a sophisticated LSTM-based neural network system for agent imitation learning in game environments. The system learns to predict agent actions (movement and rotation) based on observed features and previous actions.

## Architecture Overview

### Model Features
- **Advanced LSTM Architecture**: Multi-layer LSTM with attention mechanism
- **Uncertainty Estimation**: Optional uncertainty quantification for predictions
- **Residual Connections**: For better gradient flow
- **Action-Specific Heads**: Specialized output heads for each action type
- **Temporal Attention**: Multi-head attention for better temporal understanding

### Action Space
The model predicts 4 continuous actions:
1. `movedirection_x`: Movement along X-axis [-1, 1]
2. `movedirection_z`: Movement along Z-axis [-1, 1]
3. `lookrotation_x`: Rotation around X-axis [-10, 10]
4. `lookrotation_z`: Rotation around Z-axis [-10, 10]

## Installation

```bash
cd training/lstm
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:
```
dataset_dir/
├── features/
│   ├── {id}_features_chunk_0.npy  # Shape: (20, 20) - 20 frames, 20 features
│   ├── {id}_features_chunk_1.npy
│   └── ...
└── actions/
    ├── {id}_actions_chunk_0.npy   # Shape: (20, 10) - 20 frames, 10 actions
    ├── {id}_actions_chunk_1.npy
    └── ...
```

## Training

### Basic Training

```bash
python train.py --data_dir /path/to/dataset
```

### Advanced Training with Custom Parameters

```bash
python train.py \
    --data_dir /path/to/dataset \
    --batch_size 256 \
    --num_epochs 200 \
    --hidden_dim 512 \
    --num_lstm_layers 4 \
    --learning_rate 1e-3 \
    --model_type uncertainty \
    --use_attention \
    --use_residual \
    --use_amp
```

### Key Training Parameters

- `--batch_size`: Batch size for training (default: 256)
- `--hidden_dim`: Hidden dimension of LSTM (default: 512)
- `--num_lstm_layers`: Number of LSTM layers (default: 4)
- `--learning_rate`: Initial learning rate (default: 1e-3)
- `--dropout`: Dropout rate (default: 0.2)
- `--use_attention`: Enable attention mechanism
- `--use_residual`: Enable residual connections
- `--model_type`: Choose between 'standard' or 'uncertainty'
- `--use_amp`: Enable mixed precision training for faster training

### Resume Training

```bash
python train.py --resume checkpoints/lstm_agent/checkpoint_epoch_50.pt
```

## Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir runs/lstm_agent
```

## Inference

### Test on Dataset

```bash
python inference.py --checkpoint checkpoints/lstm_agent/best_model.pt --mode test
```

### Benchmark Performance

```bash
python inference.py --checkpoint checkpoints/lstm_agent/best_model.pt --mode benchmark
```

### Interactive Testing

```bash
python inference.py --checkpoint checkpoints/lstm_agent/best_model.pt --mode interactive
```

## Model Architecture Details

### AgentLSTM
- **Feature Encoder**: 2-layer MLP with normalization and dropout
- **Action Encoder**: Encodes previous actions for context
- **LSTM Core**: Multi-layer LSTM with optional dropout
- **Attention Module**: 8-head attention for temporal relationships
- **Output Heads**: Action-specific refinement heads

### AgentLSTMWithUncertainty
- Extends AgentLSTM with uncertainty estimation
- Provides confidence bounds for each predicted action
- Useful for safety-critical applications

## Training Tips

1. **Start Small**: Begin with a subset of data (--max_samples 10000) to verify training works
2. **Monitor Loss**: Watch for decreasing training/validation loss in TensorBoard
3. **Early Stopping**: Training automatically stops if validation loss doesn't improve for 30 epochs
4. **Learning Rate**: Uses cosine annealing with warm restarts by default
5. **Gradient Clipping**: Enabled by default to prevent exploding gradients

## Performance Optimization

1. **Mixed Precision**: Use `--use_amp` for faster training on modern GPUs
2. **Multi-GPU**: Set appropriate batch size for your GPU memory
3. **Data Loading**: Adjust `--num_workers` based on your CPU cores
4. **Caching**: Small datasets can be cached in memory for faster training

## Expected Results

With proper training on 800k samples:
- Training should converge within 50-100 epochs
- Validation loss should stabilize around 0.01-0.1 (depending on data quality)
- Inference speed: ~1000+ FPS on GPU, ~100+ FPS on CPU

## Troubleshooting

1. **Out of Memory**: Reduce batch_size or hidden_dim
2. **Slow Training**: Enable mixed precision with --use_amp
3. **Poor Performance**: Check data normalization, increase model capacity
4. **Overfitting**: Increase dropout, use weight decay, or get more data

## Integration with Unity

The trained model can be integrated with Unity using the inference API:

```python
from inference import AgentInference

# Initialize
agent = AgentInference('path/to/checkpoint.pt')

# Predict action
features = get_current_features()  # Your feature extraction
result = agent.predict(features)
actions = result['actions']

# Apply actions in Unity
move_x, move_z, look_x, look_z = actions
```

## Citation

If you use this code in your research, please cite:
```
@software{lstm_agent_2024,
  title={LSTM Agent for Imitation Learning},
  author={Your Name},
  year={2024}
}
``` 