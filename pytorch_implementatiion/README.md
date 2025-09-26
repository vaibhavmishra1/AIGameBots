# PyTorch CSGO Behavioral Cloning

This directory contains a complete PyTorch implementation of the CSGO behavioral cloning system, including:

- **DataLoader**: Equivalent to the TensorFlow `DataGenerator` from the original CSGO training script
- **Model**: Full PyTorch implementation of the EfficientNet + ConvLSTM2D architecture
- **Training**: Complete training pipeline with custom loss functions and evaluation metrics

## Overview

This implementation provides feature parity with the original TensorFlow version while leveraging PyTorch's advantages for research and deployment.

## Architecture

The CSGO model uses the following architecture:

1. **EfficientNetB0 Backbone**: Pre-trained on ImageNet for feature extraction
2. **ConvLSTM2D Layers**: For spatio-temporal feature processing
3. **Optional LSTM Layer**: For additional temporal modeling
4. **Multi-head Output**: Separate heads for keys, clicks, mouse movements, and value prediction
5. **Custom Loss Function**: Combines binary cross-entropy, categorical cross-entropy, and MSE losses

### Model Variants

- `default`: Base model with 256 ConvLSTM2D filters
- `big`: Model with 512 ConvLSTM2D filters
- `drop`: Model with dropout (0.5) for regularization
- `extra`: Model with an additional ConvLSTM2D layer
- `LSTM`: Model with LSTM layer for temporal processing
- `randinit`: Model with randomly initialized EfficientNet weights

## Components

### DataLoader (`dataloader.py`)
- **HDF5 Data Loading**: Loads game frames and labels from HDF5 files
- **Data Augmentation**: Supports brightness, contrast, and mirroring augmentations
- **Reward Calculation**: Computes rewards based on kills, deaths, and shooting
- **Batch Processing**: Handles batched data loading with multiple workers
- **Memory Efficient**: Uses lazy loading to minimize memory usage

### Model (`model.py`)
- **EfficientNetB0 Backbone**: Pre-trained feature extractor
- **ConvLSTM2D Implementation**: Custom convolutional LSTM for spatio-temporal processing
- **Multi-head Architecture**: Separate prediction heads for different action types
- **Configurable Variants**: Multiple model configurations for different experiments
- **Auxiliary Input Support**: Optional auxiliary input for previous actions

### Training (`train_model.py`)
- **Custom Loss Function**: Reproduces the original TensorFlow loss implementation
- **Training Loop**: Complete training pipeline with validation
- **Model Checkpointing**: Save and load model weights and optimizer state
- **Evaluation Metrics**: Accuracy and loss tracking for different action types
- **GPU Support**: Automatic CUDA/MPS detection and optimization

## Data Format

- **Input**: Game screenshots (150×280×3) for N_TIMESTEPS (96 frames)
- **Output**: Action labels (51 dimensions) + reward + advantage placeholder
- **Shape**: `(batch_size, N_TIMESTEPS, 150, 280, 3)` for inputs, `(batch_size, N_TIMESTEPS, 53)` for targets

## Key Features

### Action Components
1. **Keyboard inputs** (11): W, A, S, D, Space, Ctrl, Shift, 1, 2, 3, R
2. **Mouse buttons** (2): Left click, Right click
3. **Mouse X movement** (23): Discretized mouse movements
4. **Mouse Y movement** (15): Discretized mouse movements
5. **Reward** (1): Calculated from kills, deaths, and shooting
6. **Advantage** (1): Placeholder for reinforcement learning

## Usage

### Basic DataLoader Usage

```python
from dataloader import create_data_loaders

# Create data loaders
data_loaders = create_data_loaders(
    batch_size=2,
    starting_num=1,
    highest_num=30,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Get training and validation loaders
training_loader = data_loaders[0]  # training_loader_full
validation_loader = data_loaders[1]  # validation_loader_full

# Iterate through batches
for batch_x, batch_y in training_loader:
    print(f"Input shape: {batch_x.shape}")   # (batch_size, 96, 150, 280, 3)
    print(f"Target shape: {batch_y.shape}")  # (batch_size, 96, 53)
    break
```

### Model Usage

```python
from model import create_model
import torch

# Create model
model = create_model(model_name='default', pretrained=False)

# Forward pass
batch_x = torch.randn(1, 96, 150, 280, 3)  # Example input
keys_out, clicks_out, mouse_x_out, mouse_y_out, value_out = model(batch_x)

# Or get concatenated output (equivalent to TensorFlow)
concatenated_output = model.get_output_concatenated(batch_x)
print(f"Output shape: {concatenated_output.shape}")  # (1, 96, 53)
```

### Training Usage

```python
from train_model import train_model

# Train model
train_model(
    model_name='default',
    batch_size=1,
    learning_rate=0.0001,
    num_epochs=10,
    save_dir='./checkpoints',
    pretrained=False,
    starting_num=1,
    highest_num=30
)
```

### Custom Training Loop

```python
from model import create_model
from dataloader import create_data_loaders
from train_model import custom_loss
import torch

# Setup
model = create_model('default', pretrained=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
data_loaders = create_data_loaders(batch_size=1, starting_num=1, highest_num=5)

# Training loop
model.train()
for epoch in range(10):
    for batch_x, batch_y in data_loaders[0]:
        optimizer.zero_grad()

        # Forward pass
        pred = model.get_output_concatenated(batch_x)
        loss = custom_loss(pred, batch_y, n_keys=11, n_clicks=2, n_mouse_x=23, n_mouse_y=15)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Custom Dataset

```python
from dataloader import CSGODataset, CSGODataLoader

# Create custom dataset
data_list = ["1-0", "1-10", "2-0", "2-10"]  # Custom data samples
dataset = CSGODataset(
    data_list=data_list,
    folder_name='/path/to/hdf5/files',
    n_jitter=20,
    is_mirror=True,
    transform=True
)

# Create data loader
data_loader = CSGODataLoader(
    data_list=data_list,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## Data Augmentation

The dataloader supports the following augmentations:

- **Brightness adjustment**: Randomly adjusts image brightness (0.7-1.3x)
- **Contrast adjustment**: Randomly adjusts image contrast (0.7-1.3x)
- **Mirroring**: Randomly flips images horizontally and adjusts corresponding actions

```python
# Enable all augmentations
dataset = CSGODataset(
    data_list=data_list,
    is_mirror=True,  # Enable mirroring
    transform=True   # Enable brightness/contrast
)
```

## Model Integration

The dataloader works seamlessly with PyTorch models:

```python
import torch
import torch.nn as nn

class CSGOModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more layers
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=..., hidden_size=256, batch_first=True)

        # Output layers
        self.output = nn.Linear(256, 53)

    def forward(self, x):
        batch_size, seq_len, height, width, channels = x.shape

        # Reshape for CNN processing
        x_reshaped = x.view(-1, channels, height, width)
        features = self.cnn(x_reshaped)
        features = features.view(batch_size, seq_len, -1)

        # LSTM processing
        lstm_out, _ = self.lstm(features)
        output = self.output(lstm_out[:, -1, :])  # Use last timestep

        return output

# Training loop
model = CSGOModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for inputs, targets in training_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss (example - adjust based on your needs)
        loss = criterion(outputs, targets[:, -1, :])  # Use last timestep target

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()".4f"}")
```

## Configuration

Key parameters from `config.py`:

- `N_TIMESTEPS = 96`: Number of frames per sequence
- `csgo_img_dimension = (150, 280)`: Image dimensions
- `n_keys = 11`: Number of keyboard inputs
- `n_clicks = 2`: Number of mouse buttons
- `n_mouse_x = 23`: Number of discrete mouse X positions
- `n_mouse_y = 15`: Number of discrete mouse Y positions

## File Structure

- `dataloader.py`: Main dataloader implementation
- `model.py`: PyTorch model implementation with EfficientNet + ConvLSTM2D architecture
- `train_model.py`: Complete training pipeline with custom loss and evaluation
- `test_model.py`: Test script for model functionality and training
- `test_dataloader.py`: Test script for dataloader functionality
- `README.md`: This documentation file

## Performance Tips

1. **Use multiple workers**: Set `num_workers > 0` for parallel data loading
2. **Enable pin_memory**: Set `pin_memory=True` when using GPU training
3. **Batch size**: Adjust based on your GPU memory
4. **Data augmentation**: Enable for better generalization

## Compatibility

This implementation is compatible with:
- Python 3.8+
- PyTorch 1.9+
- CUDA (for GPU acceleration)
- HDF5 files generated by the original CSGO data collection pipeline

## Testing

Run the test scripts to verify everything works correctly:

### Test DataLoader
```bash
cd pytorch_implementatiion
python3 test_dataloader.py
```

### Test Model and Training
```bash
cd pytorch_implementatiion
python3 test_model.py
```

### Test Scripts Cover:
1. **DataLoader tests**: Basic functionality, data loading, and augmentation
2. **Model tests**: Model creation, forward pass, and different architectures
3. **Training tests**: Training loop integration and loss computation
4. **Data validation**: Shape verification and range checking

### Quick Test
```bash
cd pytorch_implementatiion
python3 -c "
from model import create_model
from dataloader import create_data_loaders
import torch

# Test model creation
model = create_model('default', pretrained=False)
print('✓ Model created successfully')

# Test data loading
loaders = create_data_loaders(batch_size=1, starting_num=1, highest_num=1, num_workers=0)
print('✓ DataLoader created successfully')

# Test forward pass
x = torch.randn(1, 96, 150, 280, 3)
output = model.get_output_concatenated(x)
print(f'✓ Forward pass successful, output shape: {output.shape}')
"

## Migration from TensorFlow

If you're migrating from the TensorFlow implementation:

### Data Loading
1. **DataGenerator → CSGODataset/CSGODataLoader**: Replace `DataGenerator` with `CSGODataset` or `CSGODataLoader`
2. **Batch iteration**: Same pattern with `for batch_x, batch_y in dataloader`
3. **Data shapes**: Same tensor shapes as TensorFlow version
4. **Augmentation**: Same augmentation options available
5. **Reward calculation**: Same reward function used

### Model Architecture
1. **Keras Model → PyTorch CSGOModel**: Use `create_model()` factory function
2. **Model variants**: Same naming convention (`'default'`, `'big'`, `'drop'`, etc.)
3. **EfficientNet**: Same pretrained weights option available
4. **ConvLSTM2D**: Custom implementation maintains same functionality
5. **Multi-head output**: Same output structure with separate heads

### Training
1. **Training loop**: Use `train_model()` function or implement custom loop
2. **Custom loss**: `custom_loss()` function replicates TensorFlow loss
3. **Optimizer**: Same Adam optimizer with same learning rate (0.0001)
4. **Checkpoints**: PyTorch checkpoint format with model and optimizer state
5. **Validation**: Built-in validation during training

### Key Differences
- **Device handling**: Automatic GPU detection and tensor movement
- **Gradient handling**: Manual zero_grad() required
- **Loss computation**: More explicit tensor operations
- **Model saving**: Native PyTorch checkpoint format

## Troubleshooting

### Common Issues

#### DataLoader Issues
1. **Import errors**: Make sure the config path is correct (check sys.path in imports)
2. **HDF5 file not found**: Verify the `folder_name` parameter points to correct directory
3. **Memory issues**: Reduce batch size or number of workers
4. **GPU issues**: Set `pin_memory=False` if having GPU memory issues

#### Model Issues
1. **CUDA out of memory**: Try smaller batch size or disable pretrained weights
2. **Shape mismatches**: Verify input tensor shape matches expected (batch, timesteps, height, width, channels)
3. **ConvLSTM2D errors**: Ensure input features match expected dimensions
4. **Gradient issues**: Check for NaN/inf values in inputs or disable pretrained weights

#### Training Issues
1. **Loss is NaN**: Check for invalid inputs or reduce learning rate
2. **Model not learning**: Verify loss computation and gradient flow
3. **Checkpoint loading errors**: Ensure model architecture matches saved checkpoint
4. **Slow training**: Enable mixed precision training or use larger batch sizes

### Debug Mode

#### DataLoader Debugging
```python
data_loader = CSGODataLoader(
    data_list=data_list,
    num_workers=0,        # Disable multiprocessing
    pin_memory=False,     # Disable memory pinning
    shuffle=False         # Disable shuffling for consistent results
)
```

#### Model Debugging
```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Use smaller model for debugging
model = create_model('default', pretrained=False)

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item()}")
```

#### Training Debugging
```python
# Enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

## License

This implementation is part of the CSGO behavioral cloning project. See the main project README for license information.
