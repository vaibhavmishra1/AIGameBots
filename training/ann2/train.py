import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NPYDataset
from model import create_model, get_loss_function, DualRegressionLoss
import os


# Hyperparameters
BATCH_SIZE = 256* 8
LEARNING_RATE = 0.0001
EPOCHS = 10
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# # Create dataset and dataloader
dataset = NPYDataset(data_path="/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/clash_squad_agent_trajectories_processed")

# Split dataset into train and validation sets with fixed seed for reproducibility
generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Get a sample batch from train and validation dataloaders
train_features, train_actions = next(iter(train_dataloader))
val_features, val_actions = next(iter(val_dataloader))

print("Training sample:")
print(f"Features shape: {train_features.shape}")
print(f"Actions shape: {train_actions.shape}")
print("\nValidation sample:") 
print(f"Features shape: {val_features.shape}")
print(f"Actions shape: {val_actions.shape}")

# Create model with appropriate task type
model = create_model( input_dim=360, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3)
model.to(DEVICE)

# # Get appropriate loss function
criterion = DualRegressionLoss(weight1=1.0, weight2=1.0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Dataset size: {len(dataset)}")
print(f"Device: {DEVICE}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Loss function: {criterion.__class__.__name__}")

# Training loop with best checkpoint saving
model.train()
best_val_loss = float('inf')
best_model_path = "models/best_model.pth"
os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    total_train_loss = 0.0
    num_train_batches = 0
    
    for batch_idx, (features, targets) in enumerate(train_dataloader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        num_train_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_train_loss = total_train_loss / num_train_batches
    
    # Validation phase
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for features, targets in val_dataloader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_val_loss += loss.item()
            num_val_batches += 1
    
    avg_val_loss = total_val_loss / num_val_batches
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save best model checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'hyperparameters': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'input_dim': 360,
                'hidden_dims': [512, 256, 128, 64],
                'dropout_rate': 0.3
            }
        }, best_model_path)
        print(f"New best model saved! Val Loss: {avg_val_loss:.4f}")

print("Training completed!")
print(f"Best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")

# # Test the model on a single sample
# model.eval()
# with torch.no_grad():
#     test_features, test_target = dataset[0]
#     test_features = test_features.unsqueeze(0).to(DEVICE)  # Add batch dimension
#     test_target = test_target.unsqueeze(0).to(DEVICE)
    
#     prediction = model(test_features)
    
#     print(f"\nTest sample:")
#     print(f"True target: {test_target.item():.4f}")
    
#     if task_type == 'classification':
#         binary_prediction = (prediction >= 0.5).float()
#         print(f"Predicted probability: {prediction.item():.4f}")
#         print(f"Binary prediction: {binary_prediction.item():.0f}")
#     else:
#         print(f"Predicted value: {prediction.item():.4f}")
#         print(f"Absolute error: {abs(prediction.item() - test_target.item()):.4f}")

