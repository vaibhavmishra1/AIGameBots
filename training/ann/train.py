import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NPYDataset
from model import create_model, get_loss_function
import os

# Configuration
tasks = ["moveDirection_x", "moveDirection_y", "lookRotationDelta_x", "lookRotationDelta_y", "Attack", "Reload", "thrust", "crouch", "sprint", "slide"]

# Define task types
regression_tasks = ["moveDirection_x", "moveDirection_y", "lookRotationDelta_x", "lookRotationDelta_y"]
classification_tasks = ["Attack", "Reload", "thrust", "crouch", "sprint", "slide"]

# Select task (you can change this to train different tasks)
task = tasks[4]  # "moveDirection_x" (regression task)

# Determine task type
if task in regression_tasks:
    task_type = "regression"
elif task in classification_tasks:
    task_type = "classification"
else:
    raise ValueError(f"Unknown task: {task}")

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
EPOCHS = 10
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# # Create dataset and dataloader
# dataset = NPYDataset(data_path="/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad_partitioned_chunked/", task=task)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # Create model with appropriate task type
# model = create_model('standard', input_dim=500, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3, task_type=task_type)
# model.to(DEVICE)

# # Get appropriate loss function
# criterion = get_loss_function(task_type)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print(f"Training on task: {task}")
# print(f"Task type: {task_type}")
# print(f"Dataset size: {len(dataset)}")
# print(f"Device: {DEVICE}")
# print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# print(f"Loss function: {criterion.__class__.__name__}")

# # Training loop
# model.train()
# for epoch in range(EPOCHS):
#     total_loss = 0.0
#     num_batches = 0
    
#     for batch_idx, (features, targets) in enumerate(dataloader):
#         features = features.to(DEVICE)
#         targets = targets.to(DEVICE).unsqueeze(1)  # Add dimension for batch
        
#         # Forward pass
#         optimizer.zero_grad()
#         outputs = model(features)
#         loss = criterion(outputs, targets)
        
#         # Backward pass
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#         num_batches += 1
        
#         if batch_idx % 100 == 0:
#             print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
#     avg_loss = total_loss / num_batches
#     print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

# print("Training completed!")

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

# Function to train any task
def train_task(task_name: str, epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001):
    """
    Train a model for a specific task.
    
    Args:
        task_name: Name of the task to train
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    # Determine task type
    if task_name in regression_tasks:
        task_type = "regression"
    elif task_name in classification_tasks:
        task_type = "classification"
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    print(f"\n{'='*50}")
    print(f"Training {task_name} ({task_type})")
    print(f"{'='*50}")
    
    # Create dataset and dataloader
    dataset = NPYDataset(data_path="/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad_partitioned_chunked/", task=task_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = create_model('standard', input_dim=500, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3, task_type=task_type)
    model.to(DEVICE)
    
    # Loss function and optimizer
    criterion = get_loss_function(task_type)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Create models directory if it doesn't exist
            models_dir = "/Users/vaibhavmishra/Desktop/btx-game-aicode/AIGameAgents/training/ann/models"
            os.makedirs(models_dir, exist_ok=True)
            
            # Save the best model
            model_path = os.path.join(models_dir, f"{task_name}_best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'task_type': task_type,
                'model_config': {
                    'input_dim': 500,
                    'hidden_dims': [512, 256, 128, 64],
                    'dropout_rate': 0.3,
                    'task_type': task_type
                }
            }, model_path)
            print(f"Saved best model to {model_path} (Loss: {best_loss:.4f})")
    
    # Save final model
    final_model_path = os.path.join(models_dir, f"{task_name}_final_model.pth")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'task_type': task_type,
        'model_config': {
            'input_dim': 500,
            'hidden_dims': [512, 256, 128, 64],
            'dropout_rate': 0.3,
            'task_type': task_type
        }
    }, final_model_path)
    print(f"Saved final model to {final_model_path} (Loss: {avg_loss:.4f})")
    
    print(f"Training completed for {task_name}!")
    return model


def load_model(task_name: str, model_type: str = 'best'):
    """
    Load a trained model for a specific task.
    
    Args:
        task_name: Name of the task
        model_type: Type of model to load ('best' or 'final')
        
    Returns:
        Loaded model
    """
    models_dir = "training/ann/models"
    model_path = os.path.join(models_dir, f"{task_name}_{model_type}_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Create model with same configuration
    model_config = checkpoint['model_config']
    model = create_model(
        'standard',
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate'],
        task_type=model_config['task_type']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Loaded {model_type} model for {task_name} (Loss: {checkpoint['loss']:.4f})")
    return model


def evaluate_model(model, task_name: str, num_samples: int = 1000):
    """
    Evaluate a trained model on a subset of the dataset.
    
    Args:
        model: Trained model
        task_name: Name of the task
        num_samples: Number of samples to evaluate on
    """
    # Determine task type
    if task_name in regression_tasks:
        task_type = "regression"
    elif task_name in classification_tasks:
        task_type = "classification"
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Create dataset
    dataset = NPYDataset(data_path="/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad_partitioned_chunked/", task=task_name)
    
    # Evaluate on subset
    model.eval()
    total_loss = 0.0
    criterion = get_loss_function(task_type)
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            features, target = dataset[i]
            features = features.unsqueeze(0).to(DEVICE)
            target = target.unsqueeze(0).to(DEVICE)
            
            prediction = model(features)
            loss = criterion(prediction, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / min(num_samples, len(dataset))
    print(f"Evaluation on {task_name}: Average Loss = {avg_loss:.4f}")
    return avg_loss

# Example: Train multiple tasks
if __name__ == "__main__":
    # Train the selected task
    print("Training selected task...")
    
    # Train regression tasks and save models
    for task_name in regression_tasks:
        try:
            model = train_task(task_name, epochs=10, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
            
            # Evaluate the trained model
            print(f"\nEvaluating {task_name}...")
            evaluate_model(model, task_name, num_samples=1000)
            
        except Exception as e:
            print(f"Error training {task_name}: {e}")
            continue
    
    # Example of loading a saved model
    print("\n" + "="*50)
    print("Example: Loading saved models")
    print("="*50)
    
    # Try to load the best model for the first regression task
    if regression_tasks:
        try:
            loaded_model = load_model(regression_tasks[0], model_type='best')
            print(f"Successfully loaded model for {regression_tasks[0]}")
            
            # Test prediction with loaded model
            dataset = NPYDataset(data_path="/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad_partitioned_chunked/", task=regression_tasks[0])
            test_features, test_target = dataset[0]
            test_features = test_features.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                prediction = loaded_model(test_features)
                print(f"Test prediction: {prediction.item():.4f}")
                print(f"True target: {test_target:.4f}")
                
        except Exception as e:
            print(f"Error loading model: {e}")