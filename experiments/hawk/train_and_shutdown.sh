#!/bin/bash

# Script to train model and automatically shutdown VM when complete
# This helps save GCP costs by shutting down immediately after training

set -e  # Exit on any error

echo "=== Starting Training with Auto-Shutdown ==="
echo "Start time: $(date)"
echo "Training will automatically shutdown VM when complete..."
echo ""

# Change to the correct directory
cd /home/vaibhav/AIGameBots/experiments/hawk

# Run the training command
echo "Starting training..."
python3 train.py \
  --h5_path "dataset_exp_hawk_0p02_0p3_1000000.h5" \
  --epochs 20 --batch_size 3072 \
  --lr 2e-4 --num_workers 3 --huber --temp_layers 4 --spat_layers 4 --temp_heads 8 --spat_heads 8 --d_model 512 \
  --experiment_type hawk_temporal_and_spatial > output_temporal_and_spatial.txt 2>&1

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Training completed successfully! ==="
    echo "End time: $(date)"
    echo "Output saved to: output_temporal_and_spatial.txt"
    echo "Checkpoints saved to: training/attention_v1/checkpoints/"
    echo ""
    echo "Shutting down VM in 30 seconds..."
    echo "Press Ctrl+C to cancel shutdown if needed."
    
    # Give user 30 seconds to cancel if they want to check results
    sleep 30
    
    # Shutdown the VM
    echo "Shutting down VM now..."
    sudo shutdown -h now
else
    echo ""
    echo "=== Training failed with exit code $? ==="
    echo "End time: $(date)"
    echo "Check output_temporal_and_spatial.txt for errors"
    echo "VM will NOT shutdown due to training failure."
    echo "You can manually investigate the issue."
fi
