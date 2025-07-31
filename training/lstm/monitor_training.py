#!/usr/bin/env python3
"""Monitor training progress by reading the log file."""

import time
import sys
import re
from pathlib import Path

def parse_log_line(line):
    """Parse a log line to extract metrics."""
    # Look for epoch progress
    epoch_match = re.search(r'Epoch (\d+):', line)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        
        # Extract progress percentage
        progress_match = re.search(r'(\d+)%', line)
        progress = int(progress_match.group(1)) if progress_match else 0
        
        # Extract loss values
        loss_match = re.search(r'loss=([\d.]+)', line)
        loss = float(loss_match.group(1)) if loss_match else None
        
        # Extract iterations
        iter_match = re.search(r'(\d+)/(\d+)', line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            total_iter = int(iter_match.group(2))
        else:
            current_iter = total_iter = 0
        
        # Extract speed
        speed_match = re.search(r'([\d.]+)it/s', line)
        speed = float(speed_match.group(1)) if speed_match else 0
        
        return {
            'type': 'training',
            'epoch': epoch,
            'progress': progress,
            'loss': loss,
            'current_iter': current_iter,
            'total_iter': total_iter,
            'speed': speed
        }
    
    # Look for validation
    if 'Validation:' in line:
        progress_match = re.search(r'(\d+)%', line)
        progress = int(progress_match.group(1)) if progress_match else 0
        return {
            'type': 'validation',
            'progress': progress
        }
    
    # Look for epoch summary
    if 'Epoch' in line and 'Summary:' in line:
        epoch_match = re.search(r'Epoch (\d+) Summary:', line)
        if epoch_match:
            return {
                'type': 'summary',
                'epoch': int(epoch_match.group(1))
            }
    
    # Look for metrics
    if 'Train Loss:' in line:
        loss_match = re.search(r'Train Loss: ([\d.]+)', line)
        if loss_match:
            return {
                'type': 'train_loss',
                'value': float(loss_match.group(1))
            }
    
    if 'Val Loss:' in line:
        loss_match = re.search(r'Val Loss: ([\d.]+)', line)
        if loss_match:
            return {
                'type': 'val_loss',
                'value': float(loss_match.group(1))
            }
    
    if 'Best Val Loss:' in line:
        loss_match = re.search(r'Best Val Loss: ([\d.]+)', line)
        if loss_match:
            return {
                'type': 'best_val_loss',
                'value': float(loss_match.group(1))
            }
    
    return None

def format_time(seconds):
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def monitor_log(log_path, refresh_interval=2):
    """Monitor the training log file."""
    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return
    
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop\n")
    
    last_position = 0
    current_epoch = -1
    train_loss = val_loss = best_val_loss = None
    start_time = time.time()
    
    try:
        while True:
            with open(log_path, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
            
            for line in new_lines:
                parsed = parse_log_line(line.strip())
                if parsed:
                    if parsed['type'] == 'training':
                        current_epoch = parsed['epoch']
                        if parsed['total_iter'] > 0:
                            eta_seconds = (parsed['total_iter'] - parsed['current_iter']) / max(parsed['speed'], 0.1)
                            eta_str = format_time(eta_seconds)
                            
                            # Clear line and print progress
                            sys.stdout.write('\r' + ' ' * 100 + '\r')
                            loss_str = f"{parsed['loss']:.4f}" if parsed['loss'] is not None else "N/A"
                            sys.stdout.write(
                                f"Epoch {parsed['epoch']} | "
                                f"Progress: {parsed['progress']}% "
                                f"[{parsed['current_iter']}/{parsed['total_iter']}] | "
                                f"Loss: {loss_str} | "
                                f"Speed: {parsed['speed']:.2f} it/s | "
                                f"ETA: {eta_str}"
                            )
                            sys.stdout.flush()
                    
                    elif parsed['type'] == 'validation':
                        sys.stdout.write('\r' + ' ' * 100 + '\r')
                        sys.stdout.write(f"Validation: {parsed['progress']}%")
                        sys.stdout.flush()
                    
                    elif parsed['type'] == 'train_loss':
                        train_loss = parsed['value']
                    
                    elif parsed['type'] == 'val_loss':
                        val_loss = parsed['value']
                    
                    elif parsed['type'] == 'best_val_loss':
                        best_val_loss = parsed['value']
                    
                    elif parsed['type'] == 'summary':
                        elapsed = time.time() - start_time
                        print(f"\n\nEpoch {parsed['epoch']} completed!")
                        print(f"Train Loss: {train_loss:.6f}" if train_loss is not None else "Train Loss: N/A")
                        print(f"Val Loss: {val_loss:.6f}" if val_loss is not None else "Val Loss: N/A")
                        print(f"Best Val Loss: {best_val_loss:.6f}" if best_val_loss is not None else "Best Val Loss: N/A")
                        print(f"Total time: {format_time(elapsed)}")
                        print("-" * 50)
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Last epoch: {current_epoch}")
        if best_val_loss is not None:
            print(f"Best validation loss: {best_val_loss:.6f}")
        else:
            print("Best validation loss: N/A")

if __name__ == "__main__":
    log_file = "training.log" if len(sys.argv) < 2 else sys.argv[1]
    monitor_log(log_file) 