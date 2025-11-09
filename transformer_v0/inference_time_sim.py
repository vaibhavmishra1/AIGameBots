import time
import torch
import torch.nn as nn
from model import create_vivit_model, input_shape

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simulate_inference_time(model_name='vivit_vitb16', num_frames=1000, num_runs=10, batch_size=1):
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = create_vivit_model(model_name=model_name, aux_input_on=False)  # Disable aux for simplicity
    model.to(device)
    model.eval()

    # Derive N_TIMESTEPS from input_shape
    N_TIMESTEPS = input_shape[0]

    # Dummy data: batch of sequences (but we'll process as streaming)
    # Shape: (batch, timesteps, channels, height, width) - assuming preprocessed as in dataloader
    dummy_frame = torch.randn(batch_size, 1, 3, 224, 224, device=device)  # Single frame shape

    # Extract components for timing
    vit = model.spatial_encoder  # ViT feature extractor
    temporal = model.temporal_encoder  # TransformerEncoder
    temporal_norm = model.temporal_input_norm
    heads = nn.Sequential(model.shared_dense, model.keys_output)  # Representative head for timing

    # Warmup runs
    with torch.no_grad():
        for _ in range(5):
            _ = vit(dummy_frame.squeeze(1))  # (B, 3, H, W)
            dummy_seq = torch.randn(batch_size, N_TIMESTEPS, 768, device=device)
            _ = temporal(temporal_norm(dummy_seq))

    # Time ViT for 1 frame and for N_TIMESTEPS frames
    time_vit_1 = 0.0
    time_vit_96 = 0.0
    for _ in range(num_runs):
        start = time.time()
        _ = vit(dummy_frame.squeeze(1))  # Single frame per batch
        time_vit_1 += (time.time() - start) / batch_size

        start = time.time()
        dummy_96 = torch.randn(batch_size, N_TIMESTEPS, 3, 224, 224, device=device).view(batch_size * N_TIMESTEPS, 3, 224, 224)
        _ = vit(dummy_96)
        time_vit_96 += (time.time() - start) / batch_size

    time_vit_1 /= num_runs
    time_vit_96 /= num_runs

    # Time temporal for length N_TIMESTEPS and length 1 (approximate incremental)
    time_temporal_96 = 0.0
    time_temporal_1 = 0.0
    for _ in range(num_runs):
        dummy_seq_96 = torch.randn(batch_size, N_TIMESTEPS, 768, device=device)
        start = time.time()
        _ = temporal(temporal_norm(dummy_seq_96))
        time_temporal_96 += time.time() - start

        dummy_seq_1 = torch.randn(batch_size, 1, 768, device=device)
        start = time.time()
        _ = temporal(temporal_norm(dummy_seq_1))
        time_temporal_1 += time.time() - start

    time_temporal_96 /= num_runs
    time_temporal_1 /= num_runs

    # Time heads (per sequence, but similar for both)
    time_heads = 0.0
    for _ in range(num_runs):
        dummy_shared = torch.randn(batch_size, N_TIMESTEPS, 768, device=device)
        start = time.time()
        _ = heads(dummy_shared)
        time_heads += time.time() - start
    time_heads /= num_runs

    # Simulate streaming: process num_frames, with window of N_TIMESTEPS
    # Without caching: for each new frame after N_TIMESTEPS, recompute full N_TIMESTEPS
    total_time_no_cache = (time_vit_96 + time_temporal_96 + time_heads) * (num_frames - N_TIMESTEPS + 1)  # Approximate, per "step"

    # With caching: for each new frame, vit_1 + temporal_1 (incremental) + heads (full, but could be optimized)
    # Note: Heads are per-timestep, so for streaming output only last, but timing full for conservatism
    total_time_cache = (time_vit_1 + time_temporal_1 + time_heads) * (num_frames - N_TIMESTEPS + 1)

    # Per new frame times (after initial N_TIMESTEPS)
    num_new_frames = max(num_frames - N_TIMESTEPS, 0)
    time_per_frame_no_cache = total_time_no_cache / num_new_frames if num_new_frames > 0 else 0
    time_per_frame_cache = total_time_cache / num_new_frames if num_new_frames > 0 else 0

    fps_no_cache = 1 / time_per_frame_no_cache if time_per_frame_no_cache > 0 else float('inf')
    fps_cache = 1 / time_per_frame_cache if time_per_frame_cache > 0 else float('inf')

    print(f"\nEstimated time per new frame (no caching): {time_per_frame_no_cache * 1000:.2f} ms (~{fps_no_cache:.1f} FPS)")
    print(f"Estimated time per new frame (with KV caching): {time_per_frame_cache * 1000:.2f} ms (~{fps_cache:.1f} FPS)")
    print(f"\nNote: This is a simulation; real KV caching would require model modifications for incremental decoding. Times averaged over {num_runs} runs on {device}.")

if __name__ == "__main__":
    simulate_inference_time()
