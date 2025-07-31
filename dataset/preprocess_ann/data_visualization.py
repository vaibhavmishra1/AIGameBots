import os
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Folder that contains the generated match data
folder_path = '/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad_agent_partition'

# Get list of all JSON files in the folder (sorted for reproducibility)
json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

# Global counters for tracking across all files
total_frames_all_files = 0
total_active_frames_all_files = 0

def visualize_data(frames):
    """Animate agent and target positions for a single JSON recording.

    Parameters
    ----------
    frames : list[dict]
        Each element must contain
            - 'agentPosition': {'x', 'y', 'z'}
            - 'targetPosition': {'x', 'y', 'z'}
    The function blocks until the animation window closes, ensuring
    sequential processing of multiple JSON files.
    """
    print("loading simulation ")
    if not frames:
        print("Warning: empty data; skipping visualization")
        return

    # ------------------------------------------------------------------
    # Extract x and z (ignore y) positions for faster access
    agent_pos = [(f['agentPosition']['x'], f['agentPosition']['z']) for f in frames]
    target_pos = [(f['targetPosition']['x'], f['targetPosition']['z']) for f in frames]
    
    # Calculate relative positions and find min/max
    rel_x = [t[0] - a[0] for t, a in zip(target_pos, agent_pos)]
    rel_z = [t[1] - a[1] for t, a in zip(target_pos, agent_pos)]
    
    min_x_diff = min(rel_x)
    max_x_diff = max(rel_x)
    min_z_diff = min(rel_z)
    max_z_diff = max(rel_z)
    
    print(f"X difference range: {min_x_diff:.2f} to {max_x_diff:.2f}")
    print(f"Z difference range: {min_z_diff:.2f} to {max_z_diff:.2f}")

    # Forward direction vectors (normalized) projected onto X–Z plane
    agent_fwd = [(f['agentForward']['x'], f['agentForward']['z']) for f in frames]
    target_fwd = [(f['targetForward']['x'], f['targetForward']['z']) for f in frames]

    xs_a, zs_a = zip(*agent_pos)
    xs_t, zs_t = zip(*target_pos)

    # Create a 2-D figure and configure axes with high DPI
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    
    # Enable grid
    ax.grid(True, alpha=0.3)

    # Set axis limits based on combined min/max to keep both entities in view
    # ax.set_xlim( min(min_x_diff, -100), max(max_x_diff,100))
    # ax.set_ylim(min(min_z_diff, -100), max(max_z_diff,100))
    # Note: Axis limits will be set dynamically in each frame

    # Initialize the scatter objects for agent (blue) and target (red)
    agent_scatter = ax.scatter([], [], c='blue', label='Agent', s=100)
    target_scatter = ax.scatter([], [], c='red', label='Target', s=100)

    # Initialize variables to track arrows and text
    agent_arrow = None
    target_arrow = None
    agent_text = None
    target_text = None
    agent_info_text = None
    los_line = None
    arrow_scale = 20  # Tune this if arrows appear too small/large
    ax.legend()

    def update(idx):
        """Update scatter positions each frame and close window at the end."""
    
        nonlocal agent_arrow, target_arrow, agent_text, target_text, agent_info_text, los_line
        # print("idx = ", idx)
        # Remove previous arrows and text if they exist
        if agent_arrow:
            agent_arrow.remove()
        if target_arrow:
            target_arrow.remove()
        if agent_text:
            agent_text.remove()
        if target_text:
            target_text.remove()
        if agent_info_text:
            agent_info_text.remove()
        if los_line:
            los_line.remove()
        
        # Update 2-D scatter offsets
        agent_scatter.set_offsets([[xs_a[idx], zs_a[idx]]])
        target_scatter.set_offsets([[xs_t[idx], zs_t[idx]]])
        
        # Set dynamic axis limits based on current positions with padding
        current_x_positions = [xs_a[idx], xs_t[idx]]  # Agent at 0, target relative
        current_z_positions = [zs_a[idx], zs_t[idx]]
        
        padding = 50  # Adjust this for more/less zoom
        x_min, x_max = min(current_x_positions) - padding, max(current_x_positions) + padding
        z_min, z_max = min(current_z_positions) - padding, max(current_z_positions) + padding
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        
        # Update title with game_id and game_time from current frame
        game_id = frames[idx].get('game_id', 'Unknown')
        game_time = frames[idx].get('game_time', 'Unknown')
        
        # Format game_time to 2 decimal places if it's a number
        if isinstance(game_time, (int, float)):
            game_time_str = f"{game_time:.2f}"
        else:
            game_time_str = str(game_time)
            
        ax.set_title(f'Game ID: {game_id} | Game Time: {game_time_str} | Frame: {idx}', 
                    fontsize=6)
        
        # Draw green line if line of sight is true
        is_los = frames[idx].get('islos', False)
        if is_los:
            los_line = ax.plot([xs_a[idx], xs_t[idx]], [zs_a[idx], zs_t[idx]], 
                              color='green', linewidth=2, alpha=0.7, label='Line of Sight')[0]
        
        # print("target x = ", xs_t[idx] - xs_a[idx])
        # print("target z = ", zs_t[idx] - zs_a[idx])
        # print("agent forward:", agent_fwd[idx])
        # print("target forward:", target_fwd[idx])
        
        # Create new arrows for forward direction - simplified approach
        # Agent arrow starting at (0,0)
        agent_arrow = ax.arrow(xs_a[idx], zs_a[idx], 
                              agent_fwd[idx][0] * 10, agent_fwd[idx][1] * 10,
                              head_width=4, head_length=4, 
                              fc='blue', ec='blue', alpha=0.8, linewidth=3)
        
        # Target arrow starting at target position
        target_arrow = ax.arrow(xs_t[idx], zs_t[idx],
                               target_fwd[idx][0] * 10, target_fwd[idx][1] * 10,
                               head_width=4, head_length=4,
                               fc='red', ec='red', alpha=0.8, linewidth=3)
        
        # Add text annotations showing coordinates  
        agent_text = ax.text(xs_a[idx], zs_a[idx], f'Agent\n({xs_a[idx]:.1f}, {zs_a[idx]:.1f})', 
                           ha='center', va='bottom', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        target_text = ax.text(xs_t[idx] , zs_t[idx] + 10, 
                            f'Target\n({xs_t[idx]:.1f}, {zs_t[idx]:.1f})', 
                            ha='center', va='bottom', fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        # Add agent information below the agent
        move_dir = frames[idx].get('moveDirection', {'x': 0, 'y': 0})
        look_rot = frames[idx].get('lookRotationDelta', {'x': 0, 'y': 0})
        health = frames[idx].get('health', 0)
        is_los = frames[idx].get('islos', False)
        
        info_text = (f"Move: ({move_dir.get('x', 0):.2f}, {move_dir.get('y', 0):.2f})\n"
                    f"Look: ({look_rot.get('x', 0):.2f}, {look_rot.get('y', 0):.2f})\n"
                    f"Health: {health:.1f}\n"
                    f"LOS: {is_los}")
        
        agent_info_text = ax.text(xs_a[idx], zs_a[idx] - 20, info_text, 
                                 ha='center', va='top', fontsize=8,
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        # Close the figure automatically when the last frame has been shown
        if idx == len(frames) - 1:
            # Use a timer to close the figure after a short delay
            import threading
            def close_figure():
                plt.close(fig)
            timer = threading.Timer(2.0, close_figure)  # Close after 2 seconds
            timer.start()
        return agent_scatter, target_scatter

    # Create the animation. interval controls the speed (ms between frames)
    anim = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False, repeat=False)

    # Show the plot and automatically move to next when done
    plt.show(block=False)  # Non-blocking show
    
    # Wait for the animation to complete plus a small buffer
    total_time = len(frames) * 0.05  # 50ms per frame + 3 seconds buffer
    plt.pause(total_time)
    
    # Ensure the figure is closed
    if plt.fignum_exists(fig.number):
        plt.close(fig)


def calculate_agent_activeness(data):
    """Calculate if the agent is actively moving based on moveDirection commands."""
    global total_frames_all_files, total_active_frames_all_files
    
    if not data:
        print("No data to analyze")
        return 0
    
    # Extract moveDirection commands
    move_commands = []
    for frame in data:
        if 'moveDirection' in frame:
            move_dir = frame['moveDirection']
            # Calculate magnitude of move direction vector
            if isinstance(move_dir, dict) and 'x' in move_dir and 'y' in move_dir:
                magnitude = (move_dir['x']**2 + move_dir['y']**2)**0.5
                move_commands.append(magnitude)
            else:
                move_commands.append(0)
        else:
            move_commands.append(0)
    
    if not move_commands:
        print("No moveDirection data found")
        return 0
    
    # Calculate statistics for current file
    non_zero_moves = [m for m in move_commands if m > 0.0001]  # Filter out very small movements
    total_frames = len(move_commands)
    active_frames = len(non_zero_moves)
    
    # Update global counters
    total_frames_all_files += total_frames
    total_active_frames_all_files += active_frames

    # Determine activity level based on movement ratio and magnitude
    movement_ratio = active_frames / total_frames if total_frames > 0 else 0
    
    if movement_ratio > 0.3 :  # At least 10% of frames with movement
        activity_level = 1
        print("Agent is ACTIVELY moving")
    else:
        activity_level = 0
        print("Agent is NOT actively moving")
    
    return activity_level
    

processed_count = 0
for json_file in json_files:
    print("json file = ", json_file)
    file_path = os.path.join(folder_path, json_file)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        movement = calculate_agent_activeness(data)
        if movement == 1:
            visualize_data(data)
        processed_count += 1
        print(f"Successfully simulated {json_file}")
        
        # Print cumulative statistics after each file
        cumulative_ratio = total_active_frames_all_files / total_frames_all_files if total_frames_all_files > 0 else 0
        print(f"CUMULATIVE STATS - Total files processed: {processed_count}")
        print(f"CUMULATIVE STATS - Total frames across all files: {total_frames_all_files}")
        print(f"CUMULATIVE STATS - Total active frames across all files: {total_active_frames_all_files}")
        print(f"CUMULATIVE STATS - Overall activity ratio: {cumulative_ratio:.3f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

print(f"\nTotal JSON files simulated: {processed_count}")
print(f"Total frames across all files: {total_frames_all_files}")
print(f"Total active frames across all files: {total_active_frames_all_files}")
