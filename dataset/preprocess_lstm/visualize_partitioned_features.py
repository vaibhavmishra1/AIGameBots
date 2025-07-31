import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# Define the output path
output_path = "/Users/vaibhavmishra/Desktop/Desktop/btx-game-aicode/clash_squad_partitioned_features_chunked/features"

def load_and_display_npy_files():
    """Load and display basic information about .npy files"""
    if not os.path.exists(output_path):
        print(f"Error: Directory {output_path} does not exist")
        return []
    
    npy_files = list(Path(output_path).glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {output_path}")
        return []
    
    print(f"Found {len(npy_files)} .npy files")
    print("\nFile details:")
    print("-" * 50)
    
    loaded_data = []
    for file_path in npy_files[:100]:
        try:
            data = np.load(str(file_path))
            print(f"\nFile: {file_path.name}")
            print(f"Shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Size in memory: {data.nbytes / (1024*1024):.2f} MB")
            loaded_data.append((file_path.name, data))
            print("-" * 50)
        except Exception as e:
            print(f"Error loading {file_path.name}: {str(e)}")
    
    return loaded_data

def extract_positions_from_features(features):
    """Extract agent and target positions, forward vectors, and additional data from feature vectors
    
    Based on the notebook, the feature structure is:
    - features[0]: agent x position (index 0)
    - features[1]: agent z position (index 1)
    - features[2]: agent rotation y (index 2)
    - features[3]: agent forward x (index 3)
    - features[4]: agent forward z (index 4)
    - features[7]: target x position (index 7)
    - features[8]: target z position (index 8)
    - features[9]: target rotation y (index 9)
    - features[10]: target forward x (index 10)
    - features[11]: target forward z (index 11)
    - features[14]: cross x (index 14)
    - features[15]: cross z (index 15)
    - features[17]: dot product (index 17)
    """
    agent_positions = []
    target_positions = []
    agent_forwards = []
    target_forwards = []
    agent_rotations = []
    target_rotations = []
    cross_products = []
    dot_products = []
    
    for feature_vector in features:
        if len(feature_vector) >= 18:  # Ensure we have enough features
            agent_x = feature_vector[0] * 1000  # Convert back from normalized
            agent_z = feature_vector[1] * 1000  # Convert back from normalized
            agent_rotation = feature_vector[2] * 360  # Convert back from normalized rotation
            agent_forward_x = feature_vector[3]  # Forward vector (already normalized)
            agent_forward_z = feature_vector[4]  # Forward vector (already normalized)
            
            target_x = feature_vector[7] * 1000  # Convert back from normalized
            target_z = feature_vector[8] * 1000  # Convert back from normalized
            target_rotation = feature_vector[9] * 360  # Convert back from normalized rotation
            target_forward_x = feature_vector[10]  # Forward vector (already normalized)
            target_forward_z = feature_vector[11]  # Forward vector (already normalized)
            
            cross_x = feature_vector[14]  # Cross product x component
            cross_z = feature_vector[15]  # Cross product z component
            dot_product = feature_vector[17]  # Dot product
            
            agent_positions.append((agent_x, agent_z))
            target_positions.append((target_x, target_z))
            agent_forwards.append((agent_forward_x, agent_forward_z))
            target_forwards.append((target_forward_x, target_forward_z))
            agent_rotations.append(agent_rotation)
            target_rotations.append(target_rotation)
            cross_products.append((cross_x, cross_z))
            dot_products.append(dot_product)
    
    return (np.array(agent_positions), np.array(target_positions), 
            np.array(agent_forwards), np.array(target_forwards),
            np.array(agent_rotations), np.array(target_rotations),
            np.array(cross_products), np.array(dot_products))

def create_movement_animation(agent_positions, target_positions, agent_forwards, target_forwards, agent_rotations, target_rotations, cross_products, dot_products, filename, chunk_size=20):
    """Create an animated visualization of agent and target movements with forward direction arrows"""
    
    # Limit to chunk_size frames
    num_frames = min(len(agent_positions), chunk_size)
    agent_pos = agent_positions[:num_frames]
    target_pos = target_positions[:num_frames]
    agent_fwd = agent_forwards[:num_frames]
    target_fwd = target_forwards[:num_frames]
    agent_rot = agent_rotations[:num_frames]
    target_rot = target_rotations[:num_frames]
    cross_prod = cross_products[:num_frames]
    dot_prod = dot_products[:num_frames]
    
    if num_frames == 0:
        print(f"No position data available for {filename}")
        return
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate bounds for the plot
    all_x = np.concatenate([agent_pos[:, 0], target_pos[:, 0]])
    all_z = np.concatenate([agent_pos[:, 1], target_pos[:, 1]])
    
    margin = max(np.std(all_x), np.std(all_z)) * 0.5
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    z_min, z_max = np.min(all_z) - margin, np.max(all_z) + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('X Position (units)')
    ax.set_ylabel('Z Position (units)')
    ax.set_title(f'Agent vs Target Movement Animation - {filename}')
    ax.grid(True, alpha=0.3)
    
    # Initialize plot elements
    agent_point, = ax.plot([], [], 'bo', markersize=12, label='Agent')
    target_point, = ax.plot([], [], 'ro', markersize=12, label='Target')
    agent_trail, = ax.plot([], [], 'b-', alpha=0.3, linewidth=2)
    target_trail, = ax.plot([], [], 'r-', alpha=0.3, linewidth=2)
    connection_line, = ax.plot([], [], 'g--', alpha=0.5, linewidth=1)
    
    # Forward direction arrows
    agent_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0), 
                             arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.8))
    target_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0), 
                              arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.8))
    
    # Text elements for displaying information
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    distance_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    direction_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    additional_text = ax.text(0.02, 0.65, '', transform=ax.transAxes, fontsize=9,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    
    def animate(frame):
        # Current positions and forward vectors
        agent_x, agent_z = agent_pos[frame]
        target_x, target_z = target_pos[frame]
        agent_fx, agent_fz = agent_fwd[frame]
        target_fx, target_fz = target_fwd[frame]
        agent_rotation = agent_rot[frame]
        target_rotation = target_rot[frame]
        cross_x, cross_z = cross_prod[frame]
        dot_product = dot_prod[frame]
        
        # Update current position markers
        agent_point.set_data([agent_x], [agent_z])
        target_point.set_data([target_x], [target_z])
        
        # Update trails (show last 10 positions)
        trail_start = max(0, frame - 9)
        agent_trail.set_data(agent_pos[trail_start:frame+1, 0], agent_pos[trail_start:frame+1, 1])
        target_trail.set_data(target_pos[trail_start:frame+1, 0], target_pos[trail_start:frame+1, 1])
        
        # Update connection line
        connection_line.set_data([agent_x, target_x], [agent_z, target_z])
        
        # Calculate arrow scale based on plot size
        x_range = x_max - x_min
        z_range = z_max - z_min
        arrow_scale = min(x_range, z_range) * 0.05  # 5% of the smaller dimension
        
        # Update forward direction arrows
        agent_arrow.set_position((agent_x, agent_z))
        agent_arrow.xy = (agent_x + agent_fx * arrow_scale, agent_z + agent_fz * arrow_scale)
        
        target_arrow.set_position((target_x, target_z))
        target_arrow.xy = (target_x + target_fx * arrow_scale, target_z + target_fz * arrow_scale)
        
        # Calculate distance and angles
        distance = np.sqrt((target_x - agent_x)**2 + (target_z - agent_z)**2)
        agent_angle = np.arctan2(agent_fz, agent_fx) * 180 / np.pi
        target_angle = np.arctan2(target_fz, target_fx) * 180 / np.pi
        
        # Update text information
        frame_text.set_text(f'Frame: {frame + 1}/{num_frames}')
        distance_text.set_text(f'Distance: {distance:.1f} units\nAgent: ({agent_x:.1f}, {agent_z:.1f})\nTarget: ({target_x:.1f}, {target_z:.1f})')
        direction_text.set_text(f'Agent facing: {agent_angle:.1f}° (rot: {agent_rotation:.1f}°)\nTarget facing: {target_angle:.1f}° (rot: {target_rotation:.1f}°)\nForward vectors:\nAgent: ({agent_fx:.2f}, {agent_fz:.2f})\nTarget: ({target_fx:.2f}, {target_fz:.2f})')
        additional_text.set_text(f'Cross Product: ({cross_x:.3f}, {cross_z:.3f})\nDot Product: {dot_product:.3f}\nCross Magnitude: {np.sqrt(cross_x**2 + cross_z**2):.3f}')
        
        return agent_point, target_point, agent_trail, target_trail, connection_line, agent_arrow, target_arrow, frame_text, distance_text, direction_text, additional_text
    
    # Create animation with auto-close functionality
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1000, blit=True, repeat=False)
    
    # Auto-close the figure after animation completes
    def on_animation_complete(event):
        plt.close(fig)
    
    # Set up timer to close the figure after animation duration + small buffer
    animation_duration = num_frames * 1000  # 200ms per frame + 1 second buffer
    timer = fig.canvas.new_timer(interval=animation_duration)
    timer.add_callback(on_animation_complete, None)
    timer.start()
    
    plt.tight_layout()
    plt.show()
    
    return anim

def visualize_all_features(chunk_size=20):
    """Load all .npy files and create animations for each"""
    loaded_data = load_and_display_npy_files()
    
    if not loaded_data:
        print("No data to visualize")
        return
    
    animations = []
    
    for filename, data in loaded_data:
        print(f"\nProcessing {filename}...")
        
        # Extract positions, forward vectors, and additional data from the feature data
        agent_positions, target_positions, agent_forwards, target_forwards, agent_rotations, target_rotations, cross_products, dot_products = extract_positions_from_features(data)
        
        if len(agent_positions) == 0:
            print(f"No valid position data found in {filename}")
            continue
        
        print(f"Found {len(agent_positions)} position frames")
        
        # Create animation
        anim = create_movement_animation(agent_positions, target_positions, agent_forwards, target_forwards, agent_rotations, target_rotations, cross_products, dot_products, filename, chunk_size)
        if anim:
            animations.append(anim)
    
    return animations

def analyze_single_file(filename, chunk_size=20):
    """Analyze and visualize a specific file"""
    file_path = Path(output_path) / filename
    
    if not file_path.exists():
        print(f"File {filename} not found in {output_path}")
        return
    
    try:
        data = np.load(str(file_path))
        print(f"Loaded {filename}: Shape {data.shape}")
        
        agent_positions, target_positions, agent_forwards, target_forwards, agent_rotations, target_rotations, cross_products, dot_products = extract_positions_from_features(data)
        
        if len(agent_positions) == 0:
            print("No valid position data found")
            return
        
        print(f"Found {len(agent_positions)} position frames")
        return create_movement_animation(agent_positions, target_positions, agent_forwards, target_forwards, agent_rotations, target_rotations, cross_products, dot_products, filename, chunk_size)
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    print("=== Feature Visualization Tool ===")
    print("Loading .npy files and creating movement animations...")
    
    # Visualize all features with chunk size of 20
    animations = visualize_all_features(chunk_size=20)
    
    print(f"\nCreated {len(animations) if animations else 0} animations")
    print("Close the animation windows to continue or exit the program.")
