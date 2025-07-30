import json
import re
from collections import defaultdict
import numpy as np
import os

# Add visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
sns.set_palette("husl")

def read_and_split_log_file(file_path):
    """
    Read a log file and split it into chunks wherever '}{' is found.
    The '}' goes to the previous chunk and '{' goes to the next chunk.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content = content.lstrip('\ufeff')
    
    # Split the content by '}{' pattern
    # This will split at the boundary between JSON objects
    chunks = re.split(r'}{', content)
    
    # Process chunks to properly format them as complete JSON objects
    json_objects = []
    
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # Add opening brace to all chunks except the first
        if i > 0:
            chunk = '{' + chunk
            
        # Add closing brace to all chunks except the last
        if i < len(chunks) - 1:
            chunk = chunk + '}'
        
        # Try to parse as JSON
        try:
            json_obj = json.loads(chunk)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Failed to parse chunk {i}: {e}")
            print(f"Chunk content: {chunk[:100]}...")  # Show first 100 chars
        

    return json_objects

def extract_number_from_filename(filename):
    """Extract the number from filenames like '500_recorder.log'"""
    import re
    match = re.search(r'(\d+)_recorder\.log', filename)
    return int(match.group(1)) if match else 0



def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def create_beautiful_heatmap(x_coords, z_coords, title="Agent Movement Heatmap", save_path="agent_movement_heatmap.png"):
    """
    Create a beautiful and detailed 2D heatmap of agent movement patterns
    """
    print(f"Creating heatmap with {len(x_coords)} data points...")
    
    # Convert to numpy arrays
    x_coords = np.array(x_coords)
    z_coords = np.array(z_coords)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Custom colormap for beautiful visualization
    colors = ['#000428', '#004e92', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # 1. Main Heatmap with Gaussian smoothing
    plt.subplot(2, 3, 1)
    
    # Create 2D histogram with fine-grained bins for more detail
    heatmap, xedges, yedges = np.histogram2d(x_coords, z_coords, bins=200)
    
    # Apply Gaussian smoothing for better visualization
    heatmap_smooth = gaussian_filter(heatmap, sigma=1.5)
    
    # Plot heatmap
    im1 = plt.imshow(heatmap_smooth.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     cmap=custom_cmap, aspect='auto')
    plt.colorbar(im1, label='Agent Density')
    plt.title('Agent Movement Heatmap (Smoothed)', fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Z Coordinate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 2. Scatter plot with density
    plt.subplot(2, 3, 2)
    plt.scatter(x_coords, z_coords, alpha=0.6, s=1, c='#FF6B6B', edgecolors='none')
    plt.title('Agent Positions Scatter Plot', fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Z Coordinate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 3. Hexbin plot for density visualization
    plt.subplot(2, 3, 3)
    hexbin = plt.hexbin(x_coords, z_coords, gridsize=50, cmap='viridis', alpha=0.8)
    plt.colorbar(hexbin, label='Agent Count')
    plt.title('Agent Density (Hexbin)', fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Z Coordinate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 4. Contour plot
    plt.subplot(2, 3, 4)
    # Create meshgrid for contour plot with fine-grained bins
    x_bins = np.linspace(x_coords.min(), x_coords.max(), 200)
    z_bins = np.linspace(z_coords.min(), z_coords.max(), 200)
    X, Z = np.meshgrid(x_bins, z_bins)
    
    # Calculate density for contour
    positions = np.vstack([x_coords, z_coords])
    try:
        from scipy.stats import gaussian_kde
        kernel = gaussian_kde(positions)
        density = kernel(positions)
        
        # Create contour plot
        contour = plt.tricontourf(x_coords, z_coords, density, levels=20, cmap='plasma')
    except ImportError:
        # Fallback to simple histogram if scipy.stats is not available
        heatmap_contour, _, _ = np.histogram2d(x_coords, z_coords, bins=50)
        contour = plt.imshow(heatmap_contour.T, origin='lower', cmap='plasma', aspect='auto')
    plt.colorbar(contour, label='Density')
    plt.title('Agent Movement Density (Contour)', fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Z Coordinate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 5. 2D Histogram with different colormap
    plt.subplot(2, 3, 5)
    heatmap2, xedges2, yedges2 = np.histogram2d(x_coords, z_coords, bins=200)
    im2 = plt.imshow(heatmap2.T, origin='lower', extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], 
                     cmap='inferno', aspect='auto')
    plt.colorbar(im2, label='Agent Count')
    plt.title('Agent Movement (Inferno Colormap)', fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Z Coordinate', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 6. Statistical summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate statistics
    stats_text = f"""
    AGENT MOVEMENT STATISTICS
    
    Total Data Points: {len(x_coords):,}
    
    X Coordinate Range:
    • Min: {x_coords.min():.2f}
    • Max: {x_coords.max():.2f}
    • Mean: {x_coords.mean():.2f}
    • Std: {x_coords.std():.2f}
    
    Z Coordinate Range:
    • Min: {z_coords.min():.2f}
    • Max: {z_coords.max():.2f}
    • Mean: {z_coords.mean():.2f}
    • Std: {z_coords.std():.2f}
    
    Coverage Area:
    • Width: {x_coords.max() - x_coords.min():.2f}
    • Height: {z_coords.max() - z_coords.min():.2f}
    
    Hotspots Detected: {len(np.where(heatmap > heatmap.max() * 0.8)[0])}
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Beautiful heatmap saved as: {save_path}")
    plt.show()
    
    return fig

def create_advanced_heatmap(x_coords, z_coords, save_path="advanced_agent_heatmap.png"):
    """
    Create an advanced heatmap with additional analysis
    """
    print("Creating advanced heatmap analysis...")
    
    # Convert to numpy arrays
    x_coords = np.array(x_coords)
    z_coords = np.array(z_coords)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Agent Movement Analysis', fontsize=20, fontweight='bold')
    
    # 1. Main heatmap with annotations
    ax1 = axes[0, 0]
    heatmap, xedges, yedges = np.histogram2d(x_coords, z_coords, bins=200)
    heatmap_smooth = gaussian_filter(heatmap, sigma=1.5)
    
    im1 = ax1.imshow(heatmap_smooth.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     cmap='magma', aspect='auto')
    plt.colorbar(im1, ax=ax1, label='Agent Density')
    ax1.set_title('Movement Heatmap with Hotspots', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Z Coordinate')
    ax1.grid(True, alpha=0.3)
    
    # Find and mark hotspots
    threshold = heatmap_smooth.max() * 0.7
    hotspot_coords = np.where(heatmap_smooth > threshold)
    for i, j in zip(hotspot_coords[0], hotspot_coords[1]):
        x_pos = xedges[i]
        z_pos = yedges[j]
        ax1.plot(x_pos, z_pos, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    # 2. Movement trajectory analysis
    ax2 = axes[0, 1]
    # Sample points for trajectory visualization
    sample_size = min(1000, len(x_coords))
    indices = np.random.choice(len(x_coords), sample_size, replace=False)
    
    ax2.scatter(x_coords[indices], z_coords[indices], c=range(sample_size), 
                cmap='viridis', alpha=0.7, s=20)
    ax2.set_title('Movement Trajectory (Sampled)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Z Coordinate')
    ax2.grid(True, alpha=0.3)
    
    # 3. Density distribution
    ax3 = axes[1, 0]
    # Create 2D density plot
    positions = np.vstack([x_coords, z_coords])
    try:
        from scipy.stats import gaussian_kde
        kernel = gaussian_kde(positions)
        
        # Create grid for density calculation with fine-grained bins
        x_grid = np.linspace(x_coords.min(), x_coords.max(), 200)
        z_grid = np.linspace(z_coords.min(), z_coords.max(), 200)
        X_grid, Z_grid = np.meshgrid(x_grid, z_grid)
        positions_grid = np.vstack([X_grid.ravel(), Z_grid.ravel()])
        density = kernel(positions_grid).reshape(X_grid.shape)
        
        contour = ax3.contourf(X_grid, Z_grid, density, levels=20, cmap='plasma')
        plt.colorbar(contour, ax=ax3, label='Density')
        ax3.set_title('Probability Density Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Z Coordinate')
        ax3.grid(True, alpha=0.3)
    except ImportError:
        ax3.text(0.5, 0.5, 'scipy.stats required for density plot', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Probability Density Distribution', fontsize=14, fontweight='bold')
    
    # 4. Statistical summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate advanced statistics
    coverage_area = (x_coords.max() - x_coords.min()) * (z_coords.max() - z_coords.min())
    density_score = len(x_coords) / coverage_area
    
    stats_text = f"""
    ADVANCED ANALYSIS
    
    Data Points: {len(x_coords):,}
    Coverage Area: {coverage_area:.2f}
    Density Score: {density_score:.2f}
    
    X Statistics:
    • Range: {x_coords.max() - x_coords.min():.2f}
    • Median: {np.median(x_coords):.2f}
    • 25th Percentile: {np.percentile(x_coords, 25):.2f}
    • 75th Percentile: {np.percentile(x_coords, 75):.2f}
    
    Z Statistics:
    • Range: {z_coords.max() - z_coords.min():.2f}
    • Median: {np.median(z_coords):.2f}
    • 25th Percentile: {np.percentile(z_coords, 25):.2f}
    • 75th Percentile: {np.percentile(z_coords, 75):.2f}
    
    Hotspots Found: {len(hotspot_coords[0])}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Advanced heatmap saved as: {save_path}")
    plt.show()
    
    return fig

def create_interactive_plotly_heatmap(x_coords, z_coords, save_path="interactive_agent_heatmap.html"):
    """
    Create an interactive heatmap using Plotly with zoom, pan, and hover capabilities
    """
    print("Creating interactive Plotly heatmap...")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.figure_factory as ff
    except ImportError:
        print("❌ Plotly not installed. Installing it now...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'plotly'])
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.figure_factory as ff
    
    # Convert to numpy arrays
    x_coords = np.array(x_coords)
    z_coords = np.array(z_coords)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Agent Density Heatmap', 'Scatter Plot with Density', 'Contour Plot',
                       'Hexbin Visualization', '3D Surface Plot', 'Statistics'),
        specs=[[{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'contour'}],
               [{'type': 'scatter'}, {'type': 'surface'}, {'type': 'table'}]],
        row_heights=[0.5, 0.5],
        column_widths=[0.33, 0.33, 0.34],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    # 1. Main Heatmap with double ultra-high resolution
    heatmap, xedges, yedges = np.histogram2d(x_coords, z_coords, bins=600)
    heatmap_smooth = gaussian_filter(heatmap, sigma=1.5)
    
    heatmap_trace = go.Heatmap(
        z=heatmap_smooth.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale='Viridis',
        name='Density',
        hovertemplate='X: %{x:.2f}<br>Z: %{y:.2f}<br>Density: %{z:.2f}<extra></extra>',
        colorbar=dict(title="Density", x=0.31, y=0.75, len=0.4)
    )
    fig.add_trace(heatmap_trace, row=1, col=1)
    
    # 2. Scatter plot with density coloring
    # Sample for performance with double points for maximum resolution
    sample_size = min(20000, len(x_coords))
    if len(x_coords) > sample_size:
        indices = np.random.choice(len(x_coords), sample_size, replace=False)
        x_sample = x_coords[indices]
        z_sample = z_coords[indices]
    else:
        x_sample = x_coords
        z_sample = z_coords
    
    scatter_trace = go.Scattergl(
        x=x_sample,
        y=z_sample,
        mode='markers',
        marker=dict(
            size=3,
            color=z_sample,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Z Coord", x=0.65, y=0.75, len=0.4)
        ),
        name='Agent Positions',
        hovertemplate='X: %{x:.2f}<br>Z: %{y:.2f}<extra></extra>'
    )
    fig.add_trace(scatter_trace, row=1, col=2)
    
    # 3. Contour plot
    # Create grid for contour with double ultra-high resolution
    x_grid = np.linspace(x_coords.min(), x_coords.max(), 300)
    z_grid = np.linspace(z_coords.min(), z_coords.max(), 300)
    
    # Calculate density using 2D histogram for contour
    H, xedges_c, yedges_c = np.histogram2d(x_coords, z_coords, bins=[x_grid, z_grid])
    H_smooth = gaussian_filter(H, sigma=1)
    
    contour_trace = go.Contour(
        z=H_smooth.T,
        x=xedges_c[:-1],
        y=yedges_c[:-1],
        colorscale='Hot',
        name='Density Contour',
        hovertemplate='X: %{x:.2f}<br>Z: %{y:.2f}<br>Density: %{z:.2f}<extra></extra>',
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=12, color='white'),
        ),
        colorbar=dict(title="Density", x=0.98, y=0.75, len=0.4)
    )
    fig.add_trace(contour_trace, row=1, col=3)
    
    # 4. Hexbin-style visualization using histogram2d
    hexbin_trace = go.Histogram2d(
        x=x_coords,
        y=z_coords,
        nbinsx=240,
        nbinsy=240,
        colorscale='Viridis',
        name='Hexbin',
        hovertemplate='X: %{x:.2f}<br>Z: %{y:.2f}<br>Count: %{z}<extra></extra>',
        colorbar=dict(title="Count", x=0.31, y=0.25, len=0.4)
    )
    fig.add_trace(hexbin_trace, row=2, col=1)
    
    # 5. 3D Surface plot
    # Create double ultra-high resolution grid for 3D visualization
    x_3d = np.linspace(x_coords.min(), x_coords.max(), 240)
    z_3d = np.linspace(z_coords.min(), z_coords.max(), 240)
    X_3d, Z_3d = np.meshgrid(x_3d, z_3d)
    
    # Calculate density for surface
    H_3d, _, _ = np.histogram2d(x_coords, z_coords, bins=[x_3d, z_3d])
    H_3d_smooth = gaussian_filter(H_3d, sigma=1)
    
    surface_trace = go.Surface(
        z=H_3d_smooth.T,
        x=x_3d,
        y=z_3d,
        colorscale='Viridis',
        name='3D Density',
        hovertemplate='X: %{x:.2f}<br>Z: %{y:.2f}<br>Height: %{z:.2f}<extra></extra>',
        colorbar=dict(title="Density", x=0.65, y=0.25, len=0.4)
    )
    fig.add_trace(surface_trace, row=2, col=2)
    
    # 6. Statistics table
    stats_data = [
        ['Metric', 'X Coordinate', 'Z Coordinate'],
        ['Count', f'{len(x_coords):,}', f'{len(z_coords):,}'],
        ['Min', f'{x_coords.min():.2f}', f'{z_coords.min():.2f}'],
        ['Max', f'{x_coords.max():.2f}', f'{z_coords.max():.2f}'],
        ['Mean', f'{x_coords.mean():.2f}', f'{z_coords.mean():.2f}'],
        ['Std Dev', f'{x_coords.std():.2f}', f'{z_coords.std():.2f}'],
        ['Median', f'{np.median(x_coords):.2f}', f'{np.median(z_coords):.2f}'],
        ['Range', f'{x_coords.max() - x_coords.min():.2f}', f'{z_coords.max() - z_coords.min():.2f}']
    ]
    
    table_trace = go.Table(
        header=dict(
            values=stats_data[0],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=list(zip(*stats_data[1:])),
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )
    fig.add_trace(table_trace, row=2, col=3)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Clash Squad Agent Movement Analysis',
            'font': {'size': 24, 'color': 'darkblue'},
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False,
        height=2400,
        width=4000,
        template='plotly_white',
        hovermode='closest'
    )
    
    # Update axes
    for i in range(1, 3):
        for j in range(1, 4):
            if not (i == 2 and j == 3):  # Skip table subplot
                fig.update_xaxes(title_text="X Coordinate", row=i, col=j)
                fig.update_yaxes(title_text="Z Coordinate", row=i, col=j)
    
    # Save interactive HTML
    fig.write_html(save_path, include_plotlyjs='cdn')
    print(f"✅ Interactive heatmap saved as: {save_path}")
    
    # Also create a standalone density heatmap with maximum interactivity
    fig_standalone = go.Figure()
    
    # Add main heatmap
    fig_standalone.add_trace(go.Heatmap(
        z=heatmap_smooth.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale=[
            [0, '#000428'],
            [0.2, '#004e92'],
            [0.4, '#2E86AB'],
            [0.6, '#A23B72'],
            [0.8, '#F18F01'],
            [1, '#C73E1D']
        ],
        hovertemplate='<b>Position</b><br>X: %{x:.2f}<br>Z: %{y:.2f}<br>Density: %{z:.2f}<extra></extra>',
        colorbar=dict(
            title="Agent Density",
            title_side="right",
            tickmode="linear",
            tick0=0,
            dtick=heatmap_smooth.max()/5
        )
    ))
    
    # Add scatter overlay for actual points (sampled)
    fig_standalone.add_trace(go.Scattergl(
        x=x_sample,
        y=z_sample,
        mode='markers',
        marker=dict(
            size=2,
            color='rgba(255, 255, 255, 0.3)',
            line=dict(width=0)
        ),
        name='Agent Positions',
        hovertemplate='Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
    ))
    
    # Update layout for standalone
    fig_standalone.update_layout(
        title={
            'text': 'Clash Squad Agent Movement Heatmap - Interactive (Zoom & Pan Enabled)',
            'font': {'size': 20, 'color': 'darkblue'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="X Coordinate",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(128,128,128,0.5)'
        ),
        yaxis=dict(
            title="Z Coordinate",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(128,128,128,0.5)',
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='white',
        height=2400,
        width=3200,
        hovermode='closest',
        dragmode='zoom'  # Default to zoom mode
    )
    
    # Add annotations
    fig_standalone.add_annotation(
        text=f"Total Points: {len(x_coords):,}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Save standalone version
    standalone_path = save_path.replace('.html', '_standalone.html')
    fig_standalone.write_html(standalone_path, include_plotlyjs='cdn')
    print(f"✅ Standalone interactive heatmap saved as: {standalone_path}")
    
    # Show the figure in browser
    fig_standalone.show()
    
    return fig, fig_standalone

# Define the folder path
import os

folder_path = "/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad/"
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
print("Subfolders:", subfolders)

# List to store complete paths of .log files
from tqdm import tqdm
agent_position_x_list = []
agent_position_z_list = []

for game_folder in tqdm(subfolders, desc="Processing game folders"):
    log_files = []
    # Walk through the directory and find all .log files
    # Combine folder_path and game_folder to get the full path to the game folder
    full_game_folder_path = os.path.join(folder_path, os.path.basename(game_folder))
    for root, dirs, files in os.walk(full_game_folder_path):
        for file in files:
            if file.endswith('.log'):
                # Get the complete path of the file
                complete_path = os.path.join(root, file)
                log_files.append(complete_path)

    # Sort log files by the number in their filename
    log_files.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))

    game_json_objects = []
    for log_file in log_files:
        try:
            json_objects = read_and_split_log_file(log_file)
            game_json_objects.extend(json_objects)
            
        except Exception as e:
            pass

    # Filter out JSON objects with round_number = 0
    game_json_objects = [obj for obj in game_json_objects if obj.get('round_number', 0) != 0]
    
    
    for obj in game_json_objects:
        agent_data = obj['agents_data']
        for agent in agent_data:
            agent_position = agent['agentPosition']
            agent_position_x = agent_position['x']
            agent_position_z = agent_position['z']
            agent_position_x_list.append(agent_position_x)
            agent_position_z_list.append(agent_position_z)
    
    import gc
    gc.collect()

# Create beautiful heatmaps after data collection
print(f"\n{'='*60}")
print("CREATING BEAUTIFUL AGENT MOVEMENT HEATMAPS")
print(f"{'='*60}")

# Check if we have data
if len(agent_position_x_list) > 0 and len(agent_position_z_list) > 0:
    print(f"Total agent positions collected: {len(agent_position_x_list)}")
    
    # # Create the main beautiful heatmap
    # create_beautiful_heatmap(
    #     agent_position_x_list, 
    #     agent_position_z_list,
    #     title="Clash Squad Agent Movement Analysis",
    #     save_path="clash_squad_agent_movement_heatmap.png"
    # )
    
    # # Create advanced analysis heatmap
    # create_advanced_heatmap(
    #     agent_position_x_list,
    #     agent_position_z_list,
    #     save_path="clash_squad_advanced_analysis.png"
    # )
    
    # Create interactive Plotly heatmap
    create_interactive_plotly_heatmap(
        agent_position_x_list,
        agent_position_z_list,
        save_path="clash_squad_interactive_heatmap.html"
    )
    
    print("\n🎉 Beautiful heatmaps created successfully!")
    print("📊 Files saved:")
    print("   • clash_squad_agent_movement_heatmap.png")
    print("   • clash_squad_advanced_analysis.png")
    print("   • clash_squad_interactive_heatmap.html")
    print("   • clash_squad_interactive_heatmap_standalone.html")
    
else:
    print("❌ No agent position data found. Please check your data source.")

    


