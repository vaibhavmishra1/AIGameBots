# AI Agent 2D Simulation

This simulation creates a 2D web-based environment where you can visualize AI agents moving based on predictions from your trained model.

## Features

- **Real-time Visualization**: Watch 10 agents move on a 2D canvas
- **Team-based Agents**: Agents are divided into 2 teams (team indices 1 and 2, red and blue)
- **API Integration**: Makes individual API calls for each agent with agent-specific temporal history
- **Interactive Controls**: Start, stop, reset, and step through the simulation
- **Agent History Tracking**: Maintains temporal history for accurate predictions
- **Movement Heatmap**: Visualize delta_x and delta_y predictions as a colored heatmap

## Quick Start

### Option 1: Run Everything at Once
```bash
python run_simulation.py
```

This will:
1. Automatically find an available port (starting from 8001)
2. Start the API server on the available port
3. Create a custom web environment file with the correct API URL
4. Open the web environment in your browser
5. Connect them automatically

If port 8001 is busy, it will automatically use the next available port.

### Option 2: Manual Setup
1. **Start the API server:**
   ```bash
   python api_server.py --port 8001
   ```
   (Use `--port PORT_NUMBER` to specify a custom port if 8001 is busy)

2. **Open the web environment:**
   - Open `web_environment.html` in your browser
   - Or serve it with a local web server

## How It Works

### Data Flow
1. **Agent Initialization**: 10 agents are placed within [-100, 100] range of shrinking area center (-365.63, -139.57)
2. **Temporal History**: Each agent maintains a history of positions (up to 16 timesteps)
3. **Individual API Calls**: Each agent makes its own API call with itself in the 0th row and others in subsequent rows
4. **Model Prediction**: Your model returns 5 pairs of delta movements specific to each agent's context
5. **Position Updates**: Each agent moves according to its own unique predictions
6. **Heatmap Generation**: Optional movement heatmap shows predicted delta patterns
7. **Loop**: Process repeats continuously for all agents

### Features Used
The simulation uses only the 6 core features from your model:
- `team_index`: Agent's team (0 or 1)
- `rel_pos_x`: Relative X position to zone center
- `rel_pos_z`: Relative Z position to zone center
- `shr_key`: Normalized shrinking area key
- `deltax`: Change in X position
- `deltay`: Change in Z position

### Random Features
These features are randomized since they're not used in predictions:
- `rotation`
- `move_dir_x`, `move_dir_y`
- `look_rot_delta_x`, `look_rot_delta_y`
- `attack`

## Controls

- **Start Simulation**: Begin continuous agent movement
- **Stop Simulation**: Pause the simulation
- **Reset Agents**: Generate new random positions for all agents
- **Step Forward**: Advance one timestep manually
- **Toggle Heatmap**: Show/hide movement heatmap overlay

## Technical Details

### API Endpoint
- **URL**: `http://localhost:8001/predict/unity`
- **Method**: POST
- **Input**: JSON with temporal history
- **Output**: Array of 10 floats (5 dx,dy pairs)

### Data Format
```json
{
  "temporal_history": [
    {
      "agents": [
        {
          "agent_id": 0,
          "game_time": 1.0,
          "team_index": 0,
          "pos_x": 100.5,
          "pos_z": 200.3,
          "rotation": 1.57,
          "move_dir_x": 0.1,
          "move_dir_y": -0.2,
          "look_rot_delta_x": 0.0,
          "look_rot_delta_y": 0.0,
          "attack": 0
        },
        // ... 9 more agents
      ]
    },
    // ... up to 16 timesteps
  ]
}
```

## Troubleshooting

### API Connection Issues
- Ensure the API server is running on port 8001
- Check browser console for network errors
- Verify CORS settings if accessing from different domain

### Model Loading Issues
- Ensure the model file exists at the specified path
- Check that all required dependencies are installed
- Verify PyTorch MPS/CPU compatibility

### Movement Heatmap
The heatmap visualization shows the predicted movement patterns across the environment:

- **Color Coding**: HSL color wheel represents movement direction and magnitude
  - Hue (0-360°): Movement direction (red=right, blue=left, green=up, etc.)
  - Saturation: Movement intensity (darker = stronger movement)
  - Lightness: Scaled from 40-70% based on relative strength

- **Grid Resolution**: 20x20 grid cells covering the entire canvas
- **Aggregation**: Delta values are averaged within each grid cell
- **Direction Arrows**: White arrows appear on cells with strong movement (>30% intensity)
- **Transparency**: 60% opacity overlay doesn't obscure agent visualization

### Performance Issues
- The simulation runs at ~60 FPS target but makes 10 individual API calls per timestep
- API call latency may affect performance - consider server response times
- Reduce canvas size or agent count if experiencing lag
- Check browser developer tools for network and performance bottlenecks

## Dependencies

### Python Requirements
- fastapi
- uvicorn
- torch
- numpy
- h5py
- pydantic

### Browser Requirements
- Modern browser with Canvas API support
- JavaScript enabled
- CORS support for local development

## Next Steps

- **Enhanced Visualization**: Add trails, velocity vectors, or heatmaps
- **Multi-Agent Coordination**: Implement team-based strategies
- **Performance Metrics**: Add FPS counter, prediction latency tracking
- **Export Functionality**: Save simulation data for analysis
- **Parameter Tuning**: Add controls for simulation parameters
