#!/usr/bin/env python3
"""
Script to run the AI Agent 2D Simulation
Starts the API server and opens the web environment
"""

import subprocess
import sys
import os
import time
import webbrowser
import socket
from pathlib import Path

def find_available_port(start_port=8001, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def main():
    print("🚀 Starting AI Agent 2D Simulation...")
    print("=" * 50)

    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()

    # Find an available port
    try:
        port = find_available_port()
        print(f"📡 Using port {port} for API server...")
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        return

    # Start the API server in the background with the specific port
    print("📡 Starting API server...")
    api_process = subprocess.Popen([
        "python3",
        str(script_dir / "api_server.py"),
        "--port", str(port)
    ], cwd=str(script_dir))

    # Wait a moment for the server to start
    print("⏳ Waiting for API server to initialize...")
    time.sleep(3)

    # Create a temporary HTML file with the correct API URL
    web_path = script_dir / "web_environment.html"
    temp_web_path = script_dir / "temp_web_environment.html"

    try:
        # Read the original HTML file
        with open(web_path, 'r') as f:
            html_content = f.read()

        # Replace the API URL
        html_content = html_content.replace(
            'this.apiUrl = \'http://localhost:8001/predict/unity\';',
            f'this.apiUrl = \'http://localhost:{port}/predict/unity\';'
        )

        # Write the temporary file
        with open(temp_web_path, 'w') as f:
            f.write(html_content)

        # Open the temporary web environment in the default browser
        print(f"🌐 Opening web environment with API on port {port}")
        webbrowser.open(f"file://{temp_web_path}")

    except Exception as e:
        print(f"⚠️ Warning: Could not create custom web file: {e}")
        print("🌐 Opening default web environment (you may need to manually update the API URL)")
        webbrowser.open(f"file://{web_path}")

    print("\n✅ Simulation started!")
    print("📋 Instructions:")
    print(f"   - The API server is running on http://localhost:{port}")
    print("   - The web environment should open in your browser")
    print("   - Click 'Start Simulation' to begin the agent movement")
    print("   - Use 'Reset Agents' to restart with new random positions")
    print("   - Press Ctrl+C to stop the simulation")
    print("\n" + "=" * 50)

    try:
        # Keep the script running
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down simulation...")
        api_process.terminate()
        api_process.wait()
        # Clean up temporary file
        try:
            if temp_web_path.exists():
                temp_web_path.unlink()
        except:
            pass
        print("✅ Simulation stopped.")

if __name__ == "__main__":
    main()
