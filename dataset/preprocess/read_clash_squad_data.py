import json
import re
from collections import defaultdict
import numpy as np
import os

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


def prepare_dataset(data):
    """
    prepare the dataset from data which is a per game data, 
    get the features as a dictionary with (agent, [ agent_data ] ) where agent is not a bot 

    """
    # Group samples by game_id
    
    game_agents = defaultdict(list)

    for sample in data:
        game_id = sample["game_id"]
        for agent in sample["agents_data"]:
            if agent["isBot"] == False:
                key = f"{game_id}_{agent['agent_id']}"
                game_agents[key].append(agent)
    
    return game_agents


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj



# Define the folder path
import os

folder_path = "/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad/"
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
print("Subfolders:", subfolders)

# List to store complete paths of .log files
from tqdm import tqdm

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

    dataset = prepare_dataset(game_json_objects)

    for key in dataset.keys():
        break

    for key in dataset.keys():
        agent_timeline = dataset[key]
        filename = f"/Users/vaibhavmishra/Desktop/btx-game-aicode/clash_squad_agent_partition/{key}.json"
        with open(filename, "w") as f:
            json.dump(agent_timeline, f, default=convert_numpy)
    
    del game_json_objects
    del dataset
    import gc
    gc.collect()
    
    


