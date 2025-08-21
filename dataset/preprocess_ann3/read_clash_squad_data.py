import json
import re
from collections import defaultdict
import numpy as np
import os
import h5py
# List to store complete paths of .log files
from tqdm import tqdm
import hashlib

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



def get_game_json_objects(folder_path, game_folder):
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
    # print("log files = ", log_files)
    game_json_objects = []
    for log_file in log_files:
        
        json_objects = read_and_split_log_file(log_file)
        game_json_objects.extend(json_objects)
        
        # finally:
        #     try:
        #         os.remove(log_file)
        #         # print(f"Removed {log_file}")
        #     except Exception:
        #         pass
        
    # Filter out JSON objects with round_number = 0
    
    game_json_objects = [obj for obj in game_json_objects if obj.get('round_number', 0) != 0]
    return game_json_objects


def hashit(weapon):
    # Deterministic hash for weapon string using MD5 (first 8 hex chars -> 32-bit int)
    if weapon is None:
        return 0
    if not isinstance(weapon, str):
        weapon = str(weapon)
    digest = hashlib.md5(weapon.encode('utf-8')).hexdigest()
    return int(digest[:8], 16)

def store_features(f, game_json_objects, game_id):
    # print("game_id = ",game_id)
    game_group = f.create_group(f"{game_id}")
    chunk_size =  21
    feature_length = 43
    number_of_agents = 10
    for i in range(len(game_json_objects) - chunk_size):
        game_chunk = game_json_objects[i: i + chunk_size]
        features = np.zeros((chunk_size, number_of_agents, feature_length))

        if len(game_chunk) < chunk_size:
            continue

        # Helper to build the feature vector for an agent dict
        def build_feature_vector(agent_dict):
            game_time = agent_dict['game_time'] #0
            agent_id = int(agent_dict['agent_id']) #1
            target_id = int(agent_dict['target_id']) #2 
            team_index = int(agent_dict['team_index']) #3
            pos_x = agent_dict['agentPosition']['x'] #4
            pos_y = agent_dict['agentPosition']['y'] #5
            pos_z =  agent_dict['agentPosition']['z'] #6 
            rot = agent_dict['agentRotation']['y'] #7 
            forward_x = agent_dict['agentForward']['x'] #8 
            forward_y =  agent_dict['agentForward']['y'] #9
            forward_z = agent_dict['agentForward']['z'] #10
            target_pos_x = agent_dict['targetPosition']['x'] #11
            target_pos_y = agent_dict['targetPosition']['y'] #12
            target_pos_z = agent_dict['targetPosition']['z'] #13
            target_rot = agent_dict['targetRotation']['y'] #14
            target_forward_x = agent_dict['targetForward']['x'] #15
            target_forward_y =  agent_dict['targetForward']['y'] #16
            target_forward_z = agent_dict['targetForward']['z'] #17
            distance = agent_dict['distance'] #18
            dir_to_target_x = agent_dict['directionToTarget']['x'] #19
            dir_to_target_y = agent_dict['directionToTarget']['y'] #20
            dir_to_target_z = agent_dict['directionToTarget']['z'] #21
            dot_product = agent_dict['dotProduct'] #22
            weapon = hashit(agent_dict['weapon']) #23
            move_direction_x = agent_dict['moveDirection']['x'] #24
            move_direction_y = agent_dict['moveDirection']['y'] #25
            lookRotationDelta_x = agent_dict['lookRotationDelta']['x'] #26
            lookRotationDelta_y = agent_dict['lookRotationDelta']['y'] #27
            attack = 0 if agent_dict['Attack'] == False else 1 #28
            reload = 0 if agent_dict['Reload'] == False else 1 #29
            thrust = 0 if agent_dict['thrust'] == False else 1 #30
            crouch = 0 if agent_dict['crouch'] == False else 1 #31
            sprint = 0 if agent_dict['sprint'] == False else 1 #32
            Grapple = 0 if agent_dict['Grapple'] == False else 1 #33
            slide = 0 if agent_dict['slide'] == False else 1 #34 
            revive = 0 if agent_dict['revive'] == False else 1 #35
            addtobag = 0 if agent_dict['addtobag'] == False else 1 #36
            cross_x = agent_dict['cross']['x'] #37
            cross_y = agent_dict['cross']['y'] #38
            cross_z =  agent_dict['cross']['z'] #39
            health = agent_dict['health'] #40
            isbot = 0 if agent_dict['isBot'] == False else 1 #41
            istargetbot = 0 if agent_dict['isTargetBot'] == False else 1 #42

            feature_vec = [
                game_time,           # 0
                agent_id,            # 1
                target_id,           # 2
                team_index,          # 3
                pos_x,               # 4
                pos_y,               # 5
                pos_z,               # 6
                rot,                 # 7
                forward_x,           # 8
                forward_y,           # 9
                forward_z,           # 10
                target_pos_x,        # 11
                target_pos_y,        # 12
                target_pos_z,        # 13
                target_rot,          # 14
                target_forward_x,    # 15
                target_forward_y,    # 16
                target_forward_z,    # 17
                distance,            # 18
                dir_to_target_x,     # 19
                dir_to_target_y,     # 20
                dir_to_target_z,     # 21
                dot_product,         # 22
                weapon,              # 23
                cross_x,             # 24
                cross_y,             # 25
                cross_z,             # 26
                health,              # 27
                isbot,               # 28
                istargetbot,         # 29
                move_direction_x,    # 30
                move_direction_y,    # 31
                lookRotationDelta_x, # 32
                lookRotationDelta_y, # 33
                attack,              # 34
                reload,              # 35
                thrust,              # 36
                crouch,              # 37
                sprint,              # 38
                Grapple,             # 39
                slide,               # 40
                revive,              # 41
                addtobag             # 42
            ]
            return np.array(feature_vec)

        agent_id_to_index = None
        for c in range(chunk_size):
            agents_data = game_chunk[c]["agents_data"]
            if c == 0:
                # Establish fixed mapping from agent_id -> index based on first timestep
                agent_id_to_index = {}
                j = 0
                for agent in agents_data:
                    if j >= number_of_agents:
                        break
                    idx = j
                    agent_id = int(agent['agent_id'])
                    agent_id_to_index[agent_id] = idx
                    features[c][idx] = build_feature_vector(agent)
                    j += 1
            else:
                # For subsequent timesteps, place agents according to first timestep mapping
                for agent in agents_data:
                    agent_id = int(agent['agent_id'])
                    if agent_id not in agent_id_to_index:
                        # Reject new agents appearing mid-chunk
                        continue
                    idx = agent_id_to_index[agent_id]
                    features[c][idx] = build_feature_vector(agent)

        feature_variations = transform_features(features)
        
        
        # Save all feature variations to HDF5 dataset
        for variation_idx, feature_variation in enumerate(feature_variations):
            
            chunk_dataset_name = f"chunk_{i}_variation_{variation_idx}"
            
            game_group.create_dataset(chunk_dataset_name, data=feature_variation, compression="gzip", compression_opts=9)
        


def transform_features(features):
    """
    Transform features array and identify agents present in all time steps of the chunk.
    Then calculate average move_direction for those agents and swap with 0th agent if criteria met.
    
    Args:
        features: numpy array of shape (chunk_size, number_of_agents, feature_length)
                 where index 1 contains agent_id information
    
    Returns:
        list: List of modified feature arrays (original + swapped versions)
    """
    chunk_size, number_of_agents, feature_length = features.shape
    
    # Extract agent IDs from all time steps
    # Agent ID is at index 1 in the feature vector
    agent_ids_per_timestep = []
    
    for t in range(chunk_size):
        timestep_agent_ids = set()
        for a in range(number_of_agents):
            agent_id = int(features[t, a, 1])  # Index 1 contains agent_id
            if agent_id != 0:  # Assuming 0 means no agent
                timestep_agent_ids.add(agent_id)
        agent_ids_per_timestep.append(timestep_agent_ids)
    
    # Find agents present in all time steps
    agents_in_all_timesteps = set.intersection(*agent_ids_per_timestep)
    
    # print(f"Agents present in all {chunk_size} time steps: {agents_in_all_timesteps}")
    
    # List to store all feature variations (original + swapped versions)
    feature_variations = []
    
    # For each agent present in all time steps, check if it meets the criteria
    for agent_id in agents_in_all_timesteps:
        # Find the agent's position in the features array
        agent_position = None
        for a in range(number_of_agents):
            if int(features[0, a, 1]) == agent_id:  # Check first timestep
                agent_position = a
                break
        
        if agent_position is None:
            continue
            
        # Calculate average move_direction_x and move_direction_y for this agent
        total_move_x = 0
        total_move_y = 0
        valid_timesteps = 0
        
        for t in range(chunk_size):
            move_x = features[t, agent_position, 24]  # move_direction_x at index 24
            move_y = features[t, agent_position, 25]  # move_direction_y at index 25
            
            # Check if the values are valid (not NaN or inf)
            if not (np.isnan(move_x) or np.isnan(move_y) or np.isinf(move_x) or np.isinf(move_y)):
                total_move_x += abs(move_x)  # Use absolute value
                total_move_y += abs(move_y)  # Use absolute value
                valid_timesteps += 1
        
        if valid_timesteps > 0:
            avg_move_x = total_move_x / valid_timesteps
            avg_move_y = total_move_y / valid_timesteps
            
            # print(f"Agent {agent_id} (position {agent_position}): avg_move_x={avg_move_x:.3f}, avg_move_y={avg_move_y:.3f}")
            
            # Check if average > 0.5
            if avg_move_x > 0.8 or avg_move_y > 0.8:
                # print(f"Agent {agent_id} meets criteria! Creating swapped feature set.")
                
                # Create a copy of features and swap this agent with the 0th agent
                swapped_features = features.copy()
                
                # Swap the agent data
                for t in range(chunk_size):
                    # Store 0th agent data temporarily
                    temp_agent_data = swapped_features[t, 0, :].copy()
                    # Copy current agent data to 0th position
                    swapped_features[t, 0, :] = swapped_features[t, agent_position, :].copy()
                    # Copy 0th agent data to current agent position
                    swapped_features[t, agent_position, :] = temp_agent_data
                
                # Add to feature variations
                feature_variations.append(swapped_features)
    
    # print(f"Created {len(feature_variations)} feature variations")
    return feature_variations


def inspect_hdf5_file(file_path):
    """
    Load and inspect the contents of an HDF5 file
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"📁 HDF5 File: {file_path}")
            print(f"📊 File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            print("=" * 60)
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"📄 Dataset: {name}")
                    print(f"   Shape: {obj.shape}")
                    print(f"   Dtype: {obj.dtype}")
                    print(f"   Size: {obj.size:,} elements")
                    print(f"   Memory: {obj.nbytes / (1024*1024):.2f} MB")
                    
                    # Show sample data for small datasets
                    if obj.size <= 100:
                        print(f"   Sample data: {obj[:]}")
                    else:
                        # Load data into memory first, then flatten
                        data_array = obj[:]
                        print(f"   Sample data (first 10): {data_array.flatten()[:10]}")
                        print(f"   Sample data (last 10): {data_array.flatten()[-10:]}")
                    
                    # Show statistics for numeric data
                    if np.issubdtype(obj.dtype, np.number):
                        data = obj[:]
                        print(f"   Min: {np.min(data):.6f}")
                        print(f"   Max: {np.max(data):.6f}")
                        print(f"   Mean: {np.mean(data):.6f}")
                        print(f"   Non-zero: {np.count_nonzero(data):,}")
                    
                    print()
                elif isinstance(obj, h5py.Group):
                    print(f"📁 Group: {name}")
                    print(f"   Keys: {list(obj.keys())}")
                    print()
            
            # Visit all objects in the file
            f.visititems(print_structure)
            
            # Show overall file structure
            print("=" * 60)
            print("🏗️  Overall File Structure:")
            def show_hierarchy(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}📄 {name.split('/')[-1]} (Dataset)")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name.split('/')[-1]} (Group)")
            
            f.visititems(show_hierarchy)
            
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found!")
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")

def load_and_analyze_chunk(file_path, game_id, chunk_name):
    """
    Load and analyze a specific chunk from the HDF5 file
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if game_id not in f:
                print(f"❌ Game ID {game_id} not found in file")
                return
                
            game_group = f[game_id]
            if chunk_name not in game_group:
                print(f"❌ Chunk {chunk_name} not found in game {game_id}")
                return
                
            chunk_data = game_group[chunk_name][:]
            print(f"📊 Chunk: {chunk_name}")
            print(f"   Shape: {chunk_data.shape}")
            print(f"   Dtype: {chunk_data.dtype}")
            print(f"   Size: {chunk_data.size:,} elements")
            
            # Analyze the 3D structure
            time_steps, agents, features = chunk_data.shape
            print(f"   Time steps: {time_steps}")
            print(f"   Agents: {agents}")
            print(f"   Features per agent: {features}")
            
            # Show sample data for first few time steps and agents
            print("\n🔍 Sample Data (First 3 time steps, first 2 agents):")
            for t in range(min(3, time_steps)):
                for a in range(min(2, agents)):
                    print(f"   Time {t}, Agent {a}: {chunk_data[t, a, :10]}...")  # First 10 features
            
            # Check for zero/invalid values
            zero_count = np.sum(chunk_data == 0)
            nan_count = np.sum(np.isnan(chunk_data))
            inf_count = np.sum(np.isinf(chunk_data))
            
            print(f"\n📈 Data Quality:")
            print(f"   Zero values: {zero_count:,} ({zero_count/chunk_data.size*100:.1f}%)")
            print(f"   NaN values: {nan_count:,}")
            print(f"   Inf values: {inf_count:,}")
            
    except Exception as e:
        print(f"❌ Error analyzing chunk: {e}")

def main():
    folder_path = "/Users/vaibhav/Desktop/clash_squad"
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    for game_folder in tqdm(subfolders, desc="Processing game folders"):
        game_json_objects = get_game_json_objects(folder_path, game_folder)

        if game_json_objects:
            try:
                game_id = game_json_objects[0]["game_id"]
                out_dir = "/Users/vaibhav/Desktop/game_logs_hdf5"
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/game_logs_{game_id}.h5"

                # Skip if already processed
                if os.path.exists(out_path):
                    # print(f"Skipping {game_id}: {out_path} already exists")
                    continue

                with h5py.File(out_path, "w") as f:
                    store_features(f, game_json_objects, game_id)
            except Exception as e:
                print(f"missed file for folder: {game_folder} due to error: {e}")
            


if __name__ == "__main__":
    # Uncomment the line below to inspect the HDF5 file after creation
    # inspect_hdf5_file("game_logs.h5")
    
    # Uncomment the lines below to analyze a specific chunk
    # load_and_analyze_chunk("game_logs.h5", "your_game_id", "chunk_0")
    
    main()
