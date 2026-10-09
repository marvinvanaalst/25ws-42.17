import torch
from collections import OrderedDict
import glob
import os
import json


from tqdm import tqdm

def average_model_weights(input_directory=None, destination_directory=None, model_loader_params=None, model_initializer=None, name=None):
    directory = input_directory
    """
    Averages the weights of multiple model checkpoints.
    Assumes all models have the same architecture.
    """
    # 0. Ensure the directory exists
    assert directory is not None, "Averaging Directory must be specified in average weighting module"
    assert destination_directory is not None, "Destination Directory must be specified in average weighting module"
    
    if not os.path.exists(directory):
        print("No checkpoints found.")
        return
    # 1. List of model checkpoint paths
    model_paths = glob.glob(f'{directory}/*.pt')  # or manually specify list
    loopy = tqdm(model_paths[1:], desc="Loading models")
    # 2. Load the first model's state_dict to initialize accumulator
    avg_state_dict = torch.load(model_paths[0], map_location='cpu')
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key].float()  # ensure float
    print(f"Loaded first model: {model_paths[0]} with keys: {list(avg_state_dict.keys())[:5]}...")
    
    # 3. Accumulate the weights from the rest of the models
    for idx, path in enumerate(loopy):
        loopy.set_description(f"Loading model {idx+2}/{len(model_paths)}")
        state_dict = torch.load(path, map_location='cpu')
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key].float()

    # 4. Average the weights
    for key in avg_state_dict:
        avg_state_dict[key] /= len(model_paths)

    # 5. Load into new model instance
    model = model_initializer(model_loader_params)
    model.load_state_dict(avg_state_dict)

    # Optional: Save the averaged model
    torch.save(model.state_dict(), f"{destination_directory}/average_stats=0000=_{name}.pt")
