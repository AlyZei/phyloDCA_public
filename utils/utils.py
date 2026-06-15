import os
from matplotlib import cm
import torch
import numpy as np

import inspect

def generate_colors(n):
    """Generates a list of n colors"""
    # Use the 'viridis' colormap, which is perceptually uniform
    colormap = cm.get_cmap('viridis', n)
    return [colormap(i) for i in range(n)]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

def to_tensor(array, device='cpu'):
    return torch.tensor(array, device=device) if isinstance(array, np.ndarray) else array

def createFolder(folder_name: str)-> None:
    """
    Creates a folder in the current directory if teh folder does not already exist.

    Args:
        folder_name (str): Name of the folder to be created.
    """
    if not os.path.exists(folder_name):
        # Create the new folder
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully in the current directory.")
    else:
        print(f"Folder '{folder_name}' already exists.")

def get_variable_name(var: object)-> str:
    """
    Returns the name of a variable as a string.

    Args:
        var (object): Variable whose name is to be returned.

    Returns:
        str: Name of the variable.
    """
    # Get the caller's frame
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    # Find the variable name
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

def get_all_file_paths(base_path: str)-> list:
    """Returns a list of all file paths in a directory and its subdirectories.

    Args:
        base_path (str): Path to the directory.

    Returns:
        list: List of all file paths in the directory and its subdirectories.
    """
    # List to store the file paths
    file_paths = []  
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Combine the root directory and file name to get the full path
            file_paths.append(os.path.join(root, file))
    return file_paths

def remove_charsequence(longer_string, sequence_to_remove):
    return longer_string.replace(sequence_to_remove, '')
