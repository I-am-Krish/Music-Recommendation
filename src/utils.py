import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

def save_function(file_path, obj):
    """
    Save an object to a file using pickle.

    Parameters:
    - file_path (str): Path to save the object.
    - obj (object): Object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}")
        raise CustomException(e, sys)

def load_function(file_path):
    """
    Load an object from a file using pickle.

    Parameters:
    - file_path (str): Path to load the object from.

    Returns:
    - obj (object): Loaded object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}")
        raise CustomException(e, sys)

# Add more utility functions as needed for your project