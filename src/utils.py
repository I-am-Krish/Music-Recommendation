import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

def save_function(file_path, obj):
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
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}")
        raise CustomException(e, sys)
