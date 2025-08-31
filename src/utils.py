# Import required libraries
import os      # For file and directory operations
import sys     # For system-specific parameters and functions

# Data manipulation and analysis libraries
import numpy as np 
import pandas as pd
import dill    # For serializing/deserializing Python objects
import pickle  # For serializing/deserializing Python objects

# Machine learning metrics and tools
from sklearn.metrics import r2_score  # For calculating R-squared score
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning

# Custom exception handling
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle serialization.
    
    Args:
        file_path (str): Path where the object should be saved
        obj: Any Python object to be serialized and saved
    
    Raises:
        CustomException: If any error occurs during the saving process
    """
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified file in binary mode
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple machine learning models with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Testing features
        y_test: Testing target values
        models (dict): Dictionary of model names and their corresponding objects
        param (dict): Dictionary of model names and their hyperparameter grids
    
    Returns:
        dict: Dictionary containing model names and their test R-squared scores
    
    Raises:
        CustomException: If any error occurs during model evaluation
    """
    try:
        report = {}

        # Iterate through each model
        for i in range(len(list(models))):
            # Get the current model and its parameters
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Perform grid search with cross-validation
            gs = GridSearchCV(model, para, cv=3)  # 3-fold cross-validation
            gs.fit(X_train, y_train)

            # Update model with best parameters found
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R-squared scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load a Python object from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file to load
    
    Returns:
        The deserialized Python object
    
    Raises:
        CustomException: If any error occurs during loading
    """
    try:
        # Open and load the pickled object
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
