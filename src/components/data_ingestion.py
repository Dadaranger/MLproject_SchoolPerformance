# Standard library imports
import os
import sys
from dataclasses import dataclass

# Third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Local application imports
from src.exception import CustomException
from src.logger import logging
#from src.components.data_transformation import DataTransformation, DataTransformationConfig
#from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion paths.
    Uses dataclass for automatic creation of initialization method.
    
    Attributes:
        train_data_path (str): Path where training data will be saved
        test_data_path (str): Path where test data will be saved
        raw_data_path (str): Path where raw data will be saved
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    """
    Class responsible for ingesting data from source and splitting it into training and test sets.
    """
    def __init__(self):
        """
        Initialize DataIngestion with configuration settings.
        Creates an instance of DataIngestionConfig to store file paths.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Performs the data ingestion process:
        1. Reads the source data
        2. Creates necessary directories
        3. Saves raw data
        4. Splits data into train and test sets
        5. Saves train and test sets separately
        
        Returns:
            tuple: Paths to the train and test data files
        
        Raises:
            CustomException: If any error occurs during the ingestion process
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from source
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create directories for storing the data files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split the data into training and test sets (80-20 split)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save test data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return paths to the train and test data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If any error occurs, raise custom exception with detailed error message
            raise CustomException(e, sys)

# This section only runs when the script is executed directly (not when imported as a module)
if __name__ == "__main__":
    # Create an instance of DataIngestion and perform data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Perform data transformation on the ingested data
    #data_transformation = DataTransformation()
    #train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Train and evaluate the model using the transformed data
    #modeltrainer = ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr, test_arr))



