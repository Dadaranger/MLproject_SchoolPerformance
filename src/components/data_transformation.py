import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Uses dataclass for automatic creation of initialization method.
    """
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    """
    Class responsible for transforming the data.
    Includes methods for creating a preprocessing object and applying transformations to train and test data.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create and return a preprocessing object that applies transformations to numerical and categorical features.
        
        Returns:
            ColumnTransformer: A transformer that applies different preprocessing pipelines to numerical and categorical features.
        
        Raises:
            CustomException: If any error occurs during the creation of the preprocessing object.
        """
        try:
            logging.info("Data Transformation initiated")

            # Define which columns are numerical and which are categorical
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', num_pipeline, numerical_columns),
                    ('categorical', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Data Transformation completed")

            return preprocessor
        
        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys) from e
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            # Read the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessor.transform(input_features_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object")

            save_object(
                
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys) from e
