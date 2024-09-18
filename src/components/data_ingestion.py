# import basic libraries
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass

# import other libraries from other components
from data_transformation import DataTransformation
from recommender import ModelTrainer

# ML libraries
import numpy as np # type: ignore
import pandas as pd # type: ignore

# Create the data ingestion config class 
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    logging.info('Data ingestion config created')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    #initiate the data ingestion
    def initiate_data_ingestion(self):
        try:
            # Create the df
            movies = pd.read_csv('Notebook/Dataset/tmdb_5000_movies.csv')
            credits = pd.read_csv('Notebook/Dataset/tmdb_5000_credits.csv')
            df = movies.merge(credits, on='title')
            logging.info('Data frame created')

            # make the artifacts directory 
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            #Save the df as csv file
            df.to_csv(self.data_ingestion_config.raw_data_path, index = False, header = True)
            logging.info('Data ingestion completed')

            return self.data_ingestion_config.raw_data_path
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    
    # Data Ingestion
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()

    #Data transformation 
    data_transformation = DataTransformation()
    final_data_frame = data_transformation.initiate_data_transformation(raw_data_path)

    #recommendation 
    recommender = ModelTrainer()
    movie_name = "My Date with Drew"
    recommender.initiate_recommendation(movie = movie_name, data_path= final_data_frame)