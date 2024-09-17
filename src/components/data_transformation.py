import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass
from utils import convert, convert_cast, convert_crew

# ML libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore

# Create the Data transformation config class 
@dataclass
class DataTranformationConfig:
    """Data Transformation Config Class"""
    cleaned_data_path : str = os.path.join('artifacts', 'final_data_set.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()
    
    def initiate_data_transformation(self, data):
        try:
            """
            Reduce the dimensionality of the dataset
            Columns to be keep
            1. genres
            2. id
            3. keywords
            4. title
            5. overview - important in content based
            6. cast- actors play important role in recommendation
            7. crew - directors also play imp role in recommendation
            """
            df = pd.read_csv(data)
            df = df[['movie_id', 'title', 'overview', 'genres', 'keywords','cast', 'crew']]
            
            #remove nan values
            df.dropna(inplace=True)

            #Preprocess genres
            df['genres'] = df['genres'].apply(convert)

            #preprocess keywords
            df['keywords'] = df['keywords'].apply(convert)

            #preprocess cast
            df['cast'] = df['cast'].apply(convert_cast)

            #perprocess crew
            df['crew'] = df['crew'].apply(convert_crew)

            #preprocess overview 
            df['overview'] = df['overview'].apply(lambda x: x.split())

            # Transformation to remove the space
            df['genres'] = df['genres'].apply(lambda x : [i.replace(' ', '') for i in x])
            df['keywords'] = df['keywords'].apply(lambda x : [i.replace(' ', '') for i in x])
            df['cast'] = df['cast'].apply(lambda x : [i.replace(' ', '') for i in x])
            df['crew'] = df['crew'].apply(lambda x : [i.replace(' ', '') for i in x])

            # Create the tags column
            df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']

            new_df = df.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

            #convert to string and lower case
            new_df['tags'] = new_df['tags'].apply(lambda x : " ".join(x))
            new_df['tags'] = new_df['tags'].apply(lambda x : x.lower())

            logging.info('Data Cleaning and transforming completed')

            #saving cleaned csv file
            new_df.to_csv(self.data_transformation_config.cleaned_data_path, index=False, header=True)
            logging.info('csv file created and exported to artifacts folder')

            return new_df
        
        except Exception as e:
            raise CustomException(e,sys)