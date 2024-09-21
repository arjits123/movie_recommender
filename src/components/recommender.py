import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import stemming, save_obj, fetch_poster

#Importing important libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore
import nltk # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from nltk.stem.porter import PorterStemmer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

#Model config class
@dataclass
class ModelTrainerConfig:
    recommender_file_path : str = os.path.join('artifacts', 'recommender.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_recommendation(self, movie:str, data_path):
        try:
            df =  data_path
            #perform stemming on the tags column
            df['tags'] = df['tags'].apply(stemming)

            # used Bag of words technique using Countvectorisor
            cv = CountVectorizer(max_features=5000, stop_words='english')      
            # fit_transform gives you a sparse matrix
            sparse_matrix = cv.fit_transform(df['tags'])
            # convert to numpy array
            vectors = sparse_matrix.toarray() 

            #Calculate the cosin similarity for these vectors
            similarity_matrix = cosine_similarity(vectors)

            # save_obj(file_path = self.model_trainer_config.recommender_file_path, data_frame= similarity_matrix)

            #Get the movie index
            movie_index = df[df['title'] == movie].index[0]
            distances = similarity_matrix[movie_index]
            movies_list = sorted(list(enumerate(distances)),key = lambda x: x[1], reverse=True)[1:11]

            recommendations = []
            recommended_posters = []
            for i in movies_list:
                recommendations.append(df['title'][i[0]])
                #fetch poster from tmdb api - 1e81603975601c3b788394aa6b44415f
                movie_id = df['movie_id'][i[0]]
                recommended_posters.append(fetch_poster(movie_id = movie_id))
                
            logging.info('Recommendation part completed')
            return recommendations, recommended_posters
            
        except Exception as e:
            raise CustomException(e,sys)


