import dill # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore
from src.components.recommender import ModelTrainer

movies_dict = dill.load(open('artifacts/transformed_data_dict.pkl', 'rb'))
movies_df = pd.DataFrame(movies_dict)
movies_list = movies_df['title'].values

#recommendation
recommender = ModelTrainer()

#app development
st.title('Movie Recommendation system')
st.write('This is a simple movie recommendation system built with Streamlit')

selected_movie = st.selectbox("Select the movie to watch", movies_list)

if st.button('Recommned'):
    movies = recommender.initiate_recommendation(selected_movie, movies_df)
    for name in movies:
        st.write(name)
