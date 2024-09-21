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
    movies , posters = recommender.initiate_recommendation(selected_movie, movies_df)

    # First row
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(movies[i])
            st.image(posters[i])

    # Second row
    cols = st.columns(5)
    for i, col in enumerate(cols, start=5):  # Start from index 4 for the second row
        with col:
            st.text(movies[i])
            st.image(posters[i])