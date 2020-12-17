import pickle
import pandas as pd
import streamlit as st
import content_based
import collaborative_based
import recommendation_engine as re


def main_interface(engine_kind):
    """
    :engine_kind: The type of recommendation system to use. I.e. "Content-based" or "Collaborative-based"
    :movie_data: Dataset of movies
    :return: UI Interface
    """
    st.title("Movie Recommendation Engine")

    st.subheader("Welcome, human! Want to know how you know how you will likely rate a movie?\
        Then enter the movie name, and you user ID according to our database.")

    st.text("")
    movie_title = st.text_input("Enter the title of the movie", value='Dracula: Dead and Loving It (1995)')

    if engine_kind == 'Predict Rating':
        st.text("")
        user_id = st.number_input("What is your user ID?", value=1, min_value=0, max_value=100)

        st.text("")
        button = st.button("Predict Rating")

        # Load
        model, movie_data = re.load()

        # Movie ID
        movie_id = movie_data[movie_data['title'] == movie_title]['movieId'].values[0]

        # Features
        features = re.get_features(dataset=movie_data, movie_id=movie_id, user_id=user_id)

    if engine_kind != 'Predict Rating':
        st.text("")
        button = st.button("Recommend Movies")


    if button and engine_kind != 'Predict Rating':
        if engine_kind == 'Content-Based':
            # Gettting Movies
            movies = content_based.get_recommendation(movie_title=movie_title)
        elif engine_kind == 'Collaborative-Based':
            # Gettting Movies
            movies = collaborative_based.get_similar_movies(movie_title, rating=5).iloc[:10].index

        # Rating
        st.text("")
        st.text("Recommended Movies")
        st.write("Movie 1:      {}".format(movies[0]))
        st.write("Movie 2:      {}".format(movies[1]))
        st.write("Movie 3:      {}".format(movies[2]))
        st.write("Movie 4:      {}".format(movies[3]))
        st.write("Movie 5:      {}".format(movies[4]))
        st.text("")

    if button and engine_kind == 'Predict Rating':

        # Prediction
        predicted_rating = re.predict_rating(model=model, features=features)[0]

        st.text("")
        st.write("Predicted Rating")
        st.write("Rating: {}".format(predicted_rating))
        st.text("")


def sidebar(): 
    # Type of recommendation selected
    recommendation_type = st.sidebar.selectbox(
        "Recommendation System Type",
        ("Content-Based", "Collaborative-Based", "Predict Rating")
    )

    return recommendation_type


def main():

    # Left side bar
    engine_kind = sidebar()

    # Main Interface
    main_interface(engine_kind=engine_kind)


if __name__ == "__main__":
    main()
