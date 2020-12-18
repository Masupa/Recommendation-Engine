# Import libraries
import numpy as np # linear algebra
import pandas as pd # read & manipulating data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import data
movie_data = pd.read_csv('movie_dataset.csv')

# Helper Functions

def combine_features(data):
    """
    :param data: DataFrame containing features with content
    :return: String as a result of concatenating values from row entries
    """
    return (data['genres'] + " " + data['keywords'] + " " \
            + data['original_language'] + " " + data['cast'] + " " + data['director'])


def get_index_from_title(title):
    """
    :param title: Title of movie liked by user
    :return: index of title in DataFrame
    """

    try:
        title = movie_data[movie_data.title == title]['index'].values[0]
    except:
        return "Movie not in dataset!"

    return title

def get_title_from_index(index):
    """
    :param index: Index of movie
    :return: title of movie at index 'index'
    """
    
    return movie_data[movie_data.index == index]['title'].values[0]


def get_recommendation(movie_title):
    # Movie ID
    movie_index = get_index_from_title(movie_title)

    if movie_index == "Movie not in dataset!":
        return "Movie not in dataset!"

    # List of turples of index and cosine similarity score of movies
    similar_movies = sorted(list(enumerate(cosine_sim[movie_index])), key=lambda x:x[1], reverse=True)

    # Empty List ~ Top 10 Movies
    top_10_movies = list()

    for movie in similar_movies[:10]:
        movie_title = get_title_from_index(index=movie[0])
        top_10_movies.append(movie_title)

    return top_10_movies


# Feature Selection
features = ['genres', 'keywords', 'original_language', 'cast', 'director']

# Fill empty entries with empty string
for feature in features:
    movie_data[feature].fillna(value='', inplace=True,)

# Apply "combine_features" to movie_data and assign to new column "content"
movie_data['content'] = movie_data.apply(combine_features, axis=1)


# Initialize Count Vectorizer
cv = CountVectorizer()

# Fit Transform "content"
count_matrix = cv.fit_transform(movie_data['content'])

# Cosine Similarity between movies
cosine_sim = cosine_similarity(count_matrix)


# Movie liked by user
# # movie_user_liked = "Avatar"
