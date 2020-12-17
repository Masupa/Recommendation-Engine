# Import libraries
import numpy as np  # linear algebra
import pandas as pd  # data manipulation & analysis
from sklearn.metrics.pairwise import cosine_similarity

# Import movie and ratings dataset
movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# Merge movies and ratings dataset
movies_data = pd.merge(movies, ratings, on='movieId')

# Helper Functions

def get_similar_movies(movie_title, rating):
    """
    :param movie_title: Title of movie in engine
    :param rating: Rating given to movie
    """
    
    similar_score = cosine_sim_data[movie_title] * (rating - 2.5)
    return similar_score.sort_values(ascending=False)

# With a collaborate-based filtering recommendation, we are interested in the movie
# user, and rating user gave to a movie; we can drop "timestamp", "genre" & "movieId"
movies_data.drop(labels=['movieId', 'timestamp', 'genres'], axis=1, inplace=True)

# Use Pivot table;
movies_data = pd.pivot_table(movies_data, values='rating', index='userId', columns='title')

# Drop movies with user ratings less than N, where N is an integer representing the number of ratings
# Let N be 15
movies_data.dropna(axis=1, thresh=15, inplace=True)

# Fill NA values with 0
movies_data.fillna(value=0, inplace=True)


# Standardize the ratings
def standardize(row):
    """
    :param row: Row entry from a DataFrame
    :return: Standardized values
    """
    
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row
    

# Standardizing values across row
movie_ratings_std = movies_data.apply(standardize)

# Transpose matrix to calculate similarity between items
cosine_sim = cosine_similarity(movie_ratings_std.T)

# DataFrame containing similarity scores between across movies
cosine_sim_data = pd.DataFrame(cosine_sim, index=movies_data.columns, columns=movies_data.columns)
