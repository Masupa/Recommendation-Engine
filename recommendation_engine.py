import pickle
import pandas as pd


# Load needed
def load():
    # Model
    model = pickle.load(open("model.pkl", 'rb'))

    # Reference Dataset
    movie_data = pd.read_csv("movie_ratings_features.csv")

    return model, movie_data


# Write function that takes our Dataset and returns features
def get_features(dataset, movie_id, user_id):
    """
    :param dataset: Our movies dataset is csv format
    :param movie_id: Integer representing the movie Id
    :param user_id: Integer representing the user Id
    :return: A DataFrame containing features
    """

    data = dataset

    # Drop title, genres and rating; variables not part of features
    data.drop(labels=['title', 'genres', 'rating'], axis=1, inplace=True)

    # masks based on movie_id and user_id
    movie_filter = data['movieId'] == movie_id
    user_filter = data['userId'] == user_id

    # Filter based on masks
    features = data[movie_filter & user_filter]

    # Result
    return features


# Write function the model and predicts a rating
def predict_rating(model, features):
    """
    :param model: A model to predict the ratings of a movie a user
    :param features: A series of features used to make a prediction
    :return: Rating of a movie
    """

    # Model
    predicted_rating = model.predict(features.values)
    return predicted_rating


def main():

    # Load files
    model, movie_data = load()

    # User Movie
    movie_name = "Dracula: Dead and Loving It (1995)"
    movie_id = movie_data[movie_data['title'] == movie_name]['movieId'].values[0]

    # Get user_id
    user_id = 19

    # Call get_features function passing the Dataset, movie_id and user_id
    features = get_features(dataset=movie_data, movie_id=movie_id, user_id=user_id)

    # Call get ratings function
    predicted_rating = predict_rating(model=model, features=features)

    print(predicted_rating[0])
