import pickle
import pandas as pd


# Write function that takes our Dataset and returns features
def get_features(dataset, movie_id, user_id):
    """
    :param dataset: Our movies dataset is csv format
    :param movie_id: Integer representing the movie Id
    :param user_id: Integer representing the user Id
    :return: A DataFrame containing features
    """

    data = pd.read_csv(dataset)

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

    # Features
    features = features

    # Model
    model = model
    print(model)


def main():
    # Model
    model = pickle.load(open("model.pkl", 'rb'))

    # Reference Dataset
    movie_data = "movie_ratings_features.csv"

    # Get movie_id and user_id
    user_id, movie_id = 1, 1

    # Call get_features function passing the Dataset, movie_id and user_id
    features = get_features(dataset=movie_data, movie_id=movie_id, user_id=user_id)

    # Call get ratings function
    predict_rating(model=model, features=features)


main()
