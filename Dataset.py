import pandas as pd
import re
from utilities import *


class Dataset:
    def __init__(self):
        root = "data\\MovieLensShort"
        file_rating = "ratings.csv"
        file_movie = "movies.csv"
        file_tags = "tags.csv"
        test_method = "ratings"  # "user"
        # test_method = "rating"
        pd.options.mode.chained_assignment = None
        verbose = False

        ratings = openSet(root, file_rating)  # il numero più piccolo è 0.5
        movies = openSet(root, file_movie)
        movies = movies[~(movies['genres'] == '(no genres listed)')].reset_index(drop=True)
        movies['genres'] = movies['genres'].str.replace('|', ' ')
        # change 'Sci-Fi' to 'SciFi' and 'Film-Noir' to 'Noir' otherwise there will be a genre sci, one fi, one film
        movies['genres'] = movies['genres'].str.replace('Sci-Fi', 'SciFi')
        movies['genres'] = movies['genres'].str.replace('Film-Noir', 'Noir')
        movies["title"] = movies["title"].apply(lambda x: re.sub(" [\(\[].*?[\)\]]", "", x))
        movies["title"] = movies["title"].apply(lambda x: Dataset.joinerTitle(x))

        ratings = ratings[ratings["movieId"].isin(movies["movieId"].unique())].reset_index(drop=True)

        self.ratings = ratings
        self.movies = movies
        self.tags = openSet(root, file_tags)

    # def transform(self):
    #     self.movies = self.movies[~(self.movies['genres'] == '(no genres listed)')].reset_index(drop=True)
    @staticmethod
    def joinerTitle(titolo):
        titolo = titolo.split(", ")
        if titolo[-1] in ["The", "Le", "A", "An"]:
            titolo.insert(0, titolo.pop())
            titolo = " ".join(titolo)
        else:
            titolo = ", ".join(titolo)
        return titolo


if __name__ == "__main__":
    d = Dataset()
