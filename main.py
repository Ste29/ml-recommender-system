import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from collaborative_filter import *
from content_based import *
from utilities import *
from exploratory_analysis import *


# 1. Tiri fuori la lista dei film affini
# 2. Consigli alcuni film usando sia content based che memory based
# 3. Ensamble dei 2 metodi e cerchi tra i film affini quale potresti consigliare (e poi fai una proposta anche per film
# non affini)
def interface(movies, ratings):

    while True:
        print("\nTell me a movie you recently liked or press q to quit.")
        # title = input("> ")
        title = "Monsters, Inc."
        if title == "q":
            break
        else:
            # tfidf_content_based(movies, title, 20)
            movie_recommender(1, 3, 10, ratings, movies)
            # model based collaborative




if __name__ == "__main__":
    # todo: implementare un sistema di filtraggio dei record, serve solo per movielens 25m e netflix

    # root = "data\\NetflixPrize"
    # file_rating = "combined_data_1.txt"
    # file_movies = "movie_titles.csv"
    root = "data\\MovieLensShort"
    file_rating = "ratings.csv"
    file_movie = "movies.csv"

    ratings = openSet(root, file_rating)  # il numero più piccolo è 0.5
    movies = openSet(root, file_movie)

    movies = movieAnalysis(movies, False)
    ratingAnalysis(ratings, False)

    # ratings2 = pd.merge(ratings, movies, how='inner', on='movieId')
    # df = ratings2.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    interface(movies, ratings)
