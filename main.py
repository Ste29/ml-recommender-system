import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# import fastai
# from fastai.learner import *
# from fastai.tabular.data import *

# from collaborative_filter1 import *  ######
from collaborative_filter_train import *
from content_based import *
from utilities import *
from exploratory_analysis import *
from autoencoder import AutoEncoder
from similarity import *
# from prova import *
from popular_rating import *
from matrix_factorisation import *
from lightfm_tentativo import *
from surprise_tentativo import *


def interface(movies, ratings, ratings_train, ratings_test, test_method, verbose):
    # from scipy.sparse import csr_matrix
    # df = ratings_train.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)
    # ppp = csr_matrix(df.values)
    surprise_tentativo(ratings)
    lightfm_tentativo(ratings, ratings_train, ratings_test, movies)



    distances, indices, mappa_ind = movie_recommender_train(ratings_train, number_neighbors=10)
    while True:
        print("\nOptions:\n")
        print("q: quit\npopular: get the most popular movies on the platform\ntest: test a series of ML algorithm\n")
        print("Lastly you can write the title of a movie you liked in order to get a list of related movies")
        title = input("> ")
        # title = "test"
        # title = "Monsters, Inc."
        if title == "q":
            break
        elif title == "popular":
            popular_rating(ratings_train, movies, ratings_test, True, test_name="popular", test_method=test_method)
            weighted_popular_rating(ratings_train, movies, ratings_test, True, test_name="poluplar_weighted",
                                   test_method=test_method)
        elif title == "test":
            title = "Monsters, Inc."

            matrix_factorisation(ratings_train, ratings_test, ratings, movies)
            DL_factorization(ratings, movies, ratings_train, ratings_test)
            hybrid_system(ratings, movies)
            tfidf_factorization(ratings, movies, ratings_train, ratings_test)

            popular_rating(ratings_train, movies, ratings_test, True, test_name="popular", test_method=test_method)
            # genere + tag, permette di uscire un po' di più dalla bolla
            film_simili, tfidf_matrix = tfidf_content_based(movies, "document", title, 20)  # "genres": tfidf solo sul genere

            # movie_recommender([1, 6], 3, 10, ratings, movies)  # to recommend a title
            # movie_recommender(ratings_test["userId"].unique().to_list(), 3, 10, ratings_train, movies, ratings_test)  # to test algorithm ###
            # model based collaborative
            indices, distances = indDist(tfidf_matrix, number_neighbors=6)  # questo modo di calcolare le distanze in base a linear_kernel potrebbe essere sbagliato cerca di capire cosa è sto lin_ker
            # movie_recommender_test(ratings_test, ratings_train, movies, distances, indices, movies["movieId"].to_list(),
            #                        number_neighbors=6, test_name="tfidf", test_method="rating")
            try:
                encoded = pd.read_pickle("data/autoencoder_embeddings.pkl")
                # with open('data/autoencoder_embeddings.pkl', 'rb') as fh:
                #     encoded = pickle.load(fh)
            except:
                encoded = traiAutoencoder(tfidf_matrix, movies)  # encoded lo usi poi esattamente come usavi tfidf_matrix

            content_embeddings = pd.DataFrame(encoded)
            sim_model_cont = SimilarityPredictions(content_embeddings, similarity_metric="cosine")
            # cont_output = pd.DataFrame(sim_model_cont.predict_similar_items(seed_item=0, n=26744))
            # ora per ogni item_id (è un indice! Quindi va convertito in movieId) ho un similarity score.

            indices, distances = similarityIndDist(sim_model_cont, number_neighbors=6)

            # indices, distances = indDist(encoded, number_neighbors=6)
            movie_recommender_test(ratings_test, ratings_train, movies, distances, indices, movies["movieId"].to_list(),
                                   number_neighbors=6, test_name="encoded", test_method=test_method)

            distances, indices, mappa_ind = movie_recommender_train(ratings_train, number_neighbors=6)
            movie_recommender_test(ratings_test, ratings_train, movies, distances, indices, mappa_ind,
                                   number_neighbors=6, test_name="knn", test_method=test_method)
            testing_user_user(ratings_train, ratings_test)
            break
        else:
            print("\nTF-IDF Genres approach:\n")
            film_simili, tfidf_matrix = tfidf_content_based(movies, "genres", title, 10)
            print("\nTF-IDF Tag approach:\n")
            film_simili, tfidf_matrix = tfidf_content_based(movies, "document", title, 10)

            try:
                encoded = pd.read_pickle("data/autoencoder_embeddings.pkl")
            except:
                pass
            print("\nTF-IDF Encoded approach:\n")
            content_embeddings = pd.DataFrame(encoded)
            sim_model_cont = SimilarityPredictions(content_embeddings, similarity_metric="cosine")
            closest_title, distance_score = find_closest_title(title, movies)
            cont_output = pd.DataFrame(sim_model_cont.predict_similar_items(
                seed_item=movies[movies["title"] == closest_title].index[0], n=10))
            for id in cont_output["item_id"]:
                print(movies.iloc[id].title_year)

            print("\nCollaborative filter KNN approach:\n")
            for id in indices[movies[movies["title"] == closest_title].index[0]]:
                print(movies.iloc[id].title_year)

            user_based_collaborative(ratings_train, movies, verbose)
            # break



if __name__ == "__main__":
    # todo: implementare un sistema di filtraggio dei record, serve solo per movielens 25m e netflix

    # root = "data\\NetflixPrize"
    # file_rating = "combined_data_1.txt"
    # file_movies = "movie_titles.csv"
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
    tags = openSet(root, file_tags)

    # In dataset molto grossi ha senso rimuovere i film e gli utenti poco attivi
    movies = movieAnalysis(movies, verbose)
    ratingAnalysis(ratings, verbose)
    # remove ratings to movies without metadata in movies dataset:
    ratings = ratings[ratings["movieId"].isin(movies["movieId"].unique())].reset_index(drop=True)



    # tags = pd.merge(tags, ratings, on="movieId", how="right")
    tags.fillna("", inplace=True)
    tags = pd.DataFrame(tags.groupby('movieId')['tag'].apply(lambda x: "{%s}" % ' '.join(x)))
    # tags.reset_index(inplace=True)
    movies = movies.join(tags, on="movieId", how="left")
    movies["tag"][movies["tag"].isna()] = "{}"
    movies['document'] = movies[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)

    if test_method == "user":
        ratings_train, ratings_test = splitUserTrainTest(ratings, perc=80)
    else:
        ratings_train, ratings_test = splitRatingTrainTest(ratings, perc=80)
    ratings2 = pd.merge(ratings, movies, how='inner', on='movieId')
    # df = ratings2.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    interface(movies, ratings, ratings_train, ratings_test, test_method, verbose)


# trovo la lista dei film più simili con il metodo della hybrid, poi calcolo il rating sui film visti pesati in base
# alla similarità e infine MAE. Problemi non risolti: cosa fare quando ci sono pochi film visti dall'utente o film
# completamente nuovi
# Se mi genero il vicinato come knn dei film con voti simili, oppure come autoencoder di tfidf non cambia nulla, sempre
# vicinato è

# è verosimile non avere mai tutti i tag dei prodotti, per ora comincia con lo short, poi se serve usa i tag del full

# Baseline: voto medio è 3.5, MAE=0.83, MSE=1.09, RMSE=1.04