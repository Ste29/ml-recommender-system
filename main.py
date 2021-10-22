import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from collaborative_filter1 import *  ######
from collaborative_filter_train import *
from content_based import *
from utilities import *
from exploratory_analysis import *
from autoencoder import AutoEncoder


# 1. Tiri fuori la lista dei film affini
# 2. Consigli alcuni film usando sia content based che memory based
# 3. Ensamble dei 2 metodi e cerchi tra i film affini quale potresti consigliare (e poi fai una proposta anche per film
# non affini)
def interface(movies, ratings, ratings_train, ratings_test):

    while True:
        print("\nTell me a movie you recently liked or press q to quit.")
        # title = input("> ")
        title = "Monsters, Inc."
        if title == "q":
            break
        else:
            # genere + tag, permette di uscire un po' di più dalla bolla
            film_simili, tfidf_matrix = tfidf_content_based(movies, "document", title, 20)  # "genres": tfidf solo sul genere


            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=movies.index.tolist())

            ae = AutoEncoder(tfidf_df, validation_perc=0.1, lr=1e-3, intermediate_size=600, encoded_size=50)
            ae.train_loop(epochs=30)
            losses = pd.DataFrame(data=list(zip(ae.train_losses, ae.val_losses)),
                                  columns=['train_loss', 'validation_loss'])
            losses['epoch'] = (losses.index + 1) / 3
            fig, ax = plt.subplots()
            ax.plot(losses['epoch'], losses['train_loss'])
            ax.plot(losses['epoch'], losses['validation_loss'])
            ax.set_ylabel('MSE loss')
            ax.set_xlabel('epoch')
            ax.set_title('autoencoder loss over time')
            ax.legend()
            encoded = ae.get_encoded_representations()  # ora vanno decodificati e usati

            with open('../data/autoencoder_embeddings.pkl', 'wb') as fh:
                pickle.dump(encoded, fh)


            # movie_recommender([1, 6], 3, 10, ratings, movies)  # to recommend a title
            # movie_recommender(ratings_test["userId"].unique().to_list(), 3, 10, ratings_train, movies, ratings_test)  # to test algorithm ###
            # model based collaborative
            distances, indices, mappa_ind = movie_recommender_train(ratings_train, number_neighbors=6)
            movie_recommender_test(ratings_test, ratings_train, movies, distances, indices, mappa_ind,
                                   number_neighbors=6, test_method="rating")
            break



if __name__ == "__main__":
    # todo: implementare un sistema di filtraggio dei record, serve solo per movielens 25m e netflix

    # root = "data\\NetflixPrize"
    # file_rating = "combined_data_1.txt"
    # file_movies = "movie_titles.csv"
    root = "data\\MovieLensShort"
    file_rating = "ratings.csv"
    file_movie = "movies.csv"
    file_tags = "tags.csv"

    ratings = openSet(root, file_rating)  # il numero più piccolo è 0.5
    movies = openSet(root, file_movie)
    tags = openSet(root, file_tags)


    movies = movieAnalysis(movies, False)
    ratingAnalysis(ratings, False)
    # remove ratings to movies without metadata in movies dataset:
    ratings = ratings[ratings["movieId"].isin(movies["movieId"].unique())].reset_index(drop=True)



    # tags = pd.merge(tags, ratings, on="movieId", how="right")
    tags.fillna("", inplace=True)
    tags = pd.DataFrame(tags.groupby('movieId')['tag'].apply(lambda x: "{%s}" % ' '.join(x)))
    # tags.reset_index(inplace=True)
    movies = movies.join(tags, on="movieId", how="left")
    movies["tag"][movies["tag"].isna()] = "{}"
    movies['document'] = movies[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)



    # ratings_train, ratings_test = splitUserTrainTest(ratings, perc=80)
    ratings_train, ratings_test = splitRatingTrainTest(ratings, perc=80)
    ratings2 = pd.merge(ratings, movies, how='inner', on='movieId')
    # df = ratings2.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    interface(movies, ratings, ratings_train, ratings_test)


# trovo la lista dei film più simili con il metodo della hybrid, poi calcolo il rating sui film visti pesati in base
# alla similarità e infine MAE. Problemi non risolti: cosa fare quando ci sono pochi film visti dall'utente o film
# completamente nuovi
# Se mi genero il vicinato come knn dei film con voti simili, oppure come autoencoder di tfidf non cambia nulla, sempre
# vicinato è

# è verosimile non avere mai tutti i tag dei prodotti, per ora comincia con lo short, poi se serve usa i tag del full

# Baseline: voto medio è 3.5, MAE=0.83, MSE=1.09, RMSE=1.04