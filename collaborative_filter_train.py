from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import itertools
import datetime


def movie_recommender_train(ratings_train, number_neighbors):
    df = ratings_train.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)
    return distances, indices, df.index


def movie_recommender_test(ratings_test, ratings_train, movies, distances, indices, mappa_ind, number_neighbors,
                           test_method="user"):

    # Per checkare il tuo film: movies[movies["movieId"].isin(mappa_ind[indices[0,:]])]
    mappa = pd.DataFrame({"indici": range(len(indices)), "movieId": mappa_ind,
                          "title": movies["title"][movies["movieId"].isin(mappa_ind)]})

    # test_method = "user"
    if test_method == "user":
        all_ratings = ratings_test.copy()
    elif test_method == "rating":
        all_ratings = ratings_train.copy()
    ratings_test["pred"] = None
    del ratings_train

    for index, row in ratings_test.iterrows():
        film_nuovi = []
        i = mappa["indici"][mappa["movieId"] == row["movieId"]]
        if not i.empty:   # Esistono dei film che non sono mai stati visti da nessuno splittando per user
            i = i.to_list()[0]
            sim_movies = indices[i].tolist()
            movie_distances = distances[i].tolist()
            if i in sim_movies:  # quando il film più vicino è se stesso lo rimuovo
                id_movie = sim_movies.index(i)
                sim_movies.remove(i)
                movie_distances.pop(id_movie)

            else:
                sim_movies = sim_movies[:number_neighbors - 1]  # mi assicuro di consigliare in ogni caso sempre solo n-1 film
                movie_distances = movie_distances[:number_neighbors - 1]

            # La distanza è compresa tra 0 e 1 perché lo spazio è "positivo", tutti i vettori sono disposti nello stesso
            # quadrante, quindi non possono mai avere direzioni opposte non esistendo i voti negativi-> d compresa [0,1]
            movie_similarity = [1 - x for x in movie_distances]  # similarità = 1-distanza, + sono distanti - sono simili
            movie_similarity_copy = movie_similarity.copy()
            nominator = 0

            for s in range(0, len(movie_similarity)):
                # se l'utente non ha visto il film
                voto_utente = all_ratings["rating"][(all_ratings["userId"] == row["userId"]) &
                                                    (all_ratings["movieId"] ==
                                                     mappa["movieId"][mappa["indici"] == sim_movies[s]].to_list()[0])]
                if voto_utente.empty:
                    if len(movie_similarity_copy) == (number_neighbors - 1):  # scarto il film perché non l'ha visto
                        movie_similarity_copy.pop(s)

                    else:
                        movie_similarity_copy.pop(s - (len(movie_similarity) - len(movie_similarity_copy)))
                else:
                    nominator = nominator + movie_similarity[s] * voto_utente.to_list()[0]  # nominator + similarità*voto_utente

            if len(movie_similarity_copy) > 0:  # se l'utente aveva visto almeno un film simile posso fare la previsione, se no no
                if sum(movie_similarity_copy) > 0:
                    predicted_r = nominator / sum(movie_similarity_copy)

                else:
                    predicted_r = 0
            else:
                predicted_r = 0

            if predicted_r != 0:
                ratings_test["pred"][(ratings_test["userId"] ==
                                     row["userId"]) & (ratings_test["movieId"] == row["movieId"])] = predicted_r
        else:
            # print(f"{row.movieId} never rated on this dataset! New movie!")
            film_nuovi.append(row.movieId)

    voti_non_predetti = ratings_test[ratings_test["pred"].isna()].shape[0]
    voti_predetti = ratings_test[~ratings_test["pred"].isna()]
    MSE = mean_squared_error(voti_predetti["rating"], voti_predetti["pred"], squared=False)
    RMSE = mean_squared_error(voti_predetti["rating"], voti_predetti["pred"], squared=True)
    MAE = mean_absolute_error(voti_predetti["rating"], voti_predetti["pred"])

    data = datetime.datetime.now()
    with open(f"results_run_{data.strftime('%Y_%m_%d_%H_%M_%S')}.txt", "w") as f:
        f.write(f"Metodo di test: {test_method}\n"
                f"Dimensioni del vicinato: {number_neighbors}\n\n"
                f"film nuovi rispetto al trainingset: \t{len(film_nuovi)}\n\n"
                f"Rating che non è stato possibile prevedere per mancanza di film visti nel vicinato: \t"
                f"{voti_non_predetti}\n\n"
                f"Rating predetti: \t{voti_predetti.shape[0]}\n\n"
                f"MSE: \t{MSE}\nRMSE: \t{RMSE}\nMAE: \t{MAE}")
