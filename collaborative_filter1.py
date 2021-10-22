from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import itertools


def item_based_test_recommend(user, df, df1, num_recommendation):
    recommended_movies = []
    for m in df[df[user] != 0].index.tolist():
        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
        recommended_movies.append((m, predicted_rating))
    sorted_rm = sorted(recommended_movies, key=lambda x: x[1], reverse=True)
    return sorted_rm[:num_recommendation]


def item_based_recommend_movies(user, num_recommended_movies, df, df1, movies):
    # print('The list of the Movies {} Has Watched \n'.format(user))
    #
    # for m in df[df[user] > 0][user].index.tolist():
    #     print(movies["title"][movies["movieId"] == m])
    #
    # print('\n')

    recommended_movies = []

    for m in df[df[user] == 0].index.tolist():
        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
        recommended_movies.append((m, predicted_rating))

    sorted_rm = sorted(recommended_movies, key=lambda x: x[1], reverse=True)

    print('The list of the Recommended Movies \n')
    rank = 1
    for recommended_movie in sorted_rm[:num_recommended_movies]:
        print('{}: {} - predicted rating:{}'.format(rank, movies["title"][movies["movieId"] == recommended_movie[0]],
                                                    recommended_movie[1]))
        rank = rank + 1


def predict_rating(indices, m, distances, number_neighbors, user_index, df, ratings_test):  # todo: implementare valutazione metriche per rating test invece che calcolarle su df, valutare anche split in modulo di recommend e modulo di test
    sim_movies = indices[m].tolist()  # mi trovo i film "simili"
    movie_distances = distances[m].tolist()

    if m in sim_movies:  # quando il film più vicino è se stesso lo rimuovo
        id_movie = sim_movies.index(m)
        sim_movies.remove(m)
        movie_distances.pop(id_movie)

    else:
        sim_movies = sim_movies[:number_neighbors - 1]  # mi assicuro di consigliare in ogni caso sempre solo 2 film
        movie_distances = movie_distances[:number_neighbors - 1]

    movie_similarity = [1 - x for x in movie_distances]  # similarità = 1-distanza, + sono distanti - sono simili
    movie_similarity_copy = movie_similarity.copy()
    nominator = 0

    for s in range(0, len(movie_similarity)):
        if df.iloc[sim_movies[s], user_index] == 0:  # se l'utente non ha visto il film
            if len(movie_similarity_copy) == (number_neighbors - 1):  # scarto il film perché non l'ha visto
                movie_similarity_copy.pop(s)

            else:
                movie_similarity_copy.pop(s - (len(movie_similarity) - len(movie_similarity_copy)))

        else:
            nominator = nominator + movie_similarity[s] * df.iloc[
                sim_movies[s], user_index]  # nominator + similarità*voto_utente

    if len(movie_similarity_copy) > 0:  # se l'utente aveva visto almeno un film simile posso fare la previsione, se no no
        if sum(movie_similarity_copy) > 0:
            predicted_r = nominator / sum(movie_similarity_copy)

        else:
            predicted_r = 0

    else:
        predicted_r = 0
    return predicted_r


def movie_recommender(user_list, number_neighbors, num_recommendation, ratings, movies, ratings_test=False):
    # Se ci sono dei Nan non funziona il knn, allora setto tutto a 0, tanto il minimo voto che si può assegnare è 0.5
    # sulle righe ci sono i film, sulle colonne gli utenti
    # ratings2 = pd.merge(ratings, movies, how='inner', on='movieId')
    # df = ratings2.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    df = ratings.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)
    df1 = df.copy()

    knn = NearestNeighbors(metric='cosine', algorithm='brute')  # todo: studia come funziona
    knn.fit(df.values)  # ogni riga di indices è un film, sulla prima colonna il film più vicino (se stesso) poi il
    # secondo e terzo più vicini, distances sono le distanze
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)
    # Siccome l'id dei film non coincide con le righe di df hai 2 opzioni, o fai il merge così da mettere sulle righe i
    # titoli, oppure tieni gli id e poi fai il retrieve con
    # movies[movies["movieId"]==df2.index[418]] questo è il secondo film più vicino a Toy Story
    test_recommendation = []
    for user in user_list:
        user_index = df.columns.tolist().index(user)
        # la similarità dei film è dovuta ai voti che tutti gli utenti hanno dato insieme, non dipende dallo specifico utente
        for m, t in list(enumerate(df.index)):  # m: index del film, t: movieId
            if ratings_test is False:   # con iloc ragioni per indici, con loc per id
                if df.iloc[m, user_index] == 0:  # cerco i film che lo user desiderato non ha ancora guardato
                    predicted_r = predict_rating(indices, m, distances, number_neighbors, user_index, df)
                    df1.iloc[m, user_index] = predicted_r
            else:
                if df.iloc[m, user_index] != 0:  # cerco i film che lo user desiderato ha già guardato
                    predicted_r = predict_rating(indices, m, distances, number_neighbors, user_index, df, ratings_test)
                    df1.iloc[m, user_index] = predicted_r

        if ratings_test is False:
            item_based_recommend_movies(user, num_recommendation, df, df1, movies)
        else:
            test_recommendation.append(item_based_test_recommend(user, df, df1, num_recommendation))

    all_ratings = []
    all_pred = []
    non_pred = 0
    if ratings_test is not False:
        for user in user_list:
            # user_index = df.columns.tolist().index(user)
            all_ratings.append(df.loc[df1.loc[:, user] != 0, user].to_list())
            all_pred.append(df1.loc[df1.loc[:, user] != 0, user].to_list())
            non_pred += (len(df.loc[df.loc[:, user] != 0, user].to_list()) -
                         len(df1.loc[df1.loc[:, user] != 0, user].to_list()))
        rat = list(itertools.chain.from_iterable(all_ratings))
        pred = list(itertools.chain.from_iterable(all_pred))
        MSE = mean_squared_error(rat, pred, squared=False)
        RMSE = mean_squared_error(rat, pred, squared=True)
        MAE = mean_absolute_error(rat, pred)
