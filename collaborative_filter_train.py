from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import itertools
import datetime
from similarity import *
import matplotlib.pyplot as plt


def movie_recommender_train(ratings_train, number_neighbors):
    """Collaborative item based memory method"""
    df = ratings_train.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)
    # df = ratings_train.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    # df_p_imputed = df.T.fillna(df.mean(axis=1)).T  # todo: Come cambia fillando i na con la media?

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)
    return distances, indices, df.index


def user_based_collaborative(ratings_train, movies, verbose):
    # Quando fai la similarity e sulle righe hai gli utenti sarà una user-user, se hai i film sarà item-item
    df_p = ratings_train.pivot_table(index='userId', columns='movieId', values='rating')
    # User index for recommendation
    user_index = 1

    # Number of similar users for recommendation
    n_recommendation = 10

    # Plot top n recommendations
    n_plot = 10

    # Fill in missing values, i voti che non so sono la media dei voti dati da quell'utente
    df_p_imputed = df_p.T.fillna(df_p.mean(axis=1)).T

    # Compute similarity between all users
    sim_model_cont = SimilarityPredictions(df_p_imputed, similarity_metric="cosine")
    sim_model_cont.similarity_matrix -= np.eye(sim_model_cont.similarity_matrix.shape[0])
    similarity = sim_model_cont.similarity_matrix
    diz = sim_model_cont.predict_similar_items(user_index, df_p_imputed.shape[0])
    similar_user_index, similar_user_score = pd.Series(diz["item_id"]), pd.Series(diz["similarity_score"])

    # Get unrated movies
    unrated_movies = df_p.iloc[user_index][df_p.iloc[user_index].isna()].index

    # Weight ratings of the top n most similar users with their rating and compute the mean for each movie
    mean_movie_recommendations = (df_p_imputed.loc[similar_user_index[:n_recommendation]].T
                                  * similar_user_score[:n_recommendation].values).T.mean(axis=0)

    # Filter for unrated movies and sort results
    best_movie_recommendations = mean_movie_recommendations[unrated_movies].sort_values(
        ascending=False).to_frame().join(movies.set_index('movieId'))

    if verbose:
        titles_rat = best_movie_recommendations.iloc[:n_plot, 3].astype(str)
        plt.figure(figsize=(10, 6), dpi=100)
        plt.xlim([4.2, 5])  # , ylim=(ymin, ymax))
        plt.grid(axis="x", zorder=0)
        plt.barh(list(range(1, n_plot + 1)), best_movie_recommendations.iloc[:n_plot, 0], color='#db0000', zorder=3)
        for i, v in enumerate(titles_rat):
            plt.text(best_movie_recommendations.iloc[:n_plot, 0].iloc[i] + 0.05, i + 1, v)

        plt.title('Ranking Of Top {} Recommended Movies For A User Based On Similarity'.format(n_plot))
        plt.xlabel('Recommendation-Rating')
        plt.ylabel('Movie')
        plt.show()


def testing_user_user(ratings_train, ratings_test):
    df_p = ratings_train.pivot_table(index='userId', columns='movieId', values='rating')
    df_p_imputed = df_p.T.fillna(df_p.mean(axis=1)).T
    n_recommendation = 10

    # Compute similarity between all users
    sim_model_cont = SimilarityPredictions(df_p_imputed, similarity_metric="cosine")
    sim_model_cont.similarity_matrix -= np.eye(sim_model_cont.similarity_matrix.shape[0])
    similarity = sim_model_cont.similarity_matrix

    prediction = []
    # Iterate over all testset items
    for user_id in ratings_test['userId'].unique():
        try:
            diz = sim_model_cont.predict_similar_items(user_id, df_p_imputed.shape[
                0])  # user_id_mapping[user_id], df_p_imputed.shape[0])
        except:
            for movie_id in ratings_test[ratings_test['userId'] == user_id]['movieId'].values:
                prediction.append([user_id, movie_id, np.nan])
        similar_user_index, similar_user_score = pd.Series(diz["item_id"]), pd.Series(diz["similarity_score"])

        for movie_id in ratings_test[ratings_test['userId'] == user_id]['movieId'].values:

            if movie_id in df_p_imputed.columns:
                score = (df_p_imputed.loc[similar_user_index[:n_recommendation]][movie_id]
                         * similar_user_score[:n_recommendation].values).values.sum() \
                        / similar_user_score[:n_recommendation].sum()
                prediction.append([user_id, movie_id, score])
            else:
                prediction.append([user_id, movie_id, np.nan])

    # Create prediction DataFrame
    df_pred = pd.DataFrame(prediction, columns=['userId', 'movieId', 'Prediction']).set_index(['userId', 'movieId'])
    df_pred = ratings_test.set_index(['userId', 'movieId']).join(df_pred)

    # Get labels and predictions
    y_true = df_pred['rating']
    y_pred = df_pred['Prediction']

    # Compute RMSE
    n_film_test_set = y_pred.index.unique().shape[0]
    n_rating_test_set = y_pred.shape[0]
    n_film_non_predetti = y_pred.index.unique().shape[0] - y_pred[~y_pred.isna()].index.unique().shape[0]
    n_rating_non_predetti = y_pred.shape[0] - y_pred[~y_pred.isna()].shape[0]
    y_true = y_true[~y_pred.isna()]  # can't make predictions if the movie was never seen
    y_pred = y_pred[~y_pred.isna()]
    RMSE = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    MSE = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=True)
    MAE = mean_absolute_error(y_true=y_true, y_pred=y_pred)

    data = datetime.datetime.now()
    with open(f"results_run_collaborative_useruser_{data.strftime('%Y_%m_%d_%H_%M_%S')}.txt", "w") as f:
        f.write(f"Metodo di test: ratings\n"
                f"film nuovi rispetto al trainingset: \t{n_film_non_predetti}\n\n"
                f"Rating che non è stato possibile prevedere per mancanza di film visti nel vicinato: \t"
                f"{n_rating_non_predetti}\n\n"
                f"Rating predetti: \t{y_pred[~y_pred.isna()].index.unique().shape[0]}\n\n"
                f"MSE: \t{MSE}\nRMSE: \t{RMSE}\nMAE: \t{MAE}\n\n")





def movie_recommender_test(ratings_test, ratings_train, movies, distances, indices, mappa_ind, number_neighbors,
                           test_name, test_method="user"):

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
    RMSE = mean_squared_error(voti_predetti["rating"], voti_predetti["pred"], squared=False)
    MSE = mean_squared_error(voti_predetti["rating"], voti_predetti["pred"], squared=True)
    MAE = mean_absolute_error(voti_predetti["rating"], voti_predetti["pred"])

    voti_predetti["baseline"] = voti_predetti["rating"].mean()
    MSE_baseline = mean_squared_error(voti_predetti["rating"], voti_predetti["baseline"], squared=True)
    RMSE_baseline = mean_squared_error(voti_predetti["rating"], voti_predetti["baseline"], squared=False)
    MAE_baseline = mean_absolute_error(voti_predetti["rating"], voti_predetti["baseline"])

    data = datetime.datetime.now()
    with open(f"results_run_{test_name}_{data.strftime('%Y_%m_%d_%H_%M_%S')}.txt", "w") as f:
        f.write(f"Metodo di test: {test_method}\n"
                f"Dimensioni del vicinato: {number_neighbors}\n\n"
                f"film nuovi rispetto al trainingset: \t{len(film_nuovi)}\n\n"
                f"Rating che non è stato possibile prevedere per mancanza di film visti nel vicinato: \t"
                f"{voti_non_predetti}\n\n"
                f"Rating predetti: \t{voti_predetti.shape[0]}\n\n"
                f"MSE: \t{MSE}\nRMSE: \t{RMSE}\nMAE: \t{MAE}\n\n"
                f"Baseline:\n"
                f"MSE: \t{MSE_baseline}\nRMSE: \t{RMSE_baseline}\nMAE: \t{MAE_baseline}")
