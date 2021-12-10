import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
import datetime


def popular_rating(ratings_train, movies, ratings_test, verbose=False, test_name="poluplar", test_method="user"):
    # Top films for users without ratings:
    df_p = ratings_train.pivot_table(index='userId', columns='movieId', values='rating')

    # Top n movies
    n = 10
    min_count = 10
    # Compute mean rating for all movies
    ratings_mean = df_p.mean(axis=0).sort_values(ascending=False).rename('Rating-Mean').to_frame()
    # Count ratings for all movies
    ratings_count = df_p.count(axis=0).sort_values(ascending=False).rename('Rating-Count').to_frame()
    # Combine ratings_mean, ratings_count and movie_titles
    ratings_stats = ratings_mean.join(ratings_count)
    ratings_stats = ratings_stats[ratings_stats["Rating-Count"] > min_count]
    tmp = pd.DataFrame(ratings_stats.head(n)["Rating-Mean"])
    movies = movies.set_index("movieId")
    ranking_mean_rating = tmp.join(ratings_count).join(movies.drop('year', axis=1))
    movies = movies.reset_index()
    # Join labels and predictions
    df_prediction = ratings_test.set_index('movieId').join(ratings_mean)[['rating', 'Rating-Mean']]
    y_true = df_prediction['rating']
    y_pred = df_prediction['Rating-Mean']
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
    # Computing the mean rating for all movies creates a ranking. The recommendation will be the same for all
    # users and can be used if there is no information on the user.
    # Variations of this approach can be separate rankings for each country/year/gender/... and to use them
    # individually to recommend movies/items to the user.
    #
    # It has to be noted that this approach is biased and favours movies with fewer ratings, since large numbers
    # of ratings tend to be less extreme in its mean ratings.

    if verbose:
        titles_rat = ranking_mean_rating['title'].astype(str) + ': ' + ranking_mean_rating['Rating-Count'].astype(str) \
                     + ' Ratings'
        plt.figure(figsize=(10, 6), dpi=100)
        plt.xlim([4.2, 5])  # , ylim=(ymin, ymax))
        plt.grid(axis="x", zorder=0)
        plt.barh(list(range(1, n + 1)), ranking_mean_rating['Rating-Mean'],  color='#db0000', zorder=3)
        for i, v in enumerate(titles_rat):
            plt.text(ranking_mean_rating['Rating-Mean'].iloc[i] + 0.05, i + 1, v)

        plt.title('Ranking Of Top {} Mean-Movie-Ratings: {:.4f} RMSE'.format(n, RMSE))
        plt.xlabel('Mean-Rating')
        plt.ylabel('Movie')
        plt.show()

    data = datetime.datetime.now()
    with open(f"results_run_{test_name}_{data.strftime('%Y_%m_%d_%H_%M_%S')}.txt", "w") as f:
        f.write(f"Metodo di test: {test_method}\n"
                f"film nuovi rispetto al trainingset: \t{n_film_non_predetti}\n\n"
                f"Rating che non è stato possibile prevedere per mancanza di film visti nel vicinato: \t"
                f"{n_rating_non_predetti}\n\n"
                f"Rating predetti: \t{y_pred[~y_pred.isna()].index.unique().shape[0]}\n\n"
                f"MSE: \t{MSE}\nRMSE: \t{RMSE}\nMAE: \t{MAE}\n\n")


def weighted_popular_rating(ratings_train, movies, ratings_test, verbose=False, test_name="poluplar_weighted",
                            test_method="user"):
    # The variable "m" can be seen as regularizing parameter. Changing it determines how much weight is put onto the
    # movies with many ratings.
    # Even if there is a better ranking the RMSE decreased slightly. There is a trade-off between interpretability and
    # predictive power.
    # Number of minimum votes to be considered
    m = 60
    n = 10  # number of movies to suggest
    df_p = ratings_train.pivot_table(index='userId', columns='movieId', values='rating')
    ratings_count = df_p.count(axis=0).sort_values(ascending=False).rename('Rating-Count').to_frame()
    # Mean rating for all movies
    C = df_p.stack().mean()

    # Mean rating for all movies separatly
    R = df_p.mean(axis=0).values

    # Rating count for all movies separatly
    v = df_p.count().values

    # Weighted formula to compute the weighted rating
    # Settando un numero alto di voti ti assicuri che i film che i film che ne hanno pochi abbiano una media che pesa
    # poco, se v è piccolo m domina e manda molto vicino a 0 il peso di R, se v è grosso domina e manda a 0 il peso di C
    # R è il rating medio di quello specifico film, C è il rating medio di tutti i film messi insieme.
    weighted_score = (v / (v + m) * R) + (m / (v + m) * C)
    # Sort ids to ranking
    weighted_ranking = np.argsort(weighted_score)[::-1]
    # Sort scores to ranking
    weighted_score = np.sort(weighted_score)[::-1]
    # Get movie ids
    weighted_movie_ids = df_p.columns[weighted_ranking]

    # Join labels and predictions
    df_prediction = \
    ratings_test.set_index('movieId').join(pd.DataFrame(weighted_score, index=weighted_movie_ids, columns=['Prediction']))[
        ['rating', 'Prediction']]
    y_true = df_prediction['rating']
    y_pred = df_prediction['Prediction']

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

    # Create DataFrame for plotting
    df_plot = pd.DataFrame(weighted_score[:n], columns=['Rating'])
    df_plot.index = weighted_movie_ids[:10]
    ranking_weighted_rating = df_plot.join(ratings_count).join(movies.set_index('movieId'))
    del df_plot

    if verbose:
        titles_rat = ranking_weighted_rating['title'].astype(str) + ': ' + ranking_weighted_rating['Rating-Count'].astype(str) \
                         + ' Ratings'
        plt.figure(figsize=(10, 6), dpi=100)
        plt.xlim([3.7, 4.7])  # , ylim=(ymin, ymax))
        plt.grid(axis="x", zorder=0)
        plt.barh(list(range(1, n + 1)), ranking_weighted_rating['Rating'], color='#db0000', zorder=3)
        for i, v in enumerate(titles_rat):
            plt.text(ranking_weighted_rating['Rating'].iloc[i] + 0.05, i + 1, v)

        plt.title('Ranking Of Top {} Mean-Movie-Ratings: {:.4f} RMSE'.format(n, RMSE))
        plt.xlabel('Mean-Rating')
        plt.ylabel('Movie')
        plt.show()

    data = datetime.datetime.now()
    with open(f"results_run_{test_name}_{data.strftime('%Y_%m_%d_%H_%M_%S')}.txt", "w") as f:
        f.write(f"Metodo di test: {test_method}\n"
                f"film nuovi rispetto al trainingset: \t{n_film_non_predetti}\n\n"
                f"Rating che non è stato possibile prevedere per mancanza di film visti nel vicinato: \t"
                f"{n_rating_non_predetti}\n\n"
                f"Rating predetti: \t{y_pred[~y_pred.isna()].index.unique().shape[0]}\n\n"
                f"MSE: \t{MSE}\nRMSE: \t{RMSE}\nMAE: \t{MAE}\n\n")
