from sklearn.neighbors import NearestNeighbors
import pandas as pd


def item_based_recommend_movies(user, num_recommended_movies):
    print('The list of the Movies {} Has Watched \n'.format(user))

    for m in df[df[user] > 0][user].index.tolist():
        print(m)

    print('\n')

    recommended_movies = []

    for m in df[df[user] == 0].index.tolist():
        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
        recommended_movies.append((m, predicted_rating))

    sorted_rm = sorted(recommended_movies, key=lambda x: x[1], reverse=True)

    print('The list of the Recommended Movies \n')
    rank = 1
    for recommended_movie in sorted_rm[:num_recommended_movies]:
        print('{}: {} - predicted rating:{}'.format(rank, recommended_movie[0], recommended_movie[1]))
        rank = rank + 1


def movie_recommender(user, number_neighbors, num_recommendation, ratings, movies):
    # Se ci sono dei Nan non funziona il knn, allora setto tutto a 0, tanto il minimo voto che si può assegnare è 0.5
    # sulle righe ci sono i film, sulle colonne gli utenti
    # ratings2 = pd.merge(ratings, movies, how='inner', on='movieId')
    # df = ratings2.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    df = ratings.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)

    knn = NearestNeighbors(metric='cosine', algorithm='brute')  # todo: studia come funziona
    knn.fit(df.values)  # ogni riga di indices è un film, sulla prima colonna il film più vicino (se stesso) poi il
    # secondo e terzo più vicini, distances sono le distanze
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)
    # Siccome l'id dei film non coincide con le righe di df hai 2 opzioni, o fai il merge così da mettere sulle righe i
    # titoli, oppure tieni gli id e poi fai il retrieve con
    # movies[movies["movieId"]==df2.index[418]] questo è il secondo film più vicino a Toy Story
    user_index = df.columns.tolist().index(user)

    for m, t in list(enumerate(df.index)):  # m: index del film, t: movieId
        if df.iloc[m, user_index] == 0:  # cerco i film che lo user desiderato non ha ancora guardato
            sim_movies = indices[m].tolist()  # mi trovo i film "simili"
            movie_distances = distances[m].tolist()

            if m in sim_movies:  # quando il film più vicino è se stesso lo rimuovo
                id_movie = sim_movies.index(m)
                sim_movies.remove(m)
                movie_distances.pop(id_movie)

            else:
                sim_movies = sim_movies[:number_neighbors - 1]
                movie_distances = movie_distances[:number_neighbors - 1]

            movie_similarity = [1 - x for x in movie_distances]  # similarità = 1-distanza
            movie_similarity_copy = movie_similarity.copy()
            nominator = 0

            for s in range(0, len(movie_similarity)):
                if df.iloc[sim_movies[s], user_index] == 0:
                    if len(movie_similarity_copy) == (number_neighbors - 1):
                        movie_similarity_copy.pop(s)

                    else:
                        movie_similarity_copy.pop(s - (len(movie_similarity) - len(movie_similarity_copy)))

                else:
                    nominator = nominator + movie_similarity[s] * df.iloc[sim_movies[s], user_index]

            if len(movie_similarity_copy) > 0:
                if sum(movie_similarity_copy) > 0:
                    predicted_r = nominator / sum(movie_similarity_copy)

                else:
                    predicted_r = 0

            else:
                predicted_r = 0

            df1.iloc[m, user_index] = predicted_r
    item_based_recommend_movies(user, num_recommendation)
