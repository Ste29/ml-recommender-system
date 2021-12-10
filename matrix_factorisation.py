import numpy as np
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.sparse import vstack
from utilities import *


def matrix_factorisation(ratings_train, ratings_test, ratings, movies):
    # Create user- & movie-id mapping
    user_id_mapping = {id: i for i, id in enumerate(ratings['userId'].unique())}
    movie_id_mapping = {id: i for i, id in enumerate(ratings['movieId'].unique())}
    movie_id_mapping = {id: i for i, id in zip(movies.index, movies["movieId"])}
    # Create correctly mapped train- & testset
    train_user_data = ratings_train['userId'].map(user_id_mapping)
    train_movie_data = ratings_train['movieId'].map(movie_id_mapping)

    test_user_data = ratings_test['userId'].map(user_id_mapping)
    test_movie_data = ratings_test['movieId'].map(movie_id_mapping)

    # Get input variable-sizes
    users = len(user_id_mapping)
    movies = len(movie_id_mapping)
    embedding_size = 10

    ##### Create model
    # Set input layers
    user_id_input = Input(shape=[1], name='user')
    movie_id_input = Input(shape=[1], name='movie')

    # Create embedding layers for users and movies
    user_embedding = Embedding(output_dim=embedding_size,
                               input_dim=users,
                               input_length=1,
                               name='user_embedding')(user_id_input)
    movie_embedding = Embedding(output_dim=embedding_size,
                                input_dim=movies,
                                input_length=1,
                                name='item_embedding')(movie_id_input)

    # Reshape the embedding layers
    user_vector = Reshape([embedding_size])(user_embedding)
    movie_vector = Reshape([embedding_size])(movie_embedding)

    # Compute dot-product of reshaped embedding layers as prediction
    y = Dot(1, normalize=False)([user_vector, movie_vector])

    # Setup model
    model = Model(inputs=[user_id_input, movie_id_input], outputs=y)
    model.compile(loss='mse', optimizer='adam')

    # Fit model
    model.fit([train_user_data, train_movie_data],
              ratings_train['rating'],
              batch_size=128,
              epochs=25,
              validation_split=0.1,
              shuffle=True)
    # il tizio usa una sola epoca perché lui ha una marea di batch, tu ne hai molti meno --> tante epoche
    # Qui miriamo ad imparare il corretto embedding perché non lo conosciamo a priori, ma sappiamo che l'embed della
    # matrice 1 per la 2 è il nostro output.
    # Nota che tu hai già una matrice dei film con il one-hot encoding dei film, potresti usare quella per il dot
    # product # todo: testarla!
    # Test model
    y_pred = model.predict([test_user_data, test_movie_data])
    y_true = ratings_test['rating'].values
    y_new = y_pred.copy()
    y_new[y_new > 5] = 5
    y_new[y_new < .5] = .5

    #  Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_pred=y_new, y_true=y_true))
    print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))


def DL_factorization(ratings, movies, ratings_train, ratings_test):
    user_id_mapping = {id: i for i, id in enumerate(ratings['userId'].unique())}
    movie_id_mapping = {id: i for i, id in enumerate(ratings['movieId'].unique())}
    movie_id_mapping = {id: i for i, id in zip(movies.index, movies["movieId"])}
    train_user_data = ratings_train['userId'].map(user_id_mapping)
    train_movie_data = ratings_train['movieId'].map(movie_id_mapping)
    test_user_data = ratings_test['userId'].map(user_id_mapping)
    test_movie_data = ratings_test['movieId'].map(movie_id_mapping)
    users = len(user_id_mapping)
    movies = len(movie_id_mapping)

    # Setup variables
    user_embedding_size = 20
    movie_embedding_size = 10

    ##### Create model
    # Set input layers
    user_id_input = Input(shape=[1], name='user')
    movie_id_input = Input(shape=[1], name='movie')

    # Create embedding layers for users and movies
    user_embedding = Embedding(output_dim=user_embedding_size,
                               input_dim=users,
                               input_length=1,
                               name='user_embedding')(user_id_input)
    movie_embedding = Embedding(output_dim=movie_embedding_size,
                                input_dim=movies,
                                input_length=1,
                                name='item_embedding')(movie_id_input)

    # Reshape the embedding layers
    user_vector = Reshape([user_embedding_size])(user_embedding)
    movie_vector = Reshape([movie_embedding_size])(movie_embedding)

    # # Concatenate the reshaped embedding layers
    # concat = Concatenate()([user_vector, movie_vector])
    #
    # # Combine with dense layers
    # dense = Dense(256)(concat)
    # dense = Dense(64)(dense)
    # y = Dense(1)(dense)

    dense_user = Dense(256)(user_vector)
    dense_movie = Dense(256)(movie_vector)
    y = Dot(1, normalize=False)([dense_user, dense_movie])

    # Setup model
    model = Model(inputs=[user_id_input, movie_id_input], outputs=y)
    model.compile(loss='mse', optimizer='adam')

    # Fit model
    model.fit([train_user_data, train_movie_data],
              ratings_train['rating'],
              batch_size=128,
              epochs=5,
              validation_split=0.1,
              shuffle=True)

    # Test model
    y_pred = model.predict([test_user_data, test_movie_data])
    y_true = ratings_test['rating'].values
    y_new = y_pred.copy()
    y_new[y_new > 5] = 5
    y_new[y_new < .5] = .5

    #  Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_pred=y_new, y_true=y_true))
    print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse))


def hybrid_system(ratings, movies):
    movie_metadata = pd.read_csv(
        r'C:\Users\svillata\projects\deloitte\recommendation system\data\TheMovieDataset\movies_metadata.csv',
        low_memory=False)[
        ['original_title', 'overview', 'release_date']].dropna()
    movie_metadata["original_title"] = movie_metadata.apply(
        lambda row: row["original_title"].lower()+" "+f"({row['release_date'][0:4]})", axis=1)
    movie_metadata = movie_metadata.set_index('original_title')
    movie_metadata.drop(["release_date"], axis=1, inplace=True)
    # movie_metadata = movie_metadata[movie_metadata['vote_count'] > 10].drop('vote_count', axis=1)


    # Create user- & movie-id mapping
    user_id_mapping = {id: i for i, id in enumerate(ratings['userId'].unique())}
    movie_id_mapping = {id: i for i, id in enumerate(ratings['movieId'].unique())}
    movie_id_mapping = {id: i for i, id in zip(movies.index, movies["movieId"])}

    # Use mapping to get better ids
    ratings['userId'] = ratings['userId'].map(user_id_mapping)
    ratings['movieId'] = ratings['movieId'].map(movie_id_mapping)

    ##### Combine both datasets to get movies with metadata
    # Preprocess metadata
    tmp_metadata = movie_metadata.copy()
    tmp_metadata.index = tmp_metadata.index.str.lower()

    # Preprocess titles
    tmp_titles = movies.copy()
    tmp_titles.drop(["title", "year"], axis=1, inplace=True)
    tmp_titles['title_year_tmp'] = tmp_titles['title_year'].map(movie_title_clean)
    tmp_titles['title'] = tmp_titles['title_year_tmp'].apply(lambda x: x[0])
    tmp_titles['year'] = tmp_titles['title_year_tmp'].apply(lambda x: x[1])
    tmp_titles["title_year"] = tmp_titles['title_year_tmp'].apply(lambda x: x[0]+" "+f"({x[1]})")
    tmp_titles = tmp_titles.reset_index().set_index('title_year')
    tmp_titles.index = tmp_titles.index.str.lower()
    tmp_titles.drop(["title_year_tmp"], axis=1, inplace=True)

    # Combine titles and metadata
    df_id_descriptions = tmp_titles.join(tmp_metadata).dropna()
    df_id_descriptions['movieId'] = df_id_descriptions['movieId'].map(movie_id_mapping)
    df_id_descriptions = df_id_descriptions.set_index('movieId')
    df_id_descriptions['overview'] = df_id_descriptions['overview'].str.lower()
    del tmp_metadata, tmp_titles

    # Filter all ratings with metadata
    df_hybrid = ratings.drop('date', axis=1).set_index('movieId').join(df_id_descriptions).dropna()\
        .drop(['overview', 'index'], axis=1).reset_index().rename(
        {'index': 'Movie'}, axis=1)

    # Split train- & testset
    n = 70000
    df_hybrid = df_hybrid.sample(frac=1).reset_index(drop=True)
    df_hybrid_train = df_hybrid[:n]
    df_hybrid_test = df_hybrid[n:]

    # Create tf-idf matrix for text comparison
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_hybrid = tfidf.fit_transform(df_id_descriptions['overview'])

    # Get mapping from movie-ids to indices in tfidf-matrix
    mapping = {id: i for i, id in enumerate(df_id_descriptions.index)}

    train_tfidf = []
    # Iterate over all movie-ids and save the tfidf-vector per ogni film nel training otteniamo il suo tfidf
    for id in df_hybrid_train['movieId'].values:
        index = mapping[id]
        train_tfidf.append(tfidf_hybrid[index])

    test_tfidf = []
    # Iterate over all movie-ids and save the tfidf-vector
    for id in df_hybrid_test['movieId'].values:
        index = mapping[id]
        test_tfidf.append(tfidf_hybrid[index])

    # Stack the sparse matrices
    train_tfidf = vstack(train_tfidf)
    test_tfidf = vstack(test_tfidf)

    ##### Setup the network
    # Network variables
    user_embed = 10
    movie_embed = 10

    a = nn_batch_generator(df_hybrid_train['userId'], df_hybrid_train['movieId'], train_tfidf,
                                           df_hybrid_train['rating'], 128)
    tmp = a.__next__()
    # Create two input layers
    user_id_input = Input(shape=[1], name='user')
    movie_id_input = Input(shape=[1], name='movie')
    tfidf_input = Input(shape=[tmp[0][2].shape[1]], name='tfidf', sparse=True)

    # Create separate embeddings for users and movies
    user_embedding = Embedding(output_dim=user_embed,
                               input_dim=len(user_id_mapping),
                               input_length=1,
                               name='user_embedding')(user_id_input)
    movie_embedding = Embedding(output_dim=movie_embed,
                                input_dim=len(movie_id_mapping),
                                input_length=1,
                                name='movie_embedding')(movie_id_input)

    # Dimensionality reduction with Dense layers
    tfidf_vectors = Dense(128, activation='relu')(tfidf_input)
    tfidf_vectors = Dense(32, activation='relu')(tfidf_vectors)

    # Reshape both embedding layers
    user_vectors = Reshape([user_embed])(user_embedding)
    movie_vectors = Reshape([movie_embed])(movie_embedding)

    # Concatenate all layers into one vector
    both = Concatenate()([user_vectors, movie_vectors, tfidf_vectors])

    # Add dense layers for combinations and scalar output
    dense = Dense(256, activation='relu')(both)
    dense = Dropout(0.2)(dense)
    output = Dense(1)(dense)

    # Create and compile model
    model = Model(inputs=[user_id_input, movie_id_input, tfidf_input], outputs=output)
    model.compile(loss='mse', optimizer='adam')

    # Train and test the network
    # a = nn_batch_generator(df_hybrid_train['userId'], df_hybrid_train['movieId'], train_tfidf,
    #                                        df_hybrid_train['rating'], 128)
    # a.__next__()
    model.fit(nn_batch_generator(df_hybrid_train['userId'], df_hybrid_train['movieId'], train_tfidf,
                                 df_hybrid_train['rating'], 128),
              batch_size=128,
              epochs=10,
              steps_per_epoch=df_hybrid_train['userId'].shape[0]/128,
              # validation_split=0.1,
              shuffle=True)

        # [df_hybrid_train['userId'], df_hybrid_train['movieId'], train_tfidf],
        #       df_hybrid_train['rating'],
        #       batch_size=128,
        #       epochs=15,
        #       validation_split=0.1,
        #       shuffle=True)
    # generator = batch_generator(X_train_sparse, Y_train, batch_size),
    # nb_epoch = nb_epoch,
    # samples_per_epoch = X_train_sparse.shape[0]

    y_pred = model.predict([df_hybrid_test['userId'], df_hybrid_test['movieId'], test_tfidf.toarray()])
    y_true = df_hybrid_test['rating'].values
    y_new = y_pred.copy()
    y_new[y_new > 5] = 5
    y_new[y_new < .5] = .5

    rmse = np.sqrt(mean_squared_error(y_pred=y_new, y_true=y_true))
    print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))


def nn_batch_generator(X_data_user, X_data_movie, X_data_tfidf, y_data, batch_size):
    samples_per_epoch = X_data_user.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter = 0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch_user = X_data_user[index_batch]
        X_batch_movie = X_data_movie[index_batch]
        X_batch_tfidf = X_data_tfidf[index_batch, :].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield ([np.array(X_batch_user), np.array(X_batch_movie), np.array(X_batch_tfidf)], np.array(y_batch))
        if counter > number_of_batches:
            counter = 0


def tfidf_factorization(ratings, movies, ratings_train, ratings_test):
    user_id_mapping = {id: i for i, id in enumerate(ratings['userId'].unique())}
    movie_id_mapping = {id: i for i, id in enumerate(ratings['movieId'].unique())}
    movie_id_mapping = {id: i for i, id in zip(movies.index, movies["movieId"])}

    tfidf_vector = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vector.fit_transform(movies["genres"])
    tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray())
    movies = movies.join(tfidf_matrix)
    ratings_train = ratings_train.join(movies.set_index("movieId"), on="movieId")
    ratings_test = ratings_test.join(movies.set_index("movieId"), on="movieId")

    train_user_data = ratings_train['userId'].map(user_id_mapping)
    train_movie_data = ratings_train['movieId'].map(movie_id_mapping)
    test_user_data = ratings_test['userId'].map(user_id_mapping)
    test_movie_data = ratings_test['movieId'].map(movie_id_mapping)
    users = len(user_id_mapping)
    movies = len(movie_id_mapping)

    user_embed = 20
    movie_embed = 10
    # Setup variables
    user_id_input = Input(shape=[1], name='user')
    movie_id_input = Input(shape=[1], name='movie')
    tfidf_input = Input(shape=[19], name='tfidf', sparse=True)

    # Create separate embeddings for users and movies
    user_embedding = Embedding(output_dim=user_embed,
                               input_dim=len(user_id_mapping),
                               input_length=1,
                               name='user_embedding')(user_id_input)
    movie_embedding = Embedding(output_dim=movie_embed,
                                input_dim=len(movie_id_mapping),
                                input_length=1,
                                name='movie_embedding')(movie_id_input)

    # # Dimensionality reduction with Dense layers
    # tfidf_vectors = Dense(128, activation='relu')(tfidf_input)
    # tfidf_vectors = Dense(32, activation='relu')(tfidf_vectors)
    #
    # # Reshape both embedding layers
    # user_vectors = Reshape([user_embed])(user_embedding)
    # movie_vectors = Reshape([movie_embed])(movie_embedding)
    #
    # # Concatenate all layers into one vector
    # both = Concatenate()([user_vectors, movie_vectors, tfidf_vectors])
    #
    # # Add dense layers for combinations and scalar output
    # dense = Dense(512, activation='relu')(both)
    # dense = Dropout(0.2)(dense)
    # output = Dense(1)(dense)
    # Reshape both embedding layers

    user_vector = Reshape([user_embed])(user_embedding)
    movie_vector = Reshape([movie_embed])(movie_embedding)
    # tfidf_vector = Reshape([20])(movie_embedding) # non usare
    tfidf_vector = Dense(64, activation='relu')(tfidf_input)
    # both = Dense(64, activation='relu')(tfidf_input)
    both = Concatenate()([movie_vector, tfidf_vector])
    dense_both = Dense(256, activation='relu')(both)
    dense_user = Dense(256, activation='relu')(user_vector)
    output = Dot(1, normalize=False)([dense_user, dense_both])

    # Create and compile model
    model = Model(inputs=[user_id_input, movie_id_input, tfidf_input], outputs=output)
    model.compile(loss='mse', optimizer='adam')

    # Train and test the network
    # a = nn_batch_generator(df_hybrid_train['userId'], df_hybrid_train['movieId'], train_tfidf,
    #                                        df_hybrid_train['rating'], 128)
    # a.__next__()
    model.fit([train_user_data, train_movie_data, ratings_train.iloc[:, 10:]], ratings_train['rating'],
              batch_size=128,
              epochs=5,
              # steps_per_epoch=ratings_train['userId'].shape[0] / 128,
              validation_split=0.1,
              shuffle=True)


    # Test model
    y_pred = model.predict([test_user_data, test_movie_data, ratings_test.iloc[:, 10:]])
    y_true = ratings_test['rating'].values
    y_new = y_pred.copy()
    y_new[y_new > 5] = 5
    y_new[y_new < .5] = .5

    #  Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_pred=y_new, y_true=y_true))
    print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse))