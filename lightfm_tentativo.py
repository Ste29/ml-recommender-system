from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score


def lightfm_tentativo(ratings, ratings_train, ratings_test, movies):
    # Create user- & movie-id mapping
    user_id_mapping = {id: i for i, id in enumerate(ratings['userId'].unique())}
    movie_id_mapping = {id: i for i, id in enumerate(ratings['movieId'].unique())}
    movie_id_mapping = {id: i for i, id in zip(movies.index, movies["movieId"])}

    # Create correctly mapped train- & testset
    train_user_data = ratings_train['userId'].map(user_id_mapping)
    train_movie_data = ratings_train['movieId'].map(movie_id_mapping)

    test_user_data = ratings_test['userId'].map(user_id_mapping)
    test_movie_data = ratings_test['movieId'].map(movie_id_mapping)

    # Create sparse matrix from ratings
    shape = (len(user_id_mapping), len(movie_id_mapping))
    train_matrix = coo_matrix((ratings_train['rating'].values, (train_user_data.astype(int), train_movie_data.astype(int))),
                              shape=shape)
    test_matrix = coo_matrix((ratings_test['rating'].values, (test_user_data.astype(int), test_movie_data.astype(int))),
                             shape=shape)

    # Instantiate and train the model
    model = LightFM(loss='warp', no_components=20)
    model.fit(train_matrix, epochs=50, num_threads=2)

    # Evaluate the trained model
    k = 20
    print('Train precision at k={}:\t{:.4f}'.format(k, precision_at_k(model, train_matrix, k=k).mean()))
    print('Test precision at k={}:\t\t{:.4f}'.format(k, precision_at_k(model, test_matrix, k=k).mean()))