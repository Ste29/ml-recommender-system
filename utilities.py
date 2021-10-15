import pandas as pd
import os
from collections import deque
import numpy as np


def filterMovies(df, min_movie_ratings=10000, min_user_ratings=200):
    # Filter sparse movies
    filter_movies = (df['Movie'].value_counts() > min_movie_ratings)
    filter_movies = filter_movies[filter_movies].index.tolist()

    # Filter sparse users
    # min_user_ratings = 200
    filter_users = (df['User'].value_counts() > min_user_ratings)
    filter_users = filter_users[filter_users].index.tolist()

    # Actual filtering
    df_filtered = df[(df['Movie'].isin(filter_movies)) & (df['User'].isin(filter_users))]
    del filter_movies, filter_users, min_movie_ratings, min_user_ratings
    print('Shape User-Ratings unfiltered:\t{}'.format(df.shape))
    print('Shape User-Ratings filtered:\t{}'.format(df_filtered.shape))
    return df_filtered


def splitTrainTest(df, n=100000):
    # Shuffle DataFrame, n=testingsize
    df = df.drop('Date', axis=1).sample(frac=1).reset_index(drop=True)

    # Split train- & testset
    df_train = df[:-n]
    df_test = df[-n:]
    return df_train, df_test


def extract_title(title):
    year = title[len(title) - 5:len(title) - 1]
    # some movies do not have the info about year in the column title
    if year.isnumeric():
        title_no_year = title[:len(title) - 7]
        return title_no_year
    else:
        return title


# the function to extract years
def extract_year(title):
    year = title[len(title) - 5:len(title) - 1]
    # some movies do not have the info about year in the column title. So, we should take care of the case as well.
    if year.isnumeric():
        return int(year)
    else:
        return np.nan


def openSet(root, file):  # todo: finire la funzione per aprire i diversi set (netflix movies + TheMovieDataset)
    if file.endswith("csv"):
        df = pd.read_csv(os.path.join(root, file))
        if "rating" in file:
            df.timestamp = pd.to_datetime(df.timestamp, unit="s")
            df.rename(columns={"timestamp": "date"}, inplace=True)
            # df["rating"] = df["rating"].astype(int)
        elif "movie" in file:
            # change the column name from title to title_year
            df.rename(columns={'title': 'title_year'}, inplace=True)
            # remove leading and ending whitespaces in title_year
            df['title_year'] = df['title_year'].apply(lambda x: x.strip())
            # create the columns for title and year
            df['title'] = df['title_year'].apply(extract_title)
            df['year'] = df['title_year'].apply(extract_year)
            df['year'] = df["year"].astype(pd.Int32Dtype())
    else:  # netflix dataset
        # Load single data-file
        df = pd.read_csv(os.path.join(root, file), header=None,
                         names=['userId', 'rating', 'date'], usecols=[0, 1, 2])

        # Find empty rows to slice dataframe for each movie
        tmp_movies = df[df['rating'].isna()]['userId'].reset_index()
        movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

        # Shift the movie_indices by one to get start and endpoints of all movies
        shifted_movie_indices = deque(movie_indices)
        shifted_movie_indices.rotate(-1)

        # Gather all dataframes
        user_data = []

        # Iterate over all movies
        for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):

            # Check if it is the last movie in the file
            if df_id_1 < df_id_2:
                tmp_df = df.loc[df_id_1 + 1:df_id_2 - 1].copy()
            else:
                tmp_df = df.loc[df_id_1 + 1:].copy()

            # Create movie_id column
            tmp_df['movieId'] = movie_id

            # Append dataframe to list
            user_data.append(tmp_df)

        # Combine all dataframes
        df = pd.concat(user_data)
        del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id
        print('Shape User-Ratings:\t{}'.format(df.shape))
        df.date = pd.to_datetime(df.date)
        df["rating"] = df["rating"].astype(int)

    return df
