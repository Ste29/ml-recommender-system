from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz


# create a function to find the closest title
def matching_score(a, b):
    return fuzz.ratio(a, b)  # calculates the Levenshtein Distance


# a function to convert index to title_year
def get_title_year_from_index(index, movies):
    return movies[movies.index == index]['title_year'].values[0]


# a function to convert index to title
def get_title_from_index(index, movies):
    return movies[movies.index == index]['title'].values[0]


# a function to convert title to index
def get_index_from_title(title, movies):
    return movies[movies.title == title].index.values[0]


# a function to return the most similar title to the words a user type
def find_closest_title(title, movies):
    # se combaciano perfettamente la distanza è 100, se no è minore
    leven_scores = list(enumerate(movies['title'].apply(matching_score, b=title)))
    sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
    closest_title = get_title_from_index(sorted_leven_scores[0][0], movies)
    distance_score = sorted_leven_scores[0][1]
    return closest_title, distance_score


def contents_based_recommender(movie_user_likes, sim_matrix, movies, how_many):
    # primo step: cerco se l'utente ha scritto male il titolo
    closest_title, distance_score = find_closest_title(movie_user_likes, movies)
    film_simili = []

    if distance_score == 100:  # When a user does not make misspellings
        pass
    else:  # When a user makes misspellings
        print('Did you mean ' + '\033[1m' + str(closest_title) + '\033[0m' + '?', '\n')

    movie_index = get_index_from_title(closest_title, movies)
    movie_list = list(enumerate(sim_matrix[int(movie_index)]))
    # remove the typed movie itself
    similar_movies = list(
        filter(lambda x: x[0] != int(movie_index), sorted(movie_list, key=lambda x: x[1], reverse=True)))

    print('Here\'s the list of movies similar to ' + '\033[1m' + str(closest_title) + '\033[0m' + '.\n')
    for i, s in similar_movies[:how_many]:
        print(get_title_year_from_index(i, movies))
        film_simili.append((i, s))
    return film_simili


# Content-based
def tfidf_content_based(movies, title, num_recommendation):
    # create an object for TfidfVectorizer
    tfidf_vector = TfidfVectorizer(stop_words='english')
    # apply the object to the genres column, 1 row per movie, 1 column per genre
    tfidf_matrix = tfidf_vector.fit_transform(movies['genres'])
    # in output restituisce una matrice delle similarità, sulla diagonale è max perché ovviamente la similarità di un
    # un elemento con se stesso è 100%, è la proiezione di un vettore sull'altro
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    film_simili = contents_based_recommender(title, sim_matrix, movies, num_recommendation)
    return film_simili
