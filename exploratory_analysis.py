import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# To create interactive plots
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.io as pio

from utilities import *


# Content-based recommender is the system to rely on the similarity of items when it recommends items to users.
# For example, when a user likes a movie, the system finds and recommends movies which have more similar features to the
# movie the user likes.

# User: 610 movie: 9724

# ###################################### Movies analysis ###################################### #
def movieAnalysis(movies, verbose):
    pio.renderers.default = "browser"
    # svg_renderer = pio.renderers["svg"]
    # svg_renderer.engine = 'kaleido'
    plt.switch_backend("tkAgg")

    if "genres" in movies:
        # The column genres is the only feature used for this recommendation engine. Since the movies which do not have info
        # about genres are unnecessary in this practice, I will drop those movies in the data
        r, c = movies[movies['genres'] == '(no genres listed)'].shape
        print('The number of movies which do not have info about genres:', r)
        # remove the movies without genre information and reset the index because genres is the only feature used in this
        # engine, if they do not have genres they are useless
        movies = movies[~(movies['genres'] == '(no genres listed)')].reset_index(drop=True)

        # remove '|' in the genres column
        movies['genres'] = movies['genres'].str.replace('|', ' ')
        # change 'Sci-Fi' to 'SciFi' and 'Film-Noir' to 'Noir' otherwise there will be a genre sci, one fi, one film
        movies['genres'] = movies['genres'].str.replace('Sci-Fi', 'SciFi')
        movies['genres'] = movies['genres'].str.replace('Film-Noir', 'Noir')
        # count the number of occurences for each genre in the data set
        counts = dict()
        for i in movies.index:
            for g in movies.loc[i, 'genres'].split(' '):
                if g not in counts:
                    counts[g] = 1
                else:
                    counts[g] = counts[g] + 1
        # create a bar chart
        # plt.rcParams["axes.grid.axis"] = "x"
        # plt.rcParams["axes.grid"] = True
        if verbose:
            plt.figure(figsize=(10, 6), dpi=100)
            plt.grid(axis="y", zorder=0)
            plt.bar(list(counts.keys()), counts.values(), color='#db0000', zorder=3)
            plt.xticks(rotation=45)
            plt.title(f"Movies Grouped By Genre")
            plt.xlabel('Genres')
            plt.ylabel('Counts')
            plt.show()

    # Get data
    data = movies['year'].value_counts().sort_index()
    if verbose:
        plt.rcParams["axes.grid.axis"] = "both"
        plt.plot(list(data.index), list(data.values), color='#db0000')
        plt.title(f"{movies.shape[0]} Movies Grouped By Year Of Release")
        plt.xlabel("Release Year")
        plt.ylabel("Movies")
        plt.grid()
        plt.show()

    # # Create trace
    # trace = go.Scatter(x=data.index,
    #                    y=data.values,
    #                    marker=dict(color='#db0000'))
    # # Create layout
    # layout = dict(title='{} Movies Grouped By Year Of Release'.format(movies.shape[0]),
    #               xaxis=dict(title='Release Year'),
    #               yaxis=dict(title='Movies'))
    #
    # # Create plot
    # fig = go.Figure(data=[trace], layout=layout)
    # # iplot(fig)
    # fig.show()
    return movies


# ###################################### Rating analysis ###################################### #
def ratingAnalysis(ratings, verbose):
    pio.renderers.default = "browser"
    # svg_renderer = pio.renderers["svg"]
    # svg_renderer.engine = 'kaleido'
    plt.switch_backend("tkAgg")

    # Distribuzione dei voti (stelline per film)
    data = ratings['rating'].value_counts().sort_index()  #ascending=False)
    if verbose:
        plt.figure(figsize=(9, 6), dpi=100)
        plt.grid(axis="y", zorder=0)
        plt.bar(list(map(str, data.index.to_list())), data.values, color='#db0000', zorder=3)  # data.index
        plt.title(f"Distribution Of {ratings.shape[0]} Ratings")
        plt.xlabel('Rating')
        plt.ylabel('Counts')
        for i, v in enumerate(data):
            plt.text(i - .3, v + 300, '{:.1f} %'.format(v * 100 / len(ratings["movieId"])))
        plt.show()

        data = ratings['rating'].astype(int).value_counts().sort_index()
        plt.grid(axis="y", zorder=0)
        plt.bar(data.index, data.values, color='#db0000', zorder=3)  # data.index
        plt.title(f"Distribution Of {ratings.shape[0]} Ratings")
        plt.xlabel('Rating')
        plt.ylabel('Counts')
        for i, v in enumerate(data):
            plt.text(i - .2, v + 300, '{:.1f} %'.format(v * 100 / len(ratings["movieId"])))
        plt.show()
    # The distribution is probably biased, since only people liking the movies proceed to be customers and others
    # presumably will leave the platform.

    # Grafico dinamico pyplot
    # # Create trace
    # trace = go.Bar(x=data.index,
    #                text=['{:.1f} %'.format(val) for val in (data.values / ratings.shape[0] * 100)],
    #                textposition='auto',
    #                textfont=dict(color='#000000'),
    #                y=data.values,
    #                marker=dict(color='#db0000'))
    # # Create layout
    # layout = dict(title='Distribution Of {} Netflix-Ratings'.format(ratings.shape[0]),
    #               xaxis=dict(title='Rating'),
    #               yaxis=dict(title='Count'))
    # # Create plot
    # fig = go.Figure(data=[trace], layout=layout)
    # iplot(fig)

    # Distribuzione temporale dei voti (quando sono stati dati)
    data = ratings['date'].value_counts()
    data.sort_index(inplace=True)
    if verbose:
        plt.plot(list(data.index), list(data.values), color='#db0000')
        plt.title(f"{ratings.shape[0]} Movie-Ratings Grouped By Day")
        plt.xlabel("Date")
        plt.ylabel("Ratings")
        plt.grid()
        plt.show()
    # Il dataset non è stato campionato in modo uniforme essendo il 100k

    # Distribuzione dei voti per utente
    # data = ratings.groupby('movieId')['rating'].count()  #.clip(upper=9999)
    bins = [x for x in range(0, 60, 5)]
    bins.append(400)
    groups = pd.cut(ratings.groupby('movieId')['rating'].count(), bins).value_counts()
    groups.index = groups.index.map(lambda row: str(row)[str(row).find(" ")+1:-1])
    groups = groups.sort_index()
    groups.rename(index={groups.index[-1]: f"{groups.index[-2]}+"}, inplace=True)
    # Potevi ottenere lo stesso risultato con value_counts
    # ciao = ratings["movieId"].value_counts().value_counts(bins=5)
    if verbose:
        plt.grid(axis="y", zorder=0)
        plt.bar(groups.index, groups.values, color='#db0000', zorder=3)
        plt.title(f"Distribution Of Ratings Per Movie")
        plt.xlabel('Rating Per Movie')
        plt.ylabel('Counts')
        plt.show()

    # Create trace
    # 928 film hanno ricevuto almeno 100-200 voti
    # Create trace
    # trace = go.Histogram(x=data.values,
    #                      name='Ratings',
    #                      xbins=dict(start=0,
    #                                 end=10000,
    #                                 size=100),
    #                      marker=dict(color='#db0000'))
    # # Create layout
    # layout = go.Layout(title='Distribution Of Ratings Per Movie (Clipped at 9999)',
    #                    xaxis=dict(title='Ratings Per Movie'),
    #                    yaxis=dict(title='Count'),
    #                    bargap=0.2)
    #
    # # Create plot
    # fig = go.Figure(data=[trace], layout=layout)
    # iplot(fig)

    # data = df.groupby('User')['Rating'].count()
    bins = [x for x in range(20, 200, 20)]       # TUTTI HANNO ALMENO 20 RATINGS!!!!
    bins.append(400)
    groups = pd.cut(ratings.groupby('userId')['rating'].count(), bins).value_counts()
    groups.index = groups.index.map(lambda row: str(row)[str(row).find(" ") + 1:-1])
    groups = groups.sort_index()
    groups.rename(index={groups.index[-1]: f"{groups.index[-2]}+"}, inplace=True)
    # Potevi ottenere lo stesso risultato con value_counts
    # ciao = ratings["movieId"].value_counts().value_counts(bins=5)
    if verbose:
        plt.grid(axis="y", zorder=0)
        plt.bar(groups.index, groups.values, color='#db0000', zorder=3)
        plt.title(f"Distribution Of Ratings Per User")
        plt.xlabel('Rating Per User')
        plt.ylabel('Counts')
        plt.show()


if __name__ == "__main__":
    pio.renderers.default = "browser"
    # svg_renderer = pio.renderers["svg"]
    # svg_renderer.engine = 'kaleido'
    plt.switch_backend("tkAgg")

    root = "data\\MovieLensShort"
    file_rating = "ratings.csv"
    file_movie = "movies.csv"
    df_ratings = openSet(root, file_rating)
    df_movies = openSet(root, file_movie)

    movieAnalysis(df_movies)
    ratingAnalysis(df_ratings)
