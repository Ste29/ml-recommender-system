import streamlit as st
import pandas as pd
import numpy as np
from Dataset import *
import pickle
from content_based import *
from wordcloud import WordCloud
from functions import *
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_viz import *

st.set_page_config(page_title='SteRec!',
                   page_icon='data/Img/Icon3.png',
                   layout="wide")

st.image("data/Img/sfondo5.jpg", use_column_width=True)
st.title('Movie Recommender Engine')

# ############################ INITILIAZING DATA ############################ #
# dataset = Dataset()
with open("data/Pickle/movies1.pkl", "rb") as f:
    movies1 = pickle.load(f)
with open("data/Pickle/movies2.pkl", "rb") as f:
    movies2 = pickle.load(f)
with open("data/Pickle/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("data/Pickle/mv_tags_vectors_umap.pkl", "rb") as f:
    mv_tags_vectors_umap = pickle.load(f)
with open("data/Pickle/medie_voto_film.pkl", "rb") as f:
    medie_voto_film = pickle.load(f)
with open("data/Pickle/weighted_score.pkl", "rb") as f:
    weighted_score = pickle.load(f)

movies = pd.concat([movies1,movies2])
xUmap, yUmap = mv_tags_vectors_umap.T[0], mv_tags_vectors_umap.T[1]
links = pd.read_csv(r"C:\Users\Stefano\Desktop\ml-recommender-system\data\MovieLensShort\links_new.csv")

medie_voto_film = medie_voto_film.reset_index().join(
    movies.set_index("movieId")["title"], on="movieId").sort_values(by="rating", ascending=False).dropna()
weighted_score = weighted_score.reset_index().join(
    movies.set_index("movieId")["title"], on="movieId").sort_values(by="rating", ascending=False).dropna()


# ################################ INPUT DATA ################################ #
label = "Write a movie you recently liked"
title = st.selectbox('', movies["title"].sort_values(ascending=False))

# ################################ SELECT ACTION ################################# #
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
menu_relations = st.radio(
    "",
    ("Similar Movies", "Movies Space Disposition", "Discover Movie Catalog", "MovieLens Favourite Movies"),
)

# ################################ PLOT DATA ################################# #
movies["genre"] = movies["document"].apply(lambda x: re.sub(".*?[\)\]\}] ", "", x))
movies[movies["title"].duplicated()] = movies[movies["title"].duplicated()].apply(fix_titles, axis=1)

df = pd.DataFrame({"x": xUmap, "y": yUmap, "title": movies["title"], "genre": movies["genre"],
                   "selected": "Catalog Movies", "size": .3, "color": "#a30810"})
mv_index = movies[movies["title"] == title].index.values[0]
sims = model.docvecs.most_similar(positive=[mv_index], topn=10)

if menu_relations == "Similar Movies":
    similar_movies(movies, title, links, sims)

elif menu_relations == "Discover Movie Catalog":
    fig1, fig2 = movieAnalysis(movies)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

elif menu_relations == "MovieLens Favourite Movies":
    st.header("Movie Ratings Chart")
    fig3 = scoring(movies, medie_voto_film, weighted_score)
    st.plotly_chart(fig3, use_container_width=True)
    st.header("Movie Posters")
    films = [(id,) for id in weighted_score["movieId"][0:9].to_list()]
    films = list(map(lambda x: (movies[movies["movieId"]==x[0]].index[0],), films))
    show_posters(links, films, movies, similarity=False)

elif menu_relations == "Movies Space Disposition":
    with st.spinner('Wait for it...'):
        df.loc[df["title"] == title, "selected"] = "Your Movie"
        df.loc[df["title"] == title, "size"] = 2.5
        df.loc[df["title"] == title, "color"] = "#188af5"

        for i, j in sims:
            df.loc[df["title"] == movies.loc[int(i), "title"].strip(), "selected"] = "Recommended To You"
            df.loc[df["title"] == movies.loc[int(i), "title"].strip(), "size"] = 2.5
            df.loc[df["title"] == movies.loc[int(i), "title"].strip(), "color"] = "#e07d04"

    fig = scatterUmapDistribution(df, color="selected", size="size")

    # st.write(fig)
    st.plotly_chart(fig, use_container_width=True)




