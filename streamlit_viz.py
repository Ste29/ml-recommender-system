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
# from plotly.offline import init_notebook_mode, plot, iplot


def similar_movies(movies, title, links, sims):
    st.balloons()

    # with col1:
    #     id = int(movies.loc[movies["title"] == title, "movieId"])
    #     st.image(links["imglink"][id])
    st.header("your movie")
    id = int(movies.loc[movies["title"] == title, "movieId"])
    try:
        # col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 4, 3])
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            st.write("")
            st.write("")
            st.write("")
            st.image(links[links["movieId"]==id]["imglink"].iloc[0])
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.subheader(title)
            st.write(f'Movie genre: {", ".join(movies.loc[movies["title"] == title, "genre"].iloc[0].split(" "))}')
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("From the similar movie it is computed the following word cloud")
        indici = [int(i) for i, j in sims]
        text = ','.join(movies.loc[indici, "tag_x"].values)

        # Generate a word cloud image
        wordcloud = WordCloud(width=1024, height=1024, background_color="#121212",).generate(text)

        # Display the generated image:
        fig = plt.figure(figsize=(3.8, 3.8), facecolor="#121212", edgecolor='white', layout='tight')
        plt.imshow(wordcloud, interpolation='bilinear', figure=fig)
        plt.axis("off")
        with col3:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            # st.pyplot(fig, use_container_width=True)
            st.image(buf)

    except KeyError:
        pass
    st.header("recommended")
    count = 1
    show_posters(links, sims, movies, True)


def show_posters(links, sims, movies, similarity=False):
    for i in [0, 3, 6]:
        # for i, j in sims:
        col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 2, 1, 2])
        try:
            with col1:
                id = int(movies.loc[movies["title"] == movies.loc[int(sims[i][0]), "title"].strip(), "movieId"])
                st.image(links[links["movieId"]==id]["imglink"].iloc[0])
            with col2:
                st.subheader(movies.loc[movies["title"] == movies.loc[int(sims[i][0]), "title"].strip(), "title"][int(sims[i][0])])
                listaGenere1 = movies.loc[movies["title"] == movies.loc[int(sims[i][0]), "title"].strip(),
                                          "genre"][int(sims[i][0])]
                # listaGenere1 = listaGenere1.split(" ")
                listaGenere1 = ", ".join(listaGenere1.split(" "))
                st.write("Movie Genre: {:s}".format(listaGenere1))
                if similarity:
                    st.write("similarity score: {:.2f}".format(sims[i][1]))
            with col3:
                id = int(movies.loc[movies["title"] == movies.loc[int(sims[i+1][0]), "title"].strip(), "movieId"])
                st.image(links[links["movieId"]==id]["imglink"].iloc[0])
            with col4:
                st.subheader(movies.loc[movies["title"] == movies.loc[int(sims[i+1][0]), "title"].strip(), "title"][int(sims[i+1][0])])
                listaGenere2 = movies.loc[movies["title"] == movies.loc[int(sims[i+1][0]), "title"].strip(),
                                          "genre"][int(sims[i+1][0])]
                listaGenere2 = ", ".join(listaGenere2.split(" "))
                st.write("Movie Genre: {:s}".format(listaGenere2))
                if similarity:
                    st.write("similarity score: {:.2f}".format(sims[i+1][1]))
            with col5:
                id = int(movies.loc[movies["title"] == movies.loc[int(sims[i+2][0]), "title"].strip(), "movieId"])
                st.image(links[links["movieId"] == id]["imglink"].iloc[0])
            with col6:
                st.subheader(movies.loc[movies["title"] == movies.loc[int(sims[i+2][0]), "title"].strip(), "title"][int(sims[i+2][0])])
                listaGenere3 = movies.loc[movies["title"] == movies.loc[int(sims[i+2][0]), "title"].strip(),
                                          "genre"][int(sims[i+2][0])]
                listaGenere3 = ", ".join(listaGenere3.split(" "))
                st.write("Movie Genre: {:s}".format(listaGenere3))
                if similarity:
                    st.write("similarity score: {:.2f}".format(sims[i+2][1]))
        except:
            pass
