import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import plotly.express as px
import re
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools


def fix_titles(row):
    row["title"] = f"{row['title']} ({row['year']})"
    return row

def scatterUmapDistribution(df, size=None, color=None, title_color=''):
    colormap = {"Catalog Movies": "#188af5", "Your Movie": "#a30810", "Recommended To You": "#e07d04"}
    fig = px.scatter(df,
                     x="x", y="y", color=color, size=size,
                     width=1400, height=780,  # opacity=0.5,
                     # color_continuous_scale=["#F5C518", "#F91949"],
                     # color_discrete_sequence=px.colors.qualitative.Antique,
                     # color_discrete_sequence=["#a30810", "#188af5", "#e07d04"],
                     color_discrete_map=colormap,
                     template="plotly_dark",
                     hover_name="title",
                     hover_data={'genre': True,
                                 # 'sepal_length': ':.2f',  # customize hover for column of y attribute
                                 'x': False, "y": False, "selected": False, "size": False
                                 # 'suppl_2': (':.3f', np.random.random(len(df)))
                                 },
                     labels={"Available movies": "selected"},
                     title="Movie Vector (Doc2Vec Output Visualized in 2D with UMAP)"
                     )

    if color == None:
        fig.update_traces(marker=dict(color="#F5C518"))

    fig.update_layout(coloraxis_colorbar=dict(title=title_color,),
                      legend=dict(title='Legend:', font={'size': 15}),
                      title=dict(font={'size': 22}),  # , 'color': "#F5C518"
                      )

    fig.update_xaxes(
        title_text="",
        title_font={"size": 1},
        # color="black",
        # title_standoff=20,
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False
    )

    fig.update_yaxes(
        title_text="",
        title_font={"size": 1},
        # color="black",
        # title_standoff=20,
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False
    )
    gris = '#999'
    fig.add_shape(  # horizontal axe
        type="line", line_color=gris, line_width=1, opacity=.5,
        x0=df["x"].min()-df["x"].max()/12, x1=df["x"].max()+df["x"].max()/12, xref="x",
        y0=df["x"].max()/2, y1=df["x"].max()/2, yref="y"
    )
    # fig.add_annotation(  # texto línea horizontal
    #     text="Suspenso Metascore", x=1.3, y=48, showarrow=False, font={'color': gris, 'size': 14}
    # )

    fig.add_shape(  # vertical axe
        type="line", line_color=gris, line_width=1, opacity=.5,
        x0=df["y"].max()/2, x1=df["y"].max()/2, xref="x",
        y0=df["y"].min()-df["y"].max()/12, y1=df["y"].max()+df["y"].max()/12, yref="y"
    )
    #
    # fig.add_annotation(  # texto línea vertical
    #     text="Suspenso Rating IMDb", x=3.5, y=-2, showarrow=False, font={'color': gris, 'size': 14}
    # )
    return fig


def movieAnalysis(movies):
    #todo: finire la funzione per mostrare la distribuzione dei generi e mostra anche i top popular according to ML
    counts = dict()
    for i in movies.index:
        for g in movies.loc[i, 'genres'].split(' '):
            if g not in counts:
                counts[g] = 1
            else:
                counts[g] = counts[g] + 1

    trace = go.Bar(x=list(counts.keys()), textposition='auto', textfont=dict(color='#000000'),
                   y=list(counts.values()), marker=dict(color='#08009c')) #'#db0000'))
    # Create layout
    layout = dict(title='Genres Distribution',
                  font={"size": 15},
                  xaxis=dict(title='Genre'),
                  yaxis=dict(title='# of Movies'))
    # Create plot
    fig1 = go.Figure(data=[trace], layout=layout)
    gris = '#707070'  # '#999'
    # fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor=gris)
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor=gris)
    # fig1.add_shape(type="line", line_color=gris, line_width=1, opacity=.5)
    # iplot(fig)



    data = movies['year'].value_counts().sort_index()
    trace = go.Scatter(x=data.index,
                       y=data.values,
                       marker=dict(color='#001eff'))
    # Create layout
    layout = dict(title='{} Movies Grouped By Year Of Release'.format(movies.shape[0]),
                  font={"size": 15},
                  xaxis=dict(title='Release Year'),
                  yaxis=dict(title='Movies'))

    # Create plot
    fig2 = go.Figure(data=[trace], layout=layout)
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor=gris)
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor=gris)
    # iplot(fig)
    # fig.show()
    return fig1, fig2


def scoring(movies, medie_voto_film, weighted_score):
    trace = go.Bar(x=weighted_score["rating"][0:10],
                   text=weighted_score["title"][0:10].astype(str),
                   orientation="h", textposition='outside', textfont=dict(color='#ffffff'),
                   y=list(range(1, 10+1)), marker=dict(color='#08009c')) #'#db0000'))
    # Create layout
    layout = dict(title='Genres Distribution',
                  font={"size": 15},
                  xaxis=dict(title='Weighted Rating',
                               range=(4.15, 4.6)),
                  yaxis=dict(title='# of Movies'))
    # Create plot
    fig1 = go.Figure(data=[trace], layout=layout)
    gris = '#707070'
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor=gris)
    fig1.update_yaxes(showgrid=False, gridwidth=1, gridcolor=gris)
    return fig1


if __name__ == "__main__":
    import pickle
    from plotly.offline import init_notebook_mode, plot, iplot

    with open("data/Pickle/movies.pkl", "rb") as f:
        movies = pickle.load(f)
    with open("data/Pickle/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("data/Pickle/mv_tags_vectors_umap.pkl", "rb") as f:
        mv_tags_vectors_umap = pickle.load(f)
    with open("data/Pickle/medie_voto_film.pkl", "rb") as f:
        medie_voto_film = pickle.load(f)
    with open("data/Pickle/weighted_score.pkl", "rb") as f:
        weighted_score = pickle.load(f)

    medie_voto_film = medie_voto_film.reset_index().join(
        movies.set_index("movieId")["title"], on="movieId").sort_values(by="rating", ascending=False).dropna()
    weighted_score = weighted_score.reset_index().join(
        movies.set_index("movieId")["title"], on="movieId").sort_values(by="rating", ascending=False).dropna()
    scoring(movies, medie_voto_film, weighted_score)
