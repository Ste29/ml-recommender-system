# ml-recommender-system

## Table of contents
* [General info](#general-info)
* [Streamlit app](#streamlit-app)
* [Matrix factorization](#matrix-factorization)


## General info
Recommendation systems approaches tested on Movielens dataset from [Kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset)
On top of the studies performed on different algorithms it was also built a webapp using Streamlit

## Streamlit app
Content based recommendation system. Movies are considered similar when their tags are similar.
Pickle data are created using [python notebook](https://github.com/Ste29/)

## Matrix factorization
Contains several different approaches to matrix factorization, starting from simple embedding and dot produt to more
complex ones combining neural networks and external features like tf-idf calculated once on movie descriptions and 
once on genres features