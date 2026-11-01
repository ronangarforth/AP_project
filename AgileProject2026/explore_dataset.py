import pandas as pd

# Load the MovieLens CSV files
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")
tags = pd.read_csv("dataset/tags.csv")

# Display basic information
print("MOVIE DATASET")
print("--------------------")
print("Number of movies:", len(movies))

print("\nRATING DATASET")
print("--------------------")
print("Number of ratings:", len(ratings))

print("\nTAG DATASET")
print("--------------------")
print("Number of tags:", len(tags))

# Display some example movies
print("\nEXAMPLE MOVIES")
print("--------------------")
print(movies.head(10))

# Display some example ratings
print("\nEXAMPLE RATINGS")
print("--------------------")
print(ratings.head(10))