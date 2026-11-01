import pandas as pd

# Load movie dataset
movies = pd.read_csv("dataset/movies.csv")

# Split genres into individual categories
all_genres = movies["genres"].str.split("|").explode()

# Display unique genres
unique_genres = sorted(all_genres.unique())

print("AVAILABLE MOVIE GENRES")
print("----------------------")

for genre in unique_genres:
    print(genre)

print("\nTotal number of genres:", len(unique_genres))

# Convert genres into individual binary columns
genre_data = movies["genres"].str.get_dummies(sep="|")

# Combine movie information with genre columns
movies_processed = pd.concat(
    [movies[["movieId", "title"]], genre_data],
    axis=1
)

# Display the processed data
print("\nPROCESSED MOVIE DATA")
print("--------------------")
print(movies_processed.head(10))

# Save processed data
movies_processed.to_csv(
    "dataset/movies_processed.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")