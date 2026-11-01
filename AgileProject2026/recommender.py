import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load processed movie data
movies = pd.read_csv("dataset/movies_processed.csv")

# Select genre columns
genre_columns = movies.columns.drop(["movieId", "title"])

# Calculate similarity between all movies
similarity_matrix = cosine_similarity(
    movies[genre_columns]
)

# Create a function to recommend movies
def recommend_movie(movie_title, number_of_recommendations=5):

    # Find the selected movie
    movie_index = movies[
        movies["title"].str.contains(
            movie_title, case=False, regex=False
        )
    ].index

    if len(movie_index) == 0:
        print("Movie not found.")
        return

    movie_index = movie_index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    # Sort from highest to lowest similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Exclude the selected movie itself
    similarity_scores = similarity_scores[1:number_of_recommendations + 1]

    print(f"\nMovies similar to {movies.iloc[movie_index]['title']}:")
    print("--------------------------------------")

    for index, score in similarity_scores:
        print(
            f"{movies.iloc[index]['title']} "
            f"- Similarity: {score:.2f}"
        )



def recommend_by_genres(selected_genres, number_of_recommendations=5):

    # Create a list of genre preferences
    user_preferences = pd.DataFrame(
        [[1 if genre in selected_genres else 0
          for genre in genre_columns]],
        columns=genre_columns
    )

    # Calculate similarity between user preferences and all movies
    similarity_scores = cosine_similarity(
        user_preferences,
        movies[genre_columns]
    )[0]

    # Add similarity scores to the movie data
    recommendations = movies.copy()
    recommendations["similarity"] = similarity_scores

    # Sort movies by similarity score
    recommendations = recommendations.sort_values(
        by="similarity",
        ascending=False
    )

    # Display recommendations
    print("\nRecommended movies based on your preferences:")
    print("---------------------------------------------")

    print(
        recommendations[["title", "similarity"]]
        .head(number_of_recommendations)
        .to_string(index=False)
    )



def recommend_by_weighted_genres(user_preferences, number_of_recommendations=5):

    # Convert user preferences into a DataFrame
    preferences = pd.DataFrame(
        [[user_preferences.get(genre, 0) for genre in genre_columns]],
        columns=genre_columns
    )

    # Calculate similarity between preferences and movies
    similarity_scores = cosine_similarity(
        preferences,
        movies[genre_columns]
    )[0]

    # Add similarity scores to movie data
    recommendations = movies.copy()
    recommendations["similarity"] = similarity_scores

    # Sort recommendations by similarity
    recommendations = recommendations.sort_values(
        by="similarity",
        ascending=False
    )

    # Display recommendations
    print("\nRecommendations based on weighted preferences:")
    print("-----------------------------------------------")

    print(
        recommendations[["title", "similarity"]]
        .head(number_of_recommendations)
        .to_string(index=False)
    )



recommend_by_weighted_genres({
    "Animation": 3,
    "Comedy": 2,
    "Adventure": 1
})
