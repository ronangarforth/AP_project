import pandas as pd

# Load the MovieLens ratings dataset
ratings = pd.read_csv("dataset/ratings.csv")

# Create a user-movie matrix
user_movie_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

# Display the matrix dimensions
print("\nUSER-MOVIE MATRIX")
print("--------------------")
print("Number of users:", user_movie_matrix.shape[0])
print("Number of movies:", user_movie_matrix.shape[1])


from sklearn.metrics.pairwise import cosine_similarity

# Replace missing ratings with 0
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Calculate similarity between users
user_similarity = cosine_similarity(
    user_movie_matrix_filled
)

# Convert similarity scores into a DataFrame
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# Display similar users for User 1
similar_users = user_similarity_df[1].sort_values(
    ascending=False
)

print("\nMOST SIMILAR USERS TO USER 1")
print("----------------------------")
print(similar_users.head(6))



def recommend_for_user(target_user, number_of_recommendations=10):

    # Get users most similar to the target user
    similar_users = user_similarity_df[target_user].sort_values(
        ascending=False
    )

    # Exclude the target user themselves
    similar_users = similar_users.drop(target_user)

    # Select the 10 most similar users
    similar_users = similar_users.head(10)

    # Get movies already rated by the target user
    watched_movies = user_movie_matrix.loc[target_user].dropna().index

    # Get ratings from similar users
    similar_user_ratings = user_movie_matrix.loc[
        similar_users.index
    ]

    # Calculate weighted ratings
    weighted_ratings = similar_user_ratings.mul(
        similar_users,
        axis=0
    )

    # Calculate total similarity for each movie
    similarity_totals = weighted_ratings.notna().mul(
        similar_users,
        axis=0
    ).sum()

    # Calculate weighted average ratings
    predicted_ratings = (
        weighted_ratings.sum() / similarity_totals
    )

    # Count how many similar users rated each movie
    rating_counts = similar_user_ratings.count()

    # Require at least 2 ratings
    predicted_ratings = predicted_ratings[
        rating_counts >= 2
    ]

    # Exclude movies already rated by the target user
    recommendations = predicted_ratings.drop(
        watched_movies,
        errors="ignore"
    )

    # Sort by highest predicted rating
    recommendations = recommendations.sort_values(
        ascending=False
    )

    # Get the top recommendations
    recommendations = recommendations.head(
        number_of_recommendations
    )

    # Match movie IDs with movie titles
    movies = pd.read_csv("dataset/movies.csv")

    recommendations = recommendations.reset_index()
    recommendations.columns = ["movieId", "predicted_rating"]

    recommendations = recommendations.merge(
        movies,
        on="movieId"
    )

    print(f"\nIMPROVED MOVIE RECOMMENDATIONS FOR USER {target_user}")
    print("---------------------------------------------------")

    print(
        recommendations[["title", "predicted_rating"]]
        .to_string(index=False)
    )


# Test the improved recommendations system
recommend_for_user(1)
