import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("dataset/movies_processed.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Identify genre columns
genre_columns = movies.columns.drop(["movieId", "title"])

# Create user-movie matrix
user_movie_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

# Calculate user similarity
user_similarity = cosine_similarity(
    user_movie_matrix.fillna(0)
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print("Hybrid recommendation system loaded successfully!")


def recommend_hybrid(target_user, user_preferences, number_of_recommendations=10):

    # --------------------------------
    # 1. CONTENT-BASED RECOMMENDATIONS
    # --------------------------------

    preferences = pd.DataFrame(
        [[user_preferences.get(genre, 0) for genre in genre_columns]],
        columns=genre_columns
    )

    content_scores = cosine_similarity(
        preferences,
        movies[genre_columns]
    )[0]

    # --------------------------------
    # 2. COLLABORATIVE RECOMMENDATIONS
    # --------------------------------

    similar_users = user_similarity_df[target_user].drop(
        target_user
    ).sort_values(ascending=False).head(10)

    similar_user_ratings = user_movie_matrix.loc[
        similar_users.index
    ]

    weighted_ratings = similar_user_ratings.mul(
        similar_users,
        axis=0
    )

    similarity_totals = weighted_ratings.notna().mul(
        similar_users,
        axis=0
    ).sum()

    collaborative_scores = weighted_ratings.sum() / similarity_totals

    rating_counts = similar_user_ratings.count()

    collaborative_scores = collaborative_scores[
        rating_counts >= 2
    ]

    # --------------------------------
    # 3. COMBINE BOTH METHODS
    # --------------------------------

    recommendations = movies[["movieId", "title"]].copy()

    recommendations["content_score"] = content_scores

    recommendations["collaborative_score"] = (
        recommendations["movieId"].map(collaborative_scores)
    )

    # Remove movies already rated by the target user
    watched_movies = user_movie_matrix.loc[
        target_user
    ].dropna().index

    recommendations = recommendations[
        ~recommendations["movieId"].isin(watched_movies)
    ]

    # Keep movies with collaborative recommendations
    recommendations = recommendations.dropna(
        subset=["collaborative_score"]
    )

    # Calculate hybrid score
    recommendations["hybrid_score"] = (
        recommendations["content_score"] * 0.4
        + (recommendations["collaborative_score"] / 5) * 0.6
    )

    # Sort by highest hybrid score
    recommendations = recommendations.sort_values(
        by="hybrid_score",
        ascending=False
    )

    # Display recommendations
    print(f"\nHYBRID RECOMMENDATIONS FOR USER {target_user}")
    print("---------------------------------------------")

    print(
        recommendations[
            ["title", "content_score", "collaborative_score", "hybrid_score"]
        ].head(number_of_recommendations).to_string(index=False)
    )


# Test the hybrid recommendation system
recommend_hybrid(
    target_user=1,
    user_preferences={
        "Animation": 3,
        "Comedy": 2,
        "Adventure": 1
    }
)


def recommend_for_new_user(user_preferences, number_of_recommendations=10):

    # Convert preferences into a DataFrame
    preferences = pd.DataFrame(
        [[user_preferences.get(genre, 0) for genre in genre_columns]],
        columns=genre_columns
    )

    # Calculate similarity between preferences and movies
    content_scores = cosine_similarity(
        preferences,
        movies[genre_columns]
    )[0]

    # Create recommendations
    recommendations = movies[["movieId", "title"]].copy()

    recommendations["similarity"] = content_scores

    # Sort by highest similarity
    recommendations = recommendations.sort_values(
        by="similarity",
        ascending=False
    )

    # Display recommendations
    print("\nRECOMMENDATIONS FOR A NEW USER")
    print("--------------------------------")

    print(
        recommendations[["title", "similarity"]]
        .head(number_of_recommendations)
        .to_string(index=False)
    )


# Test recommendations for a new user
recommend_for_new_user({
    "Animation": 3,
    "Comedy": 2,
    "Adventure": 1
})


test test test