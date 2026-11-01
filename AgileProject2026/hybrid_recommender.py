
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. LOAD DATASETS
# --------------------------------------------------

movies = pd.read_csv("dataset/movies_processed.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Identify all genre columns, excluding movieId and title
genre_columns = movies.columns.drop(["movieId", "title"])


# --------------------------------------------------
# 2. PREPARE COLLABORATIVE FILTERING DATA
# --------------------------------------------------

# Create a matrix where each row is a user,
# each column is a movie, and values are ratings.
user_movie_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

# Calculate similarity between users
user_similarity = cosine_similarity(
    user_movie_matrix.fillna(0)
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print("Hybrid recommendation system loaded successfully!")


# --------------------------------------------------
# 3. HYBRID RECOMMENDATIONS FOR EXISTING USERS
# --------------------------------------------------

def recommend_hybrid(
    target_user,
    user_preferences,
    number_of_recommendations=10
):

    # Calculate content-based scores
    preferences = pd.DataFrame(
        [[user_preferences.get(genre, 0) for genre in genre_columns]],
        columns=genre_columns
    )

    content_scores = cosine_similarity(
        preferences,
        movies[genre_columns]
    )[0]

    # Find the 10 most similar users
    similar_users = (
        user_similarity_df[target_user]
        .drop(target_user)
        .sort_values(ascending=False)
        .head(10)
    )

    # Retrieve ratings from similar users
    similar_user_ratings = user_movie_matrix.loc[
        similar_users.index
    ]

    # Weight ratings according to user similarity
    weighted_ratings = similar_user_ratings.mul(
        similar_users,
        axis=0
    )

    # Calculate weighted average ratings
    similarity_totals = (
        weighted_ratings.notna()
        .mul(similar_users, axis=0)
        .sum()
    )

    collaborative_scores = (
        weighted_ratings.sum() / similarity_totals
    )

    # Only consider movies rated by at least 2 similar users
    rating_counts = similar_user_ratings.count()

    collaborative_scores = collaborative_scores[
        rating_counts >= 2
    ]

    # Combine content and collaborative scores
    recommendations = movies[["movieId", "title"]].copy()

    recommendations["content_score"] = content_scores

    recommendations["collaborative_score"] = (
        recommendations["movieId"].map(collaborative_scores)
    )

    # Remove movies the user has already rated
    watched_movies = (
        user_movie_matrix.loc[target_user]
        .dropna()
        .index
    )

    recommendations = recommendations[
        ~recommendations["movieId"].isin(watched_movies)
    ]

    # Remove movies without collaborative scores
    recommendations = recommendations.dropna(
        subset=["collaborative_score"]
    )

    # Calculate the final hybrid score
    # 40% content-based + 60% collaborative filtering
    recommendations["hybrid_score"] = (
        recommendations["content_score"] * 0.4
        + (recommendations["collaborative_score"] / 5) * 0.6
    )

    # Sort by highest hybrid score
    recommendations = recommendations.sort_values(
        by="hybrid_score",
        ascending=False
    )

    print(f"\nHYBRID RECOMMENDATIONS FOR USER {target_user}")
    print("---------------------------------------------")

    print(
        recommendations[
            [
                "title",
                "content_score",
                "collaborative_score",
                "hybrid_score"
            ]
        ]
        .head(number_of_recommendations)
        .to_string(index=False)
    )


# --------------------------------------------------
# 4. RECOMMENDATIONS FOR NEW USERS
# --------------------------------------------------

def recommend_for_new_user(
    user_preferences,
    number_of_recommendations=10
):

    # Convert user preferences into a dataframe
    preferences = pd.DataFrame(
        [[user_preferences.get(genre, 0) for genre in genre_columns]],
        columns=genre_columns
    )

    # Calculate similarity between preferences and movies
    similarity_scores = cosine_similarity(
        preferences,
        movies[genre_columns]
    )[0]

    # Create recommendation results
    recommendations = movies[["movieId", "title"]].copy()

    recommendations["similarity"] = similarity_scores

    # Sort by highest similarity
    recommendations = recommendations.sort_values(
        by="similarity",
        ascending=False
    )

    print("\nRECOMMENDATIONS FOR A NEW USER")
    print("--------------------------------")

    print(
        recommendations[["title", "similarity"]]
        .head(number_of_recommendations)
        .to_string(index=False)
    )


# --------------------------------------------------
# 5. MOOD-BASED PREFERENCES
# --------------------------------------------------

def get_mood_preferences(mood):

    mood_genres = {
        "happy": {
            "Comedy": 3,
            "Animation": 2,
            "Musical": 2
        },

        "excited": {
            "Action": 3,
            "Adventure": 3,
            "Sci-Fi": 2
        },

        "thoughtful": {
            "Drama": 3,
            "Mystery": 2,
            "Documentary": 2
        },

        "scared": {
            "Horror": 3,
            "Thriller": 3
        },

        "relaxed": {
            "Comedy": 2,
            "Romance": 3,
            "Fantasy": 2
        }
    }

    if mood.lower() not in mood_genres:
        print("Mood not recognised.")
        return {}

    return mood_genres[mood.lower()]


# --------------------------------------------------
# 6. COMBINE MOOD AND GENRE PREFERENCES
# --------------------------------------------------

def combine_preferences(
    mood,
    favourite_genres=None,
    less_interested_genres=None
):

    # Start with the preferences associated with the mood
    preferences = get_mood_preferences(mood)

    if not preferences:
        return {}

    # Increase favourite genres to a preference score of 3
    if favourite_genres:

        for genre in favourite_genres:

            if genre in genre_columns:

                preferences[genre] = max(
                    preferences.get(genre, 0),
                    3
                )

            else:
                print(f"Genre not recognised: {genre}")

    # Set less-interested genres to 0
    if less_interested_genres:

        for genre in less_interested_genres:

            if genre in genre_columns:

                preferences[genre] = 0

            else:
                print(f"Genre not recognised: {genre}")

    return preferences


# --------------------------------------------------
# 7. TEST THE RECOMMENDATION SYSTEM
# --------------------------------------------------

if __name__ == "__main__":

    # Example questionnaire answers
    selected_mood = "happy"

    favourite_genres = [
        "Animation",
        "Adventure"
    ]

    less_interested_genres = [
        "Horror"
    ]

    # Combine the questionnaire answers
    combined_preferences = combine_preferences(
        selected_mood,
        favourite_genres,
        less_interested_genres
    )

    print("\nSELECTED MOOD:", selected_mood)

    print("\nFAVOURITE GENRES:", favourite_genres)

    print("\nLESS INTERESTED IN:", less_interested_genres)

    print("\nCOMBINED QUESTIONNAIRE PREFERENCES:")
    print(combined_preferences)

    # Generate recommendations for a new user
    if combined_preferences:

        recommend_for_new_user(
            combined_preferences,
            number_of_recommendations=10
        )

    # Example: Generate hybrid recommendations for an existing user
    # Uncomment the following lines to test this separately.

    # recommend_hybrid(
    #     target_user=1,
    #     user_preferences=combined_preferences
    # )
