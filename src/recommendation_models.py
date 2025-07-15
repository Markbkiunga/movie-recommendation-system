import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# --- Helper to load data (for individual testing or if not pre-loaded) ---
# In the Streamlit app, data will be loaded once.
# These functions assume necessary dataframes are passed as arguments.

# --- 1. User-Based Collaborative Filtering ---
def user_based_recommendations(user_id, user_movie_matrix, n_recommendations=10):
    """
    Generates user-based collaborative filtering recommendations.
    
    Args:
        user_id (int): The ID of the target user.
        user_movie_matrix (pd.DataFrame): User-movie rating matrix (users as index, movies as columns).
        n_recommendations (int): Number of recommendations to generate.
        
    Returns:
        pd.Series: Recommended movie IDs with their scores, sorted descending.
    """
    if user_id not in user_movie_matrix.index:
        print(f"User {user_id} not found in the user-movie matrix.")
        return pd.Series(dtype=float)

    # Calculate user similarity (cosine similarity)
    # Ensure user_movie_matrix contains only numeric values (ratings)
    user_similarity = cosine_similarity(user_movie_matrix)
    user_sim_df = pd.DataFrame(user_similarity, 
                               index=user_movie_matrix.index, 
                               columns=user_movie_matrix.index)
    
    # Get similar users (excluding the user itself)
    # Get top 10 similar users, or fewer if not enough
    similar_users = user_sim_df[user_id].sort_values(ascending=False).drop(user_id)
    if len(similar_users) > 10:
        similar_users = similar_users.head(10)
    
    # If no similar users, return empty recommendations
    if similar_users.empty:
        return pd.Series(dtype=float)

    # Get movies already rated by the target user
    user_ratings = user_movie_matrix.loc[user_id]
    
    recommendations = pd.Series(dtype=float)
    
    # Iterate through similar users to find unrated movies
    for similar_user_id, similarity_score in similar_users.items():
        similar_user_ratings = user_movie_matrix.loc[similar_user_id]
        
        # Find movies rated by similar user but not by target user
        # Filter for movies where the target user has not rated (rating == 0 in the matrix)
        unrated_movies_by_target = similar_user_ratings[user_ratings == 0]
        
        # Add weighted ratings to recommendations
        # Only consider movies actually rated by the similar user (rating > 0)
        for movie_id, rating in unrated_movies_by_target.items():
            if rating > 0: # Only consider actual ratings, not fill_value=0
                recommendations[movie_id] = recommendations.get(movie_id, 0) + (rating * similarity_score)
    
    # Remove any movies that might have slipped through (already rated by target user)
    recommendations = recommendations.drop(user_ratings[user_ratings > 0].index, errors='ignore')

    return recommendations.sort_values(ascending=False)[:n_recommendations]

# --- 2. SVD Matrix Factorization ---
# SVD model needs to be fitted once on the user_movie_matrix
# For the Streamlit app, we'll fit it on load.
svd_model = None # Placeholder for the fitted SVD model
user_movie_matrix_reduced = None # Placeholder for the transformed matrix

def fit_svd_model(user_movie_matrix, n_components=50):
    """Fits the TruncatedSVD model and transforms the user-movie matrix."""
    global svd_model, user_movie_matrix_reduced
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    user_movie_matrix_reduced = svd_model.fit_transform(user_movie_matrix)
    print(f"SVD model fitted with {n_components} components.")
    # You might want to save svd_model and user_movie_matrix_reduced to disk here
    # import joblib
    # joblib.dump(svd_model, 'models/svd_model.pkl')
    # np.save('models/user_movie_matrix_reduced.npy', user_movie_matrix_reduced)


def svd_recommendations(user_id, user_movie_matrix, n_recommendations=10):
    """
    Generates recommendations using SVD matrix factorization.
    Assumes svd_model and user_movie_matrix_reduced are globally available or passed.
    
    Args:
        user_id (int): The ID of the target user.
        user_movie_matrix (pd.DataFrame): Original user-movie rating matrix (needed for unrated movies).
        n_recommendations (int): Number of recommendations to generate.
        
    Returns:
        pd.Series: Recommended movie IDs with their predicted scores, sorted descending.
    """
    if svd_model is None or user_movie_matrix_reduced is None:
        print("SVD model not fitted. Please call fit_svd_model first.")
        return pd.Series(dtype=float)

    if user_id not in user_movie_matrix.index:
        print(f"User {user_id} not found in the user-movie matrix.")
        return pd.Series(dtype=float)

    user_index = list(user_movie_matrix.index).index(user_id)
    user_vector = user_movie_matrix_reduced[user_index]
    
    # Predict ratings for all movies
    # svd_model.components_ has shape (n_components, n_movies)
    movie_scores = np.dot(user_vector, svd_model.components_)
    movie_recommendations = pd.Series(movie_scores, index=user_movie_matrix.columns)
    
    # Remove already rated movies
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = movie_recommendations[user_ratings == 0] # Filter out movies the user has already rated
    
    return unrated_movies.sort_values(ascending=False)[:n_recommendations]

# --- 3. Content-Based Filtering (Genre-based) ---
def content_based_recommendations_genre(user_id, movies_with_features, ratings_clean, n_recommendations=10):
    """
    Generates content-based recommendations based on user's preferred genres.
    
    Args:
        user_id (int): The ID of the target user.
        movies_with_features (pd.DataFrame): Movies DataFrame with one-hot encoded genre features.
        ratings_clean (pd.DataFrame): Cleaned ratings DataFrame.
        n_recommendations (int): Number of recommendations to generate.
        
    Returns:
        pd.Series: Recommended movie IDs with their scores, sorted descending.
    """
    user_movies = ratings_clean[ratings_clean['userId'] == user_id]
    
    if len(user_movies) == 0:
        print(f"User {user_id} has no ratings for content-based recommendations.")
        return pd.Series(dtype=float)
    
    # Get genre columns (all columns that are not metadata)
    genre_columns = [col for col in movies_with_features.columns 
                     if col not in ['movieId', 'title', 'genres', 'genre_list', 'num_ratings', 'avg_rating', 'rating_std', 'num_users', 'index', 'rec_score']] # Added 'index', 'rec_score' from Streamlit output
    
    # Filter movies_with_features to only include movies that have genre data
    movies_with_features_genres_only = movies_with_features[genre_columns]

    # Get genre features for rated movies
    rated_movies_features = movies_with_features[movies_with_features['movieId'].isin(user_movies['movieId'])]
    
    # Create user profile (weighted average of genre preferences)
    user_ratings_dict = dict(zip(user_movies['movieId'], user_movies['rating']))
    user_profile = np.zeros(len(genre_columns))
    
    for _, movie in rated_movies_features.iterrows():
        movie_rating = user_ratings_dict.get(movie['movieId'], 0) # Use .get() for safety
        if movie_rating > 0: # Only consider movies actually rated by the user
            movie_genres = movie[genre_columns].values
            user_profile += movie_rating * movie_genres
    
    # Normalize user profile to avoid bias from users who rated many movies
    if np.sum(user_profile) > 0:
        user_profile = user_profile / np.sum(user_profile)
    else:
        return pd.Series(dtype=float) # User profile is all zeros, cannot recommend

    # Calculate similarity with all movies
    all_movies_features = movies_with_features_genres_only.values
    movie_scores = np.dot(all_movies_features, user_profile)
    
    # Create recommendations series
    recommendations = pd.Series(movie_scores, index=movies_with_features['movieId'])
    
    # Remove already rated movies
    rated_movie_ids = user_movies['movieId'].values
    recommendations = recommendations.drop(rated_movie_ids, errors='ignore')
    
    return recommendations.sort_values(ascending=False)[:n_recommendations]

# --- 4. Hybrid Approach (Content-based + Popularity) ---
def hybrid_content_recommendations(user_id, movies_with_features, ratings_clean, n_recommendations=10, genre_weight=0.7, popularity_weight=0.3):
    """
    Combines genre-based content recommendations with movie popularity.
    
    Args:
        user_id (int): The ID of the target user.
        movies_with_features (pd.DataFrame): Movies DataFrame with genre and popularity features.
        ratings_clean (pd.DataFrame): Cleaned ratings DataFrame.
        n_recommendations (int): Number of recommendations to generate.
        genre_weight (float): Weight for genre-based score.
        popularity_weight (float): Weight for popularity score.
        
    Returns:
        pd.Series: Recommended movie IDs with their combined scores, sorted descending.
    """
    # Get genre-based recommendations (more than final count to allow for filtering)
    genre_recs = content_based_recommendations_genre(user_id, movies_with_features, ratings_clean, n_recommendations * 5) # Get more to ensure enough after popularity filtering
    
    if genre_recs.empty:
        return pd.Series(dtype=float)
    
    # Get popularity scores (normalized)
    # Ensure 'num_ratings' exists and handle potential NaNs
    popularity_scores = movies_with_features.set_index('movieId')['num_ratings'].fillna(0)
    if popularity_scores.max() > popularity_scores.min():
        popularity_scores = (popularity_scores - popularity_scores.min()) / (popularity_scores.max() - popularity_scores.min())
    else: # Handle case where all num_ratings are the same (e.g., all zero)
        popularity_scores = pd.Series(0.5, index=popularity_scores.index) # Assign a neutral score

    # Combine genre and popularity scores
    hybrid_scores = pd.Series(index=genre_recs.index, dtype=float)
    
    for movie_id in genre_recs.index:
        genre_score = genre_recs[movie_id]
        pop_score = popularity_scores.get(movie_id, 0) # Use .get() for safety
        hybrid_scores[movie_id] = genre_weight * genre_score + popularity_weight * pop_score
    
    # Remove already rated movies (content-based already does this, but double-check)
    user_ratings = ratings_clean[ratings_clean['userId'] == user_id]
    rated_movie_ids = user_ratings['movieId'].values
    hybrid_scores = hybrid_scores.drop(rated_movie_ids, errors='ignore')

    return hybrid_scores.sort_values(ascending=False)[:n_recommendations]

