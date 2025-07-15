import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommendation_models import user_based_recommendations, fit_svd_model, svd_recommendations, content_based_recommendations_genre, hybrid_content_recommendations

# --- Load Data ---
# Use Streamlit's caching to load data only once
@st.cache_data
def load_data():
    """Loads all necessary processed dataframes."""
    try:
        ratings = pd.read_csv('data/processed/ratings_clean.csv')
        movies = pd.read_csv('data/processed/movies_clean.csv')
        user_movie_matrix = pd.read_csv('data/processed/user_movie_matrix.csv', index_col=0)
        movies_with_features = pd.read_csv('data/processed/movies_with_features.csv')
        
        # Ensure movieId is a column in user_movie_matrix for merging later
        if user_movie_matrix.columns.name == 'movieId':
            user_movie_matrix.columns = user_movie_matrix.columns.astype(int) # Ensure movie IDs are integers
        
        # Fit SVD model once on load
        fit_svd_model(user_movie_matrix) # This will set global svd_model and user_movie_matrix_reduced
        
        return ratings, movies, user_movie_matrix, movies_with_features
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure you have run src/data_pipeline.py to process the data.")
        st.stop() # Stop the app if data files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.stop()

ratings, movies, user_movie_matrix, movies_with_features = load_data()

st.title("🎬 Movie Recommendation System")
st.markdown("Get personalized movie recommendations based on your preferences!")

# --- Sidebar for User Input ---
st.sidebar.header("Recommendation Settings")

# User selection
user_ids = sorted(ratings['userId'].unique())
selected_user = st.sidebar.selectbox("Select User ID:", user_ids)

# Number of recommendations
num_recs = st.sidebar.slider("Number of recommendations:", 1, 20, 10)

# Model selection
model_type = st.sidebar.selectbox(
    "Choose recommendation method:",
    ["User-Based Collaborative Filtering", "SVD Matrix Factorization", "Content-Based Filtering", "Hybrid Approach"]
)

# Display user's rating history
if st.sidebar.button("Show User's Rating History"):
    user_ratings = ratings[ratings['userId'] == selected_user].merge(movies, on='movieId')
    user_ratings = user_ratings.sort_values('rating', ascending=False)
    st.subheader(f"User {selected_user}'s Rating History")
    st.dataframe(user_ratings[['title', 'genres', 'rating']].head(10))

# --- Main Recommendation Section ---
if st.button("Get Recommendations"):
    recommendations = pd.Series(dtype=float) # Initialize as empty Series
    try:
        if model_type == "User-Based Collaborative Filtering":
            recommendations = user_based_recommendations(selected_user, user_movie_matrix, num_recs)
        elif model_type == "SVD Matrix Factorization":
            # SVD model is fitted during load_data, so we can directly call svd_recommendations
            recommendations = svd_recommendations(selected_user, user_movie_matrix, num_recs)
        elif model_type == "Content-Based Filtering":
            recommendations = content_based_recommendations_genre(selected_user, movies_with_features, ratings, num_recs)
        elif model_type == "Hybrid Approach":
            recommendations = hybrid_content_recommendations(selected_user, movies_with_features, ratings, num_recs)
        
        if not recommendations.empty and len(recommendations) > 0:
            # Get movie details for recommendations
            # Ensure movieIds are correctly aligned for merging
            rec_movie_ids = recommendations.index.astype(int) # Convert index to int for merging
            
            # Filter movies_with_features for the recommended movie IDs
            rec_movies_details = movies_with_features[movies_with_features['movieId'].isin(rec_movie_ids)].copy()
            
            # Merge with recommendation scores
            # Reset index of recommendations to merge on movie ID
            rec_movies_details = rec_movies_details.merge(
                recommendations.reset_index().rename(columns={'index': 'movieId', 0: 'rec_score'}),
                on='movieId',
                how='left'
            )
            
            # Sort by recommendation score
            rec_movies_details = rec_movies_details.sort_values('rec_score', ascending=False)
            
            st.subheader(f"Top {num_recs} Recommendations for User {selected_user}")
            
            # Display recommendations in a nice format
            for idx, movie in rec_movies_details.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{idx+1}. {movie['title']}**") # Use idx+1 for display numbering
                        st.markdown(f"*Genres:* {movie['genres']}")
                        if 'avg_rating' in movie:
                            st.markdown(f"*Average Rating:* {movie['avg_rating']:.1f} ⭐")
                        if 'num_ratings' in movie:
                            st.markdown(f"*Number of Ratings:* {int(movie['num_ratings'])}") # Display as integer
                    
                    with col2:
                        st.metric("Recommendation Score", f"{movie['rec_score']:.3f}")
                    
                    st.markdown("---")
        else:
            st.warning("No recommendations found for this user with the selected method. Try a different user or method.")
            
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        st.exception(e) # Display full traceback for debugging

# --- Additional Features (Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Explore Movies")

# Movie search
search_term = st.sidebar.text_input("Search for a movie by title:")
if search_term:
    matching_movies = movies[movies['title'].str.contains(search_term, case=False, na=False)]
    if not matching_movies.empty:
        st.sidebar.write("Found movies (Top 5):")
        for _, movie in matching_movies.head(5).iterrows():
            st.sidebar.write(f"- {movie['title']} ({movie['genres']})")
    else:
        st.sidebar.write("No movies found matching your search.")

# Statistics
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Statistics")
st.sidebar.write(f"Total Users: {ratings['userId'].nunique()}")
st.sidebar.write(f"Total Movies: {movies['movieId'].nunique()}")
st.sidebar.write(f"Total Ratings (Cleaned): {len(ratings)}")
