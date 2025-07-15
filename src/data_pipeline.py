import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

print("--- Starting Data Preprocessing and Feature Engineering ---")

# --- 1. Data Loading (Week 1) ---
print("\n1. Loading raw datasets...")
ratings = pd.read_csv('data/raw/ratings.csv')
movies = pd.read_csv('data/raw/movies.csv')

print("Ratings dataset shape:", ratings.shape)
print("Movies dataset shape:", movies.shape)

# --- 2. Data Exploration and Cleaning (Week 2) ---
print("\n2. Performing Data Cleaning Tasks...")

# Check for missing values
print("Missing values in ratings:\n", ratings.isnull().sum())
print("\nMissing values in movies:\n", movies.isnull().sum())

# Check for duplicates
print(f"\nDuplicate ratings: {ratings.duplicated().sum()}")
if ratings.duplicated().sum() > 0:
    ratings.drop_duplicates(inplace=True)
    print("Duplicate ratings removed.")

# Remove users with less than 5 ratings
user_counts = ratings['userId'].value_counts()
active_users = user_counts[user_counts >= 5].index
ratings_filtered = ratings[ratings['userId'].isin(active_users)].copy() # Use .copy() to avoid SettingWithCopyWarning
print(f"Users after filtering (>=5 ratings): {ratings_filtered['userId'].nunique()}")

# Remove movies with less than 5 ratings
movie_counts = ratings_filtered['movieId'].value_counts()
popular_movies = movie_counts[movie_counts >= 5].index
ratings_filtered = ratings_filtered[ratings_filtered['movieId'].isin(popular_movies)].copy() # Use .copy()
print(f"Movies after filtering (>=5 ratings): {ratings_filtered['movieId'].nunique()}")

# Keep only movies that exist in both datasets after filtering
common_movies_ids = set(ratings_filtered['movieId']) & set(movies['movieId'])
ratings_clean = ratings_filtered[ratings_filtered['movieId'].isin(common_movies_ids)].copy()
movies_clean = movies[movies['movieId'].isin(common_movies_ids)].copy()

print(f"\nFinal cleaned dataset sizes:")
print(f"Clean ratings: {len(ratings_clean)}")
print(f"Clean movies: {len(movies_clean)}")
print(f"Users: {ratings_clean['userId'].nunique()}")
print(f"Movies: {ratings_clean['movieId'].nunique()}")

# Convert timestamp to datetime for temporal analysis (if needed later, though not explicitly used in final app)
if 'timestamp' in ratings_clean.columns:
    ratings_clean['datetime'] = pd.to_datetime(ratings_clean['timestamp'], unit='s')
    ratings_clean['year'] = ratings_clean['datetime'].dt.year
    ratings_clean['month'] = ratings_clean['datetime'].dt.month

# Save cleaned data
ratings_clean.to_csv('data/processed/ratings_clean.csv', index=False)
movies_clean.to_csv('data/processed/movies_clean.csv', index=False)
print("Cleaned data saved to data/processed/")

# --- 3. Feature Engineering (Week 3) ---
print("\n3. Performing Feature Engineering...")

# Create user-movie rating matrix (for Collaborative Filtering)
user_movie_matrix = ratings_clean.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)
print(f"User-movie matrix shape: {user_movie_matrix.shape}")

# Genre processing (for Content-Based Filtering)
movies_clean['genre_list'] = movies_clean['genres'].str.split('|')
movies_clean['genre_list'] = movies_clean['genre_list'].fillna('Unknown').apply(
    lambda x: ['Unknown'] if x == 'Unknown' else x
)

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies_clean['genre_list'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
movies_with_features = pd.concat([movies_clean.reset_index(drop=True), genre_df], axis=1)

print(f"Available genres: {list(mlb.classes_)}")
print(f"Movies with genre features shape: {movies_with_features.shape}")

# User statistics (e.g., for user features in hybrid models)
user_stats = ratings_clean.groupby('userId').agg({
    'rating': ['count', 'mean', 'std'],
    'movieId': 'nunique'
}).round(2)
user_stats.columns = ['num_ratings', 'avg_rating', 'rating_std', 'num_movies']
user_stats['rating_std'] = user_stats['rating_std'].fillna(0) # Handle users with only one rating

# Movie statistics (e.g., for popularity in hybrid models)
movie_stats = ratings_clean.groupby('movieId').agg({
    'rating': ['count', 'mean', 'std'],
    'userId': 'nunique'
}).round(2)
movie_stats.columns = ['num_ratings', 'avg_rating', 'rating_std', 'num_users']
movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)

# Merge movie stats with movie info
movies_with_features = movies_with_features.merge(movie_stats[['num_ratings', 'avg_rating']], on='movieId', how='left')


# User genre preferences (if needed for specific content-based approaches)
# This part of feature engineering is more complex and might be simplified for initial implementation
# user_genre_prefs = ratings_clean.merge(movies_clean, on='movieId')
# user_genre_matrix = pd.DataFrame()
# for genre in mlb.classes_:
#     genre_ratings = user_genre_prefs[user_genre_prefs['genres'].str.contains(genre, na=False)]
#     if len(genre_ratings) > 0:
#         genre_avg = genre_ratings.groupby('userId')['rating'].mean()
#         user_genre_matrix[genre] = genre_avg
# user_genre_matrix = user_genre_matrix.fillna(0)
# print(f"User genre preferences matrix shape: {user_genre_matrix.shape}")


# Save processed features
user_movie_matrix.to_csv('data/processed/user_movie_matrix.csv')
movies_with_features.to_csv('data/processed/movies_with_features.csv', index=False)
user_stats.to_csv('data/processed/user_stats.csv')
# user_genre_matrix.to_csv('data/processed/user_genre_preferences.csv') # Uncomment if you implement user_genre_matrix
print("Feature engineered data saved to data/processed/")

# --- Train/Test Split Strategy ---
print("\n4. Performing Train/Test Split...")
# Time-based split (more realistic for recommendations)
# Sort by timestamp and split 80-20
ratings_sorted = ratings_clean.sort_values('timestamp')
split_index = int(len(ratings_sorted) * 0.8)
train_ratings = ratings_sorted[:split_index].copy()
test_ratings = ratings_sorted[split_index:].copy()

print(f"Train set size: {len(train_ratings)}")
print(f"Test set size: {len(test_ratings)}")

# Create train and test matrices
train_matrix = train_ratings.pivot_table(
    index='userId', columns='movieId', values='rating', fill_value=0
)
test_matrix = test_ratings.pivot_table(
    index='userId', columns='movieId', values='rating', fill_value=0
)

# Ensure same dimensions and common users/movies for evaluation consistency
# This step is crucial for evaluation later if you want to compare predictions on test_matrix
common_users_split = list(set(train_matrix.index) & set(test_matrix.index))
common_movies_split = list(set(train_matrix.columns) & set(test_matrix.columns))

train_matrix = train_matrix.loc[common_users_split, common_movies_split]
test_matrix = test_matrix.loc[common_users_split, common_movies_split]

print(f"Train matrix shape: {train_matrix.shape}")
print(f"Test matrix shape: {test_matrix.shape}")

# Save split data
train_ratings.to_csv('data/processed/train_ratings.csv', index=False)
test_ratings.to_csv('data/processed/test_ratings.csv', index=False)
train_matrix.to_csv('data/processed/train_matrix.csv')
test_matrix.to_csv('data/processed/test_matrix.csv')
print("Train/Test split data saved to data/processed/")

print("\n--- Data Pipeline Completed Successfully ---")
