# 🎬 Movie Recommendation System

This project implements a personalized movie recommendation system using various algorithms, providing an interactive web application built with Streamlit. The system aims to suggest movies to users based on their past ratings and movie characteristics.

## ✨ Features

* **Personalized Recommendations:** Get movie suggestions tailored to individual user preferences.
* **Multiple Recommendation Algorithms:**
    * **User-Based Collaborative Filtering:** Recommends movies based on what similar users have liked.
    * **SVD Matrix Factorization:** Utilizes singular value decomposition to uncover latent features in user-movie interactions.
    * **Content-Based Filtering:** Recommends movies based on genre similarity to movies the user has previously enjoyed.
    * **Hybrid Approach:** Combines content-based filtering with movie popularity for more balanced recommendations.
* **Interactive Streamlit Interface:** A user-friendly web application for selecting users, choosing recommendation methods, and viewing results.
* **User Rating History:** View a selected user's top-rated movies to understand their past preferences.
* **Movie Search:** Quickly search for movies within the dataset.
* **Dataset Statistics:** Basic overview of the MovieLens dataset used.

## 📊 Dataset

The system uses the **MovieLens Latest Small Dataset**, which contains:
* `ratings.csv`: User ratings for movies (`userId`, `movieId`, `rating`, `timestamp`).
* `movies.csv`: Movie metadata (`movieId`, `title`, `genres`).

## 🚀 Setup and Installation

Follow these steps to get the project up and running on your local machine.

### Prerequisites

* Python (3.9, 3.10, or 3.11 are recommended for compatibility with specified library versions)
* `pip` (Python package installer)

### 1. Clone the Repository (or create project structure)

If you have a Git repository, clone it:
```bash
git clone <your-repository-url>
cd movie-recommendation-system
````

If you're building locally without Git, ensure your project structure matches:

```
movie-recommendation-system/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data_pipeline.py
│   └── recommendation_models.py
├── models/
├── app/
│   └── streamlit_app.py
├── reports/
├── README.md
└── requirements.txt
```

### 2\. Download the MovieLens Dataset

1.  Go to the MovieLens website: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
2.  Download the `ml-latest-small.zip` file.
3.  Unzip the file and copy `ratings.csv` and `movies.csv` into the `data/raw/` directory of your project.

### 3\. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment (e.g., named 'venv')
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

### 4\. Install Dependencies

Ensure your `requirements.txt` file is up-to-date with the specified versions:

```
# requirements.txt
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.4.2
scipy==1.13.0
streamlit==1.28.0
plotly==5.15.0
# surprise==1.1.1 # Uncomment if you plan to use the Surprise library for advanced CF
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 5\. Run the Data Processing Pipeline

This script cleans the raw data, performs feature engineering (like one-hot encoding genres and creating the user-movie matrix), and saves processed files to `data/processed/`.

Make sure your virtual environment is active and you are in the **root directory** of your project (`movie-recommendation-system/`).

```bash
python src/data_pipeline.py
```

This might take a few moments to run.

### 6\. Launch the Streamlit Application

Once the data pipeline completes successfully, you can launch the interactive web application.

Ensure your virtual environment is active and you are still in the **root directory** of your project.

```bash
streamlit run app/streamlit_app.py
```

This command will open the Streamlit app in your default web browser (usually at `http://localhost:8501`).

## 👨‍💻 Usage

1.  **Select User ID:** Use the dropdown in the sidebar to choose a user for whom you want recommendations.
2.  **Number of Recommendations:** Adjust the slider to specify how many movies you'd like to see.
3.  **Choose Recommendation Method:** Select your preferred algorithm (User-Based CF, SVD, Content-Based, or Hybrid).
4.  **Show User's Rating History:** (Optional) Click this button to see what movies the selected user has already rated.
5.  **Get Recommendations:** Click this button to generate and display the movie recommendations in the main section.
6.  **Explore Movies:** Use the search bar in the sidebar to find specific movies in the dataset.

## ⚙️ Recommendation Models Implemented

  * **User-Based Collaborative Filtering:** Identifies users with similar rating patterns and recommends movies that those similar users liked but the target user hasn't seen.
  * **SVD Matrix Factorization:** Decomposes the user-movie rating matrix into lower-dimensional matrices (latent factors) for users and movies, then uses these factors to predict unrated movies.
  * **Content-Based Filtering (Genre-based):** Builds a profile of the user's preferred genres based on their past ratings and recommends movies with similar genre characteristics.
  * **Hybrid Approach:** Combines the genre-based content score with a movie's overall popularity (number of ratings) to provide a blended recommendation score.

## 🚧 Key Challenges and Solutions

  * **Data Sparsity:** Most users rate only a small fraction of movies. Addressed by using matrix factorization (SVD) and collaborative filtering techniques that can handle sparse data.
  * **Cold Start Problem:** Difficulty recommending for new users or new movies with no ratings. Addressed by incorporating content-based filtering (for new items) and popularity-based components (in the hybrid model).
  * **Data Type Consistency:** Ensuring `movieId` columns across various DataFrames maintain a consistent integer type (`int64`) throughout the data pipeline and Streamlit app to prevent merge errors. This was handled with explicit type casting during data loading and processing.

## 💡 Future Enhancements

  * **Item-Based Collaborative Filtering:** Implement an item-based approach to complement user-based CF.
  * **More Sophisticated Content Features:** Incorporate movie descriptions, directors, actors, or tags for richer content-based recommendations.
  * **Advanced Hybrid Models:** Explore more complex weighting schemes or ensemble methods for combining different recommendation approaches.
  * **User Interface Improvements:** Add features like "Rate a Movie," "Add to Watchlist," or more interactive visualizations.
  * **Model Persistence:** Save trained models (e.g., SVD model, similarity matrices) to disk using `joblib` or `pickle` to avoid retraining on every app restart.
  * **Evaluation Metrics:** Implement and display evaluation metrics (RMSE, Precision@K, Recall@K) to quantitatively assess model performance.

-----