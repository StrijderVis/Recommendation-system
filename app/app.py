# Import necessary libraries
import streamlit as st
import pandas as pd
import template as t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure the layout of the Streamlit page to use a wide format and set a page title
st.set_page_config(layout="wide", page_title="Movie Recommender System")

# App title
st.title("🍿 Movie Recommender System")

# Load the dataset containing movie information
df_movies = pd.read_csv('../data/movies.csv', low_memory=False)

# Function to shorten the plot text for display
def shorten_text(text, max_sentences=3):
    sentences = text.split('. ')
    shortened_text = '. '.join(sentences[:max_sentences]) + " [...]"
    return shortened_text

# Define a function to create content-based recommendations
def get_content_based_recommendations(movie_id, df, top_n=6):
    df['combined_features'] = df['Plot'].fillna('') + ' ' + df['Title'].fillna('') + ' ' + df['Genre'].fillna('') + ' ' + df['Director'].fillna('') + ' ' + df['Country'].fillna('')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = df.index[df['movieId'] == movie_id].tolist()[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movie_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
    return df.iloc[similar_movie_indices]

# Initialize session state for storing the current movieId if not already present
if 'movieId' not in st.session_state:
    st.session_state['movieId'] = 3114  # Default movieId to start

# Search bar to find a movie by title
search_query = st.text_input("Search for a movie", "")
if search_query:
    search_results = df_movies[df_movies['Title'].str.contains(search_query, case=False, na=False)]
    if not search_results.empty:
        selected_movie_title = st.selectbox("Select a Movie", search_results['Title'].tolist())
        st.session_state['movieId'] = search_results[search_results['Title'] == selected_movie_title].iloc[0]['movieId']
    else:
        st.write("No movies found. Please try another title.")

# Filter the dataset to only include the selected movie using the stored movieId
df_movie = df_movies[df_movies['movieId'] == st.session_state['movieId']]

# Display selected movie details in two columns: poster and information
cover, info = st.columns([2, 3])

with cover:
    st.image(df_movie['Poster'].iloc[0], caption="Movie Poster")

with info:
    st.title(df_movie['Title'].iloc[0])
    st.caption(f"{df_movie['Year'].iloc[0]} | {df_movie['Runtime'].iloc[0]} | {df_movie['Genre'].iloc[0]} | {df_movie['imdbRating'].iloc[0]} | {df_movie['Actors'].iloc[0]}")
    st.markdown(f"**Plot:** {shorten_text(df_movie['Plot'].iloc[0])}")

st.subheader('Recommendations based on Frequently Reviewed Together (frequency)')
df_freq_reviewed = pd.read_csv('recommendations/recommendations-seeded-freq.csv')
movie_id = st.session_state['movieId']
df_recommendations = df_freq_reviewed[df_freq_reviewed['movie_a'] == movie_id].sort_values(by='count', ascending=False)
df_recommendations = df_recommendations.rename(columns={"movie_b": "movieId"})
df_recommendations = df_recommendations.merge(df_movies, on='movieId')
t.recommendations(df_recommendations.head(6))

# Content-Based Recommendations
st.subheader('Content-Based Recommendations')
recommended_movies = get_content_based_recommendations(st.session_state['movieId'], df_movies, top_n=6)
df_recommendations = recommended_movies.rename(columns={"movieId": "movieId"})
t.recommendations(df_recommendations)

# Display recommendations based on various criteria
st.subheader('Recommendations based on most reviewed')
df_most_reviewed = pd.read_csv('recommendations/recommendations-most-reviewed.csv')
df_most_reviewed = df_most_reviewed.merge(df_movies, on='movieId')
t.recommendations(df_most_reviewed.head(6))

st.subheader('Recommendations based on average rating')
df_avg_rating = pd.read_csv('recommendations/recommendations-ratings-avg.csv')
df_avg_rating = df_avg_rating.merge(df_movies, on='movieId')
t.recommendations(df_avg_rating.head(6))

st.subheader('Recommendations based on weighted rating')
df_weighted_rating = pd.read_csv('recommendations/recommendations-ratings-weight.csv')
df_weighted_rating = df_weighted_rating.merge(df_movies, on='movieId')
t.recommendations(df_weighted_rating.head(6))