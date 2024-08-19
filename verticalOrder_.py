import ast
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load the datasets
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
merged_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')
merged_df = merged_df[['id', 'title_x', 'genres', 'overview', 'keywords', 'cast', 'crew']]
merged_df = merged_df.rename(columns={'title_x': 'title'})

# Preprocess the data
def parse_features(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)]
    except (ValueError, TypeError, SyntaxError):
        return []

merged_df['genres'] = merged_df['genres'].apply(parse_features)
merged_df['keywords'] = merged_df['keywords'].apply(parse_features)
merged_df['cast'] = merged_df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])

def get_director(x):
    try:
        for i in ast.literal_eval(x):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except (ValueError, TypeError, SyntaxError):
        return []

merged_df['crew'] = merged_df['crew'].apply(get_director)
merged_df['combined_features'] = merged_df['genres'] + merged_df['keywords'] + merged_df['cast'] + merged_df['crew']
merged_df['combined_features'] = merged_df['combined_features'].apply(lambda x: ' '.join(x))

# Vectorize the combined features
vectorizer = CountVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(merged_df['combined_features'])
cosine_sim = cosine_similarity(feature_vectors, feature_vectors)

# Recommendation function
def recommend_movies(title, cosine_sim=cosine_sim, df=merged_df):
    idx = df[df['title'].str.lower() == title.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:11]]
    return df[['title', 'id']].iloc[sim_indices]

# Fetch movie poster from TMDb API
def get_movie_poster(title, api_key):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
    response = requests.get(url)
    data = response.json()
    if data['results']:
        return f"https://image.tmdb.org/t/p/w500{data['results'][0]['poster_path']}"
    else:
        return None

# Streamlit UI
st.title('Movie Recommendation System')

api_key = '26bdabdfa48f274b2548bf66873bba45'  # Your TMDb API key

movie_title = st.text_input('Enter a movie title:')
if st.button('Recommend'):
    if movie_title:
        try:
            recommendations = recommend_movies(movie_title)
            st.write(f'Movies similar to **{movie_title}**:')
            cols = st.columns(3)  # Display posters in 3 columns
            for i, (movie, movie_id) in enumerate(zip(recommendations['title'], recommendations['id']), 1):
                poster_url = get_movie_poster(movie, api_key)
                col = cols[i % 3]  # Cycle through columns
                if poster_url:
                    col.image(poster_url, caption=f"{i}. {movie}")
                else:
                    col.write(f"{i}. {movie} (Poster not found)")
        except IndexError:
            st.write("Movie not found in the database.")
    else:
        st.write("Please enter a movie title.")
