from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# Replace with your actual Spotify API credentials
client_id = '7c64c104f4c4482592fc0fdd6b8e3b89'
client_secret = 'fca03ba494a54bc4b18fae6d5be4ba3a'

# Set up Spotify authentication
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load your dataset
data = pd.read_csv('song-recommend.csv')

# Remove rows with null values
data = data.dropna()

# Use only the first artist's name for each track
data['artist_name'] = data['artists'].apply(lambda x: x.split(';')[0])

# Define the features you want to use for recommendation
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'loudness', 'mode']

# Function to recommend songs
def recommend_songs(song, artist, primary_feature, num_recommendations):
    # Convert input song name and artist to lowercase
    song = song.lower()
    artist = artist.lower()

    # Filter the input song by track name and artist name
    song_features = data[(data['track_name'].str.lower() == song) & (data['artist_name'].str.lower() == artist)][features]
    
    if song_features.empty:
        return []

    # Apply weights to the dataset
    feature_weights = {feature: 0.1 for feature in features}
    feature_weights[primary_feature] = 1

    X = data[features].copy()
    for feature in features:
        X[feature] *= feature_weights[feature]

    # Fit KNN model
    model = NearestNeighbors(n_neighbors=len(data), algorithm='auto')
    model.fit(X)

    # Apply weights to the input song features
    song_features_weighted = song_features.copy()
    for feature in features:
        song_features_weighted[feature] *= feature_weights[feature]

    # Find all nearest neighbors
    distances, indices = model.kneighbors(song_features_weighted)

    # Retrieve recommended songs, excluding the input song
    recommended_indices = indices[0][distances[0] > 0][:num_recommendations]
    recommended_songs = data.iloc[recommended_indices][['track_id', 'artist_name', 'track_name'] + features]

    # Fetch track images and other details from Spotify
    track_details = []
    for _, row in recommended_songs.iterrows():
        try:
            track_info = sp.track(row['track_id'])
            track_details.append({
                'image_url': track_info['album']['images'][0]['url'],
                'track_name': row['track_name'],
                'artist_name': row['artist_name']
            })
        except Exception as e:
            print(f"Error fetching track info: {e}")

    return track_details

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate', methods=['GET', 'POST'])
def index():
    song = ""
    artist = ""
    primary_feature = ""
    tracks = []
    message = ""

    if request.method == 'POST':
        if 'get_recommendations' in request.form:
            song = request.form['song']
            artist = request.form['artist']
            primary_feature = request.form['primary_feature']
            num_recommendations = 10

            tracks = recommend_songs(song, artist, primary_feature, num_recommendations)
            if not tracks:
                message = "Sorry, No songs found due to small dataset."
        elif 'clear' in request.form:
            message = "You Haven't Generated Any Songs Yet."

    return render_template('generator.html', song=song, artist=artist, primary_feature=primary_feature, tracks=tracks, features=features, message=message)

if __name__ == '__main__':
    app.run(debug=True)
