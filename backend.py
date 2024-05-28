from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load your dataset
data = pd.read_csv('song-recommend.csv')

# Remove rows with null values
data = data.dropna()

# Use only the first artist's name for each track
data['artist_name'] = data['artists'].apply(lambda x: x.split(';')[0])

# Define the features you want to use for recommendation
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'loudness', 'mode']

# Function to recommend songs
def recommend_songs(song, artist, primary_feature, num_recommendations=10):
    # Convert input song name and artist to lowercase
    song = song.lower()
    artist = artist.lower()

    # Filter the input song by track name and artist name
    song_features = data[(data['track_name'].str.lower() == song) & (data['artist_name'].str.lower() == artist)][features]
    
    if song_features.empty:
        return None

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

    return recommended_songs

@app.route('/', methods=['GET', 'POST'])
def index():
    song = ""
    artist = ""
    primary_feature = ""
    message = ""
    recommendations = None

    if request.method == 'POST':
        if 'get_recommendations' in request.form:
            song = request.form['song']
            artist = request.form['artist']
            primary_feature = request.form['primary_feature']

            recommendations = recommend_songs(song, artist, primary_feature)

            if recommendations is None or recommendations.empty:
                message = "No recommendations found. Please check the song and artist name."
                recommendations = pd.DataFrame()  # Ensure recommendations is not None
            else:
                # Redirect to the recommendations page
                return redirect(url_for('recommendations', song=song, artist=artist, primary_feature=primary_feature))

    return render_template('generator.html', song=song, artist=artist, primary_feature=primary_feature, message=message, features=features, recommendations=recommendations)

@app.route('/recommendations', methods=['GET'])
def recommendations():
    song = request.args.get('song')
    artist = request.args.get('artist')
    primary_feature = request.args.get('primary_feature')

    recommendations = recommend_songs(song, artist, primary_feature)

    return render_template('recommendations.html', song=song, artist=artist, primary_feature=primary_feature, features=features, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
