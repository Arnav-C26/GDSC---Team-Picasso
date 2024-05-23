from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load your dataset
data = pd.read_csv('song-recommend.csv')

# Remove rows with null values
data = data.dropna()

# Define the features you want to use for recommendation
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Function to recommend songs
def recommend_songs(song, primary_feature, primary_weightage, num_recommendations, total_recommended):
    # Convert input song name to lowercase
    song = song.lower()

    # Selecting the relevant features for recommendation
    X = data[features]

    # Apply primary weightage
    X[primary_feature] *= primary_weightage
    remaining_weightage = (1 - primary_weightage) / (len(features) - 1)
    for feature in features:
        if feature != primary_feature:
            X[feature] *= remaining_weightage

    # Fit KNN model
    model = NearestNeighbors(n_neighbors=len(data), algorithm='auto')  # Neighbors equal to the total number of songs
    model.fit(X)

    # Transform the input song features
    song_features = data[data['track_name'].str.lower() == song][features]

    # Apply primary weightage to input song features
    song_features[primary_feature] *= primary_weightage
    for feature in features:
        if feature != primary_feature:
            song_features[feature] *= remaining_weightage

    # Find all nearest neighbors
    distances, indices = model.kneighbors(song_features)

    # Retrieve recommended songs, excluding the input song and songs that have been recommended before
    recommended_indices = indices[0][distances[0] > 0]  # Exclude the input song
    recommended_indices = recommended_indices[~np.isin(recommended_indices, total_recommended)]  # Convert to Pandas Series to use isin
    recommended_songs = data.iloc[recommended_indices][['track_name', 'artists', 'album_name']]

    if len(recommended_songs) == 0:
        return None, total_recommended

    # Calculate similarity scores based on distances for the recommended songs only
    similarity_scores = (1 - abs(distances[0][distances[0] > 0])) * 100

    # Add rounded similarity scores to recommended songs dataframe
    recommended_songs['Similarity Score %'] = similarity_scores.round(2)

    # Keep track of total recommended songs
    total_recommended += recommended_indices.tolist()

    return recommended_songs[:num_recommendations], total_recommended

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        song = request.form['song']
        primary_feature = request.form['primary_feature']
        primary_weightage = float(request.form['primary_weightage'])
        num_recommendations = int(request.form['num_recommendations'])
        total_recommended = []

        recommendations, total_recommended = recommend_songs(song, primary_feature, primary_weightage, num_recommendations, total_recommended)

        if recommendations is None:
            message = "No more results"
        else:
            message = recommendations.to_html(index=False)

        return render_template('index.html', message=message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
