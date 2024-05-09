from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load your data
df_history = pd.read_csv("spotify2023data.csv", encoding="utf-16")
df_recommend = pd.read_csv("song-recommend.csv")

# Preprocess data as in your notebook (assuming this step is already correctly done)
# Here, add necessary data processing from your Jupyter notebook
# For example:
df_history['danceability_%'] /= 100  # Example of preprocessing

app = Flask(__name__)

@app.route('/')
def home():
    # Simple form for user input
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from form
    song_name = request.form['song_name']
    primary_feature = request.form['primary_feature']
    primary_weight = float(request.form['primary_weight'])
    rec_count = int(request.form['rec_count'])
    n_neighbors = int(request.form['n_neighbors'])
    
    # Run the recommendation model
    recommendations = weightage_recommend(song_name, primary_feature, primary_weight, rec_count, n_neighbors)
    
    # Send results back to a new HTML template or dynamically update the same page
    return render_template('results.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
