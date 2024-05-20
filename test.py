from flask import Flask, render_template, request
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# Replace with your actual Spotify API credentials
client_id = '7c64c104f4c4482592fc0fdd6b8e3b89'
client_secret = 'fca03ba494a54bc4b18fae6d5be4ba3a'

# Set up Spotify authentication
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

@app.route('/', methods=['GET', 'POST'])
def index():
    track_id = None
    track_image_url = None

    if 'track_id' in request.form:
        track_id = request.form['track_id']

        try:
            track_info = sp.track(track_id)
            track_image_url = track_info['album']['images'][0]['url']
        except Exception as e:
            print(f"Error fetching track info: {e}")

    return render_template('test.html', track_id=track_id, track_image_url=track_image_url)

if __name__ == '__main__':
    app.run(debug=True)