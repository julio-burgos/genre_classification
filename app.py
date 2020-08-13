from flask import Flask, render_template, jsonify, request
import librosa
import numpy as np
import json

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from RM import cosDisRecommender
from recommender import Recommender
app = Flask(__name__, static_folder="frontend/build/",
            template_folder="frontend/build/")


client_id = "cdb3552deca4415790a3e9cee2417299"
client_secret = ""
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def search(q):
    tracks = sp.search(q)
    tracks = tracks["tracks"]["items"]
    songs = []
    for track in tracks:
        songs.append({"id": track["id"], "name": track["name"], "preview_url": track["preview_url"], "artist": ", ".join(
            [art["name"] for art in track["artists"]])})
    return songs


def get_audio_features(spotify_ids):
    features = sp.audio_features(spotify_ids)
    return features


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/getgenre", methods=["POST"])
def getmelSpectogram():
    request_json = request.get_json()
    if request_json and 'mono' in request_json:
        mono = request_json['mono']
        genreCNN, genreSVM, recommendationCNN, recommendationSVM = Recommender(
        ).get_recommendation(np.array(mono))

        return jsonify({"predictions": genreCNN.tolist()})
    else:
        return f'Error no json or no mono propertie'


@app.route("/searchsongs")
def getSongs():

    q = request.args.get("q")
    if q:
        songs = search(q)
        return jsonify(songs)
    else:
        return f'Error no q propertie'


@app.route("/getRecommendations", methods=["POST"])
def getRecommendedSongs():
    request_json = request.get_json()
    recomendedSongs = []
    if request_json and 'selectedsongs' in request_json:
        selectedsongs = request_json['selectedsongs']
        spotify_ids = [song["id"] for song in selectedsongs]
        audio_features = get_audio_features(spotify_ids)
        keys = ['acousticness', 'danceability', 'energy',
                'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']
        for feat in audio_features:
            RM_features = {key: feat[key] for key in keys}
            songr = cosDisRecommender(RM_features)
            songrec = dict()
            key = list(songr["energy"].items())[0][0]
            for song in songr:
                songrec[song] = songr[song][key]
            recomendedSongs.append(songrec)

        return jsonify(recomendedSongs)
    else:
        return f'Error no q propertie'


app.run(host='0.0.0.0')
