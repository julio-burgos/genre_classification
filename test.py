from recommender import Recommender
import librosa
import os
song, sr = librosa.load(os.path.join('./dataset', 'rosalia.fcking_money_man_milionaria_dio_no_libre_del_dinero.mp3'), 22050, offset=90, duration=30)

print(Recommender().get_recommendation(song))