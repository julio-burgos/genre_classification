from RM import *

import random
import numpy as np
import pandas as pd

data = {'acousticness': np.random.uniform(0,1), 'danceability' :np.random.uniform(0,1), 'energy' :np.random.uniform(0,1), 'instrumentalness' :np.random.uniform(0,1), 'liveness' :np.random.uniform(0,1), 'speechiness' :np.random.uniform(0,1), 'tempo' :np.random.uniform(0,1), 'valence' :np.random.uniform(0,1)}
print(cosDisRecommender(data,False))