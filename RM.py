#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:22:29 2019
@author: anilkiroglu
"""
import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cosine 
from numpy import dot
from sklearn import preprocessing
from numpy.linalg import norm


#Reading the features file created by Federico


fma_small = pd.read_csv('features_TEST_2.csv') 
echonest = pd.read_csv( 'final.csv')
metadata = pd.read_csv( 'raw_tracks.csv')

# data as dictionary with keys as column names, keys needs to be in order of ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']
# if tempoNotIncluded = True :drops the tempo column 
def cosDisRecommender(data,tempoNotIncluded=False):
    
    user_Input = pd.DataFrame(data, index=[0]) 
    cond = echonest['track_id'].isin(fma_small['track_id']) == False
    echonest.drop(echonest[cond].index, inplace = True)

    toRecommendFrom = echonest.loc[:,'acousticness':'valence']
    
    if tempoNotIncluded: 
        toRecommendFrom = toRecommendFrom.drop(['tempo'], axis=1)
        user_Input = user_Input.drop(['tempo'], axis = 1)

    minDistIndex = []
 
    i = 0
    iterate = toRecommendFrom
    iterate = iterate.to_numpy()
    minDist = 100000  
    for row2 in iterate:
        dist = Cosine_Dist(row2, user_Input.to_numpy())
        if dist<minDist:
            minDist=dist
            index = i    
        i=i+1
    minDistIndex.append(index) 
    
    toRecommendFrom['track_id'] = echonest.loc[:,'track_id']
    
    recommendedTrackFeatures = pd.DataFrame(toRecommendFrom.iloc[minDistIndex])
    
    recommendedTrackID = recommendedTrackFeatures.loc[:,'track_id']
   
    recommendedTrackInfo = metadata.loc[metadata['track_id'].isin(recommendedTrackID.to_numpy())]

    recommendedTrackFeatures['track_title'] = recommendedTrackInfo.loc[:,'track_title'].to_numpy()
    
    recommendedTrackFeatures['artist_name'] = recommendedTrackInfo.loc[:,'artist_name'].to_numpy()
    recommendedTrackFeatures['track_url'] = recommendedTrackInfo.loc[:,'track_url'].to_numpy()
    recommendedTrackFeatures['track_genres'] = recommendedTrackInfo.loc[:,'track_genres'].to_numpy()

    recommendedTrackFeatures['min_distance'] = minDist

    return recommendedTrackFeatures.to_dict()



def Cosine_Dist(df1, df2):
    
    return cosine(df1,df2)
