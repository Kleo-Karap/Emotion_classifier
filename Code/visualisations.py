# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 12:01:27 2023

@author: kleop
"""
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_pickle('C:/Users/kleop/Documents/repos/Ergasia/Code/prepared_dataframe.pickle')
act01 = df[ df['emotion'] == 'calm' ]
act02 = df[ df['emotion'] == 'angry' ]
folder_for_figs = 'calm_angry'

# Mean Centroid - Calm/Angry
plt.clf()
plt.plot(df["mean_centroid"][df["emotion"]=="calm"], "bx", label="Calm")
plt.plot(df["mean_centroid"][df["emotion"]=="angry"], "r+", label="Angry")
plt.legend()
plt.savefig('figs/' + folder_for_figs + '/mean_centroid.png', dpi=300)

# STD Centroid - Calm/Angry
plt.clf()
plt.plot(df["std_centroid"][df["emotion"]=="calm"], "bx", label="Calm")
plt.plot(df["std_centroid"][df["emotion"]=="angry"], "r+", label="Angry")
plt.legend()
plt.savefig('figs/' + folder_for_figs + '/std_centroid.png', dpi=300)

#mean f0
plt.clf()
plt.plot(df["mean_f0"][df["emotion"]=="calm"], "bx", label="Calm")
plt.plot(df["mean_f0"][df["emotion"]=="angry"], "r+", label="Angry")
plt.legend()
plt.savefig('figs/' + folder_for_figs + '/mean_f0.png', dpi=300)
#std f0
plt.clf()
plt.plot(df["std_f0"][df["emotion"]=="calm"], "bx", label="Calm")
plt.plot(df["std_f0"][df["emotion"]=="angry"], "r+", label="Angry")
plt.legend()
plt.savefig('figs/' + folder_for_figs + '/std_f0.png', dpi=300)

#%% Plot the MFCCs with dimensionality reduction
from sklearn.decomposition import PCA
import numpy as np
act01_features = np.vstack( act01['mfcc_profile'].to_numpy() )
act02_features = np.vstack( act02['mfcc_profile'].to_numpy() )
all_features = np.vstack((act01_features, act02_features))


pca = PCA(n_components=2)
pca_features = np.vstack( all_features )
all_pca = pca.fit_transform( np.vstack( pca_features ) )

plt.clf()
plt.plot( all_pca[:act01_features.shape[0], 0] , all_pca[:act01_features.shape[0], 1] , 'bx', alpha=0.8,label="Calm" )
plt.plot( all_pca[act01_features.shape[0]:, 0] , all_pca[act01_features.shape[0]:, 1] , 'r+', alpha=0.8, label="Angry" )
plt.savefig('figs/' + folder_for_figs + '/mfcc_pca.png', dpi=300)
