# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:16:44 2021

@author: user
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

class AudioRepresentation:
    
    def __init__(self, sr=44100, n_fft=2048, hop_length=1024):
        # analysis-related constants
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        # load file
        # s, sr = librosa.load( file_path , self.sr)
        # self.audio = s
        self.audio = None
        '''
        self.extract_power_spectrum()
        # identify areas where speech is present
        self.make_useful_audio_mask()
        self.make_useful_spectrum()
        self.make_useful_area_features()
        # keep name for further processing
        # self.name = file_path.split( '\\' )[-1]
        if not keep_audio:
            del self.audio
        if not keep_aux:
            del self.power_spectrum
            del self.useful_spectrum
            del self.spectral_magnitude
            del self.useful_bandwidth
            del self.useful_centroid
            del self.useful_mask
        '''
    # end constructor
    
    def process_audio(self, s):
        self.audio = s
        self.extract_power_spectrum()
        # identify areas where speech is present
        self.make_useful_audio_mask()
        self.make_useful_spectrum()
        self.make_useful_area_features()
    
    def extract_power_spectrum(self):
        p = librosa.stft(self.audio, n_fft=2048, hop_length=1024)
        self.spectral_magnitude, _ = librosa.magphase(p)
        self.power_spectrum = librosa.amplitude_to_db( np.abs(p), ref=np.max )
    # end extract_power_spectrum
    
    def make_useful_audio_mask(self):
        self.rms = librosa.feature.rms(S=self.spectral_magnitude)
        rms = self.rms[0]
        self.useful_mask = np.zeros( rms.size )
        self.useful_mask[ rms > 0.001 ] = 1
        self.useful_mask = self.useful_mask.astype(int)
    # end extract_power_spectrum
    
    def make_useful_spectrum(self):
        self.useful_spectrum = self.power_spectrum[:,self.useful_mask == 1]
    # end extract_power_spectrum
    
    def plot_spectrum(self, range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
    # end plot_spectrum
    
    def plot_save_spectrum(self, figure_file_name='test.png', range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
        plt.savefig( figure_file_name , dpi=300 )
    # plot_save_spectrum
    
    def plot_useful_spectrum(self, range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_useful_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
    # end plot_spectrum
    
    def plot_save_useful_spectrum(self, figure_file_name='test.png', range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_useful_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
        plt.savefig( figure_file_name , dpi=300 )
    # end plot_save_spectrum
    
    def make_useful_area_features(self):
        # centroid
        c = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_centroid = c[0][ self.useful_mask == 1 ]
        # self.mean_centroid = np.mean( self.useful_centroid )
        # self.std_centroid = np.std( self.useful_centroid )
        # bandwidth
        b = librosa.feature.spectral_bandwidth(y=self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_bandwidth = b[0][ self.useful_mask == 1 ]
        # self.mean_bandwidth = np.mean( self.useful_bandwidth )
        # self.std_bandwidth = np.std( self.useful_bandwidth )
        # mfccs
        m = librosa.feature.mfcc( y=self.audio , sr=self.sr )
        self.useful_mfcc = m
        self.usefull_mfcc_normalised = (m-np.min(m))/(np.max(m)-np.min(m))
        self.useful_mfcc_profile = np.mean( self.usefull_mfcc_normalised , axis=1 )

    # end make_useful_area_features
# end class AudioFileRepresentation