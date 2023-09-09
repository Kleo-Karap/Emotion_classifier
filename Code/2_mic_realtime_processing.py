# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:48:15 2021

@author: user
"""

import pyaudio
# import wave
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from threading import Thread
import pickle
from sklearn.metrics import classification_report
import audio_representation as au

p = pyaudio.PyAudio()
# show devices
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    print(d)

# select device for input and output
mic_device_index = 1

WINDOW_SIZE = 2048
CHANNELS = 1
RATE = 44100

FFT_FRAMES_IN_SPEC = 20

# global
# n = np.zeros(1)
global_block = np.zeros( WINDOW_SIZE*2 )
fft_frame = np.array( WINDOW_SIZE//2 )
win = np.hamming(WINDOW_SIZE)
spec_img = np.zeros( ( WINDOW_SIZE//2 , FFT_FRAMES_IN_SPEC ) )

# keep separate audio blocks, ready to be concatenated
BLOCKS2KEEP = 20
audio_blocks = []
blocks_concatented = np.zeros( WINDOW_SIZE*BLOCKS2KEEP )

au_manager = au.AudioRepresentation()

# load model
modelfilename = 'random_forest_emotion.sav'
loaded_model = pickle.load(open(modelfilename, 'rb'))
user_terminated = False

#------------------------------------------------------------------------------------

# f = wave.open( 'audio_files/019.wav', 'rb' )


# %% call back with global

def callback( in_data, frame_count, time_info, status):
    global global_block, f, fft_frame, win, spec_img, audio_blocks
    # global_block = f.readframes(WINDOW_SIZE)
    n = np.frombuffer( in_data , dtype='int16' )
    # begin with a zero buffer
    b = np.zeros( (n.size , CHANNELS) , dtype='int16' )
    # 0 is left, 1 is right speaker / channel
    b[:,0] = n
    # for plotting
    # audio_data = np.fromstring(in_data, dtype=np.float32)
    if len(win) == len(n):
        frame_fft = np.fft.fft( win*n )
        p = np.abs( frame_fft )*2/np.sum(win)
        fft_frame = 20*np.log10( p[ :WINDOW_SIZE//2 ] / 32678 )
        spec_img = np.roll( spec_img , -1 , axis=1 )
        spec_img[:,-1] = fft_frame[::-1]
        # keep blocks
        audio_blocks.append( n )
        while len( audio_blocks ) > BLOCKS2KEEP:
            del audio_blocks[0]
    return (b, pyaudio.paContinue)

def user_input_function():
    k = input('press "s" to terminate (then press "Enter"): ')
    if k == 's' or k == 'S':
        global user_terminated
        user_terminated = True

# %% create output stream

output = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                input_device_index=mic_device_index,
                frames_per_buffer=WINDOW_SIZE,
                stream_callback=callback)

output.start_stream()

threaded_input = Thread( target=user_input_function )
threaded_input.start()

predictions=[]
classification_reports = []
true_labels=[]
# after starting, check when n empties (file ends) and stop
while output.is_active() and not user_terminated:
    # plt.clf()
    # plt.imshow( spec_img[ WINDOW_SIZE//4: , : ] , aspect='auto' )
    # # plt.axis([0,WINDOW_SIZE//8, -120,0])
    # plt.show()
    if len( audio_blocks ) == BLOCKS2KEEP:
        blocks_concatented = np.concatenate( audio_blocks ).astype(np.float32)/(2**15)
        au_manager.process_audio( blocks_concatented )
        mfcc_profile = np.vstack( au_manager.useful_mfcc_profile )
        X_in = mfcc_profile.reshape(-1,20)

        # make predictions from training data
        preds = loaded_model.predict( X_in )
        print(repr(preds[0]))
        plt.clf()
        plt.plot(mfcc_profile)
        plt.imshow(au_manager.usefull_mfcc_normalised, interpolation='hanning', cmap='twilight', origin='lower')
        title_text = 'Calm'
        if (preds[0] > 0.5):
            title_text = 'Angry'
            predictions.append(1)
        else: 
            predictions.append(0)

        #fig.suptitle(title_text, fontsize=16)

        # while evaluating -> not showing the mfcc figures
        plt.title( title_text )
        plt.show()
        # Generate classification report for specific recording
        true_label = 'Calm'  # Replace with the actual true label of  the emotion you are expressing in real-time recording
        classification_result = 'Angry' if preds[0] > 0.5 else 'Calm'  # Replace with the actual classification result
        report = classification_report([true_label], [classification_result])
        classification_reports.append(report)


    plt.pause(0.01)

print('stopping audio')
output.stop_stream()

#%%
# Save classification reports and predictions to a text file
with open('classification_results.txt', 'w') as file:
    for report, prediction in zip(classification_reports, predictions):
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\n")
        file.write("Prediction: {}\n".format("Angry" if prediction > 0.5 else "Calm"))
        file.write("-------------------------------------------\n")