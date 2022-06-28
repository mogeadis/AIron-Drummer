# =======================================================================================================
# # =====================================================================================================
# # Filename: featureExtraction.py
# #
# # Description: This module contains functions used to extract features from the dataset
# #
# # Author: Alexandros Iliadis
# # Project: Music Information Retrieval Techniques for Rhythmic Drum Pattern Generation
# # Faculty: Electrical & Computer Engineering | Aristotle University Of Thessaloniki
# # Date: July 2022
# # =====================================================================================================
# =======================================================================================================

# Import Modules
from audioProcessing import *
from measureSubdivisions import *
import numpy as np
import pandas as pd

# =======================================================================================================
# Function: get_onset_strength_signal(...)
# Description: This function calculates the Onset Strength Signal of an audio signal
# =======================================================================================================
def get_onset_strength_signal(audio,Fs,window_length,overlap_length,fft_length,winType = 'hann',padding = False,smoothing = True,normalization = True,start_sec = 0,end_sec = None):
    
    # Audio Spectrogram
    s,t,_ = get_spectrogram(audio,Fs,window_length,overlap_length,fft_length,winType,padding,start_sec)

    # Onset Strength Signal
    oss,t = get_oss(s,t,smoothing,normalization)

    # OSS Padding
    if end_sec != None:
        step = np.mean(np.diff(t))
        end_dist = end_sec - t[-1]
        while end_dist > step:
            oss = np.append(oss,0)
            t = np.append(t,t[-1]+step)
            end_dist = end_sec - t[-1]

    return oss,t

# =======================================================================================================
# Function: get_beat_proximity(...)
# Description: This function calculates the % proximity between all time steps and their nearest beat
# =======================================================================================================
def get_beat_proximity(beats,t):

    # Initialize Proximity List
    beat_proximity = list()

    # Loop Through Time Steps
    for t0 in t:

        # Calculate Proximity to Previous Beat
        proximity = [(t0 - beats[t1]) / (beats[t1+1] - beats[t1]) for t1 in range(len(beats)-1)]
        proximity = [value for value in proximity if value >= 0 and value <= 1][-1]

        # Normalize Proximity to Nearest Beat
        if proximity >= 0.5:
            proximity = 2*proximity-1
        else:
            proximity = 2*(1-proximity)-1

        # Append List
        beat_proximity.append(proximity)

    return beat_proximity

# =======================================================================================================
# Function: get_bar_progress(...)
# Description: This function calculates the % progress of all time steps in a bar
# =======================================================================================================
def get_bar_progress(start_sec,end_sec,t):

    # Calculate Bar Progress
    bar_progress = [(t0 - start_sec)/(end_sec - start_sec) for t0 in t]

    return bar_progress

# =======================================================================================================
# Function: get_bar_features(...)
# Description: This function extracts features from a single bar of an audio track
# =======================================================================================================
def get_bar_features(audio,start_sec,end_sec,num,den,configuration,return_time = False):

    # Configuration Variables
    Fs = configuration['Sampling Rate']
    winType = configuration['Window Type']
    overlap_factor = configuration['Overlap Factor']
    padding = configuration['Padding']
    smoothing = configuration['Smoothing']
    normalization = configuration['Normalization']
    note_value = configuration['Note Value']
    triplets = configuration['Allow Triplets']
    
    # Note Information 
    num_of_notes = get_num_of_notes(num,den,note_value,triplets)
    note_duration = get_note_positions(start_sec,end_sec,num_of_notes)[1]

    # Window Parameters
    window_length = seconds2samples(note_duration,Fs)
    overlap_length = window_length*overlap_factor
    fft_length = window_length

    # I) Onset Strength Signal
    oss,t = get_onset_strength_signal(audio,Fs,window_length,overlap_length,fft_length,winType,
                                      padding,smoothing,normalization,start_sec,end_sec)

    # II) Beat Proximity
    beats = get_beat_positions(start_sec,end_sec,num)[0]
    beat_proximity= get_beat_proximity(beats,t)

    # III) Bar Progress
    bar_progress = get_bar_progress(start_sec,end_sec,t)

    # Concatenate Features
    bar_features = np.asarray([oss,beat_proximity,bar_progress]).T

    # Return Time Steps
    if return_time == True:
        return bar_features,t

    return bar_features

# =======================================================================================================
# Function: get_features(...)
# Description: This function extracts features for all bars of an audio track
# =======================================================================================================
def get_features(audio,measures,configuration,features_cols):

    # Create Features DataFrame
    features = pd.DataFrame(columns = features_cols,index = pd.MultiIndex.from_tuples((),names = ['BAR','TIME']))

    # Measure Information
    num_of_measures = len(measures)
    measures_start_sec = measures['START_SEC'].tolist()
    measures_end_sec = measures['END_SEC'].tolist()
    measures_num = measures['NUM'].tolist()
    measures_den = measures['DEN'].tolist()

    # Split Audio To Segments
    Fs = configuration['Sampling Rate']
    audio_segments = get_audio_segments(audio,Fs,measures)

    # Loop Through Bars
    for bar in range(num_of_measures):

        # Bar Data
        audio_segment = audio_segments[bar]
        start_sec = measures_start_sec[bar]
        end_sec = measures_end_sec[bar]
        num = measures_num[bar]
        den = measures_den[bar]    

        # Bar Features
        bar_features,t = get_bar_features(audio_segment,start_sec,end_sec,num,den,configuration,return_time = True)
        bar_features = pd.DataFrame(data = bar_features,columns = features_cols,index = pd.MultiIndex.from_tuples(tuple(zip([bar+1]*len(t),t)),names = ['BAR','TIME']))

        # Update Features DataFrame
        features = features.append(bar_features)

    return features