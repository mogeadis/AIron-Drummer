# =======================================================================================================
# # =====================================================================================================
# # Filename: create_dataset.py
# #
# # Description: This script creates the dataset which will be used to train and evaluate the neural network model
# #
# # Author: Alexandros Iliadis
# # Project: AIron Drummer
# # Date: July 2022
# # =====================================================================================================
# =======================================================================================================

# Runtime Calculation
import time
start_time = time.time()
print('\n============================================================================================')
print('                              Executing file: create_dataset.py                               ')
print('============================================================================================\n')

# =======================================================================================================

# Import Modules
import os
import sys
sys.path.append(os.path.abspath('Modules'))

from config import *
from midiProcessing import *
from audioProcessing import *
from featureExtraction import *

import pandas as pd
from mido import MidiFile

# =======================================================================================================
# =======================================================================================================

# Create Dataset DataFrames
features = pd.DataFrame(columns = features_cols,index = pd.MultiIndex.from_tuples((),names = ['ID','BAR','TIME']))
targets = pd.DataFrame(columns = targets_cols,index = pd.MultiIndex.from_tuples((),names = ['ID','BAR','TICKS']))

# =======================================================================================================

# Access Dataset Tracklist
print('Accessing Dataset Tracklist ...','\n')
tracklist_path = os.path.join(dataset_path,'tracklist.txt')
with open(tracklist_path,'r') as tracklist:
    tracks = [track.rstrip() for track in tracklist]
tracklist.close()

# Loop Through Tracks
track_id = 0
for track in tracks:

    # Initiate Processing
    timer = time.time()
    track_id += 1
    print(f'Processing Track #{track_id}:',track)
    
    # Load MIDI File
    print('- Loading MIDI File ...')
    midi_path = os.path.join(dataset_path,'MIDI',track) + '.mid'
    midi = MidiFile(midi_path)
    measures = getMeasures(midi)

    # Load Audio File
    print('- Loading Audio File ...')
    audio_path = os.path.join(dataset_path,'Audio',track) + '.wav'
    audio = loadAudio(audio_path,fs)[0]

    # Extract Features
    print('- Extracting Features ...')
    track_features = getFeatures(audio,measures,configuration,features_cols)

    # Update Features DataFrame
    bar_index = track_features.index.get_level_values('BAR').values
    time_index = track_features.index.get_level_values('TIME').values
    track_features.index = pd.MultiIndex.from_tuples(tuple(zip([track_id]*len(track_features),bar_index,time_index)),names = ['ID','BAR','TIME'])
    features = features.append(track_features)

    # Create Targets
    print('- Creating Targets ...')
    drum_track = separateTracks(midi)[0]
    drum_track = mapNotes(drum_track,drum_map)
    drum_onsets = getDrumOnsets(drum_track,drum_parts)
    track_targets = getTargets(drum_onsets,measures,configuration,targets_cols)

    # Update Targets DataFrame
    bar_index = track_targets.index.get_level_values('BAR').values
    ticks_index = track_targets.index.get_level_values('TICKS').values
    track_targets.index = pd.MultiIndex.from_tuples(tuple(zip([track_id]*len(track_targets),bar_index,ticks_index)),names = ['ID','BAR','TICKS'])
    targets = targets.append(track_targets)

    # Conclude Processing
    print('Processing Completed in %.3f seconds' % (time.time() - timer))
    print('\n============================================================================================\n')

# =======================================================================================================

# Features
print('Features:')
print(features)
print('\n============================================================================================\n')

# Targets
print('Targets:')
print(targets)
print('\n============================================================================================\n')

# Save Dataset
print('Saving Dataset ...')
features.to_csv(os.path.join(dataset_path,'features.csv'))
targets.to_csv(os.path.join(dataset_path,'targets.csv'))

# =======================================================================================================
# =======================================================================================================

# Runtime Calculation
print('\n============================================================================================')
print('Runtime: %.3f seconds' % (time.time() - start_time))
print('============================================================================================\n')