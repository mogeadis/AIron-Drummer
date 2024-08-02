# =======================================================================================================
# # =====================================================================================================
# # Filename: demo.py
# #
# # Description: This script demonstrates the implemented system by generating a drum track
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
print('                                    Executing file: demo.py                                   ')
print('============================================================================================\n')

# =======================================================================================================

# Import Modules
import os
import sys
sys.path.append(os.path.abspath('Modules'))

from config import *
from midiProcessing import *
from audioProcessing import *

from mido import MidiFile
from keras.models import load_model

# =======================================================================================================
# =======================================================================================================

# Load Model
print('Loading Model ...')
path_name = os.path.abspath('Files')
model_name = 'test_model.h5'
model_path = os.path.join(path_name,model_name)
model = load_model(model_path,compile = False)

# Load MIDI File
print('Loading MIDI File ...')
path_name = os.path.abspath('Files')
midi_name = 'test_midi.mid'
midi_path = os.path.join(path_name,midi_name)
midi = MidiFile(midi_path)

# Load Audio File
print('Loading Audio File ...')
path_name = os.path.abspath('Files')
audio_name = 'test_audio.wav'
audio_path = os.path.join(path_name,audio_name)
audio = loadAudio(audio_path,fs)[0]

# Generate Drum Track
print('Generating Drum Track ...')
drum_track = generateDrumTrack(midi,audio,model,drum_parts,configuration)

# Save Drum Track
print('Saving Drum Track ...')
path_name = os.path.abspath('Files')
midi_path = os.path.join(path_name,'test_drums.mid')
saveDrumTrack(drum_track,midi,midi_path)

# =======================================================================================================
# =======================================================================================================

# Runtime Calculation
print('\n============================================================================================')
print('Runtime: %.3f seconds' % (time.time() - start_time))
print('============================================================================================\n')