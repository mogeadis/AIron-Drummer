# =======================================================================================================
# # =====================================================================================================
# # Filename: demo.py
# #
# # Description: This script demonstrates the implemented system by generating a drum track
# #
# # Author: Alexandros Iliadis
# # Project: Music Information Retrieval Techniques for Rhythmic Drum Pattern Generation
# # Faculty: Electrical & Computer Engineering | Aristotle University Of Thessaloniki
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

# Set Current Working Directory
import os
cwd = 'C:\\Users\\mogeadis\\Documents\\Projects\\Thesis\\AIron-Drummer'
os.chdir(cwd)

# Add Modules to Path
import sys
sys.path.append(os.path.abspath('Modules'))

# Import Modules
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
model_name = 'model.h5'
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
audio = load_audio(audio_path,Fs)[0]

# Generate Drum Track
print('Generating Drum Track ...')
bars = (1,None)
drum_track = generate_drum_track(midi,audio,model,drum_parts,configuration,bars)

# Save Drum Track
print('Saving Drum Track ...')
path_name = os.path.abspath('Files')
midi_path = os.path.join(path_name,'test_drums.mid')
save_drum_track(drum_track,midi,midi_path)

# =======================================================================================================
# =======================================================================================================

# Runtime Calculation
print('\n============================================================================================')
print('Runtime: %.3f seconds' % (time.time() - start_time))
print('============================================================================================\n')