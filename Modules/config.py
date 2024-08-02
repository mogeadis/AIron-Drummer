# =======================================================================================================
# # =====================================================================================================
# # Filename: config.py
# #
# # Description: This module configures various constant variables which are used in this project
# #
# # Author: Alexandros Iliadis
# # Project: AIron Drummer
# # Date: July 2022
# # =====================================================================================================
# =======================================================================================================

# Dataset Directory
dataset_path = 'C:\\Users\\mogeadis\\Projects\\GitHub\\AIron-Drummer\\Dataset'

# =======================================================================================================

# Audio & MIDI Processing Configuration
fs = 8000
configuration = {'Sampling Rate' : fs,
                   'Window Type' : 'hann',
                'Overlap Factor' : 2/3,
                       'Padding' : False,
                     'Smoothing' : True,
                 'Normalization' : True,
                    'Note Value' : 1/16,
                'Allow Triplets' : True}

# =======================================================================================================

# General MIDI Percussion Keys
drum_parts = {'KICK' : 36,
             'SNARE' : 38,
             'CRASH' : 49,
              'RIDE' : 51,
           'HH_OPEN' : 46,
         'HH_CLOSED' : 42,
           'LOW_TOM' : 43,
           'MID_TOM' : 47,
          'HIGH_TOM' : 48}
drum_map = {35 : drum_parts['KICK'],
            36 : drum_parts['KICK'],
            37 : drum_parts['SNARE'],
            38 : drum_parts['SNARE'],
            39 : drum_parts['SNARE'],
            40 : drum_parts['SNARE'],
            41 : drum_parts['LOW_TOM'],
            42 : drum_parts['HH_CLOSED'],
            43 : drum_parts['LOW_TOM'],
            44 : drum_parts['HH_CLOSED'],
            45 : drum_parts['MID_TOM'],
            46 : drum_parts['HH_OPEN'],
            47 : drum_parts['MID_TOM'],
            48 : drum_parts['HIGH_TOM'],
            49 : drum_parts['CRASH'],
            50 : drum_parts['HIGH_TOM'],
            51 : drum_parts['RIDE'],
            52 : drum_parts['CRASH'],
            53 : drum_parts['RIDE'],
            54 : drum_parts['HH_CLOSED'],
            55 : drum_parts['CRASH'],
            56 : drum_parts['RIDE'],
            57 : drum_parts['CRASH'],
            58 : drum_parts['CRASH'],
            59 : drum_parts['RIDE']}

# =======================================================================================================

# DataFrame Columns
features_cols = ['OSS','PROX','PROG']
targets_cols = list(drum_parts.keys())

# =======================================================================================================

# Random Seed
from random import seed
seed(666)

# =======================================================================================================

# Dataset Ratios
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# =======================================================================================================

# Warning Suppression
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'