# =======================================================================================================
# # =====================================================================================================
# # Filename: preprocess_dataset.py
# #
# # Description: This script preprocesses the dataset to transform the data into the format the neural network model requires
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
print('                            Executing file: preprocess_dataset.py                             ')
print('============================================================================================\n')

# =======================================================================================================

# Import Modules
import os
import sys
sys.path.append(os.path.abspath('Modules'))
from config import *
from datasetProcessing import *
import numpy as np
import pandas as pd

# =======================================================================================================
# =======================================================================================================

# Load Dataset
print('Loading Dataset ...')

# Features
print('- Loading Features ...')
features_path = os.path.join(dataset_path,'features.csv')
features = pd.read_csv(features_path,index_col = ['ID','BAR'],usecols = ['ID','BAR'] + features_cols)

# Targets
print('- Loading Targets ...')
targets_path = os.path.join(dataset_path,'targets.csv')
targets = pd.read_csv(targets_path,index_col = ['ID','BAR'],usecols = ['ID','BAR'] + targets_cols)
print('\n============================================================================================\n')

# =======================================================================================================

# Split Dataset
print('Splitting Dataset ...')
train_features,train_targets,valid_features,valid_targets,test_features,test_targets = splitDataset(features,targets,train_ratio,valid_ratio,test_ratio)
print('\n============================================================================================\n')

# =======================================================================================================

# Prepare Dataset
print('Preparing Dataset ...')

# Training Set
print(f'- Preparing Training Set ({len(train_features.index.value_counts())} Samples) ...')
train_input,train_output = prepareDataset(train_features,train_targets)

# Validation Set
print(f'- Preparing Validation Set ({len(valid_features.index.value_counts())} Samples) ...')
valid_input,valid_output = prepareDataset(valid_features,valid_targets)

# Testing Set
print(f'- Preparing Testing Set ({len(test_features.index.value_counts())} Samples) ...')
test_input,test_output = prepareDataset(test_features,test_targets)
print('\n============================================================================================\n')

# =======================================================================================================

# Save Dataset
print('Saving Dataset ...')

# Training Set
print('- Saving Training Set ...')
if type(train_input) != type(None) and type(train_output) != type(None):
    np.save(os.path.join(dataset_path,'train_input.npy'),train_input)
    np.save(os.path.join(dataset_path,'train_output.npy'),train_output)

# Validation Set
print('- Saving Validation Set ...')
if type(valid_input) != type(None) and type(valid_output) != type(None):
    np.save(os.path.join(dataset_path,'valid_input.npy'),valid_input)
    np.save(os.path.join(dataset_path,'valid_output.npy'),valid_output)

# Testing Set
print('- Saving Testing Set ...')
if type(test_input) != type(None) and type(test_output) != type(None):
    np.save(os.path.join(dataset_path,'test_input.npy'),test_input)
    np.save(os.path.join(dataset_path,'test_output.npy'),test_output)

# =======================================================================================================
# =======================================================================================================

# Runtime Calculation
print('\n============================================================================================')
print('Runtime: %.3f seconds' % (time.time() - start_time))
print('============================================================================================\n')