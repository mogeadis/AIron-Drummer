# =======================================================================================================
# # =====================================================================================================
# # Filename: datasetProcessing.py
# #
# # Description: This module contains functions that are used to process the created dataset
# #
# # Author: Alexandros Iliadis
# # Project: AIron Drummer
# # Date: July 2022
# # =====================================================================================================
# =======================================================================================================

# Import Modules
import numpy as np
import pandas as pd
from random import shuffle
from scipy.linalg import block_diag

# =============================================================================
# Function: splitDataset(...)
# Description: This function splits the dataset into training, validation and testing sets
# =============================================================================
def splitDataset(features,targets,train_ratio,valid_ratio,test_ratio):

    # Create Training DataFrames
    train_features = pd.DataFrame(columns = features.columns,index = pd.MultiIndex.from_tuples((),names = features.index.names))
    train_targets = pd.DataFrame(columns = targets.columns,index = pd.MultiIndex.from_tuples((),names = targets.index.names))

    # Create Validation DataFrames
    valid_features = pd.DataFrame(columns = features.columns,index = pd.MultiIndex.from_tuples((),names = features.index.names))
    valid_targets = pd.DataFrame(columns = targets.columns,index = pd.MultiIndex.from_tuples((),names = targets.index.names))
    
    # Create Testing DataFrames
    test_features = pd.DataFrame(columns = features.columns,index = pd.MultiIndex.from_tuples((),names = features.index.names))
    test_targets = pd.DataFrame(columns = targets.columns,index = pd.MultiIndex.from_tuples((),names = targets.index.names))

    # Approximate Samples Per Set
    total_samples = len(targets.index.value_counts())
    train_samples = round(train_ratio * total_samples)
    valid_samples = round(valid_ratio * total_samples)
    test_samples = round(test_ratio * total_samples)

    # Correct Sum of Samples
    sum_of_samples = train_samples + valid_samples + test_samples
    while  sum_of_samples < total_samples:
        train_samples += 1
        sum_of_samples = train_samples + valid_samples + test_samples
    while sum_of_samples > total_samples:
        train_samples -= 1
        sum_of_samples = train_samples + valid_samples + test_samples
 
    # Initialize Sample Counters
    train_samples_counter = 0
    valid_samples_counter = 0
    test_samples_counter = 0

    # Shuffle Track IDs
    ids = features.index.get_level_values('ID').values.tolist()
    ids = [id for i,id in enumerate(ids) if i == 0 or id != ids[i-1]]
    shuffle(ids)

    # Loop Through IDs
    for id in ids:

        # Training Samples
        if train_samples_counter < train_samples:

            # Update Features DataFrame
            temp_features = features.loc[id,:]
            temp_features.index = pd.MultiIndex.from_tuples(tuple(zip([id]*len(temp_features),temp_features.index.values)),names = ['ID','BAR'])
            train_features = train_features.append(temp_features)

            # Update Targets DataFrame
            temp_targets = targets.loc[id,:]
            temp_targets.index = pd.MultiIndex.from_tuples(tuple(zip([id]*len(temp_targets),temp_targets.index.values)),names = ['ID','BAR'])
            train_targets = train_targets.append(temp_targets)
            
            # Update Counter
            train_samples_counter = len(train_features.index.value_counts())
            continue

        # Validation Samples
        if valid_samples_counter < valid_samples:

            # Update Features DataFrame
            temp_features = features.loc[id,:]
            temp_features.index = pd.MultiIndex.from_tuples(tuple(zip([id]*len(temp_features),temp_features.index.values)),names = ['ID','BAR'])
            valid_features = valid_features.append(temp_features)

            # Update Targets DataFrame
            temp_targets = targets.loc[id,:]
            temp_targets.index = pd.MultiIndex.from_tuples(tuple(zip([id]*len(temp_targets),temp_targets.index.values)),names = ['ID','BAR'])
            valid_targets = valid_targets.append(temp_targets)
            
            # Update Counter
            valid_samples_counter = len(valid_features.index.value_counts())
            continue

        # Testing Samples
        if test_samples_counter < test_samples:

            # Update Features DataFrame
            temp_features = features.loc[id,:]
            temp_features.index = pd.MultiIndex.from_tuples(tuple(zip([id]*len(temp_features),temp_features.index.values)),names = ['ID','BAR'])
            test_features = test_features.append(temp_features)

            # Update Targets DataFrame
            temp_targets = targets.loc[id,:]
            temp_targets.index = pd.MultiIndex.from_tuples(tuple(zip([id]*len(temp_targets),temp_targets.index.values)),names = ['ID','BAR'])
            test_targets = test_targets.append(temp_targets)

            # Update Counter
            test_samples_counter = len(test_features.index.value_counts())
            continue

    # Sort Training DataFrame Indices
    train_features.sort_index(level = ['ID','BAR'],inplace = True) 
    train_targets.sort_index(level = ['ID','BAR'],inplace = True)

    # Sort Validation DataFrame Indices
    valid_features.sort_index(level = ['ID','BAR'],inplace = True) 
    valid_targets.sort_index(level = ['ID','BAR'],inplace = True)

    # Sort Testing DataFrame Indices
    test_features.sort_index(level = ['ID','BAR'],inplace = True) 
    test_targets.sort_index(level = ['ID','BAR'],inplace = True)

    # Return
    return train_features,train_targets,valid_features,valid_targets,test_features,test_targets

# =============================================================================
# Function: prepareDataset(...)
# Description: This function prepares the dataset for use with a neural network
# =============================================================================
def prepareDataset(features,targets,mask_value = -1,sos_token = True):

    # Process Empty DataFrames
    if features.empty == True or targets.empty == True:
        return None,None
    
    # Initialize Input Data Array
    num_of_input_steps = features.index.value_counts(sort = True,ascending = False)[0]
    num_of_features = len(features.columns)
    input_data = np.empty(shape = (num_of_input_steps,num_of_features))

    # Initialize Output Data Array
    num_of_output_steps = targets.index.value_counts(sort = True,ascending = False)[0]
    num_of_targets = len(targets.columns)
    if sos_token == True:
        num_of_targets += 1
        num_of_output_steps +=1
    output_data = np.empty(shape = (num_of_output_steps,num_of_targets))

    # Loop Through IDs & Bars
    index = -1
    ids = features.index.get_level_values('ID').values.tolist()
    ids = [id for i,id in enumerate(ids) if i == 0 or id != ids[i-1]]
    bars = features.index.get_level_values('BAR').values.tolist()
    bars = [bar for i,bar in enumerate(bars) if i == 0 or bar != bars[i-1]]
    for bar in bars:

        # Proceed to Next ID
        if bar == 1:
            index += 1
            id = ids[index]

        # Retrieve Input Sample
        input_sample = features.loc[id,bar].to_numpy()

        # Pad Input Sample
        pad_length = num_of_input_steps - len(input_sample)
        input_sample = np.pad(input_sample,[(0,pad_length),(0,0)],mode = 'constant',constant_values = mask_value)

        # Update Input Data Array
        input_data = np.dstack((input_data,input_sample))  

        # Retrieve Output Sample
        output_sample = targets.loc[id,bar].to_numpy()
        if sos_token == True:
            output_sample = block_diag([1],output_sample)

        # Pad Output Sample
        pad_length = num_of_output_steps - len(output_sample)
        output_sample = np.pad(output_sample,[(0,pad_length),(0,0)],mode = 'constant',constant_values = mask_value)

        # Update Output Data Array
        output_data = np.dstack((output_data,output_sample))
    
    # Reshape Input Data Array
    input_data = np.delete(input_data,0,axis = -1).astype('float32')
    input_data = np.moveaxis(input_data,[0, 1, 2],[1, 2, 0])

    # Reshape Output Data Array
    output_data = np.delete(output_data,0,axis = -1).astype('float32')
    output_data = np.moveaxis(output_data,[0, 1, 2],[1, 2, 0])

    # Return
    return input_data,output_data