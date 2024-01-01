# =======================================================================================================
# # =====================================================================================================
# # Filename: neuralNetwork.py
# #
# # Description: This module contains functions used to create, train and evaluate the neural network model
# #
# # Author: Alexandros Iliadis
# # Project: Music Information Retrieval Techniques for Rhythmic Drum Pattern Generation
# # Faculty: Electrical & Computer Engineering | Aristotle University Of Thessaloniki
# # Date: July 2022
# # =====================================================================================================
# =======================================================================================================

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Masking,LSTM,Dense
from keras.metrics import TruePositives,TrueNegatives,FalsePositives,FalseNegatives

# =============================================================================
# Function: getMaskedLoss(...)
# Description: This function creates a custom binary cross-entropy loss function which masks padded values
# =============================================================================
def getMaskedLoss(mask_value = -1):

    # Masked Loss Function
    def maskedLoss(y_true,y_pred):

        # Create Mask
        mask = K.equal(y_true,mask_value)
        mask = K.cast(mask,'float32')
        mask = 1 - mask
        
        # Calculate Binary Cross-Entropy Loss
        loss = K.binary_crossentropy(y_true,y_pred) * mask
        loss = K.sum(loss) / K.sum(mask)

        return loss

    return maskedLoss

# =============================================================================
# Function: buildModel(...)
# Description: This function builds and compiles the neural network model
# =============================================================================
def buildModel(input_size,output_size,num_of_units,dropout = 0,mask_value = -1):

    # Encoder Input Layer
    encoder_Input = Input(shape = (None,input_size),name = 'encoder_Input')
    # Encoder Masking Layer
    encoder_Masking = Masking(mask_value = mask_value,name = 'encoder_Masking')(encoder_Input)
    # Encoder LSTM Layer
    encoder_LSTM = LSTM(num_of_units,return_sequences = False,return_state = True,dropout = dropout,name = 'encoder_LSTM')
    _,state_h,state_c = encoder_LSTM(encoder_Masking)
    encoder_states = [state_h,state_c]

    # Decoder Input Layer
    decoder_Input = Input(shape = (None,output_size),name = 'decoder_Input')
    # Decoder Masking Layer
    decoder_Masking = Masking(mask_value = mask_value,name = 'decoder_Masking')(decoder_Input)
    # Decoder LSTM Layer
    decoder_LSTM = LSTM(num_of_units,return_sequences = True,return_state = True,dropout = dropout,name = 'decoder_LSTM')
    decoder_outputs,_,_ = decoder_LSTM(decoder_Masking,initial_state = encoder_states)
    # Decoder Dense Layer
    decoder_Dense = Dense(output_size,activation = 'sigmoid',name = 'decoder_Dense')
    decoder_Output = decoder_Dense(decoder_outputs)

    # Build & Compile Model
    model = Model([encoder_Input, decoder_Input],decoder_Output,name = 'Seq2Seq')
    model.compile(loss = getMaskedLoss(mask_value),optimizer = 'adam',
                  metrics = [TruePositives(name = 'TP'),TrueNegatives(name = 'TN'),
                             FalsePositives(name = 'FP'),FalseNegatives(name = 'FN')])

    return model

# =============================================================================
# Function: trainModel(...)
# Description: This function trains and validates the neural network model
# =============================================================================
def trainModel(model,train_input,train_output,valid_input = None,valid_output = None,batch_size = 64,epochs = 100,save_period = None,save_history = False):

    # Training Data
    train_input_encoder = train_input
    train_input_decoder = train_output[:,:-1,:]
    train_output_decoder = train_output[:,1:,:]
    x_train = [train_input_encoder,train_input_decoder]
    y_train = train_output_decoder

    # Validation Data
    if type(valid_input) == type(None) or type(valid_output) == type(None):
        xy_valid = None
    else:
        valid_input_encoder = valid_input
        valid_input_decoder = valid_output[:,:-1,:]
        valid_output_decoder = valid_output[:,1:,:]
        x_valid = [valid_input_encoder,valid_input_decoder]
        y_valid = valid_output_decoder
        xy_valid = (x_valid,y_valid)
        
    # Periodic Model Saving
    num_of_samples = train_input.shape[0]
    steps_per_epoch = int(np.ceil(num_of_samples/batch_size))
    if save_period == None:
        save_period = epochs
    save_freq = save_period * steps_per_epoch

    # Model Training
    history = model.fit(x_train,y_train,validation_data = xy_valid,
                        callbacks = [ModelCheckpoint('model{epoch:d}.h5',save_freq = save_freq)],
                        batch_size = batch_size,epochs = epochs,verbose = 1)
    history = history.history

    # Training Metrics
    loss = history['loss']
    TP = history['TP']
    TN = history['TN']
    FP = history['FP']
    FN = history['FN']

    # Training Classification Metrics
    recall = [0 if (tp+fn) == 0 else tp/(tp+fn) for tp,fn in zip(TP,FN)]
    precision = [0 if (tp+fp) == 0 else tp/(tp+fp) for tp,fp in zip(TP,FP)]
    f1_score = [0 if (r+p) == 0 else 2*r*p/(r+p) for r,p in zip(recall,precision)]

    # Training Similarity Metric
    jaccard_index = [tp/(tp+fp+fn) for tp,fp,fn in zip(TP,FP,FN)]

    # Training History
    train_history = {'Loss' : loss,
                       'TP' : TP,
                       'TN' : TN,
                       'FP' : FP,
                       'FN' : FN,
                   'Recall' : recall,
                'Precision' : precision,
                       'F1' : f1_score,
                  'Jaccard' : jaccard_index}

    # Model Validation
    if type(valid_input) == type(None) or type(valid_output) == type(None):
        val_history = {}
    else:
        # Validation Metrics
        val_loss = history['val_loss']
        val_TP = history['val_TP']
        val_TN = history['val_TN']
        val_FP = history['val_FP']
        val_FN = history['val_FN']

        # Validation Classification Metrics
        val_recall = [0 if (tp+fn) == 0 else tp/(tp+fn) for tp,fn in zip(val_TP,val_FN)]
        val_precision = [0 if (tp+fp) == 0 else tp/(tp+fp) for tp,fp in zip(val_TP,val_FP)]
        val_f1_score = [0 if (r+p) == 0 else 2*r*p/(r+p) for r,p in zip(val_recall,val_precision)]

        # Validation Similarity Metric
        val_jaccard_index = [tp/(tp+fp+fn) for tp,fp,fn in zip(val_TP,val_FP,val_FN)]
        
        # Validation History
        val_history = {'Val_Loss' : val_loss,
                         'Val_TP' : val_TP,
                         'Val_TN' : val_TN,
                         'Val_FP' : val_FP,
                         'Val_FN' : val_FN,
                     'Val_Recall' : val_recall,
                  'Val_Precision' : val_precision,
                         'Val_F1' : val_f1_score,
                    'Val_Jaccard' : val_jaccard_index}

    # Merge History
    history = dict(train_history,**val_history)

    # Save History
    if save_history == True:
        np.save('history.npy',history)

    return history

# =============================================================================
# Function: testModel(...)
# Description: This function tests the neural network model
# =============================================================================
def testModel(model,test_input,test_output,save_results = False):

    # Testing Data
    if type(test_input) == type(None) or type(test_output) == type(None):
        results = None
    else:
        test_input_encoder = test_input
        test_input_decoder = test_output[:,:-1,:]
        test_output_decoder = test_output[:,1:,:]
        x_test = [test_input_encoder,test_input_decoder]
        y_test = test_output_decoder

        # Model Testing
        results = model.evaluate(x_test,y_test,verbose = 0)

        # Testing Metrics
        loss = results[0]
        TP = results[1]
        TN = results[2]
        FP = results[3]
        FN = results[4]

        # Testing Classification Metrics
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1_score = 2*recall*precision/(recall+precision)

        # Testing Similarity Metric
        jaccard_index = TP/(TP+FP+FN)

        # Testing Results
        results = {'Loss' : loss,
                     'TP' : TP,
                     'TN' : TN,
                     'FP' : FP,
                     'FN' : FN,
                 'Recall' : recall,
              'Precision' : precision,
                     'F1' : f1_score,
                'Jaccard' : jaccard_index}

    # Save Results
    if save_results == True:
        np.save('results.npy',results)

    return results

# =============================================================================
# Function: plotHistory(...)
# Description: This function plots the training history of the neural network model
# =============================================================================
def plotHistory(history,validation = True,epochs = None):

    # Epochs to Plot
    total_epochs = len(history[list(history)[0]])
    t = range(1,total_epochs+1)
    if epochs == None:
        epochs = total_epochs

    # Binary Cross-Entropy Loss
    fig,ax = plt.subplots()
    ax.plot(t,history['Loss'],label = 'Training',color = '#2246E9')
    if validation == True:
        ax.plot(t,history['Val_Loss'],label = 'Validation',color = '#E92222')
    ax.set_xlim(1,epochs)
    ax.set_ylim(0,0.1)         
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.grid(visible = True)
    ax.legend(loc = 'upper right')
    title = 'Binary Cross-Entropy Loss'
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)

    # Training Fundamental Classification Metrics
    fig,ax = plt.subplots()
    ax.plot(t,history['TP'],label = 'True Positives',color = '#2246E9')
    ax.plot(t,history['FN'],label = 'False Negatives',color = '#E92222')
    ax.plot(t,history['FP'],label = 'False Positives',color = '#863486')
    ax.set_xlim(1,epochs)
    ax.set_yticklabels([])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Count')
    ax.grid(visible = True)
    ax.legend(loc = 'upper right')
    title = 'Training | Fundamental Classification Metrics'
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)

    # Training Derived Classification Metrics
    fig,ax = plt.subplots()
    ax.plot(t,history['Recall'],label = 'Recall',color = '#2246E9')
    ax.plot(t,history['Precision'],label = 'Precision',color = '#E92222')
    ax.plot(t,history['F1'],label = 'F1-Score',color = '#863486')
    ax.set_xlim(1,epochs)
    ax.set_ylim(0,1)         
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.grid(visible = True)
    ax.legend(loc = 'upper right')
    title = 'Training | Derived Classification Metrics'
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)

    if validation == True:
        # Validation Fundamental Classification Metrics
        fig,ax = plt.subplots()
        ax.plot(t,history['Val_TP'],label = 'True Positives',color = '#2246E9')
        ax.plot(t,history['Val_FN'],label = 'False Negatives',color = '#E92222')
        ax.plot(t,history['Val_FP'],label = 'False Positives',color = '#863486')
        ax.set_xlim(1,epochs)
        ax.set_yticklabels([])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Count')
        ax.grid(visible = True)
        ax.legend(loc = 'upper right')
        title = 'Validation | Fundamental Classification Metrics'
        ax.set_title(title)
        fig.canvas.manager.set_window_title(title)

        # Validation Derived Classification Metrics
        fig,ax = plt.subplots()
        ax.plot(t,history['Val_Recall'],label = 'Recall',color = '#2246E9')
        ax.plot(t,history['Val_Precision'],label = 'Precision',color = '#E92222')
        ax.plot(t,history['Val_F1'],label = 'F1-Score',color = '#863486')
        ax.set_xlim(1,epochs)
        ax.set_ylim(0,1)            
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        ax.grid(visible = True)
        ax.legend(loc = 'upper right')
        title = 'Validation | Derived Classification Metrics'
        ax.set_title(title)
        fig.canvas.manager.set_window_title(title)

    # Jaccard Index
    fig,ax = plt.subplots()
    ax.plot(t,history['Jaccard'],label = 'Training',color = '#2246E9')
    if validation == True:
        ax.plot(t,history['Val_Jaccard'],label = 'Validation',color = '#E92222')
    ax.set_xlim(1,epochs)
    ax.set_ylim(0,1)         
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.grid(visible = True)
    ax.legend(loc = 'upper right')
    title = 'Jaccard Index'
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)

# =============================================================================
# Function: buildEncoderDecoder(...)
# Description: This function builds the encoder and decoder models that are used during inference
# =============================================================================
def buildEncoderDecoder(model):

    # Encoder Input
    encoder_Input = model.get_layer('encoder_Input').output
    # Encoder States
    _,state_h,state_c = model.get_layer('encoder_LSTM').output
    encoder_states = [state_h,state_c]
    # Encoder Model
    encoder = Model(encoder_Input,encoder_states,name = 'Encoder')

    # Decoder State Inputs
    decoder_state_input_h = Input(shape = (state_h.shape[-1]),name = 'decoder_Input_h')
    decoder_state_input_c = Input(shape = (state_c.shape[-1]),name = 'decoder_Input_c')
    decoder_state_inputs = [decoder_state_input_h,decoder_state_input_c]
    # Decoder Input
    decoder_Input = model.get_layer('decoder_Input').output
    # Decoder States
    decoder_outputs,state_h,state_c = model.get_layer('decoder_LSTM')(decoder_Input,initial_state = decoder_state_inputs)
    decoder_states = [state_h,state_c]
    # Decoder Outputs
    decoder_outputs = model.get_layer('decoder_Dense')(decoder_outputs)
    # Decoder Model
    decoder = Model([decoder_Input] + decoder_state_inputs,[decoder_outputs] + decoder_states,name = 'Decoder')

    return encoder,decoder

# =============================================================================
# Function: inference(...)
# Description: This function uses the encoder and the decoder to predict the output sequence during inference
# =============================================================================
def inference(encoder,decoder,input_sequence,output_shape):

    # Encode Input Sequence
    decoder_states_input = encoder.predict(input_sequence,verbose = 0)

    # Generate Start-Of-Sequence Token
    num_of_targets = output_shape[1]
    decoder_inputs = np.asarray([1] + num_of_targets*[0]).astype('float32').reshape(1,1,num_of_targets+1)

    # Initialize Output Sequence
    output_sequence = np.empty((0,num_of_targets + 1))

    # Loop Through Time Steps
    num_of_steps = output_shape[0]
    for _ in range(num_of_steps):

        # Predict Step Output
        decoder_outputs,state_h,state_c = decoder.predict([decoder_inputs] + decoder_states_input,verbose = 0)

        # Update Output Sequence
        output_sequence = np.concatenate([output_sequence,decoder_outputs[0,0,:].reshape(1,num_of_targets+1)])

        # Update Decoder Inputs
        decoder_inputs = decoder_outputs
        decoder_states_input = [state_h,state_c]

    # Discard Start-Of-Sequence Token Column
    output_sequence = output_sequence[:,1:]

    return output_sequence