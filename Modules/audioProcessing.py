# =======================================================================================================
# # =====================================================================================================
# # Filename: audioProcessing.py
# #
# # Description: This module contains functions used to process audio data
# #
# # Author: Alexandros Iliadis
# # Project: AIron Drummer
# # Date: July 2022
# # =====================================================================================================
# =======================================================================================================

# Import Modules
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import stft,resample

# =======================================================================================================
# Function: loadAudio(...)
# Description: This function loads the samples of an audio signal
# =======================================================================================================
def loadAudio(audio_path,fs = None,mono = True,normalization = True):

    # Load Audio File
    rate,audio = read(audio_path)
    if fs == None:
        fs = rate
    else:
        num_of_samples = round(len(audio)*fs/rate)
        audio = resample(audio,num_of_samples)

    # Convert Stereo to Mono
    if mono == True:
        audio = np.mean(audio,axis = 1)

    # Normalize Audio to [-1 1]
    if normalization == True:
        audio = -1 + 2*(audio - np.min(audio,axis = 0))/np.ptp(audio,axis = 0) 

    return audio,fs

# =======================================================================================================
# Function: getAudioSegments(...)
# Description: This function splits an audio signal into segments
# =======================================================================================================
def getAudioSegments(audio,fs,measures):

    # Initialize Segments List
    audio_segments = list()

    # Measure Information
    num_of_measures = len(measures)
    measures_start_sec = measures['START_SEC'].tolist()
    measures_end_sec = measures['END_SEC'].tolist()

    # Loop Through Bars
    for bar in range(num_of_measures):

        # Segment Seconds
        start_sec = measures_start_sec[bar]
        end_sec = measures_end_sec[bar]

        # Segment Samples
        start_sample = secondsToSamples(start_sec,fs,'ceil')
        end_sample = secondsToSamples(end_sec,fs,'floor')

        # Extract Segment
        segment = audio[start_sample:end_sample+1]
        audio_segments.append(segment)

    return audio_segments

# =======================================================================================================
# Function: getSpectrogram(...)
# Description: This function calculates the spectrogram of an audio signal using the Short-Time Fourier Transform
# =======================================================================================================
def getSpectrogram(audio,fs,window_length,overlap_length,fft_length,window_type = 'hann',padding = True,start_sec = 0):

    # Padding Options
    if padding == True:
        boundary = 'zeros'
    else:
        boundary = None

    # Short-Time Fourier Transform
    f,t,s = stft(audio,fs,window_type,window_length,overlap_length,fft_length,boundary = boundary)
        
    # Time Offset
    t = t + start_sec

    return s,t,f

# =======================================================================================================
# Function: getOSS(...)
# Description: This function calculates the Onset Strength Signal of an audio signal from its spectrogram
# =======================================================================================================
def getOSS(s,t,smoothing = False,normalization = False):
    
    # Spectrogram Magnitude
    S = np.abs(s)

    # Calculate OSS
    num_of_samples = len(t)-1
    oss = np.zeros(num_of_samples)
    for n in range(num_of_samples):
        for k in range(S.shape[0]):
            fDiff = S[k,n+1]-S[k,n]
            oss[n] += fDiff*int(fDiff > 0)
    t = t[0:-1]

    # Smooth OSS
    if smoothing == True:
        oss_first = (oss[0] + oss[1])/2
        oss_mid = np.convolve(oss,np.ones(3)/3,mode = 'same')[1:-1]
        oss_last = (oss[-1] + oss[-2])/2
        oss = np.insert(oss_mid,[0,len(oss_mid)],[oss_first,oss_last])

    # Normalize OSS
    if normalization == True:
        oss = (oss-np.min(oss))/(np.max(oss)-np.min(oss))
    
    return oss,t

# =======================================================================================================
# Function: secondsToSamples(...)
# Description: This function converts seconds to samples
# =======================================================================================================
def secondsToSamples(seconds,fs,rounding = 'round'):

    # Convert Seconds to Samples
    if rounding == 'round':
        samples = int(np.round(seconds*fs))
    elif rounding == 'ceil':
        samples = int(np.ceil(seconds*fs))
    elif rounding == 'floor':
        samples = int(np.floor(seconds*fs))

    return samples