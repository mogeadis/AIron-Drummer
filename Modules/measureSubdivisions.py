# =======================================================================================================
# # =====================================================================================================
# # Filename: measureSubdivisions.py
# #
# # Description: This module contains functions that are used to divide a measure into notes and beats
# #
# # Author: Alexandros Iliadis
# # Project: Music Information Retrieval Techniques for Rhythmic Drum Pattern Generation
# # Faculty: Electrical & Computer Engineering | Aristotle University Of Thessaloniki
# # Date: July 2022
# # =====================================================================================================
# =======================================================================================================

# Import Modules
import numpy as np

# =======================================================================================================
# Function: getNumOfNotes(...)
# Description: This function calculates the maximum number of discrete notes in a measure
# =======================================================================================================
def getNumOfNotes(num,den,note_value,triplets):

    # Calculate Number of Notes
    num_of_notes = (1/note_value) * (num/den)
    if triplets == True:
        num_of_notes = 3*num_of_notes
    num_of_notes = int(num_of_notes)

    return num_of_notes

# =======================================================================================================
# Function: getNotePositions(...)
# Description: This function calculates the time positions where discrete notes can occur in a measure
# =======================================================================================================
def getNotePositions(start_t,end_t,num_of_notes,rounding = False):

    # Calculate Note Positions and Note Duration
    note_positions,note_duration = np.linspace(start_t,end_t,num_of_notes,endpoint = False,retstep = True)

    # Rounding
    if rounding == True:
        note_positions = note_positions.round().astype('int')
        note_duration = note_duration.round().astype('int')

    return note_positions,note_duration

# =======================================================================================================
# Function: getBeatPositions(...)
# Description: This function calculates the time positions where beats and offbeats occur in a measure
# =======================================================================================================
def getBeatPositions(start_t,end_t,num):

    # Calculate Beats
    beats,beat_duration = np.linspace(start_t,end_t,num+1,endpoint = True,retstep = True)

    # Calculate Offbeats
    offbeats = beats[0:-1] + beat_duration/2

    return beats,offbeats