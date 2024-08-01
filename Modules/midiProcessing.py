# =======================================================================================================
# # =====================================================================================================
# # Filename: midiProcessing.py
# #
# # Description: This module contains functions used to process MIDI data
# #
# # Author: Alexandros Iliadis
# # Project: Music Information Retrieval Techniques for Rhythmic Drum Pattern Generation
# # Faculty: Electrical & Computer Engineering | Aristotle University Of Thessaloniki
# # Date: July 2022
# # =====================================================================================================
# =======================================================================================================

# Import Modules
import numpy as np
import pandas as pd
from mido import MidiFile,MidiTrack,Message,MetaMessage
from measureSubdivisions import *
from audioProcessing import getAudioSegments
from featureExtraction import getBarFeatures
from neuralNetwork import buildEncoderDecoder,inference

# =======================================================================================================
# Function: getTempoEvents(...)
# Description: This function returns all tempo change events that occur in a MIDI file
# =======================================================================================================
def getTempoEvents(midi):

    # Initialize Data Lists
    tempo_events_sec = list()
    tempo_events_ticks = list()
    tempo_events_bpm = list()
    tempo_events_uspb = list()

    # Ticks Per Beat
    tpb = midi.ticks_per_beat

    # Loop Flag
    flag = False

    # Loop Through MIDI Tracks
    midi_tracks = midi.tracks
    num_of_tracks = len(midi_tracks)
    for id in range(num_of_tracks):

        # Select Track
        track = midi_tracks[id]
        
        # Initialize Time Counters
        last_ticks = 0
        total_ticks = 0
        total_seconds = 0
        
        # Loop Through MIDI messages
        num_of_msgs = len(track)
        for msg in range(num_of_msgs):

            # Update Tick Counters
            last_ticks += track[msg].time
            total_ticks += track[msg].time

            # Find 'Set Tempo' Meta Messages
            if track[msg].is_meta == True:
                if track[msg].type == 'set_tempo':

                    # Update Seconds Counter
                    if len(tempo_events_sec) != 0:
                        total_seconds += ticksToSeconds(tpb,bpm,last_ticks)

                    # Update Seconds List
                    tempo_events_sec.append(total_seconds)

                    # Update Ticks List
                    tempo_events_ticks.append(total_ticks)

                    # Update Microseconds Per Beat List
                    uspb = track[msg].tempo
                    tempo_events_uspb.append(uspb)

                    # Update Beats Per Minute List
                    bpm = uspbToBPM(uspb)
                    tempo_events_bpm.append(bpm)

                    # Reset Tick Counter
                    last_ticks = 0

                    # Update Flag
                    flag = True

        # Break Search Loop
        if flag == True:
            break

    # Detect False Events
    false_events = list()
    num_of_events = len(tempo_events_ticks)
    for event in range(num_of_events-1):
        if tempo_events_bpm[event] == tempo_events_bpm[event+1]:
            false_events.append(event+1)

    # Discard False Events
    tempo_events_sec = [tempo_events_sec[event] for event in range(num_of_events) if event not in false_events]
    tempo_events_ticks = [tempo_events_ticks[event] for event in range(num_of_events) if event not in false_events]
    tempo_events_bpm = [tempo_events_bpm[event] for event in range(num_of_events) if event not in false_events]
    tempo_events_uspb = [tempo_events_uspb[event] for event in range(num_of_events) if event not in false_events]

    # Create Tempo Events DataFrame
    tempo_events = pd.DataFrame({'SEC' : tempo_events_sec,
                               'TICKS' : tempo_events_ticks,
                                 'BPM' : tempo_events_bpm,
                                'USPB' : tempo_events_uspb})

    return tempo_events

# =======================================================================================================
# Function: getTimeSignatureEvents(...)
# Description: This function returns all time signature change events that occur in a MIDI file
# =======================================================================================================
def getTimeSignatureEvents(midi):

    # Get Tempo Events
    tempo_events = getTempoEvents(midi)
    tempo_events_sec = tempo_events['SEC'].tolist()
    tempo_events_ticks = tempo_events['TICKS'].tolist()
    tempo_events_bpm = tempo_events['BPM'].tolist()

    # Initialize Data Lists
    time_signature_events_sec = list() #Seconds
    time_signature_events_ticks = list() #Ticks
    time_signature_events_num = list() #Numerator
    time_signature_events_den = list() #Denominator

    # Ticks Per Beat
    tpb = midi.ticks_per_beat

    # Loop Flag
    flag = False

    # Loop Through MIDI Tracks
    midi_tracks = midi.tracks
    num_of_tracks = len(midi_tracks)
    for id in range(num_of_tracks):

        # Select Track
        track = midi_tracks[id]
        
        # Initialize Time Counters
        total_ticks = 0
        total_seconds = 0

        # Loop Through MIDI Messages
        num_of_msgs = len(track)
        for msg in range(num_of_msgs):

            # Update Tick Counter
            total_ticks += track[msg].time

            # Find 'Time Signature' Meta Messages
            if track[msg].is_meta == True:
                if track[msg].type == 'time_signature':

                    # Update Seconds Counter
                    if len(time_signature_events_sec) != 0:
                        last_tempo_event_id = len([(total_ticks - ticks) for ticks in tempo_events_ticks if (total_ticks - ticks) > 0])-1
                        bpm = tempo_events_bpm[last_tempo_event_id]
                        last_ticks = total_ticks - tempo_events_ticks[last_tempo_event_id]
                        total_seconds = tempo_events_sec[last_tempo_event_id] + ticksToSeconds(tpb,bpm,last_ticks)

                    # Update Seconds List
                    time_signature_events_sec.append(total_seconds)
                    
                    # Update Ticks List
                    time_signature_events_ticks.append(total_ticks)

                    # Update Numerator List
                    num = track[msg].numerator
                    time_signature_events_num.append(num)

                    # Update Denominator List
                    den = track[msg].denominator
                    time_signature_events_den.append(den)

                    # Update Flag
                    flag = True

        # Break Search Loop
        if flag == True:
            break

    # Detect False Events
    false_events = list()
    num_of_events = len(time_signature_events_ticks)
    for event in range(num_of_events-1):
        if time_signature_events_num[event] == time_signature_events_num[event+1] and time_signature_events_den[event] == time_signature_events_den[event+1]:
            false_events.append(event+1)

    # Discard False Events
    time_signature_events_sec = [time_signature_events_sec[event] for event in range(num_of_events) if event not in false_events]
    time_signature_events_ticks = [time_signature_events_ticks[event] for event in range(num_of_events) if event not in false_events]
    time_signature_events_num = [time_signature_events_num[event] for event in range(num_of_events) if event not in false_events]
    time_signature_events_den = [time_signature_events_den[event] for event in range(num_of_events) if event not in false_events]

    # Create Time Signature Events DataFrame
    time_signature_events = pd.DataFrame({'SEC' : time_signature_events_sec,
                                        'TICKS' : time_signature_events_ticks,
                                          'NUM' : time_signature_events_num,
                                          'DEN' : time_signature_events_den})
    
    return time_signature_events

# =======================================================================================================
# Function: getEvents(...)
# Description: This function returns all tempo and time signature change events that occur in a MIDI file
# =======================================================================================================
def getEvents(midi):

    # Get Tempo Events
    tempo_events = getTempoEvents(midi)
    tempo_events_sec = tempo_events['SEC'].tolist()
    tempo_events_ticks = tempo_events['TICKS'].tolist()
    tempo_events_bpm = tempo_events['BPM'].tolist()
    tempo_events_uspb = tempo_events['USPB'].tolist()

    # Get Time Signature Events
    time_signature_events = getTimeSignatureEvents(midi)
    time_signature_events_sec = time_signature_events['SEC'].tolist()
    time_signature_events_ticks = time_signature_events['TICKS'].tolist()
    time_signature_events_num = time_signature_events['NUM'].tolist()
    time_signature_events_den = time_signature_events['DEN'].tolist()

    # Initialize Data Lists
    events_sec = [round(seconds,3) for seconds in (tempo_events_sec + time_signature_events_sec)]
    events_ticks = tempo_events_ticks + time_signature_events_ticks
    events_bpm = list()
    events_uspb = list()
    events_num = list()
    events_den = list()

    # Discard Duplicate Values
    events_sec = list(dict.fromkeys(events_sec))
    events_ticks = list(dict.fromkeys(events_ticks))

    # Sort Lists
    events_sec.sort()
    events_ticks.sort()

    # Loop Through Events
    for total_ticks in events_ticks:

        # Last Tempo & Time Signature Events
        last_tempo_event_id = len([(total_ticks - ticks) for ticks in tempo_events_ticks if (total_ticks - ticks) >= 0])-1
        last_time_signature_event_id = len([(total_ticks - ticks) for ticks in time_signature_events_ticks if (total_ticks - ticks) >= 0])-1

        # Update Beats Per Minute List
        events_bpm.append(tempo_events_bpm[last_tempo_event_id])

        # Update Microseconds Per Beat List
        events_uspb.append(tempo_events_uspb[last_tempo_event_id])

        # Update Numerator List
        events_num.append(time_signature_events_num[last_time_signature_event_id])

        # Update Denominator List
        events_den.append(time_signature_events_den[last_time_signature_event_id])

    # Create Events DataFrame
    events = pd.DataFrame({'SEC' : events_sec,
                         'TICKS' : events_ticks,
                           'BPM' : events_bpm,
                          'USPB' : events_uspb,
                           'NUM' : events_num,
                           'DEN' : events_den})

    return events

# =======================================================================================================
# Function: getFileEndTicks(...)
# Description: This function returns the number of ticks at which a MIDI file ends
# =======================================================================================================
def getFileEndTicks(midi):

    # Initialize Variable
    file_end_ticks = 0

    # Loop Through MIDI Tracks
    midi_tracks = midi.tracks
    num_of_tracks = len(midi_tracks)
    for id in range(num_of_tracks):

        # Select Track
        track = midi_tracks[id]
        
        # Initialize Tick Counter
        total_ticks = 0

        # Loop Through MIDI messages
        num_of_msgs = len(track)
        for msg in range(num_of_msgs):

            # Update Tick Counter
            total_ticks += track[msg].time

            # Find 'End of Track' Meta Messages
            if track[msg].is_meta == True:
                if track[msg].type == 'end_of_track':
                    if total_ticks >= file_end_ticks:
                        file_end_ticks = total_ticks

    return file_end_ticks

# =======================================================================================================
# Function: getMeasures(...)
# Description: This function splits a MIDI file into music measures
# =======================================================================================================
def getMeasures(midi):

    # Get Tempo & Time Signature Events
    events = getEvents(midi)
    num_of_events = len(events)
    events_ticks = events['TICKS'].tolist()
    events_bpm = events['BPM'].tolist()
    events_uspb = events['USPB'].tolist()
    events_num = events['NUM'].tolist()
    events_den = events['DEN'].tolist()  
    
    # Get End of File Ticks
    file_end_ticks = getFileEndTicks(midi)

    # Initialize Data Lists
    measures_bar = list()
    measures_start_sec = list() 
    measures_end_sec = list()
    measures_start_ticks = list()
    measures_end_ticks = list()
    measures_bpm = list()
    measures_uspb = list()
    measures_num = list()
    measures_den = list()

    # Initialize Counters
    bar = 0;
    event = 0
    end_sec = 0
    end_ticks = 0

    # Ticks Per Beat
    tpb = midi.ticks_per_beat

    # Loop Through the Whole MIDI File
    while end_ticks < file_end_ticks:

        # Proceed to Next Bar
        bar += 1
        measures_bar.append(bar)

        # Proceed to Next Event
        if event < (num_of_events - 1):
            if end_ticks == events_ticks[event+1]:
                event += 1

        # Update Tempo Lists
        bpm = events_bpm[event]
        uspb = events_uspb[event]
        measures_bpm.append(bpm)
        measures_uspb.append(uspb)

        # Update Time Signature Lists
        num = events_num[event]
        den = events_den[event]
        measures_num.append(num)
        measures_den.append(den)

        # Update Ticks Lists
        start_ticks = end_ticks
        end_ticks = start_ticks + round(4*tpb*num/den)
        measures_start_ticks.append(start_ticks)
        measures_end_ticks.append(end_ticks)

        # Update Seconds Lists
        start_sec = end_sec
        end_sec = start_sec + ticksToSeconds(tpb,bpm,(end_ticks - start_ticks))
        measures_start_sec.append(start_sec)
        measures_end_sec.append(end_sec)        

    # Create Measures DataFrame
    measures = pd.DataFrame({'START_SEC' : measures_start_sec,
                               'END_SEC' : measures_end_sec,
                           'START_TICKS' : measures_start_ticks,
                             'END_TICKS' : measures_end_ticks,
                                   'BPM' : measures_bpm,
                                  'USPB' : measures_uspb,
                                   'NUM' : measures_num,
                                   'DEN' : measures_den},
                                   index = pd.Index(data = measures_bar,name = 'BAR'))

    return measures

# =======================================================================================================
# Function: uspbToBPM(...)
# Description: This function converts microseconds per beat to beats per minute
# =======================================================================================================
def uspbToBPM(uspb):

    #Convert USPB to BPM
    bpm = round(60*(10**6)/uspb)
    
    return bpm

# =======================================================================================================
# Function: ticksToSeconds(...)
# Description: This function converts MIDI ticks to seconds
# =======================================================================================================
def ticksToSeconds(tpb,bpm,ticks):

    # Convert Ticks to Seconds
    seconds = 60/(tpb*bpm)*ticks

    return seconds
    
#########################################################################################################

# =======================================================================================================
# Function: getDrumTrackID(...)
# Description: This function returns the drum track index of a MIDI file
# =======================================================================================================
def getDrumTrackID(midi):

    # Initialize Drum Track Index
    drum_track_id = None

    # Loop Flag
    flag = False

    # Loop Through MIDI Tracks
    midi_tracks = midi.tracks
    num_of_tracks = len(midi_tracks)
    for id in range(num_of_tracks):

        # Select Track
        track = midi_tracks[id] 

        # Loop Through MIDI Messages
        num_of_msgs = len(track)
        for msg in range(num_of_msgs):

            # Find Drum Track Index
            if track[msg].is_meta == False:
                if track[msg].channel == 9:
                    drum_track_id = id
                    flag = True
                break
        
        # Break Search Loop
        if flag == True:
            break

    return drum_track_id

# =======================================================================================================
# Function: separateTracks(...)
# Description: This function separates the drum track from the other tracks of a MIDI file
# =======================================================================================================
def separateTracks(midi):

    # Initialize Track Lists
    drum_track = list()
    other_tracks = midi.tracks[:]
    
    # Get Drum Track Index
    drum_track_id = getDrumTrackID(midi)

    # Extract Drum Track
    if drum_track_id != None:
        drum_track = other_tracks.pop(drum_track_id)

    return drum_track,other_tracks

# =======================================================================================================
# Function: mapNotes(...)
# Description: This function maps General MIDI notes to selected drum parts
# =======================================================================================================
def mapNotes(drum_track,drum_map):

    # Loop Through MIDI Messages
    num_of_msgs = len(drum_track)
    for msg in range(num_of_msgs):

        # Find 'note_on' or 'note_off' Messages
        if drum_track[msg].type == 'note_on' or drum_track[msg].type == 'note_off':

            # Map Drum Notes
            if drum_map.get(drum_track[msg].note) != None:
                drum_track[msg].note = drum_map.get(drum_track[msg].note)
            else:
                drum_track[msg].note = 0

    return drum_track

# =======================================================================================================
# Function: getDrumOnsets(...)
# Description: This function detects drum onsets of selected drum parts
# =======================================================================================================
def getDrumOnsets(drum_track,drum_parts):

    # List of Drum Parts
    drum_part_names = list(drum_parts.keys())
    num_of_drum_parts = len(drum_part_names)

    # Initialize Drum Onsets Lists
    drum_part_onsets = [[] for _ in range(num_of_drum_parts)]

    # Initialize Onset Ticks List
    onset_ticks = [-1]

    # Initialize Tick Counter
    total_ticks = 0

    # Loop Through MIDI Messages
    num_of_msgs = len(drum_track)
    for msg in range(num_of_msgs):

        # Update Tick Counter
        total_ticks += drum_track[msg].time

        # Find 'note_on' Messages
        if drum_track[msg].type == 'note_on' and drum_track[msg].note != 0:

            # Initialize Onset State 
            if onset_ticks[-1] != total_ticks:
                onset_ticks.append(total_ticks)
                for part in range(num_of_drum_parts):
                    drum_part_onsets[part].append(0)

            # Update Onset State
            for part in range(num_of_drum_parts):
                if drum_track[msg].note == drum_parts[drum_part_names[part]]:
                    drum_part_onsets[part][-1] = 1

    # Create Drum Onsets DataFrame
    drum_onsets = pd.DataFrame(data = drum_part_onsets).transpose()
    drum_onsets.index = pd.Index(data = onset_ticks[1:],name = 'TICKS' )
    drum_onsets.columns = drum_part_names

    return drum_onsets

# =======================================================================================================
# Function: getDrumSegments(...)
# Description: This function splits drum onsets into segments
# =======================================================================================================
def getDrumSegments(drum_onsets,measures):

    # Initialize Segments List
    drum_segments = list()

    # Measure Information
    num_of_measures = len(measures)
    measures_start_ticks = measures['START_TICKS'].tolist()
    measures_end_ticks = measures['END_TICKS'].tolist()

    # Loop Through Bars
    for bar in range(num_of_measures):

        # Segment Ticks
        start_ticks = measures_start_ticks[bar]
        end_ticks = measures_end_ticks[bar]

        # Extract Segment
        segment = drum_onsets.loc[start_ticks:end_ticks-1]
        drum_segments.append(segment)

    return drum_segments

# =======================================================================================================
# Function: quantizeDrumOnsets(...)
# Description: This function quantizes drum onsets to note positions
# =======================================================================================================
def quantizeDrumOnsets(drum_onsets,note_positions):

    # Create Quantized Drum Onsets DataFrame
    quant_drum_onsets= pd.DataFrame(0,columns = drum_onsets.columns,index = pd.Index(data = note_positions,name = 'TICKS'))

    # Loop Through Onset Ticks
    onset_ticks = drum_onsets.index.values
    for t in onset_ticks:

        # Find Closest Note Position
        offsets = [abs(t - t0) for t0 in note_positions]
        note_position = note_positions[offsets.index(min(offsets))]

        # Update Quantized Drum Onsets DataFrame
        quant_drum_onsets.loc[note_position] += drum_onsets.loc[t].values

    # Normalize to {0,1}
    quant_drum_onsets = quant_drum_onsets.ge(1).astype(int)

    return quant_drum_onsets

# =======================================================================================================
# Function: getBarTargets(...)
# Description: This function creates target variables for a single bar of a drum track
# =======================================================================================================
def getBarTargets(drum_onsets,start_ticks,end_ticks,num,den,configuration,return_ticks = False):

    # Configuration Variables
    note_value = configuration['Note Value']
    triplets = configuration['Allow Triplets']

    # Note Information
    num_of_notes = getNumOfNotes(num,den,note_value,triplets)
    note_positions = getNotePositions(start_ticks,end_ticks,num_of_notes,rounding = True)[0]

    # Quantize Drum Onsets
    bar_targets = quantizeDrumOnsets(drum_onsets,note_positions)

    # Return Ticks
    if return_ticks == True:
        ticks = bar_targets.index.values
        return bar_targets,ticks

    return bar_targets

# =======================================================================================================
# Function: getTargets(...)
# Description: This function creates target variables for all bars of a drum track
# =======================================================================================================
def getTargets(drum_onsets,measures,configuration,targets_cols):

    # Create Targets DataFrame
    targets = pd.DataFrame(columns = targets_cols,index = pd.MultiIndex.from_tuples((),names = ['BAR','TICKS']))

    # Measure Information
    num_of_measures = len(measures)
    measures_start_ticks = measures['START_TICKS'].tolist()
    measures_end_ticks = measures['END_TICKS'].tolist()
    measures_num = measures['NUM'].tolist()
    measures_den = measures['DEN'].tolist()

    # Split Drum Onsets to Segments
    drum_segments = getDrumSegments(drum_onsets,measures)

    # Loop Through Bars
    for bar in range(num_of_measures):

        # Bar Data
        drum_segment = drum_segments[bar]
        start_ticks = measures_start_ticks[bar]
        end_ticks = measures_end_ticks[bar]
        num = measures_num[bar]
        den = measures_den[bar]
        
        # Bar Targets
        bar_targets,ticks = getBarTargets(drum_segment,start_ticks,end_ticks,num,den,configuration,return_ticks = True)
        bar_targets.index = pd.MultiIndex.from_tuples(tuple(zip([bar+1]*len(ticks),ticks)),names = ['BAR','TICKS'])
        
        # Update Targets DataFrame
        targets = targets.append(bar_targets)

    return targets

#########################################################################################################

# =======================================================================================================
# Function: getTicksIndex(...)
# Description: This function returns the index necessary to generate a drum track
# =======================================================================================================
def getTicksIndex(measures,note_value,triplets):

    # Initialize Index Lists
    bars = list()
    ticks = list()

    # Measure Information
    num_of_measures = len(measures)
    measures_start_ticks = measures['START_TICKS'].tolist()
    measures_end_ticks = measures['END_TICKS'].tolist()
    measures_num = measures['NUM'].tolist()
    measures_den = measures['DEN'].tolist()

    # Loop Through Bars
    for bar in range(num_of_measures):
        
        # Bar Data
        start_ticks = measures_start_ticks[bar]
        end_ticks = measures_end_ticks[bar]
        num = measures_num[bar]
        den = measures_den[bar]

        # Note Information
        num_of_notes = getNumOfNotes(num,den,note_value,triplets)
        note_positions = getNotePositions(start_ticks,end_ticks,num_of_notes,rounding = True)[0]
        note_positions = (note_positions-measures_start_ticks[0]).tolist()

        # Update Index Lists
        bars += [bar+1]*len(note_positions)
        ticks += note_positions

    # Create Ticks Index
    ticks_index = pd.MultiIndex.from_tuples(tuple(zip(bars,ticks)),names = ['BAR','TICKS'])

    return ticks_index

# =============================================================================
# Function: filterOnsets(...)
# Description: This function filters drum onsets based on physical and logical playing restrictions
# =============================================================================
def filterOnsets(drum_onsets):

    # Apply Restrictions
    drum_onsets.loc[(drum_onsets['SNARE'] == 1) & (drum_onsets['KICK'] == 1),'KICK'] = 0
    drum_onsets.loc[(drum_onsets['CRASH'] == 1) & (drum_onsets['RIDE'] == 1),'RIDE'] = 0
    drum_onsets.loc[(drum_onsets['CRASH'] == 1) & (drum_onsets['HH_OPEN'] == 1),'HH_OPEN'] = 0
    drum_onsets.loc[(drum_onsets['CRASH'] == 1) & (drum_onsets['HH_CLOSED'] == 1),'HH_CLOSED'] = 0
    drum_onsets.loc[(drum_onsets['RIDE'] == 1) & (drum_onsets['HH_OPEN'] == 1),'HH_OPEN'] = 0
    drum_onsets.loc[(drum_onsets['RIDE'] == 1) & (drum_onsets['HH_CLOSED'] == 1),'HH_CLOSED'] = 0
    drum_onsets.loc[(drum_onsets['HH_OPEN'] == 1) & (drum_onsets['HH_CLOSED'] == 1),'HH_CLOSED'] = 0

    # Find Candidate Onsets
    onset_sums = drum_onsets.sum(axis = 1)
    candidates = drum_onsets.loc[onset_sums.loc[onset_sums >= 3].index]

    # Loop Through Candidates
    num_of_candidates = len(candidates)
    for id in range(num_of_candidates):

        # Select Candidate
        candidate = candidates.iloc[id]

        # Define Available Onsets
        if candidate['KICK'] == 1:
            onsets_available = 3
        else:
            onsets_available = 2

        # Loop Through Drum Parts
        for part in candidates.columns:

            # Remove Onset
            if onsets_available == 0:
                candidate[part] = 0
            
            # Update Available Onsets
            if candidate[part] == 1:
                onsets_available -= 1

    # Filter Candidate Onsets
    drum_onsets.loc[candidates.index] = candidates

    return drum_onsets

# =======================================================================================================
# Function: createDrumTrack(...)
# Description: This function creates a MIDI drum track
# =======================================================================================================
def createDrumTrack(drum_onsets,drum_parts,file_end_ticks):

    # Set Constants
    drum_channel = 9
    note_velocity = 90
    
    # Create MIDI track
    track = MidiTrack()
    track.append(MetaMessage('track_name',name = 'Drum Track',time = 0))
    track.append(Message('program_change',channel = drum_channel,program = 0,time = 0))

    # List of Drum Parts
    drum_part_names = list(drum_parts.keys())
    num_of_drum_parts = len(drum_part_names)

    # Initialize Variables
    ticks = 0
    note_states = [False] * num_of_drum_parts

    # Abstract Drum Onsets 
    drum_onsets.reset_index('BAR',drop = True,inplace = True)
    drum_onsets = drum_onsets.loc[(drum_onsets!=0).any(axis=1)]

    # Loop Through Onset Ticks
    onset_ticks = drum_onsets.index.values
    for t in onset_ticks:

        # Calculate Delta Time
        delta = t - ticks

        # Loop Through Drum Parts
        for part in range(num_of_drum_parts):

            # Add 'note_off' Messages
            if note_states[part] == True:
                track.append(Message('note_off',channel = drum_channel,note = drum_parts[drum_part_names[part]],velocity = note_velocity,time = delta))
                note_states[part] = False
                delta = 0

            # Add 'note_on' Messages
            if drum_onsets.loc[t,drum_part_names[part]] == 1:
                track.append(Message('note_on',channel = drum_channel,note = drum_parts[drum_part_names[part]],velocity = note_velocity,time = delta))
                note_states[part] = True
                delta = 0

        # Update Ticks
        ticks = t

    # Add Final 'note_off' messages
    t = file_end_ticks
    delta = t-ticks
    for part in range(num_of_drum_parts):
        if note_states[part] == True:
            track.append(Message('note_off',channel = drum_channel,note = drum_parts[drum_part_names[part]],velocity = note_velocity,time = delta))
            delta = 0

    # Add 'End of Track' Meta Message
    track.append(MetaMessage('end_of_track', time = delta))

    return track

# =======================================================================================================
# Function: generateDrumTrack(...)
# Description: This function generates a novel MIDI drum track
# =======================================================================================================
def generateDrumTrack(midi,audio,model,drum_parts,configuration,bars = (None,None)):

    # Build Encoder & Decoder
    encoder,decoder = buildEncoderDecoder(model)

    # List of Drum Parts
    drum_part_names = list(drum_parts.keys())
    num_of_drum_parts = len(drum_part_names)

    # Initialize Predictions Array
    predictions = np.empty((0,num_of_drum_parts))

    # Configuration Variables
    fs = configuration['Sampling Rate']
    note_value = configuration['Note Value']
    triplets = configuration['Allow Triplets']

    # Bar Selection
    measures = getMeasures(midi)
    start_bar = bars[0] if bars[0] != None else 1
    end_bar = bars[1] if bars[1] != None else len(measures)
    measures = measures.loc[start_bar:end_bar]

    # Measure Information
    num_of_measures = len(measures)
    measures_start_sec = measures['START_SEC'].tolist()
    measures_end_sec = measures['END_SEC'].tolist()
    measures_start_ticks = measures['START_TICKS'].tolist()
    measures_end_ticks = measures['END_TICKS'].tolist()
    measures_num = measures['NUM'].tolist()
    measures_den = measures['DEN'].tolist() 

    # Split Audio to Segments
    audio_segments = getAudioSegments(audio,fs,measures)

    # Loop Through Bars
    for bar in range(num_of_measures):
        print(f'\r--> Bar {bar+1}/{num_of_measures}',end = '')

        # Bar Data
        audio_segment = audio_segments[bar]
        start_sec = measures_start_sec[bar]
        end_sec = measures_end_sec[bar]
        num = measures_num[bar]
        den = measures_den[bar]
        
        # Bar Features
        bar_features = getBarFeatures(audio_segment,start_sec,end_sec,num,den,configuration)

        # Input Sequence
        input_sequence = np.expand_dims(bar_features.astype('float32'),axis = 0)

        # Output Shape
        num_of_notes = getNumOfNotes(num,den,note_value,triplets)
        output_shape = (num_of_notes,num_of_drum_parts)

        # Output Sequence
        output_sequence = inference(encoder,decoder,input_sequence,output_shape)

        # Update Predictions Array
        predictions = np.concatenate([predictions,output_sequence])
    print('')

    # Round Predictions to {0,1}
    predictions = np.round(predictions).astype(int)

    # Create Drum Onsets DataFrame
    drum_onsets = pd.DataFrame(predictions,columns = drum_part_names,index = getTicksIndex(measures,note_value,triplets))

    # Filter Drum Onsets
    drum_onsets = filterOnsets(drum_onsets)

    # Create Drum Track
    file_end_ticks =  measures_end_ticks[-1] - measures_start_ticks[0]
    drum_track = createDrumTrack(drum_onsets,drum_parts,file_end_ticks)

    return drum_track

# =======================================================================================================
# Function: createEventsTrack(...)
# Description: This function creates a MIDI track with all tempo and time signature change messages
# =======================================================================================================
def createEventsTrack(midi):

    # Get Events
    events = getEvents(midi)

    # Initialize Variables
    uspb = 0
    num = 0
    den = 0
    ticks = 0

    # Create MIDI Track
    events_track = MidiTrack()
    events_track.append(MetaMessage('track_name',name = 'Drum Track',time = 0))

    # Loop Through Events
    num_of_events = len(events)
    for index in range(num_of_events):

        # Calculate Delta Time
        delta = events.loc[index,'TICKS'] - ticks

        # Add 'Set Tempo' Meta Messages
        if events.loc[index,'USPB'] != uspb:
            uspb = events.loc[index,'USPB']
            events_track.append(MetaMessage('set_tempo',tempo = uspb,time = delta))
            delta = 0
        
        # Add 'Time Signature' Meta Messages
        if events.loc[index,'NUM'] != num or events.loc[index,'DEN'] != den:
            num = events.loc[index,'NUM']
            den = events.loc[index,'DEN']
            events_track.append(MetaMessage('time_signature',numerator = num,denominator = den,clocks_per_click = 24,notated_32nd_notes_per_beat = 8,time = delta))
            delta = 0

        # Update Ticks
        ticks = events.loc[index,'TICKS'] 

    # Add 'End of Track' Meta Message
    events_track.append(MetaMessage('end_of_track',time = delta))

    return events_track

# =======================================================================================================
# Function: saveMidi(...)
# Description: This function saves a MIDI file
# =======================================================================================================
def saveMidi(midi_path,midi_type,midi_tpb,midi_tracks):

    # Create MIDI File
    midi = MidiFile()
    midi.type = midi_type
    midi.ticks_per_beat = midi_tpb

    # Add MIDI Tracks
    num_of_tracks = len(midi_tracks)
    for id in range(num_of_tracks):
        midi.tracks.append(midi_tracks[id])

    # Save MIDI File
    midi.save(midi_path)

# =======================================================================================================
# Function: saveDrumTrack(...)
# Description: This function saves a drum track into a MIDI file
# =======================================================================================================
def saveDrumTrack(drum_track,midi,midi_path):

    # Create Events Track
    events_track = createEventsTrack(midi)

    # MIDI Data
    midi_type = midi.type
    midi_tpb = midi.ticks_per_beat
    midi_tracks = [events_track,drum_track]

    # Save MIDI File
    saveMidi(midi_path,midi_type,midi_tpb,midi_tracks)