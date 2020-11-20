# -*- coding: utf-8 -*-
"""
Project: "Surprising Minds" at Sea Life Brighton, by Danbee Kim, Kerry Perkins, Clive Ramble, Hazel Garnade, Goncalo Lopes, Dario Quinones, Reanna Campbell-Russo, Robb Barrett, Martin Stopps, The EveryMind Team, and Adam Kampff. 
Analysis: Measure speed of pupil

Loads daily .npz files with x position, y position, size, and size baseline data.
Calculate movement from one frame to the next and find movement peaks (saccades).
Split into calibration, unique, and octopus sequences.

@author: Adam R Kampff and Danbee Kim
"""
import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from scipy.signal import savgol_filter
from itertools import groupby
from operator import itemgetter
from scipy.signal import find_peaks
import csv
import logging

###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# FUNCTIONS
###################################

##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) dataset_dir (parent folder with intermediate files)
##########################################################
def load_data():
    dataset_dir = r'D:\data\SurprisingMinds\intermediates'
    return dataset_dir
##########################################################

def threshold_to_nan(input_array, threshold, upper_or_lower):
    for index in range(len(input_array)): 
        if upper_or_lower=='upper':
            if np.isnan(input_array[index])==False and input_array[index]>threshold:
                input_array[index] = np.nan
        if upper_or_lower=='lower':
            if np.isnan(input_array[index])==False and input_array[index]<threshold:
                input_array[index] = np.nan
    return input_array

def filter_to_nan(list_of_dicts, upper_threshold, lower_threshold):
    for dictionary in list_of_dicts:
        for key in dictionary:
            for trial in dictionary[key]:
                trial = threshold_to_nan(trial, upper_threshold, 'upper')
                trial = threshold_to_nan(trial, lower_threshold, 'lower')
    return list_of_dicts

def make_luminance_time_buckets(start_timestamp, bucket_size_ms, end_timestamp): 
    start_timestamp = start_timestamp.split('+')[0][:-3]
    end_timestamp = end_timestamp.split('+')[0][:-3]
    buckets_start_time = datetime.datetime.strptime(start_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
    buckets_end_time = datetime.datetime.strptime(end_timestamp, "%Y-%m-%dT%H:%M:%S.%f")

    current_bucket = buckets_start_time
    time_buckets = []
    window = datetime.timedelta(milliseconds=bucket_size_ms)
    while current_bucket <= buckets_end_time:
        time_buckets.append(current_bucket)
        current_bucket = current_bucket + window

    bucket_list = dict.fromkeys(time_buckets)

    for key in time_buckets: 
        bucket_list[key] = [-5]
    # -5 remains in a time bucket, this means no 'near-enough timestamp' frame was found in video

    return bucket_list

def find_nearest_timestamp_key(timestamp_to_check, dict_of_timestamps, time_window):
    for key in dict_of_timestamps.keys():
        if key <= timestamp_to_check <= (key + time_window):
            return key

def build_timebucket_avg_luminance(timestamps_and_luminance_array, bucket_size_ms, max_no_of_timebuckets):
    bucket_window = datetime.timedelta(milliseconds=bucket_size_ms)
    max_no_of_timebuckets = int(max_no_of_timebuckets)
    avg_luminance_by_timebucket = []
    index = 0
    for trial in timestamps_and_luminance_array:
        first_timestamp = trial[0][0]
        end_timestamp = trial[-1][0]
        this_trial_timebuckets = make_luminance_time_buckets(first_timestamp, bucket_size_ms, end_timestamp)
        this_trial = np.empty(max_no_of_timebuckets)
        this_trial[:] = np.nan
        for frame in trial:
            timestamp = frame[0]
            lum_val = int(frame[1])
            timestamp = timestamp.split('+')[0][:-3]
            timestamp_dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
            this_bucket = find_nearest_timestamp_key(timestamp_dt, this_trial_timebuckets, bucket_window)
            if this_trial_timebuckets[this_bucket] == [-5]:
                this_trial_timebuckets[this_bucket] = [lum_val]
            else:
                this_trial_timebuckets[this_bucket].append(lum_val)
        sorted_keys = sorted(list(this_trial_timebuckets.keys()))
        key_index = 0
        for key in sorted_keys:
            avg_luminance_for_this_bucket = np.mean(this_trial_timebuckets[key])
            this_trial[key_index] = avg_luminance_for_this_bucket
            key_index = key_index + 1
        avg_luminance_by_timebucket.append(this_trial)
        index = index + 1
    avg_lum_by_tb_thresholded = []
    for lum_array in avg_luminance_by_timebucket:
        lum_array_thresholded = threshold_to_nan(lum_array, 0, 'lower')
        avg_lum_by_tb_thresholded.append(lum_array_thresholded)
    avg_lum_by_tb_thresh_array = np.array(avg_lum_by_tb_thresholded)
    avg_lum_final = np.nanmean(avg_lum_by_tb_thresh_array, axis=0)
    return avg_lum_final

def find_windowed_peaks(time_bucket_dict, window, threshold):
    windowed_peaks = {}
    key_list = []
    for ptime in time_bucket_dict.keys():
        key_list.append(ptime)
    key_list.sort()
    for k,g in groupby(enumerate(key_list), lambda ix: ix[0] - ix[1]):
        consecutive_ptimes = list(map(itemgetter(1), g))
        #print(consecutive_ptimes)
        if len(consecutive_ptimes)<=window:
            max_val = threshold
            this_group_count = 0
            for time in consecutive_ptimes:
                this_group_count = this_group_count + time_bucket_dict[time]
            if this_group_count>max_val:
                max_time = np.median(consecutive_ptimes)
                windowed_peaks[int(max_time)] = this_group_count
        else:
            max_val = threshold
            max_times = {}
            for time in consecutive_ptimes:
                center = time
                start = int(center-(window/2))
                end = int(center+(window/2))
                this_group_count = 0
                for t in range(int(start),int(end)):
                    this_group_count = this_group_count + time_bucket_dict.get(t,0)
                if this_group_count>max_val:
                    if not max_times:
                        max_times[center] = this_group_count
                        max_val = this_group_count
                    else:
                        overlap = [x for x in max_times.keys() if start<x<end]
                        filtered_overlap = {}
                        for o in overlap:
                            temp_val = max_times.pop(o)
                            if temp_val>this_group_count:
                                filtered_overlap[o] = [temp_val]
                        if not filtered_overlap:
                            max_times[center] = this_group_count
                            max_val = this_group_count
                        else:
                            for f in filtered_overlap.items():
                                max_times[f[0]] = f[1]
            for max_time in max_times.keys():
                windowed_peaks[max_time] = max_times[max_time]
    return windowed_peaks

def calc_mvmnt_from_pos(list_of_positon_arrays, nans_threshold, movement_threshold_upper, movement_threshold_lower):
    this_stim_movements = []
    for trial in list_of_positon_arrays:
        trial_movements_min_len = len(trial)
        this_trial_movement = []
        nans_in_a_row = 0
        prev = np.nan
        for i in range(len(trial)):
            now = trial[i]
            #print("now: "+str(now))
            #print("prev: "+str(prev))
            if np.isnan(now):
                # keep the nan to understand where the dropped frames are
                this_trial_movement.append(np.nan)
                nans_in_a_row = nans_in_a_row + 1
                continue
            if nans_in_a_row>(nans_threshold):
                break
            if i==0:
                this_trial_movement.append(0)
                prev = now
                continue
            if not np.isnan(prev):
                movement = now - prev
                this_trial_movement.append(movement)
                prev = now
                nans_in_a_row = 0 
            #print("movements: " + str(this_trial_movement))
            #print("consecutive nans: " + str(nans_in_a_row))
        # filter out movements too large to be realistic saccades (120 pixels)
        trial_movement_array = np.array(this_trial_movement)
        trial_movement_array = threshold_to_nan(trial_movement_array, movement_threshold_upper, 'upper')
        trial_movement_array = threshold_to_nan(trial_movement_array, movement_threshold_lower, 'lower')
        this_stim_movements.append(trial_movement_array)  
    # filter for trial movements that are less than 4000 bins long
    output = [x for x in this_stim_movements if len(x)>=trial_movements_min_len]
    return output

def calc_avg_motion_and_peaks(list_of_movement_arrays, window):
    total_motion = np.zeros(len(list_of_movement_arrays[0]))
    nan_count = np.zeros(len(list_of_movement_arrays[0]))
    # for each frame, sum the abs(movements) on that frame
    for trial in list_of_movement_arrays:
        for t in range(len(trial)):
            if np.isnan(trial[t]):
                nan_count[t] = nan_count[t] + 1
            if not np.isnan(trial[t]):
                total_motion[t] = total_motion[t] + abs(trial[t])
    avg_motion = np.zeros(len(list_of_movement_arrays[0]))
    for f in range(len(total_motion)):
        valid_subjects_this_tbucket = len(list_of_movement_arrays) - nan_count[f]
        avg_motion[f] = total_motion[f]/valid_subjects_this_tbucket
    # smooth the average motion
    # smoothing window must be odd!
    # apply savitzky-golay filter to smooth
    avg_motion_smoothed = savgol_filter(avg_motion, window, 3)
    # find peaks in average motion
    peaks, _ = find_peaks(avg_motion_smoothed, height=(2,10), prominence=0.75)
    return avg_motion_smoothed, peaks

def find_saccades(list_of_movement_arrays, saccade_threshold, raw_count_threshold, window_size, windowed_count_threshold):
    all_trials_peaks = []
    for trial in range(len(list_of_movement_arrays)):
        all_trials_peaks.append([])
        this_trial = list_of_movement_arrays[trial]
        for time_bucket in range(len(this_trial)):
            # find timebuckets where abs(movement)>threshold
            if abs(this_trial[time_bucket])>=saccade_threshold:
                all_trials_peaks[trial].append(time_bucket)
    # count number of subjects who had peaks in the same timebuckets
    trial_peaks_totals = {}
    trial_peaks_totals = defaultdict(lambda:0, trial_peaks_totals)
    for trial in all_trials_peaks:
        for tbucket in trial:
            trial_peaks_totals[tbucket] = trial_peaks_totals[tbucket] + 1
    # filter for timebuckets when "enough" subjects had peaks
    peak_tbuckets_filtered = {}
    # combine counts of peaks within time windows
    for key in trial_peaks_totals.keys():
        count = trial_peaks_totals[key]
        if count>=raw_count_threshold:
            #print(t, count)
            peak_tbuckets_filtered[key] = count
    # combine counts of peaks within time windows
    peak_tbuckets_windowed = find_windowed_peaks(peak_tbuckets_filtered, window_size, windowed_count_threshold)
    saccades = {tbucket:total for tbucket,total in peak_tbuckets_windowed.items()}
    return saccades

##########################################################
# BEGIN SCRIPT
##########################################################
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
logging.basicConfig(filename="pm01CalcMvmnt_" + todays_datetime + ".log", filemode='w', level=logging.INFO)
###################################
# SOURCE DATA AND OUTPUT FILE LOCATIONS 
###################################
data_folder = load_data()
# set up input folders
pupil_data_downsampled = os.path.join(data_folder, 'downsampled_pupils')
# set up various output folders
calib_mvmnt_folder = os.path.join(data_folder, 'calib_movement')
octo_mvmnt_folder = os.path.join(data_folder, 'octo_movement')
unique_mvmnt_folder = os.path.join(data_folder, 'unique_movement')
# Create plots folder (and sub-folders) if it (they) does (do) not exist
if not os.path.exists(calib_mvmnt_folder):
    os.makedirs(calib_mvmnt_folder)
if not os.path.exists(octo_mvmnt_folder):
    os.makedirs(octo_mvmnt_folder)
if not os.path.exists(unique_mvmnt_folder):
    os.makedirs(unique_mvmnt_folder)

logging.info('PUPIL DATA FOLDER: %s \n CALIB MOVEMENT FOLDER: %s \n OCTO MOVEMENT FOLDER: %s \n UNIQUE MOVEMENT FOLDER: %s' % (pupil_data_downsampled, calib_mvmnt_folder, octo_mvmnt_folder, unique_mvmnt_folder))
print('PUPIL DATA FOLDER: %s \n CALIB MOVEMENT FOLDER: %s \n OCTO MOVEMENT FOLDER: %s \n UNIQUE MOVEMENT FOLDER: %s' % (pupil_data_downsampled, calib_mvmnt_folder, octo_mvmnt_folder, unique_mvmnt_folder))
###################################
# PARAMETERS
###################################
downsampled_bucket_size_ms = 40 # milliseconds
smoothing_window = 25 # in time buckets, must be odd! for savgol_filter
###################################
# SORT DATA BY STIMULUS TYPE
###################################
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
all_right_trials_contours_X = {key:[] for key in stim_vids}
all_right_trials_contours_Y = {key:[] for key in stim_vids}
all_right_trials_circles_X = {key:[] for key in stim_vids}
all_right_trials_circles_Y = {key:[] for key in stim_vids}
all_left_trials_contours_X = {key:[] for key in stim_vids}
all_left_trials_contours_Y = {key:[] for key in stim_vids}
all_left_trials_circles_X = {key:[] for key in stim_vids}
all_left_trials_circles_Y = {key:[] for key in stim_vids}
all_trials_position_X_data = [all_right_trials_contours_X, all_right_trials_circles_X, all_left_trials_contours_X, all_left_trials_circles_X]
all_trials_position_Y_data = [all_right_trials_contours_Y, all_right_trials_circles_Y, all_left_trials_contours_Y, all_left_trials_circles_Y]
activation_count = []
analysed_count = []
#########################################################
# LOAD PUPIL POSITIONS FROM CONSOLIDATED DAILY PUPIL DATA
#########################################################
daily_folders = glob.glob(pupil_data_downsampled + os.sep + '*.npz')
for daily_pupil_data in daily_folders:
    pupil_data = np.load(daily_pupil_data, allow_pickle=True)
    this_day_x_pos = pupil_data['all_pos_x']
    this_day_y_pos = pupil_data['all_pos_y']
    # extract activation and good trials count
    file_info = os.path.basename(daily_pupil_data).split('_')
    this_date = file_info[0]
    num_right_activations = int(file_info[1][6:])
    num_left_activations = int(file_info[2][6:])
    num_good_right_trials = int(file_info[3][5:])
    num_good_left_trials = int(file_info[4].split('.')[0][5:])
    analysed_count.append((num_good_right_trials, num_good_left_trials))
    activation_count.append((num_right_activations, num_left_activations))
    # append position data to global data structure
    for i in range(len(this_day_x_pos)):
        for stimulus in this_day_x_pos[i]:
            for index in range(len(this_day_x_pos[i][stimulus])):
                all_trials_position_X_data[i][stimulus].append(this_day_x_pos[i][stimulus][index])
    for i in range(len(this_day_y_pos)):
        for stimulus in this_day_y_pos[i]:
            for index in range(len(this_day_y_pos[i][stimulus])):
                all_trials_position_Y_data[i][stimulus].append(this_day_y_pos[i][stimulus][index])

#########################################################
# ACTIVATION / GOOD TRIALS
#########################################################
total_activation = sum(count[0] for count in activation_count)
total_days_activated = len(activation_count)
good_trials_right = [count[0] for count in analysed_count]
good_trials_left = [count[1] for count in analysed_count]
total_good_trials_right = sum(good_trials_right)
total_good_trials_left = sum(good_trials_left)
print("Total number of exhibit activations: {total}".format(total=total_activation))
print("Total number of good right eye camera trials: {good_total}".format(good_total=total_good_trials_right))
print("Total number of good left eye camera trials: {good_total}".format(good_total=total_good_trials_left))
logging.info("Total number of exhibit activations: {total}".format(total=total_activation))
logging.info("Total number of good right eye camera trials: {good_total}".format(good_total=total_good_trials_right))
logging.info("Total number of good left eye camera trials: {good_total}".format(good_total=total_good_trials_left))
activation_array = np.array(activation_count)
analysed_array_right = np.array(good_trials_right)
analysed_array_left = np.array(good_trials_left)
###################################
# CALCULATE PUPIL MOVEMENT
###################################
all_trials_position_right_data = [all_right_trials_contours_X, all_right_trials_contours_Y, all_right_trials_circles_X, all_right_trials_circles_Y]
all_trials_position_left_data = [all_left_trials_contours_X, all_left_trials_contours_Y, all_left_trials_circles_X, all_left_trials_circles_Y]
all_positions = [all_trials_position_right_data, all_trials_position_left_data]
# currently we are not pairing right and left eye coordinates
# measure movement from one frame to next
all_right_contours_movement_X = {key:[] for key in stim_vids}
all_right_circles_movement_X = {key:[] for key in stim_vids}
all_right_contours_movement_Y = {key:[] for key in stim_vids}
all_right_circles_movement_Y = {key:[] for key in stim_vids}
all_left_contours_movement_X = {key:[] for key in stim_vids}
all_left_circles_movement_X = {key:[] for key in stim_vids}
all_left_contours_movement_Y = {key:[] for key in stim_vids}
all_left_circles_movement_Y = {key:[] for key in stim_vids}
all_movement_right = [all_right_contours_movement_X, all_right_contours_movement_Y, all_right_circles_movement_X, all_right_circles_movement_Y]
all_movement_left = [all_left_contours_movement_X, all_left_contours_movement_Y, all_left_circles_movement_X, all_left_circles_movement_Y]
all_movements = [all_movement_right, all_movement_left]
side_names = ['Right', 'Left']
cAxis_names = ['contoursX', 'contoursY', 'circlesX', 'circlesY']
### calculate movement ###
for side in range(len(all_positions)):
    for c_axis in range(len(all_positions[side])):
        for stimuli in all_positions[side][c_axis]:
            print('Calculating movements for {side} side, {cAxis_type}, stimulus {stim}'.format(side=side_names[side], cAxis_type=cAxis_names[c_axis], stim=stimuli))
            logging.info('Calculating movements for {side} side, {cAxis_type}, stimulus {stim}'.format(side=side_names[side], cAxis_type=cAxis_names[c_axis], stim=stimuli))
            # if there are nans (dropped frames) for more than 2 seconds of video time, then toss that trial
            dropped_frames_threshold = 2000/downsampled_bucket_size_ms
            all_movements[side][c_axis][stimuli] = calc_mvmnt_from_pos(all_positions[side][c_axis][stimuli], dropped_frames_threshold, 100, -100)

###################################
# CALCULATE PUPIL MOTION (abs val of movement)
###################################
all_right_contours_X_avg_motion = {key:[] for key in stim_vids}
all_right_circles_X_avg_motion = {key:[] for key in stim_vids}
all_right_contours_Y_avg_motion = {key:[] for key in stim_vids}
all_right_circles_Y_avg_motion = {key:[] for key in stim_vids}
all_left_contours_X_avg_motion = {key:[] for key in stim_vids}
all_left_circles_X_avg_motion = {key:[] for key in stim_vids}
all_left_contours_Y_avg_motion = {key:[] for key in stim_vids}
all_left_circles_Y_avg_motion = {key:[] for key in stim_vids}
all_avg_motion_right = [all_right_contours_X_avg_motion, all_right_contours_Y_avg_motion, all_right_circles_X_avg_motion, all_right_circles_Y_avg_motion]
all_avg_motion_left = [all_left_contours_X_avg_motion, all_left_contours_Y_avg_motion, all_left_circles_X_avg_motion, all_left_circles_Y_avg_motion]
all_avg_motion = [all_avg_motion_right, all_avg_motion_left]
# motion peaks
all_RcontoursX_avg_motion_peaks = {key:[] for key in stim_vids}
all_RcirclesX_avg_motion_peaks = {key:[] for key in stim_vids}
all_RcontoursY_avg_motion_peaks = {key:[] for key in stim_vids}
all_RcirclesY_avg_motion_peaks = {key:[] for key in stim_vids}
all_LcontoursX_avg_motion_peaks = {key:[] for key in stim_vids}
all_LcirclesX_avg_motion_peaks = {key:[] for key in stim_vids}
all_LcontoursY_avg_motion_peaks = {key:[] for key in stim_vids}
all_LcirclesY_avg_motion_peaks = {key:[] for key in stim_vids}
all_avg_motion_right_peaks = [all_RcontoursX_avg_motion_peaks, all_RcontoursY_avg_motion_peaks, all_RcirclesX_avg_motion_peaks, all_RcirclesY_avg_motion_peaks]
all_avg_motion_left_peaks = [all_LcontoursX_avg_motion_peaks, all_LcontoursY_avg_motion_peaks, all_LcirclesX_avg_motion_peaks, all_LcirclesY_avg_motion_peaks]
all_avg_motion_peaks = [all_avg_motion_right_peaks, all_avg_motion_left_peaks]
# find average pixel motion per time_bucket for each stimulus
for side in range(len(all_movements)):
    for c_axis in range(len(all_movements[side])):
        for stimuli in all_movements[side][c_axis]:
            print('Calculating average motion for {side} side, {cAxis_type}, stimulus {stim}'.format(side=side_names[side], cAxis_type=cAxis_names[c_axis], stim=stimuli))
            logging.info('Calculating average motion for {side} side, {cAxis_type}, stimulus {stim}'.format(side=side_names[side], cAxis_type=cAxis_names[c_axis], stim=stimuli))
            avg_motion_this_stim, peaks_this_stim = calc_avg_motion_and_peaks(all_movements[side][c_axis][stimuli], smoothing_window)
            all_avg_motion[side][c_axis][stimuli] = avg_motion_this_stim
            all_avg_motion_peaks[side][c_axis][stimuli] = peaks_this_stim

###################################
# FIND PEAKS IN MOVEMENT
###################################
all_right_contours_X_peaks = {key:{} for key in stim_vids}
all_right_circles_X_peaks = {key:{} for key in stim_vids}
all_right_contours_Y_peaks = {key:{} for key in stim_vids}
all_right_circles_Y_peaks = {key:{} for key in stim_vids}
all_left_contours_X_peaks = {key:{} for key in stim_vids}
all_left_circles_X_peaks = {key:{} for key in stim_vids}
all_left_contours_Y_peaks = {key:{} for key in stim_vids}
all_left_circles_Y_peaks = {key:{} for key in stim_vids}
all_peaks_right = [all_right_contours_X_peaks, all_right_contours_Y_peaks, all_right_circles_X_peaks, all_right_circles_Y_peaks]
all_peaks_left = [all_left_contours_X_peaks, all_left_contours_Y_peaks, all_left_circles_X_peaks, all_left_circles_Y_peaks]
all_peaks = [all_peaks_right, all_peaks_left]
# filter through the movement to find peaks in individual traces
for side in range(len(all_movements)):
    for c_axis in range(len(all_movements[side])):
        for stim in all_movements[side][c_axis]:
            saccade_thresholds = [2.5, 5, 10, 20, 30, 40, 50, 60] # pixels
            all_peaks[side][c_axis][stim] = {key:{} for key in saccade_thresholds}
            this_stim_N = len(all_movements[side][c_axis][stim])
            count_threshold = this_stim_N/10
            windowed_count_thresholds = [this_stim_N/(i*2) for i in range(1, len(saccade_thresholds)+1)]
            for thresh in range(len(saccade_thresholds)):
                print('Looking for movements greater than {p} pixels in {side} side, {cAxis_type}, stimulus {s}'.format(p=saccade_thresholds[thresh], side=side_names[side], cAxis_type=cAxis_names[c_axis], s=stim))
                logging.info('Looking for movements greater than {p} pixels in {side} side, {cAxis_type}, stimulus {s}'.format(p=saccade_thresholds[thresh], side=side_names[side], cAxis_type=cAxis_names[c_axis], s=stim))
                peaks_window = 40 # timebuckets
                s_thresh = saccade_thresholds[thresh]
                w_thresh = windowed_count_thresholds[thresh]
                all_peaks[side][c_axis][stim][s_thresh] = find_saccades(all_movements[side][c_axis][stim], s_thresh, count_threshold, peaks_window, w_thresh)

###################################
# SPLIT INTO OCTO, UNIQUE, CALIB
###################################
stim_old_to_new = {24.0: 0, 25.0: 1, 26.0: 2, 27.0: 3, 28.0: 4, 29.0: 5}
# timing info in 40ms resolution (as opposed to timing info in sd02_detect_saccades.py, which is in 4ms resolution)
calib_start = 0
calib_end = 443
unique_start = 443
unique_ends = {0: 596, 1: 602, 2: 666, 3: 608, 4: 667, 5: 719}
octo_len = 398

for side in range(len(all_movements)):
    for c_axis in range(len(all_movements[side])):  
        # INITIATE ARRAY STRUCTURE FOR EACH SEQUENCE
        # movements
        all_calib_mvmnts = []
        all_octo_mvmnts = []
        all_unique_mvmnts = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
        # avg motion
        all_calib_avg_motion = []
        all_octo_avg_motion = []
        all_unique_avg_motion = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
        # avg motion peaks
        all_calib_avg_motion_peaks = []
        all_octo_avg_motion_peaks = []
        all_unique_avg_motion_peaks = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
        # movement peaks
        all_calib_mvmnt_peaks = {}
        all_octo_mvmnt_peaks = {}
        all_unique_mvmnts_peaks = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
        # chunk and save
        for stimuli in all_movements[side][c_axis]:
            new_stim_number = stim_old_to_new[stimuli]
            print('Chunking into calibration, octopus, and unique sequences for {side} side, {cAxis_type}, old stim number {stim}, new stim number {new_stim}'.format(side=side_names[side], cAxis_type=cAxis_names[c_axis], stim=stimuli, new_stim=new_stim_number))
            logging.info('Chunking into calibration, octopus, and unique sequences for {side} side, {cAxis_type}, old stim number {stim}, new stim number {new_stim}'.format(side=side_names[side], cAxis_type=cAxis_names[c_axis], stim=stimuli, new_stim=new_stim_number))
            # MOVEMENT
            for trial in all_movements[side][c_axis][stimuli]:
                this_trial_calib = trial[:calib_end]
                all_calib_mvmnts.append(this_trial_calib)
                this_trial_unique = trial[calib_end:unique_trials[new_stim_number]]
                all_unique_mvmnts[str(new_stim_number)].append(this_trial_unique)
                this_trial_octo = trial[unique_trials[new_stim_number]:unique_trials[new_stim_number]+octo_len]
                all_octo_mvmnts.append(this_trial_octo)
            # AVG MOTION
            this_stim_calib_avg_motion = all_avg_motion[side][c_axis][stimuli][:calib_end]
            all_calib_avg_motion.append(this_stim_calib_avg_motion)
            this_stim_unique_avg_motion = all_avg_motion[side][c_axis][stimuli][calib_end:unique_trials[new_stim_number]]
            all_unique_avg_motion[str(new_stim_number)].append(this_stim_unique_avg_motion)
            this_stim_octo_avg_motion = all_avg_motion[side][c_axis][stimuli][unique_trials[new_stim_number]:unique_trials[new_stim_number]+octo_len]
            all_octo_avg_motion.append(this_stim_octo_avg_motion)
            # AVG MOTION PEAKS
            this_stim_calib_avg_motion_peaks = [x for x in all_avg_motion_peaks[side][c_axis][stimuli] if x<calib_end]
            all_calib_avg_motion_peaks.append(this_stim_calib_avg_motion_peaks)
            this_stim_unique_avg_motion_peaks = [x for x in all_avg_motion_peaks[side][c_axis][stimuli] if calib_end<x<unique_trials[new_stim_number]]
            all_unique_avg_motion_peaks[str(new_stim_number)].append(this_stim_unique_avg_motion_peaks)
            this_stim_octo_avg_motion_peaks = [x for x in all_avg_motion_peaks[side][c_axis][stimuli] if unique_trials[new_stim_number]<x<unique_trials[new_stim_number]+octo_len]
            all_octo_avg_motion_peaks.append(this_stim_octo_avg_motion_peaks)
            # MOVEMENT PEAKS
            for saccade_threshold in all_peaks[side][c_axis][stimuli]:
                this_saccade_thresh_calib = []
                this_saccade_thresh_unique = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
                this_saccade_thresh_octo = []
                for time_bucket in all_peaks[side][c_axis][stimuli][saccade_threshold]:
                    if time_bucket<calib_end:
                        this_saccade_thresh_calib.append((timebucket, all_peaks[side][c_axis][stimuli][saccade_threshold][time_bucket]))
                    if calib_end<time_bucket<unique_trials[new_stim_number]:
                        this_saccade_thresh_unique[str(new_stim_number)].append((timebucket, all_peaks[side][c_axis][stimuli][saccade_threshold][time_bucket]))
                    if unique_trials[new_stim_number]<time_bucket<unique_trials[new_stim_number]+octo_len:
                        this_saccade_thresh_octo.append((time_bucket, all_peaks[side][c_axis][stimuli][saccade_threshold][time_bucket]))
                all_calib_mvmnt_peaks[saccade_threshold] = this_saccade_thresh_calib
                all_unique_mvmnts_peaks[new_stim_number][saccade_threshold] = this_saccade_thresh_unique
                all_octo_mvmnt_peaks[saccade_threshold] = this_saccade_thresh_octo
        # SAVE
        # movements
        all_calib_mvmnts = np.array(all_calib_mvmnts)
        all_octo_mvmnts = np.array(all_octo_mvmnts)
        N_per_unique = []
        for unique in all_unique_mvmnts:
            all_unique_mvmnts[unique] = np.array(all_unique_mvmnts[unique])
            N_per_unique.append(str(len(all_unique_mvmnts[unique])))
        unique_N_str = '-'.join(N_per_unique)
        calib_path = calib_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_calib_mvmnt_' + str(len(all_calib_mvmnts)) + '.npz'
        octo_path = octo_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_octo_mvmnt_' + str(len(all_octo_mvmnts)) + '.npz'
        unique_path = unique_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_uniques_mvmnt_' + unique_N_str + '_' + '.npz'
        print('Saving movement data to file, Calib = {c}, Octo = {o}, Unique = {u}'.format(c=len(all_calib_mvmnts), o=len(all_octo_mvmnts), u=unique_N_str))
        logging.info('Saving movement data to file, Calib = {c}, Octo = {o}, Unique = {u}'.format(c=len(all_calib_mvmnts), o=len(all_octo_mvmnts), u=unique_N_str))
        np.savez(calib_path, all_calib_mvmnts)
        np.savez(octo_path, all_octo_mvmnts)
        np.savez(unique_path, **all_unique_mvmnts)
        # avg motion
        all_calib_avg_motion = np.array(all_calib_avg_motion)
        all_octo_avg_motion = np.array(all_octo_avg_motion)
        N_per_unique = []
        for unique in all_unique_avg_motion:
            all_unique_avg_motion[unique] = np.array(all_unique_avg_motion[unique])
            N_per_unique.append(str(len(all_unique_avg_motion[unique])))
        unique_N_str = '-'.join(N_per_unique)
        calib_path = calib_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_calib_avg_motion_' + str(len(all_calib_avg_motion)) + '.npz'
        octo_path = octo_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_octo_avg_motion_' + str(len(all_octo_avg_motion)) + '.npz'
        unique_path = unique_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_uniques_avg_motion_' + unique_N_str + '_' + '.npz'
        print('Saving avg motion data to file, Calib = {c}, Octo = {o}, Unique = {u}'.format(c=len(all_calib_avg_motion), o=len(all_octo_avg_motion), u=unique_N_str))
        logging.info('Saving avg motion data to file, Calib = {c}, Octo = {o}, Unique = {u}'.format(c=len(all_calib_avg_motion), o=len(all_octo_avg_motion), u=unique_N_str))
        np.savez(calib_path, all_calib_avg_motion)
        np.savez(octo_path, all_octo_avg_motion)
        np.savez(unique_path, **all_unique_avg_motion)
        # avg motion peaks
        all_calib_avg_motion_peaks = np.array(all_calib_avg_motion_peaks)
        all_octo_avg_motion_peaks = np.array(all_octo_avg_motion_peaks)
        N_per_unique = []
        for unique in all_unique_avg_motion_peaks:
            all_unique_avg_motion_peaks[unique] = np.array(all_unique_avg_motion_peaks[unique])
            N_per_unique.append(str(len(all_unique_avg_motion_peaks[unique])))
        unique_N_str = '-'.join(N_per_unique)
        calib_path = calib_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_calib_avg_motion_peaks' + str(len(all_calib_avg_motion_peaks)) + '.npz'
        octo_path = octo_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_octo_avg_motion_peaks' + str(len(all_octo_avg_motion_peaks)) + '.npz'
        unique_path = unique_mvmnt_folder + os.sep + side_names[side] + '_' + cAxis_names[c_axis] + '_uniques_avg_motion_peaks' + unique_N_str + '_' + '.npz'
        print('Saving avg motion peaks data to file, Calib = {c}, Octo = {o}, Unique = {u}'.format(c=len(all_calib_avg_motion_peaks), o=len(all_octo_avg_motion_peaks), u=unique_N_str))
        logging.info('Saving avg motion peaks data to file, Calib = {c}, Octo = {o}, Unique = {u}'.format(c=len(all_calib_avg_motion_peaks), o=len(all_octo_avg_motion_peaks), u=unique_N_str))
        np.savez(calib_path, all_calib_avg_motion_peaks)
        np.savez(octo_path, all_octo_avg_motion_peaks)
        np.savez(unique_path, **all_unique_avg_motion_peaks)
        # movement peaks (saccades)
        

#FIN