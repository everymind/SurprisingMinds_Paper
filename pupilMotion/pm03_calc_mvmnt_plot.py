# -*- coding: utf-8 -*-
"""
Project: "Surprising Minds" at Sea Life Brighton, by Danbee Kim, Kerry Perkins, Clive Ramble, Hazel Garnade, Goncalo Lopes, Dario Quinones, Reanna Campbell-Russo, Robb Barrett, Martin Stopps, The EveryMind Team, and Adam Kampff. 
Analysis: Measure speed of pupil

Loads daily .npz files with x position, y position, size, and size baseline data.
Calculate movement from one frame to the next and find movement peaks (saccades).
Plot pupil movement and motion (absolute value of movement).

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
# 2) plots_dir (parent folder for all plots output by this script)
##########################################################
def load_data():
    dataset_dir = r'D:\data\SurprisingMinds\intermediates'
    plots_dir = r'D:\data\SurprisingMinds\plots'
    return dataset_dir, plots_dir
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
logging.basicConfig(filename="pm03CalcMvmntPlot_" + todays_datetime + ".log", filemode='w', level=logging.INFO)
###################################
# SOURCE DATA AND OUTPUT FILE LOCATIONS 
###################################
data_folder, plots_folder = load_data()
# set up input folders
pupil_data_downsampled = os.path.join(data_folder, 'downsampled_pupils')
lum_processed = os.path.join(data_folder, 'lum_processed')
# set up various plot output folders
pupil_motion_plots = os.path.join(plots_folder, "pupil_motion")
# Create plots folder (and sub-folders) if it (they) does (do) not exist
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
if not os.path.exists(pupil_motion_plots):
    os.makedirs(pupil_motion_plots)

logging.info('PUPIL DATA FOLDER: %s \n STIMULI LUMINANCE DATA FOLDER: %s \n PUPIL PLOTS FOLDER: %s' % (pupil_data_downsampled, lum_processed, pupil_motion_plots))
print('PUPIL DATA FOLDER: %s \n STIMULI LUMINANCE DATA FOLDER: %s \n PUPIL PLOTS FOLDER: %s' % (pupil_data_downsampled, lum_processed, pupil_motion_plots))
###################################
# PARAMETERS
###################################
downsampled_bucket_size_ms = 40
smoothing_window = 25 # in time buckets, must be odd! for savgol_filter
fig_size = 200 # dpi
plot_movement = False 
plot_motion = True 
plot_peaks = False
###################################
# SORT DATA BY STIMULUS TYPE
###################################
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
stim_name_to_float = {"stimuli024": 24.0, "stimuli025": 25.0, "stimuli026": 26.0, "stimuli027": 27.0, "stimuli028": 28.0, "stimuli029": 29.0}
stim_float_to_name = {24.0: "stimuli024", 25.0: "stimuli025", 26.0: "stimuli026", 27.0: "stimuli027", 28.0: "stimuli028", 29.0: "stimuli029"}
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
# FIND PEAKS IN MOTION
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
# LOAD STIMULUS LUMINANCE DATA
###################################
luminance_info_path = glob.glob(lum_processed + os.sep + '*.npz')
for lum_info_path in luminance_info_path:
    if os.path.basename(lum_info_path) == 'processed_lum.npz':
        lum = np.load(lum_info_path)
    elif os.path.basename(lum_info_path) == 'processed_lum_avg.npz':
        lum_avg = np.load(lum_info_path)
    else:
        lum_peaks = np.load(lum_info_path)

luminances_avg = {}
luminances_peaks = {}
luminances = {}
for vid_key in stim_vids:
    luminances_avg[vid_key] = lum_avg[stim_float_to_name[vid_key]]
    luminances_peaks[vid_key] = lum_peaks[stim_float_to_name[vid_key]]
    luminances[vid_key] = lum[stim_float_to_name[vid_key]]

###################################
# PLOT MOVEMENT
###################################
plotting_peaks_window = 40 # MAKE SURE THIS == peaks_window!!
cType_names = ['Contours', 'Circles']
all_movement_right_plot = [(all_right_contours_movement_X, all_right_contours_movement_Y), (all_right_circles_movement_X, all_right_circles_movement_Y)]
all_movement_left_plot = [(all_left_contours_movement_X, all_left_contours_movement_Y), (all_left_circles_movement_X, all_left_circles_movement_Y)]
all_movements_plot = [all_movement_right_plot, all_movement_left_plot]
# plot movement traces
if plot_movement:
    for side in range(len(all_movements_plot)):
        for c_type in range(len(all_movements_plot[side])):
            for stimuli in all_movements_plot[side][c_type][0]:
                plot_type_name = side_names[side] + cType_names[c_type]
                stim_name = stim_float_to_name[stimuli]
                plot_type_X = all_movements_plot[side][c_type][0][stimuli]
                plot_N_X = len(plot_type_X)
                plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
                plot_N_Y = len(plot_type_Y)
                plot_luminance = luminances_avg[stimuli]
                plot_luminance_peaks = luminances_peaks[stimuli]
                # fig name and path
                figure_name = 'MovementTraces_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
                figure_path = os.path.join(pupil_motion_plots, figure_name)
                figure_title = "Pupil movement of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPlotted on " + todays_datetime
                # draw fig
                plt.figure(figsize=(14, 14), dpi=fig_size)
                plt.suptitle(figure_title, fontsize=12, y=0.98)
                # x-axis
                plt.subplot(3,1,1)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_X:
                    plt.plot(trial, linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
                plt.xlim(-10,1250)
                plt.ylim(-80,80)
                # y-axis
                plt.subplot(3,1,2)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_Y:
                    plt.plot(trial, linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
                plt.xlim(-10,1250)
                plt.ylim(-80,80)
                # luminance
                plt.subplot(3,1,3)
                plt.ylabel('Percent change in luminance', fontsize=11)
                plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
                plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
                plt.grid(b=True, which='major', linestyle='--')
                plt.plot(plot_luminance, linewidth=0.75, color=[1.0, 0.13, 0.4, 1])
                for peak in plot_luminance_peaks:
                    plt.plot(peak, plot_luminance[peak], 'x')
                    plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-1,7)
                # save and display
                plt.subplots_adjust(hspace=0.5)
                plt.savefig(figure_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

###################################
# PLOT MOTION
###################################
all_avg_motion_right_plot = [(all_right_contours_X_avg_motion, all_right_contours_Y_avg_motion), (all_right_circles_X_avg_motion, all_right_circles_Y_avg_motion)]
all_avg_motion_left_plot = [(all_left_contours_X_avg_motion, all_left_contours_Y_avg_motion), (all_left_circles_X_avg_motion, all_left_circles_Y_avg_motion)]
all_avg_motion_plot = [all_avg_motion_right_plot, all_avg_motion_left_plot]
all_avg_motion_right_peaks_plot = [(all_RcontoursX_avg_motion_peaks, all_RcontoursY_avg_motion_peaks), (all_RcirclesX_avg_motion_peaks, all_RcirclesY_avg_motion_peaks)]
all_avg_motion_left_peaks_plot = [(all_LcontoursX_avg_motion_peaks, all_LcontoursY_avg_motion_peaks), (all_LcirclesX_avg_motion_peaks, all_LcirclesY_avg_motion_peaks)]
all_avg_motion_peaks_plot = [all_avg_motion_right_peaks_plot, all_avg_motion_left_peaks_plot]
# plot MOTION traces (abs val of movement traces)
if plot_motion:
    for side in range(len(all_movements_plot)):
        for c_type in range(len(all_movements_plot[side])):
            for stimuli in all_movements_plot[side][c_type][0]:
                plot_type_name = side_names[side] + cType_names[c_type]
                stim_name = stim_float_to_name[stimuli]
                plot_type_X = all_movements_plot[side][c_type][0][stimuli]
                plot_type_X_avg = all_avg_motion_plot[side][c_type][0][stimuli]
                plot_type_X_avg_peaks = all_avg_motion_peaks_plot[side][c_type][0][stimuli]
                plot_N_X = len(plot_type_X)
                plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
                plot_type_Y_avg = all_avg_motion_plot[side][c_type][1][stimuli]
                plot_type_Y_avg_peaks = all_avg_motion_peaks_plot[side][c_type][1][stimuli]
                plot_N_Y = len(plot_type_Y)
                plot_luminance = luminances_avg[stimuli]
                plot_luminance_peaks = luminances_peaks[stimuli]
                # fig name and path
                figure_name = 'MotionTraces-AvgMotionPeaks' + str(plotting_peaks_window) + '_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
                figure_path = os.path.join(pupil_motion_plots, figure_name)
                figure_title = "Pupil motion of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPeak finding window: " + str(plotting_peaks_window) + "\nPlotted on " + todays_datetime
                # draw fig
                plt.figure(figsize=(14, 14), dpi=fig_size)
                plt.suptitle(figure_title, fontsize=12, y=0.98)
                # x-axis
                plt.subplot(3,1,1)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_X:
                    plt.plot(abs(trial), linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
                plt.plot(plot_type_X_avg, linewidth=1, color=[0.4, 1.0, 0.27, 1])
                for peak in plot_type_X_avg_peaks:
                    if peak<1250:
                        plt.plot(peak, plot_type_X_avg[peak], 'x')
                        plt.text(peak-15, plot_type_X_avg[peak]+5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-5,40)
                # y-axis
                plt.subplot(3,1,2)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_Y:
                    plt.plot(abs(trial), linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
                plt.plot(plot_type_Y_avg, linewidth=1, color=[1.0, 1.0, 0.25, 1])
                for peak in plot_type_Y_avg_peaks:
                    if peak<1250:
                        plt.plot(peak, plot_type_Y_avg[peak], 'x')
                        plt.text(peak-15, plot_type_Y_avg[peak]+5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-5,40)
                # luminance
                plt.subplot(3,1,3)
                plt.ylabel('Percent change in luminance', fontsize=11)
                plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
                #plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
                plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled', fontsize=10, color='grey', style='italic')
                plt.grid(b=True, which='major', linestyle='--')
                plt.plot(plot_luminance, linewidth=1, color=[1.0, 0.13, 0.4, 1])
                for peak in plot_luminance_peaks:
                    plt.plot(peak, plot_luminance[peak], 'x')
                    plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-1,7)
                # save and display
                plt.subplots_adjust(hspace=0.5)
                plt.savefig(figure_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

###################################
# PLOT PEAKS
###################################
all_peaks_right_plot = [(all_right_contours_X_peaks, all_right_contours_Y_peaks), (all_right_circles_X_peaks, all_right_circles_Y_peaks)]
all_peaks_left_plot = [(all_left_contours_X_peaks, all_left_contours_Y_peaks), (all_left_circles_X_peaks, all_left_circles_Y_peaks)]
all_peaks_plot = [all_peaks_right_plot, all_peaks_left_plot]
if plot_peaks:
    for side in range(len(all_movements_plot)):
        for c_type in range(len(all_movements_plot[side])):
            for stimuli in all_movements_plot[side][c_type][0]:
                plot_type_name = side_names[side] + cType_names[c_type]
                stim_name = stim_float_to_name[stimuli]
                plot_type_X = all_movements_plot[side][c_type][0][stimuli]
                plot_type_X_peaks = all_peaks_plot[side][c_type][0][stimuli]
                plot_N_X = len(plot_type_X)
                plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
                plot_type_Y_peaks = all_peaks_plot[side][c_type][1][stimuli]
                plot_N_Y = len(plot_type_Y)
                plot_luminance = luminances_avg[stimuli]
                plot_luminance_peaks = luminances_peaks[stimuli]
                # fig name and path
                figure_name = 'MotionTraces-Saccades' + str(plotting_peaks_window) + '_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
                figure_path = os.path.join(pupil_motion_plots, figure_name)
                figure_title = "Pupil motion of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPeaks plotted at height of pixel movement threshold, peak finding window: " + str(plotting_peaks_window) + "\nPlotted on " + todays_datetime
                # begin drawing fig
                plt.figure(figsize=(14, 14), dpi=fig_size)
                plt.suptitle(figure_title, fontsize=12, y=0.98)
                # x-axis
                plt.subplot(3,1,1)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_X:
                    plt.plot(abs(trial), linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
                for threshold in plot_type_X_peaks.keys():
                    for key in plot_type_X_peaks[threshold].keys():
                        if key<1250:
                            plt.plot(key, threshold, '1', color=[0.4, 1.0, 0.27, 1.0])
                plt.xlim(-10,1250)
                plt.ylim(-5,60)
                # y-axis
                plt.subplot(3,1,2)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_Y:
                    plt.plot(abs(trial), linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
                for threshold in plot_type_Y_peaks.keys():
                    for key in plot_type_Y_peaks[threshold].keys():
                        if key<1250:
                            plt.plot(key, threshold, '1', color=[1.0, 1.0, 0.25, 1.0])
                plt.xlim(-10,1250)
                plt.ylim(-5,60)
                # luminance
                plt.subplot(3,1,3)
                plt.ylabel('Percent change in luminance (from baseline)', fontsize=11)
                plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
                plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
                plt.grid(b=True, which='major', linestyle='--')
                plt.plot(plot_luminance, linewidth=1, color=[1.0, 0.13, 0.4, 1])
                for peak in plot_luminance_peaks:
                    plt.plot(peak, plot_luminance[peak], 'x')
                    plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-1,7)
                # save and display
                plt.subplots_adjust(hspace=0.5)
                plt.savefig(figure_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

#FIN