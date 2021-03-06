# -*- coding: utf-8 -*-
"""
Project: "Surprising Minds" at Sea Life Brighton, by Danbee Kim, Kerry Perkins, Clive Ramble, Hazel Garnade, Goncalo Lopes, Dario Quinones, Reanna Campbell-Russo, Robb Barrett, Martin Stopps, The EveryMind Team, and Adam Kampff. 
Analysis preprocessing: Extract luminance info from stimulus vids

Loads .csv files with stimulus luminance info, generated by Stimuli_TimestampedLuminanceValues_wholeStimulus_batch.bonsai
Calculate baseline, smoothed average, and peaks for stimuli luminance.
Save as a .npz file.

@author: Adam R Kampff and Danbee Kim
"""
import os
import glob
import datetime
import numpy as np
from scipy.signal import savgol_filter
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
# 1) dataset_dir (folder with .csv files of luminance per frame of daily stimuli)
# 2) output_dir (folder for intermediate files output by this script)
##########################################################
def load_data():
    dataset_dir = r'D:\data\SurprisingMinds\LuminancePerFrame'
    output_dir = r'D:\data\SurprisingMinds\intermediates'
    return dataset_dir, output_dir
##########################################################
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

def threshold_to_nan(input_array, threshold, upper_or_lower):
    for index in range(len(input_array)): 
        if upper_or_lower=='upper':
            if np.isnan(input_array[index])==False and input_array[index]>threshold:
                input_array[index] = np.nan
        if upper_or_lower=='lower':
            if np.isnan(input_array[index])==False and input_array[index]<threshold:
                input_array[index] = np.nan
    return input_array

##########################################################
# BEGIN SCRIPT
##########################################################
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
logging.basicConfig(filename="pp02ExtractStimInfo_" + todays_datetime + ".log", filemode='w', level=logging.INFO)
###################################
# SOURCE DATA AND OUTPUT FILE LOCATIONS 
###################################
stim_lum_folder, intermediates_folder = load_data()
# set up various plot output folders
lum_processed_folder = os.path.join(intermediates_folder, "lum_processed")
# Create folders if they don't exist
if not os.path.exists(intermediates_folder):
    os.makedirs(intermediates_folder)
if not os.path.exists(lum_processed_folder):
    os.makedirs(lum_processed_folder)
logging.info('DATA FOLDER: %s \n PROCESSED LUMINANCE DATA FOLDER: %s' % (stim_lum_folder, lum_processed_folder))
print('DATA FOLDER: %s \n PROCESSED LUMINANCE DATA FOLDER: %s' % (stim_lum_folder, lum_processed_folder))
###################################
# EXTRACT STIMULUS INFO
###################################
# timing info
downsampled_bucket_size_ms = 40
original_bucket_size_in_ms = 4
max_length_of_stim_vid = 60000 # milliseconds
no_of_time_buckets = max_length_of_stim_vid/original_bucket_size_in_ms
downsampled_no_of_time_buckets = max_length_of_stim_vid/downsampled_bucket_size_ms
new_time_bucket_sample_rate = downsampled_bucket_size_ms/original_bucket_size_in_ms
milliseconds_for_baseline = 3000
baseline_no_buckets = int(milliseconds_for_baseline/new_time_bucket_sample_rate)
smoothing_window = 25 # in time buckets, must be odd! for savgol_filter
# stim vid info
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
stim_name_to_float = {"stimuli024": 24.0, "stimuli025": 25.0, "stimuli026": 26.0, "stimuli027": 27.0, "stimuli028": 28.0, "stimuli029": 29.0}
stim_float_to_name = {24.0: "stimuli024", 25.0: "stimuli025", 26.0: "stimuli026", 27.0: "stimuli027", 28.0: "stimuli028", 29.0: "stimuli029"}
# find average luminance of stimuli vids
luminances = {key:[] for key in stim_vids}
luminances_avg = {key:[] for key in stim_vids}
luminances_baseline = {key:[] for key in stim_vids}
luminances_peaks = {key:[] for key in stim_vids}
luminance_data_paths = glob.glob(stim_lum_folder + "/*_stimuli*_world_LuminancePerFrame.csv")
## SEPARATE BY STIMULI NUMBER
for data_path in luminance_data_paths: 
    luminance_values = np.genfromtxt(data_path, dtype=np.str, delimiter='  ')
    luminance_values = np.array(luminance_values)
    stimulus_type = data_path.split("_")[-3]
    stimulus_num = stim_name_to_float[stimulus_type]
    luminances[stimulus_num].append(luminance_values)
# build average then smooth
for stimulus in luminances:
    print('Calculating average, smoothed luminance and peaks for stimuli {s}'.format(s=stimulus)) 
    logging.info('Calculating average, smoothed luminance and peaks for stimuli {s}'.format(s=stimulus)) 
    luminance_array = np.array(luminances[stimulus])
    # build average
    average_luminance = build_timebucket_avg_luminance(luminance_array, downsampled_bucket_size_ms, no_of_time_buckets)
    luminances_avg[stimulus] = average_luminance
    # baseline average
    baseline = np.nanmean(average_luminance[0:baseline_no_buckets])
    avg_lum_baselined = [((x-baseline)/baseline) for x in average_luminance]
    avg_lum_base_array = np.array(avg_lum_baselined)
    luminances_baseline[stimulus] = avg_lum_base_array
    # smooth average
    avg_lum_smoothed = savgol_filter(avg_lum_base_array, smoothing_window-10, 3)
    luminances_avg[stimulus] = avg_lum_smoothed
    # find peaks
    lum_peaks, _ = find_peaks(avg_lum_smoothed, height=-1, prominence=0.1)
    luminances_peaks[stimulus] = lum_peaks
# store processed luminance data in .npz file
for stim_key in stim_vids:
    luminances_avg[stim_float_to_name[stim_key]] = luminances_avg.pop(stim_key, None)
    luminances_peaks[stim_float_to_name[stim_key]] = luminances_peaks.pop(stim_key, None)
    luminances[stim_float_to_name[stim_key]] = luminances.pop(stim_key, None)
lum_avg_path = lum_processed_folder + os.sep + 'processed_lum_avg.npz'
lum_peaks_path = lum_processed_folder + os.sep + 'processed_lum_peaks.npz'
lum_path = lum_processed_folder + os.sep + 'processed_lum.npz'
np.savez(lum_avg_path, **luminances_avg)
np.savez(lum_peaks_path, **luminances_peaks)
np.savez(lum_path, **luminances)
# FIN