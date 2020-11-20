# -*- coding: utf-8 -*-
"""
Project: "Surprising Minds" at Sea Life Brighton, by Danbee Kim, Kerry Perkins, Clive Ramble, Hazel Garnade, Goncalo Lopes, Dario Quinones, Reanna Campbell-Russo, Robb Barrett, Martin Stopps, The EveryMind Team, and Adam Kampff. 
Analysis preprocessing: Consolidate daily eye position and size data

Collects csv files of pupil tracking data from all subjects/days of experiment/exhibit.
Extracts, filters, and downsamples pupil tracking data.
Outputs .npz file for each day:
all_position_X_data = [R_contours_X, R_circles_X, L_contours_X, L_circles_X]
all_position_Y_data = [R_contours_Y, R_circles_Y, L_contours_Y, L_circles_Y]
all_size_data = [R_contours, R_circles, L_contours, L_circles]
all_size_baselines = [R_contours_baseline, R_circles_baseline, L_contours_baseline, L_circles_baseline]

Each item in the above lists is a dictionary with structure {key:[] for key in stim_vids}.

@author: Adam R Kampff and Danbee Kim
"""
import os
import glob
import datetime
import numpy as np
import math
import itertools
from itertools import groupby
import csv
import logging
from joblib import Parallel, delayed
###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# FUNCTIONS
###################################
##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) dataset_dir (folder with csv files of full pupil tracking dataset)
# 2) plots_dir (parent folder for all plots output by this script)
# 3) intermediates_dir (folder for output of this script)
##########################################################
def load_data():
    #dataset_dir = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows'
    #intermediates_dir = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\intermediates'
    dataset_dir = r'D:\data\SurprisingMinds\dataPythonWorkflows'
    intermediates_dir = r'D:\data\SurprisingMinds\intermediates'
    return dataset_dir, intermediates_dir
##########################################################

def load_daily_pupils(which_eye, day_csv_folder_path, max_no_of_buckets, original_bucket_size, new_bucket_size, bad_trial_cutoff): 
    if (new_bucket_size % original_bucket_size == 0):
        new_sample_rate = int(new_bucket_size/original_bucket_size)
        max_no_of_buckets = int(max_no_of_buckets)
        #print("New bucket window = {size}, need to average every {sample_rate} buckets".format(size=new_bucket_size, sample_rate=new_sample_rate))
        # List all csv trial files
        trial_files = glob.glob(day_csv_folder_path + os.sep + which_eye + "*.csv")
        num_trials = len(trial_files)
        good_trials = num_trials
        # contours
        data_contours_X = np.empty((num_trials, max_no_of_buckets+1))
        data_contours_X[:] = -6
        data_contours_Y = np.empty((num_trials, max_no_of_buckets+1))
        data_contours_Y[:] = -6
        data_contours = np.empty((num_trials, max_no_of_buckets+1))
        data_contours[:] = -6
        # circles
        data_circles_X = np.empty((num_trials, max_no_of_buckets+1))
        data_circles_X[:] = -6
        data_circles_Y = np.empty((num_trials, max_no_of_buckets+1))
        data_circles_Y[:] = -6
        data_circles = np.empty((num_trials, max_no_of_buckets+1))
        data_circles[:] = -6

        index = 0
        for trial_file in trial_files:
            trial_name = trial_file.split(os.sep)[-1]
            trial_stimulus = trial_name.split("_")[1]
            trial_stim_number = np.float(trial_stimulus[-2:])
            trial = np.genfromtxt(trial_file, dtype=np.float, delimiter=",")
            # if there are too many -5 rows (frames) in a row, don't analyse this trial
            bad_frame_count = []
            for frame in trial:
                if frame[0]==-5:
                    bad_frame_count.append(1)
                else:
                    bad_frame_count.append(0)
            clusters =  [(x[0], len(list(x[1]))) for x in itertools.groupby(bad_frame_count)]
            longest_cluster = 0
            for cluster in clusters:
                if cluster[0] == 1 and cluster[1]>longest_cluster:
                    longest_cluster = cluster[1]
            #print("For trial {name}, the longest cluster is {length}".format(name=trial_name, length=longest_cluster))
            if longest_cluster<bad_trial_cutoff:
                no_of_samples = math.ceil(len(trial)/new_sample_rate)
                this_trial_contours_X = []
                this_trial_contours_Y = []
                this_trial_contours = []
                this_trial_circles_X = []
                this_trial_circles_Y = []
                this_trial_circles = []
                # loop through the trial at given sample rate
                for sample in range(no_of_samples):
                    start = sample * new_sample_rate
                    end = (sample * new_sample_rate) + (new_sample_rate - 1)
                    this_slice = trial[start:end]
                    for line in this_slice:
                        if (line<0).any():
                            line[:] = np.nan
                        if (line>15000).any():
                            line[:] = np.nan
                    # extract pupil sizes and locations from valid time buckets
                    this_slice_contours_X = []
                    this_slice_contours_Y = []
                    this_slice_contours = []
                    this_slice_circles_X = []
                    this_slice_circles_Y = []
                    this_slice_circles = []
                    for frame in this_slice:
                        # contour x,y
                        ## DON'T PAIR X-Y YET
                        this_slice_contours_X.append(frame[0])
                        this_slice_contours_Y.append(frame[1])
                        # contour area
                        this_slice_contours.append(frame[2])
                        # circles x,y
                        ## DON'T PAIR X-Y YET
                        this_slice_circles_X.append(frame[3])
                        this_slice_circles_Y.append(frame[4])
                        # circles area
                        this_slice_circles.append(frame[5])
                    # average the pupil size and movement in this sample slice
                    this_slice_avg_contour_X = np.nanmean(this_slice_contours_X)
                    this_slice_avg_contour_Y = np.nanmean(this_slice_contours_Y)
                    this_slice_avg_contour = np.nanmean(this_slice_contours) 
                    this_slice_avg_circle_X = np.nanmean(this_slice_circles_X)
                    this_slice_avg_circle_Y = np.nanmean(this_slice_circles_Y)       
                    this_slice_avg_circle = np.nanmean(this_slice_circles)
                    # append to list of downsampled pupil sizes and movements
                    this_trial_contours_X.append(this_slice_avg_contour_X)
                    this_trial_contours_Y.append(this_slice_avg_contour_Y)
                    this_trial_contours.append(this_slice_avg_contour)
                    this_trial_circles_X.append(this_slice_avg_circle_X)
                    this_trial_circles_Y.append(this_slice_avg_circle_Y)
                    this_trial_circles.append(this_slice_avg_circle)
                # Find count of bad measurements
                bad_count_contours_X = sum(np.isnan(this_trial_contours_X))
                bad_count_contours_Y = sum(np.isnan(this_trial_contours_Y))
                bad_count_contours = sum(np.isnan(this_trial_contours))
                bad_count_circles_X = sum(np.isnan(this_trial_circles_X))
                bad_count_circles_Y = sum(np.isnan(this_trial_circles_Y))
                bad_count_circles = sum(np.isnan(this_trial_circles))
                # if more than half of the trial is NaN, then throw away this trial
                # otherwise, if it's a good enough trial...
                bad_threshold = no_of_samples/2
                if (bad_count_contours_X<bad_threshold): 
                    this_chunk_length = len(this_trial_contours_X)
                    data_contours_X[index][0:this_chunk_length] = this_trial_contours_X
                    data_contours_X[index][-1] = trial_stim_number
                if (bad_count_contours_Y<bad_threshold): 
                    this_chunk_length = len(this_trial_contours_Y)
                    data_contours_Y[index][0:this_chunk_length] = this_trial_contours_Y
                    data_contours_Y[index][-1] = trial_stim_number
                if (bad_count_contours<bad_threshold) or (bad_count_circles<bad_threshold): 
                    this_chunk_length = len(this_trial_contours)
                    data_contours[index][0:this_chunk_length] = this_trial_contours
                    data_contours[index][-1] = trial_stim_number
                if (bad_count_circles_X<bad_threshold): 
                    this_chunk_length = len(this_trial_circles_X)
                    data_circles_X[index][0:this_chunk_length] = this_trial_circles_X
                    data_circles_X[index][-1] = trial_stim_number
                if (bad_count_circles_Y<bad_threshold): 
                    this_chunk_length = len(this_trial_circles_Y)
                    data_circles_Y[index][0:this_chunk_length] = this_trial_circles_Y
                    data_circles_Y[index][-1] = trial_stim_number
                if (bad_count_circles<bad_threshold): 
                    this_chunk_length = len(this_trial_circles)
                    data_circles[index][0:this_chunk_length] = this_trial_circles
                    data_circles[index][-1] = trial_stim_number
                index = index + 1
            else:
                #print("Discarding trial {name}".format(name=trial_name))
                index = index + 1
                good_trials = good_trials - 1
        return data_contours_X, data_contours_Y, data_contours, data_circles_X, data_circles_Y, data_circles, num_trials, good_trials
    else: 
        print("Sample rate must be a multiple of {bucket}".format(bucket=original_bucket_size))
        logging.info("Sample rate must be a multiple of {bucket}".format(bucket=original_bucket_size))

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

##########################################################
# BEGIN SCRIPT
##########################################################
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
logging.basicConfig(filename="pp01ExtractPupilCSVDownsample_" + todays_datetime + ".log", filemode='a', level=logging.INFO)
###################################
# SOURCE DATA AND OUTPUT FILE LOCATIONS 
###################################
pupil_csv_folder, output_folder = load_data()
downsampled_pupils_folder = os.path.join(output_folder, "downsampled_pupils")
# Create intermediates folder if it does not exist
if not os.path.exists(downsampled_pupils_folder):
    #print("Creating downsampled_pupils_folder.")
    os.makedirs(downsampled_pupils_folder)

logging.info('DATA FOLDER: %s \n OUTPUT FOLDER: %s' % (pupil_csv_folder, downsampled_pupils_folder))
print('DATA FOLDER: %s \n OUTPUT FOLDER: %s' % (pupil_csv_folder, downsampled_pupils_folder))
###################################
# PARAMETERS
###################################
N_CPU_available = 24
###################################
# FIND DAILY PUPIL TRACKING DATA
###################################
daily_folders = glob.glob(pupil_csv_folder + os.sep + 'SurprisingMinds_*')
# If you only want to find saccades in a subset of the data...
#daily_folders = daily_folders[10:100]
# first day was a debugging session, so skip it
pupil_folders = daily_folders[1:]
### --------------------------------------------- ###
### REMOVE THIS LINE WHEN PUPIL FINDING IS DONE!! ###
# if currently still running pupil finding analysis...
#pupil_folders = daily_folders[:-1]
### --------------------------------------------- ###
###################################
# DOWNSAMPLE
###################################
# collect data from every 40ms or other multiples of 20
downsampled_bucket_size_ms = 40
original_bucket_size_in_ms = 4
max_length_of_stim_vid = 60000 # milliseconds
no_of_time_buckets = max_length_of_stim_vid/original_bucket_size_in_ms
downsampled_no_of_time_buckets = max_length_of_stim_vid/downsampled_bucket_size_ms
new_time_bucket_sample_rate = downsampled_bucket_size_ms/original_bucket_size_in_ms
milliseconds_for_baseline = 3000
baseline_no_buckets = int(milliseconds_for_baseline/new_time_bucket_sample_rate)
###################################
# CUTOFF FOR DISCARDING TRIALS
###################################
bad_trial_cutoff = 200
###################################
# BEGIN PUPIL DATA EXTRACTION
###################################
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
def process_day(day_folder):
    logging.basicConfig(filename="pm01AnalyzeCSVPupilPosition_" + todays_datetime + ".log", filemode='a', level=logging.INFO)
    # for each day...
    day_folder_path = os.path.join(pupil_csv_folder, day_folder)
    analysis_folder = os.path.join(day_folder_path, "Analysis")
    csv_folder = os.path.join(analysis_folder, "csv")
    world_folder = os.path.join(analysis_folder, "world")
    # Print/save number of users per day
    day_name = day_folder.split("_")[-1]
    try: 
        ## EXTRACT PUPIL SIZE AND POSITION
        right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles, num_right_activations, num_good_right_trials = load_daily_pupils("right", csv_folder, downsampled_no_of_time_buckets, original_bucket_size_in_ms, downsampled_bucket_size_ms, bad_trial_cutoff)
        left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles, num_left_activations, num_good_left_trials = load_daily_pupils("left", csv_folder, downsampled_no_of_time_buckets, original_bucket_size_in_ms, downsampled_bucket_size_ms, bad_trial_cutoff)
        print("On {day}, exhibit was activated {right_count} times (right) and {left_count} times (left), with {right_good_count} good right trials and {left_good_count} good left trials".format(day=day_name, right_count=num_right_activations, left_count=num_left_activations, right_good_count=num_good_right_trials, left_good_count=num_good_left_trials))
        logging.info("On {day}, exhibit was activated {right_count} times (right) and {left_count} times (left), with {right_good_count} good right trials and {left_good_count} good left trials".format(day=day_name, right_count=num_right_activations, left_count=num_left_activations, right_good_count=num_good_right_trials, left_good_count=num_good_left_trials))
        # separate by stimulus number
        R_contours_X = {key:[] for key in stim_vids}
        R_contours_Y = {key:[] for key in stim_vids}
        R_contours = {key:[] for key in stim_vids}
        R_circles_X = {key:[] for key in stim_vids}
        R_circles_Y = {key:[] for key in stim_vids}
        R_circles = {key:[] for key in stim_vids}
        L_contours_X = {key:[] for key in stim_vids}
        L_contours_Y = {key:[] for key in stim_vids}
        L_contours = {key:[] for key in stim_vids}
        L_circles_X = {key:[] for key in stim_vids}
        L_circles_Y = {key:[] for key in stim_vids}
        L_circles = {key:[] for key in stim_vids}
        stim_sorted_data_right = [R_contours_X, R_contours_Y, R_contours, R_circles_X, R_circles_Y, R_circles]
        stim_sorted_data_left = [L_contours_X, L_contours_Y, L_contours, L_circles_X, L_circles_Y, L_circles]
        stim_sorted_data_all = [stim_sorted_data_right, stim_sorted_data_left]
        extracted_data_right = [right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles]
        extracted_data_left = [left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles]
        extracted_data_all = [extracted_data_right, extracted_data_left]
        for side in range(len(extracted_data_all)):
            for dataset in range(len(extracted_data_all[side])):
                for trial in extracted_data_all[side][dataset]:
                    stim_num = trial[-1]
                    if stim_num in stim_sorted_data_all[side][dataset].keys():
                        stim_sorted_data_all[side][dataset][stim_num].append(trial[:-1])
        # filter data for outlier points
        all_position_X_data = [R_contours_X, R_circles_X, L_contours_X, L_circles_X]
        all_position_Y_data = [R_contours_Y, R_circles_Y, L_contours_Y, L_circles_Y]
        all_size_data = [R_contours, R_circles, L_contours, L_circles]
        # remove:
        # eye positions that are not realistic
        # time buckets with no corresponding frames
        # video pixel limits are (798,599)
        all_position_X_data = filter_to_nan(all_position_X_data, 798, 0)
        all_position_Y_data = filter_to_nan(all_position_Y_data, 599, 0)
        # contours/circles that are too big
        all_size_data = filter_to_nan(all_size_data, 15000, 0)
        # create a baseline for size data
        R_contours_baseline = {key:[] for key in stim_vids}
        R_circles_baseline = {key:[] for key in stim_vids}
        L_contours_baseline = {key:[] for key in stim_vids}
        L_circles_baseline = {key:[] for key in stim_vids}
        all_size_baselines = [R_contours_baseline, R_circles_baseline, L_contours_baseline, L_circles_baseline]
        for dataset in range(len(all_size_data)):
            for stimulus in all_size_data[dataset]: 
                for trial in all_size_data[dataset][stimulus]:
                    baseline = np.nanmedian(trial[:baseline_no_buckets])
                    all_size_baselines[dataset][stimulus].append(baseline)
        # save to .npz file
        this_day_all_data_path = downsampled_pupils_folder + os.sep + day_name + '_totalR{right_count}_totalL{left_count}_goodR{right_good_count}_goodL{left_good_count}.npz'.format(right_count=num_right_activations, left_count=num_left_activations, right_good_count=num_good_right_trials, left_good_count=num_good_left_trials)
        np.savez(this_day_all_data_path, all_pos_x=all_position_X_data, all_pos_y=all_position_Y_data, all_size=all_size_data, all_size_base=all_size_baselines)
        print("Day {day} succeeded!".format(day=day_name))
        logging.info("Day {day} succeeded!".format(day=day_name))
    except Exception as e:
        print("Day {day} failed!".format(day=day_name))
        print(e)
        logging.info("Day {day} failed!".format(day=day_name))
        logging.info(e)

Parallel(n_jobs=N_CPU_available)(delayed(process_day)(day_folder) for day_folder in pupil_folders)
###################################
# EXTRACTION COMPLETE
###################################

# FIN