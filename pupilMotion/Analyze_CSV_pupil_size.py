import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import itertools
import matplotlib.animation as animation

### FUNCTIONS ###
def load_daily_pupil_areas(which_eye, day_folder_path, max_no_of_buckets, original_bucket_size, new_bucket_size): 
    # "right", csv_folder, no_of_time_buckets, original_bucket_size_in_ms, downsample_rate_ms
    if (new_bucket_size % original_bucket_size == 0):
        new_sample_rate = int(new_bucket_size/original_bucket_size)
        #print("New bucket window = {size}, need to average every {sample_rate} buckets".format(size=new_bucket_size, sample_rate=new_sample_rate))
        downsampled_no_of_buckets = math.ceil(max_no_of_buckets/new_sample_rate)
        #print("Downsampled number of buckets = {number}".format(number=downsampled_no_of_buckets))
        # List all csv trial files
        trial_files = glob.glob(day_folder_path + os.sep + which_eye + "*.csv")
        num_trials = len(trial_files)
        good_trials = num_trials
        # contours
        data_contours_X = np.empty((num_trials, downsampled_no_of_buckets+1))
        data_contours_X[:] = -6
        data_contours_Y = np.empty((num_trials, downsampled_no_of_buckets+1))
        data_contours_Y[:] = -6
        data_contours = np.empty((num_trials, downsampled_no_of_buckets+1))
        data_contours[:] = -6
        # circles
        data_circles_X = np.empty((num_trials, downsampled_no_of_buckets+1))
        data_circles_X[:] = -6
        data_circles_Y = np.empty((num_trials, downsampled_no_of_buckets+1))
        data_circles_Y[:] = -6
        data_circles = np.empty((num_trials, downsampled_no_of_buckets+1))
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
            if longest_cluster<100:
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

def list_sub_folders(path_to_root_folder):
    # List all sub folders
    sub_folders = []
    for folder in os.listdir(path_to_root_folder):
        if(os.path.isdir(os.path.join(path_to_root_folder, folder))):
            sub_folders.append(os.path.join(path_to_root_folder, folder))
    return sub_folders

def threshold_to_nan(array_of_arrays, threshold, upper_or_lower):
    for array in array_of_arrays:
        for index in range(len(array)): 
            if upper_or_lower=='upper':
                if np.isnan(array[index])==False and array[index]>threshold:
                    array[index] = np.nan
            if upper_or_lower=='lower':
                if np.isnan(array[index])==False and array[index]<threshold:
                    array[index] = np.nan
    return array_of_arrays

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

def build_timebucket_avg_luminance(timestamps_and_luminance_array, new_bucket_size_ms, max_no_of_timebuckets):
    bucket_window = datetime.timedelta(milliseconds=new_bucket_size_ms)
    avg_luminance_by_timebucket = []
    index = 0
    for trial in timestamps_and_luminance_array:
        first_timestamp = trial[0][0]
        end_timestamp = trial[-1][0]
        this_trial_timebuckets = make_luminance_time_buckets(first_timestamp, new_bucket_size_ms, end_timestamp)
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
    avg_lum_by_tb_thresholded = threshold_to_nan(avg_luminance_by_timebucket, 0, 'lower')
    avg_lum_by_tb_thresh_array = np.array(avg_lum_by_tb_thresholded)
    avg_lum_final = np.nanmean(avg_lum_by_tb_thresh_array, axis=0)
    return avg_lum_final

### NEED TO WRITE THESE FUNCTIONS
### WRITE A SACCADE DETECTOR
# frame by frame change in xy
### WRITE A FUNCTION TO FIND A PERSON'S "VIEW SPACE" BASED ON CALIBRATION SEQUENCE
# make a crude linear calibration
### what is the distance between eyes and monitor in the exhibit??


### BEGIN ANALYSIS ###
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
# List relevant data locations: these are for KAMPFF-LAB-VIDEO
#root_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
current_working_directory = os.getcwd()
stimuli_luminance_folder = r"C:\Users\taunsquared\Documents\GitHub\SurprisingMinds-Analysis\PythonWithAdam\bonsai\LuminancePerFrame"

# set up log file to store all printed messages
log_filename = "pupil-plotting_log_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
log_file = os.path.join(current_working_directory, log_filename)
sys.stdout = open(log_file, "w")

# set up folders
plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
pupils_folder = os.path.join(plots_folder, "pupil")
engagement_folder = os.path.join(plots_folder, "engagement")
linReg_folder = os.path.join(plots_folder, "linReg")

# Create plots folder (and sub-folders) if it (they) does (do) not exist
if not os.path.exists(plots_folder):
    #print("Creating plots folder.")
    os.makedirs(plots_folder)
if not os.path.exists(pupils_folder):
    #print("Creating camera profiles folder.")
    os.makedirs(pupils_folder)
if not os.path.exists(engagement_folder):
    #print("Creating engagement count folder.")
    os.makedirs(engagement_folder)
if not os.path.exists(linReg_folder):
    #print("Creating engagement count folder.")
    os.makedirs(linReg_folder)

# consolidate csv files from multiple days into one data structure
day_folders = list_sub_folders(root_folder)
# first day was a debugging session, so skip it
day_folders = day_folders[1:]
# currently still running pupil finding analysis...
day_folders = day_folders[:-1]

all_right_trials_contours_X = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_right_trials_contours_Y = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_right_trials_contours = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_right_trials_circles_X = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_right_trials_circles_Y = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_right_trials_circles = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_left_trials_contours_X = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_left_trials_contours_Y = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_left_trials_contours = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_left_trials_circles_X = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_left_trials_circles_Y = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_left_trials_circles = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}

stim_name_to_float = {"stimuli024": 24.0, "stimuli025": 25.0, "stimuli026": 26.0, "stimuli027": 27.0, "stimuli028": 28.0, "stimuli029": 29.0}
stim_float_to_name = {24.0: "stimuli024", 25.0: "stimuli025", 26.0: "stimuli026", 27.0: "stimuli027", 28.0: "stimuli028", 29.0: "stimuli029"}

all_trials_position_data = [all_right_trials_contours_X, all_right_trials_contours_Y, all_right_trials_circles_X, all_right_trials_circles_Y, all_left_trials_contours_X, all_left_trials_contours_Y, all_left_trials_circles_X, all_left_trials_circles_Y]
all_trials_size_data = [all_right_trials_contours, all_right_trials_circles, all_left_trials_contours, all_left_trials_circles]
activation_count = []
analysed_count = []

# downsample = collect data from every 40ms or other multiples of 20
downsample_rate_ms = 20
original_bucket_size_in_ms = 4
no_of_time_buckets = 20000
new_time_bucket_ms = downsample_rate_ms/original_bucket_size_in_ms
milliseconds_for_baseline = 2000
baseline_no_buckets = int(milliseconds_for_baseline/new_time_bucket_ms)

for day_folder in day_folders: 
    # for each day...
    day_folder_path = os.path.join(root_folder, day_folder)
    analysis_folder = os.path.join(day_folder_path, "Analysis")
    csv_folder = os.path.join(analysis_folder, "csv")

    # Print/save number of users per day
    day_name = day_folder.split("_")[-1]
    try: 
        ## EXTRACT PUPIL SIZE AND POSITION
        right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles, num_right_activations, num_good_right_trials = load_daily_pupil_areas("right", csv_folder, no_of_time_buckets, original_bucket_size_in_ms, downsample_rate_ms)
        left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles, num_left_activations, num_good_left_trials = load_daily_pupil_areas("left", csv_folder, no_of_time_buckets, original_bucket_size_in_ms, downsample_rate_ms)

        analysed_count.append((num_good_right_trials, num_good_left_trials))
        activation_count.append((num_right_activations, num_left_activations))
        print("On {day}, exhibit was activated {right_count} times (right) and {left_count} times (left), with {right_good_count} good right trials and {left_good_count} good left trials".format(day=day_name, right_count=num_right_activations, left_count=num_left_activations, right_good_count=num_good_right_trials, left_good_count=num_good_left_trials))

        # separate by stimulus number
        R_contours_X = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        R_contours_X_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in right_area_contours_X:
            stim_num = data[-1]
            if stim_num in R_contours_X.keys():
                R_contours_X[stim_num].append(data[:-1])

        R_contours_Y = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        R_contours_Y_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in right_area_contours_Y:
            stim_num = data[-1]
            if stim_num in R_contours_Y.keys():
                R_contours_Y[stim_num].append(data[:-1])

        R_contours = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        R_contours_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in right_area_contours:
            stim_num = data[-1]
            if stim_num in R_contours.keys():
                R_contours[stim_num].append(data[:-1])

        R_circles_X = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        R_circles_X_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in right_area_circles_X:
            stim_num = data[-1]
            if stim_num in R_circles_X.keys():
                R_circles_X[stim_num].append(data[:-1])

        R_circles_Y = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        R_circles_Y_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in right_area_circles_Y:
            stim_num = data[-1]
            if stim_num in R_circles_Y.keys():
                R_circles_Y[stim_num].append(data[:-1])

        R_circles = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        R_circles_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in right_area_circles:
            stim_num = data[-1]
            if stim_num in R_circles.keys():
                R_circles[stim_num].append(data[:-1])

        L_contours_X = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        L_contours_X_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in left_area_contours_X:
            stim_num = data[-1]
            if stim_num in L_contours_X.keys():
                L_contours_X[stim_num].append(data[:-1])

        L_contours_Y = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        L_contours_Y_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in left_area_contours_Y:
            stim_num = data[-1]
            if stim_num in L_contours_Y.keys():
                L_contours_Y[stim_num].append(data[:-1])

        L_contours = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        L_contours_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in left_area_contours:
            stim_num = data[-1]
            if stim_num in L_contours.keys():
                L_contours[stim_num].append(data[:-1])

        L_circles_X = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        L_circles_X_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in left_area_circles_X:
            stim_num = data[-1]
            if stim_num in L_circles_X.keys():
                L_circles_X[stim_num].append(data[:-1])

        L_circles_Y = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        L_circles_Y_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in left_area_circles_Y:
            stim_num = data[-1]
            if stim_num in L_circles_Y.keys():
                L_circles_Y[stim_num].append(data[:-1])

        L_circles = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        L_circles_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
        for data in left_area_circles:
            stim_num = data[-1]
            if stim_num in L_circles.keys():
                L_circles[stim_num].append(data[:-1])

        # filter data for outlier points
        all_position_data = [R_contours_X, R_contours_Y, R_circles_X, R_circles_Y, L_contours_X, L_contours_Y, L_circles_X, L_circles_Y]
        all_position_baselines = [R_contours_X_baseline, R_contours_Y_baseline, R_circles_X_baseline, R_circles_Y_baseline, L_contours_X_baseline, L_contours_Y_baseline, L_circles_X_baseline, L_circles_Y_baseline]
        all_size_data = [R_contours, R_circles, L_contours, L_circles]
        all_size_baselines = [R_contours_baseline, R_circles_baseline, L_contours_baseline, L_circles_baseline]
        # remove:
        # eye positions that are not realistic
        # time buckets with no corresponding frames
        # video pixel limits are (798,599)
        for data_type in all_position_data:
            for stimulus in data_type: 
                data_type[stimulus] = threshold_to_nan(data_type[stimulus], 798, 'upper')
                data_type[stimulus] = threshold_to_nan(data_type[stimulus], 0, 'lower')
        # contours/circles that are too big
        for data_type in all_position_data:
            for stimulus in data_type: 
                data_type[stimulus] = threshold_to_nan(data_type[stimulus], 15000, 'upper')
                data_type[stimulus] = threshold_to_nan(data_type[stimulus], 0, 'lower')

        # create a baseline
        # not sure if baseline is needed for position data??
        """ for x in range(len(all_position_data)):
            for stimulus in all_position_data[x]: 
                for trial in all_position_data[x][stimulus]:
                    baseline = np.nanmedian(trial[:baseline_no_buckets])
                    all_position_baselines[x][stimulus].append(baseline) """
        for x in range(len(all_size_data)):
            for stimulus in all_size_data[x]: 
                for trial in all_size_data[x][stimulus]:
                    baseline = np.nanmedian(trial[:baseline_no_buckets])
                    all_size_baselines[x][stimulus].append(baseline)

        # normalize and append
        for x in range(len(all_position_data)):
            for stimulus in all_position_data[x]:
                for index in range(len(all_position_baselines[x][stimulus])):
                    all_position_data[x][stimulus][index] = (all_position_data[x][stimulus][index]-all_position_baselines[x][stimulus][index])/all_position_baselines[x][stimulus][index]
                    all_trials_position_data[x][stimulus].append(all_position_data[x][stimulus][index])
                    #print(all_trials_position_data[x][stimulus])
        for x in range(len(all_size_data)):
            for stimulus in all_size_data[x]:
                for index in range(len(all_size_baselines[x][stimulus])):
                    all_size_data[x][stimulus][index] = (all_size_data[x][stimulus][index]-all_size_baselines[x][stimulus][index])/all_size_baselines[x][stimulus][index]
                    all_trials_size_data[x][stimulus].append(all_size_data[x][stimulus][index])
        print("Day {day} succeeded!".format(day=day_name))
    except Exception:
        print("Day {day} failed!".format(day=day_name))


### NOTES FROM LAB MEETING ###
# based on standard dev of movement, if "eye" doesn't move enough, don't plot that trial
## during tracking, save luminance of "darkest circle"
# if there is too much variability in frame rate, then don't plot that trial
# if standard dev of diameter is "too big", then don't plot

# find average luminance of stimuli vids
# octo_clip_start for stimuli videos
# stimuli024 = 169
# stimuli025 = 173
# stimuli026 = 248
# stimuli027 = 180
# stimuli028 = 247
# stimuli029 = 314
luminances = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
luminances_avg = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
luminances_baseline = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
luminance_data_paths = glob.glob(stimuli_luminance_folder + "/*_stimuli*_world_LuminancePerFrame.csv")
## NEED TO SEPARATE BY STIMULI NUMBER
for data_path in luminance_data_paths: 
    luminance_values = np.genfromtxt(data_path, dtype=np.str, delimiter='  ')
    luminance_values = np.array(luminance_values)
    stimulus_type = data_path.split("_")[-3]
    stimulus_num = stim_name_to_float[stimulus_type]
    luminances[stimulus_num].append(luminance_values)
for stimulus in luminances: 
    luminance_array = np.array(luminances[stimulus])
    average_luminance = build_timebucket_avg_luminance(luminance_array, downsample_rate_ms, no_of_time_buckets)
    luminances_avg[stimulus].append(average_luminance)
    baseline = np.nanmean(average_luminance[0:baseline_no_buckets])
    avg_lum_baselined = [((x-baseline)/baseline) for x in average_luminance]
    avg_lum_base_array = np.array(avg_lum_baselined)
    luminances_baseline[stimulus].append(average_luminance)

### PUPILS ###
# position and movement
all_right_positions = [all_right_trials_contours_X, all_right_trials_contours_Y, all_right_trials_circles_X, all_right_trials_circles_Y]
all_left_positions = [all_left_trials_contours_X, all_left_trials_contours_Y, all_left_trials_circles_X, all_left_trials_circles_Y]
# currently we are not pairing right and left eye coordinates



# average pupil diameters
all_right_sizes = [all_right_trials_contours, all_right_trials_circles]
all_left_sizes = [all_left_trials_contours, all_left_trials_circles]
all_right_size_contours_means = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_left_size_contours_means = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_right_size_circles_means = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_left_size_circles_means = {24.0:[], 25.0:[], 26.0:[], 27.0:[], 28.0:[], 29.0:[]}
all_right_size_means = [all_right_size_contours_means, all_right_size_circles_means]
all_left_size_means = [all_left_size_contours_means, all_left_size_circles_means]
# Compute global mean
for i in range(len(all_right_sizes)):
    for stimulus in all_right_sizes[i]: 
        all_right_size_means[i][stimulus].append(np.nanmean(all_right_sizes[i][stimulus], 0))
for i in range(len(all_left_sizes)):
    for stimulus in all_left_sizes[i]: 
        all_left_size_means[i][stimulus].append(np.nanmean(all_left_sizes[i][stimulus], 0))


### EXHIBIT ACTIVITY METADATA ### 
# Save activation count to csv
engagement_count_filename = 'Exhibit_Activation_Count_measured-' + todays_datetime + '.csv'
engagement_data_folder = os.path.join(current_working_directory, 'Exhibit-Engagement')
if not os.path.exists(engagement_data_folder):
    #print("Creating plots folder.")
    os.makedirs(engagement_data_folder)
csv_file = os.path.join(engagement_data_folder, engagement_count_filename)
np.savetxt(csv_file, activation_count, fmt='%.2f', delimiter=',')

# Plot activation count - IMPROVE AXIS LABELS
total_activation = sum(count[0] for count in activation_count)
total_days_activated = len(activation_count)
good_trials_right = [count[0] for count in analysed_count]
good_trials_left = [count[1] for count in analysed_count]
total_good_trials_right = sum(good_trials_right)
total_good_trials_left = sum(good_trials_left)
print("Total number of exhibit activations: {total}".format(total=total_activation))
print("Total number of good right eye camera trials: {good_total}".format(good_total=total_good_trials_right))
print("Total number of good left eye camera trials: {good_total}".format(good_total=total_good_trials_left))
activation_array = np.array(activation_count)
analysed_array_right = np.array(good_trials_right)
analysed_array_left = np.array(good_trials_left)
# do da plot
image_type_options = ['.png', '.pdf']

### NEED BETTER PLOTS FOR EXHIBIT ENGAGEMENT
# activation based on: 
# day of the week
# time of the day
# month of the year
# language chosen
""" ## PLOT EXHIBIT ENGAGEMENT ##
for image_type in image_type_options:
    figure_name = 'TotalExhibitActivation_' + todays_datetime + image_type
    figure_path = os.path.join(engagement_folder, figure_name)
    figure_title = "Total number of exhibit activations per day (Grand Total: " + str(total_activation) + ") \n Total good trials from right eye camera (red): " + str(total_good_trials_right) + "\n Total good trials from left eye camera (green): " + str(total_good_trials_left) + "\nPlotted on " + todays_datetime
    plt.figure(figsize=(27, 9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)

    plt.ylabel('Number of activations', fontsize=11)
    plt.xlabel('Days, Total days activated: ' + str(total_days_activated), fontsize=11)
    #plt.minorticks_on()
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.plot(activation_array, color=[0.0, 0.0, 1.0])
    plt.plot(analysed_array_right, color=[1.0, 0.0, 0.0, 0.4])
    plt.plot(analysed_array_left, color=[0.0, 1.0, 0.0, 0.4])

    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close() """

### PLOTTING PUPIL STUFF ###
# Plot pupil sizes
plot_types = ["contours", "circles"]
stimuli = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
for stim_type in stimuli: 
    for i in range(len(all_right_sizes)): 
        plot_type_right = np.array(all_right_sizes[i][stim_type])
        plot_N_right = len(all_right_sizes[i][stim_type])
        plot_type_left = np.array(all_left_sizes[i][stim_type])
        plot_N_left = len(all_left_sizes[i][stim_type])
        plot_means_right = np.array(all_right_size_means[i][stim_type])[0]
        plot_means_left = np.array(all_left_size_means[i][stim_type])[0]
        plot_luminance = np.array(luminances_avg[stim_type])[0]
        plot_type_name = plot_types[i]
        stim_name = stim_float_to_name[stim_type]
        dpi_sizes = [200]
        for size in dpi_sizes: 
            figure_name = 'AveragePupilSizes_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(size) + '.png' 
            figure_path = os.path.join(pupils_folder, figure_name)
            figure_title = "Pupil sizes of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPlotted on " + todays_datetime
            plt.figure(figsize=(14, 14), dpi=size)
            plt.suptitle(figure_title, fontsize=12, y=0.98)

            plt.subplot(3,1,1)
            ax = plt.gca()
            ax.yaxis.set_label_coords(-0.09, -0.5) 
            plt.title('Right eye pupil sizes; N = ' + str(plot_N_right), fontsize=9, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.plot(plot_type_right.T, '.', MarkerSize=1, color=[0.0, 0.0, 1.0, 0.01])
            plt.plot(plot_means_right, linewidth=1.5, color=[1.0, 0.0, 0.0, 0.4])
            plt.xlim(-10,2500)
            plt.ylim(-1,1)
            
            plt.subplot(3,1,2)
            plt.ylabel('Percentage from baseline', fontsize=11)
            plt.title('Left eye pupil sizes; N = ' + str(plot_N_left), fontsize=9, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.plot(plot_type_left.T, '.', MarkerSize=1, color=[0.0, 1.0, 0.0, 0.01])
            plt.plot(plot_means_left, linewidth=1.5, color=[1.0, 0.0, 0.0, 0.4])
            plt.xlim(-10,2500)
            plt.ylim(-1,1)
            
            plt.subplot(3,1,3)
            plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsample_rate_ms) + 'ms)', fontsize=11)
            plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stim_type])), fontsize=9, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.plot(plot_luminance, linewidth=2, color=[1.0, 0.0, 1.0, 1])
            plt.xlim(-10,2500)
            #plt.ylim(-1.0,1.0)
            # mark events
            #for i in range(len(event_labels)):
            #    plt.plot((event_locations[i],event_locations[i]), (0.25,2.2-((i-1)/5)), 'k-', linewidth=1)
            #    plt.text(event_locations[i]+1,2.2-((i-1)/5), event_labels[i], fontsize='x-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.35'))
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

##################################################

### -------------------------- ###
### UNDER CONSTRUCTION!!!!!!!! ###
### -------------------------- ###
### LINEAR REGRESSION ANALYSIS ###
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D  

# offset, to account for latency of pupillary response. best latency = 20 time bucket delay
latency = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
X_test_all = []
X_test_frames_all = []
model_linX_prediction_all = []
model_linXframes_prediction_all = []
model_Xdof2_prediction_all = []
r2_lum_training_scores = []
r2_lum_test_scores = []
r2_lumFrames_training_scores = []
r2_lumFrames_test_scores = []
r2_Xdof2_training_scores = []
r2_Xdof2_test_scores = []
offsets_ms = []
for offset_frames in latency: 
    print(str(offset_frames))
    offset_ms = int(offset_frames*downsample_rate_ms)
    offsets_ms.append(offset_ms)
    print(offsets_ms)
    start = 0
    end = 480
    all_left_circles_mean_trimmed = all_left_circles_mean[(start+offset_frames):(end+offset_frames)]
    avg_lum_base_trimmed = avg_lum_base_array[start:end]
    # Build linear regression model using Avg Luminance as predictor
    # Split data into predictors X and output Y
    X = avg_lum_base_trimmed.reshape(-1,1)
    y = all_left_circles_mean_trimmed
    # divide dataset into training and test portions
    # training data - first 100 frames
    train_start = 0
    train_end = 99
    # test data - frame 101 to end
    test_start = 100
    test_end = len(X)
    X_training = X[train_start:train_end]
    X_test = X[test_start:test_end]
    y_training = y[train_start:train_end]
    y_test = y[test_start:test_end]
    # add frame number as a predictor
    relative_frame_numbers_training = np.arange(len(X_training))
    relative_frame_numbers_test = np.arange(len(X_test))
    X_frames_training = np.empty((len(X_training), 2))
    for i in range(len(X_training)):
        X_frames_training[i] = [X_training[i], relative_frame_numbers_training[i]]
    X_frames_test = np.empty((len(X_test), 2))
    for i in range(len(X_test)):
        X_test_frames[i] = [X_test[i], relative_frame_numbers_test[i]]

    # Initialise and fit model
    # linear: just luminance values
    model_linX = LinearRegression().fit(X_training, y_training)
    # multiple linear: luminance values + frame number
    model_linXframes = LinearRegression().fit(X_training_frames, y_training)
    # quadratic
    X_dof2_training = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_training)
    X_dof2_test = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_test)
    # Initialise and fit model
    model_X_dof2 = LinearRegression().fit(X_dof2_training,y_training)

    # Print Coefficients
    print(f'beta_0 = {model_linX.intercept_}')
    print(f'beta = {model_linX.coef_}')
    print(f'beta_0_frames = {model_linXframes.intercept_}')
    print(f'betas_frames = {model_linXframes.coef_}')
    print(f'beta_0_X_dof2 = {model_X_dof2.intercept_}')
    print(f'betas_X_dof2 = {model_X_dof2.coef_}')

    # predicted response
    model_linX_prediction = model_linX.predict(X_test)
    model_linX_prediction_all.append(model_linX_prediction)
    model_linXframes_prediction = model_linXframes.predict(X_frames_test)
    model_linXframes_prediction_all.append(model_linXframes_prediction)
    model_Xdof2_prediction = model_X_dof2.predict(X_dof2_test)
    model_Xdof2_prediction_all.append(model_Xdof2_prediction)

    #r^2 (coefficient of determination) regression score function.
    r2_lum_training = model_linX.score(X_training,y_training)
    r2_lum_training_scores.append(r2_lum_training)
    r2_lum_test = model_linX.score(X_test,y_test)
    r2_lum_test_scores.append(r2_lum_test)

    r2_lumFrames_training = model_linXframes.score(X_frames_training,y_training)
    r2_lumFrames_training_scores.append(r2_lumFrames_training)
    r2_lumFrames_test = model_linXframes.score(X_frames_test,y_test)
    r2_lumFrames_test_scores.append(r2_lumFrames_test)

    r2_Xdof2_training = model_X_dof2.score(X_dof2_training,y_training)
    r2_Xdof2_training_scores.append(r2_Xdof2_training)
    r2_Xdof2_test = model_X_dof2.score(X_dof2_test,y_test)
    r2_Xdof2_test_scores.append(r2_Xdof2_test)

    print(f'linear model (luminance) = {r2_lum_training}')
    print(f'multiple linear model (luminance + frame number) = {r2_lumFrames_training}')
    print(f'polynomial model, luminance, 2 dof = {r2_Xdof2_training}')

# plot R^2 scores for each offset
y_pos = np.arange(len(offsets_ms))
training_scores = [r2_lum_training_scores, r2_lumFrames_training_scores, r2_Xdof2_training_scores]
test_scores = [r2_lum_test_scores, r2_lumFrames_test_scores, r2_Xdof2_test_scores]
y_labels = ['$R^2 scores$, luminance', '$R^2 scores$, luminance+frames', '$R^2 scores$, DoF=2', '$R^2 scores$, DoF=3']

#for i in range(len(scores)): 
i = 0
r2_scores_plot_title = 'Comparison of goodness-of-fit for different latencies (ms) \nTraining data: first 100 frames'
plt.figure(dpi=200)
plt.suptitle(r2_scores_plot_title, fontsize=10, y=0.98)
bars = plt.bar(y_pos, training_scores[i], color='red', align='center', alpha=0.5)
plt.ylim(0,1)
plt.xticks(y_pos, offsets_ms)
plt.ylabel(y_labels[i])
for rect in bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.4f' % height, ha='center', va='bottom', fontsize=6, rotation=60)
plt.legend()
plt.tight_layout()
plt.show()

# plot the best model
# index of best latency
best_index = 0
# linear
figure2d_name = 'LinearReg2d_AvgLum-LeftCirclesMean_offset' + str(offset_ms[best_index]) + 'ms_' + todays_datetime + '.png'
figure2d_path = os.path.join(linReg_folder, figure2d_name)
figure2d_title = "Average Luminance of Stimuli vs Average Pupil Size (left eye) \nAverage Pupil Size offset by " + str(offset_ms[best_index]) + "ms to account for latency of pupillary response"
plt.figure(dpi=200)
plt.suptitle(figure2d_title, fontsize=10, y=0.98)

plt.scatter(X_test, y_test)
plt.plot(X_test, model_linX_prediction[best_index], 'yellow')
plt.plot(X_test, model_Xdof2_prediction[best_index], '.r')
plt.plot(X_test, model_Xdof3_prediction[best_index], 'lime')
plt.ylim(-0.25,0.4)
plt.xlabel("Average Percent Change from Baseline of Luminance of Stimuli")
plt.ylabel("Average Percent Change from Baseline of Pupil Size (left eye)")
plt.text(-0.2,0.3, '$R^2$ score, linear (luminance, yellow) = ' + str(r2_lum_scores[best_index]) + "\n$R^2$ score, polynomial (DOF=2, red) = " + str(r2_Xdof2_scores[best_index]) + "\n$R^2$ score, polynomial (DOF=3, green) = " + str(r2_Xdof3_scores[best_index]), fontsize='x-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.35'))
plt.savefig(figure2d_path)
plt.show(block=False)
plt.pause(1)
plt.close()

# multiple linear
figure2d_name = 'LinearReg2d_AvgLum-LeftCirclesMean_offset' + str(offset_ms) + 'ms_' + todays_datetime + '.png'
figure2d_path = os.path.join(linReg_folder, figure2d_name)
figure2d_title = "Average Luminance of Stimuli vs Average Pupil Size (left eye) \nAverage Pupil Size offset by " + str(offset_ms) + "ms to account for latency of pupillary response"
plt.figure(dpi=200)
plt.suptitle(figure2d_title, fontsize=10, y=0.98)

plt.scatter(relative_frame_numbers, X)
plt.plot(X_frames, model_linXframes, 'yellow')
plt.plot(X_frames, model_Xframes_dof2, '.r')
plt.plot(X_frames, model_Xframes_dof3, 'lime')
plt.ylim(-0.25,0.4)
plt.xlabel("Average Percent Change from Baseline of Luminance of Stimuli")
plt.ylabel("Average Percent Change from Baseline of Pupil Size (left eye)")
plt.text(-0.2,0.3, '$R^2$ score, linear (luminance, yellow) = ' + str(r2_lum) + "\n$R^2$ score, linear (luminance + frame number, red) = " + str(r2_lumFrames) + "\n$R^2$ score, polynomial (DOF=2, green) = " + str(r2_dof2), fontsize='x-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.35'))
plt.savefig(figure2d_path)
plt.show(block=False)
plt.pause(1)
plt.close()

# visualize this in 3d
figure3d_name = 'LinearReg3d_AvgLum-LeftCirclesMean_offset' + str(offset_ms) + 'ms_' + todays_datetime + '.png'
figure3d_path = os.path.join(linReg_folder, figure3d_name)
figure3d_title = "Average Luminance of Stimuli vs Average Pupil Size (left eye) \nAverage Pupil Size offset by " + str(offset_ms) + "ms to account for latency of pupillary response"
plt.figure(dpi=200)
plt.suptitle(figure3d_title, fontsize=10, y=0.98)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, X_frames, model_linXframes_prediction)
ax.scatter(X, X_dof2, y)
ax.view_init(elev=40., azim=-45)
ax.set_xlabel('Avg Luminance')
ax.set_ylabel('$(Avg Lum)^2$')
ax.set_zlabel('Avg pupil size')
plt.savefig(figure3d_path)
plt.show(block=False)
plt.pause(1)
plt.close

### END OF CONSTRUCTION ###
### ------------------- ###

# event locations in time - NEED TO THINK ABOUT HOW TO DO THIS 
milliseconds_until_octo_fully_decamoud = 6575
milliseconds_until_octopus_inks = 11500
milliseconds_until_octopus_disappears = 11675
milliseconds_until_camera_out_of_ink_cloud = 13000
milliseconds_until_thankyou_screen = 15225
event_labels = ['Octopus video clip starts', 'Octopus fully decamouflaged', 'Octopus begins inking', 'Octopus disappears from camera view', 'Camera exits ink cloud', 'Thank you screen']
tb_octo_decamoud = milliseconds_until_octo_fully_decamoud/downsample_rate_ms
tb_octo_inks = milliseconds_until_octopus_inks/downsample_rate_ms
tb_octo_disappears = milliseconds_until_octopus_disappears/downsample_rate_ms
tb_camera_out_of_ink_cloud = milliseconds_until_camera_out_of_ink_cloud/downsample_rate_ms
tb_thankyou_screen = milliseconds_until_thankyou_screen/downsample_rate_ms
event_locations = np.array([0, tb_octo_decamoud, tb_octo_inks, tb_octo_disappears, tb_camera_out_of_ink_cloud, tb_thankyou_screen])



#FIN