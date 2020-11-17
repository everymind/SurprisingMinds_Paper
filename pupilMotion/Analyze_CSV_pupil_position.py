## PLOTTING MOTION
# should be the difference between frame 1 and frame 0
# frame1(x,y) - frame0(x,y), then take sqrt((delta x)^2 + (delta y)^2)
# this is a saccade vs fixation vs smooth pursuit detector
# save the previous frame's (x,y) and subtract that from current frame's (x,y)
# use filtering for pupil diameter first, then build filtering for pupil motion on top of that 

import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import itertools

### FUNCTIONS ###
def load_daily_pupil_positions(which_eye, day_folder_path, max_no_of_buckets, original_bucket_size, new_bucket_size): 
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
        data_contours = np.empty((num_trials, downsampled_no_of_buckets))
        data_contours[:] = -6
        # circles
        data_circles = np.empty((num_trials, downsampled_no_of_buckets))
        data_circles[:] = -6

        index = 0
        for trial_file in trial_files:
            trial_name = trial_file.split(os.sep)[-1]
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
                this_trial_contours = []
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
                    # extract pupil sizes from valid time buckets
                    this_slice_contours = []
                    this_slice_circles = []
                    for frame in this_slice:
                        ### ADD LINES TO EXTRACT EYE MOTION ((X,Y) COORDINATES)
                        this_slice_contours.append(frame[2])
                        this_slice_circles.append(frame[5])
                    # average the pupil size in this sample slice
                    this_slice_avg_contour = np.nanmean(this_slice_contours)            
                    this_slice_avg_circle = np.nanmean(this_slice_circles)
                    # append to list of downsampled pupil sizes
                    this_trial_contours.append(this_slice_avg_contour)
                    this_trial_circles.append(this_slice_avg_circle)
                # Find count of bad measurements
                bad_count_contours = sum(np.isnan(this_trial_contours))
                bad_count_circles = sum(np.isnan(this_trial_circles))
                # if more than half of the trial is NaN, then throw away this trial
                # otherwise, if it's a good enough trial...
                if (bad_count_contours < (no_of_samples/2)): 
                    this_chunk_length = len(this_trial_contours)
                    data_contours[index][0:this_chunk_length] = this_trial_contours
                    data_circles[index][0:this_chunk_length] = this_trial_circles
                index = index + 1
            else:
                #print("Discarding trial {name}".format(name=trial_name))
                index = index + 1
                good_trials = good_trials - 1
        return data_contours, data_circles, num_trials, good_trials
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

### BEGIN ANALYSIS ###
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
# List relevant data locations: these are for KAMPFF-LAB-VIDEO
#root_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
current_working_directory = os.getcwd()
stimuli_luminance_folder = r"C:\Users\taunsquared\Documents\GitHub\SurprisingMinds-Analysis\PythonWithAdam\bonsai\StimuliVid_Luminance"

# set up log file to store all printed messages
log_filename = "pupil-plotting_log_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
log_file = os.path.join(current_working_directory, log_filename)
sys.stdout = open(log_file, "w")

# set up folders
plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
pupils_folder = os.path.join(plots_folder, "pupil")
engagement_folder = os.path.join(plots_folder, "engagement")

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

# consolidate csv files from multiple days into one data structure
day_folders = list_sub_folders(root_folder)
# first day was a debugging session, so skip it
day_folders = day_folders[1:]
# currently still running pupil finding analysis...
day_folders = day_folders[:-1]

all_right_trials_contours = []
all_right_trials_circles = []
all_left_trials_contours = []
all_left_trials_circles = []
activation_count = []
analysed_count = []

# downsample = collect data from every 40ms or other multiples of 20
downsample_rate_ms = 40
original_bucket_size_in_ms = 4
no_of_time_buckets = 10000
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
        right_area_contours, right_area_circles, num_right_activations, num_good_right_trials = load_daily_pupil_areas("right", csv_folder, no_of_time_buckets, original_bucket_size_in_ms, downsample_rate_ms)
        left_area_contours, left_area_circles, num_left_activations, num_good_left_trials = load_daily_pupil_areas("left", csv_folder, no_of_time_buckets, original_bucket_size_in_ms, downsample_rate_ms)

        analysed_count.append((num_good_right_trials, num_good_left_trials))
        activation_count.append((num_right_activations, num_left_activations))
        print("On {day}, exhibit was activated {count} times, with {good_count} good trials".format(day=day_name, count=num_right_activations, good_count=num_good_right_trials))

        ## COMBINE EXTRACTING PUPIL SIZE AND POSITION

        # filter data for outlier points
        # contours/circles that are too big
        right_area_contours = threshold_to_nan(right_area_contours, 15000, 'upper')
        right_area_circles = threshold_to_nan(right_area_circles, 15000, 'upper')
        left_area_contours = threshold_to_nan(left_area_contours, 15000, 'upper')
        left_area_circles = threshold_to_nan(left_area_circles, 15000, 'upper')
        # time buckets with no corresponding frames
        right_area_contours = threshold_to_nan(right_area_contours, 0, 'lower')
        right_area_circles = threshold_to_nan(right_area_circles, 0, 'lower')
        left_area_contours = threshold_to_nan(left_area_contours, 0, 'lower')
        left_area_circles = threshold_to_nan(left_area_circles, 0, 'lower')

        # create a baseline - take first 3 seconds, aka 75 time buckets where each time bucket is 40ms
        right_area_contours_baseline = np.nanmedian(right_area_contours[:,0:baseline_no_buckets], 1)
        right_area_circles_baseline = np.nanmedian(right_area_circles[:,0:baseline_no_buckets], 1)
        left_area_contours_baseline = np.nanmedian(left_area_contours[:,0:baseline_no_buckets], 1)
        left_area_circles_baseline = np.nanmedian(left_area_circles[:,0:baseline_no_buckets], 1)

        # normalize and append
        for index in range(len(right_area_contours_baseline)): 
            right_area_contours[index,:] = right_area_contours[index,:]/right_area_contours_baseline[index]
            all_right_trials_contours.append(right_area_contours[index,:])
        for index in range(len(right_area_circles_baseline)): 
            right_area_circles[index,:] = right_area_circles[index,:]/right_area_circles_baseline[index]
            all_right_trials_circles.append(right_area_circles[index,:])
        for index in range(len(left_area_contours_baseline)): 
            left_area_contours[index,:] = left_area_contours[index,:]/left_area_contours_baseline[index]
            all_left_trials_contours.append(left_area_contours[index,:])
        for index in range(len(left_area_circles_baseline)): 
            left_area_circles[index,:] = left_area_circles[index,:]/left_area_circles_baseline[index]
            all_left_trials_circles.append(left_area_circles[index,:])
        print("Day {day} succeeded!".format(day=day_name))
    except Exception:
        print("Day {day} failed!".format(day=day_name))

# find average luminance of stimuli vids
# octo_clip_start for stimuli videos
# stimuli024 = 169
# stimuli025 = 173
# stimuli026 = 248
# stimuli027 = 180
# stimuli028 = 247
# stimuli029 = 314
luminance = []
luminance_data_paths = glob.glob(stimuli_luminance_folder + "/*_stimuli*_LuminancePerFrame.csv")
for data_path in luminance_data_paths: 
    luminance_values = np.genfromtxt(data_path, dtype=np.str, delimiter='  ')
    luminance_values = np.array(luminance_values)
    luminance.append(luminance_values)
luminance_array = np.array(luminance)
average_luminance = build_timebucket_avg_luminance(luminance_array, downsample_rate_ms, 630)
baseline = np.nanmean(average_luminance[0:baseline_no_buckets])
avg_lum_baselined = [(x/baseline) for x in average_luminance]

### NOTES FROM LAB MEETING ###
# based on standard dev of movement, if "eye" doesn't move enough, don't plot that trial
## during tracking, save luminance of "darkest circle"
# if there is too much variability in frame rate, then don't plot that trial
# if standard dev of diameter is "too big", then don't plot

### EXHIBIT ACTIVITY METADATA ### 
# Save activation count to csv
engagement_count_filename = 'Exhibit_Activation_Count_measured-' + todays_datetime + '.csv'
engagement_data_folder = os.path.join(current_working_directory, 'Exhibit-Engagement')
if not os.path.exists(engagement_data_folder):
    #print("Creating plots folder.")
    os.makedirs(engagement_data_folder)
csv_file = os.path.join(engagement_data_folder, engagement_count_filename)
np.savetxt(csv_file, activation_count, fmt='%.2f', delimiter=',')

# Plot activation count
total_activation = sum(count[0] for count in activation_count)
total_days_activated = len(activation_count)
total_good_trials = sum(count[0] for count in analysed_count)
print("Total number of exhibit activations: {total}".format(total=total_activation))
print("Total number of good trials: {good_total}".format(good_total=total_good_trials))
activation_array = np.array(activation_count)
analysed_array = np.array(analysed_count)
# do da plot
image_type_options = ['.png', '.pdf']
for image_type in image_type_options:
    figure_name = 'TotalExhibitActivation_' + todays_datetime + image_type
    figure_path = os.path.join(engagement_folder, figure_name)
    figure_title = "Total number of exhibit activations per day (Grand Total: " + str(total_activation) + ") \n (Number of good trials: " + str(total_good_trials) + ")\nPlotted on " + todays_datetime
    plt.figure(figsize=(18, 9), dpi=200)
    plt.suptitle(figure_title, fontsize=12, y=0.98)

    plt.ylabel('Number of activations', fontsize=11)
    plt.xlabel('Days, Total days activated: ' + str(total_days_activated), fontsize=11)
    #plt.minorticks_on()
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.plot(activation_array, color=[0.0, 0.0, 1.0])
    #plt.plot(analysed_array, color=[1.0, 0.0, 0.0])

    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

### BACK TO THE PUPILS ###
all_right_trials_contours_array = np.array(all_right_trials_contours)
all_right_trials_circles_array = np.array(all_right_trials_circles)
all_left_trials_contours_array = np.array(all_left_trials_contours)
all_left_trials_circles_array = np.array(all_left_trials_circles)
trials_to_plot = [(all_right_trials_contours_array, all_left_trials_contours_array), (all_right_trials_circles_array, all_left_trials_circles_array)]

# Compute global mean
all_right_contours_mean = np.nanmean(all_right_trials_contours_array, 0)
all_right_circles_mean = np.nanmean(all_right_trials_circles_array, 0)
all_left_contours_mean = np.nanmean(all_left_trials_contours_array, 0)
all_left_circles_mean = np.nanmean(all_left_trials_circles_array, 0)
means_to_plot = [(all_right_contours_mean, all_left_contours_mean), (all_right_circles_mean, all_left_circles_mean)]

# event locations in time
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

# Plot pupil sizes
plot_types = ["contours", "circles"]
for i in range(len(trials_to_plot)):
    plot_type_right = trials_to_plot[i][0]
    plot_type_left = trials_to_plot[i][1]
    plot_means_right = means_to_plot[i][0]
    plot_means_left = means_to_plot[i][1]
    plot_type_name = plot_types[i]
    for image_type in image_type_options:
        if (image_type == '.pdf'):
            continue
        else:
            dpi_sizes = [100, 400]
        for size in dpi_sizes: 
            figure_name = 'AveragePupilSizes_' + plot_type_name + '_' + todays_datetime + '_dpi' + str(size) + image_type 
            figure_path = os.path.join(pupils_folder, figure_name)
            figure_title = "Pupil sizes of participants, N=" + str(total_good_trials) + " good trials out of " + str(total_activation) + " activations" + "\nAnalysis type: " + plot_type_name + "\nPlotted on " + todays_datetime
            plt.figure(figsize=(14, 14), dpi=size)
            plt.suptitle(figure_title, fontsize=12, y=0.98)

            plt.subplot(3,1,1)
            ax = plt.gca()
            ax.yaxis.set_label_coords(-0.09, -0.5) 
            plt.ylabel('Percentage from baseline', fontsize=11)
            plt.title('Right eye pupil sizes', fontsize=9, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.plot(plot_type_right.T, '.', MarkerSize=1, color=[0.0, 0.0, 1.0, 0.01])
            plt.plot(plot_means_right, linewidth=1.5, color=[1.0, 0.0, 0.0, 0.4])
            plt.xlim(-10,500)
            plt.ylim(0,2.0)
            
            plt.subplot(3,1,2)
            plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsample_rate_ms) + 'ms)', fontsize=11)
            plt.title('Left eye pupil sizes', fontsize=9, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.plot(plot_type_left.T, '.', MarkerSize=1, color=[0.0, 1.0, 0.0, 0.01])
            plt.plot(plot_means_left, linewidth=1.5, color=[1.0, 0.0, 0.0, 0.4])
            plt.xlim(-10,500)
            plt.ylim(0,2.0)
            
            plt.subplot(3,1,3)
            plt.title('Average luminance of stimuli video, grayscaled', fontsize=9, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.plot(avg_lum_baselined, linewidth=4, color=[1.0, 0.0, 1.0, 1])
            plt.xlim(-10,500)
            plt.ylim(0,2.0)
            # mark events
            for i in range(len(event_labels)):
                plt.plot((event_locations[i],event_locations[i]), (0.25,2.2-((i-1)/5)), 'k-', linewidth=1)
                plt.text(event_locations[i]+1,2.2-((i-1)/5), event_labels[i], fontsize='x-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.35'))

            plt.subplots_adjust(hspace=0.5)
            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()


#FIN

### NOTES FROM MEETING WITH ADAM
# one plot with event labels, one without
# plot means of pupil cameras and inverse of world camera luminance onto one plot
    # optional: add error bars to the above plot
# plot the eye movements - copy this entire script and tweak for plotting pupil movement