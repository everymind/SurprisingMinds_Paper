import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
import fnmatch
import sys
import math

### FUNCTIONS ###

def time_between_frames(timestamps_csv):
    time_diffs = []
    for t in range(len(timestamps_csv)):
        this_timestamp = timestamps_csv[t].split('+')[0][:-1]
        this_time = datetime.datetime.strptime(this_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        if (t==0):
            time_diffs.append(0)
            last_time = this_time
        else: 
            time_diff = this_time - last_time
            time_diff_milliseconds = time_diff.total_seconds() * 1000
            time_diffs.append(time_diff_milliseconds)
            last_time = this_time
    return np.array(time_diffs)

def find_target_frame(ref_timestamps_csv, target_timestamps_csv, ref_frame):
    # Find the frame in one video that best matches the timestamp of ref frame from another video
    # Get ref frame time
    ref_timestamp = ref_timestamps_csv[ref_frame]
    ref_timestamp = ref_timestamp.split('+')[0][:-1]
    ref_time = datetime.datetime.strptime(ref_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
    # Generate delta times (w.r.t. start_frame) for every frame timestamp
    frame_counter = 0
    for timestamp in target_timestamps_csv:
        timestamp = timestamp.split('+')[0][:-1]
        time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        timedelta = ref_time - time
        milliseconds_until_alignment = timedelta.total_seconds() * 1000
        if(milliseconds_until_alignment < 0):
            break
        frame_counter = frame_counter + 1
    return frame_counter

def list_sub_folders(path_to_root_folder):
    # List all sub folders
    sub_folders = []
    for folder in os.listdir(path_to_root_folder):
        if(os.path.isdir(os.path.join(path_to_root_folder, folder))):
            sub_folders.append(os.path.join(path_to_root_folder, folder))
    return sub_folders

### BEGIN ANALYSIS ###
#data_drive = r"C:\Users\taunsquared\Documents\GitHub\SurprisingMinds-Analysis\PythonWithAdam\temp"
data_drive = r"D:\Users\KAMPFF-LAB-VIDEO\SurprisingMinds-VideoBuffer\SurprisingMinds_2018-05-05"
current_working_directory = os.getcwd()
plots_folder = os.path.join(current_working_directory, "plots")
camera_profiles_folder = os.path.join(plots_folder, "camera_profiles")

# Create plots folder (and sub-folders) if it (they) does (do) not exist
if not os.path.exists(plots_folder):
    #print("Creating plots folder.")
    os.makedirs(plots_folder)
if not os.path.exists(camera_profiles_folder):
    #print("Creating camera profiles folder.")
    os.makedirs(camera_profiles_folder)

# List all trial folders
trial_folders = list_sub_folders(data_drive)
num_trials = len(trial_folders)

# create dictionary of start frames for octopus clip
octo_frames = {"stimuli024": 438, "stimuli025": 442, "stimuli026": 517, "stimuli027": 449, "stimuli028": 516, "stimuli029": 583}
    
# create dictionary to hold time differences between frames, categorized by stimuli
all_right_diffs = {"stimuli024": [], "stimuli025": [], "stimuli026": [], "stimuli027": [], "stimuli028": [], "stimuli029": []}
all_left_diffs = {"stimuli024": [], "stimuli025": [], "stimuli026": [], "stimuli027": [], "stimuli028": [], "stimuli029": []}
all_world_diffs = {"stimuli024": [], "stimuli025": [], "stimuli026": [], "stimuli027": [], "stimuli028": [], "stimuli029": []}

for trial_folder in trial_folders:
    trial_name = trial_folder.split(os.sep)[-1]
    # Load CSVs and create timestamps
    # ------------------------------
    #print("Loading csv files for {trial}...".format(trial=trial_name))
    # Get world movie timestamp csv path
    world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]
    stimuli_number = world_csv_path.split("_")[-2]

    world_octo_start = octo_frames[stimuli_number]

    # Load world CSV
    this_trial_world = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ')

    # Get eye timestamp csv paths
    right_eye_csv_path = glob.glob(trial_folder + '/*righteye.csv')[0]
    left_eye_csv_path = glob.glob(trial_folder + '/*lefteye.csv')[0]

    # Load eye CSVs
    this_trial_right = np.genfromtxt(right_eye_csv_path, dtype=np.str, delimiter=' ')
    this_trial_left = np.genfromtxt(left_eye_csv_path, dtype=np.str, delimiter=' ')

    # trim csvs to just octopus video
    right_octo = find_target_frame(this_trial_world, this_trial_right, world_octo_start)
    left_octo = find_target_frame(this_trial_world, this_trial_left, world_octo_start)

    world_octo_timestamps = this_trial_world[world_octo_start:]
    right_octo_timestamps = this_trial_right[right_octo:]
    left_octo_timestamps = this_trial_left[left_octo:]

    # Generate delta times (w.r.t. start_frame) for every frame timestamp
    right_time_diffs_array = time_between_frames(this_trial_right)
    left_time_diffs_array = time_between_frames(this_trial_left)
    world_time_diffs_array = time_between_frames(this_trial_world)

    # plot
    figure_name = stimuli_number + '_' + trial_name + '_ms-bt-frames.pdf'
    figure_path = os.path.join(camera_profiles_folder, figure_name)
    figure_title = 'Time elapsed between frames \n participant: ' + trial_name + ', video: ' + stimuli_number
    plt.figure(figsize=(7, 6.4), dpi=300)
    plt.suptitle(figure_title, fontsize=12, y=0.98)

    plt.subplot(3,1,1)
    plt.title('Right eye camera', fontsize=9, color='grey', style='italic')
    plt.minorticks_on()
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.plot(right_time_diffs_array.T,'.', MarkerSize=1, color=[1.0, 0.0, 0.0, 0.7])

    plt.subplot(3,1,2)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.09, 0.5) 
    plt.ylabel('Milliseconds between frames', fontsize=11)
    plt.title('Left eye camera', fontsize=9, color='grey', style='italic')
    plt.minorticks_on()
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.plot(left_time_diffs_array.T,'.', MarkerSize=1, color=[0.0, 1.0, 0.0, 0.7])

    plt.subplot(3,1,3)
    plt.xlabel('Frame number', fontsize=11)
    plt.title('World camera (records monitor presenting video stimuli to participants)', fontsize=9, color='grey', style='italic')
    plt.minorticks_on()
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.plot(world_time_diffs_array.T, '.', MarkerSize = 1, color=[0.0, 0.0, 1.0, 0.7])

    plt.subplots_adjust(hspace=0.7)
    #plt.tight_layout()

    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # add to dictionary of time diffs, according to stimuli number
    all_right_diffs[stimuli_number].append(right_time_diffs_array)
    all_left_diffs[stimuli_number].append(left_time_diffs_array)
    all_world_diffs[stimuli_number].append(world_time_diffs_array)

# FIN
