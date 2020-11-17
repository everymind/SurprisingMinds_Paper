import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

# List relevant data locations
root_folder = "/home/kampff/DK"
day_folder = root_folder + "/SurpriseIntelligence_2017-07-25"
save_folder = root_folder + "/Clip"

# List all trial folders
trial_folders = []
for folder in os.listdir(day_folder):
    if(os.path.isdir(day_folder + '/' + folder)):
        trial_folders.append(day_folder + '/' + folder)
num_trials = len(trial_folders)

# Set averaging parameters
align_time = 18.3 # w.r.t. end of world movie
clip_length = 360
clip_offset = -120

# Allocate empty space for average frame and movie clip
average_grayscale_clip = np.zeros((600,800,clip_length))
#average_grayscale_clip = np.zeros((120,160,clip_length))

# Load all right eye movies and average
current_trial = 0
for trial_folder in trial_folders:

    # Get world movie timestamp csv path
    world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]

    # Load world CSV
    world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ')

    # Get eye (right) timestamp csv path
    right_eye_csv_path = glob.glob(trial_folder + '/*righteye.csv')[0]

    # Load eye (right) CSV
    right_eye_timestamps = np.genfromtxt(right_eye_csv_path, dtype=np.str, delimiter=' ')

    # -----

    # Get last frame time
    last_timestamp = world_timestamps[-1]
    last_timestamp = last_timestamp.split('+')[0][:-1]
    last_time = datetime.datetime.strptime(last_timestamp, "%Y-%m-%dT%H:%M:%S.%f")

    # Generate delta times (w.r.t. last frame) for every frame timestamp
    frame_counter = 0
    for timestamp in world_timestamps:
        timestamp = timestamp.split('+')[0][:-1]
        time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        timedelta = last_time - time
        seconds_until_end = timedelta.total_seconds()
        if((seconds_until_end - align_time) < 0):
            break
        frame_counter = frame_counter + 1

    # Set temporary align frame to the frame counter closest to align_time
    temp_align_frame = frame_counter

    # Get world video filepath
    world_video_path = glob.glob(trial_folder + '/*world.avi')[0]
    
    # Open world video
    video = cv2.VideoCapture(world_video_path)

    # Jump to temprary alignment frame (position)
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, temp_align_frame)

    # Loop through frames and look for octopus
    for f in range(200):
        
        # Read frame at current position
        ret, frame = video.read()

        # Convert to grayscale
        gray = np.mean(frame,2)

        # Measure ROI intensity
        roi = gray[51:58, 63:70]
        intensity = np.mean(np.mean(roi))

        # Is there an octopus?
        if(intensity > 100):
            break
    
    # Set world align frame
    world_align_frame = temp_align_frame + f
    world_align_frame = world_align_frame

    # Set world align time
    world_align_timestamp = world_timestamps[world_align_frame]
    world_align_timestamp = world_align_timestamp.split('+')[0][:-1]
    world_align_time = datetime.datetime.strptime(world_align_timestamp, "%Y-%m-%dT%H:%M:%S.%f")

    # Find (right) eye align frame

    # Generate delta times (w.r.t. world_align_time) for every (right) eye frame timestamp
    frame_counter = 0
    for timestamp in right_eye_timestamps:
        timestamp = timestamp.split('+')[0][:-1]
        time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        timedelta = world_align_time - time
        seconds_until_alignment = timedelta.total_seconds()
        if(seconds_until_alignment < 0):
            break
        frame_counter = frame_counter + 1

    # Find (right) eye align frame
    align_frame = frame_counter + clip_offset

    # Get (right) eye video filepath
    right_video_path = glob.glob(trial_folder + '/*righteye.avi')[0]
    
    # Open video
    video = cv2.VideoCapture(right_video_path)

    # Jump to specific frame (position) for alignment purposes (currently arbitrary)
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, align_frame)

    # Loop through frames for clip extraction
    for f in range(clip_length):
        
        # Read frame at current position
        ret, frame = video.read()

        # Convert to grayscale
        gray = np.mean(frame,2)

        # Add current frame to average clip at correct slot
        average_grayscale_clip[:,:,f] = average_grayscale_clip[:,:,f] + gray

    # Report progress
    print(current_trial)
    current_trial = current_trial + 1

# Compute average clip
average_grayscale_clip = (average_grayscale_clip/num_trials)/255.0

# Save images from clip to folder
for f in range(clip_length):

    # Create file name with padded zeros
    padded_filename = str(f).zfill(4)

    # Create image file path from save folder
    image_file_path = save_folder + "/" + padded_filename + ".png" 

    # Extract gray frame from clip
    gray = np.uint8(average_grayscale_clip[:,:,f] * 255)

    # Write to image file
    ret = cv2.imwrite(image_file_path, gray)

#FIN
