import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# List relevant data locations
root_folder = "/home/kampff/DK"
day_folder = root_folder + "/SurpriseIntelligence_2017-07-25"

# List all trial folders
trial_folders = []
for folder in os.listdir(day_folder):
    if(os.path.isdir(day_folder + '/' + folder)):
        trial_folders.append(day_folder + '/' + folder)
num_trials = len(trial_folders)

# Allocate empty space for average frame
average_frame = np.zeros((600,800,3))

# Load all left eye movies and average
for trial_folder in trial_folders:

    # Get video filepath
    left_video_path = glob.glob(trial_folder + '/*lefteye.avi')[0]

    # Open video
    video = cv2.VideoCapture(left_video_path)

    # Jump to specific frame (position) for alignment purposes (currently arbitrary)
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, 100)

    # Read frame at current position
    ret, frame = video.read()

    # Add current frame to average frame
    average_frame = average_frame + frame

# Compute average (and normalize to 0.0 to 1.0)
average_frame = (average_frame/num_trials)/255.0

# Plot that shit
plt.imshow(average_frame)
plt.show()

#FIN
