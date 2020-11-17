# -*- coding: utf-8 -*-
"""
Project: "Surprising Minds" at Sea Life Brighton, by Danbee Kim, Kerry Perkins, Clive Ramble, Hazel Garnade, Goncalo Lopes, Dario Quinones, Reanna Campbell-Russo, Robb Barrett, Martin Stopps, The EveryMind Team, and Adam Kampff. 
Analysis: Measure speed of pupil

Collects csv files of pupil tracking data from all subjects/days of experiment/exhibit.
Calculate the speed of pupil movement from one frame to the next.
Outputs a .data file for each subject containing the speeds of their pupil throughout the experiment.

Resolution = 4ms per "timebucket", as that was the sampling rate used to generate the csv files of pupil tracking data. 

@author: Adam R Kampff and Danbee Kim
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os.path

###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# FUNCTIONS
###################################

##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) dataset_dir (folder with full, primary experimental dataset)
# 2) plots_dir (parent folder for all plots output by this script)
# 3) intermediates_dir (folder for output of this script)
##########################################################
def load_data():
    dataset_dir = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows'
    plots_dir = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots\saccade_detector'
    intermediates_dir = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\intermediates'
    return dataset_dir, plots_dir, intermediates_dir
##########################################################

##########################################################
# BEGIN SCRIPT
##########################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='''Measure speed of pupil.
        Collects csv files of pupil tracking data from all subjects/days of experiment/exhibit.
        Calculate the speed of pupil movement from one frame to the next.
        Outputs a .data file for each subject containing the speeds of their pupil throughout the experiment.
        WARNING: This script overwrites speed data outputted from previous runs of this script. TO SAVE OLD SPEED DATA, RENAME THE FOLDER CONTAINING OLD SPEED DATA.
        Resolution = 4ms per "timebucket", as that was the sampling rate used to generate the csv files of pupil tracking data. ''')
    parser.add_argument("--a", nargs='?', default="check_string_for_empty")
    args = parser.parse_args()
    ###################################
    # SCRIPT LOGGER
    ###################################
    # grab today's date
    now = datetime.datetime.now()
    todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
    logging.basicConfig(filename="pm01MeasureSpeeds_" + todays_datetime + ".log", filemode='w', level=logging.INFO)
    ###################################
    # SOURCE DATA AND OUTPUT FILE LOCATIONS 
    ###################################
    raw_dataset_folder, plots_folder, output_folder = load_data()
    logging.info('DATA FOLDER: %s \n PLOTS FOLDER: %s \n OUTPUT FOLDER: %s' % (raw_dataset_folder, plots_folder, output_folder))
    print('DATA FOLDER: %s \n PLOTS FOLDER: %s \n OUTPUT FOLDER: %s' % (raw_dataset_folder, plots_folder, output_folder))
    ###################################
    # FIND DAILY PUPIL TRACKING DATA
    ###################################
    daily_folders = glob.glob(data_folder + os.sep + 'SurprisingMinds*')
    # If you only want to find saccades in a subset of the data...
    #daily_folders = daily_folders[10:100]
    ###################################
    # COUNT NUMBER OF FILES
    ###################################
    num_days = len(daily_folders)
    num_files = 0
    for df_C, daily_folder_count in enumerate(daily_folders):
        # Find csv paths (platform independent)
        csv_paths_count = glob.glob(daily_folder_count + os.sep + 'analysis' + os.sep + 'csv'+ os.sep + '*.csv')
        if(len(csv_paths_count) == 0):
            csv_paths_count = glob.glob(daily_folder_count + os.sep + 'Analysis' + os.sep + 'csv'+ os.sep + '*.csv')
        num_files = len(csv_paths_count) + num_files
    logging.info('Number of files: {n}'.format(n=num_files))
    print('Number of files: {n}'.format(n=num_files))
    ###################################
    # CREATE FOLDER FOR SPEED DATA [CAUTION, DELETES ALL PREVIOUS SPEED FILES]
    ###################################
    speed_data_folder = output_folder + os.sep + 'speeds'
    if not os.path.exists(speed_data_folder):
        logging.info("Creating speed data folder.")
        print("Creating speed data folder.")
        os.makedirs(speed_data_folder)
    if os.path.exists(speed_data_folder):
        # make sure it's empty
        logging.info("Deleting old speed data...")
        print("Deleting old speed data...")
        filelist = glob.glob(os.path.join(speed_data_folder, "*.data"))
        for f in filelist:
            os.remove(f)
    ###################################
    # EXTRACT PUPIL TRACKING DATA AND GENERATE "SPEED" PER FRAME FOR EACH EYE VIDEO
    ###################################
    trial_count = 0
    stim_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for df, daily_folder in enumerate(daily_folders):

        # Find csv paths
        csv_paths = glob.glob(daily_folder + os.sep + 'analysis' + os.sep + 'csv'+ os.sep + '*.csv')
        if(len(csv_paths) == 0):
            csv_paths = glob.glob(daily_folder + os.sep + 'Analysis' + os.sep + 'csv'+ os.sep + '*.csv')
        num_files = len(csv_paths)

        # Process all csv files in a folder
        for cp, csv_path in enumerate(csv_paths):

            # Extract eye name and stimulus number
            trial_name = os.path.basename(csv_path)
            fields = trial_name.split(sep='_')
            eye = fields[0]
            stimulus = int(fields[1][-1:])-4
            stim_count[stimulus] = stim_count[stimulus] + 1

            # Load data
            data = np.genfromtxt(csv_path, delimiter=',')
            raw_x = data[:,0]
            raw_y = data[:,1]
            raw_area = data[:,2]
            x = np.copy(raw_x)
            y = np.copy(raw_y)
            area = np.copy(raw_area)
            num_samples = len(x)

            # Extract valid X and Y values
            good_indices = np.where(area > 0)[0]

            # Exclude crappy trials
            if(len(good_indices) < 200):
                break
            good_x = x[good_indices]
            good_y = y[good_indices]
            good_area = area[good_indices]
            num_valid = len(good_indices)

            # Start with first valid values
            if x[0] < 0:
                x[0] = good_x[0]
                y[0] = good_y[0]
                area[0] = good_area[0]

            # Interpolate X and Y values across tracking errors/empty frames
            count = 1
            for i in range(1, num_valid):
                next_valid_index = good_indices[i]
                next_valid_x = good_x[i]
                next_valid_y = good_y[i]
                next_valid_area = good_area[i]
                step_count = (next_valid_index - count + 1)
                step_x = (next_valid_x - x[count - 1]) / step_count
                step_y = (next_valid_y - y[count - 1]) / step_count
                step_area = (next_valid_area - area[count - 1]) / step_count
                for j in range(step_count):
                    x[count] = x[count - 1] + step_x
                    y[count] = y[count - 1] + step_y
                    area[count] = area[count - 1] + step_area
                    count += 1
            # Now we have X, Y, and Area for every time bucket (linearly interpolated)

            # Smooth (8 time-buckets: ~ 32 ms, 30 Hz)
            smooth_kernel = np.ones(8) / 8
            x = np.convolve(x, smooth_kernel, mode='same')
            y = np.convolve(y, smooth_kernel, mode='same')
            area = np.convolve(area, smooth_kernel, mode='same')

            # Measure "speed" (change in x and y)
            dx = np.diff(x, prepend=[0])
            dy = np.diff(y, prepend=[0])
            speed = np.sqrt(dx*dx + dy*dy)
            speed = np.float32(speed)

            # Store
            output_path = speed_data_folder + os.sep + 'stim%d_%s_peak_%d.data' % (stimulus, eye, trial_count)
            speed.tofile(output_path)
            trial_count = trial_count + 1

            # Plot
            plot = False
            if plot:
                plt.figure()
                plt.subplot(2,2,1)
                plt.plot(raw_x)
                plt.plot(x)
                plt.subplot(2,2,2)
                plt.plot(raw_y)
                plt.plot(y)
                plt.subplot(2,2,3)
                plt.plot(raw_area)
                plt.plot(area)
                plt.subplot(2,2,4)
                plt.plot(speed)
                plt.show()

            # Report progress
            print('Trial count: {t}'.format(t=trial_count))
    logging.info('Total trial count: {t}'.format(t=trial_count))


# FIN