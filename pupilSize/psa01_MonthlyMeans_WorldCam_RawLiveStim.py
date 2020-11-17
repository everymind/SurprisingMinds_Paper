### ------------------------------------------------------------------------- ###
### Create binary files of raw stim vid luminance values fitted to world cam stim vid presentation timings
### use world camera vids for timing, use raw vid luminance values extracted via bonsai
### also save world cam luminance as sanity check/ground truth 
### create monthly averages of both raw live stim vid and world cam sanity check
### output as data files 
### NOTE: NEED TO MODIFY FIRST FUNCTION BASED ON LOCATION OF SOURCE DATASET AND INTERMEDIATE PUPIL TRACKING DATA
### WHEN RUNNING FROM TERMINAL: add optional "restart" to delete previous runs of this script and start over
### NOTE: make sure this script is in a directory with a "__init__.py" file, so that this script can be treated as a module
### ------------------------------------------------------------------------- ###
import logging
import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
import shutil
import fnmatch
import sys
import math
import csv
import argparse
import time
###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
logging.basicConfig(filename="psa01_MonthlyMeans_WorldCam_RawLiveStim_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log", filemode='w', level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
###################################
# FUNCTIONS
###################################

##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) MAIN SURPRISING MINDS SOURCE DATASET (LEFT/RIGHT EYE CAMERA VIDEOS, WORLD CAMERA VIDEO, AND ACCOMPANYING TIMESTAMPS FOR EACH PARTICIPANT)
# AND
# 2) INTERMEDIATE PUPIL SIZE AND LOCATION FILES (WITH ACCOMPANYING WORLD CAM ALIGNMENT IMAGES)
### Current default uses a debugging source dataset
##########################################################
def load_data(location='laptop'):
    if location == 'laptop':
        data_drive = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\debuggingData"
        analysed_drive = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
    elif location == 'office_real':
        data_drive = r"\\Diskstation\SurprisingMinds"
        analysed_drive = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
    elif location == 'office_debug':
        data_drive = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\debuggingData"
        analysed_drive = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
    # collect input data subfolders
    rawStimLum_data = os.path.join(analysed_drive, "rawStimLums")
    analysed_folders = sorted(os.listdir(analysed_drive))
    daily_csv_files = fnmatch.filter(analysed_folders, 'SurprisingMinds_*')
    monthly_extracted_data = fnmatch.filter(analysed_folders, 'MeanStimuli_*')
    return data_drive, analysed_drive, rawStimLum_data, analysed_folders, daily_csv_files, monthly_extracted_data

##########################################################
def unpack_to_temp(path_to_zipped, path_to_temp):
    try:
        # copy zip file to current working directory
        #print("Copying {folder} to current working directory...".format(folder=path_to_zipped))
        current_working_directory = os.getcwd()
        copied_zipped = shutil.copy2(path_to_zipped, current_working_directory)
        path_to_copied_zipped = os.path.join(current_working_directory, copied_zipped.split(sep=os.sep)[-1])
        # unzip the folder
        #print("Unzipping files in {folder}...".format(folder=path_to_copied_zipped))
        day_unzipped = zipfile.ZipFile(path_to_copied_zipped, mode="r")
        # extract files into temp folder
        day_unzipped.extractall(path_to_temp)
        # close the unzipped file
        day_unzipped.close()
        #print("Finished unzipping {folder}!".format(folder=path_to_copied_zipped))
        # destroy copied zipped file
        #print("Deleting {file}...".format(file=path_to_copied_zipped))
        os.remove(path_to_copied_zipped)
        #print("Deleted {file}!".format(file=path_to_copied_zipped))
        return True
    except Exception: 
        logging.warning("Could not unzip {folder}".format(folder=path_to_zipped))    
        return False

def list_sub_folders(path_to_root_folder):
    # List all sub folders
    sub_folders = []
    for folder in os.listdir(path_to_root_folder):
        if(os.path.isdir(os.path.join(path_to_root_folder, folder))):
            sub_folders.append(os.path.join(path_to_root_folder, folder))
    return sub_folders

def make_time_buckets(start_timestamp, bucket_size_ms, end_timestamp, fill_pattern): 
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
        bucket_list[key] = fill_pattern
    return bucket_list

def find_nearest_timestamp_key(timestamp_to_check, dict_of_timestamps, time_window):
    for key in dict_of_timestamps.keys():
        if key <= timestamp_to_check <= (key + time_window):
            return key

def supersampled_worldCam_rawLiveVid(video_path, video_timestamps, rawStimVidData_dict, output_folder, bucket_size_ms):
    # Get video file details
    video_name = video_path.split(os.sep)[-1]
    video_date = video_name.split('_')[0]
    video_time = video_name.split('_')[1]
    video_stim_number = video_name.split('_')[2]
    # Open world video
    world_vid = cv2.VideoCapture(video_path)
    vid_width = int(world_vid.get(3))
    vid_height = int(world_vid.get(4))
    # create rawLiveVid output array
    first_timestamp = video_timestamps[0]
    last_timestamp = video_timestamps[-1]
    rawLiveVid_initializePattern = np.nan
    rawLiveVid_buckets = make_time_buckets(first_timestamp, bucket_size_ms, last_timestamp, rawLiveVid_initializePattern)
    sanityCheck_initializePattern = np.empty((vid_height*vid_width,))
    sanityCheck_initializePattern[:] = np.nan
    worldCam_sanityCheck_buckets = make_time_buckets(first_timestamp, bucket_size_ms, last_timestamp, sanityCheck_initializePattern)
    # Loop through 4ms time buckets of world video to find nearest frame and save 2-d matrix of pixel values in that frame
    # stimStructure = ['DoNotMove-English', 'Calibration', 'stimuli024', 'stimuli025', 'stimuli026', 'stimuli027', 'stimuli028', 'stimuli029', ]
    doNotMove_frameCount = rawStimVidData_dict['DoNotMove-English']['Number of Frames']
    calib_frameCount = rawStimVidData_dict['Calibration']['Number of Frames']
    # keep track of how many frames have been processed
    frame_count = 0
    for timestamp in video_timestamps: 
        # find the time bucket into which this frame falls
        timestamp = timestamp.split('+')[0][:-3]
        timestamp_dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        bucket_window = datetime.timedelta(milliseconds=bucket_size_ms)
        # fill in luminance values from world cam video as a sanity check
        currentKey_sanityCheck = find_nearest_timestamp_key(timestamp_dt, worldCam_sanityCheck_buckets, bucket_window)
        # Read frame at current position
        # should this be at current key??
        ret, frame = world_vid.read()
        # Make sure the frame exists!
        if frame is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # flatten the frame into a list
            flattened_gray = gray.ravel()
            flattened_gray = flattened_gray.astype(None)
            # append to dictionary stim_buckets
            worldCam_sanityCheck_buckets[currentKey_sanityCheck] = flattened_gray
        # fill in luminance values from raw videos based on timing of framerate in world camera timestamps
        currentKey_rLV = find_nearest_timestamp_key(timestamp_dt, rawLiveVid_buckets, bucket_window)
        if frame_count < doNotMove_frameCount:
            rawVidPhase = 'DoNotMove-English'
            frame_index = frame_count
        if doNotMove_frameCount <= frame_count < doNotMove_frameCount + calib_frameCount:
            rawVidPhase = 'Calibration'
            frame_index = frame_count - doNotMove_frameCount
        if doNotMove_frameCount + calib_frameCount <= frame_count:
            rawVidPhase = video_stim_number
            if frame_count < doNotMove_frameCount + calib_frameCount + rawStimVidData_dict[rawVidPhase]['Number of Frames']:
                frame_index = frame_count - doNotMove_frameCount - calib_frameCount
            else:
                break
        rawLiveVid_buckets[currentKey_rLV] = rawStimVidData_dict[rawVidPhase]['Luminance per Frame'][frame_index]
        #print('Processing frame %d from %s phase (total frame count: %d)' % (frame_index, rawVidPhase, frame_count))
        frame_count = frame_count + 1
    # release video capture
    world_vid.release()
    # generate rawLiveVid luminance array output
    supersampled_rawLiveVid = []
    current_lumVal = 0
    for timestamp in sorted(rawLiveVid_buckets.keys()):
        if rawLiveVid_buckets[timestamp] is not np.nan:
            supersampled_rawLiveVid.append(rawLiveVid_buckets[timestamp])
            current_lumVal = rawLiveVid_buckets[timestamp]
        else:
            supersampled_rawLiveVid.append(current_lumVal)
    supersampled_rawLiveVid_array = np.array(supersampled_rawLiveVid)
    # generate worldCam sanityCheck luminance array output
    supersampled_worldCam = []
    current_frame = sanityCheck_initializePattern
    for timestamp in sorted(worldCam_sanityCheck_buckets.keys()):
        if worldCam_sanityCheck_buckets[timestamp] is not np.nan:
            supersampled_worldCam.append(worldCam_sanityCheck_buckets[timestamp])
            current_frame = worldCam_sanityCheck_buckets[timestamp]
        else:
            supersampled_worldCam.append(current_frame)
    supersampled_worldCam_array = np.array(supersampled_worldCam)
    # return worldCam sanity check
    return vid_width, vid_height, supersampled_worldCam_array, supersampled_rawLiveVid_array

def add_to_daily_worldCam_dict(this_trial_world_vid_frames, this_trial_stim_num, daily_world_vid_dict):
    # keep track of how many videos are going into the average for this stim
    daily_world_vid_dict[this_trial_stim_num]['Vid Count'] = daily_world_vid_dict[this_trial_stim_num].get('Vid Count', 0) + 1
    this_trial_stim_vid = {}
    for tb, row in enumerate(this_trial_world_vid_frames):
        tbucket_num = tb
        flattened_frame = row
        this_trial_stim_vid[tbucket_num] = flattened_frame
    for tbucket in this_trial_stim_vid.keys():
        if tbucket in daily_world_vid_dict[this_trial_stim_num].keys():
            daily_world_vid_dict[this_trial_stim_num][tbucket]['Trial Count'] = daily_world_vid_dict[this_trial_stim_num][tbucket]['Trial Count'] + 1
            daily_world_vid_dict[this_trial_stim_num][tbucket]['Summed Frame'] = daily_world_vid_dict[this_trial_stim_num][tbucket]['Summed Frame'] + this_trial_stim_vid[tbucket]
        else: 
            daily_world_vid_dict[this_trial_stim_num][tbucket] = {'Trial Count': 1, 'Summed Frame': this_trial_stim_vid[tbucket]}

def add_to_daily_rawLiveVid_dict(this_trial_rawLive_vid_frames, this_trial_stim_num, daily_rawLive_vid_dict):
    # keep track of how many videos are going into the average for this stim
    daily_rawLive_vid_dict[this_trial_stim_num]['Vid Count'] = daily_rawLive_vid_dict[this_trial_stim_num].get('Vid Count', 0) + 1
    this_trial_stim_vid = {}
    for tb, row in enumerate(this_trial_rawLive_vid_frames):
        tbucket_num = tb
        flattened_frame = row
        this_trial_stim_vid[tbucket_num] = flattened_frame
    for tbucket in this_trial_stim_vid.keys():
        if tbucket in daily_rawLive_vid_dict[this_trial_stim_num].keys():
            daily_rawLive_vid_dict[this_trial_stim_num][tbucket]['Trial Count'] = daily_rawLive_vid_dict[this_trial_stim_num][tbucket]['Trial Count'] + 1
            daily_rawLive_vid_dict[this_trial_stim_num][tbucket]['Summed Luminance'] = daily_rawLive_vid_dict[this_trial_stim_num][tbucket]['Summed Luminance'] + this_trial_stim_vid[tbucket]
        else: 
            daily_rawLive_vid_dict[this_trial_stim_num][tbucket] = {'Trial Count': 1, 'Summed Luminance': this_trial_stim_vid[tbucket]}

def calculate_meanPerDay_worldCam(day_worldCam_tbDict):
    # intialize time bucket dict for mean worldcam for each day
    meanPerDay_worldCam = {key:{'Vid Count':0} for key in stim_vids}
    for stim_num in day_worldCam_tbDict.keys():
        for key in day_worldCam_tbDict[stim_num].keys():
            if key == 'Vid Count':
                meanPerDay_worldCam[stim_num]['Vid Count'] = meanPerDay_worldCam[stim_num]['Vid Count'] + day_worldCam_tbDict[stim_num]['Vid Count']
            else: 
                meanFrame = day_worldCam_tbDict[stim_num][key]['Summed Frame'] / day_worldCam_tbDict[stim_num][key]['Trial Count']
                meanPerDay_worldCam[stim_num][key] = {'Mean Frame': meanFrame, 'Trial Count': day_worldCam_tbDict[stim_num][key]['Trial Count']}
    return meanPerDay_worldCam

def calculate_meanPerDay_rawLiveVid(day_rawLiveVid_tbDict):
    # intialize time bucket dict for mean worldcam for each day
    meanPerDay_rawLiveVid = {key:{'Vid Count':0} for key in stim_vids}
    for stim_num in day_rawLiveVid_tbDict.keys():
        for key in day_rawLiveVid_tbDict[stim_num].keys():
            if key == 'Vid Count':
                meanPerDay_rawLiveVid[stim_num]['Vid Count'] = meanPerDay_rawLiveVid[stim_num]['Vid Count'] + day_rawLiveVid_tbDict[stim_num]['Vid Count']
            else: 
                meanLuminance = day_rawLiveVid_tbDict[stim_num][key]['Summed Luminance'] / day_rawLiveVid_tbDict[stim_num][key]['Trial Count']
                meanPerDay_rawLiveVid[stim_num][key] = {'Mean Luminance': meanLuminance, 'Trial Count': day_rawLiveVid_tbDict[stim_num][key]['Trial Count']}
    return meanPerDay_rawLiveVid

def save_daily_worldCam_meanFrames(this_day_meanWorld_dict, this_day_month, save_folder):
    for stim in this_day_meanWorld_dict.keys():
        thisStimMeanWorldCam = []
        for timebucket in this_day_meanWorld_dict[stim].keys():
            if timebucket == 'Vid Count':
                vidCount = this_day_meanWorld_dict[stim][timebucket]
            else:
                thisMeanFrame = this_day_meanWorld_dict[stim][timebucket]['Mean Frame']
                thisMeanFrame_weight = this_day_meanWorld_dict[stim][timebucket]['Trial Count']
                if np.nansum(thisMeanFrame) == 0:
                    continue
                else:
                    thisStimMeanWorldCam.append([timebucket, thisMeanFrame_weight, thisMeanFrame])
        thisStimMeanWorldCam_output = save_folder + os.sep + '%s_Stim%d_meanWorldCam_%dVids.npy' % (this_day_month, int(stim), vidCount)
        np.save(thisStimMeanWorldCam_output, thisStimMeanWorldCam)

def save_daily_rawLiveStim_meanLums(this_day_rawLiveStim_dict, this_day_month, save_folder):
    for stim in this_day_rawLiveStim_dict.keys():
        thisStimRawLive = []
        for timebucket in this_day_rawLiveStim_dict[stim].keys():
            if timebucket == 'Vid Count':
                vidCount = this_day_rawLiveStim_dict[stim][timebucket]
            else:
                thisMeanLum = this_day_rawLiveStim_dict[stim][timebucket]['Mean Luminance']
                thisMeanLum_weight = this_day_rawLiveStim_dict[stim][timebucket]['Trial Count']
                thisStimRawLive.append([timebucket, thisMeanLum_weight, thisMeanLum])
        thisStimRawLive_output = save_folder + os.sep + '%s_Stim%d_meanRawLiveStim_%dVids.npy' % (this_day_month, int(stim), vidCount)
        np.save(thisStimRawLive_output, thisStimRawLive)

def extract_daily_means_and_add_to_worldCam_or_rawLiveStim(dailyMean_binaryFiles, this_month_all_worldCam, this_month_all_rawLiveStim):
    for daily_mean_file in dailyMean_binaryFiles:
        daily_date = os.path.basename(daily_mean_file).split('_')[0]
        daily_mean_stim_num = stim_name_to_float[os.path.basename(daily_mean_file).split('_')[1]]
        daily_mean_type = os.path.basename(daily_mean_file).split('_')[2]
        daily_mean_vid_count = int(os.path.basename(daily_mean_file).split('_')[3][:-8])
        if daily_mean_type == 'meanWorldCam':
            this_file_dictionary = this_month_all_worldCam
            timebucket_mean_name = 'Mean Frame'
        if daily_mean_type == 'meanRawLiveStim':
            this_file_dictionary = this_month_all_rawLiveStim
            timebucket_mean_name = 'Mean Luminance'
        this_file_dictionary[daily_mean_stim_num]['Vid Count'] = this_file_dictionary[daily_mean_stim_num]['Vid Count'] + daily_mean_vid_count
        daily_mean = np.load(daily_mean_file, allow_pickle=True)
        # format of daily_mean: [timebucket, thisTimebucketMean_trialCount, thisTimebucketMean]
        for row in daily_mean:
            timebucket = row[0]
            thisTimebucketMean_trialCount = row[1]
            thisTimebucketMean = row[2]
            if timebucket in this_file_dictionary[daily_mean_stim_num].keys():
                this_file_dictionary[daily_mean_stim_num][timebucket][daily_date] = {'Trial Count': thisTimebucketMean_trialCount, timebucket_mean_name: thisTimebucketMean}
            else:
                this_file_dictionary[daily_mean_stim_num][timebucket] = {daily_date: {'Trial Count': thisTimebucketMean_trialCount, timebucket_mean_name: thisTimebucketMean}}

def save_monthly_weighted_meanStim(this_month_allStim_dict, stim_type):
    monthly_mean_stim = {key:{'Vid Count':0} for key in stim_vids}
    if stim_type == 'meanWorldCam':
        timebucket_mean_name = 'Mean Frame'
    if stim_type == 'meanRawLiveStim':
        timebucket_mean_name = 'Mean Luminance'
    for stim in this_month_allStim_dict.keys():
        for timebucket in this_month_allStim_dict[stim].keys():
            if timebucket == 'Vid Count':
                monthly_mean_stim[stim]['Vid Count'] = monthly_mean_stim[stim]['Vid Count'] + this_month_allStim_dict[stim]['Vid Count']
            else:
                weighted_means = []
                weights = []
                for month in this_month_allStim_dict[stim][timebucket].keys():
                    weighted_mean = this_month_allStim_dict[stim][timebucket][month]['Trial Count'] * this_month_allStim_dict[stim][timebucket][month][timebucket_mean_name]
                    weighted_means.append(weighted_mean)
                    weights.append(this_month_allStim_dict[stim][timebucket][month]['Trial Count'])
                weighted_means_array = np.array(weighted_means)
                weights_array = np.array(weights)
                this_timebucket_weighted_mean = np.sum(weighted_means, axis=0)/np.sum(weights_array)
                monthly_mean_stim[stim][timebucket] = {'Trial Count':np.sum(weights_array), timebucket_mean_name: this_timebucket_weighted_mean}
    for stim in monthly_mean_stim.keys():
        this_stim_weighted_mean = []
        for timebucket in monthly_mean_stim[stim].keys():
            if timebucket == 'Vid Count':
                vid_count = monthly_mean_stim[stim][timebucket]
            else:
                this_mean_frame = monthly_mean_stim[stim][timebucket][timebucket_mean_name]
                this_mean_frame_weight = monthly_mean_stim[stim][timebucket]['Trial Count']
                this_stim_weighted_mean.append([timebucket, this_mean_frame_weight, this_mean_frame])
        this_stim_weighted_mean_output = monthly_mean_folder + os.sep + '%s_Stim%d_%s_%dVids.npy' % (item_year_month, int(stim), stim_type, vid_count)
        np.save(this_stim_weighted_mean_output, this_stim_weighted_mean)

##########################################################
# BEGIN SCRIPT
##########################################################
if __name__=='__main__':
    # parse command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", nargs='?', default="check_string_for_empty")
    parser.add_argument("--loc", nargs='?', default='laptop')
    args = parser.parse_args()
    # clean up current working directory
    if 'world_temp' in os.listdir(current_working_directory):
        logging.info('Deleting old world_temp folder...')
        print('Deleting old world_temp folder...')
        shutil.rmtree(os.path.join(current_working_directory, 'world_temp'))
        print('Deleted!')
        time.sleep(5) # to have time to see that world_temp was in fact deleted
    zip_folders = fnmatch.filter(os.listdir(current_working_directory), '*.zip')
    if len(zip_folders) > 0:
        logging.info('Deleting old zip folders...')
        print('Deleting old zip folders...')
        for zfolder in zip_folders:
            os.remove(os.path.join(current_working_directory, zfolder))
    ###################################
    # DATA AND OUTPUT FILE LOCATIONS 
    ###################################
    data_drive, analysed_drive, rawStimLum_data, analysed_folders, daily_csv_files, monthly_extracted_data = load_data(args.loc)
    logging.info('MAIN SURPRISING MINDS SOURCE DATASET: %s \n INTERMEDIATE PUPIL SIZE AND LOCATION FILES: %s' % (data_drive, analysed_drive))
    print('MAIN SURPRISING MINDS SOURCE DATASET: %s \n INTERMEDIATE PUPIL SIZE AND LOCATION FILES: %s' % (data_drive, analysed_drive))
    ############################################################################################
    ### CHECK WHETHER COMPLETELY RESTARTING WORLD VID PROCESSING (DELETES 'world' FOLDERS!!!)... 
    ############################################################################################
    if args.a == 'check_string_for_empty':
        logging.info('Continuing world cam extraction and raw live stim creation from last session...')
        print('Continuing world cam extraction and raw live stim creation from last session...')
    elif args.a == 'restart':
        logging.warning('Restarting world cam extraction and raw live stim creation, DELETING ALL FILES FROM PREVIOUS SESSIONS!')
        print('Restarting world cam extraction and raw live stim creation, DELETING ALL FILES FROM PREVIOUS SESSIONS!')
        for folder in daily_csv_files:
            subdirs = os.listdir(os.path.join(analysed_drive, folder, 'Analysis'))
            if 'world' in subdirs:
                shutil.rmtree(os.path.join(analysed_drive, folder, 'Analysis', 'world'))
    else:
        logging.warning('%s is not a valid optional input to this script! \nContinuing world cam extraction and raw live stim creation from last session...' % (args.a))
        print('%s is not a valid optional input to this script! \nContinuing world cam extraction and raw live stim creation from last session...' % (args.a))
    ###################################
    # STIMULUS INFO
    ###################################
    stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
    stim_name_to_float = {"stimuli024": 24.0, "stimuli025": 25.0, "stimuli026": 26.0, "stimuli027": 27.0, "stimuli028": 28.0, "stimuli029": 29.0, 'Stim24':24.0, 'Stim25':25.0, 'Stim26':26.0, 'Stim27':27.0, 'Stim28':28.0, 'Stim29':29.0}
    stim_float_to_name = {24.0: "stimuli024", 25.0: "stimuli025", 26.0: "stimuli026", 27.0: "stimuli027", 28.0: "stimuli028", 29.0: "stimuli029"}
    ###################################
    # LOAD RAW VID STIM DATA
    ###################################
    rawStimLum_files = glob.glob(rawStimLum_data + os.sep + '*.csv')
    rawStimLum_dict = {}
    for rSL_file in rawStimLum_files:
        stim_phase = os.path.basename(rSL_file).split('_')[0]
        stim_lums = np.genfromtxt(rSL_file, delimiter=',')
        thisPhase_lenFrames = len(stim_lums)
        rawStimLum_dict[stim_phase] = {'Number of Frames': thisPhase_lenFrames, 'Luminance per Frame': stim_lums}
    ###################################
    # EXTRACT WORLD CAM VID TIMING AND LUMINANCE
    # SAVE RAW LIVE VIDEOS FROM EACH TRIAL AS BINARY FILE
    ###################################
    # get the subfolders, sort their names
    data_folders = sorted(os.listdir(data_drive))
    zipped_data = fnmatch.filter(data_folders, '*.zip')
    # first day was debugging the exhibit
    zipped_data = zipped_data[1:]
    zipped_names = [item[:-4] for item in zipped_data]
    # figure out which days have already been analysed
    extracted_months = [item.split('_')[1] for item in monthly_extracted_data]
    already_extracted_daily = []
    for folder in daily_csv_files:
        subdirs = os.listdir(os.path.join(analysed_drive, folder, 'Analysis'))
        if 'world' in subdirs:
            already_extracted_daily.append(folder)
    # figure out which days are the last day of data collection for each month of data collection
    last_day_each_month = []
    current_year_month_day = None
    for i, folder in enumerate(zipped_data): 
        this_year_month_day = folder.split('_')[1][:-4]
        this_year_month = this_year_month_day[:-3]
        if current_year_month_day == None:
            if folder == zipped_data[-1]:
                last_day_each_month.append(this_year_month_day)
                continue
            else:
                current_year_month_day = this_year_month_day
                continue
        if current_year_month_day[:-3] == this_year_month:
            if folder == zipped_data[-1]:
                last_day_each_month.append(this_year_month_day)
                continue
            else:
                current_year_month_day = this_year_month_day
                continue
        else:
            last_day_each_month.append(current_year_month_day)
            current_year_month_day = None
    logging.info('Last day of each month: %s' % (last_day_each_month))
    # DAYS THAT CANNOT BE UNZIPPED 
    invalid_zipped = []
    # DAYS WITH NO WORLD VIDS (no valid trials)
    no_valid_trials = []
    # BEGIN WORLD VID FRAME EXTRACTION/AVERAGING 
    for item in zipped_data:
        this_day_date = item[:-4].split('_')[1]
        ########################################################################
        # check to see if this folder has already had world vid frames extracted
        # this condition is for when the script is interrupted
        ########################################################################
        if item[:-4] in already_extracted_daily:
            logging.info("World vid frames from %s has already been extracted" % (item))
            print("World vid frames from %s has already been extracted" % (item))
            ########################################################################################
            # check to see if this folder has already been averaged into a monthly world cam average
            ########################################################################################
            item_year_month = this_day_date[:7]
            if item_year_month in extracted_months:
                logging.info("World camera frames from %s have already been consolidated into a monthly average" % (item_year_month))
                print("World camera frames from %s have already been consolidated into a monthly average" % (item_year_month))
                continue
            ########################################################################################
            # if no monthly world cam average made yet for this month
            # check that the full month has been extracted by checking for all daily mean worldCam and rawLiveStim binary files
            ########################################################################################
            this_month_extracted = fnmatch.filter(already_extracted_daily, 'SurprisingMinds_' + item_year_month + '*')
            for i, day_extracted in enumerate(this_month_extracted):
                if day_extracted in no_valid_trials:
                    logging.warning('No valid trials during %s' % (day_extracted.split('_')[1]))
                else:
                    day_extracted_files = os.listdir(os.path.join(analysed_drive, day_extracted, 'Analysis', 'world'))
                    if len(day_extracted_files) != 12:
                        this_month_extracted.pop(i)
            this_month_data = fnmatch.filter(zipped_data, 'SurprisingMinds_' + item_year_month + '*')
            this_month_invalid = fnmatch.filter(invalid_zipped, item_year_month + '*')
            if len(this_month_extracted) != len(this_month_data) + len(this_month_invalid):
                logging.info("World camera frames for %s not yet completed" % (item_year_month))
                print("World camera frames for %s not yet completed" % (item_year_month))
                continue
            this_month_no_trials = fnmatch.filter(no_valid_trials, 'SurprisingMinds_' + item_year_month + '*')
            if len(this_month_extracted) == len(this_month_no_trials):
                logging.info("No valid trials collected during %s" % (item_year_month))
                print("No valid trials collected during %s" % (item_year_month))
                continue
            ##################################################################
            # full month extracted? make monthly mean worldCam and rawLiveStim
            ##################################################################
            logging.info('This month extraction completed: %s' % (this_month_extracted))
            print('This month extraction completed: %s' % (this_month_extracted))
            # load daily mean files and organize by worldCam/rawLiveStim and by stim
            thisMonth_worldCam = {key:{'Vid Count':0} for key in stim_vids}
            thisMonth_rawLiveStim = {key:{'Vid Count':0} for key in stim_vids}
            for day_extracted in this_month_extracted:
                daily_mean_files = glob.glob(analysed_drive + os.sep + day_extracted + os.sep + 'Analysis' + os.sep + 'world' + os.sep + '*.npy')
                extract_daily_means_and_add_to_worldCam_or_rawLiveStim(daily_mean_files, thisMonth_worldCam, thisMonth_rawLiveStim)
            # create folder for this month mean files
            monthly_mean_folder = analysed_drive + os.sep + 'MeanStimuli_' + item_year_month
            if not os.path.exists(monthly_mean_folder):
                os.makedirs(monthly_mean_folder)
            # take weighted mean at each timebucket and save as monthly mean intermediate binary file
            logging.info('Saving monthly weighted mean of worldCam for %s...'%(item_year_month))
            print('Saving monthly weighted mean of worldCam for %s...'%(item_year_month))
            save_monthly_weighted_meanStim(thisMonth_worldCam, 'meanWorldCam')
            logging.info('Saving monthly weighted mean of rawLive for %s...'%(item_year_month))
            print('Saving monthly weighted mean of rawLive for %s...'%(item_year_month))
            save_monthly_weighted_meanStim(thisMonth_rawLiveStim, 'meanRawLiveStim')
            # update list of already extracted months
            logging.INFO("Updating list of extracted months...")
            analysed_folders = sorted(os.listdir(analysed_drive))
            monthly_extracted_data = fnmatch.filter(analysed_folders, 'MeanStimuli_*')
            extracted_months = [item.split('_')[1] for item in monthly_extracted_data]
            # delete daily mean intermediate files
            for day_extracted in this_month_extracted:
                daily_mean_folder = os.path.join(analysed_drive, day_extracted, 'Analysis', 'world')
                logging.INFO("Deleting daily mean worldCam and rawStim video files for %s..." % (day_extracted.split('_')[1]))
                shutil.rmtree(daily_mean_folder)
                logging.INFO("Delete successful!")
                logging.INFO("Making empty 'world' folder for %s..." % (day_extracted.split('_')[1]))
                os.makedirs(daily_mean_folder)
            logging.info("Finished averaging world video frames for %s!" % (item_year_month))
            print("Finished averaging world video frames for %s!" % (item_year_month))
            continue
        #############################################################################
        # if world vid frames in this folder haven't already been extracted, EXTRACT!
        #############################################################################
        logging.info("Extracting World Vid frames from folder %s" % (item))
        print("Extracting World Vid frames from folder %s" % (item))
        # Build relative analysis paths, these folders should already exist
        analysis_folder = os.path.join(analysed_drive, item[:-4], "Analysis")
        alignment_folder = os.path.join(analysis_folder, "alignment")
        if not os.path.exists(analysis_folder):
            logging.warning("No Analysis folder exists for folder %s!" % (item))
            continue
        # grab a folder 
        day_zipped = os.path.join(data_drive, item)
        # create Analysis subfolder for avg world vid data
        world_folder = os.path.join(analysis_folder, "world")
        # Create world_folder if it doesn't exist
        if not os.path.exists(world_folder):
            os.makedirs(world_folder)
        # create a temp folder in current working directory to store data (contents of unzipped folder)
        day_folder = os.path.join(current_working_directory, "world_temp")
        # at what time resolution to build raw live stim and world camera data?
        bucket_size = 4 #milliseconds
        #####################################################################################################
        # unzip current zipped folder into temp folder, this function checks whether the folder is unzippable
        # if it unzips, the function returns True; if it doesn't unzip, the function returns False
        #####################################################################################################
        if unpack_to_temp(day_zipped, day_folder):
            # List all trial folders
            trial_folders = list_sub_folders(day_folder)
            num_trials = len(trial_folders)
            # intialize time bucket dictionary for world vids
            this_day_worldCam_tbucket = {key:{'Vid Count':0} for key in stim_vids}
            this_day_world_vids_height = []
            this_day_world_vids_width = []
            # initialize time bucket dictionary for raw live stim vids
            this_day_rawLiveVid_tbucket = {key:{'Vid Count':0} for key in stim_vids}
            ###################################
            # extract world vid from each trial
            ###################################
            current_trial = 0
            for trial_folder in trial_folders:
                # add exception handling so that a weird day doesn't totally break everything 
                try:
                    trial_name = trial_folder.split(os.sep)[-1]
                    # check that the alignment frame for the day shows the correct start to the exhibit
                    png_filename = trial_name + '.png'
                    alignment_png_path = os.path.join(alignment_folder, png_filename)
                    if os.path.exists(alignment_png_path):
                        alignment_img = mpimg.imread(alignment_png_path)
                        alignment_gray = cv2.cvtColor(alignment_img, cv2.COLOR_RGB2GRAY)
                        monitor_zoom = alignment_gray[60:-200, 110:-110]
                        monitor_score = np.sum(monitor_zoom)
                        # pick a pixel where it should be bright because people are centering their eyes in the cameras
                        if monitor_zoom[115,200]>=0.7:
                            ###################################
                            # Load CSVs and create timestamps
                            # ------------------------------
                            # Get world movie timestamp csv path
                            world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]
                            # Get world video filepath
                            world_video_path = glob.glob(trial_folder + '/*world.avi')[0]
                            ####################################
                            # while debugging
                            #world_csv_path = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\debuggingData\SurprisingMinds_2017-10-14\2017-10-14_09-42-40\2017-10-14_09-42-40_stimuli024_world.csv"
                            #world_video_path = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\debuggingData\SurprisingMinds_2017-10-14\2017-10-14_09-42-40\2017-10-14_09-42-40_stimuli024_world.avi"
                            #world_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows\SurprisingMinds_2017-10-14\Analysis\world"
                            ####################################
                            stimuli_name = world_csv_path.split("_")[-2]
                            stimuli_number = stim_name_to_float[stimuli_name]
                            # Load world CSV
                            world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ') # row = timestamp, not frame
                            ### EXTRACT FRAMES FROM WORLD VIDS AND PUT INTO TIME BUCKETS ###
                            # create a "raw live stimulus video" array by combining framerate info from world cam with luminance values from raw vids
                            logging.INFO("Extracting world vid frames and creating raw live stim vid for %s..." % os.path.basename(world_video_path))
                            # save raw live stim vid as binary files and return world cam frames as a sanity check
                            worldCam_vidWidth, worldCam_vidHeight, worldCam_supersampledFrames, rawLiveVid_supersampledFrames = supersampled_worldCam_rawLiveVid(world_video_path, world_timestamps, rawStimLum_dict, world_folder, bucket_size)
                            #
                            # ## SANITY CHECK
                            # worldCam_meanLum = []
                            # for frame in worldCam_supersampledFrames:
                            #     worldCam_meanLum.append(np.nansum(frame))
                            # worldCam_meanLum_array = np.array(worldCam_meanLum)
                            # plt.plot(worldCam_meanLum_array)
                            # plt.show()
                            #
                            add_to_daily_worldCam_dict(worldCam_supersampledFrames, stimuli_number, this_day_worldCam_tbucket)
                            this_day_world_vids_width.append(worldCam_vidWidth)
                            this_day_world_vids_height.append(worldCam_vidHeight)
                            # ------------------------------
                            add_to_daily_rawLiveVid_dict(rawLiveVid_supersampledFrames, stimuli_number, this_day_rawLiveVid_tbucket)
                            # ------------------------------
                            # Report progress
                            cv2.destroyAllWindows()
                            logging.info("Finished Trial: %s" % (current_trial))
                            print("Finished Trial: %s" % (current_trial))
                            current_trial = current_trial + 1
                        else:
                            logging.warning("Bad trial! Stimulus did not display properly for trial %s" % (current_trial))
                            print("Bad trial! Stimulus did not display properly for trial %s" % (current_trial))
                            current_trial = current_trial + 1
                    else:
                        logging.warning("No alignment picture exists for trial %s" % (current_trial))
                        print("No alignment picture exists for trial %s" % (current_trial))
                        current_trial = current_trial + 1
                except Exception: 
                    cv2.destroyAllWindows()
                    logging.warning("Trial %s failed!" % (current_trial))
                    print("Trial %s failed!" % (current_trial))
                    current_trial = current_trial + 1
            ##################################################
            # check that all videos have same height and width
            ##################################################
            if not this_day_world_vids_height:
                logging.warning("No world vids averaged for %s" % (this_day_date))
                no_valid_trials.append(item)
                # delete temporary file with unzipped data contents
                logging.INFO("Deleting temp folder of unzipped data...")
                shutil.rmtree(day_folder)
                logging.INFO("Delete successful!")
                continue
            if all(x == this_day_world_vids_height[0] for x in this_day_world_vids_height):
                if all(x == this_day_world_vids_width[0] for x in this_day_world_vids_width):
                    unravel_height = this_day_world_vids_height[0]
                    unravel_width = this_day_world_vids_width[0]
            ###########################################
            # average worldCam sanityCheck for each day
            ###########################################
            logging.info('Calculating mean world camera videos for %s' % (this_day_date))
            print('Calculating mean world camera videos for %s' % (this_day_date))
            thisDay_meanWorldCam = calculate_meanPerDay_worldCam(this_day_worldCam_tbucket)
            logging.info('Saving non-NaN frames of daily mean world camera...')
            print('Saving non-NaN frames of daily mean world camera...')
            save_daily_worldCam_meanFrames(thisDay_meanWorldCam, this_day_date, world_folder)
            ###########################################
            # average rawLiveStim video for each day
            ###########################################
            logging.info('Calculating mean raw live stim videos for %s' % (this_day_date))
            print('Calculating mean raw live stim videos for %s' % (this_day_date))
            thisDay_meanRawLiveVid = calculate_meanPerDay_rawLiveVid(this_day_rawLiveVid_tbucket)
            logging.info('Saving daily mean raw live stim videos...')
            print('Calculating mean raw live stim videos for %s' % (this_day_date))
            save_daily_rawLiveStim_meanLums(thisDay_meanRawLiveVid, this_day_date, world_folder)
            ####################################################
            # report progress and update already_extracted_daily
            ####################################################
            already_extracted_daily.append(item[:-4])
            logging.info("Finished extracting from %s" % (day_zipped[:-4]))
            print("Finished extracting from %s" % (day_zipped[:-4]))
            ###################################################
            # delete temporary file with unzipped data contents
            ###################################################
            logging.INFO("Deleting temp folder of unzipped data...")
            shutil.rmtree(day_folder)
            logging.INFO("Delete successful!")
        else:
            logging.warning("Could not unzip data folder for day %s" % (this_day_date))
            invalid_zipped.append(this_day_date)
            logging.warning("Days that cannot be unzipped: %s" % (invalid_zipped))
        #############################################
        # check if this was the last day in the month
        #############################################
        this_day_date = item.split('_')[1][:-4]
        item_year_month = this_day_date[:7]
        if this_day_date in last_day_each_month:
            ##################################################
            # build monthly mean worldCam and rawLive vid data
            ##################################################
            logging.info("Completed world camera frame extraction and raw live stimuli creation for %s, now building monthly mean world cam and raw live data files..." % (item_year_month))
            print("Completed world camera frame extraction and raw live stimuli creation for %s, now building monthly mean world cam and raw live data files..." % (item_year_month))
            ########################################################################################
            # check to see if this folder has already been averaged into a monthly world cam average
            ########################################################################################
            if item_year_month in extracted_months:
                logging.info("World camera frames from %s have already been consolidated into a monthly average" % (item_year_month))
                print("World camera frames from %s have already been consolidated into a monthly average" % (item_year_month))
                continue
            ########################################################################################
            # if no monthly world cam average made yet for this month
            # check that the full month has been extracted by checking for all daily mean worldCam and rawLiveStim binary files
            ########################################################################################
            this_month_extracted = fnmatch.filter(already_extracted_daily, 'SurprisingMinds_' + item_year_month + '*')
            for i, day_extracted in enumerate(this_month_extracted):
                day_extracted_files = os.listdir(os.path.join(analysed_drive, day_extracted, 'Analysis', 'world'))
                if len(day_extracted_files) != 12:
                    this_month_extracted.pop(i)
            this_month_data = fnmatch.filter(zipped_data, 'SurprisingMinds_' + item_year_month + '*')
            this_month_invalid = fnmatch.filter(invalid_zipped, item_year_month)
            if len(this_month_extracted) != len(this_month_data) + len(this_month_invalid):
                logging.info("World vid frames for %s not yet completed" % (item_year_month))
                print("World vid frames for %s not yet completed" % (item_year_month))
                continue
            ##################################################################
            # full month extracted? make monthly mean worldCam and rawLiveStim
            ##################################################################
            logging.info('This month extraction completed: %s' % (this_month_extracted))
            print('This month extraction completed: %s' % (this_month_extracted))
            # load daily mean files and organize by worldCam/rawLiveStim and by stim
            thisMonth_worldCam = {key:{'Vid Count':0} for key in stim_vids}
            thisMonth_rawLiveStim = {key:{'Vid Count':0} for key in stim_vids}
            for day_extracted in this_month_extracted:
                daily_mean_files = glob.glob(analysed_drive + os.sep + day_extracted + os.sep + 'Analysis' + os.sep + 'world' + os.sep + '*.npy')
                extract_daily_means_and_add_to_worldCam_or_rawLiveStim(daily_mean_files, thisMonth_worldCam, thisMonth_rawLiveStim)
            # create folder for this month mean files
            monthly_mean_folder = analysed_drive + os.sep + 'MeanStimuli_' + item_year_month
            if not os.path.exists(monthly_mean_folder):
                os.makedirs(monthly_mean_folder)
            # take weighted mean at each timebucket and save as monthly mean intermediate binary file
            logging.info('Saving monthly weighted mean of worldCam for %s...'%(item_year_month))
            print('Saving monthly weighted mean of worldCam for %s...'%(item_year_month))
            save_monthly_weighted_meanStim(thisMonth_worldCam, 'meanWorldCam')
            logging.info('Saving monthly weighted mean of rawLive for %s...'%(item_year_month))
            print('Saving monthly weighted mean of rawLive for %s...'%(item_year_month))
            save_monthly_weighted_meanStim(thisMonth_rawLiveStim, 'meanRawLiveStim')
            # update list of already extracted months
            logging.INFO("Updating list of extracted months...")
            analysed_folders = sorted(os.listdir(analysed_drive))
            monthly_extracted_data = fnmatch.filter(analysed_folders, 'MeanStimuli_*')
            extracted_months = [item.split('_')[1] for item in monthly_extracted_data]
            # delete daily mean intermediate files
            for day_extracted in this_month_extracted:
                daily_mean_folder = os.path.join(analysed_drive, day_extracted, 'Analysis', 'world')
                logging.INFO("Deleting daily mean worldCam and rawStim video files for %s..." % (day_extracted.split('_')[1]))
                shutil.rmtree(daily_mean_folder)
                logging.INFO("Delete successful!")
                logging.INFO("Making empty 'world' folder for %s..." % (day_extracted.split('_')[1]))
                os.makedirs(daily_mean_folder)
            logging.info("Finished averaging world video frames for %s!" % (item_year_month))
            print("Finished averaging world video frames for %s!" % (item_year_month))

    logging.info("Completed world camera frame extraction and raw live stimuli creation on all data folders in this drive!")
    print("Completed world camera frame extraction and raw live stimuli creation on all data folders in this drive!")
#FIN
