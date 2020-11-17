### --------------------------------------------------------------------------- ###
# loads monthly mean raw live stim and world cam luminance data files (saved at 4ms resolution)
# saves sanity check mean world cam video for each stimulus type
# measures display latency (lag between when bonsai tells a frame to display and when it actually displays)
# output display latency and sanity check plots/videos
# NOTE: this script uses ImageMagick to easily install ffmpeg onto Windows 10 (https://www.imagemagick.org/script/download.php)
# NOTE: in command line run with optional tags 
#       1) '--a debug' to use only a subset of pupil location/size data
#       2) '--a vid_output' to generate sanity check mean world cam videos
#       3) '--a MOI' to find moments of interest in the mean world cam videos
#       4) '--loc *' to run with various root data locations (see first function below)
### --------------------------------------------------------------------------- ###
import logging
import pdb
import os
import glob
import datetime
import math
import sys
import itertools
import csv
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats
import argparse
from IPython import embed
###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
logging.basicConfig(filename="psa02_DisplayLatency_SanityChecks" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log", filemode='w', level=logging.INFO)
###################################
# FUNCTIONS
###################################

##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) root_folder (parent folder with all intermediate data)
# AND
# 2) plots_folder (parent folder for all plots output from analysis scripts)
### Current default uses a debugging source dataset
##########################################################
def load_data(location='laptop'):
    if location == 'laptop':
        root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
        plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
    elif location == 'office':
        root_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
        plots_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\plots"
    # monthly mean raw live and world cam luminances
    monthly_mean_lums_folders = fnmatch.filter(sorted(os.listdir(root_folder)), 'MeanStimuli_*')
    # display latency
    display_latency_folder = os.path.join(root_folder, 'displayLatency')
    # full dataset mean world cam output folder
    mean_world_cam_vids_folder = os.path.join(plots_folder, 'meanWorldCam')
    # sanity check
    raw_v_world_sanity_check_folder = os.path.join(plots_folder, 'rawVWorldSanityCheck')
    # Create output folders if they do not exist
    output_folders = [display_latency_folder, mean_world_cam_vids_folder, raw_v_world_sanity_check_folder]
    for output_folder in output_folders:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    return root_folder, plots_folder, monthly_mean_lums_folders, output_folders

##########################################################
def extract_stim_data(stim_data_array, world_or_raw, stim_data_dict):
    for row in stim_data_array:
        timebucket = row[0]
        weight = row[1]
        mean_data = row[2]
        this_tb_weighted_data = weight*mean_data
        if world_or_raw=='world':
            data_type = 'weighted keyframes'
        elif world_or_raw=='raw':
            data_type = 'weighted luminance'
        else: 
            print('Invalid stimulus data type! Must be world or raw.')
            logging.warning('Invalid stimulus data type! Must be world or raw.')
        if timebucket in stim_data_dict.keys():
            stim_data_dict[timebucket][data_type].append(this_tb_weighted_data)
            stim_data_dict[timebucket]['weights'].append(weight)
        else:
            stim_data_dict[timebucket] = {data_type:[this_tb_weighted_data], 'weights':[weight]}

def calculate_weighted_sums(all_weighted_stim_data_dict, world_or_raw):
    weighted_sums_stim_dict = {}
    for stim in all_weighted_stim_data_dict.keys():
        weighted_sums_stim_dict[stim] = {}
        for timebucket in sorted(all_weighted_stim_data_dict[stim].keys()):
            if world_or_raw=='world':
                this_tb_weighted_sum = np.sum(np.array(all_weighted_stim_data_dict[stim][timebucket]['weighted keyframes']), axis=0)
                weighted_sum_key = 'keyframe, weighted sum'
            elif world_or_raw=='raw':
                this_tb_weighted_sum = np.sum(all_weighted_stim_data_dict[stim][timebucket]['weighted luminance'])
                weighted_sum_key = 'luminance, weighted sum'
            else:
                print('Invalid stimulus data type! Must be world or raw.')
                logging.warning('Invalid stimulus data type! Must be world or raw.')
            this_tb_weights_sum = np.sum(all_weighted_stim_data_dict[stim][timebucket]['weights'])
            weighted_sums_stim_dict[stim][timebucket] = {weighted_sum_key:this_tb_weighted_sum, 'summed weight':this_tb_weights_sum}
    return weighted_sums_stim_dict

def sanity_check_world_v_rawLive(world_dict, worldFull_or_worldCropped, raw_dict, timebucket_size, save_folder):
    for stim in world_dict.keys():
        if worldFull_or_worldCropped == 'full':
            world_label = 'world (full)'
            # figure path and title
            figPath = os.path.join(save_folder, 'Stim%d_worldFull_versus_raw_sanityCheck.png'%(stim))
            figTitle = 'Mean luminance of world cam (full) vs mean luminance of raw live stim during Stimulus %d'%(stim)
        elif worldFull_or_worldCropped == 'cropped':
            world_label = 'world (cropped)'
            # figure path and title
            figPath = os.path.join(save_folder, 'Stim%d_worldCropped_versus_raw_sanityCheck.png'%(stim))
            figTitle = 'Mean luminance of world cam (cropped to remove display latency) \n vs mean luminance of raw live stim during Stimulus %d'%(stim)
        plt.figure(figsize=(9, 9), dpi=200)
        plt.suptitle(figTitle, fontsize=12, y=0.98)
        plt.ylabel('Mean Luminance')
        plt.xlabel('Time (sec)')
        plt.xticks(np.arange(0,len(raw_dict[stim]),1000), np.arange(0,len(raw_dict[stim]),1000)*timebucket_size/1000, rotation=50)
        plt.plot(world_dict[stim], label=world_label)
        plt.plot(raw_dict[stim], label='raw')
        plt.legend()
        plt.savefig(figPath)
        plt.close()

def sanity_check_mean_world_vid(full_world_cam_dict, world_downsample_ms, original_sample_rate_ms, save_folder):
    fps_rate = int(1000/world_downsample_ms)
    world_cam_downsample_mult = int(world_downsample_ms/original_sample_rate_ms)
    for stim in full_world_cam_dict.keys():
        tbs_to_sample = np.arange(0, len(full_world_cam_dict[stim]['all frames, weighted sum']), world_cam_downsample_mult)
        downsampled_mean_frames = []
        for current_tb in tbs_to_sample:
            this_mean_frame = full_world_cam_dict[stim]['all frames, weighted sum'][current_tb]/full_world_cam_dict[stim]['weights'][current_tb]
            downsampled_mean_frames.append(this_mean_frame)
        # reshape into original world cam dimensions
        downsampled_mean_frames_reshaped = []
        for frame in downsampled_mean_frames:
            reshaped_frame = np.reshape(frame,(120,160))
            downsampled_mean_frames_reshaped.append(reshaped_frame)
        downsampled_mean_frames_reshaped = np.array(downsampled_mean_frames_reshaped)
        # save as mp4 video file
        write_path = os.path.join(save_folder, 'Stim%d_MeanWorldCam.mp4'%(stim))
        end_tbucket = len(downsampled_mean_frames_reshaped)
        # temporarily switch matplotlib backend in order to write video
        plt.switch_backend("Agg")
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        FF_writer = animation.FFMpegWriter(fps=fps_rate, codec='h264', metadata=dict(artist='Danbee Kim', album='Surprising Minds'))
        print('Drawing frames...')
        fig = plt.figure()
        i = 0
        im = plt.imshow(downsampled_mean_frames_reshaped[i], cmap='gray', animated=True)
        def updatefig(*args):
            global i
            if (i<end_tbucket-1):
                i += 1
            else:
                i = 0
            im.set_array(downsampled_mean_frames_reshaped[i])
            return im,
        ani = animation.FuncAnimation(fig, updatefig, frames=len(downsampled_mean_frames_reshaped), interval=world_downsample_ms, blit=True)
        print("Writing average world video frames to {path}...".format(path=write_path))
        ani.save(write_path, writer=FF_writer)
        plt.close(fig)
        print("Finished writing!")
        # restore default matplotlib backend
        plt.switch_backend('TkAgg')

##########################################################
# BEGIN SCRIPT
##########################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", nargs='?', default="no_vid_output")
    parser.add_argument("--loc", nargs='?', default='laptop')
    args = parser.parse_args()
    ###################################
    # SOURCE DATA AND OUTPUT FILE LOCATIONS 
    ###################################
    root_folder, plots_folder, monthly_mean_lums_folders, output_folders = load_data(args.loc)
    display_latency_folder = output_folders[0]
    mean_world_cam_vids_folder = output_folders[1]
    raw_v_world_sanity_check_folder = output_folders[2]
    logging.info('\n MAIN DATA FOLDER: %s \n PLOTS FOLDER: %s \n MONTHLY MEAN RAW LIVE AND WORLD CAM STIMULI DATA FOLDER: %s' % (root_folder, plots_folder, monthly_mean_lums_folders))
    print('MAIN DATA FOLDER: %s \n PLOTS FOLDER: %s \n MONTHLY MEAN RAW LIVE AND WORLD CAM STIMULI DATA FOLDER: %s' % (root_folder, plots_folder, monthly_mean_lums_folders))
    ###################################
    # TIMING/SAMPLING VARIABLES FOR DATA EXTRACTION
    ###################################
    # downsample = collect data from every 40ms or other multiples of 20
    downsampled_bucket_size_ms = 40
    original_bucket_size_in_ms = 4
    max_length_of_stim_vid = 60000 # milliseconds
    no_of_time_buckets = max_length_of_stim_vid/original_bucket_size_in_ms
    downsampled_no_of_time_buckets = max_length_of_stim_vid/downsampled_bucket_size_ms
    downsample_multiplier = int(downsampled_bucket_size_ms/original_bucket_size_in_ms)
    ###################################
    # STIMULI VID INFO
    ###################################
    stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
    stim_name_to_float = {"Stim24": 24.0, "Stim25": 25.0, "Stim26": 26.0, "Stim27": 27.0, "Stim28": 28.0, "Stim29": 29.0}
    stim_float_to_name = {24.0: "Stim24", 25.0: "Stim25", 26.0: "Stim26", 27.0: "Stim27", 28.0: "Stim28", 29.0: "Stim29"}
    phase_names = ['calib', 'octo', 'unique1', 'unique2', 'unique3', 'unique4', 'unique5', 'unique6']
    ###############################################################
    # Load mean monthly raw live stim and world cam luminance files
    # extract monthly mean world cam and raw live stim data (4ms resolution)
    ###############################################################
    # world cam data structure
    all_weighted_world_keyFrames = {key:{} for key in stim_vids}
    # raw live stim data structure
    all_weighted_rawLive_timebuckets = {key:{} for key in stim_vids}
    # collect length of each stimulus type in 4ms resolution
    supersampled_length_all_stims = {key:[] for key in stim_vids}
    # extract raw live stim and world cam data
    for monthly_mean_folder in monthly_mean_lums_folders:
        raw_live_stim_files = glob.glob(root_folder + os.sep + monthly_mean_folder + os.sep + '*_meanRawLiveStim_*.npy')
        world_cam_files = glob.glob(root_folder + os.sep + monthly_mean_folder + os.sep + '*_meanWorldCam_*.npy')
        # extract world cam data
        for world_cam in world_cam_files:
            stim_type = stim_name_to_float[os.path.basename(world_cam).split('_')[1]]
            vid_count = float(os.path.basename(world_cam).split('_')[-1][:-8])
            world_cam_frames = np.load(world_cam)
            extract_stim_data(world_cam_frames, 'world', all_weighted_world_keyFrames[stim_type])
        # extract raw live data
        for raw_live_stim in raw_live_stim_files:
            stim_type = stim_name_to_float[os.path.basename(raw_live_stim).split('_')[1]]
            vid_count = float(os.path.basename(raw_live_stim).split('_')[-1][:-8])
            raw_live_array = np.load(raw_live_stim)
            supersampled_length_all_stims[stim_type].append(len(raw_live_array))
            extract_stim_data(raw_live_array, 'raw', all_weighted_rawLive_timebuckets[stim_type])
    ########################################################
    # Calculate full dataset mean world cam for each stimulus
    ########################################################
    logging.info('Calculating full dataset mean world camera for each unique stimulus...')
    print('Calculating full dataset mean world camera for each unique stimulus...')
    weighted_sums_world_keyFrames = calculate_weighted_sums(all_weighted_world_keyFrames, 'world')
    # Fill in gaps between keyframes in world cam
    weighted_sums_world_all_frames = {key:{} for key in stim_vids}
    for stim in weighted_sums_world_keyFrames.keys():
        ordered_keyframes = sorted(weighted_sums_world_keyFrames[stim].keys())
        this_stim_all_weighted_frames = []
        this_stim_all_weights = []
        for i, keyframe in enumerate(ordered_keyframes):
            if i==0 and keyframe==0:
                this_stim_all_weighted_frames.append(weighted_sums_world_keyFrames[stim][keyframe]['keyframe, weighted sum'])
                this_stim_all_weights.append(weighted_sums_world_keyFrames[stim][keyframe]['summed weight'])
            elif i==0 and keyframe!=0:
                for frame in range(keyframe-1):
                    this_stim_all_weighted_frames.append(np.nan)
                    this_stim_all_weights.append(np.nan)
                this_stim_all_weighted_frames.append(weighted_sums_world_keyFrames[stim][keyframe]['keyframe, weighted sum'])
                this_stim_all_weights.append(weighted_sums_world_keyFrames[stim][keyframe]['summed weight'])
            else:
                prev_keyframe = ordered_keyframes[i-1]
                if keyframe - prev_keyframe > 1:
                    for frame in range(prev_keyframe, keyframe-1):
                        this_stim_all_weighted_frames.append(weighted_sums_world_keyFrames[stim][prev_keyframe]['keyframe, weighted sum'])
                        this_stim_all_weights.append(weighted_sums_world_keyFrames[stim][prev_keyframe]['summed weight'])
                this_stim_all_weighted_frames.append(weighted_sums_world_keyFrames[stim][keyframe]['keyframe, weighted sum'])
                this_stim_all_weights.append(weighted_sums_world_keyFrames[stim][keyframe]['summed weight'])
        full_length_this_stim = np.min(supersampled_length_all_stims[stim])
        if ordered_keyframes[-1] < full_length_this_stim:
            for frame in range(ordered_keyframes[-1], full_length_this_stim):
                this_stim_all_weighted_frames.append(weighted_sums_world_keyFrames[stim][ordered_keyframes[-1]]['keyframe, weighted sum'])
                this_stim_all_weights.append(weighted_sums_world_keyFrames[stim][ordered_keyframes[-1]]['summed weight'])
        weighted_sums_world_all_frames[stim] = {'all frames, weighted sum':this_stim_all_weighted_frames, 'weights':this_stim_all_weights}
    # Calculate weighted mean frame and weighted mean luminance for each world cam timebucket
    world_all_weighted_mean_frames = {key:None for key in stim_vids}
    world_all_weighted_mean_luminance = {key:None for key in stim_vids}
    for stim in weighted_sums_world_all_frames.keys():
        this_stim_weighted_mean_frames = []
        this_stim_weighted_mean_lum = []
        for tb, summed_frame in enumerate(weighted_sums_world_all_frames[stim]['all frames, weighted sum']):
            this_tb_weighted_mean_frame = summed_frame/weighted_sums_world_all_frames[stim]['weights'][tb]
            this_stim_weighted_mean_frames.append(this_tb_weighted_mean_frame)
            this_tb_weighted_mean_luminance = np.sum(this_tb_weighted_mean_frame)
            this_stim_weighted_mean_lum.append(this_tb_weighted_mean_luminance)
        world_all_weighted_mean_frames[stim] = np.array(this_stim_weighted_mean_frames)
        world_all_weighted_mean_luminance[stim] = np.array(this_stim_weighted_mean_lum)
    ########################################################
    # Find timebuckets marking start and end of each phase
    # UNDER CONSTRUCTION
    ########################################################
    if args.a == 'MOI':
        reshaped_world_all_weighted_mean_frames = {key:None for key in stim_vids}
        for stim in world_all_weighted_mean_frames.keys():
            reshaped_frames = []
            for i, frame in enumerate(world_all_weighted_mean_frames[stim]):
                reshaped_frame = np.reshape(frame,(120,160))
                reshaped_frames.append(reshaped_frame)
                # figure path and title
                figPath = os.path.join(mean_world_cam_vids_folder, 'Stim%d_meanWorldCamSanityCheck_tb%06d_4msResolution.png'%(stim, i))
                figTitle = 'Stim%d: mean world cam sanity check \n timebucket: %06d'%(stim, i)
                plt.figure(figsize=(9, 9), dpi=200)
                plt.suptitle(figTitle, fontsize=12, y=0.98)
                plt.imshow(reshaped_frame)
                plt.savefig(figPath)
                plt.close()
                
            reshaped_world_all_weighted_mean_frames[stim] = np.array(reshaped_frames)
        stim_to_check = input('Which unique stimulus would you like to check for moments of interest?')
        print('Checking %s'%(stim_to_check))
        go_to_timebucket = input('Jump to timebucket:')
        embed()

def display_mean_world_vid_frame(world_vid_frames_dict, stim_num, timebucket):
    plt.imshow(world_vid_frames_dict[stim_num][timebucket])
    plt.show()

    ########################################################
    # Calculate full dataset raw live for each stimulus
    ########################################################
    logging.info('Calculating full dataset raw live vid for each unique stimulus...')
    print('Calculating full dataset raw live vid for each unique stimulus...')
    weighted_sums_raw_timebuckets = calculate_weighted_sums(all_weighted_rawLive_timebuckets, 'raw')
    # Calculate weighted mean luminance for each raw live stim timebucket
    raw_all_weighted_mean_luminance = {key:None for key in stim_vids}
    for stim in weighted_sums_raw_timebuckets.keys():
        this_stim_weighted_mean_lum = []
        for tb in sorted(weighted_sums_raw_timebuckets[stim].keys()):
            this_tb_weighted_mean_lum = weighted_sums_raw_timebuckets[stim][tb]['luminance, weighted sum']/weighted_sums_raw_timebuckets[stim][tb]['summed weight']
            this_stim_weighted_mean_lum.append(this_tb_weighted_mean_lum)
        raw_all_weighted_mean_luminance[stim] = np.array(this_stim_weighted_mean_lum)
    ########################################################
    # Calculate display latency
    # this should be when the mean luminance in world cam drops more than 400,000
    # save as binary file output
    ########################################################
    all_true_start_tb = []
    all_true_start_tb_dict = {key:None for key in stim_vids}
    for stim in world_all_weighted_mean_luminance.keys():
        true_calib_start = None
        for tb, mean_lum in enumerate(world_all_weighted_mean_luminance[stim]):
            if true_calib_start==None:
                true_calib_start = [tb, mean_lum]
            elif abs(true_calib_start[1]-mean_lum)>400000:
                true_calib_start = [tb, mean_lum]
                break
        all_true_start_tb.append([stim, true_calib_start[0]])
        all_true_start_tb_dict[stim] = true_calib_start[0]
        logging.info('Display latency for stim %d: %d'%(stim, true_calib_start[0]))
        print('Display latency for stim %d: %d'%(stim, true_calib_start[0]))
    all_true_start_tb_array = np.array(all_true_start_tb)
    logging.info('Writing display latencies to binary file...')
    print('Writing display latencies to binary file...')
    display_latency_output = display_latency_folder + os.sep + 'displayLatencies.npy'
    np.save(display_latency_output, all_true_start_tb_array)
    ########################################################
    # sanity check: plot mean world cam (cropped to "real start") + full raw live for each stimulus to visually check display latency
    # save plots
    ########################################################
    # CROP LUMINANCE ARRAYS BASED ON all_true_start_tb
    world_all_weighted_mean_luminance_cropped = {key:None for key in stim_vids}
    for stim in all_true_start_tb_dict.keys():
        true_start = all_true_start_tb_dict[stim]
        cropped_lum_array = world_all_weighted_mean_luminance[stim][true_start:]
        world_all_weighted_mean_luminance_cropped[stim] = cropped_lum_array
    # plot sanity check
    logging.info('Saving sanity check plots of mean luminance for world cam versus raw live stim...')
    print('Saving sanity check plots of mean luminance for world cam versus raw live stim...')
    sanity_check_world_v_rawLive(world_all_weighted_mean_luminance_cropped, 'cropped', raw_all_weighted_mean_luminance, original_bucket_size_in_ms, raw_v_world_sanity_check_folder)
    ########################################################
    # save full dataset mean world cam video
    # downsample mean world cam video for 25 fps (one frame every 40 ms)
    ########################################################
    if args.a == 'no_vid_output':
        logging.info('No mean world cam video output saved. To save mean world cam video, run with optional input --a vid_output.')
        print('No mean world cam video output saved. To save mean world cam video, run with optional input --a vid_output.')
    elif args.a == 'vid_output':
        logging.info('Saving sanity check videos of mean luminance for world cam...')
        print('Saving sanity check videos of mean luminance for world cam...')
        sanity_check_mean_world_vid(weighted_sums_world_all_frames, downsampled_bucket_size_ms, original_bucket_size_in_ms, mean_world_cam_vids_folder)
    else:
        logging.warning('%s is not a valid optional input to this script! \n Completing script without generating mean world cam video output...' % (args.a))
        print('%s is not a valid optional input to this script! \n Completing script without generating mean world cam video output...' % (args.a))
# FIN