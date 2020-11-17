### --------------------------------------------------------------------------- ###
# loads binary files of monthly mean raw live stim data 
# load binary file of display latencies
# split into calib, octo, unique 1-6
# outputs normalized pupil sizes and lin regression params for all phases as binary files
# creates a scatter plot comparing luminance to pupil size
# NOTE: in command line run with optional tags 
#       1) '--a debug' to use only a subset of pupil location/size data
#       2) '--a incomplete' to run this script while psa01_MonthlyMeans_WorldCam_RawLiveStim.py is still running
#       3) '--loc *' to run with various root data locations (see first function below)
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
###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
logging.basicConfig(filename="psa03_PupilSizeVLum_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log", filemode='w', level=logging.INFO)
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
    # display latencies
    display_latencies_folder = os.path.join(root_folder, 'displayLatency')
    # plots output folder
    pupilSize_folder = os.path.join(plots_folder, "pupilSizeAnalysis")
    Rco_scatter_folder = os.path.join(pupilSize_folder, 'rightContours', 'scatter')
    Rci_scatter_folder = os.path.join(pupilSize_folder, 'rightCircles', 'scatter')
    Lco_scatter_folder = os.path.join(pupilSize_folder, 'leftContours', 'scatter')
    Lci_scatter_folder = os.path.join(pupilSize_folder, 'leftCircles', 'scatter')
    Rco_rvalVsDelay_folder = os.path.join(pupilSize_folder, 'rightContours', 'rvalVsDelay')
    Rci_rvalVsDelay_folder = os.path.join(pupilSize_folder, 'rightCircles', 'rvalVsDelay')
    Lco_rvalVsDelay_folder = os.path.join(pupilSize_folder, 'leftContours', 'rvalVsDelay')
    Lci_rvalVsDelay_folder = os.path.join(pupilSize_folder, 'leftCircles', 'rvalVsDelay')
    # normed mean pupil sizes output folder
    normedMeanPupilSizes_folder = os.path.join(root_folder, 'normedMeanPupilSizes')
    pupilSizeVsDelayLinRegress_folder = os.path.join(root_folder, 'pupilSizeVsDelayLinRegress')
    # Create output folders if they do not exist
    output_folders = [pupilSize_folder, Rco_scatter_folder, Rci_scatter_folder, Lco_scatter_folder, Lci_scatter_folder, Rco_rvalVsDelay_folder, Rci_rvalVsDelay_folder, Lco_rvalVsDelay_folder, Lci_rvalVsDelay_folder, normedMeanPupilSizes_folder, pupilSizeVsDelayLinRegress_folder]
    for output_folder in output_folders:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    return root_folder, plots_folder, monthly_mean_lums_folders, display_latencies_folder, output_folders

##########################################################
def load_daily_pupils(which_eye, day_csv_folder_path, max_no_of_buckets, original_bucket_size, new_bucket_size, display_latency_dict):
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
        # iterate through trials
        index = 0
        for trial_file in trial_files:
            trial_name = trial_file.split(os.sep)[-1]
            trial_stimulus = trial_name.split("_")[1]
            trial_stim_number = np.float(trial_stimulus[-2:])
            trial = np.genfromtxt(trial_file, dtype=np.float, delimiter=",")
            # crop out display latency from beginning of trial
            this_trial_display_latency = int(display_latency_dict[trial_stim_number])
            trial = trial[this_trial_display_latency:]
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
        print('Sample rate must be a multiple of %s'%(original_bucket_size))
        logging.WARNING('Sample rate must be a multiple of %s'%(original_bucket_size))

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

def normPupilSizeData(pupilSizeArrays_allStim, eyeAnalysis_name):
    normed_pupils = []
    for i, stim_trials in enumerate(pupilSizeArrays_allStim):
        print('Normalizing trials for %s, unique stim %s'%(eyeAnalysis_name, i+1))
        looging.INFO('Normalizing trials for %s, unique stim %s'%(eyeAnalysis_name, i+1))
        thisUnique_normed = []
        for trial in stim_trials:
            trial_median = np.nanmedian(trial)
            normed_trial = trial/trial_median
            thisUnique_normed.append(normed_trial)
        thisUniqueNormed_array = np.array(thisUnique_normed)
        normed_pupils.append(thisUniqueNormed_array)
    return normed_pupils

def extract_MOI_tb_downsample_corrected(all_MOI_dict, stim_num, MOI_name, start_or_end, downsample_mult):
    MOI_timebuckets = []
    for key in all_MOI_dict[stim_num][MOI_name]:
        MOI_timebuckets.append(key)
    if start_or_end == 'start':
        MOI_timebucket = np.min(MOI_timebuckets)
    if start_or_end == 'end':
        MOI_timebucket = np.max(MOI_timebuckets)
    if start_or_end != 'start' and start_or_end != 'end':
        logging.warning('Incorrect input for parameter start_or_end! Current input: %s' % (start_or_end))
    return MOI_timebucket*downsample_mult

def trim_phase_extractions(all_weighted_raw_extractions):
    min_len_extraction = np.inf
    for extraction in all_weighted_raw_extractions:
        if len(extraction) < min_len_extraction:
            min_len_extraction = len(extraction)
    trimmed_all_weighted_raw_extractions = []
    for extraction in all_weighted_raw_extractions:
        trimmed_all_weighted_raw_extractions.append(extraction[:min_len_extraction])
    return trimmed_all_weighted_raw_extractions

def calculate_weighted_mean_lum(all_weighted_raw_lums, all_weights):
    trimmed_all_weighted_raw_lums = trim_phase_extractions(all_weighted_raw_lums)
    trimmed_all_weights = trim_phase_extractions(all_weights)
    summed_lums = np.sum(trimmed_all_weighted_raw_lums, axis=0)
    summed_weights = np.sum(trimmed_all_weights, axis=0)
    weighted_mean = summed_lums / summed_weights
    return weighted_mean

def downsample_mean_raw_live_stims(mean_RL_array, downsample_mult):
    downsampled_mean_RL = []
    for i in range(0,len(mean_RL_array), downsample_mult):
        if (i+downsample_mult-1) > len(mean_RL_array):
            this_chunk_mean = np.nanmean(mean_RL_array[i:len(mean_RL_array)])
        else:
            this_chunk_mean = np.nanmean(mean_RL_array[i:i+downsample_mult-1])
        downsampled_mean_RL.append(this_chunk_mean)
    return np.array(downsampled_mean_RL)

def phaseMeans_withDelay(delay_tb, normedPupils_array, calib_len_tb, allunique_lens_tb, octo_len_tb):
    allCalib = []
    allOcto = []
    allUnique = []
    # Split trials into calib, octo, and unique
    for i, uniqueStim in enumerate(normedPupils_array):
        thisUnique = []
        uniqueLen_tb = allunique_lens_tb[i]
        for normed_trial in uniqueStim:
            thisTrial_calib = normed_trial[delay_tb : delay_tb+calib_len_tb]
            allCalib.append(thisTrial_calib)
            thisTrial_unique = normed_trial[delay_tb+calib_len_tb+1 : delay_tb+calib_len_tb+1+uniqueLen_tb]
            thisUnique.append(thisTrial_unique)
            thisTrial_octo = normed_trial[delay_tb+calib_len_tb+1+uniqueLen_tb+1 : delay_tb+calib_len_tb+1+uniqueLen_tb+1+octo_len_tb]
            allOcto.append(thisTrial_octo)
        allUnique.append(thisUnique)
    calib_mean = np.nanmean(allCalib, axis=0)
    octo_mean = np.nanmean(allOcto, axis=0)
    unique_means = []
    for unique in allUnique:
        thisUnique_mean = np.nanmean(unique, axis=0)
        unique_means.append(thisUnique_mean)
    return calib_mean, octo_mean, unique_means

def leastSquares_pupilSize_lum(pupilSize_array, lum_array):
    # remove tb where pupil sizes are nans
    meanPupil_nonan = pupilSize_array[np.logical_not(np.isnan(pupilSize_array))]
    meanLum_nonan = lum_array[np.logical_not(np.isnan(pupilSize_array))]
    # remove tb where luminances are nans
    meanPupil_nonan = meanPupil_nonan[np.logical_not(np.isnan(meanLum_nonan))]
    meanLum_nonan = meanLum_nonan[np.logical_not(np.isnan(meanLum_nonan))]
    # calculate least squares regression line
    slope, intercept, rval, pval, stderr = stats.linregress(meanLum_nonan, meanPupil_nonan)
    return slope, intercept, rval, pval, stderr

def LumVsPupilSize_ScatterLinRegress(lum_array, pupilSize_array, phase_name, eyeAnalysis_name, pupilDelay_ms, save_folder):
    # make sure pupil size and world cam lum arrays are same size
    plotting_numTB = min(len(lum_array), len(pupilSize_array))
    lum_plot = lum_array[:plotting_numTB]
    pupil_plot = pupilSize_array[:plotting_numTB]
    # calculate least squares regression line
    slope, intercept, rval, pval, stderr = leastSquares_pupilSize_lum(pupilSize_array, lum_array)
    # figure path and title
    figPath = os.path.join(save_folder, '%s_meanLum-mean%s_delay%dms.png'%(phase_name, eyeAnalysis_name, pupilDelay_ms))
    figTitle = 'Mean luminance of world cam vs mean pupil size (%s) during %s, pupil delay = %dms'%(eyeAnalysis_name, phase_name, pupilDelay_ms)
    print('Plotting %s'%(figTitle))
    logging.INFO('Plotting %s'%(figTitle))
    # draw scatter plot
    plt.figure(figsize=(9, 9), dpi=200)
    plt.suptitle(figTitle, fontsize=12, y=0.98)
    plt.ylabel('Mean pupil size (percent change from median of full trial)')
    plt.xlabel('Mean luminance of world cam')
    plt.plot(lum_plot, pupil_plot, '.', label='original data')
    # draw regression line
    plt.plot(lum_plot, intercept+slope*lum_plot, 'r', label='fitted line, r-squared: %f'%(rval**2))
    plt.legend()
    plt.savefig(figPath)
    plt.close()
    return slope, intercept, rval, pval, stderr

def splitPupils_withDelay_plotScatterLinRegress(delay_tb, downsample_ms, lum_array, pupilSize_array, calib_len_tb, unique_lens_tb, octo_len_tb, eyeAnalysis_name, savePlotsFolder, saveDataFolder):
    # split normalized pupil size data into trial phases
    pupil_calib_mean, pupil_octo_mean, pupil_unique_means = phaseMeans_withDelay(delay_tb, pupilSize_array, calib_len_tb, unique_lens_tb, octo_len_tb)
    # save normalized, split and averaged pupil size data as intermediate files
    ## output path
    calib_output = saveDataFolder + os.sep + 'meanNormedPupilSize_calib_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    octo_output = saveDataFolder + os.sep + 'meanNormedPupilSize_octo_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique1_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u1_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique2_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u2_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique3_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u3_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique4_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u4_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique5_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u5_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique6_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u6_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    ## save file
    np.save(calib_output, pupil_calib_mean)
    np.save(octo_output, pupil_octo_mean)
    np.save(unique1_output, pupil_unique_means[0])
    np.save(unique2_output, pupil_unique_means[1])
    np.save(unique3_output, pupil_unique_means[2])
    np.save(unique4_output, pupil_unique_means[3])
    np.save(unique5_output, pupil_unique_means[4])
    np.save(unique6_output, pupil_unique_means[5])
    # recombine to create a "master" scatter plot with regression
    all_phases_pupil_sizes = np.concatenate((pupil_calib_mean, pupil_octo_mean, pupil_unique_means[0], pupil_unique_means[1], pupil_unique_means[2], pupil_unique_means[3], pupil_unique_means[4], pupil_unique_means[5]), axis=0)
    all_phases_mean_lum = np.concatenate((lum_array[0], lum_array[1], lum_array[2], lum_array[3], lum_array[4], lum_array[5], lum_array[6], lum_array[7]), axis=0)
    # plot scatter plots with regression line
    slope_allPhases, intercept_allPhases, rval_allPhases, pval_allPhases, stderr_allPhases = LumVsPupilSize_ScatterLinRegress(all_phases_mean_lum, all_phases_pupil_sizes, 'AllPhases', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    slope_calib, intercept_calib, rval_calib, pval_calib, stderr_calib = LumVsPupilSize_ScatterLinRegress(lum_array[0], pupil_calib_mean, 'calib', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    slope_octo, intercept_octo, rval_octo, pval_octo, stderr_octo = LumVsPupilSize_ScatterLinRegress(lum_array[1], pupil_octo_mean, 'octo', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    slope_u1, intercept_u1, rval_u1, pval_u1, stderr_u1 = LumVsPupilSize_ScatterLinRegress(lum_array[2], pupil_unique_means[0], 'unique01', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    slope_u2, intercept_u2, rval_u2, pval_u2, stderr_u2 = LumVsPupilSize_ScatterLinRegress(lum_array[3], pupil_unique_means[1], 'unique02', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    slope_u3, intercept_u3, rval_u3, pval_u3, stderr_u3 = LumVsPupilSize_ScatterLinRegress(lum_array[4], pupil_unique_means[2], 'unique03', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    slope_u4, intercept_u4, rval_u4, pval_u4, stderr_u4 = LumVsPupilSize_ScatterLinRegress(lum_array[5], pupil_unique_means[3], 'unique04', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    slope_u5, intercept_u5, rval_u5, pval_u5, stderr_u5 = LumVsPupilSize_ScatterLinRegress(lum_array[6], pupil_unique_means[4], 'unique05', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    slope_u6, intercept_u6, rval_u6, pval_u6, stderr_u6 = LumVsPupilSize_ScatterLinRegress(lum_array[7], pupil_unique_means[5], 'unique06', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    # return correlation coefficients
    return [[slope_allPhases, intercept_allPhases, rval_allPhases, pval_allPhases, stderr_allPhases], [slope_calib, intercept_calib, rval_calib, pval_calib, stderr_calib], [slope_octo, intercept_octo, rval_octo, pval_octo, stderr_octo], [slope_u1, intercept_u1, rval_u1, pval_u1, stderr_u1], [slope_u2, intercept_u2, rval_u2, pval_u2, stderr_u2], [slope_u3, intercept_u3, rval_u3, pval_u3, stderr_u3], [slope_u4, intercept_u4, rval_u4, pval_u4, stderr_u4], [slope_u5, intercept_u5, rval_u5, pval_u5, stderr_u5], [slope_u6, intercept_u6, rval_u6, pval_u6, stderr_u6]]

def drawFitScoresVsDelay_full(allPhases_fullLinRegress, num_delays, eyeAnalysis_name, downsample_ms, save_folder):
    rvals = []
    for delay in allPhases_fullLinRegress: 
        rvals.append(delay[2])
    rvals_plot = np.array(rvals)
    # optimal delay
    best_rval = min(rvals_plot)
    best_delay = rvals.index(best_rval)
    # figure path and title
    figPath = os.path.join(save_folder, 'AllPhases_rValsVsDelays_%s.png'%(eyeAnalysis_name))
    figTitle = 'Correlation coefficients (r val) vs delays in pupil response time \n All Phases, %s; Best delay = %dms (rval = %f)'%(eyeAnalysis_name, best_delay*downsample_ms, best_rval)
    print('Plotting %s'%(figTitle))
    logging.INFO('Plotting %s'%(figTitle))
    # draw fit scores vs delay
    plt.figure(dpi=150)
    plt.suptitle(figTitle, fontsize=12, y=0.98)
    plt.xlabel('Delay of pupil size data (ms)')
    plt.ylabel('Correlation coefficient')
    plt.xticks(np.arange(num_delays), np.arange(num_delays)*downsample_ms, rotation=50)
    plt.plot(rvals_plot, 'g')
    plt.tight_layout(rect=[0,0.03,1,0.93])
    # save figure and close
    plt.savefig(figPath)
    plt.close()

def drawFitScoresVsDelay_byPhase(linRegress_allPhases_list, num_delays, phases_strList, eyeAnalysis_name, downsample_ms, save_folder):
    for i, phase in enumerate(linRegress_allPhases_list):
        rvals = []
        for delay in phase: 
            rvals.append(delay[2])
        rvals_plot = np.array(rvals)
        # optimal delay
        best_rval = min(rvals_plot)
        best_delay = rvals.index(best_rval)
        # figure path and title
        figPath = os.path.join(save_folder, '%s_rValsVsDelays_%s.png'%(phases_strList[i], eyeAnalysis_name))
        figTitle = 'Correlation coefficients (r val) vs delays in pupil response time \n Phase: %s; %s; Best delay = %dms (rval = %f)'%(phases_strList[i], eyeAnalysis_name, best_delay*downsample_ms, best_rval)
        print('Plotting %s'%(figTitle))
        logging.INFO('Plotting %s'%(figTitle))
        # draw fit scores vs delay
        plt.figure(dpi=150)
        plt.suptitle(figTitle, fontsize=12, y=0.98)
        plt.xlabel('Delay of pupil size data (ms)')
        plt.ylabel('Correlation coefficient')
        plt.xticks(np.arange(num_delays), np.arange(num_delays)*downsample_ms, rotation=50)
        plt.plot(rvals_plot, 'g')
        plt.tight_layout(rect=[0,0.03,1,0.93])
        # save figure and close
        plt.savefig(figPath)
        plt.close()

def worldCam_MOIs_all_stim():
    # Moments of interest for each stimulus type, counted in 4ms timebuckets
    # For each start moment: [first timebucket when current phase appears and may be overlapping with previous phase, first timebucket when previous phase is entirely gone]
    # For each end moment: [last timebucket when current phase is showing with no overlap from next phase, last timebucket when current phase is showing and may be overlapping with next phase]
    all_avg_world_moments = {}
    # Stimulus 24.0
    all_avg_world_moments[24.0] = {'please center eyes': [0,0],
    'do not move your head': [35,51],
    'upper left dot appears': [1027,1047],
    'lower right dot appears': [1727,1727],
    'lower left dot appears': [2391,2391],
    'upper right dot appears': [3071,3071],
    'center dot appears': [3751,3751],
    'calibration end': [],
    'unique start': [4431,4431],
    'cat appears': [4615,4615],
    'front paws fully visible': [4720,4720],
    'front paws first contact with toy': {513:['2017-10'], 514:['2018-05']},
    'cat back paws bounce': {549:['2017-10'],547:['2018-05']},
    'unique end': [5922,5962],
    'octo start': [5923,5963],
    'fish turns': {645:['2017-10','2018-05']},
    'octopus fully decamouflaged': {766:['2018-05'], 767:['2017-10']},
    'camera zooms in on octopus': {860:['2017-10','2018-05']},
    'octopus inks': {882:['2017-10'],883:['2017-11','2018-03']},
    'camera clears ink cloud': {916:['2017-10'],920:['2018-05']},
    'octo end': [10803,10803]}
    # Stimulus 25.0
    all_avg_world_moments[25.0] = {'please center eyes': [0,0],
    'do not move your head': [35,51],
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {239:['2017-10'],238:['2018-05']},
    'upper right dot appears': {307:['2017-10'],306:['2018-05']},
    'center dot appears': {375:['2017-10'],374:['2018-05']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2018-03'],443:['2017-10','2017-11']},
    'fingers appear': {443:['2017-10'], 442:['2018-05']},
    'bird flies towards fingers': {462:['2018-05'],463:['2017-10']},
    'beak contacts food': {491:['2017-10'],492:['2018-05']},
    'wings at top of frame': {535:['2017-10','2018-05']},
    'bird flutters': {553:['2017-10'], 553:['2018-05']},
    'bird lands': {561:['2017-10'], 562:['2018-05']},
    'bird flies past fingers': {573:['2017-10','2018-05']},
    'unique end': {599:['2017-10'],600:['2017-11'],601:['2018-03']},
    'octo start': {599:['2017-10','2017-11','2018-03']},
    'fish turns': {649:['2017-10','2018-05']},
    'octopus fully decamouflaged': {770:['2017-10','2018-05']},
    'camera zooms in on octopus': {863:['2018-05'],864:['2017-10']},
    'octopus inks': {885:['2017-10','2018-03'],886:['2017-11']},
    'camera clears ink cloud': {919:['2017-10'],923:['2018-05']},
    'octo end': {989:['2017-10'],993:['2017-11'],994:['2018-03']}}
    # Stimulus 26.0
    all_avg_world_moments[26.0] = {'please center eyes': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {2:['2018-05'],3:['2017-10']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2017-10','2018-05']},
    'center dot appears': {374:['2017-10','2018-05']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2017-10','2018-03'],443:['2017-11']},
    'eyespots appear': {449:['2017-10', '2018-05']},
    'eyespots disappear, eyes darken': {487:['2017-10','2018-05']},
    'arms spread': {533:['2017-10'], 534:['2018-05']},
    'arms in, speckled mantle': {558:['2017-10'], 561:['2018-05']},
    'unique end': {663:['2017-10'],665:['2017-11','2018-03']},
    'octo start': {662:['2017-10'],663:['2018-03'],664:['2017-11']},
    'fish turns': {712:['2017-10','2018-05']},
    'octopus fully decamouflaged': {833:['2017-10','2018-05']},
    'camera zooms in on octopus': {927:['2017-10','2018-05']},
    'octopus inks': {949:['2017-10'],951:['2017-11','2018-03']},
    'camera clears ink cloud': {983:['2017-10'],987:['2018-05']},
    'octo end': {1054:['2017-10'],1059:['2017-11','2018-03']}}
    # Stimulus 27.0
    all_avg_world_moments[27.0] = {'please center eyes': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {3:['2017-10','2018-05']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2018-05'],307:['2017-10']},
    'center dot appears': {374:['2018-05'],375:['2017-10']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {443:['2017-10','2017-11','2018-03']},
    'cuttlefish appears': {443:['2017-10','2018-05']},
    'tentacles go ballistic': {530:['2017-10','2018-05']},
    'unique end': {606:['2017-10'],607:['2017-11','2018-03']},
    'octo start': {605:['2017-10','2017-11'],606:['2018-03']},
    'fish turns': {655:['2017-10','2018-05']},
    'octopus fully decamouflaged': {776:['2017-10','2018-05']},
    'camera zooms in on octopus': {869:['2018-05'],870:['2017-10']},
    'octopus inks': {892:['2017-10'],893:['2017-11','2018-03']},
    'camera clears ink cloud': {926:['2017-10'],929:['2018-05']},
    'octo end': {996:['2017-10'],1000:['2017-11','2018-03']}}
    # Stimulus 28.0
    all_avg_world_moments[28.0] = {'please center eyes': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {2:['2018-05'],3:['2017-10']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2017-10','2018-05']},
    'center dot appears': {374:['2018-05'],375:['2017-10']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2018-03'],443:['2017-10','2017-11']},
    'fish scatter': {456:['2017-10','2018-04','2018-10']},
    'center fish turns': {469:['2017-10'], 470:['2018-04'], 471:['2018-10']},
    'center fish swims to left': {494:['2018-04','2018-10'], 495:['2017-10']},
    'camera clears red ferns': {503:['2017-10'],506:['2018-04'],509:['2018-10']},
    'unique end': {662:['2017-10'],663:['2017-11'],666:['2018-03']},
    'octo start': {661:['2017-10'],662:['2018-03'],663:['2017-11']},
    'fish turns': {711:['2017-10','2018-05']},
    'octopus fully decamouflaged': {832:['2017-10'],834:['2018-05']},
    'camera zooms in on octopus': {927:['2017-10','2018-05']},
    'octopus inks': {948:['2017-10'],950:['2017-11','2018-03']},
    'camera clears ink cloud': {982:['2017-10'],986:['2018-05']},
    'octo end': {1054:['2017-10'],1056:['2017-11'],1059:['2018-03']}}
    # Stimulus 29.0
    all_avg_world_moments[29.0] = {'please center eyes': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {3:['2017-10','2018-05']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2017-10','2018-05']},
    'center dot appears': {374:['2017-10','2018-05']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2017-10'],443:['2017-11','2018-03']},
    'fish 1 appears': {457:['2017-10','2018-05']},
    'fish 1 turns': {495:['2017-10','2018-05']}, 
    'fish 2 appears': {538:['2017-10','2018-05']},
    'fish 2 touches mirror image': {646:['2017-10','2018-05']},
    'fish 2 disappears': {661:['2017-10','2018-05']}, 
    'fish 1 touches mirror image': {685:['2017-10','2018-05']},
    'fish 1 disappears': {702:['2017-10','2018-05']}, 
    'unique end': {717:['2017-10','2017-11'],718:['2018-03']},
    'octo start': {716:['2017-10','2018-03'],717:['2017-11']},
    'fish turns': {766:['2017-10','2018-03']},
    'octopus fully decamouflaged': {887:['2017-10','2018-05']},
    'camera zooms in on octopus': {981:['2017-10','2018-05']},
    'octopus inks': {1003:['2017-10'],1004:['2017-11','2018-03']},
    'camera clears ink cloud': {1037:['2017-10'],1041:['2018-05']},
    'octo end': {1108:['2017-10'],1110:['2017-11'],1112:['2018-03']}}
    return all_avg_world_moments

##########################################################
# BEGIN SCRIPT
##########################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", nargs='?', default="check_string_for_empty")
    parser.add_argument("--loc", nargs='?', default='laptop')
    args = parser.parse_args()
    ###################################
    # SOURCE DATA AND OUTPUT FILE LOCATIONS 
    ###################################
    root_folder, plots_folder, monthly_mean_lums_folders, display_latencies_folder, output_folders = load_data(args.loc)
    pupilSize_folder = output_folders[0]
    Rco_scatter_folder = output_folders[1]
    Rci_scatter_folder = output_folders[2]
    Lco_scatter_folder = output_folders[3]
    Lci_scatter_folder = output_folders[4]
    Rco_rvalVsDelay_folder = output_folders[5]
    Rci_rvalVsDelay_folder = output_folders[6]
    Lco_rvalVsDelay_folder = output_folders[7]
    Lci_rvalVsDelay_folder = output_folders[8]
    normedMeanPupilSizes_folder = output_folders[9]
    pupilSizeVsDelayLinRegress_folder = output_folders[10]
    logging.info('ROOT FOLDER: %s \n PLOTS FOLDER: %s \n MONTHLY MEAN RAW LIVE AND WORLD CAM STIMULI DATA FOLDER: %s' % (root_folder, plots_folder, monthly_mean_lums_folders))
    print('ROOT FOLDER: %s \n PLOTS FOLDER: %s \n MONTHLY MEAN RAW LIVE AND WORLD CAM STIMULI DATA FOLDER: %s' % (root_folder, plots_folder, monthly_mean_lums_folders))
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
    ########################################################
    # COLLECT TIMING INFO FOR CALIB, OCTO, AND UNIQUE PHASES
    ########################################################
    all_avg_world_moments = worldCam_MOIs_all_stim() # these are in 40ms timebuckets
    # convert start and end timebuckets into 4ms resolution timebuckets
    do_not_move_start = {key:{} for key in stim_vids}
    do_not_move_end = {key:{} for key in stim_vids}
    pulsing_dots_start = {key:{} for key in stim_vids}
    pulsing_dots_end = {key:{} for key in stim_vids}
    uniques_start = {key:{} for key in stim_vids}
    uniques_end = {key:{} for key in stim_vids}
    octo_start = {key:{} for key in stim_vids}
    octo_end = {key:{} for key in stim_vids}
    for stim_type in stim_vids:
        # world cam
        dnm_start_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'do not move your head', 'start', downsample_multiplier)
        dnm_end_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'upper left dot appears', 'end', downsample_multiplier)
        pd_start_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'upper left dot appears', 'start', downsample_multiplier)
        pd_end_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'calibration end', 'end', downsample_multiplier)
        u_start_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'unique start', 'start', downsample_multiplier)
        u_end_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'unique end', 'end', downsample_multiplier)
        o_start_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'octo start', 'start', downsample_multiplier)
        o_end_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'octo end', 'end', downsample_multiplier)
        do_not_move_start[stim_type]['world'] = dnm_start_world
        do_not_move_end[stim_type]['world'] = dnm_end_world
        pulsing_dots_start[stim_type]['world'] = pd_start_world
        pulsing_dots_end[stim_type]['world'] = pd_end_world
        uniques_start[stim_type]['world'] = u_start_world
        uniques_end[stim_type]['world'] = u_end_world
        octo_start[stim_type]['world'] = o_start_world
        octo_end[stim_type]['world'] = o_end_world
        # raw live
        dnm_start_raw = dnm_start_world - dnm_start_world
        dnm_end_raw = dnm_end_world - dnm_start_world
        pd_start_raw = pd_start_world - dnm_start_world
        pd_end_raw = pd_end_world - dnm_start_world
        u_start_raw = u_start_world - dnm_start_world
        u_end_raw = u_end_world - dnm_start_world
        o_start_raw = o_start_world - dnm_start_world
        o_end_raw = o_end_world - dnm_start_world
        do_not_move_start[stim_type]['raw'] = dnm_start_raw
        do_not_move_end[stim_type]['raw'] = dnm_end_raw
        pulsing_dots_start[stim_type]['raw'] = pd_start_raw
        pulsing_dots_end[stim_type]['raw'] = pd_end_raw
        uniques_start[stim_type]['raw'] = u_start_raw
        uniques_end[stim_type]['raw'] = u_end_raw
        octo_start[stim_type]['raw'] = o_start_raw
        octo_end[stim_type]['raw'] = o_end_raw
    ###############################################################
    # Load mean monthly raw live stim
    # split into Do Not Move, Pulsing Dots, Unique, and Octo phases
    # calculate weighted mean of each phase
    # save as binary files
    ###############################################################
    # raw live stim
    all_weighted_raw_doNotMove = []
    all_weights_raw_doNotMove = []
    all_weighted_raw_pulsingDots = []
    all_weights_raw_pulsingDots = []
    all_weighted_raw_unique = {key:[] for key in stim_vids}
    all_weights_raw_unique = {key:[] for key in stim_vids}
    all_weighted_raw_octo = []
    all_weights_raw_octo = []
    # collect length of each stimulus type in 4ms resolution
    supersampled_length_all_stims = {key:[] for key in stim_vids}
    # extract raw live stim
    for monthly_mean_folder in monthly_mean_lums_folders:
        raw_live_stim_files = glob.glob(root_folder + os.sep + monthly_mean_folder + os.sep + '*_meanRawLiveStim_*.npy')
        # raw live - extract and split into phases
        for raw_live_stim in raw_live_stim_files:
            stim_type = stim_name_to_float[os.path.basename(raw_live_stim).split('_')[1]]
            vid_count = float(os.path.basename(raw_live_stim).split('_')[-1][:-8])
            raw_live_array = np.load(raw_live_stim)
            supersampled_length_all_stims[stim_type].append(len(raw_live_array))
            this_file_weighted_doNotMove = []
            this_file_weights_doNotMove = []
            this_file_weighted_pulsingDots = []
            this_file_weights_pulsingDots = []
            this_file_weighted_unique = []
            this_file_weights_unique = []
            this_file_weighted_octo = []
            this_file_weights_octo = []
            for row in raw_live_array:
                timebucket = row[0]
                weight = row[1]
                mean_lum = row[2]
                this_tb_weighted_lum = weight*mean_lum
                if do_not_move_start[stim_type]['raw'] < timebucket < do_not_move_end[stim_type]['raw']:
                    this_file_weighted_doNotMove.append(this_tb_weighted_lum)
                    this_file_weights_doNotMove.append(weight)
                    continue
                elif pulsing_dots_start[stim_type]['raw'] < timebucket < pulsing_dots_end[stim_type]['raw']:
                    this_file_weighted_pulsingDots.append(this_tb_weighted_lum)
                    this_file_weights_pulsingDots.append(weight)
                elif uniques_start[stim_type]['raw'] < timebucket < uniques_end[stim_type]['raw']:
                    this_file_weighted_unique.append(this_tb_weighted_lum)
                    this_file_weights_unique.append(weight)
                elif octo_start[stim_type]['raw'] < timebucket < octo_end[stim_type]['raw']:
                    this_file_weighted_octo.append(this_tb_weighted_lum)
                    this_file_weights_octo.append(weight)
            all_weighted_raw_doNotMove.append(np.array(this_file_weighted_doNotMove))
            all_weights_raw_doNotMove.append(np.array(this_file_weights_doNotMove))
            all_weighted_raw_pulsingDots.append(np.array(this_file_weighted_pulsingDots))
            all_weights_raw_pulsingDots.append(np.array(this_file_weights_pulsingDots))
            all_weighted_raw_unique[stim_type].append(np.array(this_file_weighted_unique))
            all_weights_raw_unique[stim_type].append(np.array(this_file_weights_unique))
            all_weighted_raw_octo.append(np.array(this_file_weighted_octo))
            all_weights_raw_octo.append(np.array(this_file_weights_octo))
    # mean raw live luminance arrays
    mean_raw_live_doNotMove = calculate_weighted_mean_lum(all_weighted_raw_doNotMove, all_weights_raw_doNotMove)
    mean_raw_live_pulsingDots = calculate_weighted_mean_lum(all_weighted_raw_pulsingDots, all_weights_raw_pulsingDots)
    mean_raw_live_calib = np.concatenate([mean_raw_live_doNotMove, mean_raw_live_pulsingDots])
    mean_raw_live_uniques = {key:None for key in stim_vids}
    for stim in all_weighted_raw_unique:
        mean_raw_live_uniques[stim] = calculate_weighted_mean_lum(all_weighted_raw_unique[stim], all_weights_raw_unique[stim])
    mean_raw_live_octo = calculate_weighted_mean_lum(all_weighted_raw_octo, all_weights_raw_octo)
    ############################################################################
    # Downsample raw live stim data to match pupil data (4ms to 40ms resolution)
    ############################################################################
    downsampled_mean_RL_calib = downsample_mean_raw_live_stims(mean_raw_live_calib, downsample_multiplier)
    downsampled_mean_RL_octo = downsample_mean_raw_live_stims(mean_raw_live_octo, downsample_multiplier)
    downsampled_mean_RL_uniques = {key:None for key in stim_vids}
    for stim in mean_raw_live_uniques:
        downsampled_mean_RL_uniques[stim] = downsample_mean_raw_live_stims(mean_raw_live_uniques[stim], downsample_multiplier)
    downsampled_mean_RL_all_phases = [downsampled_mean_RL_calib, downsampled_mean_RL_octo, downsampled_mean_RL_uniques[24.0], downsampled_mean_RL_uniques[25.0], downsampled_mean_RL_uniques[26.0], downsampled_mean_RL_uniques[27.0], downsampled_mean_RL_uniques[28.0], downsampled_mean_RL_uniques[29.0]]
    calib_len = len(downsampled_mean_RL_calib)
    octo_len = len(downsampled_mean_RL_octo)
    unique_lens = [len(downsampled_mean_RL_uniques[24.0]), len(downsampled_mean_RL_uniques[25.0]), len(downsampled_mean_RL_uniques[26.0]), len(downsampled_mean_RL_uniques[27.0]), len(downsampled_mean_RL_uniques[28.0]), len(downsampled_mean_RL_uniques[29.0])]
    ###################################
    # ADJUST FOR WORLD CAM CAPTURING 2-3 FRAMES OF "PLEASE CENTER EYE" PHASE (display latency)
    # load display latencies and crop the beginning of the pupil arrays
    ###################################
    display_latencies_file = display_latencies_folder + os.sep + 'displayLatencies.npy'
    display_latencies = np.load(display_latencies_file)
    disp_latency_dict = dict(display_latencies)
    ###################################
    # BEGIN PUPIL DATA EXTRACTION 
    ###################################
    # prepare to sort pupil data by stimulus
    all_right_trials_contours_X = {key:[] for key in stim_vids}
    all_right_trials_contours_Y = {key:[] for key in stim_vids}
    all_right_trials_contours = {key:[] for key in stim_vids}
    all_right_trials_circles_X = {key:[] for key in stim_vids}
    all_right_trials_circles_Y = {key:[] for key in stim_vids}
    all_right_trials_circles = {key:[] for key in stim_vids}
    all_left_trials_contours_X = {key:[] for key in stim_vids}
    all_left_trials_contours_Y = {key:[] for key in stim_vids}
    all_left_trials_contours = {key:[] for key in stim_vids}
    all_left_trials_circles_X = {key:[] for key in stim_vids}
    all_left_trials_circles_Y = {key:[] for key in stim_vids}
    all_left_trials_circles = {key:[] for key in stim_vids}
    all_trials_position_X_data = [all_right_trials_contours_X, all_right_trials_circles_X, all_left_trials_contours_X, all_left_trials_circles_X]
    all_trials_position_Y_data = [all_right_trials_contours_Y, all_right_trials_circles_Y, all_left_trials_contours_Y, all_left_trials_circles_Y]
    all_trials_size_data = [all_right_trials_contours, all_right_trials_circles, all_left_trials_contours, all_left_trials_circles]
    activation_count = {}
    analysed_count = {}
    stimuli_tbucketed = {key:[] for key in stim_vids}
    # consolidate csv files from multiple days into one data structure
    day_folders = sorted(os.listdir(root_folder))
    # find pupil data on dropbox
    pupil_folders = fnmatch.filter(day_folders, 'SurprisingMinds_*')
    # first day was a debugging session, so skip it
    pupil_folders = pupil_folders[1:]
    ########################################################
    if args.a == 'check_string_for_empty':
        logging.info('Extracting all pupil data...')
        print('Extracting all pupil data...')
    elif args.a == 'debug':
        logging.warning('Extracting debugging subset of pupil data...')
        print('Extracting debugging subset of pupil data...')
        pupil_folders = pupil_folders[5:10]
    elif args.a == 'incomplete':
        # if currently still running pupil finding analysis...
        logging.warning('Currently still running pupil finding analysis...')
        print('Currently still running pupil finding analysis...')
        pupil_folders = pupil_folders[:-1]
    else:
        logging.warning('%s is not a valid optional input to this script! \nExtracting all pupil data...' % (args.a))
        print('%s is not a valid optional input to this script! \nExtracting all pupil data...' % (args.a))
    ########################################################
    # collect dates for which pupil extraction fails
    failed_days = []
    for day_folder in pupil_folders:
        # for each day...
        day_folder_path = os.path.join(root_folder, day_folder)
        analysis_folder = os.path.join(day_folder_path, "Analysis")
        csv_folder = os.path.join(analysis_folder, "csv")
        world_folder = os.path.join(analysis_folder, "world")
        #
        # Print/save number of users per day
        day_name = day_folder.split("_")[-1]
        try:
            ## EXTRACT PUPIL SIZE AND POSITION
            right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles, num_right_activations, num_good_right_trials = load_daily_pupils("right", csv_folder, downsampled_no_of_time_buckets, original_bucket_size_in_ms, downsampled_bucket_size_ms, disp_latency_dict)
            left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles, num_left_activations, num_good_left_trials = load_daily_pupils("left", csv_folder, downsampled_no_of_time_buckets, original_bucket_size_in_ms, downsampled_bucket_size_ms, disp_latency_dict)
            #
            analysed_count[day_name] = [num_good_right_trials, num_good_left_trials]
            activation_count[day_name] = [num_right_activations, num_left_activations]
            print("On {day}, exhibit was activated {right_count} times (right) and {left_count} times (left), with {right_good_count} good right trials and {left_good_count} good left trials".format(day=day_name, right_count=num_right_activations, left_count=num_left_activations, right_good_count=num_good_right_trials, left_good_count=num_good_left_trials))
            logging.INFO("On {day}, exhibit was activated {right_count} times (right) and {left_count} times (left), with {right_good_count} good right trials and {left_good_count} good left trials".format(day=day_name, right_count=num_right_activations, left_count=num_left_activations, right_good_count=num_good_right_trials, left_good_count=num_good_left_trials))
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
            #
            stim_sorted_data_right = [R_contours_X, R_contours_Y, R_contours, R_circles_X, R_circles_Y, R_circles]
            stim_sorted_data_left = [L_contours_X, L_contours_Y, L_contours, L_circles_X, L_circles_Y, L_circles]
            stim_sorted_data_all = [stim_sorted_data_right, stim_sorted_data_left]
            #
            extracted_data_right = [right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles]
            extracted_data_left = [left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles]
            extracted_data_all = [extracted_data_right, extracted_data_left]
            #
            for side in range(len(extracted_data_all)):
                for dataset in range(len(extracted_data_all[side])):
                    for trial in extracted_data_all[side][dataset]:
                        stim_num = trial[-1]
                        if stim_num in stim_sorted_data_all[side][dataset].keys():
                            stim_sorted_data_all[side][dataset][stim_num].append(trial[:-1])
            #
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
            #
            # append position data to global data structure
            for i in range(len(all_position_X_data)):
                for stimulus in all_position_X_data[i]:
                    for index in range(len(all_position_X_data[i][stimulus])):
                        all_trials_position_X_data[i][stimulus].append(all_position_X_data[i][stimulus][index])
            for i in range(len(all_position_Y_data)):
                for stimulus in all_position_Y_data[i]:
                    for index in range(len(all_position_Y_data[i][stimulus])):
                        all_trials_position_Y_data[i][stimulus].append(all_position_Y_data[i][stimulus][index])
            # append size data to global data structure
            for i in range(len(all_size_data)):
                for stimulus in all_size_data[i]:
                    for index in range(len(all_size_data[i][stimulus])):
                        all_trials_size_data[i][stimulus].append(all_size_data[i][stimulus][index])
            print("Day {day} succeeded!".format(day=day_name))
            logging.INFO("Day {day} succeeded!".format(day=day_name))
        except Exception:
            failed_days.append(day_name)
            print("Day {day} failed!".format(day=day_name))
            logging.WARNING("Day {day} failed!".format(day=day_name))
    ###################################
    # Normalize pupil size data 
    ###################################
    # Right Contours
    R_contours_allStim = [all_trials_size_data[0][24.0], all_trials_size_data[0][25.0], all_trials_size_data[0][26.0], all_trials_size_data[0][27.0], all_trials_size_data[0][28.0], all_trials_size_data[0][29.0]]
    Rco_normed = normPupilSizeData(R_contours_allStim, 'right contours')
    # Right Circles
    R_circles_allStim = [all_trials_size_data[1][24.0], all_trials_size_data[1][25.0], all_trials_size_data[1][26.0], all_trials_size_data[1][27.0], all_trials_size_data[1][28.0], all_trials_size_data[1][29.0]]
    Rci_normed = normPupilSizeData(R_circles_allStim, 'right circles')
    # Left Contours
    L_contours_allStim = [all_trials_size_data[2][24.0], all_trials_size_data[2][25.0], all_trials_size_data[2][26.0], all_trials_size_data[2][27.0], all_trials_size_data[2][28.0], all_trials_size_data[2][29.0]]
    Lco_normed = normPupilSizeData(L_contours_allStim, 'left contours')
    # Left Circles
    L_circles_allStim = [all_trials_size_data[3][24.0], all_trials_size_data[3][25.0], all_trials_size_data[3][26.0], all_trials_size_data[3][27.0], all_trials_size_data[3][28.0], all_trials_size_data[3][29.0]]
    Lci_normed = normPupilSizeData(L_circles_allStim, 'left circles')
    ###################################
    # split normed and cropped pupil size arrays based on different delays of pupil reaction
    # save split normed and cropped pupil size arrays as binary files
    # create scatter plot of pupil size against world cam luminance values
    # include least squares regression line in scatter plot
    ###################################
    delays = 25
    # by phase
    Rco_calibLinRegress_allDelays = []
    Rci_calibLinRegress_allDelays = []
    Lco_calibLinRegress_allDelays = []
    Lci_calibLinRegress_allDelays = []
    Rco_octoLinRegress_allDelays = []
    Rci_octoLinRegress_allDelays = []
    Lco_octoLinRegress_allDelays = []
    Lci_octoLinRegress_allDelays = []
    Rco_u1LinRegress_allDelays = []
    Rci_u1LinRegress_allDelays = []
    Lco_u1LinRegress_allDelays = []
    Lci_u1LinRegress_allDelays = []
    Rco_u2LinRegress_allDelays = []
    Rci_u2LinRegress_allDelays = []
    Lco_u2LinRegress_allDelays = []
    Lci_u2LinRegress_allDelays = []
    Rco_u3LinRegress_allDelays = []
    Rci_u3LinRegress_allDelays = []
    Lco_u3LinRegress_allDelays = []
    Lci_u3LinRegress_allDelays = []
    Rco_u4LinRegress_allDelays = []
    Rci_u4LinRegress_allDelays = []
    Lco_u4LinRegress_allDelays = []
    Lci_u4LinRegress_allDelays = []
    Rco_u5LinRegress_allDelays = []
    Rci_u5LinRegress_allDelays = []
    Lco_u5LinRegress_allDelays = []
    Lci_u5LinRegress_allDelays = []
    Rco_u6LinRegress_allDelays = []
    Rci_u6LinRegress_allDelays = []
    Lco_u6LinRegress_allDelays = []
    Lci_u6LinRegress_allDelays = []
    # all phases
    Rco_allPhasesConcatLinRegress_allDelays = []
    Rci_allPhasesConcatLinRegress_allDelays = []
    Lco_allPhasesConcatLinRegress_allDelays = []
    Lci_allPhasesConcatLinRegress_allDelays = []
    for delay in range(delays):
        print('Delay: %d timebucket(s)'%(delay))
        logging.INFO('Delay: %d timebucket(s)'%(delay))
        # Right Contours
        linRegress_Rco = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, downsampled_mean_RL_all_phases, Rco_normed, calib_len, unique_lens, octo_len, 'RightContours', Rco_scatter_folder, normedMeanPupilSizes_folder)
        Rco_allPhasesConcatLinRegress_allDelays.append(linRegress_Rco[0])
        Rco_calibLinRegress_allDelays.append(linRegress_Rco[1])
        Rco_octoLinRegress_allDelays.append(linRegress_Rco[2])
        Rco_u1LinRegress_allDelays.append(linRegress_Rco[3])
        Rco_u2LinRegress_allDelays.append(linRegress_Rco[4])
        Rco_u3LinRegress_allDelays.append(linRegress_Rco[5])
        Rco_u4LinRegress_allDelays.append(linRegress_Rco[6])
        Rco_u5LinRegress_allDelays.append(linRegress_Rco[7])
        Rco_u6LinRegress_allDelays.append(linRegress_Rco[8])
        # Right Circles
        linRegress_Rci = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, downsampled_mean_RL_all_phases, Rci_normed, calib_len, unique_lens, octo_len, 'RightCircles', Rci_scatter_folder, normedMeanPupilSizes_folder)
        Rci_allPhasesConcatLinRegress_allDelays.append(linRegress_Rci[0])
        Rci_calibLinRegress_allDelays.append(linRegress_Rci[1])
        Rci_octoLinRegress_allDelays.append(linRegress_Rci[2])
        Rci_u1LinRegress_allDelays.append(linRegress_Rci[3])
        Rci_u2LinRegress_allDelays.append(linRegress_Rci[4])
        Rci_u3LinRegress_allDelays.append(linRegress_Rci[5])
        Rci_u4LinRegress_allDelays.append(linRegress_Rci[6])
        Rci_u5LinRegress_allDelays.append(linRegress_Rci[7])
        Rci_u6LinRegress_allDelays.append(linRegress_Rci[8])
        # Left Contours
        linRegress_Lco = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, downsampled_mean_RL_all_phases, Lco_normed, calib_len, unique_lens, octo_len, 'LeftContours', Lco_scatter_folder, normedMeanPupilSizes_folder)
        Lco_allPhasesConcatLinRegress_allDelays.append(linRegress_Lco[0])
        Lco_calibLinRegress_allDelays.append(linRegress_Lco[1])
        Lco_octoLinRegress_allDelays.append(linRegress_Lco[2])
        Lco_u1LinRegress_allDelays.append(linRegress_Lco[3])
        Lco_u2LinRegress_allDelays.append(linRegress_Lco[4])
        Lco_u3LinRegress_allDelays.append(linRegress_Lco[5])
        Lco_u4LinRegress_allDelays.append(linRegress_Lco[6])
        Lco_u5LinRegress_allDelays.append(linRegress_Lco[7])
        Lco_u6LinRegress_allDelays.append(linRegress_Lco[8])
        # Left Circles
        linRegress_Lci = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, downsampled_mean_RL_all_phases, Lci_normed, calib_len, unique_lens, octo_len, 'LeftCircles', Lci_scatter_folder, normedMeanPupilSizes_folder)
        Lci_allPhasesConcatLinRegress_allDelays.append(linRegress_Lci[0])
        Lci_calibLinRegress_allDelays.append(linRegress_Lci[1])
        Lci_octoLinRegress_allDelays.append(linRegress_Lci[2])
        Lci_u1LinRegress_allDelays.append(linRegress_Lci[3])
        Lci_u2LinRegress_allDelays.append(linRegress_Lci[4])
        Lci_u3LinRegress_allDelays.append(linRegress_Lci[5])
        Lci_u4LinRegress_allDelays.append(linRegress_Lci[6])
        Lci_u5LinRegress_allDelays.append(linRegress_Lci[7])
        Lci_u6LinRegress_allDelays.append(linRegress_Lci[8])
    ###################################
    # plot fit scores (rvals) vs delay
    ###################################
    # all phases combined
    drawFitScoresVsDelay_full(Rco_allPhasesConcatLinRegress_allDelays, delays, 'RightContours', downsampled_bucket_size_ms, Rco_rvalVsDelay_folder) 
    drawFitScoresVsDelay_full(Rci_allPhasesConcatLinRegress_allDelays, delays, 'RightCircles', downsampled_bucket_size_ms, Rci_rvalVsDelay_folder) 
    drawFitScoresVsDelay_full(Lco_allPhasesConcatLinRegress_allDelays, delays, 'LeftContours', downsampled_bucket_size_ms, Lco_rvalVsDelay_folder) 
    drawFitScoresVsDelay_full(Lci_allPhasesConcatLinRegress_allDelays, delays, 'LeftCircles', downsampled_bucket_size_ms, Lci_rvalVsDelay_folder) 
    # by phase
    allRcoPhases = [Rco_calibLinRegress_allDelays, Rco_octoLinRegress_allDelays, Rco_u1LinRegress_allDelays, Rco_u2LinRegress_allDelays, Rco_u3LinRegress_allDelays, Rco_u4LinRegress_allDelays, Rco_u5LinRegress_allDelays, Rco_u6LinRegress_allDelays]
    drawFitScoresVsDelay_byPhase(allRcoPhases, delays, phase_names, 'RightContours', downsampled_bucket_size_ms, Rco_rvalVsDelay_folder)
    allRciPhases = [Rci_calibLinRegress_allDelays, Rci_octoLinRegress_allDelays, Rci_u1LinRegress_allDelays, Rci_u2LinRegress_allDelays, Rci_u3LinRegress_allDelays, Rci_u4LinRegress_allDelays, Rci_u5LinRegress_allDelays, Rci_u6LinRegress_allDelays]
    drawFitScoresVsDelay_byPhase(allRciPhases, delays, phase_names, 'RightCircles', downsampled_bucket_size_ms, Rci_rvalVsDelay_folder)
    allLcoPhases = [Lco_calibLinRegress_allDelays, Lco_octoLinRegress_allDelays, Lco_u1LinRegress_allDelays, Lco_u2LinRegress_allDelays, Lco_u3LinRegress_allDelays, Lco_u4LinRegress_allDelays, Lco_u5LinRegress_allDelays, Lco_u6LinRegress_allDelays]
    drawFitScoresVsDelay_byPhase(allLcoPhases, delays, phase_names, 'LeftContours', downsampled_bucket_size_ms, Lco_rvalVsDelay_folder)
    allLciPhases = [Lci_calibLinRegress_allDelays, Lci_octoLinRegress_allDelays, Lci_u1LinRegress_allDelays, Lci_u2LinRegress_allDelays, Lci_u3LinRegress_allDelays, Lci_u4LinRegress_allDelays, Lci_u5LinRegress_allDelays, Lci_u6LinRegress_allDelays]
    drawFitScoresVsDelay_byPhase(allLciPhases, delays, phase_names, 'LeftCircles', downsampled_bucket_size_ms, Lci_rvalVsDelay_folder)
    ###################################
    # save linear regression parameters as binary files
    ###################################
    # output path
    Rco_allPhasesConcat_linRegress_output = pupilSizeVsDelayLinRegress_folder + os.sep + 'pupilSizeVsDelayLinRegressParams_RightContours_allPhasesConcat_%dTBDelays.npy'%(delays)
    Rci_allPhasesConcat_linRegress_output = pupilSizeVsDelayLinRegress_folder + os.sep + 'pupilSizeVsDelayLinRegressParams_RightCircles_allPhasesConcat_%dTBDelays.npy'%(delays)
    Lco_allPhasesConcat_linRegress_output = pupilSizeVsDelayLinRegress_folder + os.sep + 'pupilSizeVsDelayLinRegressParams_LeftContours_allPhasesConcat_%dTBDelays.npy'%(delays)
    Lci_allPhasesConcat_linRegress_output = pupilSizeVsDelayLinRegress_folder + os.sep + 'pupilSizeVsDelayLinRegressParams_LeftCircles_allPhasesConcat_%dTBDelays.npy'%(delays)
    # save file
    np.save(Rco_allPhasesConcat_linRegress_output, Rco_allPhasesConcatLinRegress_allDelays)
    np.save(Rci_allPhasesConcat_linRegress_output, Rci_allPhasesConcatLinRegress_allDelays)
    np.save(Lco_allPhasesConcat_linRegress_output, Lco_allPhasesConcatLinRegress_allDelays)
    np.save(Lci_allPhasesConcat_linRegress_output, Lci_allPhasesConcatLinRegress_allDelays)

# FIN