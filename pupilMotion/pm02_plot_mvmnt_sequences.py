# -*- coding: utf-8 -*-
"""
Project: "Surprising Minds" at Sea Life Brighton, by Danbee Kim, Kerry Perkins, Clive Ramble, Hazel Garnade, Goncalo Lopes, Dario Quinones, Reanna Campbell-Russo, Robb Barrett, Martin Stopps, The EveryMind Team, and Adam Kampff. 
Analysis: Measure speed of pupil

Loads .npz files with movement/motion data that has been chunked into calib, unique, and octo sequences
Plot pupil movement and motion (absolute value of movement).

@author: Adam R Kampff and Danbee Kim
"""

import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import logging

###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# FUNCTIONS
###################################

##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) dataset_dir (parent folder with intermediate files)
# 2) plots_dir (parent folder for all plots output by this script)
##########################################################
def load_data():
    dataset_dir = r'D:\data\SurprisingMinds\intermediates'
    plots_dir = r'D:\data\SurprisingMinds\plots'
    return dataset_dir, plots_dir
##########################################################

##########################################################
# BEGIN SCRIPT
##########################################################
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
logging.basicConfig(filename="pm02PlotMvmntSeq_" + todays_datetime + ".log", filemode='w', level=logging.INFO)
###################################
# SOURCE DATA AND OUTPUT FILE LOCATIONS 
###################################
data_folder, plots_folder = load_data()
# set up input folders
calib_mvmnt_folder = os.path.join(data_folder, 'calib_movement')
octo_mvmnt_folder = os.path.join(data_folder, 'octo_movement')
unique_mvmnt_folder = os.path.join(data_folder, 'unique_movement')
# set up plot output folders
pupil_motion_plots = os.path.join(plots_folder, "pupil_motion")
# Create plots folder (and sub-folders) if it (they) does (do) not exist
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
if not os.path.exists(pupil_motion_plots):
    os.makedirs(pupil_motion_plots)

logging.info('PUPIL DATA FOLDER: %s \n CALIB DATA FOLDER: %s \n UNIQUE DATA FOLDER: %s \n OCTO DATA FOLDER: %s \n PUPIL PLOTS FOLDER: %s' % (pupil_data_downsampled, calib_mvmnt_folder, unique_mvmnt_folder, octo_mvmnt_folder, pupil_motion_plots))
print('PUPIL DATA FOLDER: %s \n CALIB DATA FOLDER: %s \n UNIQUE DATA FOLDER: %s \n OCTO DATA FOLDER: %s \n PUPIL PLOTS FOLDER: %s' % (pupil_data_downsampled, calib_mvmnt_folder, unique_mvmnt_folder, octo_mvmnt_folder, pupil_motion_plots))
###################################
# PARAMETERS
###################################
downsampled_bucket_size_ms = 40
smoothing_window = 25 # in time buckets, must be odd! for savgol_filter
fig_size = 200 # dpi
plot_movement = False 
plot_avg_motion = True 
plot_peaks = False
#########################################################
# LOAD MOVEMENT DATA
#########################################################
calib_files = glob.glob(calib_mvmnt_folder + os.sep + '*.npz')
unique_files = glob.glob(unique_mvmnt_folder + os.sep + '*.npz')
octo_files = glob.glob(octo_mvmnt_folder + os.sep + '*.npz')

for octo_file in octo_files:
    file_info = os.path.basename(octo_file).split('_')
    side = file_info[0]
    c_axis = file_info[1]
    seq = file_info[2]
    movement_type = file_info[3]
    if side == 'Left':
        if c_axis == 'contoursX':
            if seq == 'octo':
                if movement_type == 'mvmnt':
                    pupil_data = np.load(octo_file)
                    pupil_movement = pupil_data['arr_0']

seq_trial_count = len(pupil_movement)
abs_val_pupil_movement = np.abs(pupil_movement)
mean_pupil_movement = np.nanmean(abs_val_pupil_movement, axis=0)
std_pupil_movement = np.nanstd(abs_val_pupil_movement, axis=0)
sem_pupil_movement = std_pupil_movement/np.sqrt(seq_trial_count)
upper_bound = mean_pupil_movement+sem_pupil_movement
lower_bound = mean_pupil_movement-sem_pupil_movement
figure_name = 'Octo_motion_update' + todays_datetime + '.png'
figure_path = os.path.join(plots_folder, figure_name)
figure_title = 'Avg motion during sequence {s}, with 95 CI, N={n}'.format(s=seq, n=seq_trial_count)
plt.figure(figsize=(18, 6), dpi=fsize)
plt.suptitle(figure_title, fontsize=12, y=0.98)
plt.grid(b=True, which='major', linestyle='--')
x_frames = range(len(mean_pupil_movement))
plot_xticks = np.arange(0, len(mean_pupil_movement), step=25)
plt.xticks(plot_xticks, ['%.1f'%(x*0.04) for x in plot_xticks])
plt.plot(mean_pupil_movement, color='orange')
plt.fill_between(x_frames, upper_bound, lower_bound, color='yellow')
plt.show()

    pupil_data = np.load(daily_pupil_data, allow_pickle=True)
    this_day_x_pos = pupil_data['all_pos_x']
    this_day_y_pos = pupil_data['all_pos_y']
    
###################################
# LOAD STIMULUS LUMINANCE DATA
###################################
luminance_info_path = glob.glob(lum_processed + os.sep + '*.npz')
for lum_info_path in luminance_info_path:
    if os.path.basename(lum_info_path) == 'processed_lum.npz':
        lum = np.load(lum_info_path)
    elif os.path.basename(lum_info_path) == 'processed_lum_avg.npz':
        lum_avg = np.load(lum_info_path)
    else:
        lum_peaks = np.load(lum_info_path)

luminances_avg = {}
luminances_peaks = {}
luminances = {}
for vid_key in stim_vids:
    luminances_avg[vid_key] = lum_avg[stim_float_to_name[vid_key]]
    luminances_peaks[vid_key] = lum_peaks[stim_float_to_name[vid_key]]
    luminances[vid_key] = lum[stim_float_to_name[vid_key]]

###################################
# PLOT MOVEMENT
###################################
plotting_peaks_window = 40 # MAKE SURE THIS == peaks_window!!
cType_names = ['Contours', 'Circles']
all_movement_right_plot = [(all_right_contours_movement_X, all_right_contours_movement_Y), (all_right_circles_movement_X, all_right_circles_movement_Y)]
all_movement_left_plot = [(all_left_contours_movement_X, all_left_contours_movement_Y), (all_left_circles_movement_X, all_left_circles_movement_Y)]
all_movements_plot = [all_movement_right_plot, all_movement_left_plot]
# plot movement traces
if plot_movement:
    for side in range(len(all_movements_plot)):
        for c_type in range(len(all_movements_plot[side])):
            for stimuli in all_movements_plot[side][c_type][0]:
                plot_type_name = side_names[side] + cType_names[c_type]
                stim_name = stim_float_to_name[stimuli]
                plot_type_X = all_movements_plot[side][c_type][0][stimuli]
                plot_N_X = len(plot_type_X)
                plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
                plot_N_Y = len(plot_type_Y)
                plot_luminance = luminances_avg[stimuli]
                plot_luminance_peaks = luminances_peaks[stimuli]
                # fig name and path
                figure_name = 'MovementTraces_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
                figure_path = os.path.join(pupil_motion_plots, figure_name)
                figure_title = "Pupil movement of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPlotted on " + todays_datetime
                # draw fig
                plt.figure(figsize=(14, 14), dpi=fig_size)
                plt.suptitle(figure_title, fontsize=12, y=0.98)
                # x-axis
                plt.subplot(3,1,1)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_X:
                    plt.plot(trial, linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
                plt.xlim(-10,1250)
                plt.ylim(-80,80)
                # y-axis
                plt.subplot(3,1,2)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_Y:
                    plt.plot(trial, linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
                plt.xlim(-10,1250)
                plt.ylim(-80,80)
                # luminance
                plt.subplot(3,1,3)
                plt.ylabel('Percent change in luminance', fontsize=11)
                plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
                plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
                plt.grid(b=True, which='major', linestyle='--')
                plt.plot(plot_luminance, linewidth=0.75, color=[1.0, 0.13, 0.4, 1])
                for peak in plot_luminance_peaks:
                    plt.plot(peak, plot_luminance[peak], 'x')
                    plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-1,7)
                # save and display
                plt.subplots_adjust(hspace=0.5)
                plt.savefig(figure_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

###################################
# PLOT MOTION
###################################
all_avg_motion_right_plot = [(all_right_contours_X_avg_motion, all_right_contours_Y_avg_motion), (all_right_circles_X_avg_motion, all_right_circles_Y_avg_motion)]
all_avg_motion_left_plot = [(all_left_contours_X_avg_motion, all_left_contours_Y_avg_motion), (all_left_circles_X_avg_motion, all_left_circles_Y_avg_motion)]
all_avg_motion_plot = [all_avg_motion_right_plot, all_avg_motion_left_plot]
all_avg_motion_right_peaks_plot = [(all_RcontoursX_avg_motion_peaks, all_RcontoursY_avg_motion_peaks), (all_RcirclesX_avg_motion_peaks, all_RcirclesY_avg_motion_peaks)]
all_avg_motion_left_peaks_plot = [(all_LcontoursX_avg_motion_peaks, all_LcontoursY_avg_motion_peaks), (all_LcirclesX_avg_motion_peaks, all_LcirclesY_avg_motion_peaks)]
all_avg_motion_peaks_plot = [all_avg_motion_right_peaks_plot, all_avg_motion_left_peaks_plot]
# plot MOTION traces (abs val of movement traces)
if plot_motion:
    for side in range(len(all_movements_plot)):
        for c_type in range(len(all_movements_plot[side])):
            for stimuli in all_movements_plot[side][c_type][0]:
                plot_type_name = side_names[side] + cType_names[c_type]
                stim_name = stim_float_to_name[stimuli]
                plot_type_X = all_movements_plot[side][c_type][0][stimuli]
                plot_type_X_avg = all_avg_motion_plot[side][c_type][0][stimuli]
                plot_type_X_avg_peaks = all_avg_motion_peaks_plot[side][c_type][0][stimuli]
                plot_N_X = len(plot_type_X)
                plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
                plot_type_Y_avg = all_avg_motion_plot[side][c_type][1][stimuli]
                plot_type_Y_avg_peaks = all_avg_motion_peaks_plot[side][c_type][1][stimuli]
                plot_N_Y = len(plot_type_Y)
                plot_luminance = luminances_avg[stimuli]
                plot_luminance_peaks = luminances_peaks[stimuli]
                # fig name and path
                figure_name = 'MotionTraces-AvgMotionPeaks' + str(plotting_peaks_window) + '_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
                figure_path = os.path.join(pupil_motion_plots, figure_name)
                figure_title = "Pupil motion of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPeak finding window: " + str(plotting_peaks_window) + "\nPlotted on " + todays_datetime
                # draw fig
                plt.figure(figsize=(14, 14), dpi=fig_size)
                plt.suptitle(figure_title, fontsize=12, y=0.98)
                # x-axis
                plt.subplot(3,1,1)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_X:
                    plt.plot(abs(trial), linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
                plt.plot(plot_type_X_avg, linewidth=1, color=[0.4, 1.0, 0.27, 1])
                for peak in plot_type_X_avg_peaks:
                    if peak<1250:
                        plt.plot(peak, plot_type_X_avg[peak], 'x')
                        plt.text(peak-15, plot_type_X_avg[peak]+5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-5,40)
                # y-axis
                plt.subplot(3,1,2)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_Y:
                    plt.plot(abs(trial), linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
                plt.plot(plot_type_Y_avg, linewidth=1, color=[1.0, 1.0, 0.25, 1])
                for peak in plot_type_Y_avg_peaks:
                    if peak<1250:
                        plt.plot(peak, plot_type_Y_avg[peak], 'x')
                        plt.text(peak-15, plot_type_Y_avg[peak]+5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-5,40)
                # luminance
                plt.subplot(3,1,3)
                plt.ylabel('Percent change in luminance', fontsize=11)
                plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
                #plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
                plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled', fontsize=10, color='grey', style='italic')
                plt.grid(b=True, which='major', linestyle='--')
                plt.plot(plot_luminance, linewidth=1, color=[1.0, 0.13, 0.4, 1])
                for peak in plot_luminance_peaks:
                    plt.plot(peak, plot_luminance[peak], 'x')
                    plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-1,7)
                # save and display
                plt.subplots_adjust(hspace=0.5)
                plt.savefig(figure_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

###################################
# PLOT PEAKS
###################################
all_peaks_right_plot = [(all_right_contours_X_peaks, all_right_contours_Y_peaks), (all_right_circles_X_peaks, all_right_circles_Y_peaks)]
all_peaks_left_plot = [(all_left_contours_X_peaks, all_left_contours_Y_peaks), (all_left_circles_X_peaks, all_left_circles_Y_peaks)]
all_peaks_plot = [all_peaks_right_plot, all_peaks_left_plot]
if plot_peaks:
    for side in range(len(all_movements_plot)):
        for c_type in range(len(all_movements_plot[side])):
            for stimuli in all_movements_plot[side][c_type][0]:
                plot_type_name = side_names[side] + cType_names[c_type]
                stim_name = stim_float_to_name[stimuli]
                plot_type_X = all_movements_plot[side][c_type][0][stimuli]
                plot_type_X_peaks = all_peaks_plot[side][c_type][0][stimuli]
                plot_N_X = len(plot_type_X)
                plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
                plot_type_Y_peaks = all_peaks_plot[side][c_type][1][stimuli]
                plot_N_Y = len(plot_type_Y)
                plot_luminance = luminances_avg[stimuli]
                plot_luminance_peaks = luminances_peaks[stimuli]
                # fig name and path
                figure_name = 'MotionTraces-Saccades' + str(plotting_peaks_window) + '_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
                figure_path = os.path.join(pupil_motion_plots, figure_name)
                figure_title = "Pupil motion of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPeaks plotted at height of pixel movement threshold, peak finding window: " + str(plotting_peaks_window) + "\nPlotted on " + todays_datetime
                # begin drawing fig
                plt.figure(figsize=(14, 14), dpi=fig_size)
                plt.suptitle(figure_title, fontsize=12, y=0.98)
                # x-axis
                plt.subplot(3,1,1)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_X:
                    plt.plot(abs(trial), linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
                for threshold in plot_type_X_peaks.keys():
                    for key in plot_type_X_peaks[threshold].keys():
                        if key<1250:
                            plt.plot(key, threshold, '1', color=[0.4, 1.0, 0.27, 1.0])
                plt.xlim(-10,1250)
                plt.ylim(-5,60)
                # y-axis
                plt.subplot(3,1,2)
                plt.ylabel('Change in pixels', fontsize=11)
                plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
                plt.minorticks_on()
                plt.grid(b=True, which='major', linestyle='--')
                for trial in plot_type_Y:
                    plt.plot(abs(trial), linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
                for threshold in plot_type_Y_peaks.keys():
                    for key in plot_type_Y_peaks[threshold].keys():
                        if key<1250:
                            plt.plot(key, threshold, '1', color=[1.0, 1.0, 0.25, 1.0])
                plt.xlim(-10,1250)
                plt.ylim(-5,60)
                # luminance
                plt.subplot(3,1,3)
                plt.ylabel('Percent change in luminance (from baseline)', fontsize=11)
                plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
                plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
                plt.grid(b=True, which='major', linestyle='--')
                plt.plot(plot_luminance, linewidth=1, color=[1.0, 0.13, 0.4, 1])
                for peak in plot_luminance_peaks:
                    plt.plot(peak, plot_luminance[peak], 'x')
                    plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                plt.xlim(-10,1250)
                plt.ylim(-1,7)
                # save and display
                plt.subplots_adjust(hspace=0.5)
                plt.savefig(figure_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

#FIN