### ------------------------------------------------------------------------- ###
# DON'T NEED THIS SCRIPT ANYMORE????
### Create CSV files with average luminance per frame of stimulus vids
### use world camera vids for timing, stretch and interpolate raw stim vid lum values
### output as data files categorized by calibration, octopus, and unique sequences.
### ------------------------------------------------------------------------- ###
import os
import glob
import datetime
import csv
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
from scipy import interpolate

###################################
# FUNCTIONS
###################################
def load_avg_world_unraveled(avg_world_folder_path):
    # List all world camera csv files
    stim_files = glob.glob(avg_world_folder_path + os.sep + "*Avg-World-Vid-tbuckets.csv")
    world_vids_tbucketed = {}
    for stim_file in stim_files:
        stim_filename = stim_file.split(os.sep)[-1]
        stim_type = stim_filename.split('_')[1]
        stim_number = stim_name_to_float[stim_type]
        world_vids_tbucketed[stim_number] = {}
        extracted_rows = []
        print("Extracting from {name}".format(name=stim_filename))
        with open(stim_file) as f:
            csvReader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in csvReader:
                extracted_rows.append(row)
        print("Unraveling average frame data...")
        for i in range(len(extracted_rows)):
            if i==0:
                unravel_height = int(extracted_rows[i][0])
                unravel_width = int(extracted_rows[i][1])
                world_vids_tbucketed[stim_number]["Vid Dimensions"] = [unravel_height, unravel_width]
            elif i==1:
                vid_count = int(extracted_rows[i][0])
                world_vids_tbucketed[stim_number]["Vid Count"] = vid_count
            else:
                tbucket_num = extracted_rows[i][0]
                flattened_frame = extracted_rows[i][1:]
                flat_frame_array = np.array(flattened_frame)
                unraveled_frame = np.reshape(flat_frame_array,(unravel_height,unravel_width))
                world_vids_tbucketed[stim_number][tbucket_num] = unraveled_frame
    return world_vids_tbucketed

def downsample_avg_world_vids(unraveled_world_vids_dict, original_bucket_size_ms, new_bucket_size_ms):
    if (new_bucket_size_ms % original_bucket_size_ms == 0):
        new_sample_rate = int(new_bucket_size_ms/original_bucket_size_ms)
        downsampled_world_vids_dict = {}
        for stim in unraveled_world_vids_dict.keys():
            print("Working on stimulus {s}".format(s=stim))
            downsampled_world_vids_dict[stim] = {}
            vid_metadata_keys = sorted([x for x in unraveled_world_vids_dict[stim].keys() if type(x) is str])
            for metadata in vid_metadata_keys:
                downsampled_world_vids_dict[stim][metadata] = unraveled_world_vids_dict[stim][metadata]
            this_stim_avg_vid_dimensions = unraveled_world_vids_dict[stim][vid_metadata_keys[1]]
            tbuckets = sorted([x for x in unraveled_world_vids_dict[stim].keys() if type(x) is float])
            padding = new_sample_rate - (int(tbuckets[-1]) % new_sample_rate)
            original_tbuckets_sliced = range(0, int(tbuckets[-1]+padding), new_sample_rate)
            new_tbucket = 0
            for i in original_tbuckets_sliced:
                start = i
                end = i + new_sample_rate - 1
                this_slice_summed_frame = np.zeros((this_stim_avg_vid_dimensions[0], this_stim_avg_vid_dimensions[1]))
                this_slice_tbuckets = []
                this_slice_count = 0
                for tbucket in tbuckets:
                    if start<=tbucket<=end:
                        this_slice_tbuckets.append(tbucket)
                for bucket in this_slice_tbuckets:
                    this_slice_summed_frame = this_slice_summed_frame + unraveled_world_vids_dict[stim][bucket]
                    this_slice_count = this_slice_count + 1
                this_slice_avg_frame = this_slice_summed_frame/float(this_slice_count)
                downsampled_world_vids_dict[stim][new_tbucket] = this_slice_avg_frame
                new_tbucket = new_tbucket + 1
        return downsampled_world_vids_dict
    else:
        print("Sample rate must be a multiple of {bucket}".format(bucket=original_bucket_size))

def matchArrays_RawVsWorld(inputArrayRaw, inputArrayWorld, phaseName, plot_saveFolder):
    # create array of nans, size = larger array (either World or Raw)
    meanAdjusted_outputArray = np.empty((len(inputArrayWorld),))
    meanAdjusted_outputArray.fill(np.nan)
    # get first and last values of Raw
    meanAdjusted_outputArray[0] = inputArrayRaw[0]
    meanAdjusted_outputArray[-1] = inputArrayRaw[-1]
    # get major peaks and troughs of Raw
    goodDataPoints = []
    inputArrayRaw_maximas = argrelextrema(inputArrayRaw, np.greater)
    inputArrayRaw_minimas = argrelextrema(inputArrayRaw, np.less)
    for maxima in inputArrayRaw_maximas[0]:
        adjustedIndex = int(round((maxima/len(inputArrayRaw))*len(inputArrayWorld), 0))
        meanAdjusted_outputArray[adjustedIndex] = inputArrayRaw[maxima]
        goodDataPoints.append(adjustedIndex)
    for minima in inputArrayRaw_minimas[0]:
        adjustedIndex = int(round((minima/len(inputArrayRaw))*len(inputArrayWorld), 0))
        meanAdjusted_outputArray[adjustedIndex] = inputArrayRaw[minima]
        goodDataPoints.append(adjustedIndex)
    # fill in as many values as can be transferred
    for i,value in enumerate(inputArrayRaw):
        adjustedIndex = int(round((i/len(inputArrayRaw))*len(inputArrayWorld), 0))
        if adjustedIndex not in goodDataPoints:
            meanAdjusted_outputArray[adjustedIndex] = value
            goodDataPoints.append(adjustedIndex)
    # if World array is longer than Raw, interpolate to fill in remaining nans
    if len(inputArrayWorld)>len(inputArrayRaw):
        goodDataPoints.sort()
        num_valid = len(goodDataPoints)
        count = 1
        for i in range(1, num_valid):
            next_valid_index = goodDataPoints[i]
            next_valid_lum = meanAdjusted_outputArray[next_valid_index]
            step_count = (next_valid_index - count + 1)
            step_lum = (next_valid_lum - meanAdjusted_outputArray[count - 1]) / step_count
            for j in range(step_count):
                meanAdjusted_outputArray[count] = meanAdjusted_outputArray[count - 1] + step_lum
                count += 1
    # draw output array against world cam avg array
    meanWorldScaled_array = inputArrayWorld*(inputArrayRaw[0]/inputArrayWorld[0])
    # figure path and title
    figPath = os.path.join(plot_saveFolder, 'meanWorldScaled%s_Vs_meanAdjusted%s.png'%(phaseName, phaseName))
    figTitle = 'Mean luminance of world cam (scaled) vs mean luminance of raw stimulus (adjusted) during %s'%(phaseName)
    print('Plotting %s'%(figTitle))
    # draw comparison plot
    plt.figure(figsize=(9, 9), dpi=150)
    plt.suptitle(figTitle, fontsize=12, y=0.98)
    plt.ylabel('Timebuckets')
    plt.xlabel('Mean luminance')
    plt.plot(meanWorldScaled_array, label='World cam (scaled)')
    plt.plot(meanAdjusted_outputArray, label='Raw Stim (adjusted)')
    plt.legend()
    plt.savefig(figPath)
    plt.close()
    return meanAdjusted_outputArray

###################################
# DATA AND OUTPUT FILE LOCATIONS
###################################
# List relevant data locations: this is for laptop
root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
# List relevant data locations: this is for office desktop (windows)
#root_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
# set up folders
rawStim_lums_folder = os.path.join(root_folder, "rawStimLums")
stimVid_lums_folder = os.path.join(root_folder, "stimVidLums")
stimVid_plots = os.path.join(plots_folder, "stimulusAvgLum")
# Create folders they do not exist
output_folders = [stimVid_lums_folder, stimVid_plots]
for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

###################################
# TIMING/SAMPLING VARIABLES FOR DATA EXTRACTION
###################################
# downsample = collect data from every 40ms or other multiples of 20
downsampled_bucket_size_ms = 40
original_bucket_size_in_ms = 4
max_length_of_stim_vid = 60000 # milliseconds
no_of_time_buckets = max_length_of_stim_vid/original_bucket_size_in_ms
downsampled_no_of_time_buckets = max_length_of_stim_vid/downsampled_bucket_size_ms
new_time_bucket_sample_rate = downsampled_bucket_size_ms/original_bucket_size_in_ms
milliseconds_for_baseline = 3000
baseline_no_buckets = int(milliseconds_for_baseline/new_time_bucket_sample_rate)
###################################
# STIMULI VID INFO
###################################
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
stim_name_to_float = {"Stimuli24": 24.0, "Stimuli25": 25.0, "Stimuli26": 26.0, "Stimuli27": 27.0, "Stimuli28": 28.0, "Stimuli29": 29.0}
stim_float_to_name = {24.0: "Stimuli24", 25.0: "Stimuli25", 26.0: "Stimuli26", 27.0: "Stimuli27", 28.0: "Stimuli28", 29.0: "Stimuli29"}

###################################
### EXTRACT, UNRAVEL, SAVE TO FILE TIME BINNED STIM VIDEOS
###################################
allMonths_meanWorldVidArrays = {}
for unique_stim in stim_vids:
    allMonths_meanWorldVidArrays[unique_stim] = {}
    allMonths_meanWorldVidArrays[unique_stim]['Vid Count'] = 0
# update list of completed world vid average folders on dropbox
day_folders = sorted(os.listdir(root_folder))
avg_world_vid_folders = fnmatch.filter(day_folders, 'WorldVidAverage_*')
updated_folders_to_extract = []
for avg_world_vid_folder in avg_world_vid_folders:
    folder_year_month = avg_world_vid_folder.split('_')[1]
    if folder_year_month not in allMonths_meanWorldVidArrays.keys():
        updated_folders_to_extract.append(avg_world_vid_folder)

#### WHILE DEBUGGING ####
#updated_folders_to_extract = updated_folders_to_extract[4:6]
#debugging_output_folder = os.path.join(root_folder, 'test_stimVidLums')
#### --------------- ####

# extract, unravel, calculate mean luminance of each frame, create array of mean luminances for each stim type
for month_folder in updated_folders_to_extract:
    month_name = month_folder.split('_')[1]
    month_folder_path = os.path.join(root_folder, month_folder)
    # unravel
    unraveled_monthly_world_vids = load_avg_world_unraveled(month_folder_path)
    # downsample
    print("Downsampling monthly averaged stimulus videos for {month}".format(month=month_name))
    downsampled_monthly_world_vids = downsample_avg_world_vids(unraveled_monthly_world_vids, original_bucket_size_in_ms, downsampled_bucket_size_ms)
    # now need to convert these frame arrays into luminance value, one per timebucket
    for unique_stim in downsampled_monthly_world_vids:
        thisMonth_thisStim_frames = downsampled_monthly_world_vids[unique_stim]
        thisMonth_thisStim_lums = []
        for key in thisMonth_thisStim_frames:
            if key == 'Vid Count':
                allMonths_meanWorldVidArrays[unique_stim]['Vid Count'] = allMonths_meanWorldVidArrays[unique_stim]['Vid Count'] + thisMonth_thisStim_frames['Vid Count']
                continue
            if key == 'Vid Dimensions':
                continue
            else:
                frame = thisMonth_thisStim_frames[key]
                lum = np.nanmean(frame[:])
                thisMonth_thisStim_lums.append(lum)
        thisMonth_thisStim_lums_array = np.array(thisMonth_thisStim_lums)
        allMonths_meanWorldVidArrays[unique_stim][month_name] = thisMonth_thisStim_lums_array

###################################
# AVERAGE ACROSS ALL MONTHS
###################################
for unique_stim in allMonths_meanWorldVidArrays:
    allMonthlyMeans = []
    shortest = 2000
    for key in allMonths_meanWorldVidArrays[unique_stim]:
        if key == 'Vid Count':
            continue
        else:
            thisMonthMean = allMonths_meanWorldVidArrays[unique_stim][key]
            if len(thisMonthMean)<shortest:
                shortest = len(thisMonthMean)
            allMonthlyMeans.append(thisMonthMean)       
    # make all arrays same length
    allMonthlyMeans_truncated = []
    for monthlyMean in allMonthlyMeans:
        monthlyMean_truncated = monthlyMean[:shortest]
        allMonthlyMeans_truncated.append(monthlyMean_truncated)
    allMonthlyMeans_array = np.array(allMonthlyMeans_truncated)
    thisStimMeanLum = np.nanmean(allMonthlyMeans_array, axis=0)
    allMonths_meanWorldVidArrays[unique_stim]['All Months'] = thisStimMeanLum

###################################
# SPLIT ARRAYS INTO CALIB, OCTO, AND UNIQUE PHASES
###################################
# Moments of interest for each stimulus type
all_avg_world_moments = {}
# Stimulus 24.0
all_avg_world_moments[24.0] = {'calibration start': {0:['2017-10','2018-05']},
'do not move your head': {3:['2017-10','2018-05']},
'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
'lower right dot appears': {170:['2017-10','2018-05']},
'lower left dot appears': {238:['2017-10','2018-05']},
'upper right dot appears': {306:['2017-10','2018-05']},
'center dot appears': {374:['2017-10','2018-05']},
'calibration end': {441:['2017-10','2017-11','2018-03']},
'unique start': {442:['2017-10','2018-03','2018-05'],443:['2017-11']},
'cat appears': {463:['2017-10','2018-01','2018-05'], 464:['2017-11']},
'cat front paws visible': {473:['2017-10','2018-01','2018-05'], 474:['2017-11']},
'cat lands on toy': {513:['2017-10'], 514:['2018-05']},
'cat back paws bounce': {549:['2017-10'],547:['2018-05']},
'unique end': {596:['2017-10','2017-11'],598:['2018-03']},
'octo start': {595:['2017-10','2018-03'],596:['2017-11']},
'fish turns': {645:['2017-10','2018-05']},
'octopus fully decamouflaged': {766:['2018-05'], 767:['2017-10']},
'camera zooms in on octopus': {860:['2017-10','2018-05']},
'octopus inks': {882:['2017-10'],883:['2017-11','2018-03']},
'camera clears ink cloud': {916:['2017-10'],920:['2018-05']},
'octo end': {987:['2017-10'],989:['2017-11'],990:['2018-03']}}
# Stimulus 25.0
all_avg_world_moments[25.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
'do not move your head': {3:['2017-10','2018-05']},
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
all_avg_world_moments[26.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
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
all_avg_world_moments[27.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
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
all_avg_world_moments[28.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
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
all_avg_world_moments[29.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
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
# split world vid lum arrays
uniqueWeights = {}
allWeightedDoNotMove = []
allWeightedPulsingDots = []
allWeightedOcto = []
allWeightedUnique = []
doNotMoveLens = []
pulsingDotsLens = []
octoLens = []
uniqueLens = []
uniqueOrder = []
shortestDoNotMove = 2000
shortestPulsingDots = 2000
shortestOcto = 2000
# cut out each phase of the stimuli
for unique_stim in allMonths_meanWorldVidArrays:
    thisUniqueStim_weight = allMonths_meanWorldVidArrays[unique_stim]['Vid Count']
    uniqueWeights[unique_stim] = thisUniqueStim_weight
    fullWeightedMeanWorldVid = allMonths_meanWorldVidArrays[unique_stim]['All Months']*thisUniqueStim_weight
    ## CALIB
    # Do Not Move section
    calibStart = []
    for key in all_avg_world_moments[unique_stim]['calibration start']:
        calibStart.append(key)
    calibStart_tb = np.min(calibStart)
    doNotMoveEnd = []
    for key in all_avg_world_moments[unique_stim]['upper left dot appears']:
        doNotMoveEnd.append(key)
    doNotMoveEnd_tb = np.min(doNotMoveEnd) - 1
    # pulsing dots section
    pulsingDotsStart_tb = doNotMoveEnd_tb + 1
    calibEnd = []
    for key in all_avg_world_moments[unique_stim]['calibration end']:
        calibEnd.append(key)
    calibEnd_tb = np.max(calibEnd)
    # cut out Do Not Move section of calib phase from full weighted mean world vid lum array
    thisStim_weightedMeanDoNotMove = fullWeightedMeanWorldVid[calibStart_tb:doNotMoveEnd_tb]
    if len(thisStim_weightedMeanDoNotMove)<shortestDoNotMove:
        shortestDoNotMove = len(thisStim_weightedMeanDoNotMove)
    allWeightedDoNotMove.append(thisStim_weightedMeanDoNotMove)
    thisStim_weightedMeanPulsingDots = fullWeightedMeanWorldVid[pulsingDotsStart_tb:calibEnd_tb]
    if len(thisStim_weightedMeanPulsingDots)<shortestPulsingDots:
        shortestPulsingDots = len(thisStim_weightedMeanPulsingDots)
    allWeightedPulsingDots.append(thisStim_weightedMeanPulsingDots)
    print('Unique Stim %d, "Do Not Move" length: %d, Pulsing Dots length: %d'%(unique_stim, len(thisStim_weightedMeanDoNotMove), len(thisStim_weightedMeanPulsingDots)))
    doNotMoveLen = doNotMoveEnd_tb - calibStart_tb
    doNotMoveLens.append(doNotMoveLen)
    pulsingDotsLen = calibEnd_tb - pulsingDotsStart_tb
    pulsingDotsLens.append(pulsingDotsLen)
    ## OCTO
    octoStart = []
    for key in all_avg_world_moments[unique_stim]['octo start']:
        octoStart.append(key)
    octoStart_tb = np.min(octoStart)
    octoEnd = []
    for key in all_avg_world_moments[unique_stim]['octo end']:
        octoEnd.append(key)
    octoEnd_tb = np.max(octoEnd)
    # cut out octo phase from full world vid lum array
    thisStim_weightedMeanOcto = fullWeightedMeanWorldVid[octoStart_tb:octoEnd_tb]
    if len(thisStim_weightedMeanOcto)<shortestOcto:
        shortestOcto = len(thisStim_weightedMeanOcto)
    allWeightedOcto.append(thisStim_weightedMeanOcto)
    octoLen = octoEnd_tb - octoStart_tb
    octoLens.append(octoLen)
    ### UNIQUE
    thisUniqueStart = []
    for key in all_avg_world_moments[unique_stim]['unique start']:
        thisUniqueStart.append(key)
    thisUniqueStart_tb = np.min(thisUniqueStart)
    thisUniqueEnd = []
    for key in all_avg_world_moments[unique_stim]['unique end']:
        thisUniqueEnd.append(key)
    thisUniqueEnd_tb = np.max(thisUniqueEnd)
    uniqueLen = thisUniqueEnd_tb - thisUniqueStart_tb
    uniqueLens.append(uniqueLen)
    # cut out unique phase from full world vid lum array
    thisStim_weightedMeanUnique = fullWeightedMeanWorldVid[thisUniqueStart_tb:thisUniqueEnd_tb]
    allWeightedUnique.append(thisStim_weightedMeanUnique)
    uniqueOrder.append(unique_stim)

# calculate weighted mean of doNotMove, pulsingDots, octo and unique phases
total_worldVids = sum(uniqueWeights.values())
allDoNotMove_truncated = []
for doNotMove in allWeightedDoNotMove:
    doNotMove_truncated = doNotMove[:shortestDoNotMove]
    allDoNotMove_truncated.append(doNotMove_truncated)
meanWorld_doNotMove = np.nansum(allDoNotMove_truncated, axis=0)/total_worldVids
allPulsingDots_truncated = []
for pulsingDots in allWeightedPulsingDots:
    pulsingDots_truncated = pulsingDots[:shortestPulsingDots]
    allPulsingDots_truncated.append(pulsingDots_truncated)
meanWorld_pulsingDots = np.nansum(allPulsingDots_truncated, axis=0)/total_worldVids
allOcto_truncated = []
for octo in allWeightedOcto:
    octo_truncated = octo[:shortestOcto]
    allOcto_truncated.append(octo_truncated)
meanWorld_octo = np.nansum(allOcto_truncated, axis=0)/total_worldVids
allMeanWorld_unique = []
for i, unique in enumerate(allWeightedUnique):
    thisMeanWorld_unique = unique/uniqueWeights[uniqueOrder[i]]
    allMeanWorld_unique.append(thisMeanWorld_unique)
meanWorld_u1 = allMeanWorld_unique[0]
meanWorld_u2 = allMeanWorld_unique[1]
meanWorld_u3 = allMeanWorld_unique[2]
meanWorld_u4 = allMeanWorld_unique[3]
meanWorld_u5 = allMeanWorld_unique[4]
meanWorld_u6 = allMeanWorld_unique[5]

###################################
# LOAD CSV OF RAW STIM VIDEOS
###################################
rawStimLum_files = glob.glob(rawStim_lums_folder + os.sep + '*.csv')
allRaw_doNotMove = []
allRaw_doNotMove_languageOrder = []
allRaw_pulsingDots = []
allRaw_unique = []
allRaw_unique_order = []
allRaw_octo = []
rawUniqueLens_frames = {'stimuli024':168, 'stimuli025':172, 'stimuli026':247, 'stimuli027':179, 'stimuli028':246, 'stimuli029':313}
for rawStimLumFile in rawStimLum_files:
    stim_phase = os.path.basename(rawStimLumFile).split('_')[0].split('-')[0]
    if stim_phase == 'Calibration':
        rawPulsingDots = np.genfromtxt(rawStimLumFile, delimiter=',')
        allRaw_pulsingDots.append(rawPulsingDots)
        continue
    if stim_phase == 'DoNotMove':
        stim_language = os.path.basename(rawStimLumFile).split('_')[0].split('-')[1]
        raw_doNotMove = np.genfromtxt(rawStimLumFile, delimiter=',')
        allRaw_doNotMove.append(raw_doNotMove)
        allRaw_doNotMove_languageOrder.append(stim_language)
        continue
    if stim_phase == 'CenterEye' or stim_phase == 'Replay' or stim_phase == 'RestingState':
        print('Skipping %s raw stimulus video'%(stim_phase))
        continue
    else:
        rawUniqueLen_frames = rawUniqueLens_frames[stim_phase]
        rawUniqueStim_full = np.genfromtxt(rawStimLumFile, delimiter=',')
        thisRawUnique = rawUniqueStim_full[:rawUniqueLen_frames]
        allRaw_unique.append(thisRawUnique)
        allRaw_unique_order.append(stim_phase)
        thisRawOcto = rawUniqueStim_full[rawUniqueLen_frames+1:]
        allRaw_octo.append(thisRawOcto)
        print('%s, unique phase: %d frames, octo phase: %d frames'%(stim_phase, len(thisRawUnique), len(thisRawOcto)))

###################################
# BUILD MEAN RAW STIM VID LUMINANCE ARRAYS
###################################
allLanguages_activationCount = {'Chinese':15, 'English':934, 'French':44, 'German':95, 'Italian':15}
# start counting again at 2017-12-20
# DO NOT MOVE
total_activations = sum(allLanguages_activationCount.values())
allWeighted_DoNotMove = []
for language in allLanguages_activationCount.keys():
    thisLanguage_weighted = allRaw_doNotMove[allRaw_doNotMove_languageOrder.index(language)]*allLanguages_activationCount[language]
    allWeighted_DoNotMove.append(thisLanguage_weighted)
meanRaw_doNotMove = sum(allWeighted_DoNotMove)/total_activations
# PULSING DOTS
meanRaw_pulsingDots = sum(allRaw_pulsingDots)/len(allRaw_pulsingDots)
# OCTO
meanRaw_octo = sum(allRaw_octo)/len(allRaw_octo)
# UNIQUE SEQUENCES
meanRaw_u1 = allRaw_unique[0]
meanRaw_u2 = allRaw_unique[1]
meanRaw_u3 = allRaw_unique[2]
meanRaw_u4 = allRaw_unique[3]
meanRaw_u5 = allRaw_unique[4]
meanRaw_u6 = allRaw_unique[5]

###################################
# MATCH SIZES OF RAW STIM PHASES TO WORLD CAM STIM PHASES
# look for peaks/important moments in raw vid and match timing in world vids
###################################
# CALIB
## do not move
meanAdjusted_doNotMove = np.empty((len(meanWorld_doNotMove),))
meanAdjusted_doNotMove.fill(meanRaw_doNotMove[0])
## pulsing dots
meanAdjusted_pulsingDots = matchArrays_RawVsWorld(meanRaw_pulsingDots, meanWorld_pulsingDots, 'pulsingDots', stimVid_plots)
# FULL CALIB - concatenate doNotMove and pulsingDots
meanWorld_calib = np.concatenate((meanWorld_doNotMove, meanWorld_pulsingDots), axis=0)
meanRaw_calib = np.concatenate((meanRaw_doNotMove, meanRaw_pulsingDots), axis=0)
meanAdjusted_calib = np.concatenate((meanAdjusted_doNotMove, meanAdjusted_pulsingDots), axis=0)
# OCTO
meanAdjusted_octo = matchArrays_RawVsWorld(meanRaw_octo, meanWorld_octo, 'octo', stimVid_plots)
# UNIQUE
meanAdjusted_u1 = matchArrays_RawVsWorld(meanRaw_u1, meanWorld_u1, 'u1', stimVid_plots)
meanAdjusted_u2 = matchArrays_RawVsWorld(meanRaw_u2, meanWorld_u2, 'u2', stimVid_plots)
meanAdjusted_u3 = matchArrays_RawVsWorld(meanRaw_u3, meanWorld_u3, 'u3', stimVid_plots)
meanAdjusted_u4 = matchArrays_RawVsWorld(meanRaw_u4, meanWorld_u4, 'u4', stimVid_plots)
meanAdjusted_u5 = matchArrays_RawVsWorld(meanRaw_u5, meanWorld_u5, 'u5', stimVid_plots)
meanAdjusted_u6 = matchArrays_RawVsWorld(meanRaw_u6, meanWorld_u6, 'u6', stimVid_plots)

###################################
# SAVE INTERMEDIATE DATA FILES
###################################
# generate file names to include number of videos that went into the mean lum array
totalVidCount = 0
for unique_stim in allMonths_meanWorldVidArrays:
    totalVidCount = totalVidCount + allMonths_meanWorldVidArrays[unique_stim]['Vid Count']
# filepaths
calib_output = stimVid_lums_folder + os.sep + 'meanAdjustedCalib_%sVids_%dTBs.npy' % (totalVidCount, len(meanAdjusted_calib))
octo_output = stimVid_lums_folder + os.sep + 'meanAdjustedOcto_%sVids_%dTBs.npy' % (totalVidCount, len(meanAdjusted_octo))
unique24_output = stimVid_lums_folder + os.sep + 'meanAdjustedU1_%sVids_%dTBs.npy' % (allMonths_meanWorldVidArrays[24.0]['Vid Count'], len(meanAdjusted_u1))
unique25_output = stimVid_lums_folder + os.sep + 'meanAdjustedU2_%sVids_%dTBs.npy' % (allMonths_meanWorldVidArrays[25.0]['Vid Count'], len(meanAdjusted_u2))
unique26_output = stimVid_lums_folder + os.sep + 'meanAdjustedU3_%sVids_%dTBs.npy' % (allMonths_meanWorldVidArrays[26.0]['Vid Count'], len(meanAdjusted_u3))
unique27_output = stimVid_lums_folder + os.sep + 'meanAdjustedU4_%sVids_%dTBs.npy' % (allMonths_meanWorldVidArrays[27.0]['Vid Count'], len(meanAdjusted_u4))
unique28_output = stimVid_lums_folder + os.sep + 'meanAdjustedU5_%sVids_%dTBs.npy' % (allMonths_meanWorldVidArrays[28.0]['Vid Count'], len(meanAdjusted_u5))
unique29_output = stimVid_lums_folder + os.sep + 'meanAdjustedU6_%sVids_%dTBs.npy' % (allMonths_meanWorldVidArrays[29.0]['Vid Count'], len(meanAdjusted_u6))

# save to file
np.save(calib_output, meanAdjusted_calib)
np.save(octo_output, meanAdjusted_octo)
np.save(unique24_output, meanAdjusted_u1)
np.save(unique25_output, meanAdjusted_u2)
np.save(unique26_output, meanAdjusted_u3)
np.save(unique27_output, meanAdjusted_u4)
np.save(unique28_output, meanAdjusted_u5)
np.save(unique29_output, meanAdjusted_u6)

# FIN