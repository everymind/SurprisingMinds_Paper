### --------------------------------------------------------------------------- ###
# loads normalised mean pupil sizes for each stimulus phase
# loads linear regression params for all phases
# generates prediction of pupil size based on avg lum of stimuli
# compares to observed pupil sizes to find moments where observed deviates from predicted pupil size
# NOTE: in command line run with optional tags 
#       1) '--loc *' to run with various root data locations (see first function below)
### --------------------------------------------------------------------------- ###
import os
import numpy as np 
import matplotlib.pyplot as plt
import glob
import datetime
import logging
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
logging.basicConfig(filename="psa03_LumBasedPupilSizePredictor_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log", filemode='w', level=logging.INFO)
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
    # pupil size data
    normedMeanPupils_folder = os.path.join(root_folder, 'normedMeanPupilSizes')
    # linear regression data
    linRegressParams_folder = os.path.join(root_folder, 'pupilSizeVsDelayLinRegress')
    # stim vid luminance data
    stimVidLums_folder = os.path.join(root_folder, 'stimVidLums')
    # output folders
    pupilSize_folder = os.path.join(plots_folder, "pupilSizeAnalysis")
    Rco_predictedVsReal_folder = os.path.join(pupilSize_folder, 'rightContours', 'predVsReal')
    Rci_predictedVsReal_folder = os.path.join(pupilSize_folder, 'rightCircles', 'predVsReal')
    Lco_predictedVsReal_folder = os.path.join(pupilSize_folder, 'leftContours', 'predVsReal')
    Lci_predictedVsReal_folder = os.path.join(pupilSize_folder, 'leftCircles', 'predVsReal')
    # Create output folders if they do not exist
    output_folders = [pupilSize_folder, Rco_predictedVsReal_folder, Rci_predictedVsReal_folder, Lco_predictedVsReal_folder, Lci_predictedVsReal_folder]
    for output_folder in output_folders:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    return root_folder, plots_folder, normedMeanPupils_folder, linRegressParams_folder, stimVidLums_folder, output_folders

##########################################################
def predictPupilSizeFromStimVidLums(stimVidLumArrays_allPhases, linRegParams_allDelays, bestDelayIndex):
    # for each delay, params are [slope, intercept, rval, pval, stderr]
    slope = linRegParams_allDelays[bestDelayIndex][0]
    intercept = linRegParams_allDelays[bestDelayIndex][1]
    predictedPupilSizes_allPhases = []
    for phase in stimVidLumArrays_allPhases:
        predictedPupilSizes_thisPhase = []
        for timebucket in phase:
            pupilSize = slope*timebucket + intercept
            predictedPupilSizes_thisPhase.append(pupilSize)
        predictedPupilSizes_allPhases.append(np.array(predictedPupilSizes_thisPhase))
    return predictedPupilSizes_allPhases

def drawPredictedVsRealPupilSize(predictedPupilSizes_allPhases, realPupilSizes_allPhases_bestDelay, phaseOrderStrList, bestDelay_ms, downsample_ms, saveFolder):
    for i, phase in enumerate(predictedPupilSizes_allPhases):
        lenOfPhase = len(realPupilSizes_allPhases_bestDelay[i])
        # figure path and title
        figPath = os.path.join(saveFolder, '%s_predVsRealPupilSizes_%s.png'%(phaseOrderStrList[i], 'RightContours'))
        figTitle = 'Predicted (based on stimulus luminance) vs Real Pupil Sizes \n Phase: %s; %s; Best delay = %dms'%(phaseOrderStrList[i], 'RightContours', bestDelay_ms)
        print('Plotting %s'%(figTitle))
        logging.INFO('Plotting %s'%(figTitle))
        # draw predicted vs real pupil size
        plt.figure(dpi=150)
        plt.suptitle(figTitle, fontsize=12, y=0.98)
        plt.xlabel('Time (ms)')
        plt.ylabel('Pupil diameter (normalised change from baseline)')
        plt.xticks(np.arange(0,lenOfPhase,50), np.arange(0,lenOfPhase,50)*downsample_ms)
        plt.plot(predictedPupilSizes_allPhases[i], label='predicted')
        plt.plot(realPupilSizes_allPhases_bestDelay[i], label='real')
        plt.legend()
        plt.tight_layout(rect=[0,0.03,1,0.93])
        plt.savefig(figPath)
        plt.close()

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
    root_folder, plots_folder, normedMeanPupils_folder, linRegressParams_folder, stimVidLums_folder, output_folders = load_data(args.loc)
    pupilSize_folder = output_folders[0]
    Rco_predictedVsReal_folder = output_folders[1]
    Rci_predictedVsReal_folder = output_folders[2]
    Lco_predictedVsReal_folder = output_folders[3]
    Lci_predictedVsReal_folder = output_folders[4]
    logging.info('ROOT FOLDER: %s \n PLOTS FOLDER: %s \n Pupil Size folder: %s \n Right Contours folder: %s \n Right Circles folder: %s \n Left Contours folder: %s \n Left Circles folder: %s' % (root_folder, plots_folder, pupilSize_folder, Rco_predictedVsReal_folder, Rci_predictedVsReal_folder, Lco_predictedVsReal_folder, Lci_predictedVsReal_folder))
    print('ROOT FOLDER: %s \n PLOTS FOLDER: %s \n Pupil Size folder: %s \n Right Contours folder: %s \n Right Circles folder: %s \n Left Contours folder: %s \n Left Circles folder: %s' % (root_folder, plots_folder, pupilSize_folder, Rco_predictedVsReal_folder, Rci_predictedVsReal_folder, Lco_predictedVsReal_folder, Lci_predictedVsReal_folder))
    ###################################
    # TIMING/SAMPLING VARIABLES FOR DATA EXTRACTION
    ###################################
    # downsample = collect data from every 40ms or other multiples of 20
    downsampled_bucket_size_ms = 40
    original_bucket_size_in_ms = 4
    ###################################
    # load linear regression parameters for pupil size vs avg luminance of stimulus video
    # organise by eyeAnalysis_type, phase_type, and delay_ms
    ###################################
    # extract linear regression parameters to find best delay
    linRegressParams_files = glob.glob(linRegressParams_folder + os.sep + '*.npy')
    for linReg_file in linRegressParams_files:
        eyeAnalysis = os.path.basename(linReg_file).split('.')[0].split('_')[1]
        thisEyeAnalysis_linRegParams = np.load(linReg_file)
        best_rVal = 0
        best_delay = 0
        for i, delay in enumerate(thisEyeAnalysis_linRegParams):
            if delay[2]<best_rVal:
                best_rVal = delay[2]
                best_delay = i
        if eyeAnalysis == 'RightContours':
            Rco_linRegParams = thisEyeAnalysis_linRegParams
            Rco_bestRVal = best_rVal
            Rco_bestDelay_index = best_delay
        if eyeAnalysis == 'RightCircles':
            Rci_linRegParams = thisEyeAnalysis_linRegParams
            Rci_bestRVal = best_rVal
            Rci_bestDelay_index = best_delay
        if eyeAnalysis == 'LeftContours':
            Lco_linRegParams = thisEyeAnalysis_linRegParams
            Lco_bestRVal = best_rVal
            Lco_bestDelay_index = best_delay
        if eyeAnalysis == 'LeftCircles':
            Lci_linRegParams = thisEyeAnalysis_linRegParams
            Lci_bestRVal = best_rVal
            Lci_bestDelay_index = best_delay
    # now we should have:
    # Rco_linRegParams, Rco_bestRVal, Rco_bestDelay_index
    # Rci_linRegParams, Rci_bestRVal, Rci_bestDelay_index
    # Lco_linRegParams, Lco_bestRVal, Lco_bestDelay_index
    # Lci_linRegParams, Lci_bestRVal, Lci_bestDelay_index
    Rco_bestDelay_ms = Rco_bestDelay_index*downsampled_bucket_size_ms
    Rci_bestDelay_ms = Rci_bestDelay_index*downsampled_bucket_size_ms
    Lco_bestDelay_ms = Lco_bestDelay_index*downsampled_bucket_size_ms
    Lci_bestDelay_ms = Lci_bestDelay_index*downsampled_bucket_size_ms
    ###################################
    # load stim vid avg luminances
    ###################################
    stimVidLums_files = glob.glob(stimVidLums_folder + os.sep + '*.npy')
    unique_lums = []
    for stimVid_file in stimVidLums_files:
        phaseType = os.path.basename(stimVid_file).split('.')[0].split('_')[0][12]
        thisPhase_avgLums = np.load(stimVid_file)
        if phaseType == 'C':
            meanAdjusted_calibLums = thisPhase_avgLums
        if phaseType == 'O':
            meanAdjusted_octoLums = thisPhase_avgLums
        if phaseType == 'U':
            unique_lums.append(thisPhase_avgLums)
    stimVidLums_allPhases = [meanAdjusted_calibLums, meanAdjusted_octoLums, unique_lums[0], unique_lums[1], unique_lums[2], unique_lums[3], unique_lums[4], unique_lums[5]]
    ###################################
    # build prediction from linear regression params
    ###################################
    Rco_predictedPupilSizes_allPhases = predictPupilSizeFromStimVidLums(stimVidLums_allPhases, Rco_linRegParams, Rco_bestDelay_index)
    Rci_predictedPupilSizes_allPhases = predictPupilSizeFromStimVidLums(stimVidLums_allPhases, Rci_linRegParams, Rci_bestDelay_index)
    Lco_predictedPupilSizes_allPhases = predictPupilSizeFromStimVidLums(stimVidLums_allPhases, Lco_linRegParams, Lco_bestDelay_index)
    Lci_predictedPupilSizes_allPhases = predictPupilSizeFromStimVidLums(stimVidLums_allPhases, Lci_linRegParams, Lci_bestDelay_index)
    ###################################
    # load normalised mean pupil size files
    # organise by eyeAnalysis_type, phase_type, and delay_ms
    ###################################
    # extract mean pupil sizes for best delay
    normedMeanPupils_files = glob.glob(normedMeanPupils_folder + os.sep + '*.npy')
    Rco_bestDelay_allPhases = []
    Rco_bestDelay_phaseOrder = []
    Rci_bestDelay_allPhases = []
    Rci_bestDelay_phaseOrder = []
    Lco_bestDelay_allPhases = []
    Lco_bestDelay_phaseOrder = []
    Lci_bestDelay_allPhases = []
    Lci_bestDelay_phaseOrder = []
    for nmpSize_file in normedMeanPupils_files:
        delay_ms = int(os.path.basename(nmpSize_file).split('_')[2][:-7])
        eyeAnalysis_type = os.path.basename(nmpSize_file).split('_')[3][:-4]
        phase_type = os.path.basename(nmpSize_file).split('_')[1]
        this_nmpSize = np.load(nmpSize_file)
        if eyeAnalysis_type == 'RightContours':
            if delay_ms == Rco_bestDelay_ms:
                Rco_bestDelay_allPhases.append(this_nmpSize)
                Rco_bestDelay_phaseOrder.append(phase_type)
                continue
        if eyeAnalysis_type == 'RightCircles':
            if delay_ms == Rci_bestDelay_ms:
                Rci_bestDelay_allPhases.append(this_nmpSize)
                Rci_bestDelay_phaseOrder.append(phase_type)
                continue
        if eyeAnalysis_type == 'LeftContours':
            if delay_ms == Lco_bestDelay_ms:
                Lco_bestDelay_allPhases.append(this_nmpSize)
                Lco_bestDelay_phaseOrder.append(phase_type)
                continue
        if eyeAnalysis_type == 'LeftCircles':
            if delay_ms == Lci_bestDelay_ms:
                Lci_bestDelay_allPhases.append(this_nmpSize)
                Lci_bestDelay_phaseOrder.append(phase_type)
                continue
    ###################################
    # plot predicted vs real pupil size
    ###################################
    drawPredictedVsRealPupilSize(Rco_predictedPupilSizes_allPhases, Rco_bestDelay_allPhases, Rco_bestDelay_phaseOrder, Rco_bestDelay_ms, downsampled_bucket_size_ms, Rco_predictedVsReal_folder)
    drawPredictedVsRealPupilSize(Rci_predictedPupilSizes_allPhases, Rci_bestDelay_allPhases, Rci_bestDelay_phaseOrder, Rci_bestDelay_ms, downsampled_bucket_size_ms, Rci_predictedVsReal_folder)
    drawPredictedVsRealPupilSize(Lco_predictedPupilSizes_allPhases, Lco_bestDelay_allPhases, Lco_bestDelay_phaseOrder, Lco_bestDelay_ms, downsampled_bucket_size_ms, Lco_predictedVsReal_folder)
    drawPredictedVsRealPupilSize(Lci_predictedPupilSizes_allPhases, Lci_bestDelay_allPhases, Lci_bestDelay_phaseOrder, Lci_bestDelay_ms, downsampled_bucket_size_ms, Lci_predictedVsReal_folder)

# FIN