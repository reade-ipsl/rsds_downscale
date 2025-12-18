"""
Created on 25.08.25
@author Rosie Eade

Code to train and compare ML architectures using pytorch, e.g. 1D CNN
- With comprehensive performance output in terms of prediction skill

Aim: Downscale solar irradiance datasets to increase temporaral and spatial resolution.

-----------------------------------
Run by typing into terminal:
module purge
module load pytorch/2.6.0 # example python environment to make sure have pytorch
python -i downscale_ts/rsdsMain_ObsPrediction.py 'W3033' 16 '10cS' 3 'France'
python downscale_ts/rsdsMain_ObsPrediction.py 'W0001' 16 '10cS' 3 'France'

Or use a script to submit, e.g.
run_rsdsMain_ObsPrediction.bash

Requirements:
# Memory needed = 1Gb per run
# Time needed = 2-10min per run
# Storage needed = 8Mb per run

----------------------------------
Command Line Parameters
----------
sys.argv[1] : string - Code representing loss function definition (string)
sys.argv[2] : int - Batch size number (int)
sys.argv[3] : string - Code representing ML model architecture to use (or list of architectures, see model_name_list in main())
sys.argv[4] : int - Code representing years used for training ML model and years (e.g. out of sample) for testing ML model after fully trained
sys.argv[5] : string - Namecode representing geographical location for training and testing

Hard Coded Parameters
----------
Outside/above main()
# GLOBAL DATA PATHS
- define director OUT_DIR = directory to save output models and plots


Inside main()
- Input Parameters to define how to setup models and evaluation methods
# Section INPUT PARAMETERS
# Parameters for data preparation
# Parameters for training the ML model
e.g. train_test_case: number code defines partioning of input data years for A: training+validation and B: testing prediction
* Training : Data used to train ML model (set to be all but the final year of data input A)
* Validation : Data used to validate ML model within training loops (set to be final year of data input A)
* Testing Prediction : Out of sample data used to test ML model predictions once have final version (data input B)

variable names:
short => 3hour input (8 values per day)
long  => 30min input (48 values per day)

----------------------------------
OUTPUT : 
----------

Output files to pdir = 25Mb if plot everything

Filename codes:
 PR = prediction period
 TR = training period
 mlr = multiple linear regression model output e.g. mlr56 (input 56 variables = 8 rsds total 3hr values, 48 clearsky direct 30min values)
 ML = machine learning model output e.g. ML10cS
 TOT = Total solar radiation
 DIR = Direct only solar radiation

------Machine Learning Model------

pdir:
Output Trained Machine Learning model(s) and graph of Loss Function
  *model.pth
  *_LossFn.png
  
------AND RESULTS TABLES of PERFORMANCE METRICS ON PREDICTION (and training) PERIODS------

pdir:
Output loss function statistics and simple performance metrics
  rsdsTOTnDIR_SARAH3_*_TR_LossFn_TRPR_RMSE.txt # TOTal and DIRect Performance Metrics on TRaining and PRediction period
      TR Loss = Loss function for scaled output data from MLR and ML models vs Target (ie before revert to W/m2 units) on TRaining period
      TR RMSE = RMSE for scaled output data from MLR and ML models vs Target (ie before revert to W/m2 units) on TRaining period
      PR RMSE = RMSE for scaled output data from MLR and ML models vs Target (ie before revert to W/m2 units) on PRediction period

  rsdsTOTnDIR_SARAH3_*_PR_PerfMetrics.txt # TOTal and DIRect Performance Metrics on PRediction period
    CLM = Metrics to assess Climatology of model output on 30 min timesteps vs Target
      CLM SD(R-MR56) = Standard Deviation of Climatology for 'ML model radiation output MINUS MLR56 model radiation output'
      CLM RMSE(RSDS) = RMSE of Model radiation output vs Target radiation 
      CLM WD(RSDS)   = Wasserstein Distance of Model radiation output vs Target radiation 
    DSD = Metrics to assess standard deviation of sub-daily radiation within each day
      DSD RMSE(R-MR56) = RMSE of 'Model minus MLR56' vs 'Target minus MLR56'
      DSD WD(R-MR56)   = Wasserstein Distance of 'Model minus MLR56' vs 'Target minus MLR56' 
      DSD RMSE(RSDS)   = RMSE of 'Model' vs 'Targe'
      DSD WD(RSDS)     = Wasserstein Distance of 'Model' vs 'Target'

  rsdsCSKYTOT_SARAH3_*_PR_PerfMetrics.txt # Clearsky Total Performance Metrics on PRediction period
      RMSE(CSKYTOT) = RMSE of 'Model Daily mean of sub-daily Max Radiation' vs 'Daily mean of Smooth curve estimate of sub-daily clear sky total'
                      Just for timesteps identified as being representative of clearsky days

------AND RESULTS PLOTS ON PREDICTION PERIODS------

pdir:
Output plots of estimate of clearsky total on PRediction period
  rsdsTOT_*PR_target_MEANCSKYunsc_DAYtsMN.png: Target max daily rsds, including scaled and unscaled estimate of clearsky total
  rsdsTOT_*PR_target_MEANCSKY_DAYtsMN.png: Target max daily rsds, including scaled estimate of clearsky total (OBS Smoothed)
  rsdsTOT_*PR_mlr56_B5SUBsampleMEANCSKY_DAYtsMN.png: Target and MLR Model max daily rsds, including scaled estimate of clearsky total (OBS Smoothed)
  rsdsTOT_*PR_ML10cS_B5SUBsampleMEANCSKY_DAYtsMN.png: Target and ML Model max daily rsds, including scaled estimate of clearsky total (OBS Smoothed)

# PLOT Climatology Mean
Output climatology of total and direct radation on PRediction period
  rsds???_*ANNts.png: Total and Direct solar radiation climatology 

# PLOT Example day time series
pdir/ex_dayts:
Example sub-daily time series of solar radiation data for 1st Jan, Apr, Jul, Oct (Day0000 etc)
  rsdsINPUTTARGET_SARAH3*.png: Example Input and Target data after scaled for ML Model (see data preparation section)
  rsds???_SARAH3*.png: Total and Direct data for Input, Target, MLR Model output and ML model output (and clearsky total and direct estimates)
  rsds???mMLR56_SARAH3*.png: Total and Direct solar radiation Target and ML model output MINUS MLR56 model output

# PLOT Example week time series
pdir/ex_weekts:
Example sub-daily time series for 1st-7th Jan, Apr, Jul, Oct
  rsds???_SARAH3*.png: Total and Direct data for Input, Target, MLR Model output and ML model output (and clearsky total and direct estimates)
  rsds???mMLR56_SARAH3*.png: Total and Direct solar radiation Target and ML model output MINUS MLR56 model output

----------------------------------
ML model codes
    10c: Encoder MaxPool1d and Conv1d; Decoder ConvTranspose1d (concatenate so increase number of channels)
----------------------------------

"""

import datetime
now = datetime.datetime.now()
print("--------------")
print("start time")
print(now.time())
print("--------------")

import sys
import os
import importlib
import pandas as pd
import numpy as np
import scipy as sp
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg') # Improves plotting efficiency. Remove if want to pause code and make plots within run.
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import calendar
import pdb
import rsds_data_prep as rsdp		# DATA PREPARATION FUNCTIONS
import rsds_ml_models as rsmlm	# MACHINE LEARNING MODEL FUNCTIONS and MULTIPLE-LINEAR REGRESSION
import rsds_performance_metrics as rspm # PERFORMANCE METRIC FUNCTIONS

# Can reimport libraries using importlib.reload() e.g. importlib.reload(rsmlm) importlib.reload(rsdp)

# ----------------------------------------------------------------------------
# GLOBAL DATA PATHS
# ----------------------------------------------------------------------------

# DIRECTORY For Output Plots 
OUT_DIR='results/rsds_ObsPrediction/'

# ----------------------------------------------------------------------------
def seed_worker(worker_id):
    # Implemented for reproducibility
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


# ----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------------------------------
def plot_loss(loss_out, filepath='file.png', FIGWIDTH=8, FIGHEIGHT=6):
    """
    Plot loss function from ML model training

    Parameters:
    -----------
    loss_out : numpy.ndarray
        Array of loss output, assumes of form [training, validation]
    filepath : string
        path to save .png file to (include '.png' in filepath)
    FIGWIDTH : int
        width of figure in output file
    FIGHEIGHT : int
        height of figure in output file

    Returns:
    --------
    Saves png file to filepath

    """

    nepoch=loss_out.shape[1] # no. epochs trained over
    
    # Setup output figure and plot training and validation loss functions together
    fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
    ax.plot(np.arange(nepoch)+1,loss_out[0], '-', color='black', linewidth=2, label='Train') # Training
    ax.plot(np.arange(nepoch)+1,loss_out[1], '-', color='gray', linewidth=2, label='Valid') # Validation
    ax.legend(loc="upper right")
    #ax.set_ylim(bottom=mlr_params.numpy().min(), top=mlr_params.numpy().max())
    plt.title('Loss Function')
    plt.savefig(filepath) # saves figure to designated file path (do this before show plot, else will plot nothing to file)
    plt.close('all') # plt.show()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def main():

    #####################################################
    # Section INPUT PARAMETERS
    wlist_txt = str(sys.argv[1]) # Code defining custom loss function weights 'W0001'
    batch_size1 = int(sys.argv[2]) # size of batches e.g. 16, ..., 128
    model_name_list1 = str(sys.argv[3]) # Code defining ML model architectures  e.g. '10cS'
    train_test_case1 = int(sys.argv[4]) # Code defining partitioning of years into trainging vs prediction e.g. 2, 12 
    location1 = str(sys.argv[5]) # Code defining location of tile series data e.g. 'France'
    
    # _____________________________________________________
    # Parameters for data preparation
    train_test_case = train_test_case1 # 1, 12, 2 # Define years and locations of input and target data (function define_traintest_years in rsds_data_prep.py)
    SM_Thresh=10.0 # For method where use rsds/clearsky, when clearsky<SM_Thresh set ratio to ==1 so doesn't blow up due to dividing by small values ('def fn_cleanup_ratio')
    maxFac=1.05 # 1.1, 1.05, 1.02, 1.0 stretch max clearsky total curve by factor as tends to underestimate actual max

    SIDvar='SID'

    # _____________________________________________________
    # Parameters for training the ML model    
    save_model=True # False # Option to store ML model
    batch_size = batch_size1 # 16 # 32 # Set batch size to use when training ML model
    num_epochs = 30 # Set number of epoch loops to use when training ML model
    
    weight_list = rsmlm.get_loss_weights(wlist_txt)

    no_target_direct=False    
    bsize_txt='B'+str(batch_size) # bsize_txt+wlist_txt

    # Setup Output Directories
    pdir = OUT_DIR + 'test'+str(train_test_case)+'_'+ model_name_list1 + '_' + bsize_txt + wlist_txt + '/' + location1  + '/'
    # Make directories if don't already exist
    if not os.path.exists(pdir): os.makedirs(pdir)
    if not os.path.exists(pdir+'timeseries'): os.makedirs(pdir+'timeseries')
    if not os.path.exists(pdir+'ex_dayts'): os.makedirs(pdir+'ex_dayts')
    if not os.path.exists(pdir+'ex_weekts'): os.makedirs(pdir+'ex_weekts')
    if not os.path.exists(pdir+'2dpdfs'): os.makedirs(pdir+'2dpdfs')
    if not os.path.exists(pdir+'1dpdfs'): os.makedirs(pdir+'1dpdfs')

   
    # CHOOSE 1 Model to test
    # ===================
    nLOOP=1 # Test just 1 Model
    model_name_list=[model_name_list1]	# e.g. ['10cS']
    mlist_code=model_name_list1		# e.g. '10cS'
    
    pnum_list=[''] 
    
    # Example list of colours for figures that include multiple ML models on the same graph
    col_list=['red','blue', 'green','orange','yellow','pink','brown','violet','darkblue','darkgreen'] * 2

    # ====================================================


    # _____________________________________________________
    # Split available data into samples for training and testing (split by years)
    yearS_List, yrS_txt, yearS_P_List, yrS_txt_P = rsdp.define_traintest_years(train_test_case=train_test_case)
    # Convert location name into lists of the same length as the number of years lists
    site_name_txt=location1
    site_name_List=[location1] * len(yearS_List)
    site_name_P_txt=location1
    site_name_P_List=[location1] * len(yearS_P_List)
    print(train_test_case)

    # _____________________________________________________
    # Read in sample data into numpy arrays: X=input, y=target
    # ---------
    # data_length_list=[num_years, num_days, daylen_short]

    # =========================
    # Read in and prepare TRaining Data (OBS)
    # ---------
    # - INPUT
    OBSxTR_RAWshortTOT, OBSxTR_RAWlongCSKYDIR, OBSxTR_RAWshortCSKYDIR, OBSxTR_shortTOTALnLeap_MAX, OBSxTR_shortCSKYDIRnLeap_MAX, OBSxTR_short_timestamps_nLeap, OBSxTRst_shortCSKYDIRnLeap_MAX, OBSxTRst_short_timestamps_nLeap, OBSxTR_longCSKYDIR_maxday, OBSxTR_data_length_list = rsdp.fn_get_rawinput_data(year_list=yearS_List, site_list=site_name_List, dataset='SARAH3', variable='SIS')
    # ---------
    # - TARGET
    OBSyTR_RAWtargetTOT, OBSyTR_RAWtargetDIR, OBSyTR_targetTOTALnLeap_MAX, OBSyTR_longCSKYDIRnLeap_MAX, OBSyTR_target_timestamps_nLeap, OBSyTRst_longCSKYDIRnLeap_MAX, OBSyTRst_target_timestamps_nLeap, OBSyTR_data_length_list = rsdp.fn_get_rawtarget_data(year_list=yearS_List, site_list=site_name_List, dataset='SARAH3', variable2='SID')
    # ==========================
    # Compute Climatology & Apply Normalisation
    # - INPUT
    numTR_days=OBSxTR_data_length_list[1]
    numTR_years=OBSxTR_data_length_list[0]
    # Compute Climatology clearsky total curve on TRaining data
    OBSxTR3HR_rsds_total_max_est, OBSxTR3HR_MLRmx_params = rsdp.fn_compute_smoothed_clim(OBSxTR_shortCSKYDIRnLeap_MAX, OBSxTR_short_timestamps_nLeap, sample_clim=OBSxTR_shortTOTALnLeap_MAX, TrainApply=True, GTPYS=True)
    # Stretch clearysky total curve by *maxFac (given in input section of main) as sample max likely underestimates real max due to small number of years in dataset
    #  (this is applied after compute max curve MLR parameters so need to reapply if remake curve with parameters!)
    OBSxTR3HR_rsds_total_max_est[0]=OBSxTR3HR_rsds_total_max_est[0]*maxFac
    OBSxTR3HR_rsds_total_max_est[1]=OBSxTR3HR_rsds_total_max_est[1]*maxFac
    # Compute Climatology clearsky total curve using same function but applied to Standard timesteps 00:00, 00:30, ...
    ST_OBSxTR3HR_rsds_total_max_est, tmp_params = rsdp.fn_compute_smoothed_clim(OBSxTRst_shortCSKYDIRnLeap_MAX, OBSxTRst_short_timestamps_nLeap, ntsteps=8, GiveParams=OBSxTR3HR_MLRmx_params, TrainApply=False)
    ST_OBSxTR3HR_rsds_total_max_est[0]=ST_OBSxTR3HR_rsds_total_max_est[0]*maxFac
    ST_OBSxTR3HR_rsds_total_max_est[1]=ST_OBSxTR3HR_rsds_total_max_est[1]*maxFac
    # Correct offset timestep towards Standard timestep 00:00, 00:30, ... : *(CSKYTOT(Standard time)/CSKTOT(Offset time)
    TShFact3HR=[]
    TShFact3HR.append(rsdp.fn_compute_clean_ratio(ST_OBSxTR3HR_rsds_total_max_est[0], OBSxTR3HR_rsds_total_max_est[0], SM_Thresh=10.0, set_num=1.0))
    TShFact3HR.append(rsdp.fn_compute_clean_ratio(ST_OBSxTR3HR_rsds_total_max_est[1], OBSxTR3HR_rsds_total_max_est[1], SM_Thresh=10.0, set_num=1.0))
    TR_TShFact3HR=rsdp.fn_compute_matched_clim(yearS_List, TShFact3HR[0], TShFact3HR[1])
    OBSxTR_RAWshortTOT=OBSxTR_RAWshortTOT*TR_TShFact3HR    
    # Apply Normalisation using TRaining data : Total/ClearskyTotal, ClearskyDirect/MaxDayCSKYDIR
    OBSxTR_NmFactor_TOT, OBSxTR_NmFactor_DIR = rsdp.fn_get_input_NmFactors(ST_OBSxTR3HR_rsds_total_max_est, OBSxTR_longCSKYDIR_maxday, num_days=OBSxTR_RAWshortTOT.shape[0], year_list=yearS_List)
    OBSxTR_NMshortTOT, OBSxTR_NMlongCSKYDIR = rsdp.fn_normalise_input(OBSxTR_RAWshortTOT, OBSxTR_RAWlongCSKYDIR, SM_Thresh=SM_Thresh, NmFactor_TOTAL=OBSxTR_NmFactor_TOT, NmFactor_DIRECT=OBSxTR_NmFactor_DIR)
    # ---------
    # Compute Climatology & Apply Normalisation
    # - TARGET
    # Compute Climatology clearsky total curve on TRaining data
    OBSyTR30MIN_rsds_total_max_est, OBSyTR30MIN_MLRmx_params = rsdp.fn_compute_smoothed_clim(OBSyTR_longCSKYDIRnLeap_MAX, OBSyTR_target_timestamps_nLeap, sample_clim=OBSyTR_targetTOTALnLeap_MAX, ntsteps=48, TrainApply=True, GTPYS=True)
    # Stretch clearysky total curve by *maxFac (given in input section of main) as sample max likely underestimates real max due to small number of years in dataset
    #  (this is applied after compute max curve MLR parameters so need to reapply if remake curve with parameters!)
    OBSyTR30MIN_rsds_total_max_est[0]=OBSyTR30MIN_rsds_total_max_est[0]*maxFac
    OBSyTR30MIN_rsds_total_max_est[1]=OBSyTR30MIN_rsds_total_max_est[1]*maxFac    
    # Compute Climatology clearsky total curve using same function but applied to Standard timesteps 00:00, 00:30, ...
    ST_OBSyTR30MIN_rsds_total_max_est, tmp_params = rsdp.fn_compute_smoothed_clim(OBSyTRst_longCSKYDIRnLeap_MAX, OBSyTRst_target_timestamps_nLeap, ntsteps=48, GiveParams=OBSyTR30MIN_MLRmx_params, TrainApply=False)
    ST_OBSyTR30MIN_rsds_total_max_est[0]=ST_OBSyTR30MIN_rsds_total_max_est[0]*maxFac
    ST_OBSyTR30MIN_rsds_total_max_est[1]=ST_OBSyTR30MIN_rsds_total_max_est[1]*maxFac
    # Correct offset timestep towards Standard timestep 00:00, 00:30, ... *(CSKYTOT(Standard time)/CSKTOT(Offset time)
    TShFact30MIN=[]
    TShFact30MIN.append(rsdp.fn_compute_clean_ratio(ST_OBSyTR30MIN_rsds_total_max_est[0], OBSyTR30MIN_rsds_total_max_est[0], SM_Thresh=10.0, set_num=1.0))
    TShFact30MIN.append(rsdp.fn_compute_clean_ratio(ST_OBSyTR30MIN_rsds_total_max_est[1], OBSyTR30MIN_rsds_total_max_est[1], SM_Thresh=10.0, set_num=1.0))
    TR_TShFact30MIN=rsdp.fn_compute_matched_clim(yearS_List, TShFact30MIN[0], TShFact30MIN[1])
    OBSyTR_RAWtargetTOT=OBSyTR_RAWtargetTOT*TR_TShFact30MIN
    OBSyTR_RAWtargetDIR=OBSyTR_RAWtargetDIR*TR_TShFact30MIN
    # Apply Normalisation using TRaining data : Total/ClearskyTotal, Direct/Total
    OBSyTR_NmFactor_TOT = rsdp.fn_get_target_NmFactors(ST_OBSyTR30MIN_rsds_total_max_est, num_days=OBSyTR_RAWtargetTOT.shape[0], year_list=yearS_List)
    OBSyTR_NMtargetTOT, OBSyTR_NMtargetDIR = rsdp.fn_normalise_target(OBSyTR_RAWtargetTOT, OBSyTR_RAWtargetDIR, SM_Thresh=SM_Thresh, NmFactor_TOTAL=OBSyTR_NmFactor_TOT)
    # ========================= # CHECKED sample and seem identical to previous version

    # Store year info used in TRaining data
    yearS_LenList=[]
    for yearS in yearS_List: yearS_LenList.append(rsdp.get_yr_length(yearS))
    yearS_LenList0=[]
    for yy in range(numTR_years): yearS_LenList0.append(sum(yearS_LenList[0:yy]))
    yearS_LenList1=[]
    for yy in range(numTR_years): yearS_LenList1.append(sum(yearS_LenList[0:yy+1]))

    print('loaded training data')
    # ====================================================
    


    # =========================
    # Read in and prepare PRediction Data (OBS SARAH)
    # - Pysolar is an algorithm so can be computed for any Prediction period
    # - Max rsds is based on the estimate from the TRaining data, but with year lengths matched to the PRediction data
    # - Can also compute mean rsds based on the estimate from the TRaining data, but with year lengths matched to the PRediction data
    # ---------
    # data_length_list=[num_years, num_days, daylen_short]

    # =========================
    # Read in and prepare PRediction Data (OBS) For comparison of perfect model vs GCM
    # ---------
    # - INPUT
    OBSxPR_RAWshortTOT, OBSxPR_RAWlongCSKYDIR, OBSxPR_RAWshortCSKYDIR, OBSxPR_shortTOTALnLeap_MAX, OBSxPR_shortCSKYDIRnLeap_MAX, OBSxPR_short_timestamps_nLeap, OBSxPRst_shortCSKYDIRnLeap_MAX, OBSxPRst_short_timestamps_nLeap, OBSxPR_longCSKYDIR_maxday, OBSxPR_data_length_list = rsdp.fn_get_rawinput_data(year_list=yearS_P_List, site_list=site_name_P_List, dataset='SARAH3', variable='SIS')
    # ---------
    # - TARGET
    OBSyPR_RAWtargetTOT, OBSyPR_RAWtargetDIR, OBSyPR_targetTOTALnLeap_MAX, OBSyPR_longCSKYDIRnLeap_MAX, OBSyPR_target_timestamps_nLeap, OBSyPRst_longCSKYDIRnLeap_MAX, OBSyPRst_target_timestamps_nLeap, OBSyPR_data_length_list = rsdp.fn_get_rawtarget_data(year_list=yearS_P_List, site_list=site_name_P_List, dataset='SARAH3', variable2='SID')
    # ==========================
    # - INPUT
    numPR_days=OBSxPR_data_length_list[1]
    numPR_years=OBSxPR_data_length_list[0]  
    # Correct offset timestep towards Standard timestep 00:00, 00:30, ... *(CSKYTOT(Standard time)/CSKTOT(Offset time)
    PR_TShFact3HR=rsdp.fn_compute_matched_clim(yearS_P_List, TShFact3HR[0], TShFact3HR[1]) # Correction computed on TRaining data
    OBSxPR_RAWshortTOT=OBSxPR_RAWshortTOT*PR_TShFact3HR    
    # Apply Normalisation using TRaining data : Total/ClearskyTotal, ClearskyDirect/MaxDayCSKYDIR
    OBSxPR_RAWshortCSKYTOT = rsdp.fn_compute_matched_clim(yearS_P_List, ST_OBSxTR3HR_rsds_total_max_est[0], ST_OBSxTR3HR_rsds_total_max_est[1]) # TRaining Clearsky Total matched to PR years
    OBSxPR_NmFactor_TOT, OBSxPR_NmFactor_DIR = rsdp.fn_get_input_NmFactors(ST_OBSxTR3HR_rsds_total_max_est, OBSxTR_longCSKYDIR_maxday, num_days=OBSxPR_RAWshortTOT.shape[0], year_list=yearS_P_List) # Daily matched clearsky direct OR TR climatologies
    OBSxPR_NMshortTOT, OBSxPR_NMlongCSKYDIR = rsdp.fn_normalise_input(OBSxPR_RAWshortTOT, OBSxPR_RAWlongCSKYDIR, SM_Thresh=SM_Thresh, NmFactor_TOTAL=OBSxPR_NmFactor_TOT, NmFactor_DIRECT=OBSxPR_NmFactor_DIR)
    # ---------
    # - TARGET
    # Correct offset timestep towards Standard timestep 00:00, 00:30, ... *(CSKYTOT(Standard time)/CSKTOT(Offset time)
    PR_TShFact30MIN=rsdp.fn_compute_matched_clim(yearS_P_List, TShFact30MIN[0], TShFact30MIN[1])
    OBSyPR_RAWtargetTOT=OBSyPR_RAWtargetTOT*PR_TShFact30MIN
    OBSyPR_RAWtargetDIR=OBSyPR_RAWtargetDIR*PR_TShFact30MIN
    ST_OBSyPR_targetTOTALnLeap_MAX=OBSyPR_targetTOTALnLeap_MAX*TShFact30MIN[0]
    # Apply Normalisation using TRaining data : Total/ClearskyTotal, ClearskyDirect/MaxDayCSKYDIR
    OBSyPR_RAWlongCSKYTOT = rsdp.fn_compute_matched_clim(yearS_P_List, ST_OBSyTR30MIN_rsds_total_max_est[0], ST_OBSyTR30MIN_rsds_total_max_est[1]) # TRaining Clearsky Total matched to PR years (smoothed)
    OBSyPR_NmFactor_TOT = rsdp.fn_get_target_NmFactors(ST_OBSyTR30MIN_rsds_total_max_est, num_days=OBSyPR_RAWtargetTOT.shape[0], year_list=yearS_P_List)
    OBSyPR_NMtargetTOT, OBSyPR_NMtargetDIR = rsdp.fn_normalise_target(OBSyPR_RAWtargetTOT, OBSyPR_RAWtargetDIR, SM_Thresh=SM_Thresh, NmFactor_TOTAL=OBSyPR_NmFactor_TOT)
    # =========================
    
    # Store year info used in PRediction data
    yearP_LenList=[]
    for yearP in yearS_P_List: yearP_LenList.append(rsdp.get_yr_length(yearP))
    yearP_LenList0=[]
    for yy in range(numPR_years): yearP_LenList0.append(sum(yearP_LenList[0:yy]))
    yearP_LenList1=[]
    for yy in range(numPR_years): yearP_LenList1.append(sum(yearP_LenList[0:yy+1]))
    
    print('loaded prediction data')
    # ====================================================


    # =========================
    fsetup_txt='_SARAH3_'+mlist_code+'_e'+str(num_epochs)+bsize_txt+wlist_txt+'_TR'+yrS_txt+site_name_txt+'_PR'+'_'+yrS_txt_P+site_name_P_txt
    # =========================

    # _____________________________________________________
    # Prepare Data
    #
    # --- TRAINING DATA
    # Split numpy data into training and validation subsets
    # - Don't shuffle (Default is to first shuffle the data unless set shuffle=False)
    # - test_size => proportion of data in validation set
    # - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    dindex=np.arange(numTR_days)
    test_size=1/numTR_years 	# So validation part is roughly equal to a single year
    OBSx_shortTOT_train, OBSx_shortTOT_val, OBSx_longCSKY_train, OBSx_longCSKY_val, OBSy_targetTOT_train, OBSy_targetTOT_val, OBSy_targetDIR_train, OBSy_targetDIR_val, dindex_train, dindex_val = train_test_split(OBSxTR_NMshortTOT, OBSxTR_NMlongCSKYDIR, OBSyTR_NMtargetTOT, OBSyTR_NMtargetDIR, dindex, test_size=test_size, random_state=42, shuffle=False) # , shuffle=True
    
    # CONVERT TRaining data NumPy arrays to PyTorch tensors
    OBSx_shortTOT_train_tn=torch.from_numpy(OBSx_shortTOT_train).float()
    OBSx_longCSKY_train_tn=torch.from_numpy(OBSx_longCSKY_train).float()
    OBSy_targetTOT_train_tn=torch.from_numpy(OBSy_targetTOT_train).float()
    OBSy_targetDIR_train_tn=torch.from_numpy(OBSy_targetDIR_train).float()
    OBSx_shortTOT_val_tn=torch.from_numpy(OBSx_shortTOT_val).float()
    OBSx_longCSKY_val_tn=torch.from_numpy(OBSx_longCSKY_val).float()
    OBSy_targetTOT_val_tn=torch.from_numpy(OBSy_targetTOT_val).float()
    OBSy_targetDIR_val_tn=torch.from_numpy(OBSy_targetDIR_val).float()

    # Create dataset for training model
    train_dataset = TensorDataset(OBSx_shortTOT_train_tn, OBSx_longCSKY_train_tn, OBSy_targetTOT_train_tn, OBSy_targetDIR_train_tn)
    valid_dataset = TensorDataset(OBSx_shortTOT_val_tn, OBSx_longCSKY_val_tn, OBSy_targetTOT_val_tn, OBSy_targetDIR_val_tn)

    # Create data loaders for training model
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    
    # --- PREDICTION on TESTING DATA
    # CONVERT PRediction data NumPy arrays to PyTorch tensors
    OBSx_shortTOT_pred_tn=torch.from_numpy(OBSxPR_NMshortTOT).float()
    OBSx_longCSKY_pred_tn=torch.from_numpy(OBSxPR_NMlongCSKYDIR).float()
    OBSy_targetTOT_pred_tn=torch.from_numpy(OBSyPR_NMtargetTOT).float()
    OBSy_targetDIR_pred_tn=torch.from_numpy(OBSyPR_NMtargetDIR).float()
   
    print('prepared data')
    # ====================================================

    # _____________________________________________________
    # TRAIN Standard Multiple Linear Regression (MLR) and Machine Learning (ML) models on TRaining data (OBS)
    # AND Apply models to PRediction testing data

    # Compute simple standard MSE Loss Function on scaled TRaining data
    criterion = nn.MSELoss()

    # _____________________________________________________
    # TRAIN Multiple Linear Regression (MLR) models with LSE loss function on TRaining data (OBS)
    # - Use training part of [training, validation] split so same as for the ML models
    # - Individual timesteps and TOT and DIR treated as independent (Params are the same whether cat TOT and DIR or compute separately as targets) 
    # - Need to include a ONEs column so can have a non-zero intercept

    # * MLR With 3hr rsds as input (mlr8 as 8 input variables)
    ones=OBSx_shortTOT_train_tn[:,0:1]*0.0+1.0
    OBScombined_featuresTR_8X = torch.cat([OBSx_shortTOT_train_tn,ones], dim=1)
    OBScombined_featuresTR_8Y = torch.cat([OBSy_targetTOT_train_tn,OBSy_targetDIR_train_tn], dim=1)
    mlr8_params=rsmlm.Tensor_least_squares_regression(OBScombined_featuresTR_8X, OBScombined_featuresTR_8Y)

    # * MLE With 3hr rsds and 30min clear-sky as input (mlr56 as 8+48=56 input variables)
    OBScombined_featuresTR_56X = torch.cat([OBSx_shortTOT_train_tn,OBSx_longCSKY_train_tn,ones], dim=1)
    OBScombined_featuresTR_56Y = torch.cat([OBSy_targetTOT_train_tn,OBSy_targetDIR_train_tn], dim=1)
    mlr56_params=rsmlm.Tensor_least_squares_regression(OBScombined_featuresTR_56X, OBScombined_featuresTR_56Y)

    # APPLY MLR Model
    with torch.no_grad():
        # Training data (in sample)
        OBStrainingTOTDIRSca_mlr8 = rsmlm.Tensor_LSR_predict(OBScombined_featuresTR_8X, mlr8_params)
        OBStrainingTOTSca_mlr8 = OBStrainingTOTDIRSca_mlr8[:,0:48]
        OBStrainingDIRSca_mlr8 = OBStrainingTOTDIRSca_mlr8[:,48:96]
        OBStrainingTOTDIRSca_mlr56 = rsmlm.Tensor_LSR_predict(OBScombined_featuresTR_56X, mlr56_params)
        OBStrainingTOTSca_mlr56 = OBStrainingTOTDIRSca_mlr56[:,0:48]
        OBStrainingDIRSca_mlr56 = OBStrainingTOTDIRSca_mlr56[:,48:96]

        # Prediction data (out of sample)
        ones=OBSx_shortTOT_pred_tn[:,0:1]*0.0+1.0

        OBScombined_featuresPR_8X = torch.cat([OBSx_shortTOT_pred_tn,ones], dim=1)
        OBScombined_featuresPR_8Y = torch.cat([OBSy_targetTOT_pred_tn,OBSy_targetDIR_pred_tn], dim=1)
        OBSpredictionTOTDIRSca_mlr8 = rsmlm.Tensor_LSR_predict(OBScombined_featuresPR_8X, mlr8_params)
        OBSpredictionTOTSca_mlr8 = OBSpredictionTOTDIRSca_mlr8[:,0:48]
        OBSpredictionDIRSca_mlr8 = OBSpredictionTOTDIRSca_mlr8[:,48:96]

        OBScombined_featuresPR_56X = torch.cat([OBSx_shortTOT_pred_tn,OBSx_longCSKY_pred_tn,ones], dim=1)
        OBScombined_featuresPR_56Y = torch.cat([OBSy_targetTOT_pred_tn,OBSy_targetDIR_pred_tn], dim=1)
        OBSpredictionTOTDIRSca_mlr56 = rsmlm.Tensor_LSR_predict(OBScombined_featuresPR_56X, mlr56_params)
        OBSpredictionTOTSca_mlr56 = OBSpredictionTOTDIRSca_mlr56[:,0:48]
        OBSpredictionDIRSca_mlr56 = OBSpredictionTOTDIRSca_mlr56[:,48:96]

    # Compute simple MSE loss
    OBSTR_lossTOT_mlr8 = criterion(OBStrainingTOTSca_mlr8,OBSy_targetTOT_train_tn)
    OBSTR_lossDIR_mlr8 = criterion(OBStrainingDIRSca_mlr8,OBSy_targetDIR_train_tn)
    OBSTR_loss_mlr8 = OBSTR_lossTOT_mlr8 + OBSTR_lossDIR_mlr8
    OBSTR_lossTOT_mlr56 = criterion(OBStrainingTOTSca_mlr56,OBSy_targetTOT_train_tn)
    OBSTR_lossDIR_mlr56 = criterion(OBStrainingDIRSca_mlr56,OBSy_targetDIR_train_tn)
    OBSTR_loss_mlr56 = OBSTR_lossTOT_mlr56 + OBSTR_lossDIR_mlr56

    print('computed MLR models')

    # _____________________________________________________
    # TRAIN ML model(s) on TRaining data with MSE based custom loss function
    # AND Apply ML models to PRediction testing data

    # Setup lists to store ML model output
    NO_MLmodels=1
    trained_model_list=[]

    OBStrainingTOTSca_MLlist=[]
    OBStrainingDIRSca_MLlist=[]
    OBSTR_lossTOT_MLlist=[]
    OBSTR_lossDIR_MLlist=[]
    OBSTR_loss_MLlist=[]
    
    OBSpredictionTOTSca_MLlist=[]
    OBSpredictionDIRSca_MLlist=[]
    
    mcount=0
    # -----------
    # TRAIN Model
    ml_model_name = model_name_list[0]
    model_ml=rsmlm.get_ml_model(ml_model_name)
    weight_list = rsmlm.get_loss_weights(wlist_txt)

    # ============
    # Train model (trained model) and store loss output (loss_out) for complete and partial elements of loss function for each epoch
    #   TOT+DIR Training Loss; TOT+DIR Validation Loss; TOT Training Loss; TOT Validation Loss; DIR Training Loss; DIR Validation Loss

    trained_model, loss_out = rsmlm.train_model(model_ml, train_loader, valid_loader, epochs=num_epochs, Wlist=weight_list)

    print('final_loss')
    print('TR TOT+DIR')
    for lcount in range(2): print(loss_out[lcount,num_epochs-1]) 
    print('TR TOT')
    for lcount in range(2): print(loss_out[lcount+2,num_epochs-1])
    print('TR DIR')
    for lcount in range(2): print(loss_out[lcount+4,num_epochs-1])
    # ============

    # Plot Loss Functions (separately and average of MSE for Total and MSE for Direct)
    filename=pdir+'rsds'+fsetup_txt+'_ML'+model_name_list[0]+'_LossFn'+pnum_list[0]
    plot_loss(loss_out[0:2,:], filepath=filename+'.png')
    filename=pdir+'rsdsTOT'+fsetup_txt+'_ML'+model_name_list[0]+'_LossFn'+pnum_list[0]
    plot_loss(loss_out[2:4,:], filepath=filename+'.png')
    filename=pdir+'rsdsDIR'+fsetup_txt+'_ML'+model_name_list[0]+'_LossFn'+pnum_list[0]
    plot_loss(loss_out[4:6,:], filepath=filename+'.png')

    # Save ML Model
    filename=pdir+'rsdsDIR'+fsetup_txt+'_ML'+model_name_list[0]+'_model'+pnum_list[0]
    if save_model: torch.save(trained_model.state_dict(), filename+'.pth') # Optional: save the model

    ## To look at weights of model, pause here and output:
    # pdb.set_trace()
    ## trained_model.decoderTOT[0].weight[30].detach().numpy()
    ## trained_model.decoderDIR[0].weight[30].detach().numpy()
    ## mlr56_params[:,48+30]

    # -----------
    # APPLY ML Model (trained_model) to TRaining and PRediction data
    trained_model.eval()
    with torch.no_grad():
        # TRaining data (in sample)
        # ============
        OBStrainingTOT_test, OBStrainingDIR_test = trained_model(OBSx_shortTOT_train_tn, OBSx_longCSKY_train_tn) # torch.Size([no. days, 48])
        # ============
        OBStrainingTOTSca_MLlist.append(OBStrainingTOT_test)
        OBStrainingDIRSca_MLlist.append(OBStrainingDIR_test)
        OBSTR_lossTOT_MLlist.append(criterion(OBStrainingTOT_test,OBSy_targetTOT_train_tn))
        OBSTR_lossDIR_MLlist.append(criterion(OBStrainingDIR_test,OBSy_targetDIR_train_tn))
        OBSTR_loss_MLlist.append(criterion(OBStrainingTOT_test,OBSy_targetTOT_train_tn)+criterion(OBStrainingDIR_test,OBSy_targetDIR_train_tn))
        # ============

        # PRediction data (out of sample)
        # ============
        OBSpredictionTOT_test, OBSpredictionDIR_test = trained_model(OBSx_shortTOT_pred_tn, OBSx_longCSKY_pred_tn) # torch.Size([no. days, 48])
        # ============
        OBSpredictionTOTSca_MLlist.append(OBSpredictionTOT_test)
        OBSpredictionDIRSca_MLlist.append(OBSpredictionDIR_test)

    print('computed ML')
    
    # ======================================
    # ASSESS PERFORMANCE of ML(R) models trained on OBServations
    #
    # Output MSE Loss Function type statistic for ML and MLR on scaled OBS so can compare to MLR
    # - torch.Tensor 
    
    # Write to text file
    #-------------------------

    mcount=0
    #-------------------------
    filename=pdir+'rsdsTOTnDIR'+fsetup_txt+'_TR_LossFn_TRPR_RMSE.txt'
    f = open(filename, 'w')

    # TOTAL
    #-------------------------
    col_heads='TOT              TR Loss   TR RMSE   PR RMSE'
    f.write(col_heads),
    f.write("\n"),
    f.write("%14s" % "MLR8"),
    f.write("%10.5f" % OBSTR_lossTOT_mlr8),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBStrainingTOTSca_mlr8, OBSy_targetTOT_train_tn)),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBSpredictionTOTSca_mlr8, OBSy_targetTOT_pred_tn)),
    f.write("\n"),
    f.write("%14s" % "MLR56"),
    f.write("%10.5f" % OBSTR_lossTOT_mlr56),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBStrainingTOTSca_mlr56, OBSy_targetTOT_train_tn)),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBSpredictionTOTSca_mlr56, OBSy_targetTOT_pred_tn)),
    f.write("\n"),
    f.write("%14s" % (wlist_txt+' ML '+model_name_list[0])),
    f.write("%10.5f" % OBSTR_lossTOT_MLlist[0]),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBStrainingTOTSca_MLlist[0], OBSy_targetTOT_train_tn)),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBSpredictionTOTSca_MLlist[0], OBSy_targetTOT_pred_tn)),
    f.write("\n"),

    # DIRECT
    #-------------------------
    col_heads='DIR              TR Loss   TR RMSE   PR RMSE'
    f.write(col_heads),
    f.write("\n"),
    f.write("%14s" % "MLR8"),
    f.write("%10.5f" % OBSTR_lossDIR_mlr8),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBStrainingDIRSca_mlr8, OBSy_targetDIR_train_tn)),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBSpredictionDIRSca_mlr8, OBSy_targetDIR_pred_tn)),
    f.write("\n"),
    f.write("%14s" % "MLR56"),
    f.write("%10.5f" % OBSTR_lossDIR_mlr56),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBStrainingDIRSca_mlr56, OBSy_targetDIR_train_tn)),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBSpredictionDIRSca_mlr56, OBSy_targetDIR_pred_tn)),
    f.write("\n"),
    f.write("%14s" % (wlist_txt+' ML '+model_name_list[0])),
    f.write("%10.5f" % OBSTR_lossDIR_MLlist[0]),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBStrainingDIRSca_MLlist[0], OBSy_targetDIR_train_tn)),
    f.write("%10.5f" % rspm.RMSE_NPARRAY(OBSpredictionDIRSca_MLlist[0], OBSy_targetDIR_pred_tn)),
    f.write("\n"),
    f.close()
    # ======================================


    # _____________________________________________________
    # ASSESS performance of predictions
    # Revert back to original rsds units (W/m2)
    # 
    # rsdsTOT/clearskyTotal; rsdsDIR/MaxDay (find day with max rsdsDIR and divide all days by this)

    OBSxTR_longCSKYDIR_wm2=OBSxTR_NMlongCSKYDIR.copy() # numpy.ndarray
    OBSxPR_longCSKYDIR_wm2=OBSxPR_NMlongCSKYDIR.copy() # numpy.ndarray

    # Revert clearsky direct back to w/m2 units
    for dcount in range(OBSxTR_longCSKYDIR_wm2.shape[0]): OBSxTR_longCSKYDIR_wm2[dcount]=OBSxTR_NMlongCSKYDIR[dcount]*OBSxTR_longCSKYDIR_maxday
    for dcount in range(OBSxPR_longCSKYDIR_wm2.shape[0]): OBSxPR_longCSKYDIR_wm2[dcount]=OBSxPR_NMlongCSKYDIR[dcount]*OBSxPR_longCSKYDIR_maxday
    
    # Un-normalisation of data
    #   PR means data has same length and is matched to the PR dates/years
    CLIMxPR_shortMxTOT=OBSxPR_RAWshortCSKYTOT.copy()	# TR climatology of Clearsky Total (smooth estimate)
    CLIMxPR_longCSKY=OBSxPR_longCSKYDIR_wm2.copy()	# Actual clearsky direct for PRediction period (matched) (Un-normalised)
    CLIMyPR_longMxTOT=OBSyPR_RAWlongCSKYTOT.copy()	# TR climatology of Clearsky Total (smooth estimate)

    # Total
    OBSpredictionTOT_MLlist_wm2=[]
    OBSxPR_shortTOT_wm2=rsdp.revertTOT_to_wm2(OBSxPR_NMshortTOT, CLIMxPR_shortMxTOT)
    OBSyPR_targetTOT_wm2=rsdp.revertTOT_to_wm2(OBSyPR_NMtargetTOT, CLIMyPR_longMxTOT)
    OBSpredictionTOT_mlr8_wm2=rsdp.revertTOT_to_wm2(OBSpredictionTOTSca_mlr8, CLIMyPR_longMxTOT)
    OBSpredictionTOT_mlr56_wm2=rsdp.revertTOT_to_wm2(OBSpredictionTOTSca_mlr56, CLIMyPR_longMxTOT)
    OBSpredictionTOT_MLlist_wm2.append(rsdp.revertTOT_to_wm2(OBSpredictionTOTSca_MLlist[0], CLIMyPR_longMxTOT))

    # Direct
    OBSpredictionDIR_MLlist_wm2=[]
    OBSyPR_targetDIR_wm2=rsdp.revertDIR_to_wm2(OBSyPR_NMtargetDIR, OBSyPR_targetTOT_wm2)
    OBSpredictionDIR_mlr8_wm2=rsdp.revertDIR_to_wm2(OBSpredictionDIRSca_mlr8, OBSpredictionTOT_mlr8_wm2)
    OBSpredictionDIR_mlr56_wm2=rsdp.revertDIR_to_wm2(OBSpredictionDIRSca_mlr56, OBSpredictionTOT_mlr56_wm2)
    OBSpredictionDIR_MLlist_wm2.append(rsdp.revertDIR_to_wm2(OBSpredictionDIRSca_MLlist[0],OBSpredictionTOT_MLlist_wm2[0]))



    # ======================================
    # Store prediction of TOT and DIR output as csv files (1 per year as for input)
    # - on standard timesteps 
    # - OBS Target
    # - ML Model output
    # - MLR Model output
    # index=True has index column; index=False prevents an index column from being added.
    ycount=0
    for yrpred in yearS_P_List:
        start_date=datetime.datetime(yrpred, 1, 1, 0, 0)
        end_date=datetime.datetime(yrpred, 12, 31, 23, 30)
        date_list = pd.date_range(start_date, end_date, freq='30min')
        yrpred_len=365
        if calendar.isleap(yrpred_len): yrpred_len=366

        TOTtarget=OBSyPR_targetTOT_wm2[yearP_LenList0[ycount]:yearP_LenList1[ycount],:].reshape(yrpred_len*48)
        dataset = pd.DataFrame({'SIS': TOTtarget}, index=date_list)
        dataset.to_csv(pdir+'timeseries/'+'SARAH3_DATEst_SIS_30MIN_'+str(yrpred)+'_'+location1+'.csv', index=True, index_label='time', float_format='%6.1f')

        DIRtarget=OBSyPR_targetDIR_wm2[yearP_LenList0[ycount]:yearP_LenList1[ycount],:].reshape(yrpred_len*48)
        dataset = pd.DataFrame({'SID': DIRtarget}, index=date_list)
        dataset.to_csv(pdir+'timeseries/'+'SARAH3_DATEst_SID_30MIN_'+str(yrpred)+'_'+location1+'.csv', index=True, index_label='time', float_format='%6.1f')

        TOTmloutput=OBSpredictionTOT_MLlist_wm2[0][yearP_LenList0[ycount]:yearP_LenList1[ycount],:].reshape(yrpred_len*48)
        dataset = pd.DataFrame({'SIS': TOTmloutput}, index=date_list)
        dataset.to_csv(pdir+'timeseries/'+'ML'+mlist_code+'_DATEst_SIS_30MIN_'+str(yrpred)+'_'+location1+'.csv', index=True, index_label='time', float_format='%6.1f')

        DIRmloutput=OBSpredictionDIR_MLlist_wm2[0][yearP_LenList0[ycount]:yearP_LenList1[ycount],:].reshape(yrpred_len*48)
        dataset = pd.DataFrame({'SID': DIRmloutput}, index=date_list)
        dataset.to_csv(pdir+'timeseries/'+'ML'+mlist_code+'_DATEst_SID_30MIN_'+str(yrpred)+'_'+location1+'.csv', index=True, index_label='time', float_format='%6.1f')

        TOTmlroutput=OBSpredictionTOT_mlr56_wm2[yearP_LenList0[ycount]:yearP_LenList1[ycount],:].reshape(yrpred_len*48)
        dataset = pd.DataFrame({'SIS': TOTmlroutput}, index=date_list)
        dataset.to_csv(pdir+'timeseries/'+'MLR56_DATEst_SIS_30MIN_'+str(yrpred)+'_'+location1+'.csv', index=True, index_label='time', float_format='%6.1f')

        DIRmlroutput=OBSpredictionDIR_mlr56_wm2[yearP_LenList0[ycount]:yearP_LenList1[ycount],:].reshape(yrpred_len*48)
        dataset = pd.DataFrame({'SID': DIRmlroutput}, index=date_list)
        dataset.to_csv(pdir+'timeseries/'+'MLR56_DATEst_SID_30MIN_'+str(yrpred)+'_'+location1+'.csv', index=True, index_label='time', float_format='%6.1f')

        ycount=ycount+1



    # ======================================
    # ASSESS PREDICTION PERFORMANCE of ML(R) models (trained on OBServations)
    #
    # SD RMSE WasDistance
    RowHeadingListMN=["Target", "MLR56", "MLR8", wlist_txt+' ML '+model_name_list[0]]
    ColHeadingList1=["CLM SD(R-MR56)", "CLM RMSE(RSDS)", "CLM WD(RSDS)", "DSD RMSE(R-MR56)", "DSD WD(R-MR56)", "DSD RMSE(RSDS)", "DSD WD(RSDS)"]
    
    # ---------------------------------------
    # TOTAL 
    # -------------
    TOT_LIST1=[]
    mcount=0
    # SD CLIM(TOT-MLR56) [Target, nan, MLR8, ML MEAN]
    rsdsTOTmMLR56_AMEAN_SD=[]
    rsdsTOTmMLR56_AMEAN_SD.append((OBSyPR_targetTOT_wm2.mean(axis=0)-OBSpredictionTOT_mlr56_wm2.mean(axis=0)).std())
    rsdsTOTmMLR56_AMEAN_SD.append(np.nan)
    rsdsTOTmMLR56_AMEAN_SD.append((OBSpredictionTOT_mlr8_wm2.mean(axis=0)-OBSpredictionTOT_mlr56_wm2.mean(axis=0)).std())
    rsdsTOTmMLR56_AMEAN_SD.append((OBSpredictionTOT_MLlist_wm2[0].mean(axis=0)-OBSpredictionTOT_mlr56_wm2.mean(axis=0)).std())
    TOT_LIST1.append(rsdsTOTmMLR56_AMEAN_SD)

    # RMSE CLIM(TOT,Target) [nan, MLR56, MLR8, ML MEAN]
    rsdsTOT_AMEAN_RMSE=[]
    rsdsTOT_AMEAN_RMSE.append(np.nan)
    rsdsTOT_AMEAN_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionTOT_mlr56_wm2.mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)))
    rsdsTOT_AMEAN_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionTOT_mlr8_wm2.mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)))
    rsdsTOT_AMEAN_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionTOT_MLlist_wm2[0].mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)))
    TOT_LIST1.append(rsdsTOT_AMEAN_RMSE)
    
    # WD CLIM(TOT,Target) [nan, MLR56, MLR8, ML MEAN]
    rsdsTOT_AMEAN_WD=[]
    rsdsTOT_AMEAN_WD.append(np.nan)
    rsdsTOT_AMEAN_WD.append(sp.stats.wasserstein_distance(OBSpredictionTOT_mlr56_wm2.mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)))
    rsdsTOT_AMEAN_WD.append(sp.stats.wasserstein_distance(OBSpredictionTOT_mlr8_wm2.mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)))
    rsdsTOT_AMEAN_WD.append(sp.stats.wasserstein_distance(OBSpredictionTOT_MLlist_wm2[0].mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)))
    TOT_LIST1.append(rsdsTOT_AMEAN_WD)
    
    # RMSE(DSD(TOT-MLR56),DSD(Target-MLR56)) [nan, nan, MLR8, ML MEAN]
    rsdsTOTmMLR56_DSD_RMSE=[]
    rsdsTOTmMLR56_DSD_RMSE.append(np.nan)
    rsdsTOTmMLR56_DSD_RMSE.append(np.nan)
    rsdsTOTmMLR56_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionTOT_mlr8_wm2-OBSpredictionTOT_mlr56_wm2).std(axis=1), (OBSyPR_targetTOT_wm2-OBSpredictionTOT_mlr56_wm2).std(axis=1)))
    rsdsTOTmMLR56_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionTOT_MLlist_wm2[0]-OBSpredictionTOT_mlr56_wm2).std(axis=1), (OBSyPR_targetTOT_wm2-OBSpredictionTOT_mlr56_wm2).std(axis=1)))
    TOT_LIST1.append(rsdsTOTmMLR56_DSD_RMSE)

    # WD(DSD(TOT-MLR56),DSD(Target-MLR56)) [nan, nan, MLR8, ML MEAN]
    rsdsTOTmMLR56_DSD_WD=[]
    rsdsTOTmMLR56_DSD_WD.append(np.nan)
    rsdsTOTmMLR56_DSD_WD.append(np.nan)
    rsdsTOTmMLR56_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionTOT_mlr8_wm2-OBSpredictionTOT_mlr56_wm2).std(axis=1), (OBSyPR_targetTOT_wm2-OBSpredictionTOT_mlr56_wm2).std(axis=1)))
    rsdsTOTmMLR56_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionTOT_MLlist_wm2[0]-OBSpredictionTOT_mlr56_wm2).std(axis=1), (OBSyPR_targetTOT_wm2-OBSpredictionTOT_mlr56_wm2).std(axis=1)))
    TOT_LIST1.append(rsdsTOTmMLR56_DSD_WD)
    
    # RMSE(DSD(TOT),DSD(Target)) [nan, MLR56, MLR8, ML MEAN]
    rsdsTOT_DSD_RMSE=[]
    rsdsTOT_DSD_RMSE.append(np.nan)
    rsdsTOT_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionTOT_mlr56_wm2).std(axis=1), (OBSyPR_targetTOT_wm2).std(axis=1)))
    rsdsTOT_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionTOT_mlr8_wm2).std(axis=1), (OBSyPR_targetTOT_wm2).std(axis=1)))
    rsdsTOT_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionTOT_MLlist_wm2[0]).std(axis=1), (OBSyPR_targetTOT_wm2).std(axis=1)))
    TOT_LIST1.append(rsdsTOT_DSD_RMSE)

    # WD(DSD(TOT),DSD(Target)) [nan, MLR56, MLR8, ML MEAN]
    rsdsTOT_DSD_WD=[]
    rsdsTOT_DSD_WD.append(np.nan)
    rsdsTOT_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionTOT_mlr56_wm2).std(axis=1), (OBSyPR_targetTOT_wm2).std(axis=1)))
    rsdsTOT_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionTOT_mlr8_wm2).std(axis=1), (OBSyPR_targetTOT_wm2).std(axis=1)))
    rsdsTOT_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionTOT_MLlist_wm2[0]).std(axis=1), (OBSyPR_targetTOT_wm2).std(axis=1)))
    TOT_LIST1.append(rsdsTOT_DSD_WD)

    # ---------------------------------------
    # DIRECT 
    # -------------
    DIR_LIST1=[]
    mcount=0
    # SD CLIM(DIR-MLR56) [Target, nan, MLR8, ML MEAN]
    rsdsDIRmMLR56_AMEAN_SD=[]
    rsdsDIRmMLR56_AMEAN_SD.append((OBSyPR_targetDIR_wm2.mean(axis=0)-OBSpredictionDIR_mlr56_wm2.mean(axis=0)).std())
    rsdsDIRmMLR56_AMEAN_SD.append(np.nan)
    rsdsDIRmMLR56_AMEAN_SD.append((OBSpredictionDIR_mlr8_wm2.mean(axis=0)-OBSpredictionDIR_mlr56_wm2.mean(axis=0)).std())
    rsdsDIRmMLR56_AMEAN_SD.append((OBSpredictionDIR_MLlist_wm2[0].mean(axis=0)-OBSpredictionDIR_mlr56_wm2.mean(axis=0)).std())
    DIR_LIST1.append(rsdsDIRmMLR56_AMEAN_SD)

    # RMSE CLIM(DIR,Target) [nan, MLR56, MLR8, ML MEAN]
    rsdsDIR_AMEAN_RMSE=[]
    rsdsDIR_AMEAN_RMSE.append(np.nan)
    rsdsDIR_AMEAN_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionDIR_mlr56_wm2.mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)))
    rsdsDIR_AMEAN_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionDIR_mlr8_wm2.mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)))
    rsdsDIR_AMEAN_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionDIR_MLlist_wm2[0].mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)))
    DIR_LIST1.append(rsdsDIR_AMEAN_RMSE)
    
    # WD CLIM(DIR,Target) [nan, MLR56, MLR8, ML MEAN]
    rsdsDIR_AMEAN_WD=[]
    rsdsDIR_AMEAN_WD.append(np.nan)
    rsdsDIR_AMEAN_WD.append(sp.stats.wasserstein_distance(OBSpredictionDIR_mlr56_wm2.mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)))
    rsdsDIR_AMEAN_WD.append(sp.stats.wasserstein_distance(OBSpredictionDIR_mlr8_wm2.mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)))
    rsdsDIR_AMEAN_WD.append(sp.stats.wasserstein_distance(OBSpredictionDIR_MLlist_wm2[0].mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)))
    DIR_LIST1.append(rsdsDIR_AMEAN_WD)
    
    # RMSE(DSD(DIR-MLR56),DSD(Target-MLR56)) [nan, nan, MLR8, ML MEAN]
    rsdsDIRmMLR56_DSD_RMSE=[]
    rsdsDIRmMLR56_DSD_RMSE.append(np.nan)
    rsdsDIRmMLR56_DSD_RMSE.append(np.nan)
    rsdsDIRmMLR56_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionDIR_mlr8_wm2-OBSpredictionDIR_mlr56_wm2).std(axis=1), (OBSyPR_targetDIR_wm2-OBSpredictionDIR_mlr56_wm2).std(axis=1)))
    rsdsDIRmMLR56_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionDIR_MLlist_wm2[0]-OBSpredictionDIR_mlr56_wm2).std(axis=1), (OBSyPR_targetDIR_wm2-OBSpredictionDIR_mlr56_wm2).std(axis=1)))
    DIR_LIST1.append(rsdsDIRmMLR56_DSD_RMSE)

    # WD(DSD(DIR-MLR56),DSD(Target-MLR56)) [nan, nan, MLR8, ML MEAN]
    rsdsDIRmMLR56_DSD_WD=[]
    rsdsDIRmMLR56_DSD_WD.append(np.nan)
    rsdsDIRmMLR56_DSD_WD.append(np.nan)
    rsdsDIRmMLR56_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionDIR_mlr8_wm2-OBSpredictionDIR_mlr56_wm2).std(axis=1), (OBSyPR_targetDIR_wm2-OBSpredictionDIR_mlr56_wm2).std(axis=1)))
    rsdsDIRmMLR56_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionDIR_MLlist_wm2[0]-OBSpredictionDIR_mlr56_wm2).std(axis=1), (OBSyPR_targetDIR_wm2-OBSpredictionDIR_mlr56_wm2).std(axis=1)))
    DIR_LIST1.append(rsdsDIRmMLR56_DSD_WD)
    
    # RMSE(DSD(DIR),DSD(Target)) [nan, MLR56, MLR8, ML MEAN]
    rsdsDIR_DSD_RMSE=[]
    rsdsDIR_DSD_RMSE.append(np.nan)
    rsdsDIR_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionDIR_mlr56_wm2).std(axis=1), (OBSyPR_targetDIR_wm2).std(axis=1)))
    rsdsDIR_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionDIR_mlr8_wm2).std(axis=1), (OBSyPR_targetDIR_wm2).std(axis=1)))
    rsdsDIR_DSD_RMSE.append(rspm.RMSE_NPARRAY((OBSpredictionDIR_MLlist_wm2[0]).std(axis=1), (OBSyPR_targetDIR_wm2).std(axis=1)))
    DIR_LIST1.append(rsdsDIR_DSD_RMSE)

    # WD(DSD(DIR),DSD(Target)) [nan, MLR56, MLR8, ML MEAN]
    rsdsDIR_DSD_WD=[]
    rsdsDIR_DSD_WD.append(np.nan)
    rsdsDIR_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionDIR_mlr56_wm2).std(axis=1), (OBSyPR_targetDIR_wm2).std(axis=1)))
    rsdsDIR_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionDIR_mlr8_wm2).std(axis=1), (OBSyPR_targetDIR_wm2).std(axis=1)))
    rsdsDIR_DSD_WD.append(sp.stats.wasserstein_distance((OBSpredictionDIR_MLlist_wm2[0]).std(axis=1), (OBSyPR_targetDIR_wm2).std(axis=1)))
    DIR_LIST1.append(rsdsDIR_DSD_WD)

    # Write to text file
    filename=pdir+'rsdsTOTnDIR'+fsetup_txt+'_PR_PerfMetrics.txt'
    f = open(filename, 'w')
    # Col Headings
    f.write("%20s" % 'TOT'),
    for ccount in range(len(ColHeadingList1)):
        f.write("%18s" % ColHeadingList1[ccount]),
    f.write("\n"),
    for rcount in range(len(RowHeadingListMN)):
        f.write("%20s" % RowHeadingListMN[rcount]),
        for ccount in range(len(ColHeadingList1)):
            f.write("%18.5f" %  TOT_LIST1[ccount][rcount]),
        f.write("\n"),
    # Col Headings
    f.write("%20s" % 'DIR'),
    for ccount in range(len(ColHeadingList1)):
        f.write("%18s" % ColHeadingList1[ccount]),
    f.write("\n"),
    for rcount in range(len(RowHeadingListMN)):
        f.write("%20s" % RowHeadingListMN[rcount]),
        for ccount in range(len(ColHeadingList1)):
            f.write("%18.5f" %  DIR_LIST1[ccount][rcount]),
        f.write("\n"),
    f.close()
    # ======================================



    # ---------------------------------
    # ClearSky Total
    # Test PREDICTION performance for subset of days where sample max is closest to the estimate of CSKY TOTAL (Target)

    # Method:
    # - Subset of values where % difference between TR Smooth estimate and PR Sample estimate is less than B=5% 
    BratioThresh=5.0
    Bratio=100*np.abs(ST_OBSyTR30MIN_rsds_total_max_est[0].mean(axis=1) - ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1))/ST_OBSyTR30MIN_rsds_total_max_est[0].mean(axis=1)

    # Plot example of which points will be assessed
    B5StarPoints=ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1).copy()
    B5StarPoints[Bratio > BratioThresh] = np.nan

    # ClearSky Total RSDS Daily Timesteps
    #-------------------------
    OBSpredictionTOT_mlr56_wm2_nLeap=rsdp.fn_remove_29feb_array(OBSpredictionTOT_mlr56_wm2, yearP_LenList, ntsteps=48)
    OBSpredictionTOT_mlr56_wm2_CMAX=rsdp.fn_compute_timestep_day_max(OBSpredictionTOT_mlr56_wm2_nLeap, num_years=len(yearP_LenList), ntsteps=48, yrlen=365)
    OBSpredictionTOT_mlr8_wm2_nLeap=rsdp.fn_remove_29feb_array(OBSpredictionTOT_mlr8_wm2, yearP_LenList, ntsteps=48)
    OBSpredictionTOT_mlr8_wm2_CMAX=rsdp.fn_compute_timestep_day_max(OBSpredictionTOT_mlr8_wm2_nLeap, num_years=len(yearP_LenList), ntsteps=48, yrlen=365)

    FIGWIDTH=8
    FIGHEIGHT=6

    # -----------------------------------------------------
    filename=pdir+'rsdsTOT_30MIN'+fsetup_txt+'_PR_target_MEANCSKYunsc_DAYtsMN'
    fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
    ax.plot(ST_OBSyTR30MIN_rsds_total_max_est[0].mean(axis=1), color='black', label='OBS Smoothed')
    ax.plot(ST_OBSyTR30MIN_rsds_total_max_est[0].mean(axis=1)/maxFac, color='red', label='OBS Smoothed UnSc')
    ax.plot(ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1), color='gray', label='OBS Target')
    ax.legend(loc="upper right")
    plt.title('TOTAL Daily TS MeanCSKY RSDS 30MIN')
    plt.xlabel('Day')
    plt.ylabel('RSDS, W/m2')
    ax.set_ylim(bottom=0.0, top=400.0)
    plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
    plt.close('all') # plt.show() 
    
    ST_OBSyTR30MIN_rsds_total_max_estUS0=ST_OBSyTR30MIN_rsds_total_max_est[0]/maxFac
    ratio=ST_OBSyPR_targetTOTALnLeap_MAX/ST_OBSyTR30MIN_rsds_total_max_estUS0
    ratioNI=ratio[ST_OBSyTR30MIN_rsds_total_max_estUS0>0]
    ratioGT1=ratioNI[ratioNI>1]
    print('30min Total Clearsky average bias of fit < sample')
    print(ratioGT1.mean())
    print(np.percentile(ratioGT1,50))
    print(np.percentile(ratioGT1,75))
    print(ratioGT1.max())
    
    # -----------------------------------------------------
    filename=pdir+'rsdsTOT_30MIN'+fsetup_txt+'_PR_target_MEANCSKY_DAYtsMN'
    fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
    ax.plot(ST_OBSyTR30MIN_rsds_total_max_est[0].mean(axis=1), color='black', label='OBS Smoothed')
    ax.plot(ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1), color='gray', label='OBS Target')
    ax.legend(loc="upper right")
    plt.title('TOTAL Daily TS MeanCSKY RSDS 30MIN')
    plt.xlabel('Day')
    plt.ylabel('RSDS, W/m2')
    ax.set_ylim(bottom=0.0, top=400.0)
    plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
    plt.close('all') # plt.show()  

    # -----------------------------------------------------
    filename=pdir+'rsdsTOT_30MIN'+fsetup_txt+'_PR_mlr56_B5SUBsampleMEANCSKY_DAYtsMN'
    fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
    ax.plot(ST_OBSyTR30MIN_rsds_total_max_est[0].mean(axis=1), color='black', label='OBS Smoothed')
    ax.plot(ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1), color='gray', label='OBS Target')
    ax.plot(OBSpredictionTOT_mlr56_wm2_CMAX.mean(axis=1), color='red', label='OBS MLR56')
    ax.plot(B5StarPoints, color='black', marker='x', markersize=3, linestyle='')
    ax.legend(loc="upper right")
    plt.title('TOTAL Daily TS MeanCSKY RSDS 30MIN MLR56')
    plt.xlabel('Day')
    plt.ylabel('RSDS, W/m2')
    ax.set_ylim(bottom=0.0, top=400.0)
    plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
    plt.close('all') # plt.show()  

    # -----------------------------------------------------
    for mcount in range(1): 
        OBSpredictionTOT_mltmp_wm2_nLeap=rsdp.fn_remove_29feb_array(OBSpredictionTOT_MLlist_wm2[mcount], yearP_LenList, ntsteps=48)
        OBSpredictionTOT_mltmp_wm2_CMAX=rsdp.fn_compute_timestep_day_max(OBSpredictionTOT_mltmp_wm2_nLeap, num_years=len(yearP_LenList), ntsteps=48, yrlen=365)

        filename=pdir+'rsdsTOT_30MIN'+fsetup_txt+'_PR_ML'+model_name_list[mcount]+pnum_list[mcount]+'_B5SUBsampleMEANCSKY_DAYtsMN'
        fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
        ax.plot(ST_OBSyTR30MIN_rsds_total_max_est[0].mean(axis=1), color='black', label='OBS Smoothed')
        ax.plot(ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1), color='gray', label='OBS Target')
        ax.plot(OBSpredictionTOT_mltmp_wm2_CMAX.mean(axis=1), color='red', label='OBS ML'+model_name_list[mcount])
        ax.plot(B5StarPoints, color='black', marker='x', markersize=3, linestyle='')
        ax.legend(loc="upper right")
        plt.title('TOTAL Daily TS MeanCSKY RSDS 30MIN ML'+model_name_list[mcount]+' '+bsize_txt+wlist_txt)
        plt.xlabel('Day')
        plt.ylabel('RSDS, W/m2')
        ax.set_ylim(bottom=0.0, top=400.0)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
    # -----------------------------------------------------

    RowHeadingListMN=["MLR56", "MLR8", wlist_txt+' ML '+model_name_list[0]]
    ColHeadingList1=["RMSE(CSKYTOT)"]
    
    # TOTAL
    # -------------
    TOT_LIST1=[]
    mcount=0
    # RMSE [MLR56, MLR8, ML MEAN]
    rsdsCSKYTOT_RMSE=[]
    rsdsCSKYTOT_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionTOT_mlr56_wm2_CMAX.mean(axis=1)[Bratio <= 5], ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1)[Bratio <= 5]))
    rsdsCSKYTOT_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionTOT_mlr8_wm2_CMAX.mean(axis=1)[Bratio <= 5], ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1)[Bratio <= 5]))
    OBSpredictionTOT_mltmp_wm2_nLeap=rsdp.fn_remove_29feb_array(OBSpredictionTOT_MLlist_wm2[0], yearP_LenList, ntsteps=48)
    OBSpredictionTOT_mltmp_wm2_CMAX=rsdp.fn_compute_timestep_day_max(OBSpredictionTOT_mltmp_wm2_nLeap, num_years=len(yearP_LenList), ntsteps=48, yrlen=365)
    rsdsCSKYTOT_RMSE.append(rspm.RMSE_NPARRAY(OBSpredictionTOT_mltmp_wm2_CMAX.mean(axis=1)[Bratio <= 5], ST_OBSyPR_targetTOTALnLeap_MAX.mean(axis=1)[Bratio <= 5]))

    # Write to text file
    filename=pdir+'rsdsCSKYTOT'+fsetup_txt+'_PR_PerfMetrics.txt'
    f = open(filename, 'w')
    # Col Headings
    f.write("%20s" % ''),
    for ccount in range(len(ColHeadingList1)):
        f.write("%15s" % ColHeadingList1[ccount]),
    f.write("\n"),
    for rcount in range(len(RowHeadingListMN)):
        f.write("%20s" % RowHeadingListMN[rcount]),
        f.write("%15.5f" % rsdsCSKYTOT_RMSE[rcount]),
        f.write("\n"),
    f.close()
    # ======================================


    # -----------------------------------------------------
    # Plots to assess performance

    FIGWIDTH=8
    FIGHEIGHT=6
    TFIGWIDTH=12
    T2FIGWIDTH=18
    TFIGHEIGHT=6
    SFIGWIDTH=8
    SFIGHEIGHT=8

    lwd=1
    if NO_MLmodels < 2: lwd=2

    xtimevec=np.arange(48)*0.5
    xtime3HRvec=np.arange(8)*3+1.5
    x48timelist8=['0200','0500','0800','1100','1400','1700','2000','2300']
    x48timelist12=['0400','0530','0700','0830','1000','1130','1300','1430','1600','1730','1900','2030']
    xtimevec_1week=np.arange(48*7)*0.5
    xtime3HRvec_1week=np.arange(8*7)*3+1.5
    xtimevec_2weeks=np.arange(48*14)*0.5
    xtime3HRvec_2weeks=np.arange(8*14)*3+1.5
    
    mcount=0

    # _____________________________________________________
    # PLOT Climatology Mean
    # -----------------------------------------------------
    # TOTAL RSDS PLOT simple ANNUAL MEAN RSDS values to test bias and smoothness
    # (Could repeat for seasonal values?)
    # ------
    # Prediction on testing data ANN Mean (raw)
    filename=pdir+'rsdsTOT'+fsetup_txt+'_PR_MEAN_ANNts'
    fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
    ax.plot(xtimevec,CLIMyPR_longMxTOT.mean(axis=0), label='OBS CskyTot', color='cyan', linewidth=2, linestyle='-')	# OBS TOTAL clearsky
    ax.plot(xtimevec,CLIMxPR_longCSKY.mean(axis=0), label='OBS CskyDir', color='cyan', linewidth=1, linestyle='-')	# OBS DIRECT clearsky
    ax.plot(xtimevec,OBSpredictionTOT_mlr56_wm2.mean(axis=0), label='%8s' % 'MLR56' + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_mlr56_wm2.mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)), color='gray', linewidth=1, linestyle='--') # MLR Prediction ann mean
    ax.plot(xtimevec,OBSpredictionTOT_mlr8_wm2.mean(axis=0), label='%8s' % 'MLR8' + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_mlr8_wm2.mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)), color='gray', linewidth=1, linestyle=':')  # MLR Prediction ann mean
    ax.plot(xtimevec,OBSpredictionTOT_MLlist_wm2[0].mean(axis=0), label='%8s' % ('ML '+model_name_list[0])+', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_MLlist_wm2[0].mean(axis=0), OBSyPR_targetTOT_wm2.mean(axis=0)), color=col_list[0], linewidth=lwd, linestyle='-') # ML Predicted 30min
    ax.plot(xtimevec,OBSyPR_targetTOT_wm2.mean(axis=0), label='OBS Target', color='black', linewidth=2, linestyle='-')	 # Prediction target ann mean
    ax.plot(xtimevec,OBSpredictionTOT_mlr56_wm2.mean(axis=0), color='gray', linewidth=1, linestyle='--') # MLR Prediction ann mean
    ax.plot(xtimevec,OBSpredictionTOT_mlr8_wm2.mean(axis=0), color='gray', linewidth=1, linestyle=':')  # MLR Prediction ann mean
    ax.set_ylim(bottom=-20, top=None)
    ax.legend(loc="upper right")
    plt.title('TOTAL '+model_name_list1+' '+bsize_txt+wlist_txt)
    plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
    plt.close('all') # plt.show()
    # -----------------------------------------------------
    # -----------------------------------------------------
    # DIRECT RSDS PLOT simple ANNUAL MEAN RSDS values to test smoothness
    # (Could repeat for seasonal values?)
    # ------
    # Prediction on testing data ANN Mean
    filename=pdir+'rsdsDIR'+fsetup_txt+'_PR_MEAN_ANNts'
    fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
    ax.plot(xtimevec,CLIMxPR_longCSKY.mean(axis=0), label='OBS CskyDir', color='cyan', linewidth=1, linestyle='-')	# OBS DIRECT clearsky
    ax.plot(xtimevec,OBSpredictionDIR_mlr56_wm2.mean(axis=0), label='%8s' % 'MLR56' + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_mlr56_wm2.mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)), color='gray', linewidth=1, linestyle='--') # MLR Prediction ann mean
    ax.plot(xtimevec,OBSpredictionDIR_mlr8_wm2.mean(axis=0), label='%8s' % 'MLR8' + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_mlr8_wm2.mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)), color='gray', linewidth=1, linestyle=':') # MLR Prediction ann mean
    ax.plot(xtimevec,OBSpredictionDIR_MLlist_wm2[0].mean(axis=0), label='%8s' % ('ML '+model_name_list[0])+', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_MLlist_wm2[0].mean(axis=0), OBSyPR_targetDIR_wm2.mean(axis=0)), color=col_list[0], linewidth=lwd, linestyle='-') # ML Predicted 30min
    ax.plot(xtimevec,OBSyPR_targetDIR_wm2.mean(axis=0), label='OBS Target', color='black', linewidth=2, linestyle='--')	 # Prediction target ann mean OBS
    ax.plot(xtimevec,OBSpredictionDIR_mlr56_wm2.mean(axis=0), color='gray', linewidth=1, linestyle='--') # MLR Prediction ann mean
    ax.plot(xtimevec,OBSpredictionDIR_mlr8_wm2.mean(axis=0), color='gray', linewidth=1, linestyle=':') # MLR Prediction ann mean
    ax.set_ylim(bottom=-20, top=None)
    ax.legend(loc="upper right")
    plt.title('DIRECT '+model_name_list1+' '+bsize_txt+wlist_txt)
    plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
    plt.close('all') # plt.show()
    # -----------------------------------------------------

    # _____________________________________________________
    # PLOT Example day time series
    # -----------------------------------------------------
    # SUB-DAILY VARIABILITY (DAY)
    # Plot examples of daily input and target with ML and MLR predictions
    # - Make list of indices for 1st of month on all prediction years
    # - Include daily climatology curve
    # -----------------------------------------------------
    list_1stmon=[]
    numEXyears=np.array([numPR_years,2]).min()
    sum_yrlen=0
    for yy in range(numEXyears):
        tmp_yrlen=yearP_LenList[yy]
        leapyear=False
        if tmp_yrlen>365: leapyear=True
        for mmon in range(12): list_1stmon.append(rsdp.get_daynum(mmon+1, 1, leapyear=leapyear)+sum_yrlen-1)
        sum_yrlen=sum_yrlen+tmp_yrlen
    print(list_1stmon)
    mon_list=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']*numEXyears

    # -----------------------------------------------------
    # Just plot 1st year Jan, Apr, Jul, Oct ie months in middle of each season
    print('Single days')
    for ii in range(numEXyears*4):
        daynum=list_1stmon[ii*3]
        daynumtxt=str(daynum)
        datetxt='1'+mon_list[ii*3]
        if daynum < 1000: daynumtxt='0'+str(daynum)
        if daynum < 100: daynumtxt='00'+str(daynum)
        if daynum < 10: daynumtxt='000'+str(daynum)

        # PLOT EXAMPLE of INPUT and TARGET data
        #--------------------------------
        # TOTAL Plot Sample prediction and target
        # - AND include original predictand
        filename=pdir+'ex_dayts/'+'rsdsINPUTTARGET'+fsetup_txt+'_PR_Day'+daynumtxt
        fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
        ax.plot(xtime3HRvec,OBSxPR_NMshortTOT[daynum], label='Input 3HR Tot/CskyTot', color='black', linewidth=2, linestyle='--')	 # Prediction input
        ax.plot(xtimevec,OBSxPR_NMlongCSKYDIR[daynum], label='Input CskyDir/maxday', color='cyan', linewidth=2, linestyle='--')	       # clearsky Direct
        ax.plot(xtimevec,OBSyPR_NMtargetTOT[daynum], label='Target Tot/CskyTot', color='red', linewidth=2, linestyle='-')	       # clearsky Direct
        ax.plot(xtimevec,OBSyPR_NMtargetDIR[daynum], label='Target Dir/Tot', color='blue', linewidth=2, linestyle='-')	       # clearsky Direct
        ax.set_ylim(bottom=None, top=None)
        ax.legend(loc="upper right")
        plt.title(site_name_P_txt+' '+yrS_txt_P+' Day '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #--------------------------------
        #--------------------------------

        #--------------------------------
        # TOTAL Plot Sample prediction and target
        # - AND include original predictand
        filename=pdir+'ex_dayts/'+'rsdsTOT'+fsetup_txt+'_PR_Day'+daynumtxt
        fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
        ax.plot(xtime3HRvec,OBSxPR_shortTOT_wm2[daynum], label='Input 3HR Tot', color='black', linewidth=1, linestyle='--')	 # Prediction input
        ax.plot(xtimevec,CLIMyPR_longMxTOT[daynum], label='CskyTot', color='cyan', linewidth=2, linestyle='-')	       # clearsky Total
        ax.plot(xtimevec,CLIMxPR_longCSKY[daynum], label='CskyDir', color='cyan', linewidth=1, linestyle='-')	       # clearsky Direct
        label_tmp='%8s' % 'MLR56' +', SD=' + '%5.1f' % OBSpredictionTOT_mlr56_wm2[daynum].std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_mlr56_wm2[daynum], OBSyPR_targetTOT_wm2[daynum])
        ax.plot(xtimevec,OBSpredictionTOT_mlr56_wm2[daynum], label=label_tmp, color='gray', linewidth=2, linestyle='-')  # MLR Prediction
        label_tmp='%8s' % ('ML '+model_name_list[0])+', SD=' + '%5.1f' % OBSpredictionTOT_MLlist_wm2[0][daynum].std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_MLlist_wm2[0][daynum], OBSyPR_targetTOT_wm2[daynum])
        ax.plot(xtimevec,OBSpredictionTOT_MLlist_wm2[0][daynum], label=label_tmp, color=col_list[0], linewidth=2) # ML Predicted 30min
        label_tmp='%8s' % 'Target' +', SD=' + '%5.1f' % OBSyPR_targetTOT_wm2[daynum].std()
        ax.plot(xtimevec,OBSyPR_targetTOT_wm2[daynum], label=label_tmp, color='black', linewidth=2.0, linestyle='-')    # Prediction target
        ax.set_ylim(bottom=-20, top=None)
        ax.legend(loc="upper right")
        plt.title(site_name_P_txt+' '+yrS_txt_P+' Day '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #-------------------------------
        # DIRECT Plot Sample prediction and target
        # - AND include original predictand
        filename=pdir+'ex_dayts/'+'rsdsDIR'+fsetup_txt+'_PR_Day'+daynumtxt
        fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
        ax.plot(xtimevec,CLIMxPR_longCSKY[daynum], label='OBS CskyDir', color='cyan', linewidth=1, linestyle='-') # OBS clearsky Direct
        label_tmp='%8s' % 'MLR56' +', SD=' + '%5.1f' % OBSpredictionDIR_mlr56_wm2[daynum].std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_mlr56_wm2[daynum], OBSyPR_targetDIR_wm2[daynum])
        ax.plot(xtimevec,OBSpredictionDIR_mlr56_wm2[daynum], label=label_tmp, color='gray', linewidth=2, linestyle='-')  # MLR Prediction
        label_tmp='%8s' % ('ML '+model_name_list[0])+', SD=' + '%5.1f' % OBSpredictionDIR_MLlist_wm2[0][daynum].std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_MLlist_wm2[0][daynum], OBSyPR_targetDIR_wm2[daynum])
        ax.plot(xtimevec,OBSpredictionDIR_MLlist_wm2[0][daynum], label=label_tmp, color=col_list[0], linewidth=2) # ML Predicted 30min
        label_tmp='%8s' % 'Target' +', SD=' + '%5.1f' % OBSyPR_targetDIR_wm2[daynum].std()
        ax.plot(xtimevec,OBSyPR_targetDIR_wm2[daynum], label=label_tmp, color='black', linewidth=2.0, linestyle='-')    # Prediction target
        ax.set_ylim(bottom=-20, top=None)
        ax.legend(loc="upper right")
        plt.title(site_name_P_txt+' DIR '+yrS_txt_P+' Day '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #--------------------------------
        #--------------------------------
        # TOTAL Plot Sample prediction and target - MLR56
        # - AND include original predictand
        filename=pdir+'ex_dayts/'+'rsdsTOTmMLR56'+fsetup_txt+'_PR_Day'+daynumtxt
        fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
        label_tmp='%8s' % ('ML '+model_name_list[0])+', SD=' + '%5.1f' % (OBSpredictionTOT_MLlist_wm2[0][daynum]-OBSpredictionTOT_mlr56_wm2[daynum]).std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_MLlist_wm2[0][daynum]-OBSpredictionTOT_mlr56_wm2[daynum], OBSyPR_targetTOT_wm2[daynum]-OBSpredictionTOT_mlr56_wm2[daynum])
        ax.plot(xtimevec,OBSpredictionTOT_MLlist_wm2[0][daynum]-OBSpredictionTOT_mlr56_wm2[daynum], label=label_tmp, color=col_list[0], linewidth=2) # ML Predicted 30min
        label_tmp='%8s' % 'Target' +', SD=' + '%5.1f' % (OBSyPR_targetTOT_wm2[daynum]-OBSpredictionTOT_mlr56_wm2[daynum]).std()
        ax.plot(xtimevec,OBSyPR_targetTOT_wm2[daynum]-OBSpredictionTOT_mlr56_wm2[daynum], label=label_tmp, color='black', linewidth=2.0, linestyle='-')    # Prediction target
        ax.set_ylim(bottom=None, top=None)
        ax.legend(loc="upper right")
        plt.title('TOT-MLR56 '+site_name_P_txt+' '+yrS_txt_P+' Day '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #-------------------------------
        # DIRECT Plot Sample prediction and target - MLR56
        # - AND include original predictand
        filename=pdir+'ex_dayts/'+'rsdsDIRmMLR56'+fsetup_txt+'_PR_Day'+daynumtxt
        fig, ax = plt.subplots(figsize=(FIGWIDTH,FIGHEIGHT), layout='constrained')
        label_tmp='%8s' % ('ML '+model_name_list[0])+', SD=' + '%5.1f' % (OBSpredictionDIR_MLlist_wm2[0][daynum]-OBSpredictionDIR_mlr56_wm2[daynum]).std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_MLlist_wm2[0][daynum]-OBSpredictionDIR_mlr56_wm2[daynum], OBSyPR_targetDIR_wm2[daynum]-OBSpredictionDIR_mlr56_wm2[daynum])
        ax.plot(xtimevec,OBSpredictionDIR_MLlist_wm2[0][daynum]-OBSpredictionDIR_mlr56_wm2[daynum], label=label_tmp, color=col_list[0], linewidth=2) # ML Predicted 30min
        label_tmp='%8s' % 'Target' +', SD=' + '%5.1f' % (OBSyPR_targetDIR_wm2[daynum]-OBSpredictionDIR_mlr56_wm2[daynum]).std()
        ax.plot(xtimevec,OBSyPR_targetDIR_wm2[daynum]-OBSpredictionDIR_mlr56_wm2[daynum], label=label_tmp, color='black', linewidth=2.0, linestyle='-')    # Prediction target
        ax.set_ylim(bottom=None, top=None)
        ax.legend(loc="upper right")
        plt.title('TOT-MLR56 '+site_name_P_txt+' DIR '+yrS_txt_P+' Day '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #-------------------------------



    # _____________________________________________________
    # PLOT Example week time series
    # -----------------------------------------------------
    # SUB-DAILY VARIABILITY (WEEK)
    print('Single week')
    for ii in range(numEXyears*4):
        daynum=list_1stmon[ii*3]
        daynumtxt=str(daynum)
        datetxt='1_7'+mon_list[ii*3]
        if daynum < 1000: daynumtxt='0'+str(daynum)
        if daynum < 100: daynumtxt='00'+str(daynum)
        if daynum < 10: daynumtxt='000'+str(daynum)

        #--------------------------------
        # TOTAL Plot Sample prediction and target
        # - AND include original predictand
        filename=pdir+'ex_weekts/'+'rsdsTOT'+fsetup_txt+'_PR_1week'+daynumtxt
        fig, ax = plt.subplots(figsize=(T2FIGWIDTH,FIGHEIGHT), layout='constrained')
        ax.plot(xtime3HRvec_1week,OBSxPR_shortTOT_wm2[daynum:(daynum+7)].flatten(), label='Input 3HR Tot', color='black', linewidth=1, linestyle='--')	 # Prediction input
        ax.plot(xtimevec_1week,CLIMyPR_longMxTOT[daynum:(daynum+7)].flatten(), label='CskyTot', color='cyan', linewidth=2, linestyle='-')	       # clearsky Total
        ax.plot(xtimevec_1week,CLIMxPR_longCSKY[daynum:(daynum+7)].flatten(), label='CskyDir', color='cyan', linewidth=1, linestyle='-')	       # clearsky Direct
        label_tmp='%8s' % 'MLR56' +', SD=' + '%5.1f' % OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)].std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)].flatten(), OBSyPR_targetTOT_wm2[daynum:(daynum+7)].flatten())
        ax.plot(xtimevec_1week,OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)].flatten(), label=label_tmp, color='gray', linewidth=2, linestyle='-')  # MLR Prediction
        label_tmp='%8s' % ('ML '+model_name_list[0])+', SD=' + '%5.1f' % OBSpredictionTOT_MLlist_wm2[0][daynum:(daynum+7)].std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_MLlist_wm2[0][daynum:(daynum+7)].flatten(), OBSyPR_targetTOT_wm2[daynum:(daynum+7)].flatten())
        ax.plot(xtimevec_1week,OBSpredictionTOT_MLlist_wm2[0][daynum:(daynum+7)].flatten(), label=label_tmp, color=col_list[0], linewidth=2) # ML Predicted 30min
        label_tmp='%8s' % 'Target' +', SD=' + '%5.1f' % OBSyPR_targetTOT_wm2[daynum:(daynum+7)].std()
        ax.plot(xtimevec_1week,OBSyPR_targetTOT_wm2[daynum:(daynum+7)].flatten(), label=label_tmp, color='black', linewidth=2.0, linestyle='-')    # Prediction target
        ax.set_ylim(bottom=-20, top=None)
        ax.legend(loc="upper right")
        plt.title(site_name_P_txt+' '+yrS_txt_P+' Week '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #-------------------------------
        # DIRECT Plot Sample prediction and target
        # - AND include original predictand
        filename=pdir+'ex_weekts/'+'rsdsDIR'+fsetup_txt+'_PR_1week'+daynumtxt
        fig, ax = plt.subplots(figsize=(T2FIGWIDTH,FIGHEIGHT), layout='constrained')
        ax.plot(xtimevec_1week,CLIMxPR_longCSKY[daynum:(daynum+7)].flatten(), label='OBS CskyDir', color='cyan', linewidth=1, linestyle='-') # OBS clearsky Direct
        label_tmp='%8s' % 'MLR56' +', SD=' + '%5.1f' % OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)].std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)].flatten(), OBSyPR_targetDIR_wm2[daynum:(daynum+7)].flatten())
        ax.plot(xtimevec_1week,OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)].flatten(), label=label_tmp, color='gray', linewidth=2, linestyle='-')  # MLR Prediction
        label_tmp='%8s' % ('ML '+model_name_list[0])+', SD=' + '%5.1f' % OBSpredictionDIR_MLlist_wm2[0][daynum:(daynum+7)].std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_MLlist_wm2[0][daynum:(daynum+7)].flatten(), OBSyPR_targetDIR_wm2[daynum:(daynum+7)].flatten())
        ax.plot(xtimevec_1week,OBSpredictionDIR_MLlist_wm2[0][daynum:(daynum+7)].flatten(), label=label_tmp, color=col_list[0], linewidth=2) # ML Predicted 30min
        label_tmp='%8s' % 'Target' +', SD=' + '%5.1f' % OBSyPR_targetDIR_wm2[daynum:(daynum+7)].std()
        ax.plot(xtimevec_1week,OBSyPR_targetDIR_wm2[daynum:(daynum+7)].flatten(), label=label_tmp, color='black', linewidth=2.0, linestyle='-')    # Prediction target
        ax.set_ylim(bottom=-20, top=None)
        ax.legend(loc="upper right")
        plt.title(site_name_P_txt+' DIR '+yrS_txt_P+' Week '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #--------------------------------
        #--------------------------------
        # TOTAL Plot Sample prediction and target - MLR56
        # - AND include original predictand
        filename=pdir+'ex_weekts/'+'rsdsTOTmMLR56'+fsetup_txt+'_PR_1week'+daynumtxt
        fig, ax = plt.subplots(figsize=(T2FIGWIDTH,FIGHEIGHT), layout='constrained')
        label_tmp='ML '+model_name_list[0]+', SD=' + '%5.1f' % (OBSpredictionTOT_MLlist_wm2[0][daynum:(daynum+7)]-OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)]).std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionTOT_MLlist_wm2[0][daynum:(daynum+7)].flatten()-OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)].flatten(), OBSyPR_targetTOT_wm2[daynum:(daynum+7)].flatten()-OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)].flatten())
        ax.plot(xtimevec_1week,OBSpredictionTOT_MLlist_wm2[0][daynum:(daynum+7)].flatten()-OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)].flatten(), label=label_tmp, color=col_list[0], linewidth=2) # ML Predicted 30min
        label_tmp='%8s' % ('ML '+model_name_list[0])+', SD=' + '%5.1f' % (OBSyPR_targetTOT_wm2[daynum:(daynum+7)]-OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)]).std()
        ax.plot(xtimevec_1week,OBSyPR_targetTOT_wm2[daynum:(daynum+7)].flatten()-OBSpredictionTOT_mlr56_wm2[daynum:(daynum+7)].flatten(), label=label_tmp, color='black', linewidth=2.0, linestyle='-')    # Prediction target
        ax.set_ylim(bottom=None, top=None)
        ax.legend(loc="upper right")
        plt.title('TOT-MLR56 '+site_name_P_txt+' '+yrS_txt_P+' Week '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #-------------------------------
        # DIRECT Plot Sample prediction and target - MLR56
        # - AND include original predictand
        filename=pdir+'ex_weekts/'+'rsdsDIRmMLR56'+fsetup_txt+'_PR_1week'+daynumtxt
        fig, ax = plt.subplots(figsize=(T2FIGWIDTH,FIGHEIGHT), layout='constrained')
        label_tmp='%8s' % ('ML '+model_name_list[0])+', SD=' + '%5.1f' % (OBSpredictionDIR_MLlist_wm2[0][daynum:(daynum+7)]-OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)]).std() + ', WD='+ '%4.1f' % sp.stats.wasserstein_distance(OBSpredictionDIR_MLlist_wm2[0][daynum:(daynum+7)].flatten()-OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)].flatten(), OBSyPR_targetDIR_wm2[daynum:(daynum+7)].flatten()-OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)].flatten())
        ax.plot(xtimevec_1week,OBSpredictionDIR_MLlist_wm2[0][daynum:(daynum+7)].flatten()-OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)].flatten(), label=label_tmp, color=col_list[0], linewidth=2) # ML Predicted 30min
        label_tmp='%8s' % 'Target' +', SD=' + '%5.1f' % (OBSyPR_targetDIR_wm2[daynum:(daynum+7)]-OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)]).std()
        ax.plot(xtimevec_1week,OBSyPR_targetDIR_wm2[daynum:(daynum+7)].flatten()-OBSpredictionDIR_mlr56_wm2[daynum:(daynum+7)].flatten(), label=label_tmp, color='black', linewidth=2.0, linestyle='-')    # Prediction target
        ax.set_ylim(bottom=None, top=None)
        ax.legend(loc="upper right")
        plt.title('TOT-MLR56 '+site_name_P_txt+' DIR '+yrS_txt_P+' Week '+datetxt+' '+bsize_txt+wlist_txt)
        plt.savefig(filename+'.png') # saves figure to designated file path (do this before show plot, else will plot nothing to file)
        plt.close('all') # plt.show()
        #-------------------------------


    now = datetime.datetime.now()
    print("--------------")
    print("end time")
    print(now.time())
    print("--------------")   

    #pdb.set_trace()
    
    return



if __name__ == '__main__':
    main()

    now = datetime.datetime.now()
    print("--------------")
    print("end time")
    print(now.time())
    print("--------------")

