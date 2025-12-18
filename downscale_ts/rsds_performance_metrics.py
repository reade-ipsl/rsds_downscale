'''
Created on 13.07.25
@author Rosie Eade

PERFORMANCE METRICS 
For code to compare performance of ML model architectures using pytorch.

e.g. 
python rsdsMain_ObsPrediction.py

'''

import numpy as np
import pdb

# ----------------------------------------------------------------------------
def RMSE_NPARRAY(NPARR1, NPARR2):
    """
    # Compute RMSE between input numpy arrays

    Parameters:
    -----------
    NPARR1 : numpy.ndarray
        Model Prediction
    NPARR2 : numpy.ndarray
        Target e.g. observations

    Returns:
    --------
    OUT : float
        RMSE between inputs
    """
    
    OUT=NPARR1-NPARR2
    OUT=OUT*OUT
    OUT=OUT.mean()
    OUT=np.sqrt(OUT)
    return OUT
# ----------------------------------------------------------------------------
def SRMSE_NPARRAY(NPARR1, NPARR2):
    """
    # Compute Standardised RMSE between input numpy arrays

    Parameters:
    -----------
    NPARR1 : numpy.ndarray
        Model Prediction
    NPARR2 : numpy.ndarray
        Target e.g. observations, sd used for standardisation

    Returns:
    --------
    OUT : float
        SRMSE between inputs
    """
    
    OUT=NPARR1-NPARR2
    OUT=OUT*OUT
    OUT=OUT.mean()
    OUT=np.sqrt(OUT)
    OUT=OUT/NPARR2.std()
    return OUT
# ----------------------------------------------------------------------------
def ME_NPARRAY(NPARR1, NPARR2, ABSE=False):
    """
    # Compute Mean Error (Bias) or Mean Absolute Error

    Parameters:
    -----------
    NPARR1 : numpy.ndarray
        Model Prediction
    NPARR2 : numpy.ndarray
        Target e.g. observations
    ABSE : boolean
        Option to apply abs to error i.e. True= MAE, False=ME

    Returns:
    --------
    OUT : float
        Error between inputs
    """
    OUT=NPARR1-NPARR2
    if ABSE==True: OUT=abs(OUT)
    OUT=OUT.mean()
    return OUT
# ----------------------------------------------------------------------------
def PCOR_NPARRAY(NPARR1, NPARR2):
    """
    # Compute Pearson Correlation between input numpy arrays

    Parameters:
    -----------
    NPARR1 : numpy.ndarray
        Model Prediction
    NPARR2 : numpy.ndarray
        Target e.g. observations

    Returns:
    --------
    OUT : float
        Correlation between inputs
    """
    
    OUT=np.corrcoef(NPARR1,NPARR2)[0,1]
    OUT=np.nan_to_num(OUT,nan=0.0)
    return OUT
# ----------------------------------------------------------------------------
def fn_autocorr_array_lag1(input_array, axis=0):
    """
    Compute lag 1 auto-correlation on given axis (i.e. between neighbouring values along given axis)
    - Assumes input 2d array

    Parameters:
    -----------
    input_array : numpy.ndarray
        2d array
    axis : int
        axis to compute auto-correlation on

    Returns:
    --------
    output_array : float
        Auto-correlation of input
    """
    
    ishape=input_array.shape
    output_array=input_array.mean(axis=axis)*0.0
    if axis==1:
        for ii in range(ishape[0]): output_array[ii]=np.corrcoef(input_array[ii,0:(ishape[1]-1)],input_array[ii,1:(ishape[1])])[0,1]
    if axis==0:
        for ii in range(ishape[1]): output_array[ii]=np.corrcoef(input_array[0:(ishape[0]-1),ii],input_array[1:(ishape[0]),ii])[0,1]

    output_array=np.nan_to_num(output_array, nan=0.0) # Tidy up array set nan and inf to 0

    return output_array
# ----------------------------------------------------------------------------
def fn_autocorr_array_lagN(input_array, lagN, axis=0):
    """
    Compute lag N auto-correlation on given axis (i.e. between neighbouring Nth values along given axis)
    - Assumes input 2d array

    Parameters:
    -----------
    input_array : numpy.ndarray
        2d array
    lagN : int
        lag value e.g. 1 is same as fn_autocorr_array_lag1
    axis : int
        axis to compute auto-correlation on

    Returns:
    --------
    output_array : float
        Auto-correlation of input
    """
    
    ishape=input_array.shape
    output_array=input_array.mean(axis=axis)*0.0
    if axis==1:
        for ii in range(ishape[0]): output_array[ii]=np.corrcoef(input_array[ii,0:(ishape[1]-lagN)],input_array[ii,lagN:(ishape[1])])[0,1]
    if axis==0:
        for ii in range(ishape[1]): output_array[ii]=np.corrcoef(input_array[0:(ishape[0]-lagN),ii],input_array[lagN:(ishape[0]),ii])[0,1]

    output_array=np.nan_to_num(output_array, nan=0.0) # Tidy up array set nan and inf to 0

    return output_array
# ----------------------------------------------------------------------------
