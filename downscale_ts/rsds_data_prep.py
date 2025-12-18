"""
Created on 10.06.25
@author Rosie Eade

DATA PREPARATION FUNCTIONS
For code to compare ML architectures using pytorch, e.g. 1D CNN

e.g. 
python rsdsMain_ObsPrediction.py

"""

import pandas as pd
import numpy as np
import scipy as sp
import torch
import calendar
import pdb

# ----------------------------------------------------------------------------
# GLOBAL DATA PATHS
# ----------------------------------------------------------------------------
# DIRECTORY For input data
cSARdir='data/sarah3obs_subdaily_ts/'       # Obs Data better: 3Hr mean = e.g. France [00:40 to 03:10] chosen so close to GCM [00:30 to 03:00]

# ----------------------------------------------------------------------------
# DATA PREPARATION FUNCTIONS FOR ML MODELS
# ----------------------------------------------------------------------------
def get_yr_length(yearNum):
    """
    Compute the length of a given year

    Parameters:
    -----------
    yearNum : int
        Input year e.g. 2024

    Returns:
    --------
    yrLength : int
        No. of days in input year e.g. 365
    """

    yrLength = 365
    if calendar.isleap(yearNum): yrLength=366    #Check if a leap year
    return yrLength

# ----------------------------------------------------------------------------
def get_daynum(month, day, leapyear=False):
    """
    Compute the day number (1 to 366) for a given date (month and day in month)

    Parameters:
    -----------
    month : int
        Input month e.g. 2 (= February)
    day : int
        Input date e.g. 1 (= 1st of the month)
    leapyear : boolean
        True => leapyear; False => not a leap year

    Returns:
    --------
    day_num : int
        Output day of year (not date) e.g. 1st February => day 32
    """
    
    mon_num=np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12])
    mon_len=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    if leapyear: mon_len=np.array([31,29,31,30,31,30,31,31,30,31,30,31])
    
    if month==1: day_num=day
    if month>1:  day_num=(sum(mon_len[0:(month-1)])+day).item() # else returns np.int64(day_num)
    
    return day_num

# ----------------------------------------------------------------------------
def get_monthday(day_num, leapyear=False):
    """
    Compute the date (month and day in month) for a given day number in single year (1 to 366)

    Parameters:
    -----------
    day_num : int
        Input day of year (not date) e.g. day 32 => 1st February
    leapyear : boolean
        True => leapyear; False => not a leap year

    Returns:
    --------
    month : int
        Output month e.g. 2 (= February)
    day : int
        Output date e.g. 1 (= 1st of the month)
    """

    if day_num < 1: return np.nan, np.nan
    if leapyear==False and day_num > 365: return np.nan, np.nan
    if leapyear==True and day_num > 366: return np.nan, np.nan
    
    mon_num=np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12])
    mon_len=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    if leapyear: mon_len=np.array([31,29,31,30,31,30,31,31,30,31,30,31])
    mon_len_cum=mon_len.copy()
    for mcount in range(11): mon_len_cum[mcount+1]=int(sum(mon_len[0:(mcount+2)]))
    day_mon_len_cum=day_num-mon_len_cum
    max_nonpos=day_mon_len_cum[day_mon_len_cum<=0].max()
    month=int(mon_num[np.where(day_mon_len_cum==max_nonpos)[0]][0])
    if month==1: day=int(day_num)
    if month>1: day=int(day_num-mon_len_cum[month-2])
    
    return month, day

# ----------------------------------------------------------------------------
def get_yearmonthday(day_num, year_list):
    """
    Compute the date (month and day in month) for a given day number in a list of years (1 to ...)

    Parameters:
    -----------
    day_num : int
        Input day of year (not date) e.g. day 32 => 1st February
    year_list : list of ints
        List of years e.g. [2021, 2022, 2023, 2024]

    Returns:
    --------
    month : int
        Output month e.g. 2 (= February)
    day : int
        Output date e.g. 1 (= 1st of the month)
    """

    if day_num < 1: return np.nan, np.nan, np.nan
    
    nyrs=len(year_list)
    total_ndays=0
    if nyrs==1: total_ndays=get_yr_length(year_list[0])
    if nyrs>1: 
        for ycount in range(nyrs): total_ndays=total_ndays+get_yr_length(year_list[ycount])
    if day_num>total_ndays: return np.nan, np.nan, np.nan

    year_np=np.array(year_list)
    if nyrs==1:
        month, day=get_monthday(day_num, leapyear=calendar.isleap(year_list[0]))
        year=year_list[0]
    
    yr1_len=get_yr_length(year_list[0])
    if day_num<=yr1_len: 
        month, day=get_monthday(day_num, leapyear=calendar.isleap(year_list[0]))
        year=year_list[0]
    
    if nyrs>1 and day_num>yr1_len: 
        yr_len=np.zeros(nyrs)
        yr_len_cum=np.zeros(nyrs)
        for ycount in range(nyrs): yr_len[ycount]=get_yr_length(year_list[ycount])
        yr_len_cum[0]=yr_len[0]
        for ycount in range(nyrs-1): yr_len_cum[ycount+1]=int(sum(yr_len[0:ycount+2]))
        day_yr_len_cum=day_num-yr_len_cum
        max_nonpos=day_yr_len_cum[day_yr_len_cum<=0].max()
        year=int(year_np[np.where(day_yr_len_cum==max_nonpos)[0]][0])
        day_num1yr=day_num-yr_len_cum[np.where(day_yr_len_cum==max_nonpos)[0]-1]
        month, day=get_monthday(day_num1yr, leapyear=calendar.isleap(year))
    
    return year, month, day

# ----------------------------------------------------------------------------
def get_sincos_datetime(dFrame, ntsteps=8, add_day=False, timestamp=False):
    """
    Convert date and time into pairs of sin and cos functions

    Parameters:
    -----------
    dFrame : data frame of datetimes
        Input list of subdaily datetimes for a year (assumes length 365 days)       
    ntsteps : int
        No. of timesteps in a day e.g. 8 (3 hourly steps) or 48 (30min steps)
    add_day : boolean
        If True, extend output functions by 1 day to represent a leap year
    timestamp : boolean
        Defines method to read in dFrame

    Returns:
    --------
    date_vec : numpy.ndarray
        Output pairs of sin and cos of day/no. days per year, for each date and time in year (length 365 days, or 366 if add_day==True)
    time_vec : numpy.ndarray
        Output pairs of sin and cos time/no. seconds per day, for each date and time in year (length 365 days, or 366 if add_day==True)
    """

    yrlen=365 # Default year length
    
    newyrlen=yrlen
    if add_day==True: newyrlen=yrlen+1 # Option to extend output to represent leapyear
    
    # Setup numpy arrays to output sin and cos pairs
    date_vec=np.zeros((newyrlen*ntsteps, 2))   # s,c(time)
    time_vec=np.zeros((newyrlen*ntsteps, 2))   # s,c(time)
    nsec_perday=60*60*24 # No. seconds per day
    ndays_peryr=365.25 # No. days in year (or 365? 366?)
    
    # Loop over each day in year to get sin and cos pairs of date
    ncount=0
    for dd in range(yrlen):
        # dFrame['time'][ncount] Looks for the key with number ncount, NOT the ncount'th row! So fails for leap year where removed some rows
        icount=dFrame.index[ncount]
        if timestamp==False: daynum=get_daynum(dFrame['time'][icount].month, dFrame['time'][icount].day, leapyear=False) # day number within the year
        if timestamp==True: daynum=get_daynum(dFrame[icount].month, dFrame[icount].day, leapyear=False) # day number within the year
        # Loop over each timestep in day to get sin and cos pairs of time
        for tstep in range(ntsteps):
            icount=dFrame.index[ncount]
            if timestamp==False: timesec=dFrame['time'][icount].hour*60*60+dFrame['time'][icount].minute*60+dFrame['time'][icount].second
            if timestamp==True: timesec=dFrame[icount].hour*60*60+dFrame[icount].minute*60+dFrame[icount].second
            time_vec[dd*ntsteps+tstep,0]=np.sin((2*np.pi*(timesec))/nsec_perday)    # sin(time in sec)
            time_vec[dd*ntsteps+tstep,1]=np.cos((2*np.pi*(timesec))/nsec_perday)    # sin(time in sec)
            date_vec[dd*ntsteps+tstep,0]=np.sin((2*np.pi*(daynum-1))/ndays_peryr) # sin(daynum)
            date_vec[dd*ntsteps+tstep,1]=np.cos((2*np.pi*(daynum-1))/ndays_peryr) # cos(daynum)
            ncount=ncount+1

    # Add extra days of datetime info on end for Leap-Years (time just repeats the same each day)
    if add_day==True:
        daynum=daynum+1
        dd=yrlen
        for tstep in range(ntsteps):
            time_vec[dd*ntsteps+tstep,0]=time_vec[(yrlen-1)*ntsteps+tstep,0]    # sin(time in sec)
            time_vec[dd*ntsteps+tstep,1]=time_vec[(yrlen-1)*ntsteps+tstep,1]    # sin(time in sec)
            date_vec[dd*ntsteps+tstep,0]=np.sin((2*np.pi*(daynum-1))/ndays_peryr) # sin(daynum)
            date_vec[dd*ntsteps+tstep,1]=np.cos((2*np.pi*(daynum-1))/ndays_peryr) # cos(daynum)
            ncount=ncount+1

    return date_vec, time_vec

# ----------------------------------------------------------------------------
def fn_30minTo3hr_np(in_array, in_timestamps):
    """
    Convert numpy array of 30min data to 3hour means

    Parameters:
    -----------
    in_array : numpy.ndarray
        Input array of 30 minute data [no. days, 48]

    Returns:
    --------
    out_array_3HR : numpy.ndarray
        Output array of 3 hour mean data [no. days, 8]

    """

    in_array_vec=in_array.reshape(in_array.shape[0]*in_array.shape[1])
    df_in_array = pd.DataFrame({'datetime': in_timestamps, 'SIS': in_array_vec}).set_index('datetime')
    df_in_array_3HR = df_in_array['SIS'].resample(rule='3h', origin='start').mean()
    out_array_3HR = df_in_array_3HR.to_numpy()
    out_array_3HR = out_array_3HR.reshape(in_array.shape[0], 8)
    
    return out_array_3HR
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# FUNCTIONS TO READ in Irradiance data from *.csv files
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def define_traintest_years(train_test_case=2):
    """
    Options for partioning data years into training (train) and prediction (test) subsets 

    Parameters:
    -----------
    train_test_case : int
        int code refers to specific partitioning case

    Returns:
    --------
    yearS_List : List of int
        List of years to include in training subset
    yrS_txt : string
        Text description of list of years in yearS_List e.g. '2013_2022'
    yearS_P_List : List of int
        List of years to include in prediction subset
    yrS_txt_P : string
        Text description of list of years in yearS_P_List e.g. '2023_2024'

    """
    
    # -----------------
    # Data for training method (TR)
    
    # Use all available years for training and testing, as perfect model setup
    if train_test_case == 1:
        yearS_List=    [ 2013,    2014,    2015,    2016,    2017,    2018,    2019,    2020,    2021,    2022,    2023,    2024]
        yrS_txt='2013_2024'
    # Use all-2 years of available data for training, so 2 years left over for out of sample testing
    if train_test_case == 2:
        yearS_List=    [ 2013,    2014,    2015,    2016,    2017,    2018,    2019,    2020,    2021,    2022]
        yrS_txt='2013_2022'
    # Use 4 years to train, 2 for prediction
    if train_test_case == 3:
        yearS_List=    [2019,    2020,    2021,    2022]
        yrS_txt='2019_2022'
    # Use all-2 years of available data for training (as 2), but apply prediction testing to all 12 years
    if train_test_case == 4:
        yearS_List=    [ 2013,    2014,    2015,    2016,    2017,    2018,    2019,    2020,    2021,    2022]
        yrS_txt='2013_2022'

    # -----------------
    # Data for testing prediction method (PR) e.g. outside training years
    
    # Use all available years for training and testing, as perfect model setup
    if train_test_case == 1:
        yearS_P_List=    [ 2013,    2014,    2015,    2016,    2017,    2018,    2019,    2020,    2021,    2022,    2023,    2024]
        yrS_txt_P='2013_2024'
    # Use all-2 years of available data for training, so 2 years left over for out of sample testing
    if train_test_case == 2:
        yearS_P_List=    [ 2023,    2024]
        yrS_txt_P='2023_2024'
    # Use 4 years to train, 2 for prediction
    if train_test_case == 3:
        yearS_P_List=    [ 2023,    2024]
        yrS_txt_P='2023_2024'
    # Use all-2 years of available data for training (as 2), but apply prediction testing to all 12 years
    if train_test_case == 4:
        yearS_P_List=    [ 2013,    2014,    2015,    2016,    2017,    2018,    2019,    2020,    2021,    2022,    2023,    2024]
        yrS_txt_P='2013_2024'

    return yearS_List, yrS_txt, yearS_P_List, yrS_txt_P

# ----------------------------------------------------------------------------
def fn_extract_month_annarray(input_vec, year_LenList, mon_num, RmLeap=False):
    """
    Extract single month (mon_num=1 => Jan, 2=Feb, ...) from input_vec array of form [No. Days, No. Timesteps]
     - Assumes years start on 1st Jan and finish 31st Dec
    Parameters:
    -----------
    input_vec : numpy.ndarray
        Input array of data [no. days, no. subdaily timesteps]
    year_LenList : numpy.ndarray
        np array containing the number of days in each input year within input_vec
    mon_num : int
        Code to define month: 1=Jan, 2=Feb, ...
    RmLeap : boolean
        if RmLeap==True: Removes 29th Feb

    Returns:
    --------
    output_vec_mon : numpy.ndarray
        Output array of data for just the days in the specified season
    
    """


    # no. days per month
    mon_lenNlp=np.array([31,28,31,30,31,30,31,31,30,31,30,31])	# Non leap-year
    mon_lenLp=np.array([31,29,31,30,31,30,31,31,30,31,30,31])	# Leap-year
    
    # extract just the days in the given month
    d_list=[]
    for yy in range(len(year_LenList)):
        if year_LenList[yy]==365: mon_len=mon_lenNlp
        if year_LenList[yy]==366: mon_len=mon_lenLp
        for dd in range(mon_len[mon_num-1]): d_list.append(sum(year_LenList[0:yy])+sum(mon_len[0:(mon_num-1)])+dd)
        if year_LenList[yy]==366 and RmLeap==True and mon_num==2: d_list=d_list[0:-1] # skip 29th Feb
    output_vec_mon=input_vec[d_list]

    return output_vec_mon

# ----------------------------------------------------------------------------
def fn_extract_season_annarray(input_vec, year_LenList, seas_num, RmLeap=False):
    """
    Extract single 3 month seasons (seas_num=1 => DJF, 2=MAM, 3=JJA, 4=SON) from input_vec array of form [No. Days, No. Timesteps]
     - Assumes years start on 1st Jan and finish 31st Dec
     
    Parameters:
    -----------
    input_vec : numpy.ndarray
        Input array of data [no. days, no. subdaily timesteps]
    year_LenList : numpy.ndarray
        np array containing the number of days in each input year within input_vec
    seas_num : int
        Code to define season: 1=DJF, 2=MAM, 3=JJA, 4=SON
    RmLeap : boolean
        if RmLeap==True: Removes 29th Feb

    Returns:
    --------
    output_vec_mon : numpy.ndarray
        Output array of data for just the days in the specified season
    
    """
    
    # no. days per month
    mon_lenNlp=np.array([31,28,31,30,31,30,31,31,30,31,30,31])	# Non leap-year
    mon_lenLp=np.array([31,29,31,30,31,30,31,31,30,31,30,31])	# Leap-year
    
    # match code to season choice
    if seas_num==1: seas_vec=np.array([1,2,12])
    if seas_num==2: seas_vec=np.array([3,4,5])
    if seas_num==3: seas_vec=np.array([6,7,8])
    if seas_num==4: seas_vec=np.array([9,10,11])
    
    # extract just the days in the given season
    d_list=[]
    for mon_num in seas_vec:
        for yy in range(len(year_LenList)):
            if year_LenList[yy]==365: mon_len=mon_lenNlp
            if year_LenList[yy]==366: mon_len=mon_lenLp
            for dd in range(mon_len[mon_num-1]): d_list.append(sum(year_LenList[0:yy])+sum(mon_len[0:(mon_num-1)])+dd)
            if year_LenList[yy]==366 and RmLeap==True and mon_num==2: d_list=d_list[0:-1] # skip 29th Feb
    output_vec_mon=input_vec[d_list]

    return output_vec_mon

# ----------------------------------------------------------------------------
def fn_get_rawinput_data(year_list=[2000], site_list=['France'], dataset='SARAH3', variable='SIS'):
    """
    Read in solar irradiance data for input in ML or MLR models, including data for normalisation
    - NB SARAH3 Approximates 3hr model mean data using SARAH3 data averaged over CMIP6 GCM style grid box region e.g. IPSL-CM6A-LR
        
    Parameters:
    -----------
    year_list : list of int
        List of years of data to read in
    site_list : list of strings
        Location to select (list of same length as year_list)
    dataset : string
        Code identifying dataset to read in 
          e.g. 'SARAH3' for SARAH3 satellite observations
    variable : string
        Code identifying variable to read in e.g. 'SIS' for Total allsky solar irradiance 
          (SARAH3 also has SID; GCMs also have SIScs i.e. rsdscs)
    
    Returns:
    --------
    x_shortTOTAL : numpy.ndarray
        ML Model INPUT: Total Allsky - 3 Hour Mean, Regional Mean, original timesteps (SARAH3 offset, GCM standard)
          array [total number of days, 8 time steps per day]
    x_longCSKYDIR : numpy.ndarray
        ML Model INPUT: Direct Clearsky - 30 Minute Instantaneous, at Target Point Value and timesteps; np array [total number of days, 48 time steps per day]
          array [total number of days, 48 time steps per day]
    x_shortCSKYDIR : numpy.ndarray
        Other Data: Direct Clearsky - 3 Hour Mean, Regional Mean (time matches x_shortTOTAL); np array [total number of days, 8 time steps per day]
    x_shortTOTALnLeap_MAX : numpy.ndarray
        Normalisation Data: Sample Max of x_shortTOTAL across all years on original timesteps; function(day in year, timestep) (first remove 29th Feb so all length 365 days)
    x_shortCSKYDIRnLeap_MAX : numpy.ndarray
        Other Data: Sample Max of x_shortCSKYDIR across all years on original timesteps
    x_short_timestamps_nLeap : pandas.core.series.Series
        Datetime: timestamps associated with x_shortTOTALnLeap_MAX
    xST_shortCSKYDIRnLeap_MAX : numpy.ndarray
        Other Data: Sample Max of CSKYDIR across all years on STANDARD timesteps
    xST_short_timestamps_nLeap : pandas.core.series.Series
        Datetime: timestamps associated with xST_shortTOTALnLeap_MAX
    x_longCSKYDIR_maxday : numpy.ndarray
        Normalisation Data: Max day in year of Direct Clearsky - 30 Minute Instantaneous, at Target Point Value
    data_length_list : list of ints
        Summary data: [no. years, total no. days, length of short day=8]

    """
    
    # -------------------------
    # Filename information
    
    # Variable name
    var_key='SIS'
    if variable=='SID': var_key='SID'
    if variable=='SIScs': var_key='SIScs'
    
    # 3 Hourly irradiance data from SARAH3 or GCM on original timesteps
    # e.g. SARAH3 in France 00:10, 00:40, ...; GCM is standard 00:00, 00:30, ...
    fname_start3hr='SARAH3_DATE_'+variable+'_3HR_IPSLCM6_'
    if dataset=='IPSL-CM6A-LR_r1': fname_start3hr='IPSL-CM6A-LR_r1_DATE_'+variable+'_3HR_'
    indir3hr=cSARdir
    if dataset=='IPSL-CM6A-LR_r1': indir3hr=c3HRdir
    
    # 3 Hourly clearsky direct data estimated by pysolar on Standard timesteps (00:00, 00:30, ...)
    pysolar_start3hr='PYSOLAR_DATE_SID_3HR_IPSLCM6_' 
    
    # 30 Minute clearsky direct data estimated by pysolar on STANDARD timesteps (00:00, 00:30, ...)
    fname_start30min='PYSOLAR_DATE_SID_30MIN_'
    indir30min=cSARdir
    if dataset=='IPSL-CM6A-LR_r1': indir30min=c3HRdir

    # Length of data expected
    num_years = len(year_list) # No. of years
    num_days = 0
    for yr in year_list: num_days = num_days + get_yr_length(yr) # Total number of days over all years (checks if a leap year)

    daylen_short=8 # No. sub-daily time steps (3 hourly)
    daylen_long=48 # No. sub-daily time steps (30 minute)

    # Create data arrays
    # Raw Data arrays
    x_shortTOTAL=np.zeros((num_days, daylen_short))      	# 3hr means rsds TOTAL original timesteps
    x_shortCSKYDIR=np.zeros((num_days, daylen_short)) 		# 3hr means clearsky DIRECT (pysolar) original timesteps
    x_longCSKYDIR=np.zeros((num_days, daylen_long))      	# 30min clearsky DIRECT (pysolar) standard timesteps

    # Non-leap year versions of Raw Data arrays, i.e. without 29th Feb
    x_shortTOTALnLeap=np.zeros((num_years*365, daylen_short))	# 3hr means rsds TOTAL original timesteps
    x_shortCSKYDIRnLeap=np.zeros((num_years*365, daylen_short))	# 3hr means clearsky DIRECT (pysolar) original timesteps
    xST_shortCSKYDIRnLeap=np.zeros((num_years*365, daylen_short)) # 3hr means clearsky DIRECT (pysolar) standard timesteps

    # Populate data arrays, looping over a year at a time
    SampleStart=0 # First data row for a specific year
    SampleStartnLeap=0
    for ycount in range(num_years):
        site_name=site_list[ycount]
        yearS=year_list[ycount]
        yearS_txt=str(yearS)
        num_days_tmp=get_yr_length(yearS)    

        # Read 3 Hourly irradiance data, Raw data on original timesteps
        SIS3HR=pd.read_csv(indir3hr+site_name+'/'+fname_start3hr+yearS_txt+'_'+site_name+'.csv')  
        SIS3HR["time"] = pd.to_datetime(SIS3HR["time"])
        for dd in range(num_days_tmp): x_shortTOTAL[dd+SampleStart,:] = SIS3HR[[var_key]].values[(dd*daylen_short):(dd*daylen_short+daylen_short),0] # Use 3HR input file (not version where repeated 30min steps)
        for dd in range(num_days_tmp): x_shortCSKYDIR[dd+SampleStart,:] = SIS3HR[['Clearsky']].values[(dd*daylen_short):(dd*daylen_short+daylen_short),0] # Use 3HR input file where mean already computed

        # Create non-leap year versions for leap years so can calculate non-leap climatology over all years
        SIS3HRnLeap=SIS3HR.copy()
        if num_days_tmp==366: maskS=SIS3HR['time'].between(yearS_txt+'-02-29 00:00:00+00:00', yearS_txt+'-02-29 23:59:59+00:00')
        if num_days_tmp==366: SIS3HRnLeap=SIS3HR[maskS==False]
        for dd in range(365): x_shortTOTALnLeap[dd+SampleStartnLeap,:] = SIS3HRnLeap[[var_key]].values[(dd*daylen_short):(dd*daylen_short+daylen_short),0]
        for dd in range(365): x_shortCSKYDIRnLeap[dd+SampleStartnLeap,:] = SIS3HRnLeap[['Clearsky']].values[(dd*daylen_short):(dd*daylen_short+daylen_short),0]
        if ycount==(num_years-1): x_short_timestamps_nLeap=pd.to_datetime(SIS3HRnLeap['time'])

        # Read 3 Hourly clearsky direct data estimated by pysolar, Raw data on  Standard timesteps
        if dataset=='SARAH3' or dataset=='SARAH3sh0010': CSKYDIR3HR=pd.read_csv(indir3hr+site_name+'/'+pysolar_start3hr+yearS_txt+'_'+site_name+'.csv')  
        if dataset=='IPSL-CM6A-LR_r1': CSKYDIR3HR=SIS3HR.copy()
        CSKYDIR3HRnLeap=CSKYDIR3HR.copy()
        if num_days_tmp==366: maskS=CSKYDIR3HR['time'].between(yearS_txt+'-02-29 00:00:00+00:00', yearS_txt+'-02-29 23:59:59+00:00')
        if num_days_tmp==366: CSKYDIR3HRnLeap=CSKYDIR3HR[maskS==False]
        for dd in range(365): xST_shortCSKYDIRnLeap[dd+SampleStartnLeap,:] = CSKYDIR3HRnLeap[['Clearsky']].values[(dd*daylen_short):(dd*daylen_short+daylen_short),0]
        if ycount==(num_years-1): xST_short_timestamps_nLeap=pd.to_datetime(CSKYDIR3HRnLeap['time'])

        # Read 30 Minute clearsky direct data estimated by pysolar, Raw data on  Standard timesteps
        SIS30MIN=pd.read_csv(indir30min+site_name+'/'+fname_start30min+yearS_txt+'_'+site_name+'.csv')
        SIS30MIN["time"] = pd.to_datetime(SIS30MIN["time"])
        for dd in range(num_days_tmp): x_longCSKYDIR[dd+SampleStart,:] = SIS30MIN[['Clearsky']].values[(dd*daylen_long):(dd*daylen_long+daylen_long),0]

        SampleStart=SampleStart+num_days_tmp
        SampleStartnLeap=SampleStartnLeap+365

    # Compute Sample Max of Total (non-leap year)
    x_shortTOTALnLeap_MAX=fn_compute_timestep_day_max(x_shortTOTALnLeap, num_years=num_years, ntsteps=daylen_short, yrlen=365)
    x_shortCSKYDIRnLeap_MAX=fn_compute_timestep_day_max(x_shortCSKYDIRnLeap, num_years=num_years, ntsteps=daylen_short, yrlen=365)
    xST_shortCSKYDIRnLeap_MAX=fn_compute_timestep_day_max(xST_shortCSKYDIRnLeap, num_years=num_years, ntsteps=daylen_short, yrlen=365)

    # Compute max clearsky day in whole period E.g. day with max total clearsky (i.e. max mean)
    aindex=np.arange(num_days)
    clearskyDayMN=x_longCSKYDIR.mean(axis=1)
    maxMNday=aindex[clearskyDayMN==clearskyDayMN.max()]
    x_longCSKYDIR_maxday=x_longCSKYDIR[maxMNday]
    
    # Summary data
    data_length_list=[num_years, num_days, daylen_short]

    return x_shortTOTAL, x_longCSKYDIR, x_shortCSKYDIR, x_shortTOTALnLeap_MAX, x_shortCSKYDIRnLeap_MAX, x_short_timestamps_nLeap, xST_shortCSKYDIRnLeap_MAX, xST_short_timestamps_nLeap, x_longCSKYDIR_maxday, data_length_list

# ----------------------------------------------------------------------------
def fn_get_rawtarget_data(year_list=[2000], site_list=['France'], dataset='SARAH3', variable2='SID'):

    """
    Read in solar irradiance data for target in ML or MLR models, including data for normalisation
     - variable2 default is SID so can model Direct radiation as well as Total (available from SARAH3)
     - CMIP6 models don't output Direct so either ignore (variable2='SIS' so repeat) or for interest can output 'SIScs' clearsky TOTAL

    Parameters:
    -----------
    year_list : list of int
        List of years of data to read in
    site_list : list of strings
        Location to select (list of same length as year_list)
    dataset : string
        Code identifying dataset to read in 
        e.g. 'SARAH3' for SARAH3 satellite observations
    variable : string
        Code identifying variable to read in e.g. 'SIS' for Total allsky solar irradiance 
        (SARAH3 also has SID; GCMs also have SIScs i.e. rsdscs)
    
    Returns:
    --------
    y_targetTOTAL : numpy.ndarray
        ML Model TARGET: Total Allsky - 30 Minute Instantaneous, at Target Point Value, original timesteps (SARAH3 offset, GCM standard)
          array [total number of days, 48 time steps per day]
    y_targetDIRECT : numpy.ndarray 
        ML Model TARGET: Direct Allsky - 30 Minute Instantaneous, at Target Point Value, original timesteps (SARAH3 offset, GCM standard)
          array [total number of days, 48 time steps per day]
    y_targetTOTALnLeap_MAX : numpy.ndarray 
        Normalisation Data: Sample Max of y_targetTOTAL across all years on original timesteps; function(day in year, timestep) (first remove 29th Feb so all length 365 days)
          array [365, 48 time steps per day]
    y_longCSKYDIRnLeap_MAX : numpy.ndarray 
        Other Data: Sample Max of CSKYDIR across all years on original timesteps    
          array [365, 48 time steps per day]
    y_target_timestamps_nLeap : pandas.core.series.Series 
        Datetime: timestamps associated with y_targetTOTALnLeap_MAX, on original timesteps    
          series [365*48 time steps per day]
    yST_longCSKYDIRnLeap_MAX : numpy.ndarray 
        Normalisation Data: Sample Max of CSKYDIR across all years on STANDARD timesteps    
          array [total number of days, 48 time steps per day]
    yST_target_timestamps_nLeap: pandas.core.series.Series
        Datetime: timestamps associated with yST_longCSKYDIRnLeap_MAX, on STANDARD timesteps
           series [365*48 time steps per day]
    data_length_list : list of ints
        Summary data: [no. years, total no. days, length of long day=48]

    """ 


    # -------------------------
    # Filename information
    
    # Variable name
    var_key2='SID'
    if variable2=='SIScs': var_key2='SIScs'
    if variable2=='SIS': var_key2='SIS'
    
    # 30 Minute irradiance data from SARAH3 or GCM on original timesteps
    # e.g. SARAH3 in France 00:10, 00:40, ...; GCM is standard 00:00, 00:30, ...
    fname_start30min='SARAH3_DATE_SIS_30MIN_'
    if dataset=='IPSL-CM6A-LR_r1': fname_start30min='IPSL-CM6A-LR_r1_DATE_SIS_30MIN_'
    indir30min=cSARdir
    if dataset=='IPSL-CM6A-LR_r1': indir30min=c30MINdir

    fname_start30min2='SARAH3_DATE_'+variable2+'_30MIN_'
    if dataset=='IPSL-CM6A-LR_r1': fname_start30min2='IPSL-CM6A-LR_r1_DATE_'+variable2+'_30MIN_'
    indir30min2=cSARdir
    if dataset=='IPSL-CM6A-LR_r1': indir30min2=c30MINdir
    
    # 30 Minute clearsky direct data estimated by pysolar on STANDARD timesteps (00:00, 00:30, ...)
    fname_start30min3='PYSOLAR_DATE_SID_30MIN_'
    indir30min3=cSARdir
    if dataset=='IPSL-CM6A-LR_r1': indir30min3=c30MINdir
    
    # Length of data expected
    num_years = len(year_list)
    num_days = 0
    for yr in year_list: num_days = num_days + get_yr_length(yr) # Total number of days over all years (checks if a leap year)

    daylen_long=48 # No. sub-daily time steps (30 minute)
    
    # Create data arrays
    # Raw Data arrays
    y_targetTOTAL=np.zeros((num_days, daylen_long))		# 30min rsds TOTAL original timesteps
    y_targetDIRECT=np.zeros((num_days, daylen_long))		# 30min rsds DIRECT original timesteps

    # Non-leap year versions of Raw Data arrays, i.e. without 29th Feb
    y_targetTOTALnLeap=np.zeros((num_years*365, daylen_long))	# 30min rsds TOTAL original timesteps
    y_longCSKYDIRnLeap=np.zeros((num_years*365, daylen_long))	# 30min clearsky DIRECT (pysolar) original timesteps
    yST_longCSKYDIRnLeap=np.zeros((num_years*365, daylen_long))	# 30min clearsky DIRECT (pysolar) standard timesteps

    # Populate data arrays, looping over a year at a time
    SampleStart=0 # First data row for a specific year
    SampleStartnLeap=0
    for ycount in range(num_years):
        site_name=site_list[ycount]
        yearS=year_list[ycount]
        yearS_txt=str(yearS)
        num_days_tmp=get_yr_length(yearS)

        # Read 30 min irradiance data, Raw data on original timesteps
        SIS30MIN=pd.read_csv(indir30min+site_name+'/'+fname_start30min+yearS_txt+'_'+site_name+'.csv')
        SIS30MIN["time"] = pd.to_datetime(SIS30MIN["time"])
        if variable2=='SIS':
            SID30MIN=SIS30MIN.copy()
        if variable2!='SIS':
            SID30MIN=pd.read_csv(indir30min2+site_name+'/'+fname_start30min2+yearS_txt+'_'+site_name+'.csv')
        SID30MIN["time"] = pd.to_datetime(SID30MIN["time"])
        for dd in range(num_days_tmp): y_targetTOTAL[dd+SampleStart,:] = SIS30MIN[['SIS']].values[(dd*daylen_long):(dd*daylen_long+daylen_long),0]
        for dd in range(num_days_tmp): y_targetDIRECT[dd+SampleStart,:] = SID30MIN[[var_key2]].values[(dd*daylen_long):(dd*daylen_long+daylen_long),0]

        # Create non-leap year versions for leap years so can calculate non-leap climatology over all years
        SIS30MINnLeap=SIS30MIN.copy()
        if num_days_tmp==366: maskS=SIS30MIN['time'].between(yearS_txt+'-02-29 00:00:00+00:00', yearS_txt+'-02-29 23:59:59+00:00')
        if num_days_tmp==366: SIS30MINnLeap=SIS30MIN[maskS==False]
        for dd in range(365): y_targetTOTALnLeap[dd+SampleStartnLeap,:] = SIS30MINnLeap[['SIS']].values[(dd*daylen_long):(dd*daylen_long+daylen_long),0]
        for dd in range(365): y_longCSKYDIRnLeap[dd+SampleStartnLeap,:] = SIS30MINnLeap[['Clearsky']].values[(dd*daylen_long):(dd*daylen_long+daylen_long),0]
        if ycount==(num_years-1): y_target_timestamps_nLeap=pd.to_datetime(SIS30MINnLeap['time'])

        # Reade 30 min clearsky direct estimated by pysolar, Raw data on  Standard timesteps
        if dataset=='SARAH3' or dataset=='SARAH3sh0010': ST_SIS30MIN=pd.read_csv(indir30min3+site_name+'/'+fname_start30min3+yearS_txt+'_'+site_name+'.csv')
        if dataset=='IPSL-CM6A-LR_r1': ST_SIS30MIN=SIS30MIN.copy()
        ST_SIS30MINnLeap=ST_SIS30MIN.copy()
        if num_days_tmp==366: maskS=ST_SIS30MIN['time'].between(yearS_txt+'-02-29 00:00:00+00:00', yearS_txt+'-02-29 23:59:59+00:00')
        if num_days_tmp==366: ST_SIS30MINnLeap=ST_SIS30MIN[maskS==False]
        for dd in range(365): yST_longCSKYDIRnLeap[dd+SampleStartnLeap,:] = ST_SIS30MINnLeap[['Clearsky']].values[(dd*daylen_long):(dd*daylen_long+daylen_long),0]
        if ycount==(num_years-1): yST_target_timestamps_nLeap=pd.to_datetime(ST_SIS30MINnLeap['time'])

        SampleStart=SampleStart+num_days_tmp
        SampleStartnLeap=SampleStartnLeap+365

    # Compute Sample Max of Total (non-leap year)
    y_targetTOTALnLeap_MAX=fn_compute_timestep_day_max(y_targetTOTALnLeap, num_years=num_years, ntsteps=daylen_long, yrlen=365)
    y_longCSKYDIRnLeap_MAX=fn_compute_timestep_day_max(y_longCSKYDIRnLeap, num_years=num_years, ntsteps=daylen_long, yrlen=365)
    yST_longCSKYDIRnLeap_MAX=fn_compute_timestep_day_max(yST_longCSKYDIRnLeap, num_years=num_years, ntsteps=daylen_long, yrlen=365)
    
    # Summary data
    data_length_list=[num_years, num_days, daylen_long]
  
    return y_targetTOTAL, y_targetDIRECT, y_targetTOTALnLeap_MAX, y_longCSKYDIRnLeap_MAX, y_target_timestamps_nLeap, yST_longCSKYDIRnLeap_MAX, yST_target_timestamps_nLeap, data_length_list

# ----------------------------------------------------------------------------
def fn_get_input_NmFactors(x3HR_rsds_total_max_est, x_longCSKYdir_maxday, num_days=365, year_list=[2020]):
    """
    Find normalisation factors for Input Data: TOT/CSKYDIR, DIR/DIR_maxday

    Parameters:
    -----------
    x3HR_rsds_total_max_est : list of 2 numpy.ndarray
        1 Year of values from smooth estimate of the function Max Clearsky Total RSDS, fn(time, date)
        [0] 365 days long; [1] 366 days long
    x_longCSKYdir_maxday : numpy.ndarray
        1 Day of values from day identified as having the maximum Clearsky Direct RSDS
    num_days : int
        Number of days in year, assumed to be 365
    year_list : list of int
        List of years to be normalised (so output has correct number of days per year)
    
    Returns:
    --------
    NmFactor_TOTAL : numpy.ndarray
        Normalisation factors for RSDS Total Input
    NmFactor_DIRECT : numpy.ndarray
        Normalisation factors for RSDS Clearsky Direct Input

    """
    # Length of day e.g. 48 timesteps, 30 minutes apart
    daylen_long=x_longCSKYdir_maxday.shape[1]
    
    # Repeat input Max Clearsky Total so total number of days in a year matches years in input year_list
    NmFactor_TOTAL = fn_compute_matched_clim(year_list, x3HR_rsds_total_max_est[0], x3HR_rsds_total_max_est[1])
    NmFactor_DIRECT=np.ones((num_days, daylen_long))
    for dcount in range(num_days): NmFactor_DIRECT[dcount]=x_longCSKYdir_maxday # repeat so same length as training data

    return NmFactor_TOTAL, NmFactor_DIRECT

# ----------------------------------------------------------------------------
def fn_get_target_NmFactors(y30MIN_rsds_total_max_est, num_days=365, year_list=[2020]):
    """
    Find normalisation factors for Input Data : TOT/ClearskyTotal, DIR/Total

    Parameters:
    -----------
    y30MIN_rsds_total_max_est : list of 2 numpy.ndarray
        1 Year of values from smooth estimate of the function Max Clearsky Total RSDS, fn(time, date)
        [0] 365 days long; [1] 366 days long
    num_days : int
        Number of days in year, assumed to be 365
    year_list : list of int
        List of years to be normalised (so output has correct number of days per year)
    
    Returns:
    --------
    NmFactor_TOTAL : numpy.ndarray
        Normalisation factors for RSDS Total Input

    """

    NmFactor_TOTAL = fn_compute_matched_clim(year_list, y30MIN_rsds_total_max_est[0], y30MIN_rsds_total_max_est[1])

    return NmFactor_TOTAL

# ----------------------------------------------------------------------------
def fn_normalise_input(x_shortTOTAL, x_longCSKYdir, SM_Thresh=10.0, NmFactor_TOTAL=1.0, NmFactor_DIRECT=1.0):
    """
    Normalise rsds data for input in ML or MLR models : TOT/CSKYDIR, DIR/DIR_maxday

    Parameters:
    -----------
    x_shortTOTAL : numpy.ndarray
        Total solar irradiance array [no. days, no. subdaily timesteps]
    x_longCSKYdir : numpy.ndarray
        Clearsky Direct solar irradiance array [no. days, no. subdaily timesteps]
    SM_Thresh : float
        Threshold to avoid dividing by very small values so number/[value < SM_Thresh] := 1
    NmFactor_TOTAL : float or numpy.ndarray
        Normalisation factor, assumed have same dimensions as input arrays x_shortTOTAL
    NmFactor_DIRECT : float or numpy.ndarray
        Normalisation factor, assumed have same dimensions as input arrays x_longCSKYdir
    
    Returns:
    --------
    x_shortTOTAL : numpy.ndarray
        Normalised x_shortTOTAL
    x_longCSKYdir : numpy.ndarray
        Normalised x_longCSKYdir

    """

    # Divide Total rsds by Max clearsky Total radiation estimated from max Total rsds
    # - NB need to match year length as leap-year or non-leap-year
    x_shortTOTAL = x_shortTOTAL/NmFactor_TOTAL # x_shortCSKYtot
    x_shortTOTAL = fn_cleanup_ratio(x_shortTOTAL, NmFactor_TOTAL, SM_Thresh=SM_Thresh, set_num=0.0)
    x_longCSKYdir = x_longCSKYdir/NmFactor_DIRECT # x_longCSKYdir_maxdayRP
    x_longCSKYdir = fn_cleanup_ratio(x_longCSKYdir, NmFactor_DIRECT, SM_Thresh=SM_Thresh, set_num=0.0)

    return x_shortTOTAL, x_longCSKYdir

# ----------------------------------------------------------------------------
def fn_normalise_target(y_targetTOTAL, y_targetDIRECT, SM_Thresh=10.0, NmFactor_TOTAL=1):
    """
    Normalise rsds data for target in ML or MLR models : TOT/CSKYDIR, DIR/DIR_maxday

    Parameters:
    -----------
    y_targetTOTAL : numpy.ndarray
        Total solar irradiance array [no. days, no. subdaily timesteps]
    y_targetDIRECT : numpy.ndarray
        Direct solar irradiance array [no. days, no. subdaily timesteps]
    SM_Thresh : float
        Threshold to avoid dividing by very small values so number/[value < SM_Thresh] := 1
    NmFactor_TOTAL : float or numpy.ndarray
        Normalisation factor, assumed have same dimensions as input arrays y_targetTOTAL and y_targetDIRECT
    
    Returns:
    --------
    y_targetTOTAL : numpy.ndarray
        Normalised y_targetTOTAL
    y_targetDIRECT : numpy.ndarray
        Normalised y_targetDIRECT

    """

    # Divide Direct by Total rsds (so can constrain Direct <= Total)
    y_targetDIRECT=y_targetDIRECT/y_targetTOTAL
    y_targetDIRECT = fn_cleanup_ratio(y_targetDIRECT, y_targetTOTAL, SM_Thresh=0.0, set_num=0.0)

    # Divide Total rsds by Max clearsky Total radiation estimated from max Total rsds
    # - NB need to match year length as leap-year or non-leap-year
    y_targetTOTAL = y_targetTOTAL/NmFactor_TOTAL # y_longCSKYtot
    y_targetTOTAL = fn_cleanup_ratio(y_targetTOTAL, NmFactor_TOTAL, SM_Thresh=SM_Thresh, set_num=0.0)

    return y_targetTOTAL, y_targetDIRECT

# ----------------------------------------------------------------------------
def fn_cleanup_ratio(ratioarray, denomarray, SM_Thresh=10.0, set_num=1.0):
    """
    Check for values of ratio that blow up due to dividing by too-small a value
    - i.e. value < SM_Thresh, 
    - Or inf and NaN values
    - Set to have ratio := set_num (eg. 1 so controlled by clear sky)

    Parameters:
    -----------
    ratioarray : numpy.ndarray
        array of ratio data e.g. Total RSDS/Clearsky Total RSDS
    denomarray : numpy.ndarray
        denominator part of ratio e.g. Clearsky Total RSDS
    SM_Thresh : float
        Threshold to define SMall numbers
    set_num : float
        Number to replace values with where the ratio would otherwise blow up

    Returns:
    --------
    outputarray : numpy.ndarray
        Cleaned up version of ratio array

    """
    
    # Setup output array
    outputarray=ratioarray.copy()
    
    # Use set_num to replace nan, inf, and values where denominator very small
    nan_index=np.isnan(ratioarray)
    inf_index=np.isinf(ratioarray)
    small_index=denomarray<SM_Thresh
    naninf_index=nan_index+inf_index+small_index
    outputarray[naninf_index] = set_num

    return outputarray

# ----------------------------------------------------------------------------
def fn_compute_clean_ratio(inputarray, denomarray, SM_Thresh=10.0, set_num=1.0):
    # Check for values of ratio that blow up due to dividing by too-small a value
    # i.e. value < SM_Thresh, 
    # Or inf and NaN values
    # Set to have ratio ;= set_num (eg. 1 so controlled by clear sky)
    
    """
    Compute ratio and check for values of ratio that blow up due to dividing by too-small a value
    - i.e. value < SM_Thresh, 
    - Or inf and NaN values
    - Set to have ratio := set_num (eg. 1 so controlled by clear sky)

    Parameters:
    -----------
    inputarray : numpy.ndarray
        numerator part of ratio data e.g. Total RSDS
    denomarray : numpy.ndarray
        denominator part of ratio e.g. Clearsky Total RSDS
    SM_Thresh : float
        Threshold to define SMall numbers
    set_num : float
        Number to replace values with where the ratio would otherwise blow up

    Returns:
    --------
    outputarray : numpy.ndarray
        array of ratio data inputarray/denomarray e.g. Total RSDS/Clearsky Total RSDS

    """

    # Compute ratio and cleanup
    ratioarray=inputarray/denomarray
    outputarray=fn_cleanup_ratio(ratioarray, denomarray, SM_Thresh=10.0, set_num=1.0)

    return outputarray

# ----------------------------------------------------------------------------
def revertTOT_to_wm2(inputTOTsc, inputMxTOT):
    """
    Un-normalise Total radiation data e.g. TOT/CSKYDIR -> TOT so back in units W/m2
    i.e. inverse of fn_normalise_target

    Parameters:
    -----------
    inputTOTsc : numpy.ndarray
        Total solar irradiance array (normalised) e.g. Total/Clearsky Total RSDS [no. days, no. subdaily timesteps]
    inputMxTOT : numpy.ndarray
        Normalisation factor e.g. Clearsky Total RSDS

    Returns:
    --------
    outputTOT_wm2 : numpy.ndarray
        Total solar irradiance array [no. days, no. subdaily timesteps]

    """

    # Multiply Total rsds by clearsky Total (estimated from max Total rsds)
    # - Max rsds is based on the estimate from the TRaining data, but with year lengths matched to the PRediction data
    outputTOT_wm2=inputTOTsc*inputMxTOT

    if hasattr(outputTOT_wm2, 'detach'): outputTOT_wm2 = outputTOT_wm2.detach().cpu().numpy() # Convert to numpy if pytorch tensor

    return outputTOT_wm2

# ----------------------------------------------------------------------------
def revertDIR_to_wm2(inputDIRsc, inputTOT_wm2):
    """
    Un-normalise Direct radiation data e.g. DIR/TOT -> DIR so back in units W/m2
    i.e. inverse of fn_normalise_target

    Parameters:
    -----------
    inputDIRsc : numpy.ndarray
        Direct solar irradiance array (normalised) e.g. Direct/Total RSDS [no. days, no. subdaily timesteps]
    inputTOT_wm2 : numpy.ndarray
        Normalisation factor e.g. Allsky Total RSDS

    Returns:
    --------
    outputDIR_wm2 : numpy.ndarray
        Direct solar irradiance array [no. days, no. subdaily timesteps]

    """

    # Multiply Direct by Total rsds
    outputDIR_wm2=inputDIRsc*inputTOT_wm2
    
    if hasattr(outputDIR_wm2, 'detach'): outputDIR_wm2 = outputDIR_wm2.detach().cpu().numpy() # Convert to numpy if pytorch tensor

    return outputDIR_wm2
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# FUNCTIONS TO ESTIMATE Max (Mean) Clearsky Total Irradiance as smooth functions of date and time
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def fn_compute_smoothed_clim(sample_clim_pysolar, clim_timestamps, sample_clim=np.zeros(1), ntsteps=8, GiveParams=np.zeros(6), TrainApply=True, GTPYS=False):
    """
    Fit MLR to sample max (or mean) time series for Non-Leap Year data 
      using predictors sin and cos pairs on normalised date' and time'
      and clearsky direct irradiance from pysolar for matching dates and times
    Max RSDS = linear fn(sin(day'), cos(day'), sin(time'), cos(time'), clearsky direct)
    
    Parameters:
    -----------
    sample_clim_pysolar : numpy.ndarray
        Sample climatology from pysolar i.e. sample max clearsky direct
        array [365, ntsteps]
    clim_timestamps : pandas.core.series.Series
        datetimes for all days and timesteps in sample climatologies
    sample_clim : numpy.ndarray
        Sample climatology solar irradiance needed if Training MLR function e.g. sample max of Total SIS
        array [365, ntsteps]
    ntsteps=8 : int
        no. sub-daily timesteps, Default = 8 (3 hour steps), alternative is 48 (30 min steps)
    GiveParams : numpy.ndarray
        Option to input parameters from an already trained model
        Default is np.zeros(6) ie no model diven
    TrainApply=True : boolean
        True => Train MLR on inputs and apply (Default); 
        False => Apply MLR given parameters (GiveParams; in this case sample_clim is not needed)
    GTPYS : boolean
        True ensures sample_clim>sample_clim_pysolar e.g. so that Max Total > Max Direct 
          (not appropriate for mean smoothing as this may not be true)

    Returns:
    --------
    sample_clim_sm_list : list of numpy.ndarray
        Smoothed time series output from fitted MLR function for a leap year and a non-leap year list of datetimes
    MLR_params : numpy.ndarray
        MLR parameters for the fitted MLR function

    """
    
    yrlen=365

    # Compute date time info for final year (for max so don't need to include the year or location)
    date_vec, time_vec = get_sincos_datetime(clim_timestamps, ntsteps=ntsteps, timestamp=True)
    date_veclp, time_veclp = get_sincos_datetime(clim_timestamps, ntsteps=ntsteps, add_day=True, timestamp=True) # Version for leap years   

    # NON-LEAP YEAR: Estimate smoothed climatology function(date, time, pysolar)
    if TrainApply==True: sample_clim_sm, MLR_params = fn_compute_max_sis_pysolardate(date_vec, time_vec, sample_clim_pysolar, sis_MAXarrayIN=sample_clim, TrainApply=True, GTPYS=GTPYS)
    if TrainApply==False: sample_clim_sm, MLR_params = fn_compute_max_sis_pysolardate(date_vec, time_vec, sample_clim_pysolar, GiveParams=GiveParams, TrainApply=False, GTPYS=GTPYS)

    # LEAP YEAR: Compute smoothed climatology function(date, time, pysolar) using fitted MLR_params
    # - Use max pysolar for non-leap year, but substitute 29th Feb with 28th Feb value
    sample_clim_pysolarLeap=np.zeros([366,ntsteps])
    sample_clim_pysolarLeap[0:59]=sample_clim_pysolar[0:59]	# 1st Jan to 28th Feb, 31+28=59, -1=58
    sample_clim_pysolarLeap[59]=sample_clim_pysolar[58]		# 29th Feb, 31+29=60, -1=59
    sample_clim_pysolarLeap[60:366]=sample_clim_pysolar[59:365]	# 1st Mar to 31st Dec

    sample_clim_smLP, MLR_paramsLP = fn_compute_max_sis_pysolardate(date_veclp, time_veclp, sample_clim_pysolarLeap, GiveParams=MLR_params, TrainApply=False)
    sample_clim_sm_list=[sample_clim_sm, sample_clim_smLP]
    
    return sample_clim_sm_list, MLR_params

# ----------------------------------------------------------------------------
def fn_compute_matched_clim(year_List, inputClim, inputClimLP):
    """
    Repeat climatology array so matches the shape and size of desired data array (defined by list of years)
    i.e. concatenate non-leap year (inputClim) and leap year (inputClimLP) climatologies so match the order of years in year_List

    Parameters:
    -----------
    year_List : list of ints
        List of years needed
    inputClim : numpy.ndarray
        Solar irradiance climatology for non-leap years 
        array [365, no. subdaily timesteps]
    inputClimLP : numpy.ndarray
        Solar irradiance climatology for leap years 
        array [366, no. subdaily timesteps]
    
    Returns:
    --------
    outputClim: numpy.ndarray
        Repeated climatology to match length of all years combined

    """

    # Setup output numpy array
    num_tsteps=inputClim.shape[1]	# No. subdaily timesteps
    num_years=len(year_List)		# No. years
    total_days=0
    for yr in year_List: total_days=total_days+get_yr_length(yr) # Total no. days over all years
    outputClim=np.ones((total_days, num_tsteps))

    # Concatenate climatologies
    SampleStart=0 # First data row for a specific year
    for ycount in range(num_years):
        num_days_tmp=get_yr_length(year_List[ycount])  
        if num_days_tmp==365:
            for dd in range(num_days_tmp): outputClim[dd+SampleStart,:] = inputClim[dd,:]
        if num_days_tmp==366:
            for dd in range(num_days_tmp): outputClim[dd+SampleStart,:] = inputClimLP[dd,:]
        SampleStart=SampleStart+num_days_tmp

    return outputClim

# ----------------------------------------------------------------------------
def fn_remove_29feb_array(input_vec, year_LenList, ntsteps=48):
    """
    Removes 29th Feb so can compute climatologies over a mix of leap and non-leap years
    Assumes input_vec matches criteria of key_words
    
    Parameters:
    -----------
    input_vec : numpy.ndarray
        Solar irradiance data on sub-daily timesteps
        array [number days, ntsteps]
    year_LenList : list of ints
        A list containing the number of days in each input year within input_vec
    ntsteps : int
        Number of sub-daily timesteps in a day
    
    Returns:
    --------
    output_vec_nLeap: numpy.ndarray
        Solar irradiance data on sub-daily timesteps but with 29th Feb removed

    """

    # Setup output numpy array
    output_vec_nLeap=np.zeros([365*len(year_LenList),ntsteps])
    # Remove 29th Feb where exists
    for yy in range(len(year_LenList)):
        if yy==0: dindex=np.arange(365)
        if yy>0: dindex=np.arange(365)+sum(year_LenList[0:yy])
        if year_LenList[yy]==366: dindex[(31+28):365]=dindex[(31+28):365]+1 # skip 29th Feb
        output_vec_nLeap[np.arange(365)+365*yy]=input_vec[dindex]

    return output_vec_nLeap

# ----------------------------------------------------------------------------
def fn_compute_timestep_day_max(input_vec, num_years=12, ntsteps=8, yrlen=365):

    """
    Compute climatological maxima for each timestep and day
    Assumes input_vec matches criteria of key_words
    Assumes all years have same length (yrlen) e.g. 365 after removing 29th Feb from any leap-years
    
    Parameters:
    -----------
    input_vec : numpy.ndarray
        Solar irradiance data on sub-daily timesteps
        array [number days, ntsteps]
    num_years : int
        Number of years in input_vec
    ntsteps : int
        Number of sub-daily timesteps in a day
    yrlen : int
        Length of each year (assumed the same e.g. 365 if all non-leap years)
    
    Returns:
    --------
    output_MAXvec: numpy.ndarray
        Climatology maxima for each time step and day over all years

    """

    # Setup output numpy array
    output_MAXvec = np.zeros([yrlen, ntsteps])
    # Compute climatology max for each timestep and day
    for tt in range(ntsteps): 
        for dd in range(yrlen): output_MAXvec[dd,tt]=input_vec[dd:(dd+num_years*yrlen):yrlen, tt].max()

    return output_MAXvec

# ----------------------------------------------------------------------------
def fn_compute_timestep_day_mean(input_vec, num_years=12, ntsteps=8, yrlen=365):
    """
    Compute climatological means for each timestep and day
    Assumes input_vec matches criteria of key_words
    Assumes all years have same length (yrlen) e.g. 365 after removing 29th Feb from any leap-years
    
    Parameters:
    -----------
    input_vec : numpy.ndarray
        Solar irradiance data on sub-daily timesteps
        array [number days, ntsteps]
    num_years : int
        Number of years in input_vec
    ntsteps : int
        Number of sub-daily timesteps in a day
    yrlen : int
        Length of each year (assumed the same e.g. 365 if all non-leap years)
    
    Returns:
    --------
    output_MAXvec: numpy.ndarray
        Climatology average for each time step and day over all years

    """
    
    # Setup output numpy array
    output_MAXvec = np.zeros([yrlen, ntsteps])
    # Compute climatology mean for each timestep and day
    for tt in range(ntsteps): 
        for dd in range(yrlen): output_MAXvec[dd,tt]=input_vec[dd:(dd+num_years*yrlen):yrlen, tt].mean()

    return output_MAXvec

# ----------------------------------------------------------------------------
def fn_compute_max_sis_pysolardate(date_vec, time_vec, pysolar_MAXarrayIN, sis_MAXarrayIN=np.zeros(1), GiveParams=np.zeros(6), TrainApply=True, GTPYS=True):
    """
    Estimate clearsky total by smoothing Max SIS using MLR
    Fit MLR to sample max (or mean) time series for Non-Leap Year data 
    - Create and fit the model to DAYTIME data only; apply to ALL data; re-force NIGHT values ==0
      - predictors sin and cos pairs on normalised date' and time'
      - AND predictor clearsky direct irradiance from pysolar for matching dates and times
    Max RSDS = linear fn(sin(day'), cos(day'), sin(time'), cos(time'), clearsky direct)
    
    Parameters:
    -----------
    date_vec : numpy.ndarray
        sin and cos pairs of normalised date
    time_vec : numpy.ndarray
        sin and cos pairs of normalised time
    pysolar_MAXarrayIN : numpy.ndarray
        Sample climatology from pysolar i.e. sample max clearsky direct
        array [365, ntsteps]
    sis_MAXarrayIN : numpy.ndarray
        Sample climatology solar irradiance e.g. sample max of Total SIS
        array [365, ntsteps]
    GiveParams : numpy.ndarray
        Option to input parameters from an already trained model
        Default is np.zeros(6) ie no model diven
    TrainApply=True : boolean
        True => Train MLR on inputs and Apply (Default); 
        False => Apply MLR using given parameters (GiveParams; in this case sample_clim is not needed)
    GTPYS : boolean
        True ensures sample_clim>sample_clim_pysolar e.g. so that Max Total > Max Direct 
          (not appropriate for mean smoothing as this may not be true)

    Returns:
    --------
    pred_linalg : numpy.ndarray
        Smoothed time series output from fitted MLR function
    MLRmx_params : numpy.ndarray
        MLR parameters for the fitted MLR function

    """

    DayThresh=1.0 # Threshold used when force small values to zero at end
    
    pysolar_MAXarray=pysolar_MAXarrayIN.copy()

    pysolar_MAXarray_vec=pysolar_MAXarray.reshape(pysolar_MAXarray.shape[0]*pysolar_MAXarray.shape[1])
    ones_datetime=date_vec[:,0:1]*0.0+1.0
    datetimeonesMAX_vec=np.column_stack([pysolar_MAXarray_vec, date_vec, time_vec, ones_datetime])
    datetimeonesMAX_vecday=datetimeonesMAX_vec[pysolar_MAXarray_vec > 0.0]

    if TrainApply==True:
        # Compute MLR function for predictors sin and cos of normalised date and time, and clearsky direct from pysolar
        sis_MAXarray=sis_MAXarrayIN.copy()
        # Check that sis (max total) is always greater than clearsky direct
        if GTPYS==True: sis_MAXarray[sis_MAXarray<pysolar_MAXarray]=pysolar_MAXarray[sis_MAXarray<pysolar_MAXarray]
        sis_MAXarray_vec=sis_MAXarray.reshape(sis_MAXarray.shape[0]*sis_MAXarray.shape[1])
        sis_MAXarray_vec_day=sis_MAXarray_vec[pysolar_MAXarray_vec > 0.0] # Should == where max SIS > 0.
        MLRmx_params = np.linalg.lstsq(datetimeonesMAX_vecday, sis_MAXarray_vec_day, rcond=None)[0]

    if TrainApply==False: MLRmx_params=GiveParams.copy() # Use Given Parameters instead of fitting model to data

    # Apply MLR model to datetime and clearsky direct input
    pred_linalg=MLRmx_params[0]*datetimeonesMAX_vec[:,0]+MLRmx_params[1]*datetimeonesMAX_vec[:,1]+MLRmx_params[2]*datetimeonesMAX_vec[:,2]+MLRmx_params[3]*datetimeonesMAX_vec[:,3]+MLRmx_params[4]*datetimeonesMAX_vec[:,4]+MLRmx_params[5]*datetimeonesMAX_vec[:,5]
    pred_linalg=pred_linalg.reshape(pysolar_MAXarray.shape[0],pysolar_MAXarray.shape[1])
    # Force nighttime, small and negative values to be 0
    pred_linalg[pred_linalg<=0.0]=0.0
    pred_linalg[pysolar_MAXarray<DayThresh]=0.0 # 
    pred_linalg[pred_linalg<pysolar_MAXarray]=pysolar_MAXarray[pred_linalg<pysolar_MAXarray]

    return pred_linalg, MLRmx_params
## ----------------------------------------------------------------------------
