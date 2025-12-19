'''
Created on 08.08.25
@author Rosie Eade

Code provided to:
Read in SARAH-3 Surface incoming total radiation data using xarray.
Store data as panda array for a single point location and year of data.
+ Emulate a CMIP6 style Global Climate Model (low temporal and spatial resolution) with 3hr mean starting 00:30 (+offset)

--------------------
Run by typing:
module purge
module load python/meso-3.11  # example module to load python environment
pip install pysolar
python -i write_data_files/write_rsds_sarah3_to_panda_xarray.py 'SIS' 'France' 2018
python write_data_files/write_rsds_sarah3_to_panda_xarray.py 'SIS' 'France' 2014

Or use a script to submit, e.g.
run_write_rsds_sarah3_to_panda_xarray.bash

Requirements:
# Memory needed = 1Gb per run (single point location and year)
# Time needed = 35 minutes per run (single point location and year)

--------------------
INPUT:
SARAH-3 instantaneous surface incoming total radiation data (SIS/SID netcdf files + ancillary file for time offset)

Command line input:
  Choose variable 'SIS' or 'SID'
  Choose specific point location e.g. 'France', see function get_site_info
  Choose specific single year e.g. 2014

Hardcoded input, see main below for:
  # GLOBAL DATA PATHS
  def get_directory
  
  Set TEST_EXAMPLE==True to test on just one day

Reads in netcdf files using xarray.

--------------------
OUTPUT:
STORE radiation timeseries as panda data file with columns:
- time, SIS, Zenith, Clearsky
These represent
- datetime stamp, Surface incoming total radiation from netcdf files
- Zenith angle, Clearsky direct estimate from pysolar
  https://github.com/pingswept/pysolar

Output Files are created in 3 different formats:

1. RSDS instantaneous 30 minute single point output
   - timestamp = offset for location as satellite moves
   - SIS for point location and 30min intervals
   - Zenith and Clearsky for same point location and 30min intervals
   e.g. SARAH3_DATE_SIS_30MIN_2021_France.csv

2. RSDS Low resolution emulating 3 hour mean for GCM-like grid box:
   - SIS 3 hourly mean and spatial average representing a GCM-like grid box
   - Zenith and Clearsky averaged over time for the point at the centre of the GCM-like grid box
   - timestamp = start of time averaging period 
     (3 hourly window, computes average on 6x30min values starting 00:30+)
   e.g. IPSL CMIP6 low-resolution model: 2.5x1.27 (lon 144 by lat 143)
        SARAH3_DATE_SIS_3HR_IPSLCM6_2021_France.csv

3. Clearsky Direct instantaneous 30 minute single point output
   - timestamp = 00:00, 00:30, ... i.e. without offset for location of satellite
   - Zenith and Clearsky for same point location and 30min intervals
   e.g. PYSOLAR_DATE_SID_30MIN_2021_France.csv

Output all to 1dp accuracy (same as input netcdf files)
--------------------
SARAH-3 satellite estimate of rsds is available to download as:
* 30-min instantaneous values (used here to compute 3HR mean)
  - 1983-present. 1 file per day, 0.05 x 0.05 grid (2600 by 2600)
  - (SIS/SID Year 2014 missing 2 whole days 2/12/2014 3/12/2014 so overwrite with data from neighbouring day)
- Put in a catch to fill in missing data: fn_interp_md_rsds_array
  (e.g. missing data values flagged by -999)

SARAH-3 data also available as:
* daily mean values 
  - 1983-present. 1 file per day, 0.05 x 0.05 grid (2600 by 2600), 7MB
  - has SIS and clear sky version SISC in same file
* monthly mean values
  - 1983-present. 
https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V003 # (SIS)
https://navigator.eumetsat.int/product/EO:EUM:DAT:0863

'''

import datetime
now=datetime.datetime.now()
print("--------------")
print("start time")
print(now.time())
print("--------------")

import os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from pysolar.solar import *
import calendar
import pdb # used for checking code part way through

# ----------------------------------------------------------------------------
# GLOBAL DATA PATHS
# ----------------------------------------------------------------------------

# DIRECTORY For Output
output_dir=Path('data/sarah3obs_subdaily_ts/')
if not os.path.exists(output_dir): os.makedirs(output_dir)

# -------------------------------------------------------------------#
def get_directory(year=2018, rsdsType='SIS'):
    """
    Write code here to identify directories containing SARAH-3 data
    Used in functions:
      fn_read_one_date_sarah3_30MIN_np_xr
      fn_read_one_date_spavg_sarah3_30MIN_np_xr
    e.g. may be spread over multiple directories
    """
    
    # Data paths for SIS and SID Data (SARAH-3)
    sarah3_dir_30MIN=Path('data/sarah3nc/inst/')
    if rsdsType=='SID': sarah3_dir_30MIN=Path('data/sarah3nc/instDir/')
    
    #if year < 2019 and rsdsType=='SIS': sarah3_dir_30MIN='' # <2019
    #if year < 2019 and rsdsType=='SID': sarah3_dir_30MIN='' # <2019

    # Ancillary File For Datetime info (offset due to satellite motion)
    sarah3_file_30MIN_ancil=Path('data/sarah3nc/ancil/AuxilaryData_SARAH-3.nc')

    return sarah3_dir_30MIN, sarah3_file_30MIN_ancil


# ----------------------------------------------------------------------------
# GLOBAL VARIABLES
# ----------------------------------------------------------------------------

month_listTxt=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_listTno=['01','02','03','04','05','06','07','08','09','10','11','12']
month_listNumDaysNL=[31,28,31,30,31,30,31,31,30,31,30,31]
month_listNumDaysL=[31,29,31,30,31,30,31,31,30,31,30,31]
day_listTno=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
# -------------------------------------------------------------------#


# -------------------------------------------------------------------#
def fn_read_one_date_sarah3_30MIN_np_xr(year=2019, mon=1, day=1, latIN=48.70999908, lonIN=2.20000005, rsdsType='SIS'):
    """
    Read 1 day at a time (netcdf file) using xarray
    Only load the required spatial point (nearest neighbour)
    rsdsType = 'SIS' or 'SID'
    Output as a numpy array and a datetime array

    Parameters:
    -----------
    year : int
        Year to select
    mon : int
        Month to select
    day : int
        Day to select
    latIN : float
        Latitude coordinate
    lonIN : float
        Longitude coordinate
    rsdsType : string
        Choice of variable e.g. 'rsds' or 'rsdscs'
    
    Returns:
    --------
    np.array(scube_30MIN.values) : numpy.ndarray
        30 minute radiation data at point location, shape (48)

    datetime30MIN_list : list of datetimes
        date times of 30min data values on standard time steps
    datetime30MINoffset_list : list of datetimes

    date times of 30min data values on offset time steps (due to satellite movement)

    act_lat, act_lon : floats
        Actual lat and lon values of nearest grid box

    """
    
    # Get directory and file info (this function needs to be adjusted to own data paths)
    sarah3_dir_30MIN, sarah3_file_30MIN_ancil = get_directory(year=year, rsdsType=rsdsType)
    
    # Define variable names based on rsdsType
    if rsdsType == 'SIS': 
        var_name = 'SIS' # 'surface_downwelling_shortwave_flux_in_air'
    if rsdsType == 'SID': 
        var_name = 'SID' # 'surface_direct_downwelling_shortwave_flux_in_air'
    
    # Handle longitude wrapping
    circ_lon = lonIN
    if lonIN > 180: 
        circ_lon = lonIN - 360
    
    # Create date code for required file
    date_code = month_listTno[mon-1] + day_listTno[day-1]
    print(date_code)
    
    # Build file path
    fpath = sarah3_dir_30MIN / Path(rsdsType + 'in' + str(year) + date_code + '*MA.nc')
    
    # Special case for corrupt data files (all data missing, only checked 2013-2024)
    if year == 2014 and mon == 12 and day == 2: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20141201*MA.nc')
    if year == 2014 and mon == 12 and day == 3: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20141204*MA.nc')
    if year == 2016 and mon == 10 and day == 15: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20161013*MA.nc')
    if year == 2016 and mon == 10 and day == 16: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20161014*MA.nc')
    if year == 2016 and mon == 10 and day == 17: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20161018*MA.nc')
	   

    # Get file list (should be just 1 file)
    #file_list = glob.glob(fpath) # doesn't work with Path
    parent_dir = fpath.parent if fpath.parent.exists() else Path('.')
    pattern = fpath.name
    file_list = list(parent_dir.glob(pattern))

    if not file_list:
        raise FileNotFoundError(f"No files found matching pattern: {fpath}")
    print(file_list)
    
    # First pass: find lat/lon indices using the first file (without loading data)
    with xr.open_dataset(file_list[0]) as ds:
        # Convert coordinate arrays to numpy for nearest neighbor search
        lat_values = ds.lat.values
        lon_values = ds.lon.values
        
        # Find nearest neighbor indices for lat/lon
        lat_idx = np.argmin(np.abs(lat_values - latIN))
        lon_idx = np.argmin(np.abs(lon_values - circ_lon))
        
        # Get actual lat/lon values
        act_lat = float(lat_values[lat_idx])
        act_lon = float(lon_values[lon_idx])
    
    # Second pass: load only the required spatial point from each file
    data_arrays = []
    for file in file_list:
        with xr.open_dataset(file) as ds:
            # Check variable exists
            if var_name not in ds.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in dataset. Available variables: {list(ds.data_vars.keys())}")

            # Load only the single spatial point for all time steps
            point_data = ds[var_name].isel(lat=lat_idx, lon=lon_idx).load()
            # Drop the scalar lat/lon coordinates to avoid dimension issues
            point_data = point_data.drop_vars(['lat', 'lon'], errors='ignore')
            data_arrays.append(point_data)
    
    # Concatenate along time dimension
    if len(data_arrays) == 1:
        scube_30MIN = data_arrays[0]
    else:
        scube_30MIN = xr.concat(data_arrays, dim='time')
    
    # Load time adjustment data - only the required point
    # - Not tested on years before 2013
    with xr.open_dataset(sarah3_file_30MIN_ancil) as ancil_ds:
        if year <= 2005: 
            sarah_time_add = ancil_ds['time_difference_MVIRI'].isel(lat=lat_idx, lon=lon_idx).load()
            print('WARNING: Not tested MVIRI years ie 2005 and earlier')
        if year >= 2006: 
            sarah_time_add = ancil_ds['time_difference_SEVIRI'].isel(lat=lat_idx, lon=lon_idx).load()
        
        # Drop scalar coordinates if they exist
        sarah_time_add = sarah_time_add.drop_vars(['lat', 'lon'], errors='ignore')
    
    # Adjust time coordinates
    time_adjustment = sarah_time_add.values
    
    # Convert time to datetime objects
    time_coord = scube_30MIN.time
    datetime30MIN_list = []
    datetime30MINoffset_list = []
    
    for time_val in time_coord.values:
        # Convert numpy datetime64 to python datetime
        dt = time_val.astype('datetime64[s]').astype(datetime.datetime)
        
        # Ensure timezone is UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        
        # Special case for corrupt data files (all data missing)
        if year == 2014 and mon == 12 and day == 2: 
            dt = dt.replace(day=2)
        if year == 2014 and mon == 12 and day == 3: 
            dt = dt.replace(day=3)
        if year == 2016 and mon == 10 and day == 15: 
            dt = dt.replace(day=15)
        if year == 2016 and mon == 10 and day == 16: 
            dt = dt.replace(day=16)
        if year == 2016 and mon == 10 and day == 17: 
            dt = dt.replace(day=17)

        datetime30MIN_list.append(dt)
        
        # Adjust time by the time difference (assuming it's in appropriate units)
        # * xarray automatically converts time units to nanoseconds since epoch 
        #   as part of its internal time handling, regardless of what the original 
        #   units were in the NetCDF file.

        dt = dt + datetime.timedelta(seconds=float(time_adjustment/1000000000.0))
        
        datetime30MINoffset_list.append(dt)

    return np.array(scube_30MIN.values), datetime30MIN_list, datetime30MINoffset_list, act_lat, act_lon

# -------------------------------------------------------------------#


def fn_read_one_date_spavg_sarah3_30MIN_np_xr(year=2019, mon=1, day=1, latIN=[47.5,48.8], lonIN=[1.25,3.75], rsdsType='SIS'):
    """
    Read 1 day at a time (netcdf file) using xarray
    Compute spatial average (spavg) for a region of grid points (representing a GCM-like box)
    rsdsType = 'SIS' or 'SID'
    Output as a numpy array and a datetime array

    Parameters:
    -----------
    year : int
        Year to select
    mon : int
        Month to select
    day : int
        Day to select
    latIN : float
        Latitude coordinate
    lonIN : float
        Longitude coordinate
    rsdsType : string
        Choice of variable e.g. 'rsds' or 'rsdscs'
    
    Returns:
    --------
    np.array(scube_30MIN.values) : numpy.ndarray
        30 minute radiation data, average over region, shape (48)

    datetime30MIN_list : list of datetimes
        date times of 30min data values on standard time steps
    datetime30MINoffset_list : list of datetimes

    date times of 30min data values on offset time steps (due to satellite movement)

    act_lat, act_lon : floats
        Actual lat and lon values of nearest grid box

    """
    
    # Get directory and file info (this function needs to be adjusted to own data paths)
    sarah3_dir_30MIN, sarah3_file_30MIN_ancil = get_directory(year=year, rsdsType=rsdsType)
    
    # Set variable name based on rsdsType
    if rsdsType == 'SIS': 
        var_name = 'SIS' # 'surface_downwelling_shortwave_flux_in_air'
    if rsdsType == 'SID': 
        var_name = 'SID' # 'surface_direct_downwelling_shortwave_flux_in_air'
    
    # Create date code
    date_code = month_listTno[mon-1] + day_listTno[day-1]
    print(date_code)
    
    # Build file path
    fpath = sarah3_dir_30MIN / Path(rsdsType + 'in' + str(year) + date_code + '*MA.nc')
    
    # Special case for corrupt data files (all data missing, only checked 2013-2024)
    if year == 2014 and mon == 12 and day == 2: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20141201*MA.nc')
    if year == 2014 and mon == 12 and day == 3: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20141204*MA.nc')
    if year == 2016 and mon == 10 and day == 15: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20161013*MA.nc')
    if year == 2016 and mon == 10 and day == 16: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20161014*MA.nc')
    if year == 2016 and mon == 10 and day == 17: 
        fpath = sarah3_dir_30MIN / Path(rsdsType + 'in20161018*MA.nc')
    
    # Get file list (should be just 1 file)
    #files = glob.glob(fpath) # doesn't work with Path
    parent_dir = fpath.parent if fpath.parent.exists() else Path('.')
    pattern = fpath.name
    files = list(parent_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {fpath}")
    
    # Load data using xarray with spatial subsetting to minimize memory usage
    # Process files one by one to avoid loading full global data
    # Coordinate names
    lat_name = 'lat'
    lon_name = 'lon'    

    regional_datasets = []
    for file in files:
        with xr.open_dataset(file) as ds:
            # Check variable exists
            if var_name not in ds.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in dataset. Available variables: {list(ds.data_vars.keys())}")
            
            # Load only the spatial subset into memory
            regional_subset = ds[var_name].sel({lat_name: slice(latIN[0], latIN[1]), lon_name: slice(lonIN[0], lonIN[1])}).load()
            regional_datasets.append(regional_subset)

    # Concatenate the regional subsets along time dimension
    if len(regional_datasets) == 1:
        regional_data = regional_datasets[0]
    else:
        regional_data = xr.concat(regional_datasets, dim='time')
    
    # Calculate area weights for spatial averaging
    lat_coord = regional_data[lat_name]
    lon_coord = regional_data[lon_name]
    
    # Calculate area weights using cosine of latitude
    if len(lat_coord) > 1:
        # Create weights array that broadcasts correctly
        cos_lat = np.cos(np.radians(lat_coord))
        # Create 2D weight array matching the spatial dimensions
        weights = xr.ones_like(regional_data.isel(time=0))
        weights = weights * cos_lat
    else:
        weights = xr.ones_like(regional_data.isel(time=0))
    
    # Compute weighted spatial average
    scube_30MIN = regional_data.weighted(weights).mean(dim=[lat_name, lon_name])
    
    # Get actual lat/lon coordinates (center of selected region)
    act_lat = float(lat_coord.mean())
    act_lon = float(lon_coord.mean())
    
    # Load time adjustment data - only the required point
    # - Not tested on years before 2013
    with xr.open_dataset(sarah3_file_30MIN_ancil) as ancil_ds:
        if year <= 2005: 
            sarah_time_add = ancil_ds['time_difference_MVIRI'].sel({lat_name: slice(latIN[0], latIN[1]), lon_name: slice(lonIN[0], lonIN[1])}).load()
            print('WARNING: Not tested MVIRI years ie 2005 and earlier')
        if year >= 2006: 
            sarah_time_add = ancil_ds['time_difference_SEVIRI'].sel({lat_name: slice(latIN[0], latIN[1]), lon_name: slice(lonIN[0], lonIN[1])}).load()

        lat_values = sarah_time_add.lat.values
        lon_values = sarah_time_add.lon.values
        
        # Find nearest neighbor indices for lat/lon
        lat_idx = np.argmin(np.abs(lat_values - act_lat))
        lon_idx = np.argmin(np.abs(lon_values - act_lon))
        sarah_time_add = sarah_time_add[lat_idx,lon_idx]

        # Drop scalar coordinates if they exist
        sarah_time_add = sarah_time_add.drop_vars(['lat', 'lon'], errors='ignore')
    
    # Adjust time coordinates
    time_adjustment = sarah_time_add.values
    
    # Convert time coordinate to datetime objects
    time_coord = scube_30MIN.time
    datetime30MIN_list = []
    datetime30MINoffset_list = []
    
    for time_val in time_coord.values:
        # Convert numpy datetime64 to python datetime
        dt = time_val.astype('datetime64[s]').astype(datetime.datetime)
        
        # Ensure timezone is UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        
        # Special case for corrupt data files (all data missing)
        if year == 2014 and mon == 12 and day == 2: 
            dt = dt.replace(day=2)
        if year == 2014 and mon == 12 and day == 3: 
            dt = dt.replace(day=3)
        if year == 2016 and mon == 10 and day == 15: 
            dt = dt.replace(day=15)
        if year == 2016 and mon == 10 and day == 16: 
            dt = dt.replace(day=16)
        if year == 2016 and mon == 10 and day == 17: 
            dt = dt.replace(day=17)

        datetime30MIN_list.append(dt)
        
        # Adjust time by the time difference (assuming it's in appropriate units)
        # * xarray automatically converts time units to nanoseconds since epoch 
        #   as part of its internal time handling, regardless of what the original 
        #   units were in the NetCDF file.

        dt = dt + datetime.timedelta(seconds=float(time_adjustment/1000000000.0))
        
        datetime30MINoffset_list.append(dt)
    
    # Memory cleanup - explicitly delete large objects
    del regional_data, weights
  
    return np.array(scube_30MIN.values), datetime30MIN_list, datetime30MINoffset_list, act_lat, act_lon

# -------------------------------------------------------------------#
def fn_interp_nonmissing_neigbours_circular(in_vec, tTime, md_val=-999):
    """
    Find nearest non-missing neighbours
    Assumes vector is circular

    Parameters:
    -----------
    in_vec : numpy.ndarray
        1 day of sub-daily data that contains missing data
    tTime : int
        position of single missing data point within in_vec
    md_val : float
        missing data value
    
    Returns:
    --------
    newY : float
        value to replace missing data with

    """
    # repeat single day timeseries so have 3 day cycle 
    # - so can find nearest neigbours earlier AND later in day
    in_vec_tile=np.tile(in_vec,3)
    # Define point of interest in 2nd day within 3 day cycle
    tTime_tile=tTime+len(in_vec)
    # Identify nearest non-missing values
    nm_index=np.where(in_vec_tile != md_val)[0]
    resL=nm_index[np.where(nm_index < tTime_tile)].max()
    resU=nm_index[np.where(nm_index > tTime_tile)].min()
    # Interpolate between nearest non-missing values to replace missing value
    newY = in_vec_tile[resL] + (in_vec_tile[resU]-in_vec_tile[resL])*(tTime_tile-resL)/(resU-resL)    
    return newY

# -------------------------------------------------------------------#
def fn_interp_md_rsds_array(in_array, in_daily_clearsky, md_val=-999):
    """
    Fill in missing data using linear interpolation
    Being careful of zero values (night) and consecutive missing values

    Parameters:
    -----------
    in_array : numpy.ndarray
        multiple days of sub-daily SIS or SID data, shape (number of days, number timesteps)
    in_daily_clearsky : numpy.ndarray
        multiple days of sub-daily clearsky direct data (from pysolar), shape (number of days, number timesteps)
    md_val : float
        missing data value
    
    Returns:
    --------
    out_array : numpy.ndarray
        multiple days of sub-daily SIS or SID data, with missing data replaced

    """
    out_array=in_array.copy()
    md_index=np.where(in_array == md_val) # tuple of arrays of indices
    npoints=len(md_index[0])
    print("NumMissingPOINTS")
    print(npoints)
    # Fill in missing nighttime values with zero
    for pcount in range(npoints):
        tDay=md_index[0][pcount]  #in_array[tDay,:]   # All times for a single day
        tTime=md_index[1][pcount] #in_array[:,tTime] # All days at a single time
        # Check if time step should be zero
        if in_daily_clearsky[tDay,tTime]<=0: out_array[tDay,tTime]=0
    md_index=np.where(out_array == md_val) # tuple of arrays of indices (just non-night times)
    npoints=len(md_index[0])
    print(npoints)
    # Fill in missing daytime values with linear interpolation
    for pcount in range(npoints):
        tDay=md_index[0][pcount]
        tTime=md_index[1][pcount]
        # Linearly interpolate with nearest neighbours if not zero
        # Check if time step should be zero
        if in_daily_clearsky[tDay,tTime]<=0: out_array[tDay,tTime]=0
        if in_daily_clearsky[tDay,tTime]>0: 
            DVec=out_array[tDay,:]
            res=fn_interp_nonmissing_neigbours_circular(DVec, tTime, md_val=-999)
            # Linear interpolation overestimates near sunrise and sunset
            if res > in_daily_clearsky[tDay,tTime]: res=in_daily_clearsky[tDay,tTime]
            out_array[tDay,tTime]=res
    return out_array


# -------------------------------------------------------------------#
def get_site_info(site_name='France'):
    """
    Some hard coded example site information for France and other locations in CFsubhr MIP of CMIP6
    * Can generalise this by choosing an observation location point and then
      reading in an example GCM field to find coordinates of the nearest grid box
    NB CFsubhr location is not generally the centre of an actual GCM grid box (interpolated or use nearest box?)
       or indeed the centre of an actual SARAH3 gridbox

    Parameters:
    -----------
    site_name : string
        Name of location in dictionary site_info
    
    Returns:
    --------
    site_num : int
        CFsubhr MIP site location index

    lat_given, lon_given : floats
        lat and lon values of CFsubhr MIP site location

    latRange_given, lonRange_given : lists of floats
        lat and lon values of closest points in SARAH3 data (0.05 deg grid)

    """

    #-------------
    # Example Location Info 
    # Name, site number, [lat, lon], [lat range], [lon range]
    # - site number refers to CMIP6 CFsubhr location if available (else set as -1)
    # - lat and lon bounds refer to example CMIP6 GCM grid box (IPSL-CM6A-LR used here)
    site_info = {'France': [90, 48.70999908, 2.20000005, [47.53521156311035, 48.80281639099121], [1.25, 3.75]],
                 'Senegal': [115, 15.0, -17.0, [14.577464580535889, 15.845069885253906], [-18.75, -16.25]]}
    #-------------

    site_num=site_info[site_name][0]	# Refers to CFsubhr if available
    lat_given=site_info[site_name][1]	# Location latitude
    lon_given=site_info[site_name][2]	# Location longitude

    # Adjust Location lat and lon range to nearest gridpoints in SARAH3 data 0.05deg grid
    latRange_given=site_info[site_name][3]
    latRange_given.sort()
    latRange_given[0]=int(latRange_given[0]/0.05)*0.05+0.025
    latRange_given[1]=int(latRange_given[1]/0.05)*0.05-0.025

    lonRange_given=site_info[site_name][4]
    lonRange_given.sort()
    lonRange_given[0]=int(lonRange_given[0]/0.05)*0.05+0.025
    lonRange_given[1]=int(lonRange_given[1]/0.05)*0.05-0.025

    return site_num, lat_given, lon_given, latRange_given, lonRange_given


# -------------------------------------------------------------------#
def main():
        
    #####################################################
    # INPUT SECTION
    rsdsType = str(sys.argv[1]) # 'SIS' 'SID'
    site_name = str(sys.argv[2]) # 'France' 'Senegal'
    yearS = int(sys.argv[3]) # 2018
    
    TEST_EXAMPLE=False # True => test on just 1 day of data
    
    print(site_name)
    print(yearS)
    print(rsdsType)
    
    # Define output directory
    fdir=output_dir / site_name  # Output path for csv data files, see GLOBAL DATA PATHS
    if not os.path.exists(fdir): os.makedirs(fdir)
    
    site_num, lat_given, lon_given, latRange_given, lonRange_given=get_site_info(site_name=site_name)
    yrS_txt=str(yearS)
    
    #Check if a leap year
    month_listNumDays=month_listNumDaysNL
    if calendar.isleap(yearS): month_listNumDays=month_listNumDaysL
    total_days=365
    if calendar.isleap(yearS): total_days=366
    
    if TEST_EXAMPLE==True: total_days=1
    tfn='' # test file name
    if TEST_EXAMPLE==True: tfn='TEST'

    #__________________________________
    # Read in SARAH-3 as numpy array 30MIN rsds (W/m2) instantaneous
    # - for point daily_patterns_orig and regional average dailyRAVG_patterns_orig
    # - and store datetime info
    # - NB datetime for regional average is based on centre of grid box, so subtly different to point value
    daily_patterns_orig = np.zeros((total_days, 48)) # assumes all years and days available
    dailyRAVG_patterns_orig = np.zeros((total_days, 48)) # assumes all years and days available
    dateone=datetime.datetime(yearS, 1, 1, tzinfo=datetime.timezone.utc)
    daily_date_list=[]
    daily_standarddate_list=[]
    dcount=0
    for dd in range(total_days):
        time_changeS = datetime.timedelta(days=dd)
        tmpDate = dateone + time_changeS
        tmp_30min_out, tmp_datetime_list, tmp_datetimeoffset_list, act_lat, act_lon=fn_read_one_date_sarah3_30MIN_np_xr(year=tmpDate.year, mon=tmpDate.month, day=tmpDate.day,latIN=lat_given,lonIN=lon_given, rsdsType=rsdsType)
        tmpRAVG_30min_out, tmpRAVG_datetime_list, tmpRAVG_datetimeoffset_list, actRAVG_lat, actRAVG_lon=fn_read_one_date_spavg_sarah3_30MIN_np_xr(year=tmpDate.year, mon=tmpDate.month, day=tmpDate.day,latIN=latRange_given,lonIN=lonRange_given, rsdsType=rsdsType)
        tmp_30min_out[np.argwhere(np.isnan(tmp_30min_out))]=-999 # xarray converts to nan
        tmpRAVG_30min_out[np.argwhere(np.isnan(tmpRAVG_30min_out))]=-999 # xarray converts to nan
        daily_patterns_orig[dcount,:]=tmp_30min_out
        dailyRAVG_patterns_orig[dcount,:]=tmpRAVG_30min_out
        daily_date_list.append(tmp_datetimeoffset_list)
        daily_standarddate_list.append(tmp_datetime_list)
        dcount=dcount+1

    print("--------------")
    print("T: Read 30MIN")
    print(datetime.datetime.now().time())
    print("--------------")



    #__________________________________
    # Estimate OFFSET TIME clear-sky direct radiation info using pysolar (W/m2) 
    # - What if get value = -0.0 ? Catch and set to zero.

    # 30 MIN, POINT LOCATION, at SARAH3 offset timestep (e.g. 00:10, 00:40, ... for France point)
    # - Compute at centre of SARAH3 high resolution grid box (actual lat and lon)
    daily_clearsky = np.zeros((total_days, 48))
    daily_azimuth = np.zeros((total_days, 48))
    daily_altitude = np.zeros((total_days, 48))
    daily_zenith = np.zeros((total_days, 48))
    daily_date_list1D=[]
    dcount=0
    for yy in range(total_days):
        tmpDateList=daily_date_list[yy]
        for tt in range(48):
            date=tmpDateList[tt]
            daily_clearsky[yy,tt]=radiation.get_radiation_direct(date, get_altitude(act_lat, act_lon, date))* math.cos(math.radians(90-get_altitude(act_lat, act_lon, date)))
            daily_azimuth[yy,tt]=get_azimuth(act_lat, act_lon, date)
            daily_altitude[yy,tt]=get_altitude(act_lat, act_lon, date)
            daily_zenith[yy,tt]=90.0-get_altitude(act_lat, act_lon, date)
            daily_date_list1D.append(date)
    nan_index=np.argwhere(np.isnan(daily_clearsky))
    for nn in range(len(nan_index)): daily_clearsky[nan_index[nn][0],nan_index[nn][1]]=0.0 # Check this is always ok???
    
    # 3HR MEAN, represent REGIONAL MEAN, at SARAH3 offset timestep
    # - Compute at centre of SARAH3 low resolution grid box (lat and lon at centre of GCM-like grid box)
    dailyRAVG_clearsky = np.zeros((total_days, 48))
    dailyRAVG_azimuth = np.zeros((total_days, 48))
    dailyRAVG_altitude = np.zeros((total_days, 48))
    dailyRAVG_zenith = np.zeros((total_days, 48))
    dailyRAVG_date_list1D=[]
    dcount=0
    for yy in range(total_days):
        tmpDateList=daily_date_list[yy]
        for tt in range(48):
            date=tmpDateList[tt]
            dailyRAVG_clearsky[yy,tt]=radiation.get_radiation_direct(date, get_altitude(actRAVG_lat, actRAVG_lon, date))* math.cos(math.radians(90-get_altitude(actRAVG_lat, actRAVG_lon, date)))
            dailyRAVG_azimuth[yy,tt]=get_azimuth(actRAVG_lat, actRAVG_lon, date)
            dailyRAVG_altitude[yy,tt]=get_altitude(actRAVG_lat, actRAVG_lon, date)
            dailyRAVG_zenith[yy,tt]=90.0-get_altitude(actRAVG_lat, actRAVG_lon, date)
            dailyRAVG_date_list1D.append(date)
    nanRAVG_index=np.argwhere(np.isnan(dailyRAVG_clearsky))
    for nn in range(len(nanRAVG_index)): dailyRAVG_clearsky[nanRAVG_index[nn][0],nanRAVG_index[nn][1]]=0.0 # Check this is always ok???

    #--------------------------------
    # Fill in any missing data found in SARAH3 netcdf files
    daily_patterns_fill=fn_interp_md_rsds_array(daily_patterns_orig, daily_clearsky, md_val=-999)
    dailyRAVG_patterns_fill=fn_interp_md_rsds_array(dailyRAVG_patterns_orig, dailyRAVG_clearsky, md_val=-999)

    
    #__________________________________
    # Create pandas data frames of data
    
    # -----------------------
    # 30 min original data
    datetime_1Dnp=np.array(daily_date_list1D)
    sis_1Dnp=daily_patterns_fill.reshape(total_days*48)
    zen_1Dnp=daily_zenith.reshape(total_days*48)
    csky_1Dnp=daily_clearsky.reshape(total_days*48)
    
    # Version with missing data filled by interpolated values
    datetime_1Ddf=pd.to_datetime(np.array(daily_date_list1D))
    dataset = pd.DataFrame({rsdsType: sis_1Dnp, 'Zenith': zen_1Dnp, 'Clearsky': csky_1Dnp}, index=datetime_1Ddf)
    dataset.to_csv(fdir / Path('SARAH3_DATE_'+rsdsType+'_30MIN_'+yrS_txt+'_'+site_name+tfn+'.csv'), index=True, index_label='time', float_format='%6.1f') # index=False prevents an index column from being added.

    # Version with missing data kept in
    sis_1DnpRaw=daily_patterns_orig.reshape(total_days*48)
    dataset2 = pd.DataFrame({rsdsType: sis_1DnpRaw}, index=datetime_1Ddf)
    dataset2.to_csv(fdir / Path('SARAH3_DATE_'+rsdsType+'miss_30MIN_'+yrS_txt+'_'+site_name+tfn+'.csv'), index=True, index_label='time') # index=False prevents an index column from being added.

    # -----------------------
    # 3 hourly mean version of data
    # - Clear sky and zenith also averaged over time
    # - Timestamp is start of the 3hourly window
    # - For sis, zenith and clearsky data, shift all rows by 1 as 3HR mean starts 00:30(+offset) to match GCM 
    # - (substitute final row with first row, assumes both zero anyway)
    sisRAVG_1Dnp=dailyRAVG_patterns_fill.reshape(total_days*48)
    sisRAVG_1DnpCP=sisRAVG_1Dnp.copy()
    sisRAVG_1DnpCP[0:(total_days*48-1)]=sisRAVG_1Dnp[1:total_days*48]
    sisRAVG_1DnpCP[total_days*48-1]=sisRAVG_1Dnp[0]
    sisRAVG_1Dnp=sisRAVG_1DnpCP.copy()
    zenRAVG_1Dnp=dailyRAVG_zenith.reshape(total_days*48)
    zenRAVG_1DnpCP=zenRAVG_1Dnp.copy()
    zenRAVG_1DnpCP[0:(total_days*48-1)]=zenRAVG_1Dnp[1:total_days*48]
    zenRAVG_1DnpCP[total_days*48-1]=zenRAVG_1Dnp[0]
    zenRAVG_1Dnp=zenRAVG_1DnpCP.copy()
    cskyRAVG_1Dnp=dailyRAVG_clearsky.reshape(total_days*48)
    cskyRAVG_1DnpCP=cskyRAVG_1Dnp.copy()
    cskyRAVG_1DnpCP[0:(total_days*48-1)]=cskyRAVG_1Dnp[1:total_days*48]
    cskyRAVG_1DnpCP[total_days*48-1]=cskyRAVG_1Dnp[0]
    cskyRAVG_1Dnp=cskyRAVG_1DnpCP.copy()
    datasetRAVG2 = pd.DataFrame({rsdsType: sisRAVG_1Dnp, 'Zenith': zenRAVG_1Dnp, 'Clearsky': cskyRAVG_1Dnp}, index=datetime_1Ddf)
    SIS_3HR=datasetRAVG2[rsdsType].resample(rule='3h', origin='start').mean() # include origin else time defaults to nearest 3 hour eg 0000, 0300 so no timesteps match those of the 30min file
    ZENRAVG_3HR=datasetRAVG2['Zenith'].resample(rule='3h', origin='start').mean()
    CSKYRAVG_3HR=datasetRAVG2['Clearsky'].resample(rule='3h', origin='start').mean()
    dataset3HR = pd.DataFrame({rsdsType: SIS_3HR, 'Zenith': ZENRAVG_3HR, 'Clearsky': CSKYRAVG_3HR}) # , index=datetime_1Ddf)
    dataset3HR.to_csv(fdir / Path('SARAH3_DATE_'+rsdsType+'_3HR_IPSLCM6_'+yrS_txt+'_'+site_name+tfn+'.csv'), index=True, index_label='time', float_format='%6.1f') # index=False prevents an index column from being added.


    #__________________________________
    # Estimate STANDARD TIME clear-sky radiation using pysolar (W/m2) 
    # - What if get value = -0.0 ? Catch and set to zero.

    # 30 MIN, POINT LOCATION, standard timestep (00:00, 00:30, ...)
    # - Compute at centre of SARAH3 high resolution grid box (actual lat and lon)
    daily_clearsky = np.zeros((total_days, 48))
    daily_azimuth = np.zeros((total_days, 48))
    daily_altitude = np.zeros((total_days, 48))
    daily_zenith = np.zeros((total_days, 48))
    daily_date_list1D=[]
    dcount=0
    for yy in range(total_days):
        tmpDateList=daily_standarddate_list[yy]
        for tt in range(48):
            date=tmpDateList[tt]
            daily_clearsky[yy,tt]=radiation.get_radiation_direct(date, get_altitude(act_lat, act_lon, date))* math.cos(math.radians(90-get_altitude(act_lat, act_lon, date)))
            daily_azimuth[yy,tt]=get_azimuth(act_lat, act_lon, date)
            daily_altitude[yy,tt]=get_altitude(act_lat, act_lon, date)
            daily_zenith[yy,tt]=90.0-get_altitude(act_lat, act_lon, date)
            daily_date_list1D.append(date)
    nan_index=np.argwhere(np.isnan(daily_clearsky))
    for nn in range(len(nan_index)): daily_clearsky[nan_index[nn][0],nan_index[nn][1]]=0.0 # Check this is always ok???

    # 30 MIN, POINT LOCATION, represent REGIONAL MEAN, standard timestep (00:00, 00:30, ...)
    # - Compute at centre of SARAH3 low resolution grid box (lat and lon at centre of GCM-like grid box)
    dailyRAVG_clearsky = np.zeros((total_days, 48))
    dailyRAVG_azimuth = np.zeros((total_days, 48))
    dailyRAVG_altitude = np.zeros((total_days, 48))
    dailyRAVG_zenith = np.zeros((total_days, 48))
    dailyRAVG_date_list1D=[]
    dcount=0
    for yy in range(total_days):
        tmpDateList=daily_standarddate_list[yy]
        for tt in range(48):
            date=tmpDateList[tt]
            dailyRAVG_clearsky[yy,tt]=radiation.get_radiation_direct(date, get_altitude(actRAVG_lat, actRAVG_lon, date))* math.cos(math.radians(90-get_altitude(actRAVG_lat, actRAVG_lon, date)))
            dailyRAVG_azimuth[yy,tt]=get_azimuth(actRAVG_lat, actRAVG_lon, date)
            dailyRAVG_altitude[yy,tt]=get_altitude(actRAVG_lat, actRAVG_lon, date)
            dailyRAVG_zenith[yy,tt]=90.0-get_altitude(actRAVG_lat, actRAVG_lon, date)
            dailyRAVG_date_list1D.append(date)
    nanRAVG_index=np.argwhere(np.isnan(dailyRAVG_clearsky))
    for nn in range(len(nanRAVG_index)): dailyRAVG_clearsky[nanRAVG_index[nn][0],nanRAVG_index[nn][1]]=0.0 # Check this is always ok???


    #__________________________________
    # Create pandas data frames of STANDARD TIME data
    # - This output is the same regardless of whether computing for SIS or SID
    
    # -----------------------
    # 30 min original data
    datetime_1Dnp=np.array(daily_date_list1D)
    zen_1Dnp=daily_zenith.reshape(total_days*48)
    csky_1Dnp=daily_clearsky.reshape(total_days*48)
    
    datetime_1Ddf=pd.to_datetime(np.array(daily_date_list1D))
    dataset = pd.DataFrame({'Zenith': zen_1Dnp, 'Clearsky': csky_1Dnp}, index=datetime_1Ddf)
    dataset.to_csv(fdir / Path('PYSOLAR_DATE_SID_30MIN_'+yrS_txt+'_'+site_name+tfn+'.csv'), index=True, index_label='time', float_format='%6.1f') # index=False prevents an index column from being added.

    # -----------------------
    # 3 hourly mean version of data
    # - Clear sky and zenith also averaged over time
    # - Timestamp is start of the 3hourly window
    # - For zenith and clearsky data, shift all rows by 1 as 3HR mean starts 00:30 to match GCM
    # - (substitute final row with first row, assumes both zero anyway)
    zenRAVG_1Dnp=dailyRAVG_zenith.reshape(total_days*48)
    zenRAVG_1DnpCP=zenRAVG_1Dnp.copy()
    zenRAVG_1DnpCP[0:(total_days*48-1)]=zenRAVG_1Dnp[1:total_days*48]
    zenRAVG_1DnpCP[total_days*48-1]=zenRAVG_1Dnp[0]
    zenRAVG_1Dnp=zenRAVG_1DnpCP.copy()
    cskyRAVG_1Dnp=dailyRAVG_clearsky.reshape(total_days*48)
    cskyRAVG_1DnpCP=cskyRAVG_1Dnp.copy()
    cskyRAVG_1DnpCP[0:(total_days*48-1)]=cskyRAVG_1Dnp[1:total_days*48]
    cskyRAVG_1DnpCP[total_days*48-1]=cskyRAVG_1Dnp[0]
    cskyRAVG_1Dnp=cskyRAVG_1DnpCP.copy()
    datasetRAVG2 = pd.DataFrame({'Zenith': zenRAVG_1Dnp, 'Clearsky': cskyRAVG_1Dnp}, index=datetime_1Ddf)
    ZENRAVG_3HR=datasetRAVG2['Zenith'].resample(rule='3h', origin='start').mean()
    CSKYRAVG_3HR=datasetRAVG2['Clearsky'].resample(rule='3h', origin='start').mean()
    dataset3HR = pd.DataFrame({'Zenith': ZENRAVG_3HR, 'Clearsky': CSKYRAVG_3HR}) # , index=datetime_1Ddf)
    dataset3HR.to_csv(fdir / Path('PYSOLAR_DATE_SID_3HR_IPSLCM6_'+yrS_txt+'_'+site_name+tfn+'.csv'), index=True, index_label='time', float_format='%6.1f') # index=False prevents an index column from being added.

    print('----LAT LON----')
    print(site_name)
    print('act_lat')
    print(act_lat)
    print('act_lon')
    print(act_lon)
    print('actRAVG_lat')
    print(actRAVG_lat)
    print('actRAVG_lon')
    print(actRAVG_lon)
    print('---------------')
    
    #Check output csv files
    #readData=pd.read_csv(fdir / Path('SARAH3_DATE_SIS_30MIN_'+yrS_txt+'_'+site_name+tfn+'.csv'))
    #readData=pd.read_csv(fdir / Path('SARAH3_DATE_SIS_3HRto30min_'+yrS_txt+'_'+site_name+tfn+'.csv'))
    #readData=pd.read_csv(fdir / Path('SARAH3_DATE_SID_30MIN_'+yrS_txt+'_'+site_name+tfn+'.csv'))
    #readData=pd.read_csv(fdir / Path('SARAH3_DATE_SID_3HRto30min_'+yrS_txt+'_'+site_name+tfn+'.csv'))

#    pdb.set_trace()

if __name__ == '__main__':
    main()

    now = datetime.datetime.now()
    print("--------------")
    print("end time")
    print(now.time())
    print("--------------")

