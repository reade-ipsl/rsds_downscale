# rsds_downscale
Downscale solar irradiance to increase temporal resolution at a point location

# Method:

Train a multiple-linear regression (MLR) model and a machine learning (ML) model on SARAH-3 satellite observations to convert low spatial and temporal resolution solar Total irradiance data into the target 30 minute output at a point location (Total and also Direct only).

# Solar Irradiance Data:

* Download SARAH-3 data using wget script in write_data_files (including the ancillary file containing timestamp adjustments)

* Create a set of input and target time series files to train the models using python code in write_data_files (for a specific location)
- Input = CMIP6 General Circulation Model (GCM) 'like' time series, emulated using SARAH-3 data averaged over a region similar to a GCM sized grid box and a 3-hour time window
- Target = SARAH-3 time series at single point within the input region (GCM grid box) on 30 minute timesteps (instantaneous) for Total and also Direct only solar irradiance.

# Downscaling Models:

* See python code in downscale_ts
- rsdsMain_ObsTraining.py Trains a single ML model with some performance analysis (tables of metrics).
- rsdsMain_ObsPrediction.py Trains a single ML model and outputs predicted time series as .csv files, with some performance analysis (tables of metrics and plots of example output). 
- rsdsMain_ObsPredictionLOOPs.py Trains the ML model multiple times and outputs a summary of the performance so ML model can be tuned with respect to the loss function, batch size etc. (tables of metrics and plots of example output). *.com files provided to concatenate/montage output across differently tuned ML models.

# Requirements:
pytorch https://pytorch.org/ (BSD-3 license)
pysolar https://github.com/pingswept/pysolar/ (GPL-3.0 license)
see python code for other python needs

# License:
CC BY 4.0