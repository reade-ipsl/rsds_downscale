#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=30
#SBATCH --mem 2G
#SBATCH --output=log_files/rsdsMain_ObsPredictionLOOPs.log

# Example script to run rsdsMain_ObsPredictionLOOPs.py
#
# sbatch downscale_ts/run_rsdsMain_ObsPredictionLOOPs.bash &
# check job status
# sacct --format=jobname%50,jobid,elapsed,timelimit,maxrss,reqmem,exitcode # last day's jobs
# sacct --format=jobname%50,jobid,elapsed,timelimit,maxrss,reqmem,exitcode -S 2025-12-09 # jobs since date
# squeue -u username
# exit code 0.0 = exited successfully.
#
# Example limits to memory available e.g. 3840 MB per node
#
# *--mem Memory needed = 1.2Gb per run (per run with 10 loops)
# *--time Time needed = 15min (per run with 10 loops)
# *--output Change to own file path for log of run. Default is to make log_files directory in current directory.
#
# *Storage needed = 11Mb

mkdir log_files

module purge # avoids interference from other envs
module load pytorch/2.6.0 # example module to load python environment

# Choice of Training Test Case (partitioning of years for training and prediction)
ttc=3

# Choice of batch size for ML model
bsize=16

python -i downscale_ts/rsdsMain_ObsPredictionLOOPs.py 'W1000' $bsize '10cS' $ttc 'France'

echo $suite COMPLETE

# montage file1.png file2.png -geometry +10+10 -tile 2x1 file3.png 
