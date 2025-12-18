#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=120
#SBATCH --mem 1G
#SBATCH --output=log_files/write_rsds_sarah3_to_panda_xarray.log

# Example script to run write_rsds_sarah3_to_panda_xarray.py
#
# sbatch run_write_rsds_sarah3_to_panda_xarray.bash &
# check job status
# sacct --format=jobname%50,jobid,elapsed,timelimit,maxrss,reqmem,exitcode # last day's jobs
# sacct --format=jobname%50,jobid,elapsed,timelimit,maxrss,reqmem,exitcode -S 2025-01-27 # jobs since date
# squeue -u username
# exit code 0.0 = exited successfully.
#
# Example limits to memory available e.g. 3840 MB per node
#
# *--mem Memory needed = 1Gb per run (single point location and year)
# *--time Time needed = 35min per run (single point location and year) quicker depending on data disk usage?
# *--output Change to own file path for log of run. Default is to make log_files directory in current directory.
#
# *Storage needed = 4Mb per year (SIS + SID + pysolar combined)

mkdir log_files

module purge # avoids interference from other envs
module load python/meso-3.11 # example module to load python environment
pip install pysolar

#for year in {2021..2024}
for year in 2018
 do
 python write_rsds_sarah3_to_panda_xarray.py 'SIS' 'France' $year
 python write_rsds_sarah3_to_panda_xarray.py 'SID' 'France' $year
 done

echo $suite COMPLETE
