# Concatenate table output from downscale_ts/rsdsMain_ObsPredictionLOOPs.py
# So can compare results for e.g. different loss functions, different batch sizes etc.
#
# chmod u+x downscale_ts/concat_loop_output.com
# downscale_ts/concat_loop_output.com
#
# Example below is for specific training and test case

site_name=France
TrainTestCase=TR2013_2022France_PR_2023_2024France

cd results/rsds_ObsPredictionLOOPs

mkdir -p table_txt/$site_name

# Performance Metrics for ALLSKY TOTAL solar radiation
head -n 3 test2_10cS_B16W1000/$site_name/rsdsTOT_SARAH3_10cS_e30B16W1000_$TrainTestCase\_PR_PerfMetricsMN.txt > table_txt/$site_name/rsdsTOT_SARAH3_10cS_e30B16WALL_$TrainTestCase\_PR_PerfMetricsMN.txt 
tail -n +4 -q test2_10cS_B16W*/$site_name/rsdsTOT_SARAH3_10cS_e30B16W*_$TrainTestCase\_PR_PerfMetricsMN.txt >> table_txt/$site_name/rsdsTOT_SARAH3_10cS_e30B16WALL_$TrainTestCase\_PR_PerfMetricsMN.txt 

# Performance Metrics for ALLSKY DIRECT solar radiation
head -n 3 test2_10cS_B16W1000/$site_name/rsdsDIR_SARAH3_10cS_e30B16W1000_$TrainTestCase\_PR_PerfMetricsMN.txt > table_txt/$site_name/rsdsDIR_SARAH3_10cS_e30B16WALL_$TrainTestCase\_PR_PerfMetricsMN.txt
tail -n +4 -q test2_10cS_B16W*/$site_name/rsdsDIR_SARAH3_10cS_e30B16W*_$TrainTestCase\_PR_PerfMetricsMN.txt >> table_txt/$site_name/rsdsDIR_SARAH3_10cS_e30B16WALL_$TrainTestCase\_PR_PerfMetricsMN.txt

# Performance Metrics for CLEARSKY TOTAL solar radiation
head -n 3 test2_10cS_B16W1000/$site_name/rsdsCSKYTOT_SARAH3_10cS_e30B16W1000_$TrainTestCase\_PR_PerfMetricsMN.txt > table_txt/$site_name/rsdsCSKYTOT_SARAH3_10cS_e30B16WALL_$TrainTestCase\_PR_PerfMetricsMN.txt
tail -n +4 -q test2_10cS_B16W*/$site_name/rsdsCSKYTOT_SARAH3_10cS_e30B16W*_$TrainTestCase\_PR_PerfMetricsMN.txt >> table_txt/$site_name/rsdsCSKYTOT_SARAH3_10cS_e30B16WALL_$TrainTestCase\_PR_PerfMetricsMN.txt

