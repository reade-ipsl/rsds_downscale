# Montage graph output from downscale_ts/$site_name/rsdsMain_ObsPredictionLOOPs.py
# So can compare results for e.g. different loss functions, different batch sizes etc.
#
# chmod u+x downscale_ts/montage_loop_output.com
# downscale_ts/montage_loop_output.com
#
# Example below is for specific training and test case

site_name=France
TrainTestCase=TR2013_2022France_PR_2023_2024France

cd results/rsds_ObsPredictionLOOPs

mkdir -p matrix_plots/$site_name

# -----------------------
# Single Loss Functions

montage test2_10cS_B16W1000/$site_name/rsdsTOT_SARAH3_10cS_e30B16W1000_$TrainTestCase\_PR_MEAN_ANNts.png test2_10cS_B16W3033/$site_name/rsdsTOT_SARAH3_10cS_e30B16W3033_$TrainTestCase\_PR_MEAN_ANNts.png -geometry +10+10 -tile 1x2 matrix_plots/$site_name/rsdsTOT_SARAH3_10cS_e30B16W1ALL_$TrainTestCase\_PR_MEAN_ANNts.png

montage test2_10cS_B16W1000/$site_name/rsdsTOT_30MIN_SARAH3_10cS_e30B16W1000_$TrainTestCase\_PR_ML10cS1_B5SUBsampleMEANCSKY_DAYtsMN.png test2_10cS_B16W3033/$site_name/rsdsTOT_30MIN_SARAH3_10cS_e30B16W3033_$TrainTestCase\_PR_ML10cS1_B5SUBsampleMEANCSKY_DAYtsMN.png -geometry +10+10 -tile 1x2 matrix_plots/$site_name/rsdsTOT_30MIN_SARAH3_10cS_e30B16W1ALL_$TrainTestCase\_PR_ML10cS1_B5SUBsampleMEANCSKY_DAYtsMN.png

montage test2_10cS_B16W1000/$site_name/Examples/rsdsTOT_SARAH3_10cS_e30B16W1000_$TrainTestCase\_PR_Day0090.png test2_10cS_B16W3033/$site_name/Examples/rsdsTOT_SARAH3_10cS_e30B16W3033_$TrainTestCase\_PR_Day0090.png -geometry +10+10 -tile 1x2 matrix_plots/$site_name/rsdsTOT_SARAH3_10cS_e30B16W1ALL_$TrainTestCase\_PR_Day0090.png

montage test2_10cS_B16W1000/$site_name/Examples/rsdsTOT_SARAH3_10cS_e30B16W1000_$TrainTestCase\_PR_Day0181.png test2_10cS_B16W3033/$site_name/Examples/rsdsTOT_SARAH3_10cS_e30B16W3033_$TrainTestCase\_PR_Day0181.png -geometry +10+10 -tile 1x2 matrix_plots/$site_name/rsdsTOT_SARAH3_10cS_e30B16W1ALL_$TrainTestCase\_PR_Day0181.png

