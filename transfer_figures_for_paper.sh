#!/bin/bash

DIR="./Figures"
 _DIR="./_Figures"

 if [ -d "$DIR" ]; then
    echo "'$DIR' found and now copying files in '_$DIR' for backup, please wait ..."
    cp -R $DIR $_DIR
    echo "deleting '$DIR' ....."
   rm -r $DIR
 fi
 
 
mkdir -p ./Figures/
mkdir -p ./Figures/Pendulum/ 
mkdir -p ./Figures/Pendulum/cubic/
mkdir -p ./Figures/Pendulum/linear/
mkdir -p ./Figures/Pendulum/quad/

mkdir -p ./Figures/NLO/ 
mkdir -p ./Figures/NLO/cubic/
mkdir -p ./Figures/NLO/linear/
mkdir -p ./Figures/NLO/quad/

mkdir -p ./Figures/LV/ 
mkdir -p ./Figures/LV/cubic/
mkdir -p ./Figures/LV/linear/
mkdir -p ./Figures/LV/quad/

mkdir -p ./Figures/NLS/ 
mkdir -p ./Figures/NLS/cubic/
mkdir -p ./Figures/NLS/linear/
mkdir -p ./Figures/NLS/quad/
mkdir -p ./Figures/NLS/linear_opinf/

mkdir -p ./Figures/Wave/ 
mkdir -p ./Figures/Wave/cubic/
mkdir -p ./Figures/Wave/linear/
mkdir -p ./Figures/Wave/quad/
mkdir -p ./Figures/Wave/linear_opinf/


cp .//Results/Pendulum/phase_plot.pdf ./Figures/Pendulum/
cp ./Results/Pendulum/error_analysis.pdf ./Figures/Pendulum/
cp ./Results/Pendulum/cubic/plot_groundtruth_time_0.pdf ./Figures/Pendulum/cubic/
cp ./Results/Pendulum/linear/plot_learned_time_0.pdf ./Figures/Pendulum/linear/
cp ./Results/Pendulum/quad/plot_learned_time_0.pdf ./Figures/Pendulum/quad/
cp ./Results/Pendulum/cubic/plot_learned_time_0.pdf ./Figures/Pendulum/cubic/
cp ./Results/Pendulum/cubic/plot_groundtruth_time_5.pdf ./Figures/Pendulum/cubic/
cp ./Results/Pendulum/linear/plot_learned_time_5.pdf ./Figures/Pendulum/linear/
cp ./Results/Pendulum/quad/plot_learned_time_5.pdf ./Figures/Pendulum/quad/
cp ./Results/Pendulum/cubic/plot_learned_time_5.pdf ./Figures/Pendulum/cubic/


cp ./Results/NLO/phase_plot.pdf ./Figures/NLO/
cp ./Results/NLO/error_analysis.pdf ./Figures/NLO/
cp ./Results/NLO/cubic/plot_groundtruth_time_21.pdf ./Figures/NLO/cubic/
cp ./Results/NLO/linear/plot_learned_time_21.pdf ./Figures/NLO/linear/
cp ./Results/NLO/quad/plot_learned_time_21.pdf ./Figures/NLO/quad/
cp ./Results/NLO/cubic/plot_learned_time_21.pdf ./Figures/NLO/cubic/
cp ./Results/NLO/cubic/plot_groundtruth_time_1.pdf ./Figures/NLO/cubic/
cp ./Results/NLO/linear/plot_learned_time_1.pdf ./Figures/NLO/linear/
cp ./Results/NLO/quad/plot_learned_time_1.pdf ./Figures/NLO/quad/
cp ./Results/NLO/cubic/plot_learned_time_1.pdf ./Figures/NLO/cubic/


cp ./Results/LV/phase_plot.pdf ./Figures/LV/
cp ./Results/LV/error_analysis.pdf ./Figures/LV/
cp ./Results/LV/cubic/plot_groundtruth_time_14.pdf ./Figures/LV/cubic/
cp ./Results/LV/linear/plot_learned_time_14.pdf ./Figures/LV/linear/
cp ./Results/LV/quad/plot_learned_time_14.pdf ./Figures/LV/quad/
cp ./Results/LV/cubic/plot_learned_time_14.pdf ./Figures/LV/cubic/
cp ./Results/LV/cubic/plot_groundtruth_time_5.pdf ./Figures/LV/cubic/
cp ./Results/LV/linear/plot_learned_time_5.pdf ./Figures/LV/linear/
cp ./Results/LV/quad/plot_learned_time_5.pdf ./Figures/LV/quad/
cp ./Results/LV/cubic/plot_learned_time_5.pdf ./Figures/LV/cubic/

cp ./Results/NLS/cubic/svd_plot.pdf ./Figures/NLS/cubic/
cp ./Results/NLS/cubic/energy_plot.pdf ./Figures/NLS/cubic/
cp ./Results/NLS/cubic/pod_coeffs_ground_truth_0.pdf ./Figures/NLS/cubic/
cp ./Results/NLS/linear/pod_coeffs_learned_0.pdf ./Figures/NLS/linear/
cp ./Results/NLS/quad/pod_coeffs_learned_0.pdf ./Figures/NLS/quad/
cp ./Results/NLS/cubic/pod_coeffs_learned_0.pdf ./Figures/NLS/cubic/

cp ./Results/NLS/linear/pod_coeffs_ground_truth_phasespace_testing_0.pdf ./Figures/NLS/linear/
cp ./Results/NLS/linear/pod_coeffs_learned_phasespace_testing_0.pdf ./Figures/NLS/linear/
cp ./Results/NLS/quad/pod_coeffs_learned_phasespace_testing_0.pdf ./Figures/NLS/quad/
cp ./Results/NLS/cubic/pod_coeffs_learned_phasespace_testing_0.pdf ./Figures/NLS/cubic/
cp ./Results/NLS/linear_opinf/pod_coeffs_learned_phasespace_testing_0.pdf ./Figures/NLS/linear_opinf/

cp ./Results/NLS/linear/pod_coeffs_ground_truth_phasespace_testing_1.pdf ./Figures/NLS/linear/
cp ./Results/NLS/linear/pod_coeffs_learned_phasespace_testing_1.pdf ./Figures/NLS/linear/
cp ./Results/NLS/quad/pod_coeffs_learned_phasespace_testing_1.pdf ./Figures/NLS/quad/
cp ./Results/NLS/cubic/pod_coeffs_learned_phasespace_testing_1.pdf ./Figures/NLS/cubic/
cp ./Results/NLS/linear_opinf/pod_coeffs_learned_phasespace_testing_1.pdf ./Figures/NLS/linear_opinf/

cp ./Results/NLS/linear/q_compare_0.png ./Figures/NLS/linear/
cp ./Results/NLS/linear/p_compare_0.png ./Figures/NLS/linear/


cp ./Results/Wave/cubic/svd_plot.pdf ./Figures/Wave/cubic/
cp ./Results/Wave/cubic/energy_plot.pdf ./Figures/Wave/cubic/
cp ./Results/Wave/testing_err.pdf ./Figures/Wave/

cp ./Results/Wave/linear/q_compare_1.png ./Figures/Wave/linear/
cp ./Results/Wave/linear/p_compare_1.png ./Figures/Wave/linear/

cp ./Results/Wave/linear/q_compare_2.png ./Figures/Wave/linear/
cp ./Results/Wave/linear/p_compare_2.png ./Figures/Wave/linear/



