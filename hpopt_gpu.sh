#!/bin/bash
while getopts s: flag
do
    case "${flag}" in
        s) solvent_classification=${OPTARG};;
    esac
done

# Check if solvent classification is either 'coarse' or 'fine'
if [[ "$solvent_classification" != "coarse" && "$solvent_classification" != "fine" ]]; then
    echo "Invalid solvent classification. Please use 'coarse' or 'fine'."
    exit 1
fi

if [[ "$solvent_classification" == "coarse" ]]; then
    target_cols="coarse_solvent base"
else
    target_cols="fine_solvent base"
fi

source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate ml-env

python src/scripts/perform_hpopt_across_splits.py --model_type cgr_gb -i data/parsed_jacs_data/splits -o data/parsed_jacs_data/${solvent_classification}_solvs/cgr_gbm_results -t $target_cols --num_jobs_per_hpopt 1 --num_trials 50 --n_jobs 1

conda deactivate
conda activate dl-env

python src/scripts/perform_hpopt_across_splits.py --model_type cgr_mtnn -i data/parsed_jacs_data/splits -o hpopt/${solvent_classification}_solvs/cgr_mtnn_results -t $target_cols --num_jobs_per_hpopt 1 --num_trials 50 --n_jobs 1 --job_name cgr_mtnn
python src/scripts/perform_hpopt_across_splits.py --model_type cgr_dmpnn -i data/parsed_jacs_data/splits -o data/parsed_jacs_data/${solvent_classification}_solvs/cgr_dmpnn_results -t $target_cols --num_trials 50 --n_jobs 1 --num_jobs_per_hpopt 32