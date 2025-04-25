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

python src/scripts/perform_hpopt_across_splits.py --model_type cgr_rf -i data/parsed_jacs_data/splits -o data/parsed_jacs_data/${solvent_classification}_solvs/cgr_rf_results -t $target_cols --num_jobs_per_hpopt 1 --num_trials 20 --n_jobs 5
