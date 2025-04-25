#!/bin/bash

# Parse command line arguments for solvent classification and coarse_hparams
use_coarse_hparams=false
while getopts s:c flag
do
    case "${flag}" in
        s) solvent_classification=${OPTARG};;
        c) use_coarse_hparams=true;;
    esac
done

# Check if solvent classification is either 'coarse' or 'fine'
if [[ "$solvent_classification" != "coarse" && "$solvent_classification" != "fine" ]]; then
    echo "Invalid solvent classification. Please use 'coarse' or 'fine'."
    exit 1
fi

# Activate the conda environment 'ml-env'
source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate ml-env

# Run the test models script with the specified solvent classification and optional coarse_hparams flag
if $use_coarse_hparams; then
    python src/scripts/test_models_across_splits.py --model_type pop -s "$solvent_classification" --n_splits 5 --n_jobs 5 --n_folds 5 --use_coarse_hparams
    python src/scripts/test_models_across_splits.py --model_type knn -s "$solvent_classification" --n_splits 5 --n_jobs 5 --n_folds 5 --use_coarse_hparams
    python src/scripts/test_models_across_splits.py --model_type cgr_rf -s "$solvent_classification" --n_splits 5 --n_jobs 25 --n_folds 5 --use_coarse_hparams
else
    python src/scripts/test_models_across_splits.py --model_type pop -s "$solvent_classification" --n_splits 5 --n_jobs 5 --n_folds 5
    python src/scripts/test_models_across_splits.py --model_type knn -s "$solvent_classification" --n_splits 5 --n_jobs 5 --n_folds 5
    python src/scripts/test_models_across_splits.py --model_type cgr_rf -s "$solvent_classification" --n_splits 5 --n_jobs 25 --n_folds 5
fi