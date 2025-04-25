#!/bin/bash

# Activate the conda environment 'ml-env'
source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate dl-env

python src/scripts/test_models_across_splits.py --model_type cgr_mtnn -s exact --n_splits 5 --n_jobs 1 --n_folds 5 --predict
python src/scripts/test_models_across_splits.py --model_type morgan_mtnn -s exact --n_splits 5 --n_jobs 1 --n_folds 5 --predict