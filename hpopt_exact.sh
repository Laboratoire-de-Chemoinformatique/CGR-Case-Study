#!/bin/bash

python src/scripts/perform_hpopt_across_splits.py --model_type cgr_mtnn -i data/parsed_jacs_data/splits -o hpopt/exact_solvs/cgr_mtnn_results -t exact_base exact_solvent --num_jobs_per_hpopt 1 --num_trials 50 --n_jobs 1 --job_name cgr_mtnn --no_likelihood_ranking

python src/scripts/perform_hpopt_across_splits.py --model_type morgan_mtnn -i data/parsed_jacs_data/splits -o hpopt/exact_solvs/morgan_mtnn_results -t exact_base exact_solvent --num_jobs_per_hpopt 1 --num_trials 50 --n_jobs 1 --job_name morgan_mtnn --no_likelihood_ranking