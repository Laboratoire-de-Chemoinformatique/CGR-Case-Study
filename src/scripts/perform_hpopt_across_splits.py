"""
Performs hyperparameter optimisation for the models.

Optimises hyperparameters for a single split (repetition) and a single fold.
"""

import argparse
import multiprocessing
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--splits_dir",
        help="Directory containing folders corresponding to each fold (and their respective splits).",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Directory to save the output files.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_type",
        help="Model to perform hyperparameter optimisation for.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--target_cols",
        help="Target columns to predict.",
        nargs="+",
        default=["coarse_solvent", "base"],
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        help="Number of jobs to run in parallel.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--n_splits",
        help="Number of splits to perform hyperparameter optimisation for.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--num_jobs_per_hpopt",
        help="Number of jobs to run in parallel for each hyperparameter optimisation.",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--num_trials",
        help="Number of trials to run for each hyperparameter optimisation.",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--job_name",
        help="Name of the job, to be used when performing hpopt on cgr-mtnn.",
        default="cgr_mtnn_hpopt",
        type=str,
    )
    parser.add_argument(
        "--no_likelihood_ranking",
        help="Whether to use likelihood ranking or not, in the case where this is not used, and there are multiple target columns, optuna will perform multi-objective optimisation, using the top-1 accuracies of each individual target.",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "cgr_rf":
        command = "python src/modelling/rf.py"
    elif args.model_type == "cgr_gb":
        command = "python src/modelling/gb.py"
    elif args.model_type in ["cgr_mtnn", "morgan_mtnn"]:
        command = "python src/modelling/cgr_mtnn_opt.py"
    elif args.model_type == "cgr_dmpnn":
        command = "python src/modelling/cgr_dmpnn.py"
    else:
        raise NotImplementedError(
            f"Model type {args.model_type} not implemented. Currently supported models are: cgr_rf, cgr_gb, cgr_mtnn, cgr_dmpnn."
        )

    commands = []
    for i in range(args.n_splits):
        input_dir = f"{args.splits_dir}/split_seed_{i + 1}/fold_1"
        output_dir = f"{args.output_dir}/split_seed_{i + 1}/fold_1"
        data_csv = "data.csv"

        if set(args.target_cols) == {"coarse_solvent", "base"}:
            solvent_classification = "coarse"
        elif set(args.target_cols) == {"fine_solvent", "base"}:
            solvent_classification = "fine"
        elif set(args.target_cols) == {"exact_solvent", "exact_base"}:
            solvent_classification = "exact"
        else:
            solvent_classification = ""

        if args.model_type in ["cgr_rf", "cgr_gb"]:
            command += f" -i {input_dir} -o {output_dir} -d {data_csv} -t {' '.join(args.target_cols)} -c {args.num_jobs_per_hpopt} -n {args.num_trials} --target_cols {' '.join(args.target_cols)} --model_output_dir models/{solvent_classification}_solvs/{args.model_type}/split_seed_{i + 1}/fold_1 --config_save_dir hpopt/{solvent_classification}_solvs/{args.model_type}_results/split_seed_{i + 1}/fold_1"
            if args.model_type == "cgr_gb":
                command += " --gpu"
        elif args.model_type in ["cgr_mtnn", "morgan_mtnn"]:
            command += f" -i {input_dir}/{data_csv} -o {output_dir} -t {' '.join(args.target_cols)} -n {args.num_trials} -j {args.job_name} -m {args.model_type}"
            if args.model_type == "cgr_mtnn":
                command += f" -f {input_dir}/id_to_frag.pkl"
            if args.no_likelihood_ranking:
                command += " --no_likelihood_ranking"
        elif args.model_type == "cgr_dmpnn":
            command += f" -i {input_dir}/{data_csv} -o {output_dir} -t {' '.join(args.target_cols)} -n {args.num_trials} -hp hpopt/{solvent_classification}_solvs/{args.model_type}_results/split_seed_{i + 1}/fold_1 --num_train_epochs 50 --num_hpopt_epochs 20 -s models/{solvent_classification}_solvs/{args.model_type}/split_seed_{i + 1}/fold_1 -c {args.num_jobs_per_hpopt}"

        commands.append(command)

    with multiprocessing.Pool(args.n_jobs) as pool:
        pool.map(os.system, commands)
        pool.close()
        pool.join()
