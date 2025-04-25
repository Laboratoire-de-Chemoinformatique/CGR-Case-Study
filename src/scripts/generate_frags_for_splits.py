"""Fragment generation for all splits."""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits_dir", type=str, default="data/parsed_jacs_data/splits"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5]
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        help="Number of folds for cross validation",
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    commands = []
    for seed in args.seeds:
        if args.n_folds is not None:
            for i in range(args.n_folds):
                command = f"python src/scripts/generate_cgr_frags.py --input_csv {args.splits_dir}/split_seed_{seed}/fold_{i + 1}/data.csv --n_components 1500 --run_pca -o {args.splits_dir}/split_seed_{seed}/fold_{i + 1}"
                commands.append(command)
        else:
            command = f"python src/scripts/generate_cgr_frags.py --input_csv {args.splits_dir}/split_seed_{seed}/data_seed_{seed}.csv --n_components 1500 --run_pca -o {args.splits_dir}/split_seed_{seed}"
            commands.append(command)

    for command in commands:
        os.system(command)
