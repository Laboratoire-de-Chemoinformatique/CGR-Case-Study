"""Splits the dataset into training, validation and testing sets, depending
on the number of seeds provided. Optionally, for each random seed (and therefore
data split), the dataset can be split a further n times for cross validation.
"""

import argparse
import os

import polars as pl
from sklearn.model_selection import StratifiedKFold, train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input_csv",
        type=str,
        required=True,
        help="Path to input csv file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Seeds for random number generator, passing multiple seeds will create multiple datasets.",
    )
    parser.add_argument(
        "-v",
        "--val_size",
        type=float,
        default=0.1,
        help="Size of validation set.",
    )
    parser.add_argument(
        "-f",
        "--num_folds",
        type=int,
        default=1,
        help="Number of folds for cross validation.",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--stratification_label",
        type=str,
        default="fine_solvent",
        help="Column to use for stratification.",
        required=False,
    )
    return parser.parse_args()


def split_data(
    df: pl.DataFrame,
    val_size: float,
    seed: int,
    output_dir: str,
    num_folds: int,
    stratification_label: str = "fine_solvent",
) -> None:
    """Splits the given DataFrame into training, validation, and test sets, ensuring that all entries with the same
    'parsed_rxn' value are in the same dataset. The resulting datasets are saved as CSV files in the specified
    output directory.

    :param df: The input DataFrame containing the data to be split.
    :type df: pl.DataFrame
    :param test_size: The proportion of the data to be used for the test set.
    :type test_size: float
    :param val_size: The proportion of the training data to be used for the validation set.
    :type val_size: float
    :param seed: The random seed for reproducibility.
    :type seed: int
    :param output_dir: The directory where the split datasets will be saved.
    :type output_dir: str
    :param num_folds: The number of folds for cross validation.
    :type num_folds: int
    :param stratification_label: The column to use for stratification.
    :type stratification_label: str
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(
        skf.split(df, df[stratification_label])
    ):
        fold_train_df = df[train_index]
        fold_test_df = df[test_index]

        fold_train_df = fold_train_df.with_columns(
            [
                pl.Series("dataset", ["train"] * len(fold_train_df)),
            ]
        )

        fold_test_df = fold_test_df.with_columns(
            [
                pl.Series("dataset", ["test"] * len(fold_test_df)),
            ]
        )

        # Further split training data into training and validation sets
        fold_train_df, fold_val_df = train_test_split(
            fold_train_df,
            test_size=val_size,
            random_state=seed,
            stratify=fold_train_df[stratification_label],
        )

        fold_val_df = fold_val_df.with_columns(
            [
                pl.Series("dataset", ["val"] * len(fold_val_df)),
            ]
        )

        df_fold = pl.concat([fold_train_df, fold_val_df, fold_test_df])

        fold_output_dir = os.path.join(
            output_dir, f"split_seed_{seed}", f"fold_{fold + 1}"
        )
        os.makedirs(fold_output_dir, exist_ok=True)

        df_fold.write_csv(
            os.path.join(fold_output_dir, "data.csv"),
        )


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pl.read_csv(args.input_csv)

    for seed in args.seeds:
        split_data(
            df=df,
            val_size=args.val_size,
            seed=seed,
            output_dir=args.output_dir,
            num_folds=args.num_folds,
            stratification_label=args.stratification_label,
        )
