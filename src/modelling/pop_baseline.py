"""Calculates a naive popularity baseline for predicting reaction conditions."""

import argparse
import os

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score

TOP_K_TO_MATCH = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
COARSE_CONDITION_COLS = ["coarse_solvent", "base"]
FINE_CONDITION_COLS = ["fine_solvent", "base"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-i",
        "--input_path",
        help="The path to the reaction database csv",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The directory where the output csv file should be written",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--target_cols",
        help="The target columns to calculate the baseline for",
        type=str,
        nargs="+",
        default=COARSE_CONDITION_COLS,
    )
    return parser.parse_args()


def create_naive_reagent_pop_baseline(
    df: pl.DataFrame,
    output_path: str,
    reagent_cols: list = COARSE_CONDITION_COLS,
) -> None:
    """Creates a naive popularity baseline, predicting the nth most popular
    reagent as the nth prediction for each reagent type.

    :param df: Input DataFrame
    :type df: pl.DataFrame
    :param output_path: Path to save the accuracy results
    :type output_path: str
    :param reagent_cols: The columns to calculate the accuracies for, defaults to COARSE_CONDITION_COLS
    :type reagent_cols: list, optional
    """
    train_df = df.filter(pl.col("dataset") == "train")
    preds = {}
    for reagent_type in reagent_cols:
        preds[reagent_type] = (
            train_df[reagent_type]
            .value_counts()
            .sort("count", descending=True)[reagent_type]
            .to_numpy()
        )

    test_df = df.filter(pl.col("dataset") == "test")

    for reagent_type in reagent_cols:
        # We retrieve the index of the reagent in the sorted list of reagents (+1 to give the k value)
        test_df = test_df.with_columns(
            pl.Series(
                f"{reagent_type}_top_k",
                test_df[reagent_type].map_elements(
                    lambda x: np.where(preds[reagent_type] == x)[0][0] + 1
                    if x in preds[reagent_type]
                    else len(preds[reagent_type]),
                    return_dtype=pl.Int64,
                ),
            )
        )

    reagent_top_k_accuracies = {}
    reagent_metrics = {
        "metric_name": [],
        "value": [],
    }
    for reagent_type in reagent_cols:
        n_classes = len(preds[reagent_type])
        reagent_top_k_accuracies[reagent_type] = []
        for k in range(1, n_classes + 1):
            reagent_top_k_accuracies[reagent_type].append(
                (
                    test_df.filter(pl.col(f"{reagent_type}_top_k") <= k)[
                        f"{reagent_type}_top_k"
                    ].count()
                    / len(test_df)
                )
            )

            # Here we calculate additional metrics for the top-1 predictions
            if k == 1:
                y_true = test_df[f"{reagent_type}_top_k"].to_numpy()

                # We know that the k value of all predictions is 1, the most popular reagent
                y_pred = np.ones(len(y_true))

                macro_precision = precision_score(
                    y_true, y_pred, average="macro"
                )
                macro_recall = recall_score(y_true, y_pred, average="macro")
                micro_f1 = f1_score(y_true, y_pred, average="micro")
                macro_f1 = f1_score(y_true, y_pred, average="macro")

                for metric_name, value in zip(
                    [
                        "macro_precision",
                        "macro_recall",
                        "micro_f1",
                        "macro_f1",
                    ],
                    [macro_precision, macro_recall, micro_f1, macro_f1],
                ):
                    reagent_metrics["metric_name"].append(
                        f"{reagent_type}_{metric_name}"
                    )
                    reagent_metrics["value"].append(value)

    pl.DataFrame(reagent_metrics).write_csv(
        os.path.join(
            os.path.dirname(output_path),
            "independent_metrics.csv",
        )
    )

    reagent_type_col = []
    top_k_col = []
    accuracy_col = []
    for reagent_type in reagent_cols:
        for k, accuracy in enumerate(reagent_top_k_accuracies[reagent_type]):
            reagent_type_col.append(reagent_type)
            top_k_col.append(k + 1)
            accuracy_col.append(accuracy)

    pl.DataFrame(
        {
            "reagent_type": reagent_type_col,
            "top_k": top_k_col,
            "accuracy": accuracy_col,
        }
    ).write_csv(output_path)


def create_dependent_pop_baseline(
    df: pl.DataFrame,
    output_path: str,
    reagent_cols: list = COARSE_CONDITION_COLS,
) -> None:
    """Creates a dependent popularity baseline, predicting the nth most popular
    *combination* of reagents as the nth prediction *overall*.

    This leads to the possibility of the same reagent being predicted multiple times,
    and gives n_reagents_i * n_reagents_j * ... * n_reagents_k total predictions.

    :param df: Input DataFrame
    :type df: pl.DataFrame
    :param output_path: Path to save the accuracy results
    :type output_path: str
    :param reagent_cols: The columns to calculate the accuracies for, defaults to COARSE_CONDITION_COLS
    :type reagent_cols: list, optional
    """
    train_df = df.filter(pl.col("dataset") == "train")
    preds = (
        train_df.group_by(reagent_cols)
        .agg(pl.len())
        .sort("len", descending=True)
        .select(reagent_cols)
        .to_numpy()
    )

    test_df = df.filter(pl.col("dataset") == "test")
    labels = test_df[reagent_cols].to_numpy()

    top_k_accuracies = {"reagent_type": [], "top_k": [], "accuracy": []}

    # Calculate the overall accuracy
    for k in range(1, max(TOP_K_TO_MATCH) + 1):
        overall_accuracy = np.mean(
            [np.any(np.all(label == preds[:k], axis=1)) for label in labels]
        )
        top_k_accuracies["reagent_type"].append("overall")
        top_k_accuracies["top_k"].append(k)
        top_k_accuracies["accuracy"].append(overall_accuracy)

    # Calculate the accuracy for each reagent type
    for col_idx, col in enumerate(reagent_cols):
        for k in range(1, max(TOP_K_TO_MATCH) + 1):
            accuracy = np.mean(
                [label[col_idx] in preds[:k, col_idx] for label in labels]
            )
            top_k_accuracies["reagent_type"].append(col)
            top_k_accuracies["top_k"].append(k)
            top_k_accuracies["accuracy"].append(accuracy)

    pl.DataFrame(top_k_accuracies).write_csv(output_path)


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pl.read_csv(args.input_path)

    create_naive_reagent_pop_baseline(
        df=df,
        output_path=os.path.join(
            args.output_dir,
            "independent_top_k_accuracies.csv",
        ),
        reagent_cols=args.target_cols,
    )
    create_dependent_pop_baseline(
        df=df,
        output_path=os.path.join(
            args.output_dir,
            "top_k_accuracies.csv",
        ),
        reagent_cols=args.target_cols,
    )
