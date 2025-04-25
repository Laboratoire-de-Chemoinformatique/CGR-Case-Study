"""Performs kNN search using CGR fragments."""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

N_NEIGHBOURS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Performs kNN search using CGR fragments"
    )
    parser.add_argument(
        "-i",
        "--input_csv",
        type=str,
        default="data/parsed_jacs_data/splits/data_seed_1.csv",
        help="Path to the data file",
    )
    parser.add_argument(
        "-f",
        "--id_to_frag_path",
        type=str,
        default="data/parsed_jacs_data/id_to_frag.pkl",
        help="Path to the fragment file",
    )
    parser.add_argument(
        "-t",
        "--target_cols",
        nargs="+",
        default=["coarse_solvent", "base"],
        help="Columns to predict",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="data/parsed_jacs_data/knn_results",
        help="Path to save the results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    id_to_frag = pickle.load(open(args.id_to_frag_path, "rb"))

    train_ids = df[df["dataset"] == "train"]["rxn_id"].values
    test_ids = df[df["dataset"] == "test"]["rxn_id"].values

    train_frags = [id_to_frag[id] for id in train_ids]
    train_frags = np.array(train_frags)
    test_frags = [id_to_frag[id] for id in test_ids]
    test_frags = np.array(test_frags)

    mlb_dict = {}
    for target_col in args.target_cols:
        mlb = MultiLabelBinarizer()
        mlb.fit([[target] for target in df[target_col].values])
        mlb_dict[target_col] = mlb

    train_targets = {
        target_col: mlb.transform(
            [
                [target]
                for target in df[df["dataset"] == "train"][target_col].values
            ]
        )
        for target_col, mlb in mlb_dict.items()
    }

    test_targets = {
        target_col: mlb.transform(
            [
                [target]
                for target in df[df["dataset"] == "test"][target_col].values
            ]
        )
        for target_col, mlb in mlb_dict.items()
    }

    neighbors = NearestNeighbors(n_neighbors=N_NEIGHBOURS)
    neighbors.fit(train_frags)

    _, indices = neighbors.kneighbors(test_frags)

    independent_matches_dict = {
        target_col: [] for target_col in args.target_cols + ["overall"]
    }
    for test_idx, training_idxs_for_testing_point in enumerate(indices):
        testing_point_matches = {
            target_col: [] for target_col in args.target_cols + ["overall"]
        }
        for target_col in args.target_cols:
            reagent_target = test_targets[target_col][test_idx]

            reagent_preds = []
            for training_idx in training_idxs_for_testing_point:
                reagent_pred = train_targets[target_col][training_idx]
                # Make sure that the prediction is not already in the list
                if not any(
                    [
                        np.array_equal(
                            train_targets[target_col][training_idx],
                            existing_pred,
                        )
                        for existing_pred in reagent_preds
                    ]
                ):
                    reagent_preds.append(reagent_pred)

            match_indices = []
            for k, pred in enumerate(reagent_preds):
                if np.array_equal(reagent_target, pred):
                    match_indices.append(k + 1)

            if match_indices:
                testing_point_matches[target_col] = match_indices
            else:
                testing_point_matches[target_col] = [50]

        # Get the lowest k for a match for each target column
        for target_col in args.target_cols:
            independent_matches_dict[target_col].append(
                min(testing_point_matches[target_col])
            )

    matches_dict = {
        target_col: [] for target_col in args.target_cols + ["overall"]
    }

    # Tracking these for metrics
    independent_top_1_preds = {
        target_col: [] for target_col in args.target_cols
    }
    independent_top_1_labels = {
        target_col: [] for target_col in args.target_cols
    }
    for test_idx, training_idxs_for_testing_point in enumerate(indices):
        # Next calculate the overall matches:
        true_overall_target = np.concatenate(
            [
                test_targets[target_col][test_idx]
                for target_col in args.target_cols
            ]
        )
        overall_preds = []
        for k, training_idx in enumerate(training_idxs_for_testing_point):
            if k == 1:
                for target_col in args.target_cols:
                    independent_top_1_preds[target_col].append(
                        train_targets[target_col][training_idx]
                    )
                    independent_top_1_labels[target_col].append(
                        test_targets[target_col][test_idx]
                    )

            overall_pred = np.concatenate(
                [
                    train_targets[target_col][training_idx]
                    for target_col in args.target_cols
                ]
            )
            # Make sure that the prediction is not already in the list
            if not any(
                [
                    np.array_equal(overall_pred, existing_pred)
                    for existing_pred in overall_preds
                ]
            ):
                overall_preds.append(overall_pred)

        overall_match_indices = []
        reagent_match_indices = {
            target_col: [] for target_col in args.target_cols
        }
        for k, pred in enumerate(overall_preds):
            if np.array_equal(true_overall_target, pred):
                overall_match_indices.append(k + 1)

            start_idx = 0
            for target_col in args.target_cols:
                reagent_target = test_targets[target_col][test_idx]

                end_idx = start_idx + len(train_targets[target_col][0])
                reagent_pred = pred[start_idx:end_idx]

                if np.array_equal(reagent_target, reagent_pred):
                    reagent_match_indices[target_col].append(k + 1)

                start_idx = end_idx

        if overall_match_indices:
            matches_dict["overall"].append(min(overall_match_indices))
        else:
            matches_dict["overall"].append(50)

        for target_col in args.target_cols:
            if reagent_match_indices[target_col]:
                matches_dict[target_col].append(
                    min(reagent_match_indices[target_col])
                )
            else:
                matches_dict[target_col].append(50)

    for target_col in args.target_cols + ["overall"]:
        independent_matches_dict[target_col] = np.array(
            independent_matches_dict[target_col]
        )
        matches_dict[target_col] = np.array(matches_dict[target_col])

    independent_top_k_accuracys = {
        target_col: [] for target_col in args.target_cols
    }
    top_k_accuracys = {
        target_col: [] for target_col in args.target_cols + ["overall"]
    }

    for k in range(1, 11):
        for target_col in args.target_cols:
            top_k_accuracys[target_col].append(
                np.mean(matches_dict[target_col] <= k)
            )
            independent_top_k_accuracys[target_col].append(
                np.mean(independent_matches_dict[target_col] <= k)
            )

        top_k_accuracys["overall"].append(
            np.mean(matches_dict["overall"] <= k)
        )

    acc_df = pd.DataFrame(top_k_accuracys)
    acc_df["top_k"] = acc_df.index + 1
    acc_df = acc_df.melt(
        id_vars="top_k", var_name="reagent_type", value_name="accuracy"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    acc_df.to_csv(
        os.path.join(args.output_dir, "top_k_accuracies.csv"), index=False
    )

    independent_acc_df = pd.DataFrame(independent_top_k_accuracys)
    independent_acc_df["top_k"] = independent_acc_df.index + 1
    independent_acc_df = independent_acc_df.melt(
        id_vars="top_k", var_name="reagent_type", value_name="accuracy"
    )
    independent_acc_df.to_csv(
        os.path.join(args.output_dir, "independent_top_k_accuracies.csv"),
        index=False,
    )

    # Finally calculate the metrics:
    metrics = {"metric_name": [], "value": []}
    for target_col in args.target_cols:
        y_true = np.array(independent_top_1_labels[target_col])
        y_pred = np.array(independent_top_1_preds[target_col])

        for metric_name, value in zip(
            [
                "macro_precision",
                "macro_recall",
                "micro_f1",
                "macro_f1",
            ],
            [
                precision_score(y_true, y_pred, average="macro"),
                recall_score(y_true, y_pred, average="macro"),
                f1_score(y_true, y_pred, average="micro"),
                f1_score(y_true, y_pred, average="macro"),
            ],
        ):
            metrics["metric_name"].append(f"{target_col}_macro_{metric_name}")
            metrics["value"].append(value)

    reagent_metrics_df = pd.DataFrame(metrics)
    reagent_metrics_df.to_csv(
        os.path.join(
            args.output_dir,
            "independent_metrics.csv",
        ),
        index=False,
    )
