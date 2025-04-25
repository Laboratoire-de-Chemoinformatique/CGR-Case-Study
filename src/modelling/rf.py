"""Builds, optimises and evaluates random forest models for the given dataset."""

import argparse
import os
import pathlib
import pickle
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import optuna
import polars as pl
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import get_lrm_predictions, get_top_k_accuracies

SEED = 0


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory, containing the id_to_frag.pkl file and data.csv",
    )
    parser.add_argument(
        "-d",
        "--data_csv_name",
        type=str,
        required=True,
        help="Name of the data csv file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        type=int,
        default=10,
        help="Number of trials to run for hyperparameter optimisation.",
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=1,
        help="Number of cores to use for hyperparameter optimisation.",
    )
    parser.add_argument(
        "-m",
        "--path_to_model",
        type=str,
        help="Path to the model config to be loaded. Models should be in a .yml file, containing hyperparameters for both solvent and base.",
    )
    parser.add_argument(
        "--target_cols",
        "-t",
        nargs="+",
        default=["coarse_solvent", "base"],
        help="Columns to predict.",
    )
    parser.add_argument(
        "-k",
        "--model_dictionary_keys",
        nargs="+",
        default=["coarse_solvent", "base"],
        help="Keys corresponding to the desired models for the target columns. To be used when the target columns differ in name from the saved models.",
        required=False,
    )
    parser.add_argument(
        "-mo",
        "--model_output_dir",
        type=str,
        required=False,
        help="If specified, the (fitted) models, on the training data, will be saved to this directory, otherwise models are saved to the output directory.",
    )
    parser.add_argument(
        "-cs",
        "--config_save_dir",
        type=str,
        required=False,
        help="If specified, the config file of the best hyperparameters will be saved to this directory.",
    )
    return parser.parse_args()


def load_data(input_dir: str, data_csv_name: str) -> Tuple[pl.DataFrame, Dict]:
    """
    Load the dataset and fragment data.

    :param input_dir: Path to the input directory.
    :type input_dir: str
    :param data_csv_name: Name of the data csv file.
    :type data_csv_name: str
    :return: DataFrame containing the dataset and dictionary mapping IDs to fragments.
    :rtype: Tuple[pl.DataFrame, Dict]
    """
    df = pl.read_csv(os.path.join(input_dir, data_csv_name))
    id_to_frags = pickle.load(
        open(os.path.join(input_dir, "id_to_frag.pkl"), "rb")
    )
    return df, id_to_frags


def prepare_conditions(
    df: pl.DataFrame, reagent_cols: List[str] = ["coarse_solvent", "base"]
) -> Dict:
    """
    Prepare the conditions for the dataset.

    :param df: DataFrame containing the dataset.
    :type df: pl.DataFrame
    :param reagent_cols: List of reagent columns.
    :type reagent_cols: List[str]
    :return: Dictionary mapping IDs to conditions.
    :rtype: Dict
    """
    conditions = [
        df.select(reagent_type).to_numpy() for reagent_type in reagent_cols
    ]
    conditions = np.array(conditions).T.reshape(-1, len(reagent_cols))
    id_to_cond = {
        id: cond for id, cond in zip(df["rxn_id"].to_numpy(), conditions)
    }
    return id_to_cond


def split_data(
    df: pl.DataFrame, id_to_frags: Dict, id_to_cond: Dict
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Split the data into training and testing sets.

    :param df: DataFrame containing the dataset.
    :type df: pl.DataFrame
    :param id_to_frags: Dictionary mapping IDs to fragments.
    :type id_to_frags: Dict
    :param id_to_cond: Dictionary mapping IDs to conditions.
    :type id_to_cond: Dict
    :return: Arrays containing fragments and conditions for the training, testing and validation sets.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    train_frags, train_conds, val_frags, val_conds, test_frags, test_conds = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for row in df.iter_rows(named=True):
        if row["dataset"] == "train":
            train_frags.append(id_to_frags[row["rxn_id"]])
            train_conds.append(id_to_cond[row["rxn_id"]])
        elif row["dataset"] == "val":
            val_frags.append(id_to_frags[row["rxn_id"]])
            val_conds.append(id_to_cond[row["rxn_id"]])
        else:
            test_frags.append(id_to_frags[row["rxn_id"]])
            test_conds.append(id_to_cond[row["rxn_id"]])
    return (
        np.array(train_frags),
        np.array(train_conds),
        np.array(val_frags),
        np.array(val_conds),
        np.array(test_frags),
        np.array(test_conds),
    )


def train_or_load_models(
    train_frags: np.ndarray,
    train_conds: np.ndarray,
    val_frags: np.ndarray,
    val_conds: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, RandomForestClassifier]:
    """
    Train or load random forest models.

    :param train_frags: Training fragments.
    :type train_frags: np.ndarray
    :param train_conds: Training conditions.
    :type train_conds: np.ndarray
    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    :return: Dictionary of trained random forest models.
    :rtype: Dict[str, RandomForestClassifier]
    """
    models = {}
    train_labels = {}
    val_labels = {}

    for i, reagent_type in enumerate(args.target_cols):
        train_labels[reagent_type] = train_conds[:, i]

    # If path to model is provided, load the model
    if args.path_to_model:
        models_config = yaml.safe_load(open(args.path_to_model, "r"))

        # This is to treat the case where we are loading models trained on different target columns (but want to keep the same model config otherwise).
        if args.model_dictionary_keys:
            models = {
                key: RandomForestClassifier(**models_config[value])
                for key, value in zip(
                    args.target_cols, args.model_dictionary_keys
                )
            }
        else:
            models = {
                reagent_type: RandomForestClassifier(
                    **models_config[reagent_type]
                )
                for reagent_type in args.target_cols
            }

    # Otherwise, perform hyperparameter optimisation
    else:
        for i, reagent_type in enumerate(args.target_cols):
            val_labels[reagent_type] = val_conds[:, i]

        param_distributions = {
            "n_estimators": optuna.distributions.IntDistribution(
                100, 500, step=100
            ),
            "max_depth": optuna.distributions.IntDistribution(
                10, 100, step=10
            ),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "class_weight": optuna.distributions.CategoricalDistribution(
                ["balanced", None]
            ),
        }
        for reagent_type in args.target_cols:
            models[reagent_type] = optimize_model(
                model=RandomForestClassifier(random_state=SEED),
                param_distributions=param_distributions,
                train_frags=train_frags,
                train_labels=train_labels[reagent_type],
                val_frags=val_frags,
                val_labels=val_labels[reagent_type],
                args=args,
            )

    for reagent_type in args.target_cols:
        models[reagent_type].fit(train_frags, train_labels[reagent_type])

    # If specified, save the models to the model_output_dir, otherwise save to the output_dir
    if args.model_output_dir:
        save_models(models, args.model_output_dir)
    else:
        save_models(models, args.output_dir)

    return models


def optimize_model(
    model: RandomForestClassifier,
    param_distributions: Dict,
    train_frags: np.ndarray,
    train_labels: np.ndarray,
    val_frags: np.ndarray,
    val_labels: np.ndarray,
    args: argparse.Namespace,
) -> RandomForestClassifier:
    """
    Optimize the model using Optuna.

    :param model: Random forest classifier.
    :type model: RandomForestClassifier
    :param param_distributions: Parameter distributions for optimization.
    :type param_distributions: Dict
    :param train_frags: Training fragments.
    :type train_frags: np.ndarray
    :param train_labels: Training labels.
    :type train_labels: np.ndarray
    :param val_frags: Validation fragments.
    :type val_frags: np.ndarray
    :param val_labels: Validation labels.
    :type val_labels: np.ndarray
    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    :return: Best estimator found by Optuna.
    :rtype: RandomForestClassifier
    """
    combined_train_val_frags = np.concatenate([train_frags, val_frags])
    combined_train_val_labels = np.concatenate([train_labels, val_labels])
    search = optuna.integration.OptunaSearchCV(
        model,
        param_distributions,
        refit=True,
        scoring="f1_micro",
        n_trials=args.n_trials,
        n_jobs=args.cores,
        random_state=SEED,
    )
    search.fit(combined_train_val_frags, combined_train_val_labels)
    return search.best_estimator_


def save_models(
    models: Dict[str, RandomForestClassifier],
    output_dir: str,
) -> None:
    """
    Save the trained models to disk.

    :param models: Dictionary of trained random forest models.
    :type models: Dict[str, RandomForestClassifier]
    :param output_dir: Directory to save the models.
    :type output_dir: str
    """
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(
        models,
        open(os.path.join(output_dir, "cgr_rf_models.pkl"), "wb"),
    )


def evaluate_models(
    models: Dict[str, RandomForestClassifier],
    test_frags: np.ndarray,
    test_conds: np.ndarray,
    output_dir: str,
    reagent_cols: List[str] = ["coarse_solvent", "base"],
) -> None:
    """
    Evaluate the trained models on the test set.

    :param models: Dictionary of trained random forest models.
    :type models: Dict[str, RandomForestClassifier]
    :param test_frags: Test fragments.
    :type test_frags: np.ndarray
    :param test_conds: Test conditions.
    :type test_conds: np.ndarray
    :param output_dir: Directory to save the evaluation results.
    :type output_dir: str
    :param reagent_cols: List of reagent columns.
    :type reagent_cols: List[str]
    """
    class_probabilities = {}
    test_labels = {}
    mlbs = {}

    for i, reagent_type in enumerate(reagent_cols):
        class_probabilities[reagent_type] = np.array(
            models[reagent_type].predict_proba(test_frags)
        )
        mlbs[reagent_type] = MultiLabelBinarizer(
            classes=models[reagent_type].classes_
        )
        test_labels[reagent_type] = mlbs[reagent_type].fit_transform(
            [[reagent] for reagent in test_conds[:, i]]
        )

    combined_probs = np.concatenate(
        [class_probabilities[reagent_type] for reagent_type in reagent_cols],
        axis=1,
    )
    combined_labels = np.concatenate(
        [test_labels[reagent_type] for reagent_type in reagent_cols], axis=1
    )
    reagent_class_sizes = [
        len(models[reagent_type].classes_) for reagent_type in reagent_cols
    ]

    pred_labels, true_labels = get_lrm_predictions(
        class_probabilities=torch.Tensor(combined_probs),
        mhe_labels=combined_labels,
        reagent_class_sizes=reagent_class_sizes,
        multilabel_mask=[False for _ in reagent_cols],
        reagent_encoders=[mlbs[reagent_type] for reagent_type in reagent_cols],
    )

    overall_top_k_accuracy, reagent_top_k_accuracies_dict = (
        get_top_k_accuracies(
            predictions=pred_labels,
            labels=true_labels,
            top_k=10,
            reagent_classes=reagent_cols,
        )
    )

    reagent_types, reagent_top_k_accuracies = [], []

    for (
        reagent_type,
        reagent_top_k_accuracy_dict,
    ) in reagent_top_k_accuracies_dict.items():
        reagent_types.append(reagent_type)
        reagent_top_k_accuracies.append(reagent_top_k_accuracy_dict)

    save_accuracies(
        reagent_types=reagent_types,
        reagent_top_k_accuracy_dicts=reagent_top_k_accuracies,
        output_dir=output_dir,
        overall_top_k_accuracy_dict=overall_top_k_accuracy,
    )

    reagent_top_k_accuracies = []
    for reagent_type in reagent_cols:
        reagent_top_k_accuracies.append(
            {
                k + 1: top_k_accuracy_score(
                    y_true=np.argmax(test_labels[reagent_type], axis=1),
                    y_score=class_probabilities[reagent_type],
                    k=k + 1,
                )
                for k in range(len(models[reagent_type].classes_))
            }
        )

    save_accuracies(
        reagent_types=reagent_cols,
        reagent_top_k_accuracy_dicts=reagent_top_k_accuracies,
        output_dir=output_dir,
        filename="independent_top_k_accuracies.csv",
    )

    metric_dict = {
        "metric_name": [],
        "value": [],
    }
    for reagent_type in reagent_cols:
        reagent_probabilities = class_probabilities[reagent_type]
        reagent_labels = test_labels[reagent_type]
        for metric, function in [
            ("macro_precision", precision_score),
            ("macro_recall", recall_score),
        ]:
            metric_dict["metric_name"].append(f"{reagent_type}_{metric}")
            metric_dict["value"].append(
                function(
                    np.argmax(reagent_labels, axis=1),
                    np.argmax(reagent_probabilities, axis=1),
                    average="macro",
                )
            )

        metric_dict["metric_name"].append(f"{reagent_type}_micro_f1")
        metric_dict["value"].append(
            f1_score(
                np.argmax(reagent_labels, axis=1),
                np.argmax(reagent_probabilities, axis=1),
                average="micro",
            )
        )

        metric_dict["metric_name"].append(f"{reagent_type}_macro_f1")
        metric_dict["value"].append(
            f1_score(
                np.argmax(reagent_labels, axis=1),
                np.argmax(reagent_probabilities, axis=1),
                average="macro",
            )
        )

    pl.DataFrame(metric_dict).write_csv(
        os.path.join(output_dir, "independent_metrics.csv")
    )


def save_accuracies(
    reagent_types: List[str],
    reagent_top_k_accuracy_dicts: List[Dict],
    output_dir: str,
    filename: str = "top_k_accuracies.csv",
    overall_top_k_accuracy_dict: Union[dict, None] = None,
) -> None:
    """
    Save the top-k accuracies to a CSV file.

    :param reagent_types: List of reagent types.
    :type reagent_types: List[str]
    :param reagent_top_k_accuracy_dicts: List of dictionaries containing top-k accuracies for each reagent type.
    :type reagent_top_k_accuracy_dicts: List[Dict]
    :param output_dir: Directory to save the CSV file.
    :type output_dir: str
    :param filename: Name of the CSV file.
    :type filename: str
    :param overall_top_k_accuracy_dict: Dictionary containing overall top-k accuracies, defaults to None.
    :type overall_top_k_accuracy_dict: Union[dict, None], optional
    """
    complete_top_k_accuracies = {
        "reagent_type": [],
        "top_k": [],
        "accuracy": [],
    }
    for reagent_type, top_k_accuracies in zip(
        reagent_types, reagent_top_k_accuracy_dicts
    ):
        for top_k, accuracy in top_k_accuracies.items():
            complete_top_k_accuracies["reagent_type"].append(reagent_type)
            complete_top_k_accuracies["top_k"].append(top_k)
            complete_top_k_accuracies["accuracy"].append(accuracy)

    if overall_top_k_accuracy_dict is not None:
        for top_k, accuracy in overall_top_k_accuracy_dict.items():
            complete_top_k_accuracies["reagent_type"].append("overall")
            complete_top_k_accuracies["top_k"].append(top_k)
            complete_top_k_accuracies["accuracy"].append(accuracy)

    pl.DataFrame(complete_top_k_accuracies).write_csv(
        os.path.join(output_dir, filename)
    )


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, id_to_frags = load_data(args.input_dir, args.data_csv_name)
    id_to_cond = prepare_conditions(df, reagent_cols=args.target_cols)
    train_frags, train_conds, val_frags, val_conds, test_frags, test_conds = (
        split_data(df, id_to_frags, id_to_cond)
    )
    models = train_or_load_models(
        train_frags=train_frags,
        train_conds=train_conds,
        val_frags=val_frags,
        val_conds=val_conds,
        args=args,
    )
    evaluate_models(
        models=models,
        test_frags=test_frags,
        test_conds=test_conds,
        output_dir=args.output_dir,
        reagent_cols=args.target_cols,
    )

    if args.config_save_dir:
        os.makedirs(args.config_save_dir, exist_ok=True)
        with open(
            os.path.join(args.config_save_dir, "model_configs.yml"), "w"
        ) as f:
            yaml.dump(
                {
                    reagent_type: models[reagent_type].get_params()
                    for reagent_type in args.target_cols
                },
                f,
            )
