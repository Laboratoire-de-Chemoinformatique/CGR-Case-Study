"""Builds, optimises and evaluates gradient boosting models for the given dataset."""

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
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from xgboost import XGBClassifier

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
        "-e",
        "--path_to_encoders",
        type=str,
        help="Path to the encoders to be loaded, if a model is being loaded. If left empty, will default to the same directory as the models. Passing 'fit' will fit the encoders and save them to the output directory.",
        required=False,
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training.",
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
    :param reagent_cols: List of reagent columns, defaults to ["coarse_solvent", "base"]
    :type reagent_cols: List, optional
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
    :return: Arrays containing fragments and conditions for training, validation and testing sets.
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
) -> Tuple[XGBClassifier, XGBClassifier, OrdinalEncoder, OrdinalEncoder]:
    """
    Train or load gradient boosting models.

    Creates ordinal encoders for the reagents if they are not loaded from disk. Then either optimises
    the models using Optuna or loads the models from disk. Finally, fits the models on the new training data.

    :param train_frags: Training fragments.
    :type train_frags: np.ndarray
    :param train_conds: Training conditions (array of conditions of shape [n_samples, n_target_cols]).
    :type train_conds: np.ndarray
    :param val_frags: Validation fragments.
    :type val_frags: np.ndarray
    :param val_conds: Validation conditions (array of conditions of shape [n_samples, n_target_cols]).
    :type val_conds: np.ndarray
    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    :return: Trained gradient boosting models for solvent and base, and their encoders.
    :rtype: Tuple[XGBClassifier, XGBClassifier, OrdinalEncoder, OrdinalEncoder]
    """
    train_labels = {}
    val_labels = {}
    encoded_train_labels = {}
    encoded_val_labels = {}
    encoders = {}
    models = {}

    for i, reagent_type in enumerate(args.target_cols):
        train_labels[reagent_type] = train_conds[:, i]

    # If we don't specify a path to the model, we need to perform hyperparameter optimisation
    if not args.path_to_model:
        encoders = create_encoders(
            target_cols=args.target_cols,
            train_labels=train_labels,
        )
        for i, reagent_type in enumerate(args.target_cols):
            encoded_train_labels[reagent_type] = (
                encoders[reagent_type]
                .transform(train_labels[reagent_type].reshape(-1, 1))
                .ravel()
            )

            val_labels[reagent_type] = val_conds[:, i]
            encoded_val_labels[reagent_type] = (
                encoders[reagent_type]
                .transform(val_labels[reagent_type].reshape(-1, 1))
                .ravel()
            )

        for reagent_type in args.target_cols:
            models[reagent_type] = optimize_model(
                train_frags=train_frags,
                train_labels=encoded_train_labels[reagent_type],
                val_frags=val_frags,
                val_labels=encoded_val_labels[reagent_type],
                args=args,
            )
    # If we specify a path to the model, we load the model and encoders
    else:
        if args.path_to_encoders == "fit":
            encoders = create_encoders(
                target_cols=args.target_cols,
                train_labels=train_labels,
            )
        # Load the encoders, since we have provided a path to the model, this assumes that the encoders are in the same directory
        elif not args.path_to_encoders:
            encoders = pickle.load(
                open(
                    os.path.join(
                        pathlib.Path(args.path_to_model).parent, "encoders.pkl"
                    ),
                    "rb",
                )
            )
        # Otherwise, if we explicitly provide a path to the encoders, we load them from that path
        else:
            encoders = pickle.load(open(args.path_to_encoders, "rb"))

        # Encode the train labels:
        for i, reagent_type in enumerate(args.target_cols):
            encoded_train_labels[reagent_type] = (
                encoders[reagent_type]
                .transform(train_labels[reagent_type].reshape(-1, 1))
                .ravel()
            )

            val_labels[reagent_type] = val_conds[:, i]
            encoded_val_labels[reagent_type] = (
                encoders[reagent_type]
                .transform(val_labels[reagent_type].reshape(-1, 1))
                .ravel()
            )

        model_configs = yaml.safe_load(open(args.path_to_model, "r"))

        # This is used when the target columns differ in name from the saved models, e.g. using the coarse hyperparameters on the fine solvent column
        config_target_cols = args.target_cols
        if args.model_dictionary_keys:
            config_target_cols = args.model_dictionary_keys

        for i, reagent_type in enumerate(config_target_cols):
            # Check agreement between the number of classes in the model and the number of classes in the data
            target_col = args.target_cols[i]
            model_configs[reagent_type]["num_class"] = len(
                encoders[target_col].categories_[0]
            )
            models[target_col] = XGBClassifier(**model_configs[reagent_type])

    # Fit the models on the training data
    for reagent_type in args.target_cols:
        models[reagent_type].fit(
            train_frags,
            encoded_train_labels[reagent_type],
            eval_set=[(val_frags, encoded_val_labels[reagent_type])],
            verbose=False,
        )

    # Save the trained models:
    model_encoder_save_dir = args.output_dir
    if args.model_output_dir:
        model_encoder_save_dir = args.model_output_dir

    # Ensure the directory exists before saving encoders
    os.makedirs(model_encoder_save_dir, exist_ok=True)

    with open(os.path.join(model_encoder_save_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    save_models(models=models, output_dir=model_encoder_save_dir)

    return models, encoders


def create_encoders(
    target_cols: list[str], train_labels: np.ndarray, output_dir: str = None
) -> dict:
    """Creates and optionally saves encoders for the reagents.

    :param target_cols: A list of the target columns to transform
    :type target_cols: list[str]
    :param train_labels: The training labels
    :type train_labels: np.ndarray
    :param output_dir: The directory to save the encoders. If None, the encoders are not saved, defaults to None
    :type output_dir: str
    :return: A dictionary containing the encoders
    :rtype: dict
    """
    encoders = {}
    for reagent_type in target_cols:
        encoders[reagent_type] = OrdinalEncoder()
        encoders[reagent_type].fit(train_labels[reagent_type].reshape(-1, 1))

    if output_dir is not None:
        with open(os.path.join(output_dir, "encoders.pkl"), "wb") as f:
            pickle.dump(encoders, f)

    return encoders


def optimize_model(
    train_frags: np.ndarray,
    train_labels: np.ndarray,
    val_frags: np.ndarray,
    val_labels: np.ndarray,
    args: argparse.Namespace,
) -> XGBClassifier:
    """
    Optimize the model using Optuna.
    """
    train_data = (train_frags, train_labels)
    val_data = (val_frags, val_labels)

    if args.gpu:
        device = "gpu"
    else:
        device = "cpu"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, device=device),
        n_trials=args.n_trials,
        n_jobs=args.cores,
        show_progress_bar=True,
    )

    best_model = study.best_trial.user_attrs["best_model"]

    return best_model


def objective(
    trial: optuna.Trial,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    device: str = "gpu",
) -> float:
    """
    Objective function for Optuna.

    :param trial: Optuna trial.
    :type trial: optuna.Trial
    :param args: Parsed command line arguments.
    :type args: argparse.Namespace
    :return: Top-1 accuracy.
    :rtype: float
    """
    train_frags, train_labels = train_data
    val_frags, val_labels = val_data

    param = {
        "verbosity": 0,
        "objective": "multi:softprob",
        "tree_method": "hist",
        "device": device,
        "num_class": np.unique(train_labels).shape[0],
        "random_state": SEED,
        "early_stopping_rounds": 5,
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 32),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float(
            "learning_rate", low=0.01, high=0.1, step=0.01
        ),
    }

    xgb_model = XGBClassifier(
        **param,
    )

    encoded_train_labels = OrdinalEncoder().fit_transform(
        train_labels.reshape(-1, 1)
    )
    encoded_val_labels = OrdinalEncoder().fit_transform(
        val_labels.reshape(-1, 1)
    )

    xgb_model.fit(
        train_frags,
        encoded_train_labels,
        eval_set=[(val_frags, encoded_val_labels)],
        verbose=False,
    )

    val_preds = xgb_model.predict(val_frags)

    trial.set_user_attr("best_model", xgb_model)

    return f1_score(val_labels, val_preds, average="micro")


def save_models(
    models: Dict[str, XGBClassifier],
    output_dir: str,
) -> None:
    """
    Save the trained models to disk.

    :param models: Dictionary containing trained models.
    :type models: Dict[str, XGBClassifier]
    :param output_dir: Directory to save the models.
    :type output_dir: str
    """
    pickle.dump(
        models,
        open(os.path.join(output_dir, "cgr_gbm_models.pkl"), "wb"),
    )


def evaluate_models(
    models: Dict[str, XGBClassifier],
    encoders: Dict[str, OrdinalEncoder],
    test_frags: np.ndarray,
    test_conds: np.ndarray,
    output_dir: str,
    reagent_cols: List[str] = ["coarse_solvent", "base"],
) -> None:
    """
    Evaluate the trained models on the test set.

    :param models : Dictionary containing trained models.
    :type models: Dict[str, XGBClassifier]
    :param encoders: Dictionary containing encoders for the reagents.
    :type encoders: Dict[str, OrdinalEncoder]
    :param test_frags: Test fragments.
    :type test_frags: np.ndarray
    :param test_conds: Test conditions.
    :type test_conds: np.ndarray
    :param output_dir: Directory to save the evaluation results.
    :type output_dir: str
    """
    class_probabilities = {}
    for reagent_type, model in models.items():
        class_probabilities[reagent_type] = np.array(
            model.predict_proba(test_frags)
        )

    combined_probs = np.concatenate(
        [class_probabilities[reagent_type] for reagent_type in reagent_cols],
        axis=1,
    )

    mlbs = {}
    test_labels = {}

    for i, reagent_type in enumerate(reagent_cols):
        mlbs[reagent_type] = MultiLabelBinarizer(
            classes=encoders[reagent_type].categories_[0]
        )
        test_labels[reagent_type] = mlbs[reagent_type].fit_transform(
            [[reagent] for reagent in test_conds[:, i]]
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

    # These are the top k accuracies when we predict the entire vector (i.e. the classes predictions aren't independent)
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

    # Save rankings of the predictions when treating the classes as independent
    reagent_top_k_accuracies = []
    for reagent_type in reagent_cols:
        reagent_top_k_accuracies.append(
            {
                k + 1: top_k_accuracy_score(
                    np.argmax(test_labels[reagent_type], axis=1),
                    class_probabilities[reagent_type],
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

    # Add other metrics:
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
    id_to_cond = prepare_conditions(df=df, reagent_cols=args.target_cols)
    train_frags, train_conds, val_frags, val_conds, test_frags, test_conds = (
        split_data(df, id_to_frags, id_to_cond)
    )
    models, encoders = train_or_load_models(
        train_frags=train_frags,
        train_conds=train_conds,
        val_frags=val_frags,
        val_conds=val_conds,
        args=args,
    )

    evaluate_models(
        models=models,
        encoders=encoders,
        test_frags=test_frags,
        test_conds=test_conds,
        output_dir=args.output_dir,
        reagent_cols=args.target_cols,
    )

    # Save the configs for the models
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
