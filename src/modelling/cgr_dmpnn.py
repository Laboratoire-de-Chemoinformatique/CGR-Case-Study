"""Trains and tests ChemProp DMPNN models.

If no model config is supplied, this script will optimise the hyperparameters of a DMPNN
model using Optuna.

With a trained/loaded model, the script will train the model on the training data and
evaluate it on the test data, saving the metrics to the output directory."""

import argparse
import os
import pathlib
import sys

import lightning.pytorch as pl
import numpy as np
import optuna
import torch
import yaml
from chemprop import data, featurizers, models, nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import polars

from utils import get_lrm_predictions, get_top_k_accuracies

MAX_EPOCHS = 50

# This is so we can set the trainer to be deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

pl.seed_everything(123, workers=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--data_path",
        type=str,
        required=True,
        help="Path to the input csv file",
    )
    parser.add_argument(
        "-t",
        "--target_cols",
        type=str,
        nargs="+",
        required=True,
        help="List of target columns",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results and metrics, hyperparameter optimisation checkpoints will also be saved here.",
    )
    parser.add_argument(
        "-s",
        "--model_save_dir",
        type=str,
        required=True,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "-m",
        "--model_config_path",
        type=str,
        required=False,
        help="Path to the model config file, if not specified this script will optimise the hyperparameters",
    )
    parser.add_argument(
        "-hp",
        "--hpopt_output_dir",
        type=str,
        required=False,
        help="Output directory for the hyperparameter optimisation config file.",
    )
    parser.add_argument(
        "-n",
        "--num_trials",
        type=int,
        default=100,
        help="Number of trials to run",
    )
    parser.add_argument(
        "-e",
        "--num_train_epochs",
        type=int,
        default=50,
        help="Number of epochs to train for, during testing",
    )
    parser.add_argument(
        "-he",
        "--num_hpopt_epochs",
        type=int,
        default=10,
        help="Number of epochs to train for, during hyperparameter optimisation",
    )
    parser.add_argument(
        "-c",
        "--num_cpus",
        type=int,
        default=1,
        help="Number of cpus to use for dataloaders",
    )
    return parser.parse_args()


def get_rxns_and_indicies(
    train_df: polars.DataFrame,
    val_df: polars.DataFrame,
    test_df: polars.DataFrame,
) -> tuple:
    train_rxns = train_df["parsed_rxn"]
    val_rxns = val_df["parsed_rxn"]
    test_rxns = test_df["parsed_rxn"]

    rxns = np.concatenate([train_rxns, val_rxns, test_rxns])
    train_val_test_indicies = [
        np.arange(len(train_rxns)),
        np.arange(len(train_rxns), len(train_rxns) + len(val_rxns)),
        np.arange(len(train_rxns) + len(val_rxns), len(rxns)),
    ]

    return rxns, train_val_test_indicies


def encode_reagents(
    train_df: polars.DataFrame,
    val_df: polars.DataFrame,
    test_df: polars.DataFrame,
    target_cols: list,
) -> tuple:
    ordinal_encoders = {col: OrdinalEncoder() for col in target_cols}

    ### ENCODING LABELS
    encoded_labels = {}
    for col in target_cols:
        train_labels = ordinal_encoders[col].fit_transform(
            train_df[col].to_numpy().reshape(-1, 1)
        )
        val_labels = ordinal_encoders[col].transform(
            val_df[col].to_numpy().reshape(-1, 1)
        )
        test_labels = ordinal_encoders[col].transform(
            test_df[col].to_numpy().reshape(-1, 1)
        )

        encoded_labels[col] = np.concatenate(
            [train_labels, val_labels, test_labels], axis=0
        )

    encoded_reagents = np.concatenate(
        [encoded_labels[col] for col in target_cols], axis=1
    )

    return ordinal_encoders, encoded_reagents


def prepare_dataloaders(
    rxns: np.ndarray,
    encoded_reagents: np.ndarray,
    train_val_test_indicies: list,
    num_cpus: int = 1,
) -> tuple:
    all_data = [
        data.ReactionDatapoint.from_smi(rxn, tuple(reagent_set))
        for rxn, reagent_set in zip(rxns, encoded_reagents)
    ]

    train_indicies = train_val_test_indicies[0]
    val_indicies = train_val_test_indicies[1]
    test_indicies = train_val_test_indicies[2]

    train_data, val_data, test_data = data.split_data_by_indices(
        data=all_data,
        train_indices=[train_indicies],
        val_indices=[
            val_indicies,
        ],
        test_indices=[
            test_indicies,
        ],
    )

    featurizer = featurizers.CondensedGraphOfReactionFeaturizer()

    train_dset = data.ReactionDataset(train_data[0], featurizer)
    val_dset = data.ReactionDataset(val_data[0], featurizer)
    test_dset = data.ReactionDataset(test_data[0], featurizer)

    train_loader = data.build_dataloader(train_dset, num_workers=num_cpus)
    val_loader = data.build_dataloader(
        val_dset, shuffle=False, num_workers=num_cpus
    )
    test_loader = data.build_dataloader(
        test_dset, shuffle=False, num_workers=num_cpus
    )

    return train_loader, val_loader, test_loader, featurizer


def build_model(
    config: dict,
    featurizer: featurizers.CondensedGraphOfReactionFeaturizer,
    batch_norm=True,
):
    fdims = featurizer.shape
    agg = nn.MeanAggregation()
    mp = nn.BondMessagePassing(*fdims, **config["message_passing"])
    ffn = nn.MulticlassClassificationFFN(**config["ffn"])

    metric_list = [nn.metrics.MulticlassMCCMetric()]

    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    return mpnn


def train_model(
    model: models.MPNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model_checkpoint_path: str,
    num_epochs: int = MAX_EPOCHS,
    enable_model_summary: bool = True,
):
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", verbose=False
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_checkpoint_path,
        filename="best_val_loss",
        save_top_k=1,
        mode="min",
        save_last=False,
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=enable_model_summary,
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs,
        callbacks=[early_stopping, checkpoint],
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    lowest_val_loss_model = models.MPNN.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path
    )

    return lowest_val_loss_model


def optimise_or_load_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    target_cols: list,
    featurizer: featurizers.CondensedGraphOfReactionFeaturizer,
    model_config_path: str = None,
    hpopt_output_dir: str = None,
    num_trials: int = 100,
    encoders: dict = None,
    reagent_class_sizes: dict = None,
    num_epochs: int = MAX_EPOCHS,
):
    if model_config_path is None:
        if hpopt_output_dir is None:
            raise ValueError(
                "If no model config is supplied, an output directory for the hyperparameter optimisation config file must be provided."
            )

        mpnn = optimise_model(
            train_loader=train_loader,
            val_loader=val_loader,
            target_cols=target_cols,
            featurizer=featurizer,
            output_dir=hpopt_output_dir,
            num_trials=num_trials,
            ordinal_encoders=encoders,
            reagent_class_sizes=reagent_class_sizes,
            num_epochs=num_epochs,
        )

    else:
        mpnn = load_model(
            model_config_path=model_config_path,
            target_cols=target_cols,
            featurizer=featurizer,
            reagent_class_sizes=reagent_class_sizes,
        )

    return mpnn


def optimise_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    target_cols: list,
    featurizer: featurizers.CondensedGraphOfReactionFeaturizer,
    num_trials: int,
    output_dir: str,
    ordinal_encoders: dict,
    reagent_class_sizes: dict,
    num_epochs: int = MAX_EPOCHS,
):
    # Make the sampler behave in a deterministic way.
    sampler = optuna.samplers.TPESampler(seed=123)

    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name="cgr_dmpnn_hpopt"
    )
    study.optimize(
        lambda trial: objective(
            trial=trial,
            train_loader=train_loader,
            val_loader=val_loader,
            target_cols=target_cols,
            featurizer=featurizer,
            output_dir=output_dir,
            ordinal_encoders=ordinal_encoders,
            reagent_class_sizes=reagent_class_sizes,
            num_epochs=num_epochs,
            enable_model_summary=False,
        ),
        n_trials=num_trials,
        show_progress_bar=True,
    )

    best_config = study.best_trial.user_attrs["best_config"]
    best_model = build_model(config=best_config, featurizer=featurizer)

    with open(os.path.join(output_dir, "best_model_config.yml"), "w") as f:
        yaml.dump(best_config, f)

    return best_model


def objective(
    trial: optuna.Trial,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    target_cols: list,
    featurizer: featurizers.CondensedGraphOfReactionFeaturizer,
    output_dir: str,
    ordinal_encoders: dict,
    reagent_class_sizes: dict,
    num_epochs: int = MAX_EPOCHS,
    enable_model_summary: bool = False,
):
    config = {"ffn": {}, "message_passing": {}}

    config["ffn"]["dropout"] = trial.suggest_float(
        "dropout", 0, 0.2, step=0.05
    )
    config["ffn"]["n_layers"] = trial.suggest_int("n_layers", 1, 2)
    config["ffn"]["hidden_dim"] = trial.suggest_categorical(
        "hidden_dim", [256, 512, 1024]
    )
    config["ffn"]["n_classes"] = max(
        reagent_class_sizes.values(),
    )
    config["ffn"]["n_tasks"] = len(target_cols)

    config["message_passing"]["dropout"] = trial.suggest_float(
        "mpnn_dropout", 0, 0.2, step=0.05
    )
    config["message_passing"]["depth"] = trial.suggest_int("mpnn_depth", 4, 7)

    mpnn_model = build_model(config=config, featurizer=featurizer)

    mpnn_model = train_model(
        model=mpnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_checkpoint_path=os.path.join(
            output_dir, "checkpoints", f"trial_{trial.number}"
        ),
        num_epochs=num_epochs,
        enable_model_summary=enable_model_summary,
    )

    dependent_top_k_accuracies, _, _ = test_model(
        model=mpnn_model,
        test_loader=val_loader,
        target_cols=target_cols,
        reagent_class_sizes=reagent_class_sizes,
        encoders=ordinal_encoders,
    )

    trial.set_user_attr("best_model", mpnn_model)
    trial.set_user_attr("best_config", config)

    # get the index where reagent type is overall and top_k is 1
    overall_top_1_index = [
        i
        for i, (reagent_type, top_k) in enumerate(
            zip(
                dependent_top_k_accuracies["reagent_type"],
                dependent_top_k_accuracies["top_k"],
            )
        )
        if reagent_type == "overall" and top_k == 1
    ][0]

    return dependent_top_k_accuracies["accuracy"][overall_top_1_index]


def load_model(
    model_config_path: str,
    target_cols: list,
    reagent_class_sizes: dict,
    featurizer: featurizers.CondensedGraphOfReactionFeaturizer,
):
    config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)

    # We do an additional check here to ensure that the number of classes and number of tasks
    # is correct:
    config["ffn"]["n_tasks"] = len(target_cols)

    # This is a bit of a hack, ChemProp only allows MT models with the same number of classes per task,
    # so we set the number of classes to the maximum number of classes in the dataset.
    config["ffn"]["n_classes"] = max(reagent_class_sizes.values())

    mpnn = build_model(config=config, featurizer=featurizer)

    return mpnn


def test_model(
    model: models.MPNN,
    test_loader: torch.utils.torch.utils.data.DataLoader,
    target_cols: list,
    reagent_class_sizes: dict,
    encoders: dict,
):
    preds = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch.bmg.to(model.device)
            batch_preds = model(batch.bmg)
            preds.append(batch_preds)

            batch_labels = batch.Y
            labels.append(batch_labels)

    preds_tensor = torch.cat(preds, dim=0).detach().cpu()
    labels_tensor = torch.cat(labels, dim=0).detach().cpu()

    reagent_class_sizes = {
        col: int(reagents[:, i].max() + 1) for i, col in enumerate(target_cols)
    }

    reagent_preds = {}
    for i, col in enumerate(target_cols):
        reagent_preds[col] = preds_tensor[:, i, : reagent_class_sizes[col]]

    independent_top_k_accuracies = {
        "reagent_type": [],
        "top_k": [],
        "accuracy": [],
    }
    for i, col in enumerate(target_cols):
        for top_k in range(1, reagent_class_sizes[col] + 1):
            acc = MulticlassAccuracy(
                num_classes=reagent_class_sizes[col],
                top_k=top_k,
                average="micro",
            )
            independent_top_k_accuracies["reagent_type"].append(col)
            independent_top_k_accuracies["top_k"].append(top_k)
            independent_top_k_accuracies["accuracy"].append(
                acc(
                    reagent_preds[col],
                    labels_tensor[:, i].long(),
                )
            )

    # Add other metrics:
    independent_metrics = {
        "metric_name": [],
        "value": [],
    }
    for i, reagent_type in enumerate(target_cols):
        reagent_labels = labels_tensor[:, i]
        for metric_name, function in [
            ("macro_precision", MulticlassPrecision),
            ("macro_recall", MulticlassRecall),
        ]:
            metric = function(
                num_classes=reagent_class_sizes[reagent_type],
                average="macro",
            )
            independent_metrics["metric_name"].append(
                f"{reagent_type}_{metric_name}"
            )
            independent_metrics["value"].append(
                metric(
                    reagent_preds[reagent_type],
                    reagent_labels.long(),
                )
            )

        for average in ["micro", "macro"]:
            independent_metrics["metric_name"].append(
                f"{reagent_type}_{average}_f1"
            )
            metric = MulticlassF1Score(
                num_classes=reagent_class_sizes[reagent_type],
                average=average,
            )
            independent_metrics["value"].append(
                metric(
                    reagent_preds[reagent_type],
                    reagent_labels.long(),
                )
            )

    dependent_top_k_accuracies = get_dependent_top_k_accuracies(
        preds=reagent_preds,
        labels_tensor=labels_tensor,
        ordinal_encoders=encoders,
        reagent_class_sizes=reagent_class_sizes,
        target_cols=target_cols,
    )

    return (
        dependent_top_k_accuracies,
        independent_top_k_accuracies,
        independent_metrics,
    )


def get_dependent_top_k_accuracies(
    preds: dict,
    labels_tensor: torch.Tensor,
    ordinal_encoders: dict,
    reagent_class_sizes: dict,
    target_cols: list,
) -> dict:
    combined_preds = torch.cat([preds[col] for col in target_cols], dim=1)

    mhe_encoders = {
        col: MultiLabelBinarizer(
            classes=ordinal_encoders[col].categories_[0].tolist()
        )
        for col in target_cols
    }

    combined_mhe_labels = None
    for i, col in enumerate(target_cols):
        str_labels = ordinal_encoders[col].inverse_transform(
            labels_tensor[:, i].reshape(-1, 1)
        )
        mhe_labels = mhe_encoders[col].fit_transform(str_labels)

        if combined_mhe_labels is None:
            combined_mhe_labels = mhe_labels
        else:
            combined_mhe_labels = np.concatenate(
                [combined_mhe_labels, mhe_labels], axis=1
            )

    predicted_conditions, true_conditions = get_lrm_predictions(
        class_probabilities=combined_preds,
        mhe_labels=combined_mhe_labels,
        reagent_class_sizes=[reagent_class_sizes[col] for col in target_cols],
        reagent_encoders=[mhe_encoders[col] for col in target_cols],
        multilabel_mask=[False for _ in target_cols],
    )

    overall_top_k_accuracies, reagent_top_k_accuracies = get_top_k_accuracies(
        predictions=predicted_conditions,
        labels=true_conditions,
        top_k=10,
        reagent_classes=target_cols,
    )

    dependent_top_k_accuracies = {
        "reagent_type": [],
        "top_k": [],
        "accuracy": [],
    }

    for k, accuracy in overall_top_k_accuracies.items():
        dependent_top_k_accuracies["reagent_type"].append("overall")
        dependent_top_k_accuracies["top_k"].append(k)
        dependent_top_k_accuracies["accuracy"].append(accuracy[0])

    for col in target_cols:
        for k, accuracy in reagent_top_k_accuracies[col].items():
            dependent_top_k_accuracies["reagent_type"].append(col)
            dependent_top_k_accuracies["top_k"].append(k)
            dependent_top_k_accuracies["accuracy"].append(accuracy)

    return dependent_top_k_accuracies


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = polars.read_csv(
        args.data_path,
    )

    train_df = df.filter(polars.col("dataset") == "train")
    val_df = df.filter(polars.col("dataset") == "val")
    test_df = df.filter(polars.col("dataset") == "test")

    rxns, train_val_test_indicies = get_rxns_and_indicies(
        train_df=train_df, val_df=val_df, test_df=test_df
    )

    ordinal_encoders, reagents = encode_reagents(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_cols=args.target_cols,
    )

    train_loader, val_loader, test_loader, featurizer = prepare_dataloaders(
        rxns=rxns,
        encoded_reagents=reagents,
        train_val_test_indicies=train_val_test_indicies,
        num_cpus=args.num_cpus,
    )

    reagent_class_sizes = {
        col: int(reagents[:, i].max() + 1)
        for i, col in enumerate(args.target_cols)
    }

    mpnn_model = optimise_or_load_model(
        model_config_path=args.model_config_path,
        hpopt_output_dir=args.hpopt_output_dir,
        num_trials=args.num_trials,
        train_loader=train_loader,
        val_loader=val_loader,
        target_cols=args.target_cols,
        featurizer=featurizer,
        reagent_class_sizes=reagent_class_sizes,
        encoders=ordinal_encoders,
        num_epochs=args.num_hpopt_epochs,
    )

    mpnn_model = train_model(
        model=mpnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_checkpoint_path=args.model_save_dir,
        num_epochs=args.num_train_epochs,
    )

    (
        dependent_top_k_accuracies,
        independent_top_k_accuracies,
        independent_metrics,
    ) = test_model(
        model=mpnn_model,
        test_loader=test_loader,
        target_cols=args.target_cols,
        reagent_class_sizes=reagent_class_sizes,
        encoders=ordinal_encoders,
    )

    dependent_top_k_accuracies_df = polars.DataFrame(
        dependent_top_k_accuracies
    )
    dependent_top_k_accuracies_df.write_csv(
        f"{args.output_dir}/top_k_accuracies.csv"
    )

    independent_top_k_accuracies_df = polars.DataFrame(
        independent_top_k_accuracies
    )
    independent_top_k_accuracies_df.write_csv(
        f"{args.output_dir}/independent_top_k_accuracies.csv"
    )

    independent_metrics_df = polars.DataFrame(independent_metrics)
    independent_metrics_df.write_csv(
        f"{args.output_dir}/independent_metrics.csv"
    )
    independent_metrics_df = polars.DataFrame(independent_metrics)
    independent_metrics_df.write_csv(
        f"{args.output_dir}/independent_metrics.csv"
    )
