"""Optimising the hyperparameters of the MT-CGR model using Optuna.

Optimises the model for the top-1 accuracy of the validation set. If multiple target
columns are used, and likelihood ranking is not used, then the model is optimised for
the top-1 accuracy of each individual target column. If likelihood ranking is used, then
the model is optimised for the top-1 accuracy of the 'overall' prediction."""

import argparse
import os
import pathlib
import sys

import lightning.pytorch as pl
import optuna
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from multitask_nn import MultitaskModel

from data import FragDatamodule, MorganFPReactionDataModule

# This is so we can set the trainer to be deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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
        "-f",
        "--id_to_frag_path",
        type=str,
        required=False,
        help="Path to the id to fragment csv file",
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
        "-n",
        "--num_trials",
        type=int,
        default=100,
        help="Number of trials to run",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Output directory",
    )
    parser.add_argument(
        "-j",
        "--job_name",
        type=str,
        default="cgr_mtnn",
        help="Job name",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="cgr_mtnn",
        help="Model type to optimise the hyperparameters for, this will normally be 'cgr_mtnn', but could be 'morgan_mtnn' for the 'exact' solvents",
    )
    parser.add_argument(
        "--no_likelihood_ranking",
        action="store_true",
        help="Whether to use likelihood ranking or not, in the case where this is not used, and there are multiple target columns, optuna will perform multi-objective optimisation, using the top-1 accuracies of each individual target.",
    )
    return parser.parse_args()


def objective(trial, args: argparse.Namespace):
    n_shared_layers = trial.suggest_int("n_shared_layers", 1, 2)
    shared_layer_sizes = []
    for i in range(n_shared_layers):
        shared_layer_sizes.append(
            trial.suggest_categorical(
                f"shared_layer_size_{i}", [128, 256, 512, 1024]
            )
        )

    activation_fn = trial.suggest_categorical("activation_fn", ["ReLU", "ELU"])
    dropout = trial.suggest_float("dropout_rate", 0.0, 0.2)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

    if activation_fn == "ReLU":
        activation_fn = nn.ReLU
    elif activation_fn == "ELU":
        activation_fn = nn.ELU

    if args.model_type == "cgr_mtnn":
        datamodule = FragDatamodule(
            data_path=args.data_path,
            id_to_frag_path=args.id_to_frag_path,
            target_cols=args.target_cols,
        )
    elif args.model_type == "morgan_mtnn":
        datamodule = MorganFPReactionDataModule(
            data_path=args.data_path,
            target_cols=args.target_cols,
        )
    else:
        raise NotImplementedError(
            f"Model type {args.model_type} is not implemented"
        )

    use_likelihood_ranking = not args.no_likelihood_ranking

    model = MultitaskModel(
        task_head_params={
            target_col: {
                "hidden_layers": None,
                "output_size": datamodule.output_dims[target_col],
                "loss_fn": nn.CrossEntropyLoss(),
            }
            for target_col in datamodule.target_cols
        },
        input_size=datamodule.input_dim,
        shared_layers=shared_layer_sizes,
        activation_fn=activation_fn,
        dropout_rate=dropout,
        learning_rate=lr,
        target_cols=datamodule.target_cols,
        reagent_encoders=datamodule.reagent_encoders,
        use_likelihood_ranking=use_likelihood_ranking,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        logger=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                filename=f"best_val_loss_trial_{trial.number}",
            ),
        ],
        enable_model_summary=False,
        deterministic=True,
    )

    trainer.fit(model, datamodule)

    lowest_val_loss_model = MultitaskModel.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path
    )

    # We ensure that we optimising on the validation accuracy/metrics, not the testing set.
    if len(datamodule.target_cols) > 1:
        top_k_accuracies = trainer.test(
            lowest_val_loss_model,
            dataloaders=datamodule.val_dataloader(),
            verbose=False,
        )
        # Return the mean top-1 accuracy over each target column
        if args.no_likelihood_ranking:
            combined_idp_accuracies = 0
            for target_col in datamodule.target_cols:
                combined_idp_accuracies += top_k_accuracies[0][
                    f"{target_col}_top_1_accuracy"
                ]
            return combined_idp_accuracies / len(datamodule.target_cols)
        # Return the 'overall' top-1 accuracy
        else:
            return top_k_accuracies[0]["lrm_overall_top_1_acc"]
    else:
        independent_top_k_accuracies = trainer.test(
            lowest_val_loss_model,
            dataloaders=datamodule.val_dataloader(),
            verbose=False,
        )
        return independent_top_k_accuracies[0][
            f"{datamodule.target_cols[0]}_top_1_accuracy"
        ]


if __name__ == "__main__":
    args = parse_args()

    # Make the sampler behave in a deterministic way.
    sampler = optuna.samplers.TPESampler(seed=123)

    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name=args.job_name
    )
    study.optimize(
        lambda trial: objective(trial, args=args),
        n_trials=args.num_trials,
        show_progress_bar=True,
    )

    # Save to a .yml file:
    os.makedirs(args.output_dir, exist_ok=True)

    with open(
        os.path.join(args.output_dir, f"{args.job_name}_best_params.yml"), "w"
    ) as file:
        yaml.dump(study.best_params, file)
