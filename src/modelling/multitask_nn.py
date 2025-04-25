"""Multitask model based on Morgan fingerprints."""

import os
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import data  # noqa: F401
from utils import get_lrm_predictions, get_top_k_accuracies

seed_everything(123, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MultitaskModel(pl.LightningModule):
    def __init__(
        self,
        task_head_params: Dict[str, Dict[str, Any]],
        input_size: int = 1500,
        shared_layers: list[int] = [64, 128],
        activation_fn: Callable[[int], nn.Module] = nn.ReLU,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        target_cols: list[str] = ["coarse_solvent", "base"],
        reagent_encoders: Dict[str, Any] = None,
        multilabel_mask: List[bool] = [False, False],
        metric_save_dir: Optional[str] = None,
        use_likelihood_ranking: bool = True,
        prediction_top_k_values: int = 10,
    ):
        """
        Initialize the multitask model.

        Args:
            input_size (int): The size of the input features.
            shared_layers (List[int]): Sizes of hidden layers in the shared backbone.
            activation_fn (Callable[[int], nn.Module]): Function to create activation modules.
            task_heads (Dict[str, Dict[str, Any]]): Specifications for each task head.
                Each head should be defined as:
                {
                    "output_size": int,      # Number of classes for the task
                    "hidden_layers": List[int], # Sizes of layers in the task-specific head
                    "activation_fn": Callable[[int], nn.Module], # Activation for head
                }
            learning_rate (float): Learning rate for training.
        """
        super().__init__()
        self.save_hyperparameters()

        self.task_head_params = task_head_params

        # Build the shared backbone
        layers = []
        last_size = input_size
        for hidden_size in shared_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(activation_fn(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            last_size = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Build task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, task_params in self.task_head_params.items():
            head_layers = []
            last_size = shared_layers[-1]
            if task_params["hidden_layers"] is not None:
                for hidden_size in task_params["hidden_layers"]:
                    head_layers.append(nn.Linear(last_size, hidden_size))
                    head_layers.append(
                        task_params["activation_fn"](hidden_size)
                    )
                    head_layers.append(nn.Dropout(task_params["dropout_rate"]))
                    last_size = hidden_size

            head_layers.append(
                nn.Linear(last_size, task_params["output_size"])
            )
            self.task_heads[task_name] = nn.Sequential(*head_layers)

        self.learning_rate = learning_rate

        self.reagent_encoders = reagent_encoders
        self.multilabel_mask = multilabel_mask
        self.target_cols = target_cols

        self.reagent_class_sizes = [
            len(self.reagent_encoders[task_head].classes_)
            for task_head in self.target_cols
        ]

        self.test_step_preds = []
        self.test_step_labels = []

        self.metric_save_dir = metric_save_dir
        if self.metric_save_dir is not None:
            os.makedirs(os.path.dirname(self.metric_save_dir), exist_ok=True)

        self.use_likelihood_ranking = use_likelihood_ranking
        self.prediction_top_k_values = prediction_top_k_values

    def forward(self, x):
        """
        Forward pass through the shared backbone and task-specific heads.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            Dict[str, torch.Tensor]: Predictions for each task head.
        """
        shared_output = self.shared_layers(x)
        task_outputs = {
            task_name: head(shared_output)
            for task_name, head in self.task_heads.items()
        }
        return task_outputs

    def training_step(self, batch, batch_idx):
        """
        Training step to compute losses for each task and backpropagate.
        Args:
            batch (Dict[str, torch.Tensor]): Input data and labels for each task.
        """
        x, y = batch["x"], batch["y"]
        outputs = self(x)
        total_loss = 0
        for task_name, task_params in self.task_head_params.items():
            task_loss = task_params["loss_fn"](
                outputs[task_name], y[task_name]
            )
            total_loss += task_loss

        self.log("train_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step to compute and log metrics.
        Args:
            batch (Dict[str, torch.Tensor]): Input data and labels for each task.
        """
        x, y = batch["x"], batch["y"]
        outputs = self(x)
        total_loss = 0
        for task_name, task_params in self.task_head_params.items():
            task_loss = task_params["loss_fn"](
                outputs[task_name], y[task_name]
            )
            total_loss += task_loss

        self.log("val_loss", total_loss, prog_bar=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        """
        Test step.
        """
        x, y = batch["x"], batch["y"]
        preds = self(x)

        for i, reagent_type in enumerate(self.target_cols):
            if self.multilabel_mask[i]:
                preds[reagent_type] = nn.Sigmoid()(preds[reagent_type])
            else:
                preds[reagent_type] = nn.Softmax(dim=1)(preds[reagent_type])

        self.test_step_preds.append(preds)
        self.test_step_labels.append(y)

    def on_test_epoch_end(self):
        """
        Compute metrics over the entire test set.
        """
        if self.metric_save_dir is not None:
            os.makedirs(self.metric_save_dir, exist_ok=True)

        combined_preds = {}
        combined_labels = {}
        for target_col in self.target_cols:
            combined_preds[target_col] = torch.cat(
                [
                    batch_pred[target_col]
                    for batch_pred in self.test_step_preds
                ],
                dim=0,
            )
            combined_labels[target_col] = torch.cat(
                [
                    batch_label[target_col]
                    for batch_label in self.test_step_labels
                ],
                dim=0,
            )

        independent_top_k_accuracies = {
            "reagent_type": [],
            "top_k": [],
            "accuracy": [],
        }
        independent_metrics = {
            "metric_name": [],
            "value": [],
        }
        for reagent_type, reagent_class_size in zip(
            self.target_cols, self.reagent_class_sizes
        ):
            reagent_preds = combined_preds[reagent_type]
            reagent_label_idxs = torch.argmax(
                combined_labels[reagent_type], dim=1
            )
            # Compute top-k accuracies
            for k in range(1, reagent_class_size + 1):
                acc = MulticlassAccuracy(
                    top_k=k,
                    num_classes=reagent_class_size,
                    average="micro",
                ).to(self.device)(
                    reagent_preds,
                    reagent_label_idxs,
                )
                independent_top_k_accuracies["reagent_type"].append(
                    reagent_type
                )
                independent_top_k_accuracies["top_k"].append(k)
                independent_top_k_accuracies["accuracy"].append(acc)

                self.log(
                    f"{reagent_type}_top_{k}_accuracy",
                    acc,
                )

            # Compute F1-Scores, Recall and Precision
            micro_f1 = MulticlassF1Score(
                num_classes=reagent_class_size, average="micro"
            ).to(self.device)(
                reagent_preds,
                reagent_label_idxs,
            )
            self.log(f"{reagent_type}_micro_f1", micro_f1)
            independent_metrics["metric_name"].append(
                f"{reagent_type}_micro_f1"
            )
            independent_metrics["value"].append(micro_f1)

            macro_f1 = MulticlassF1Score(
                num_classes=reagent_class_size, average="macro"
            ).to(self.device)(
                reagent_preds,
                reagent_label_idxs,
            )
            self.log(f"{reagent_type}_macro_f1", macro_f1)
            independent_metrics["metric_name"].append(
                f"{reagent_type}_macro_f1"
            )
            independent_metrics["value"].append(macro_f1)

            macro_recall = MulticlassRecall(
                num_classes=reagent_class_size, average="macro"
            ).to(self.device)(
                reagent_preds,
                reagent_label_idxs,
            )
            self.log(f"{reagent_type}_macro_recall", macro_recall)
            independent_metrics["metric_name"].append(
                f"{reagent_type}_macro_recall"
            )
            independent_metrics["value"].append(macro_recall)

            macro_precision = MulticlassPrecision(
                num_classes=reagent_class_size, average="macro"
            ).to(self.device)(
                reagent_preds,
                reagent_label_idxs,
            )
            self.log(f"{reagent_type}_macro_precision", macro_precision)
            independent_metrics["metric_name"].append(
                f"{reagent_type}_macro_precision"
            )
            independent_metrics["value"].append(macro_precision)

        independent_top_k_accuracies["accuracy"] = torch.Tensor(
            independent_top_k_accuracies["accuracy"]
        ).cpu()
        independent_metrics["value"] = torch.Tensor(
            independent_metrics["value"]
        ).cpu()

        independent_accuracy_df = pd.DataFrame(independent_top_k_accuracies)
        independent_metrics_df = pd.DataFrame(independent_metrics)

        if self.metric_save_dir:
            independent_accuracy_df.to_csv(
                os.path.join(
                    self.metric_save_dir, "independent_top_k_accuracies.csv"
                ),
                index=False,
            )
            independent_metrics_df.to_csv(
                os.path.join(self.metric_save_dir, "independent_metrics.csv"),
                index=False,
            )

        # Combine the predictions and calculate likelihood rankings of combinations
        # only if we have more than one target column, otherwise this is a standard
        # single-task model
        if len(self.target_cols) > 1 and self.use_likelihood_ranking:
            predicted_conditions, true_conditions = get_lrm_predictions(
                class_probabilities=torch.cat(
                    [
                        combined_preds[target_col]
                        for target_col in self.target_cols
                    ],
                    dim=1,
                ).cpu(),
                mhe_labels=torch.cat(
                    [
                        combined_labels[target_col]
                        for target_col in self.target_cols
                    ],
                    dim=1,
                ).cpu(),
                reagent_class_sizes=self.reagent_class_sizes,
                reagent_encoders=[
                    self.reagent_encoders[target_col]
                    for target_col in self.target_cols
                ],
                multilabel_mask=[False, False],
            )

            (
                dependent_overall_top_k_accuracies,
                dependent_top_k_accuracies,
            ) = get_top_k_accuracies(
                predictions=predicted_conditions,
                labels=true_conditions,
                top_k=10,
                reagent_classes=self.target_cols,
            )

            lrm_top_k_accuracies_to_save = {
                "reagent_type": [],
                "top_k": [],
                "accuracy": [],
            }

            lrm_overall_accuracies = {}
            for k, acc in dependent_overall_top_k_accuracies.items():
                lrm_top_k_accuracies_to_save["reagent_type"].append("overall")
                lrm_top_k_accuracies_to_save["top_k"].append(k)
                lrm_top_k_accuracies_to_save["accuracy"].append(acc[0])

                lrm_overall_accuracies[f"lrm_overall_top_{k}_acc"] = acc[0]

            self.log_dict(lrm_overall_accuracies)

            for reagent_type in self.target_cols:
                lrm_reagent_accuracies = {}
                for k, acc in dependent_top_k_accuracies[reagent_type].items():
                    lrm_top_k_accuracies_to_save["reagent_type"].append(
                        reagent_type
                    )
                    lrm_top_k_accuracies_to_save["top_k"].append(k)
                    lrm_top_k_accuracies_to_save["accuracy"].append(acc)

                    lrm_reagent_accuracies[
                        f"lrm_{reagent_type}_top_{k}_acc"
                    ] = acc

                self.log_dict(lrm_reagent_accuracies)

            lrm_accuracy_df = pd.DataFrame(lrm_top_k_accuracies_to_save)

            if self.metric_save_dir:
                lrm_accuracy_df.to_csv(
                    os.path.join(self.metric_save_dir, "top_k_accuracies.csv"),
                    index=False,
                )

            return independent_top_k_accuracies, lrm_overall_accuracies

        else:
            return independent_top_k_accuracies

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """A prediction step."""
        x = batch["x"]
        ids = batch["id"]
        preds = self(x)
        labels = batch["y"]

        for i, reagent_type in enumerate(self.target_cols):
            if self.multilabel_mask[i]:
                preds[reagent_type] = nn.Sigmoid()(preds[reagent_type])
            else:
                preds[reagent_type] = nn.Softmax(dim=1)(preds[reagent_type])

        self.predict_step_preds.append(preds)
        self.predict_step_labels.append(labels)
        self.predict_step_ids.append(ids)

        return

    def on_predict_start(self):
        self.predict_step_preds = []
        self.predict_step_labels = []
        self.predict_step_ids = []
        return

    def on_predict_end(self):
        """Combine predictions, labels and ids, to output a single dataframe."""
        if self.metric_save_dir is not None:
            os.makedirs(self.metric_save_dir, exist_ok=True)

        combined_preds = {}
        combined_labels = {}
        combined_ids = []
        for target_col in self.target_cols:
            combined_preds[target_col] = torch.cat(
                [
                    batch_pred[target_col]
                    for batch_pred in self.predict_step_preds
                ],
                dim=0,
            )
            combined_labels[target_col] = torch.cat(
                [
                    batch_label[target_col]
                    for batch_label in self.predict_step_labels
                ],
                dim=0,
            )
        combined_ids = torch.cat(self.predict_step_ids, dim=0)

        predictions_dict = {
            "rxn_id": [],
            "top_k": [],
        }
        predictions_dict.update(
            {target_col: [] for target_col in self.target_cols}
        )

        for idx, i in enumerate(combined_ids.cpu().numpy()):
            top_k_decoded_predictions = {
                target_col: [] for target_col in self.target_cols
            }
            for tc_idx, target_col in enumerate(self.target_cols):
                if self.multilabel_mask[tc_idx]:
                    raise NotImplementedError(
                        "Multilabel prediction not implemented yet."
                    )
                else:
                    # Get the OHE vector for the top-kth prediction
                    top_k_ohe_predictions = self.probabilities_to_top_k_ohe(
                        combined_preds[target_col][idx],
                        top_k=self.prediction_top_k_values,
                    )

                top_k_decoded_prediction = self.reagent_encoders[
                    target_col
                ].inverse_transform(top_k_ohe_predictions.numpy())
                top_k_decoded_predictions[target_col] = (
                    top_k_decoded_prediction
                )

            for k in range(self.prediction_top_k_values):
                predictions_dict["rxn_id"].append(int(i))
                predictions_dict["top_k"].append(k + 1)
                for target_col in self.target_cols:
                    predictions_dict[target_col].append(
                        top_k_decoded_predictions[target_col][k]
                    )

        predictions_df = pd.DataFrame(predictions_dict)
        predictions_df.to_csv(
            os.path.join(self.metric_save_dir, "predictions.csv"),
            index=False,
        )

        return predictions_df

    def probabilities_to_top_k_ohe(
        self, probabilities: torch.Tensor, top_k: int = 10
    ) -> torch.Tensor:
        """Converts a single vector of probabilities into k one-hot encoded vectors.

        :param probabilities: Predicted class probabilities, from a softmax layer.
        :type probabilities: torch.Tensor
        :param top_k: Top-K predictions to process, defaults to 10
        :type top_k: int, optional
        :return: One-hot encoded vectors for the top-k predictions, a tensor of shape k, num_classes.
        :rtype: torch.Tensor
        """
        sorted_indices = torch.argsort(probabilities, descending=True)
        top_k_indices = sorted_indices[:top_k]

        ohe_top_k_preds = torch.zeros(
            size=(top_k, probabilities.shape[0]), dtype=torch.float32
        )
        ohe_top_k_preds[torch.arange(top_k), top_k_indices] = 1
        return ohe_top_k_preds

    def configure_optimizers(self):
        """
        Define optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class MultiTaskCLI(LightningCLI):
    """A subclass of the LightningCLI which allows us to calculate the input size and target columns dynamically."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.input_dim", "model.input_size", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.target_cols", "model.target_cols", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.reagent_encoders",
            "model.reagent_encoders",
            apply_on="instantiate",
        )
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_config")
        parser.add_lightning_class_args(EarlyStopping, "early_stopping_config")

        parser.set_defaults(
            {
                "early_stopping_config.monitor": "val_loss",
                "early_stopping_config.mode": "min",
                "early_stopping_config.patience": 3,
                "early_stopping_config.min_delta": 0.0,
                "checkpoint_config.monitor": "val_loss",
                "checkpoint_config.mode": "min",
                "checkpoint_config.save_top_k": 1,
                "checkpoint_config.filename": "best_val_loss",
                "checkpoint_config.verbose": True,
                "checkpoint_config.save_last": True,
            }
        )


def main():
    MultiTaskCLI(
        model_class=MultitaskModel,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
