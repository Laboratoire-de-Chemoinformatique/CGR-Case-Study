"""DataModules and Datasets for training neural networks on reaction data."""

import os
import pathlib
import pickle
import sys
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch import seed_everything
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpgen
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

seed_everything(123, workers=True)


class MorganFPReactionDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target_cols: List[str],
        reagent_encoders: Dict[str, MultiLabelBinarizer],
        multilabel_mask: Optional[list[bool]] = None,
        fp_save_path: Optional[str] = None,
        fp_length: int = 512,
        fp_radius: int = 3,
        precompute_fps: bool = True,
        reload_fps: bool = False,
    ):
        """
        Dataset for reaction data with optional precomputed fingerprints.

        :param data: DataFrame containing reaction data.
        :type data: pd.DataFrame
        :param target_cols: List of target column names.
        :type target_cols: List[str]
        :param encoders: Dictionary of encoders for target columns.
        :type encoders: Dict[str, MultiLabelBinarizer]
        :param multilabel_mask: Optional list indicating which targets are multilabel.
        :type multilabel_mask: Optional[list[bool]]
        :param fp_save_path: Path to save/load precomputed fingerprints, defaults to None.
        :type fp_save_path: Optional[str], optional
        :param fp_length: Length of the Morgan fingerprints, defaults to 512.
        :type fp_length: int, optional
        :param fp_radius: Radius of the Morgan fingerprints, defaults to 3.
        :type fp_radius: int, optional
        :param precompute_fps: Whether to compute fingerprints if not already saved, defaults to True.
        :type precompute_fps: bool, optional
        :param reload_fps: Whether to reload fingerprints from the saved file, defaults to False.
        :type reload_fps: bool, optional
        """
        self.data = data
        self.fp_save_path = fp_save_path
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        self.target_cols = target_cols
        self.multilabel_mask = multilabel_mask
        self.reagent_encoders = reagent_encoders

        if reload_fps and fp_save_path and os.path.exists(fp_save_path):
            with open(fp_save_path, "rb") as f:
                self.fps = pickle.load(f)
        else:
            self.fps = None

        if precompute_fps and self.fps is None:
            self.fps = self._compute_fps()
            if fp_save_path:
                self._save_fps()

        self.labels = {}
        for target_col in target_cols:
            encoder = self.reagent_encoders[target_col]
            target_col_label = encoder.transform(
                self.data[target_col].to_numpy().reshape(-1, 1)
            )
            self.labels[target_col] = torch.tensor(
                target_col_label, dtype=torch.float32
            )
            self.reagent_encoders[target_col] = encoder

        self.ids = torch.Tensor(self.data["rxn_id"].to_numpy())

    def _compute_fps(self) -> List[torch.Tensor]:
        """
        Compute Morgan fingerprints for the reactions.

        :return: List of fingerprints.
        :rtype: List[torch.Tensor]
        """
        fps = []
        morgan_generator = fpgen.GetMorganGenerator(
            radius=self.fp_radius, fpSize=self.fp_length
        )
        for _, row in self.data.iterrows():
            reactant_smiles = row["parsed_rxn"].split(">")[
                0
            ]  # Assuming SMILES structure: reactants>agents>products
            reactants = [
                Chem.MolFromSmiles(smi) for smi in reactant_smiles.split(".")
            ]
            combined_reactant_fp = None
            for reactant in reactants:
                fp = torch.Tensor(morgan_generator.GetFingerprint(reactant))

                # Concat the fingerprints together
                if combined_reactant_fp is None:
                    combined_reactant_fp = fp
                else:
                    combined_reactant_fp = torch.cat(
                        (combined_reactant_fp, fp), dim=0
                    )
            fps.append(combined_reactant_fp)

        return fps

    def _save_fps(self):
        """
        Save the precomputed fingerprints to a file.
        """
        with open(self.fp_save_path, "wb") as f:
            pickle.dump(self.fps, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
         Get a single data point from the dataset.

        :param idx: Index of the data point.
        :type idx: int
        """
        fp = self.fps[idx]

        return {
            "id": self.ids[idx],
            "x": fp.float(),
            "y": {
                target_col: self.labels[target_col][idx]
                for target_col in self.target_cols
            },
        }


class MorganFPReactionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        target_cols: List[str],
        fp_save_path: Optional[str] = None,
        fp_length: int = 512,
        fp_radius: int = 3,
        precompute_fps: bool = True,
        reload_fps: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        multilabel_mask: Optional[List[bool]] = None,
    ):
        """
        DataModule for reaction data with optional precomputed fingerprints.

        :param data_path: Path to the CSV file containing reaction data.
        :type data_path: str
        :param target_cols: List of target column names.
        :type target_cols: List[str]
        :param fp_save_path: Path to save/load precomputed fingerprints, defaults to None.
        :type fp_save_path: Optional[str], optional
        :param fp_length: Length of the Morgan fingerprints, defaults to 512.
        :type fp_length: int, optional
        :param fp_radius: Radius of the Morgan fingerprints, defaults to 3.
        :type fp_radius: int, optional
        :param precompute_fps: Whether to compute fingerprints if not already saved, defaults to True.
        :type precompute_fps: bool, optional
        :param reload_fps: Whether to reload fingerprints from the saved file, defaults to False.
        :type reload_fps: bool, optional
        :param batch_size: Batch size for DataLoader, defaults to 32.
        :type batch_size: int, optional
        :param num_workers: Number of workers for DataLoader, defaults to 4.
        :type num_workers: int, optional
        :param multilabel_mask: Optional list indicating which targets are multilabel.
        :type multilabel_mask: Optional[List[bool]], optional
        """
        super().__init__()
        self.data_path = data_path
        self.target_cols = target_cols
        self.fp_save_path = fp_save_path
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        self.precompute_fps = precompute_fps
        self.reload_fps = reload_fps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.reagent_encoders = {}
        self.multilabel_mask = multilabel_mask

        self.input_dim = self.fp_length * 2  # 2 Reactants per reaction
        self.output_dims = {}

        self.setup()

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for training, validation, and testing.

        """
        # Split the data into training, validation, and test sets
        data = pd.read_csv(self.data_path)
        train_data = data[data["dataset"] == "train"]
        val_data = data[data["dataset"] == "val"]
        test_data = data[data["dataset"] == "test"]

        # Create the encoders:
        for target_col in self.target_cols:
            encoder = MultiLabelBinarizer()
            encoder.fit(data[target_col].to_numpy().reshape(-1, 1))
            self.reagent_encoders[target_col] = encoder

        self.train_dataset = MorganFPReactionDataset(
            data=train_data,
            target_cols=self.target_cols,
            reagent_encoders=self.reagent_encoders,
            multilabel_mask=None,
            fp_save_path=self.fp_save_path,
            fp_length=self.fp_length,
            fp_radius=self.fp_radius,
            precompute_fps=self.precompute_fps,
            reload_fps=self.reload_fps,
        )
        self.val_dataset = MorganFPReactionDataset(
            data=val_data,
            target_cols=self.target_cols,
            reagent_encoders=self.reagent_encoders,
            multilabel_mask=self.multilabel_mask,
            fp_save_path=None,  # Don't save the validation FPs
            fp_length=self.fp_length,
            fp_radius=self.fp_radius,
            precompute_fps=True,
            reload_fps=False,
        )
        self.test_dataset = MorganFPReactionDataset(
            data=test_data,
            target_cols=self.target_cols,
            reagent_encoders=self.reagent_encoders,
            multilabel_mask=self.multilabel_mask,
            fp_save_path=None,  # Don't save the test FPs
            fp_length=self.fp_length,
            fp_radius=self.fp_radius,
            precompute_fps=True,
            reload_fps=False,
        )

        self.output_dims = {
            target_col: len(encoder.classes_)
            for target_col, encoder in self.reagent_encoders.items()
        }

    def train_dataloader(self):
        """Returns training dataloader."""
        print("Using MorganFPReactionDataModule")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Returns validation dataloader."""
        print("Using MorganFPReactionDataModule")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Returns test dataloader."""
        print("Using MorganFPReactionDataModule")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """Returns predict dataloader.

        For now this is the same as the test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class FragDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        id_to_frag_path: str,
        reagent_encoders: dict,
        target_cols: list,
    ):
        super(FragDataset, self).__init__()
        self.df = df
        self.id_to_frag = pickle.load(open(id_to_frag_path, "rb"))
        self.reagent_encoders = reagent_encoders
        self.target_cols = target_cols

        self.frags = torch.Tensor(
            [self.id_to_frag[id] for id in self.df["rxn_id"]]
        )
        self.labels = {}
        self.ids = torch.Tensor(self.df["rxn_id"].to_numpy())

        for target_col in target_cols:
            encoder = self.reagent_encoders[target_col]
            target_col_label = encoder.transform(
                df[target_col].to_numpy().reshape(-1, 1)
            )
            self.labels[target_col] = torch.Tensor(target_col_label)
            self.reagent_encoders[target_col] = encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "x": self.frags[idx].float(),
            "y": {
                target_col: self.labels[target_col][idx]
                for target_col in self.target_cols
            },
        }


class FragDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        id_to_frag_path: str,
        target_cols: list[str],
        batch_size: int = 64,
    ):
        super(FragDatamodule, self).__init__()
        self.data_path = data_path
        self.id_to_frag_path = id_to_frag_path
        self.target_cols = target_cols
        self.batch_size = batch_size

        self.id_to_frag = pickle.load(open(id_to_frag_path, "rb"))

        self.reagent_encoders = {}

        self.setup()

    def setup(self, stage=None):
        # Split the data into training, validation, and test sets
        data = pd.read_csv(self.data_path)
        train_data = data[data["dataset"] == "train"]
        val_data = data[data["dataset"] == "val"]
        test_data = data[data["dataset"] == "test"]

        # Create the encoders:
        for target_col in self.target_cols:
            encoder = MultiLabelBinarizer()
            encoder.fit(data[target_col].to_numpy().reshape(-1, 1))
            self.reagent_encoders[target_col] = encoder

        # Create the datasets
        self.train_dataset = FragDataset(
            train_data,
            self.id_to_frag_path,
            self.reagent_encoders,
            self.target_cols,
        )
        self.val_dataset = FragDataset(
            val_data,
            self.id_to_frag_path,
            self.reagent_encoders,
            self.target_cols,
        )
        self.test_dataset = FragDataset(
            test_data,
            self.id_to_frag_path,
            self.reagent_encoders,
            self.target_cols,
        )

        self.input_dim = self.train_dataset.frags.shape[1]

        self.output_dims = {
            target_col: len(encoder.classes_)
            for target_col, encoder in self.reagent_encoders.items()
        }

    def train_dataloader(self):
        """Returns training dataloader."""
        print("Using FragDatamodule")
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        """Returns validation dataloader."""
        print("Using FragDatamodule")
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        """Returns test dataloader."""
        print("Using FragDatamodule")
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def predict_dataloader(self):
        """Returns predict dataloader.

        For now this is the same as the test dataloader.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
