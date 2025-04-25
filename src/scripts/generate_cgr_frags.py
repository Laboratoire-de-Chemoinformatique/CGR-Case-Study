"""Loads in an RDF files corresponding to training and testing reactions. Generates
ISIDA fragments for each and saves them to a file."""

#! usr/bin/env python3

import argparse
import os
import pickle
import sys
from typing import Tuple

import CGRtools
import numpy as np
import pandas as pd
import tqdm
import yaml
from CIMtools.preprocessing import CGR, Fragmentor
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    :return: Command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate ISIDA fragments for an RDF file of reactions. Training reactions should be marked with 'train' in the 'dataset' column"
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        type=str,
        help="Path to the CSV file containing the mapped reactions.",
    )
    parser.add_argument(
        "--run_pca",
        action="store_true",
        help="Whether to run PCA on the fragments.",
    )
    parser.add_argument(
        "--save_raw_frags",
        action="store_true",
        help="Whether to save the raw fragments before PCA.",
    )
    parser.add_argument(
        "--save_add_info",
        action="store_true",
        help="Whether to save additional information about the fragments.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to the directory to save the output files.",
    )
    parser.add_argument(
        "--frag_type",
        type=int,
        default=9,
        help="Type of fragmentation to use. See ISIDA documentation for details.",
    )
    parser.add_argument(
        "--min_frag_length",
        type=int,
        default=2,
        help="Minimum length of fragments. See ISIDA documentation for details.",
    )
    parser.add_argument(
        "--max_frag_length",
        type=int,
        default=4,
        help="Maximum length of fragments. See ISIDA documentation for details.",
    )
    parser.add_argument(
        "--use_formal_charge",
        type=bool,
        default=True,
        help="Whether to use formal charge in the fragments. See ISIDA documentation for details.",
    )
    parser.add_argument(
        "--cgr_dynbonds",
        type=int,
        default=0,
        help="Number of dynamic bonds in the CGR. See ISIDA documentation for details.",
    )
    parser.add_argument(
        "--n_components",
        type=float,
        default=100,
        help="Number of components to keep in the PCA.",
    )
    parser.add_argument(
        "--input_frags",
        required=False,
        type=str,
        help="Path to the file containing the fragments. If specified, no fragmenting will be done.",
    )
    parser.add_argument(
        "--min_num_occurrences",
        type=int,
        default=5,
        help="Minimum number of occurrences of a fragment to be included in the final features.",
    )
    return parser.parse_args()


def remove_reagents(
    r: CGRtools.ReactionContainer, keep_reagents: bool = False
):
    """
    Preprocess reaction according to mapping, using the following idea: molecules(each separated graph) will be
    placed to reagents if it is not changed in the reaction (no bonds, charges reorders)

    Return True if any reagent found.
    """
    reaction = r.copy()
    cgr = reaction.compose()
    if cgr.center_atoms:
        active = set(cgr.center_atoms)
        reactants = []
        products = []
        reagents = set(reaction.reagents)
        for i in reaction.reactants:
            if not active.isdisjoint(i):
                reactants.append(i)
            else:
                reagents.add(i)
        for i in reaction.products:
            if not active.isdisjoint(i):
                products.append(i)
            else:
                reagents.add(i)
        if keep_reagents:
            tmp = []
            for m in reaction.reagents:
                if m in reagents:
                    tmp.append(m)
                    reagents.discard(m)
            tmp.extend(reagents)
            reagents = tuple(tmp)
        else:
            reagents = ()

        if (
            len(reactants) != len(reaction.reactants)
            or len(products) != len(reaction.products)
            or len(reagents) != len(reaction.reagents)
        ):
            reaction._ReactionContainer__reactants = tuple(reactants)
            reaction._ReactionContainer__products = tuple(products)
            reaction._ReactionContainer__reagents = reagents
            reaction.flush_cache()
            reaction.fix_positions()
            return reaction, True
        return reaction, False


def run_pca(
    fragment_features: np.ndarray, pca_components: float
) -> Tuple[np.ndarray, IncrementalPCA]:
    """Runs PCA dimnesionality reduction on the fragment features.

    :param fragment_features: Unprocessed fragment features.
    :type fragment_features: np.ndarray
    :param pca_components: Number of components to keep in the PCA, if <1, the number of components to explain the variance, if >1 then converted to an integer.
    :type pca_components: float
    :return: Processed fragment features and the PCA object.
    :rtype: tuple[np.ndarray, IncrementalPCA]
    """
    if pca_components >= 1:
        pca = IncrementalPCA(n_components=pca_components)

        print("Running PCA...")
        fitted_pca = pca.fit(fragment_features)
        pca_features = fitted_pca.transform(fragment_features)
        print(f"PCA Done. Features Final Shape: {pca_features.shape}")

    else:
        pca = IncrementalPCA(n_components=None)

        print(f"Running PCA to explain {pca_components * 100}% of variance...")
        fitted_pca = pca.fit(fragment_features)
        cumulative_variance = np.cumsum(fitted_pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumulative_variance >= pca_components) + 1
        print(
            f"Number of components to explain {pca_components * 100}% variance: {n_components_95}"
        )

        pca = IncrementalPCA(n_components=n_components_95)
        pca_features = pca.fit_transform(fragment_features)
        print(f"PCA Done. Features Final Shape: {pca_features.shape}")

    return pca_features, pca


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_frags:
        frags = np.load(args.input_frags)

        if not args.run_pca:
            raise ValueError(
                "Input fragments provided but PCA not run. Please specify --run_pca."
            )
        else:
            n_components = (
                int(args.n_components)
                if args.n_components >= 1
                else args.n_components
            )
            features = run_pca(frags, n_components)

            np.save(
                os.path.join(
                    args.output_dir,
                    f"pca_frags_{n_components}_dim.npy",
                ),
                features,
            )

        sys.exit(0)

    if not args.save_raw_frags and not args.run_pca:
        raise ValueError(
            "No action specified. Please specify either --save_raw_frags or --run_pca or both."
        )

    train_rxns = []
    test_rxns = []
    val_rxns = []
    train_rxn_ids = []
    test_rxn_ids = []
    val_rxn_ids = []

    # This is for the generation of CGRs, just to make sure that the reagent information is not included in the descriptors.
    train_rxns_no_reagents = []
    test_rxns_no_reagents = []
    val_rxns_no_reagents = []

    df = pd.read_csv(args.input_csv)

    for _, row in tqdm.tqdm(
        df.iterrows(), desc="Reading in reactions...", total=len(df)
    ):
        train = row["dataset"] == "train"
        rxn = CGRtools.smiles(row["parsed_rxn"])
        cleaned_rxn, _ = remove_reagents(rxn, keep_reagents=True)

        if train:
            train_rxns.append(cleaned_rxn)
        elif row["dataset"] == "val":
            val_rxns.append(cleaned_rxn)
        else:
            test_rxns.append(cleaned_rxn)

        rxn_no_reagent = CGRtools.ReactionContainer(
            reactants=cleaned_rxn.reactants, products=cleaned_rxn.products
        )

        if train:
            train_rxn_ids.append(row["rxn_id"])
            train_rxns_no_reagents.append(rxn_no_reagent)
        elif row["dataset"] == "val":
            val_rxn_ids.append(row["rxn_id"])
            val_rxns_no_reagents.append(rxn_no_reagent)
        else:
            test_rxn_ids.append(row["rxn_id"])
            test_rxns_no_reagents.append(rxn_no_reagent)

    create_cgr_and_fragment = Pipeline(
        [
            ("cgr", CGR()),
            (
                "frg",
                Fragmentor(
                    fragment_type=args.frag_type,
                    min_length=args.min_frag_length,
                    max_length=args.max_frag_length,
                    useformalcharge=args.use_formal_charge,
                    cgr_dynbonds=args.cgr_dynbonds,
                    version="2017",
                    verbose=False,
                    header=None,
                    # remove_rare_ratio=0.001,
                ),
            ),
        ]
    )  # All descriptors were normalized to zero mean and unit variance.

    cgr_pipeline = ColumnTransformer(
        [("create_cgr_and_fragment", create_cgr_and_fragment, [0])], n_jobs=1
    )

    print("Fragmenting reactions...")
    scaler = StandardScaler()
    unprocessed_train_features = cgr_pipeline.fit_transform(
        np.array(train_rxns_no_reagents).reshape(-1, 1)
    )
    # Ensure that the fragment appears in at least min_num_occurrences reactions
    valid_cols = (
        unprocessed_train_features.sum(axis=0) > args.min_num_occurrences
    )
    unprocessed_train_features = unprocessed_train_features[:, valid_cols]
    unprocessed_train_features = unprocessed_train_features.reshape(
        len(train_rxns_no_reagents), -1
    )
    unprocessed_train_features = scaler.fit_transform(
        unprocessed_train_features
    )

    unprocessed_test_features = cgr_pipeline.transform(
        np.array(test_rxns_no_reagents).reshape(-1, 1)
    )
    unprocessed_test_features = unprocessed_test_features[:, valid_cols]
    unprocessed_test_features = unprocessed_test_features.reshape(
        len(test_rxns_no_reagents), -1
    )
    unprocessed_test_features = scaler.transform(unprocessed_test_features)

    unprocessed_val_features = cgr_pipeline.transform(
        np.array(val_rxns_no_reagents).reshape(-1, 1)
    )
    unprocessed_val_features = unprocessed_val_features[:, valid_cols]
    unprocessed_val_features = unprocessed_val_features.reshape(
        len(val_rxns_no_reagents), -1
    )
    unprocessed_val_features = scaler.transform(unprocessed_val_features)
    print("Finished fragmenting reactions.")

    rxn_ids = train_rxn_ids + test_rxn_ids + val_rxn_ids
    unprocessed_features = np.concatenate(
        (
            unprocessed_train_features,
            unprocessed_test_features,
            unprocessed_val_features,
        ),
        axis=0,
    )

    if args.save_raw_frags:
        id_to_raw_frag = {
            rxn_id: frag for rxn_id, frag in zip(rxn_ids, unprocessed_features)
        }

        with open(
            os.path.join(args.output_dir, "id_to_raw_frag.pkl"), "wb"
        ) as f:
            pickle.dump(id_to_raw_frag, f)

    if args.run_pca:
        n_components = args.n_components
        if n_components >= 1:
            n_components = int(n_components)

        train_features, pca = run_pca(unprocessed_train_features, n_components)
        test_features = pca.transform(unprocessed_test_features)
        val_features = pca.transform(unprocessed_val_features)

        combined_features = np.concatenate(
            (train_features, test_features, val_features), axis=0
        )

        id_to_frag = {
            rxn_id: frag for rxn_id, frag in zip(rxn_ids, combined_features)
        }

        with open(os.path.join(args.output_dir, "id_to_frag.pkl"), "wb") as f:
            pickle.dump(id_to_frag, f)

    with open(os.path.join(args.output_dir, "frag_config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
