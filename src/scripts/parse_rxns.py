"""Parses and curates reaction data from a csv and writes to a new file.

Involves a small amount of curation, if specified, can rerun the atom-mapping.
Uses atom-mapping to determine which species are reagents dependent on if they
contribute an atom to the product. Also combines ions into a single species,
this only works if there is a single ionic species in the reaction (so that the
combination of ions is unambiguous).

Adds a parsed_rxn column to the original csv file containing the reaction
string without any reagents, also contains a parsed_rxn column with the
reagents as a comma separated list.
"""

import argparse
import os
import pathlib
import sys

import polars as pl

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import parse_rxn_series


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input_csv", help="Path to the input csv file."
    )
    parser.add_argument(
        "-o", "--output_csv", help="Path to the output csv file."
    )
    parser.add_argument(
        "--rerun_atom_mapping",
        action="store_true",
        help="Rerun the atom mapping.",
    )
    parser.add_argument(
        "--rxn_column",
        default="NM_RxnSmiles",
        help="Name of the column containing the reaction strings.",
    )
    parser.add_argument(
        "--add_rxn_id",
        action="store_true",
        help="Add a column containing the reaction id.",
    )
    parser.add_argument(
        "--parse_jacs_paper_data",
        action="store_true",
        help="Special flag to parse the JACS paper data (Beker et al. 2022). Input file is a .txt with no header.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    if args.parse_jacs_paper_data:
        rxn_df = pl.read_csv(args.input_csv, separator=";")
        rxn_df = rxn_df.with_columns(
            pl.concat_str(
                [pl.col("Reactant1"), pl.col("Reactant2")], separator="."
            ).alias("Reactants"),
        )
        rxn_df = rxn_df.with_columns(
            pl.concat_str(
                [pl.col("Reactants"), pl.col("Product")], separator=">>"
            ).alias("NM_RxnSmiles"),
        )
        rxn_df = rxn_df.rename(
            {
                "S": "coarse_solvent",
                "B": "base",
            }
        )
        rxn_df = rxn_df.drop(["Reactant1", "Reactant2", "Product"])
    else:
        rxn_df = pl.read_csv(args.input_csv)

    parsed_rxn_strs, parsed_reagent_strs = parse_rxn_series(
        rxn_df[args.rxn_column], remap_rxn=args.rerun_atom_mapping
    )

    parsed_rxn_df = rxn_df.with_columns(
        [
            pl.Series("parsed_rxn", parsed_rxn_strs),
            pl.Series("parsed_reagents", parsed_reagent_strs),
        ]
    )

    if args.add_rxn_id:
        parsed_rxn_df = parsed_rxn_df.with_columns(
            pl.arange(0, parsed_rxn_df.height).alias("rxn_id")
        )

    parsed_rxn_df.write_csv(args.output_csv)
