"""Merges the 2 JACS datasets and drops duplicates, before saving the result to a new file."""

import argparse
import os
import pathlib

import loguru
import polars as pl

logger = loguru.logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge the 2 JACS datasets and drop duplicates."
    )
    parser.add_argument(
        "-i",
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file, assumes that the other base7solv13.csv file is in the same folder.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory to save the merged CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mapped_df = pl.read_csv(args.input_csv)
    fine_solv_df = pl.read_csv(
        os.path.join(pathlib.Path(args.input_csv).parent, "base7solv13.csv"),
        separator=";",
    )
    fine_solv_df = fine_solv_df.rename({"S": "fine_solvent"})
    mapped_df = mapped_df.with_columns(
        [
            pl.Series("fine_solvent", fine_solv_df["fine_solvent"].to_numpy()),
        ]
    )

    # Drop duplicates
    logger.info(
        f"Number of reactions before dropping duplicates: {len(mapped_df)}"
    )
    deduped_df = mapped_df.unique(
        subset=["parsed_rxn", "base", "coarse_solvent", "fine_solvent"]
    )
    logger.info(
        f"Number of reactions after dropping duplicates: {len(deduped_df)}"
    )

    # Save the merged dataset
    output_path = os.path.join(args.output_dir, "valid_rxns.csv")
    deduped_df.write_csv(output_path)
