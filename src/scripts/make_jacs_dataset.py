import argparse
import glob
import os
import subprocess

JACS_DATA_EXTRACTION_DIR = "../../jacs_data_extraction_scripts/data_extraction"


def parse_args():
    parser = argparse.ArgumentParser(description="Makes dataset for modelling")
    parser.add_argument(
        "-i",
        "--raw_file_path",
        type=str,
        default="../dataset/suzuki_USPTO_with_hetearomatic.txt",
        help="Path to the raw file",
    )
    parser.add_argument(
        "-p",
        "--output_prefix",
        type=str,
        default="jacs_data_",
        help="Output prefix for the parsed files",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="data", help="Output directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    command = f"python parseUSPTOdata.py -i {args.raw_file_path} -o {args.output_prefix}"
    subprocess.run(command, shell=True)

    files_to_keep = [
        f"{JACS_DATA_EXTRACTION_DIR}/{args.output_prefix}parsed_het.txt"
    ]

    # We then remove the external files:
    for file in glob.glob(f"{JACS_DATA_EXTRACTION_DIR}/*.txt"):
        if file not in files_to_keep:
            os.remove(file)

    command = f"python makeReactionFromParsed.py --parsed {args.output_prefix}parsed_het.txt --outprefix base7solv13 --makecanonsmiles --outputformat 'GAT' --auxoutformat 'base7solv13'"
    subprocess.run(command, shell=True)

    files_to_keep.append(f"{JACS_DATA_EXTRACTION_DIR}/base7solv13_valid0.txt")
    for file in glob.glob(f"{JACS_DATA_EXTRACTION_DIR}/*.txt"):
        if file not in files_to_keep:
            os.remove(file)

    command = f"python makeReactionFromParsed.py --parsed {args.output_prefix}parsed_het.txt --outprefix base7solv6 --makecanonsmiles --outputformat 'GAT' --auxoutformat 'base7solv6'"
    subprocess.run(command, shell=True)

    files_to_keep.append(f"{JACS_DATA_EXTRACTION_DIR}/base7solv6_valid0.txt")
    files_to_keep.remove(
        f"{JACS_DATA_EXTRACTION_DIR}/{args.output_prefix}parsed_het.txt"
    )
    for file in glob.glob(f"{JACS_DATA_EXTRACTION_DIR}/*.txt"):
        if file not in files_to_keep:
            os.remove(file)

    # Finally move these 3 files to the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for file in files_to_keep:
        os.rename(
            file,
            os.path.join(
                args.output_dir,
                os.path.basename(file).split("_")[0] + ".csv",
            ),
        )
