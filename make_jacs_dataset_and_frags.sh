#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate ml-env

cd jacs_data_extraction_scripts/data_extraction

python ../../src/scripts/make_jacs_dataset.py -i ../dataset/suzuki_USPTO_with_hetearomatic.txt -o ../../data/parsed_jacs_data

cd ../../

python src/scripts/parse_rxns.py -i data/parsed_jacs_data/base7solv6.csv \
-o data/parsed_jacs_data/mapped_base7solv6.csv \
--rerun_atom_mapping --add_rxn_id --parse_jacs_paper_data 

python src/scripts/merge_datasets_drop_dupes.py -i data/parsed_jacs_data/mapped_base7solv6.csv \
-o data/parsed_jacs_data

rm data/parsed_jacs_data/base7solv6.csv data/parsed_jacs_data/base7solv13.csv data/parsed_jacs_data/mapped_base7solv6.csv

python src/scripts/split_data.py -i data/parsed_jacs_data/valid_rxns.csv -o data/parsed_jacs_data/ -s 1 2 3 4 5 -f 5

mkdir data/parsed_jacs_data/splits
mv data/parsed_jacs_data/split* data/parsed_jacs_data/splits

conda deactivate
conda activate cgr-frag

python src/scripts/generate_frags_for_splits.py --splits_dir data/parsed_jacs_data/splits --n_folds 5 \

conda deactivate