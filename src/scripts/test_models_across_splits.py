"""Launches a series of experiments to test models across different data splits.

Assumes that the initial optimisation has been performed, and that the models have
been saved to disk in data/parsed_jacs_data/{solvent_classification}_solvs/{model_type}_results/split_seed_1/{model_type}_models.pkl.
"""

import argparse
import multiprocessing
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="rf",
        help="Type of model to use. Choices are 'cgr_rf', 'cgr_gbm', 'morgan_mtnn', 'cgr_mtnn', 'cgr_dmpnn', 'knn' and 'pop'",
    )
    parser.add_argument(
        "-s",
        "--solvent_classification",
        type=str,
        default="coarse",
        help="Type of solvent classification to use. Choices are 'coarse', 'fine' and 'exact'",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of data splits to test.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel.",
        required=False,
    )
    parser.add_argument(
        "--use_coarse_hparams",
        action="store_true",
        help="Whether to use the coarse hyperparameters for the fine solvent classification.",
        required=False,
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds to use for cross-validation.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="If enabled, will run predictions for the pytorch lightning-based models, rather than testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    command_str = ""
    output_dir = ""

    if args.model_type == "cgr_rf":
        command_str += "python src/modelling/rf.py"
    elif args.model_type == "cgr_gbm":
        command_str += "python src/modelling/gb.py"
    elif args.model_type in ["morgan_mtnn", "cgr_mtnn"]:
        command_str += "python src/modelling/multitask_nn.py"
    elif args.model_type == "cgr_dmpnn":
        command_str += "python src/modelling/cgr_dmpnn.py"
    elif args.model_type == "knn":
        command_str += "python src/modelling/knn.py"
    elif args.model_type == "pop":
        command_str += "python src/modelling/pop_baseline.py"
    else:
        raise ValueError(
            "Invalid model type. Choices are 'cgr_rf', 'cgr_gbm', 'morgan_mtnn', 'cgr_mtnn', 'knn' and 'pop'"
        )

    if args.solvent_classification == "fine":
        target_cols = ["fine_solvent", "base"]
    elif args.solvent_classification == "coarse":
        target_cols = ["coarse_solvent", "base"]
    elif args.solvent_classification == "exact":
        target_cols = ["exact_solvent", "exact_base"]
    else:
        raise ValueError(
            "Invalid solvent classification. Choices are 'coarse', 'fine' and 'exact'"
        )

    if args.model_type in ["cgr_rf", "cgr_gbm", "cgr_dmpnn"]:
        commands = []

        for split in range(1, args.n_splits + 1):
            if args.model_type == "cgr_rf":
                output_dir = (
                    f"{args.solvent_classification}_solvs/cgr_rf_results"
                )
            elif args.model_type == "cgr_gbm":
                output_dir = (
                    f"{args.solvent_classification}_solvs/cgr_gbm_results"
                )
            elif args.model_type == "cgr_dmpnn":
                output_dir = (
                    f"{args.solvent_classification}_solvs/cgr_dmpnn_results"
                )

            # If we are using the fine solvent classification, and we want to use the coarse hyperparameters:
            if (
                args.solvent_classification == "fine"
                and args.use_coarse_hparams
            ):
                model_input_path = f"hpopt/coarse_solvs/{args.model_type}_results/split_seed_{split}/fold_1/model_configs.yml"
                output_dir += "_coarse_hparams"

                # Since we are using the coarse hyperparameters, we won't have tested on the 1st split.
                fold_start_idx = 1
            else:
                model_input_path = f"hpopt/{args.solvent_classification}_solvs/{args.model_type}_results/split_seed_{split}/fold_1/model_configs.yml"
                fold_start_idx = 2

            # We don't necessarily start the loop at 1, since the hyperparameter optimisation will test on the 1st split.
            for fold in range(fold_start_idx, args.n_folds + 1):
                if args.model_type in ["cgr_rf", "cgr_gbm"]:
                    if (
                        args.use_coarse_hparams
                        and args.solvent_classification == "fine"
                    ):
                        model_output_dir = f"models/fine_solvs_coarse_hparams/{args.model_type}/split_seed_{split}/fold_{fold}"
                    else:
                        model_output_dir = f"models/{args.solvent_classification}_solvs/{args.model_type}/split_seed_{split}/fold_{fold}"

                    command_str = f"{command_str} -i data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/ -d data.csv -o data/parsed_jacs_data/{output_dir}/split_seed_{split}/fold_{fold} -m {model_input_path} --target_cols {' '.join(target_cols)} --model_output_dir {model_output_dir}"

                    if args.model_type == "cgr_rf":
                        if (
                            args.use_coarse_hparams
                            and args.solvent_classification == "fine"
                        ):
                            command_str += " -k coarse_solvent base"
                    else:
                        # GBM can use GPU
                        command_str = f"{command_str} --gpu"
                        # If we are using fine solvent classification, and the coarse hparams, we will need to refit the encoders:
                        if (
                            args.use_coarse_hparams
                            and args.solvent_classification == "fine"
                        ):
                            # Only fit the encoders once per split:
                            if fold == 1:
                                command_str += " --path_to_encoders fit -k coarse_solvent base"
                            else:
                                command_str += f" --path_to_encoders models/fine_solvs_coarse_hparams/cgr_gbm/split_seed_{split}/fold_1/encoders.pkl -k coarse_solvent base"
                        else:
                            if fold == 1:
                                command_str += " --path_to_encoders fit"
                            else:
                                command_str += f" --path_to_encoders models/{args.solvent_classification}_solvs/cgr_gbm/split_seed_{split}/fold_1/encoders.pkl"

                elif args.model_type == "cgr_dmpnn":
                    num_workers = 32
                    if (
                        args.use_coarse_hparams
                        and args.solvent_classification == "fine"
                    ):
                        model_input_path = f"hpopt/coarse_solvs/{args.model_type}_results/split_seed_{split}/fold_1/best_model_config.yml"
                        command_str = f"{command_str} -i data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/data.csv -o data/parsed_jacs_data/{output_dir}/split_seed_{split}/fold_{fold} -t {' '.join(target_cols)} -m {model_input_path} -c {num_workers} -s models/fine_solvs_coarse_hparams/{args.model_type}/split_seed_{split}/fold_{fold}"
                    else:
                        model_input_path = f"hpopt/{args.solvent_classification}_solvs/{args.model_type}_results/split_seed_{split}/fold_1/best_model_config.yml"
                        command_str = f"{command_str} -i data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/data.csv -o data/parsed_jacs_data/{output_dir}/split_seed_{split}/fold_{fold} -t {' '.join(target_cols)} -m {model_input_path} -c {num_workers} -s models/{args.solvent_classification}_solvs/{args.model_type}/split_seed_{split}/fold_{fold}"

                commands.append(command_str)

        if args.model_type == "cgr_rf":
            # Run the experiments in parallel
            pool = multiprocessing.Pool(args.n_jobs)
            pool.map(os.system, commands)
            pool.close()
            pool.join()
        else:
            # GBM runs individually on GPU
            for command in commands:
                os.system(command)

    elif args.model_type in ["morgan_mtnn", "cgr_mtnn"]:
        train_commands = []
        test_commands = []
        evaluation_arg = "predict" if args.predict else "test"
        for split in range(1, args.n_splits + 1):
            for fold in range(1, args.n_folds + 1):
                if args.model_type == "morgan_mtnn":
                    model_datamodule = "MorganFPReactionDataModule"
                elif args.model_type == "cgr_mtnn":
                    model_datamodule = "FragDatamodule"

                if args.model_type == "morgan_mtnn":
                    if args.solvent_classification == "exact":
                        config_dir_path = f"configs/{args.solvent_classification}_solvs/{args.model_type}/split_seed_{split}"
                    else:
                        config_dir_path = f"configs/{args.solvent_classification}_solvs/{args.model_type}"

                    model_checkpoint_dir = f"models/{args.solvent_classification}_solvs/{args.model_type}/split_seed_{split}/fold_{fold}"
                elif (
                    args.use_coarse_hparams
                    and args.solvent_classification == "fine"
                ):
                    config_dir_path = f"configs/fine_solvs_coarse_hparams/{args.model_type}/split_seed_{split}"
                    model_checkpoint_dir = f"models/fine_solvs_coarse_hparams/{args.model_type}/split_seed_{split}/fold_{fold}"
                else:
                    config_dir_path = f"configs/{args.solvent_classification}_solvs/{args.model_type}/split_seed_{split}"
                    model_checkpoint_dir = f"models/{args.solvent_classification}_solvs/{args.model_type}/split_seed_{split}/fold_{fold}"

                train_command = """
                {command_str} fit \
                -c {config_dir_path}/train.yml \
                --data data.{model_datamodule} \
                --data.data_path data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/data.csv \
                --data.target_cols "{target_cols}" \
                --checkpoint_config.dirpath {model_checkpoint_dir} \
                """.format(
                    command_str=command_str,
                    config_dir_path=config_dir_path,
                    model_datamodule=model_datamodule,
                    fold=fold,
                    split=split,
                    target_cols=target_cols,
                    model_checkpoint_dir=model_checkpoint_dir,
                )

                test_command = """
                {command_str} {evaluation_argument} \
                -c {config_dir_path}/test.yml \
                --data data.{model_datamodule} \
                --data.data_path data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/data.csv \
                --data.target_cols "{target_cols}" \
                --ckpt_path {model_checkpoint_dir}/best_val_loss.ckpt \
                """.format(
                    evaluation_argument=evaluation_arg,
                    command_str=command_str,
                    config_dir_path=config_dir_path,
                    model_datamodule=model_datamodule,
                    fold=fold,
                    split=split,
                    target_cols=target_cols,
                    model_checkpoint_dir=model_checkpoint_dir,
                )

                if (
                    args.use_coarse_hparams
                    and args.solvent_classification == "fine"
                ):
                    test_command += f" --model.metric_save_dir data/parsed_jacs_data/{args.solvent_classification}_solvs/{args.model_type}_results_coarse_hparams/split_seed_{split}/fold_{fold}"
                else:
                    test_command += f" --model.metric_save_dir data/parsed_jacs_data/{args.solvent_classification}_solvs/{args.model_type}_results/split_seed_{split}/fold_{fold}"

                if args.model_type == "cgr_mtnn":
                    train_command += " --data.id_to_frag_path data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/id_to_frag.pkl".format(
                        fold=fold, split=split
                    )
                    test_command += " --data.id_to_frag_path data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/id_to_frag.pkl".format(
                        fold=fold, split=split
                    )

                train_commands.append(train_command)
                test_commands.append(test_command)

        for command in train_commands:
            os.system(command)

        for command in test_commands:
            os.system(command)
    elif args.model_type in ["pop", "knn"]:
        commands = []
        for split in range(1, args.n_splits + 1):
            for fold in range(1, args.n_folds + 1):
                commands.append(
                    f"{command_str} -i data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/data.csv -o data/parsed_jacs_data/{args.solvent_classification}_solvs/baseline_{args.model_type}_results/split_seed_{split}/fold_{fold} -t {' '.join(target_cols)}"
                )

        if args.model_type == "knn":
            for i, command in enumerate(commands):
                command += f" -f data/parsed_jacs_data/splits/split_seed_{split}/fold_{fold}/id_to_frag.pkl "
                commands[i] = command

        # Run the experiments in parallel
        pool = multiprocessing.Pool(args.n_jobs)
        pool.map(os.system, commands)
        pool.close()
        pool.join()
        pool.join()
