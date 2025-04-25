"""Various utility functions for dealing with reactions."""

from functools import reduce
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import rdkit
import torch
import tqdm


def canonicalize_rxn(smiles_str: str) -> str:
    """Canonicalizes a smiles string.

    :param smiles_str: SMILES string
    :type smiles_str: str
    :return: Canonicalized SMILES string
    :rtype: str
    """
    can_smiles_str = rdkit.Chem.MolToSmiles(
        rdkit.Chem.MolFromSmiles(smiles_str)
    )
    return can_smiles_str


def parse_rxn(r_str: str, remap_rxn: bool = False) -> Tuple[str, str]:
    """Parses a reaction string performing some curation.

    Returns a mapped reaction string, without reagents and a comma separated
    string of reagents. Reagents are defined as species that do not contain any
    mapped atoms in the product.

    :param r_str: Input reaction string.
    :type r_str: str
    :param remap_rxn: Boolean flag to re-map the reaction, using Chython, defaults to False
    :type remap_rxn: bool, optional
    :return: Tuple containing the mapped reaction string and reagents.
    :rtype: Tuple[str, str]
    """
    import chython

    r = chython.smiles(r_str)
    if remap_rxn:
        r.reset_mapping()

    r.contract_ions()
    r.remove_reagents(keep_reagents=True)

    reagent_str = ""
    for reagent in r.reagents:
        can_reagent = canonicalize_rxn(str(reagent))
        reagent_str += can_reagent + ","

    # Remove the reagents from the reaction:
    r.remove_reagents(keep_reagents=False)

    return format(r, "m"), reagent_str


def parse_rxn_series(
    r_str_series: Iterable[str], remap_rxn: bool = False
) -> Tuple[list, list]:
    """Parses a series of reaction strings.

    For details see `parse_rxn`.

    :param r_str_series: Iterable of reaction strings.
    :type r_str_series: Iterable[str]
    :param remap_rxn: Flag to re-map reactions, defaults to False
    :type remap_rxn: bool, optional
    :return: Tuple containing lists of parsed reaction strings and reagents.
    :rtype: Tuple[list, list]
    """
    parsed_rxn_strings = []
    parsed_reagent_strings = []
    for r_str in tqdm.tqdm(r_str_series):
        parsed_rxn_string, parsed_reagent_string = parse_rxn(
            r_str, remap_rxn=remap_rxn
        )
        parsed_rxn_strings.append(parsed_rxn_string)
        parsed_reagent_strings.append(parsed_reagent_string)

    return parsed_rxn_strings, parsed_reagent_strings


def get_condition_information(
    condition_strs: Iterable[str],
    rxn_ids: Union[Iterable[int], None] = None,
    verbose: bool = False,
    separator: str = ",",
    return_errors: bool = False,
) -> Tuple[np.ndarray, list, list, list]:
    """Parses a string of conditions. Attempts to assign each reagent to a role,
    e.g. solvent, base, ligand etc. Combines into a single MHE condition vector.

    If we can't assign a role to a given reagent, this function skips the reagent.
    However, we will skip an entire condition string if we don't have a reagent
    in each role. This is to ensure that the conditions are 'valid' for Suzuki
    couplings.

    :param condition_strs: An iterable of condition strings.
    :type condition_strs: Iterable[str]
    :param rxn_ids: Optional, an iterable of corresponding rxn_ids, useful for debugging and investigating 'incorrect' conditions, defaults to None
    :type rxn_ids: Union[Iterable[int], None], optional
    :param verbose: Boolean flag for verbose parsing of conditions, defaults to False
    :type verbose: bool, optional
    :param separator: The separator that separates individual reagents, defaults to ","
    :type separator: str, optional
    :param return_errors: Boolean flag to return errors, defaults to False
    :type return_errors: bool, optional
    :return:
    :rtype: Tuple[np.ndarray, list, list, list]
    """
    valid_rxn_mask, rxn_conds, rxn_cond_vecs, other_reagents = [], [], [], []

    if return_errors:
        errors = []
    if rxn_ids is not None:
        for condition_str, rxn_id in tqdm.tqdm(
            zip(condition_strs, rxn_ids), total=len(condition_strs)
        ):
            processed_conditions = process_condition_str(
                condition_str=condition_str,
                rxn_id=rxn_id,
                verbose=verbose,
                separator=separator,
                return_error=return_errors,
            )

            valid_rxn = processed_conditions[0]

            valid_rxn_mask.append(valid_rxn)
            if valid_rxn:
                rxn_conds.append(processed_conditions[1])
                rxn_cond_vecs.append(processed_conditions[2])
                other_reagents.append(processed_conditions[3])
            else:
                if return_errors:
                    errors.append(processed_conditions[-1])

    else:
        for condition_str in tqdm.tqdm(condition_strs):
            processed_conditions = process_condition_str(
                condition_str=condition_str,
                verbose=verbose,
                separator=separator,
                return_error=return_errors,
            )
            valid_rxn = processed_conditions[0]

            valid_rxn_mask.append(valid_rxn)
            if valid_rxn:
                rxn_conds.append(processed_conditions[1])
                rxn_cond_vecs.append(processed_conditions[2])
                other_reagents.append(processed_conditions[3])
            else:
                if return_errors:
                    errors.append(processed_conditions[-1])

    if return_errors:
        return (
            np.array(valid_rxn_mask),
            rxn_conds,
            rxn_cond_vecs,
            other_reagents,
            errors,
        )
    else:
        return (
            np.array(valid_rxn_mask),
            rxn_conds,
            rxn_cond_vecs,
            other_reagents,
        )


def enumerate_conditions(reagent_class_sizes, multilabel_mask):
    from itertools import product

    all_condition_possibilities = None
    for reagent_class_size, multilabel in zip(
        reagent_class_sizes, multilabel_mask
    ):
        if multilabel:
            all_reagent_possiblities = list(
                [
                    list(comb)
                    for comb in product([0, 1], repeat=reagent_class_size)
                ]
            )
        else:
            all_reagent_possiblities = list(
                [
                    list(comb)
                    for comb in product([0, 1], repeat=reagent_class_size)
                    if sum(comb) == 1
                ]
            )

        if all_condition_possibilities is None:
            all_condition_possibilities = all_reagent_possiblities
        else:
            all_condition_possibilities = [
                comb1 + comb2
                for comb1 in all_condition_possibilities
                for comb2 in all_reagent_possiblities
            ]

    return torch.Tensor(all_condition_possibilities)


def get_condition_rankings(
    predictions: torch.Tensor,
    all_condition_possibilities: torch.Tensor,
    reagent_class_sizes: List[int],
    multilabel_mask: List[bool],
) -> torch.Tensor:
    """Gets the rankings of the different conditions based on the predicted probabilities.

    We deal with multilabel conditions differently from single-label conditions. In single-label,
    we multiply the predicted probabilities with the condition possibilities and sum over the
    reagents, which essentially just gets the predicted probability of the true reagent label.

    In multilabel, we multiply the predicted probabilities with the condition possibilities,
    but in the case where the true reagent label is not present, we take 1-probability of the
    reagent label.

    :param predictions: Predicted probabilities for each reagent class.
    :type predictions: torch.Tensor
    :param all_condition_possibilities: All possible conditions for the reagents.
    :type all_condition_possibilities: torch.Tensor
    :param reagent_class_sizes: Sizes of each reagent class.
    :type reagent_class_sizes: List[int]
    :param multilabel_mask: Mask indicating which reagent classes are multilabel.
    :type multilabel_mask: List[bool]
    :return: Ranked multi-head ensemble conditions.
    :rtype: torch.Tensor
    """
    n_samples = predictions.shape[0]
    total_conditions = all_condition_possibilities.shape[0]

    # Prepare a tensor to store likelihoods for all samples
    likelihoods = torch.zeros((n_samples, total_conditions))

    # Loop through each reagent class
    start_idx = 0
    for i, (reagent_class_size, multilabel) in enumerate(
        zip(reagent_class_sizes, multilabel_mask)
    ):
        end_idx = start_idx + reagent_class_size
        current_predictions = predictions[
            :, start_idx:end_idx
        ]  # Shape: (n_samples, reagent_class_size)

        # Current reagent condition possibilities
        current_condition_possibilities = all_condition_possibilities[
            :, start_idx:end_idx
        ]  # Shape: (n_conditions, reagent_class_size)

        if multilabel:
            # Multilabel: Include (1-p) for absence of labels
            current_likelihoods = (
                current_condition_possibilities
                * current_predictions.unsqueeze(1)
                + (1 - current_condition_possibilities)
                * (1 - current_predictions.unsqueeze(1))
            ).prod(dim=-1)  # Shape: (n_samples, n_conditions)
        else:
            # Single-label (multiclass): Simple multiplication
            current_likelihoods = (
                current_condition_possibilities
                * current_predictions.unsqueeze(1)
            ).sum(dim=-1)

        # Combine likelihoods for this reagent class
        if i == 0:
            likelihoods = current_likelihoods  # Initialize likelihoods for the first class
        else:
            likelihoods = (
                likelihoods * current_likelihoods
            )  # Combine across classes

        start_idx = end_idx

    # Reshape likelihoods to combine reagent possibilities and samples
    likelihoods = likelihoods.reshape(n_samples, total_conditions)

    # Rank conditions for each sample
    rankings = torch.argsort(likelihoods, dim=-1, descending=True)
    ranked_mhe_conditions = all_condition_possibilities[rankings]

    return ranked_mhe_conditions


def get_predicted_and_true_conditions(
    reagent_encoders: list,
    ranked_mhe_conditions: np.ndarray,
    labels: np.ndarray,
    reagent_class_sizes: list,
):
    """
    Enumerates different possibilities of conditions given the reagent class sizes and the multiclass mask.

    This function takes in encoders for reagents, ranked multi-head ensemble (MHE) conditions, true labels, and the sizes of reagent classes.
    It returns the predicted and true conditions for each reagent class.

    :param reagent_encoders: List of encoders for each reagent class.
    :type reagent_encoders: list
    :param ranked_mhe_conditions: Ranked multi-head ensemble conditions.
    :type ranked_mhe_conditions: numpy.ndarray
    :param labels: True labels for the conditions, in encoded form.
    :type labels: numpy.ndarray
    :param reagent_class_sizes: Sizes of each reagent class.
    :type reagent_class_sizes: list of int

    :return: A tuple containing:
        - condition_classes (np.ndarray): Predicted condition classes for each reagent (in string form).
        - indv_true_condition_classes (np.ndarray): True condition classes for each reagent (in string form).
    :rtype: tuple
    """
    condition_classes = None
    indv_true_condition_classes = None
    for i, (reagent_class_size, reagent_encoder) in enumerate(
        zip(reagent_class_sizes, reagent_encoders)
    ):
        lb = reagent_encoder

        start_idx = sum(reagent_class_sizes[:i])
        end_idx = start_idx + reagent_class_size

        indv_pred_condition_classes = None
        for j in range(ranked_mhe_conditions.shape[0]):
            if indv_pred_condition_classes is None:
                indv_pred_condition_classes = np.expand_dims(
                    np.array(
                        lb.inverse_transform(
                            ranked_mhe_conditions[j, :, start_idx:end_idx]
                        )
                    ).reshape(1, -1),
                    axis=-1,
                )

            else:
                indv_pred_condition_classes = np.concatenate(
                    [
                        indv_pred_condition_classes,
                        np.expand_dims(
                            np.array(
                                lb.inverse_transform(
                                    ranked_mhe_conditions[
                                        j, :, start_idx:end_idx
                                    ]
                                )
                            ).reshape(1, -1),
                            axis=-1,
                        ),
                    ],
                    axis=0,
                )

        if condition_classes is None:
            condition_classes = indv_pred_condition_classes
        else:
            condition_classes = np.concatenate(
                [
                    condition_classes,
                    indv_pred_condition_classes,
                ],
                axis=-1,
            )

        if indv_true_condition_classes is None:
            indv_true_condition_classes = lb.inverse_transform(
                labels[:, start_idx:end_idx]
            )
        else:
            indv_true_condition_classes = np.concatenate(
                [
                    indv_true_condition_classes,
                    lb.inverse_transform(labels[:, start_idx:end_idx]),
                ],
                axis=-1,
            )

    return condition_classes, indv_true_condition_classes


def get_top_k_accuracies(
    predictions: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    reagent_classes: List[str],
) -> Tuple[dict, dict]:
    """
    Calculates the top-k accuracies for overall matches and individual reagent matches.

    :param predictions: Predicted labels for each test point. Shape: (n_samples, n_reagent_classes, n_reagents)
    :type predictions: np.ndarray
    :param labels: True labels for each test point. Shape: (n_samples, n_reagents)
    :type labels: np.ndarray
    :param top_k: The maximum value of k for which to calculate the top-k accuracy.
    :type top_k: int
    :param reagent_classes: List of reagent class names.
    :type reagent_classes: List[str]
    :return: A tuple containing two dictionaries:
        - top_k_overall_accuracy: Dictionary with top-k overall accuracy values.
        - top_k_reagent_accuracy: Dictionary with top-k accuracy values for each reagent class.
    :rtype: Tuple[dict, dict]
    """
    overall_matches = []
    reagent_matches = []

    for test_point in range(labels.shape[0]):
        true_condition_label = labels[test_point]

        # Getting the overall match index
        overall_match_index = np.where(
            np.all(predictions[test_point] == true_condition_label, axis=1)
        )[0]
        overall_matches.append(overall_match_index)

        idv_reagent_matches = []
        for reagent_class_index in range(len(reagent_classes)):
            true_reagent_label = true_condition_label[reagent_class_index]
            predicted_reagent_labels = predictions[
                test_point, :, reagent_class_index
            ]

            match_indices = np.where(
                predicted_reagent_labels == true_reagent_label
            )[0]

            if len(match_indices) > 0:
                first_match_index = match_indices[0]
                idv_reagent_matches.append(first_match_index)

        reagent_matches.append(idv_reagent_matches)

    overall_matches = np.array(overall_matches)
    reagent_matches = np.array(reagent_matches)

    top_k_overall_accuracy = {}
    for i in range(top_k):
        top_k_overall_accuracy[i + 1] = sum(overall_matches < i + 1) / len(
            overall_matches
        )

    top_k_reagent_accuracy = {}
    for idx, reagent_class in enumerate(reagent_classes):
        top_k_reagent_accuracy[reagent_class] = {}
        for i in range(top_k):
            top_k_reagent_accuracy[reagent_class][i + 1] = sum(
                reagent_matches[:, idx] < i + 1
            ) / len(reagent_matches)

    return top_k_overall_accuracy, top_k_reagent_accuracy


def calc_top_k_condition_matches(
    df: pl.DataFrame,
    k_values: Optional[List[int]] = None,
    conditions_to_match: Optional[List[str]] = None,
    reaction_identifiers: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Calculates the top-k matches for each reaction, and for each condition specified.

    Also calculates the overall match for each reaction, corresponding to whether all
    conditions match.

    :param df: The dataframe to calculate the top-k matches for. **MUST** contain the columns:
    `{condition}` for the ground truth conditions, and `pred_{condition}` for the predicted conditions.
    :type df: pl.DataFrame
    :param k_values: A list of the top-k values to calculate matches for, defaults to
    [1, 3, 5, 10, 15]
    :type k_values: list[int], optional
    :param conditions_to_match: A list containing the different conditions to find matches for,
    defaults to [ "catalyst1", "solvent1", "solvent2", "reagent1", "reagent2"]
    :type conditions_to_match: list[str], optional
    :param reaction_identifiers: A list of column names from which a unique reaction can be uniquely
    identified, in this case, it is the canonical reaction and conditions, defaults to ["rxn_id"]
    :type reaction_identifiers: list[str], optional
    :return: The dataframe with the top-k matches for each condition and reaction. For each
    reaction, contains the original columns + len(k_values) * len(conditions_to_match) columns. With the columns containing booleans indicating whether the condition was a hit in the top-k
    predictions.
    :rtype: pl.DataFrame
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 15]
    if conditions_to_match is None:
        conditions_to_match = [
            "catalyst1",
            "solvent1",
            "solvent2",
            "reagent1",
            "reagent2",
        ]
    if reaction_identifiers is None:
        reaction_identifiers = ["rxn_id"]

    # Check match for each condition
    for cond in conditions_to_match:
        df = df.with_columns(
            [
                (
                    (
                        pl.col(f"{cond}") == pl.col(f"pred_{cond}")
                    )  # Check if values match
                    | (
                        pl.col(f"{cond}").is_null()
                        & pl.col(f"pred_{cond}").is_null()
                    )  # Check if both are null
                ).alias(f"{cond}_match")
            ]
        )

    # Compute the "overall" match by checking if all conditions match:
    overall_match_conditions = reduce(
        lambda x, y: x & y,
        [pl.col(f"{cond}_match") for cond in conditions_to_match],
    ).alias("overall_match")

    matched_df = df.with_columns([overall_match_conditions])

    # Compute the top-k hit (boolean) for each condition and reaction
    agg_exprs = []
    for cond in conditions_to_match + ["overall"]:
        for k in k_values:
            # Check if there's a hit (any match) within the top-k predictions
            expr = (
                (
                    pl.col(f"{cond}_match")
                    .sort_by("pred_number")
                    .head(k)
                    .max()
                    > 0
                )
                .fill_null(False)
                .alias(f"{cond}_top_{k}_hit")
            )
            agg_exprs.append(expr)

    # Group by the rxn_ids
    grouped_df = matched_df.group_by(reaction_identifiers).agg(agg_exprs)

    # Reshape the final dataframe to the desired format
    merged_unpivot_df = grouped_df.unpivot(
        index=reaction_identifiers,
        on=[
            f"{cond}_top_{k}_hit"
            for cond in conditions_to_match + ["overall"]
            for k in k_values
        ],
        variable_name="condition",
        value_name="hit",
    )

    # Reshape the dataframe to show top-k hits for each condition and reaction
    merged_pivot_df = merged_unpivot_df.pivot(
        index=reaction_identifiers, on="condition", values="hit"
    )

    return merged_pivot_df


def convert_match_df_to_summary_df(
    match_df: pl.DataFrame,
    groupby_cols: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    pivot_on_class_name: bool = True,
) -> pl.DataFrame:
    """Converts the matched dataframe of the type created by `calc_top_k_condition_matches` to a summary dataframe.

    The summary dataframe contains the percentage of hits within the top-k predictions for the group defined by groupby_cols.

    :param match_df: The matched dataframe containing boolean columns for each condition and top-k pair.
    :type match_df: pl.DataFrame
    :param groupby_cols: The columns to group by in the summary dataframe.
    :type groupby_cols: list[str]
    :param conditions: The conditions to calculate the top-k hits for, defaults to ["catalyst1", "solvent1", "solvent2", "reagent1", "reagent2", "overall"]
    :type conditions: list[str], optional
    :param k_values: The top-k values to calculate the hits for, defaults to [1, 3, 5, 10, 15]
    :type k_values: list[int], optional
    :return: The summary dataframe containing the percentage of top_k hits for each condition and group.
    :rtype: pl.DataFrame
    """
    if conditions is None:
        conditions = [
            "catalyst1",
            "solvent1",
            "solvent2",
            "reagent1",
            "reagent2",
            "overall",
        ]
    if k_values is None:
        k_values = [1, 3, 5, 10, 15]
    # Prepare aggregation expressions for each condition and top-k value
    agg_exprs = []
    for cond in conditions:
        for k in k_values:
            # Calculate the percentage of hits within the top-k predictions
            hit_expr = (pl.col(f"{cond}_top_{k}_hit").mean()).alias(
                f"{cond}_top_{k}"
            )
            agg_exprs.append(hit_expr)

    if groupby_cols is not None:
        # Group by rxn_class_name and compute top-k hit percentages
        grouped_df = match_df.group_by(groupby_cols).agg(agg_exprs)
    else:
        # Calculate the overall top-k hit percentages
        grouped_df = match_df.with_columns(agg_exprs)

    # Reshape the final dataframe to the desired format
    merged_pivot_df = grouped_df.unpivot(
        index=groupby_cols,
        on=[f"{cond}_top_{k}" for cond in conditions for k in k_values],
        variable_name="condition_and_k",
        value_name="percentage",
    )

    # Extract condition and k values from the 'condition' column
    merged_pivot_df = merged_pivot_df.with_columns(
        pl.col("condition_and_k")
        .str.split_exact("_top_", 1)
        .struct.rename_fields(["condition", "top_k"])
        .alias("fields")
    ).unnest("fields")

    # Pivot the dataframe so each row represents rxn_class_name, condition and top-k hit percentages
    if pivot_on_class_name:
        merged_pivot_df = merged_pivot_df.pivot(
            index=["rxn_class_name", "condition"],
            columns="top_k",
            values="percentage",
            aggregate_function="mean",
        )
    else:
        merged_pivot_df = merged_pivot_df.pivot(
            index="condition",
            columns="top_k",
            values="percentage",
            aggregate_function="mean",
        )

    # Rename columns to match top-k format
    merged_pivot_df = merged_pivot_df.rename(
        {str(k): f"top-{k}" for k in k_values}
    )

    return merged_pivot_df


def convert_df_for_plots(df: pl.DataFrame, data_source="") -> pl.DataFrame:
    """Converts dataframes produced by `convert_match_df_to_summary_df` to a format
      suitable for plotting.

    :param df: The summary dataframe
    :type df: pl.DataFrame
    :param data_source: The label to give this data, e.g. the method used to produce it,
    defaults to "".
    :type data_source: str, optional
    :return: The dataframe in a format suitable for plotting.
    :rtype: pl.DataFrame
    """
    overall_df = df.filter(pl.col("condition") == "overall")
    overall_df = overall_df.drop("condition")

    transposed_df = overall_df.transpose(
        include_header=True, header_name="top_k", column_names=["accuracy"]
    )

    transformed_df = transposed_df.with_columns(
        pl.col("top_k")
        .str.split("-")
        .list.get(-1)
        .cast(pl.Int32)
        .alias("top_k")
    )

    final_df = transformed_df.with_columns(
        pl.Series("method", [data_source] * len(transformed_df)).alias(
            "data_source"
        )
    )

    return final_df


def get_lrm_predictions(
    class_probabilities: Union[torch.Tensor, np.ndarray],
    mhe_labels: Union[torch.Tensor, np.ndarray],
    reagent_class_sizes: List[int],
    multilabel_mask: List[bool],
    reagent_encoders: List,
):
    """Get ranked predictions from input probabilities, using the likelihood ranking method.

    :param class_probabilities: Predicted probabilities for each reagent class, of shape [n_samples, total_n_classes]
    :type class_probabilities: Union(torch.Tensor, np.ndarray)
    :param mhe_labels: True labels for the conditions, in MHE form.
    :type mhe_labels: Union(torch.Tensor, np.ndarray)
    :param reagent_class_sizes: Sizes of each reagent class.
    :type reagent_class_sizes: List
    :param multilabel_mask: Mask indicating which reagent classes are multilabel.
    :type multilabel_mask: List

    :return: A tuple containing:
        - predicted_conditions (np.ndarray): Predicted condition classes for each reagent (in string form) of shape [n_samples, n_predictions, n_reagent_classes]
        - true_conditions (np.ndarray): True condition classes for each reagent (in string form) of shape [n_samples, n_reagent_classes]
    :rtype: tuple
    """
    all_condition_combinations = enumerate_conditions(
        reagent_class_sizes=reagent_class_sizes,
        multilabel_mask=multilabel_mask,
    )

    ranked_conditions = get_condition_rankings(
        predictions=class_probabilities,
        all_condition_possibilities=all_condition_combinations,
        reagent_class_sizes=reagent_class_sizes,
        multilabel_mask=multilabel_mask,
    )

    predicted_conditions, true_conditions = get_predicted_and_true_conditions(
        ranked_mhe_conditions=ranked_conditions,
        labels=mhe_labels,
        reagent_encoders=reagent_encoders,
        reagent_class_sizes=reagent_class_sizes,
    )

    return predicted_conditions, true_conditions
