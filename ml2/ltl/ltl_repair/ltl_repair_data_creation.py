""" Library to create a dataset for repair training based on given raw data (e.g. evaluation data from ltl_repair_gen) """


import os
import json
import shutil
from typing import Dict, List, Optional, Tuple
import pandas
from Levenshtein import distance as lev
import matplotlib.pyplot as plt
from ml2.globals import LOCAL_STORAGE_DIR, LTL_REP_BUCKET_DIR
from ml2.ltl.ltl_repair.ltl_repair_data import LTLRepairSplitData
from ml2.ltl.ltl_repair.ltl_repair_data_gen import LTLRepairGenData
import argparse
import warnings

from tqdm.auto import tqdm

warnings.warn("This module is deprecated and may not work", DeprecationWarning)


def load_from_file(path: str, split: str) -> pandas.DataFrame:
    dataframe = pandas.read_csv(os.path.join(path, split + ".csv"))
    dataframe = dataframe.fillna("")
    return dataframe[dataframe["status"] != "Timeouts"]


def calculate_levenshtein(
    dataframe: pandas.DataFrame,
    over: List[Dict] = [
        {
            "between_1": "repair_circuit",
            "between_2": "circuit",
            "res": "levenshtein_distance",
        }
    ],
    set_match: bool = True,
) -> pandas.DataFrame:
    def apply(x: pandas.DataFrame, i: int, cols: List[str]) -> pandas.DataFrame:
        res = []
        for el in cols:
            if el == "status" and set_match:
                res.append(
                    "Match"
                    if set_match and lev(x["repair_circuit"], x["circuit"]) == 0
                    else x["status"]
                )
            else:
                res.append(x[el])
        res.append(lev(x[over[i]["between_1"]], x[over[i]["between_2"]]))
        return res

    for i in range(len(over)):
        cols = dataframe.columns
        if over[i]["between_1"] not in cols or over[i]["between_2"] not in cols:
            raise ValueError("Arguments passed to calculate_levenshtein not consistent")
        dataframe = dataframe.apply(
            lambda x: apply(x, i, cols),
            axis=1,
            result_type="expand",
        )
        cols = cols.append(pandas.Index([over[i]["res"]]))
        dataframe.columns = cols
    return dataframe


def filter_matches(dataframe: pandas.DataFrame, max_match_fraction: float) -> pandas.DataFrame:
    if max_match_fraction == 0.0:
        return dataframe[dataframe["status"] != "Match"]
    match = dataframe[(dataframe["status"] == "Satisfied") | (dataframe["status"] == "Match")]
    if dataframe.count()[0] == match.count()[0] and max_match_fraction != 1.0:
        print("Warning: tried filtering with match fraction but dataset contains 100% matches")
        return dataframe
    new_fraction = (
        min(
            (
                ((dataframe.count()[0] - match.count()[0]) / (1 - max_match_fraction))
                - (dataframe.count()[0] - match.count()[0])
            )
            / match.count()[0],
            1,
        )
        if max_match_fraction < 1
        else 1
    )
    new_matches = match.sample(frac=new_fraction)
    return (
        dataframe[(dataframe["status"] != "Satisfied") & (dataframe["status"] != "Match")]
        .append(new_matches)
        .reset_index()
    )


def add_satisfied_token(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    raise NotImplementedError()


def filter_distance_bigger(dataframe: pandas.DataFrame, distance: int) -> pandas.DataFrame:
    return dataframe[dataframe["levenshtein_distance"] <= distance]


def replace_satisfied_with_match(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    tqdm.pandas(desc="replacing target with predicted circuit for satisfied samples")
    df = dataframe.progress_apply(
        lambda x: [
            "Match" if x["status"] == "Satisfied" else x["status"],
            x["assumptions"],
            x["guarantees"],
            x["repair_circuit"],
            x["inputs"],
            x["outputs"],
            x["realizable"],
            x["repair_circuit"] if x["status"] == "Satisfied" else x["circuit"],
            0 if x["status"] == "Satisfied" else lev(x["repair_circuit"], x["circuit"]),
        ],
        axis=1,
        result_type="expand",
    )
    df.columns = [
        "status",
        "assumptions",
        "guarantees",
        "repair_circuit",
        "inputs",
        "outputs",
        "realizable",
        "circuit",
        "levenshtein_distance",
    ]
    return df


def find_minimal_distance(
    row: pandas.DataFrame, dataframe_reference: pandas.DataFrame
) -> Tuple[str, int]:
    guarantees_match = dataframe_reference["guarantees"] == row["guarantees"]
    assumptions_match = dataframe_reference["assumptions"] == row["assumptions"]
    same_spec = dataframe_reference[assumptions_match & guarantees_match]
    lv = same_spec.apply(
        lambda x: [x["circuit"], lev(x["circuit"], row["repair_circuit"])],
        axis=1,
        result_type="expand",
    ).reset_index()
    if len(lv) == 0 or (lv.iloc[lv[1].idxmin()][1] > row["levenshtein_distance"]):
        return row["circuit"], row["levenshtein_distance"]
    else:
        return lv.iloc[lv[1].idxmin()][0], lv.iloc[lv[1].idxmin()][1]


def replace_minimal_distance(
    dataframe: pandas.DataFrame, dataframe_reference: pandas.DataFrame
) -> pandas.DataFrame:
    def apply_fn(row: pandas.DataFrame, dataframe_reference: pandas.DataFrame) -> pandas.DataFrame:
        if row["status"] == "Match":
            return [
                row["status"],
                row["assumptions"],
                row["guarantees"],
                row["repair_circuit"],
                row["inputs"],
                row["outputs"],
                row["realizable"],
                row["circuit"],
                0,
            ]
        circuit, lv = find_minimal_distance(row, dataframe_reference)
        return [
            row["status"],
            row["assumptions"],
            row["guarantees"],
            row["repair_circuit"],
            row["inputs"],
            row["outputs"],
            row["realizable"],
            circuit,
            lv,
        ]

    dataframe_match_replaced: pandas.DataFrame = replace_satisfied_with_match(dataframe=dataframe)
    dataframe_reference_match_replaced: pandas.DataFrame = replace_satisfied_with_match(
        dataframe=dataframe_reference
    )
    dataframe_reference_match_replaced = dataframe_reference_match_replaced[
        dataframe_reference_match_replaced["levenshtein_distance"] == 0
    ]

    tqdm.pandas(desc="replacing misleading target with minimal distance target")
    dataframe_minimal_replaced = dataframe_match_replaced.progress_apply(
        lambda x: apply_fn(row=x, dataframe_reference=dataframe_reference_match_replaced),
        axis=1,
        result_type="expand",
    )
    dataframe_minimal_replaced.columns = [
        "status",
        "assumptions",
        "guarantees",
        "repair_circuit",
        "inputs",
        "outputs",
        "realizable",
        "circuit",
        "levenshtein_distance",
    ]
    return dataframe_minimal_replaced


def counts(dataframe: pandas.DataFrame) -> Dict:
    realizable = dataframe["realizable"] == 1
    unrealizable = dataframe["realizable"] == 0
    invalid = dataframe["status"] == "Invalid"
    error_ = dataframe["status"] == "Error"
    violated = dataframe["status"] == "Violated"
    satisfied = dataframe["status"] == "Satisfied"
    match = dataframe["status"] == "Match"
    return {
        "Invalid": {
            "realizable": len(dataframe[realizable & invalid]),
            "unrealizable": len(dataframe[unrealizable & invalid]),
        },
        "Error": {
            "realizable": len(dataframe[realizable & error_]),
            "unrealizable": len(dataframe[unrealizable & error_]),
        },
        "Violated": {
            "realizable": len(dataframe[realizable & violated]),
            "unrealizable": len(dataframe[unrealizable & violated]),
        },
        "Satisfied": {
            "realizable": len(dataframe[realizable & satisfied]),
            "unrealizable": len(dataframe[unrealizable & satisfied]),
        },
        "Match": {
            "realizable": len(dataframe[realizable & match]),
            "unrealizable": len(dataframe[unrealizable & match]),
        },
    }


def metrics(dataframe: pandas.DataFrame) -> Dict:
    return {
        "realizable_fraction": len(dataframe[dataframe["realizable"] == 1]) / len(dataframe),
        "satisfied_fraction": len(dataframe[dataframe["status"] == "Satisfied"]) / len(dataframe),
        "match_fraction": len(dataframe[dataframe["levenshtein_distance"] == 0]) / len(dataframe),
        "distance_lower_10_fraction": len(dataframe[dataframe["levenshtein_distance"] < 10])
        / len(dataframe),
        "distance_lower_50_fraction": len(dataframe[dataframe["levenshtein_distance"] < 50])
        / len(dataframe),
        "distance_broken_mean": dataframe[dataframe["levenshtein_distance"] != 0][
            "levenshtein_distance"
        ].mean(),
        "distance_broken_std": dataframe[dataframe["levenshtein_distance"] != 0][
            "levenshtein_distance"
        ].std(),
        "distance_broken_median": dataframe[dataframe["levenshtein_distance"] != 0][
            "levenshtein_distance"
        ].median(),
        "distance_all_mean": dataframe["levenshtein_distance"].mean(),
        "distance_all_std": dataframe["levenshtein_distance"].std(),
        "distance_all_median": dataframe["levenshtein_distance"].median(),
        "realizable_count": len(dataframe[dataframe["realizable"] == 1]),
        "satisfied_count": len(dataframe[dataframe["status"] == "Satisfied"]),
        "match_count": len(dataframe[dataframe["levenshtein_distance"] == 0]),
        "distance_lower_10_count": len(dataframe[dataframe["levenshtein_distance"] < 10]),
        "distance_lower_50_count": len(dataframe[dataframe["levenshtein_distance"] < 50]),
        "total_count": len(dataframe),
    }


def save_plots_to_file(
    dataframe: pandas.DataFrame, target_path: str, split: str, before: bool = False
) -> None:
    before_string = "before" if before else "after"
    if len(dataframe[dataframe["levenshtein_distance"] > 0]):
        plot_levenshtein_counts_overview(dataframe)
        plt.savefig(os.path.join(target_path, split + "_lv_" + before_string + "_overview.png"))
        plt.show()  # weirdly needs this
        plot_levenshtein_counts_detail(dataframe)
        plt.savefig(os.path.join(target_path, split + "_lv_" + before_string + "_details.png"))
        plt.show()  # weirdly needs this
    dataframe.boxplot(column="levenshtein_distance")
    plt.savefig(os.path.join(target_path, split + "_lv_" + before_string + "_boxplot.png"))
    plt.show()  # weirdly needs this


def plot_levenshtein_counts_detail(dataframe: pandas.DataFrame) -> None:
    filter_zero = dataframe.levenshtein_distance[dataframe["levenshtein_distance"] != 0]
    ax = filter_zero.value_counts(sort=False).sort_index()
    ax[:100].plot(kind="bar", figsize=(25, 7))


def plot_levenshtein_counts_overview(dataframe: pandas.DataFrame) -> None:
    bins = [0.9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 260]
    ax = dataframe.levenshtein_distance.value_counts(bins=bins, sort=False).sort_index()
    ax.plot(kind="bar", figsize=(15, 7))


def save_split_to_file(dataframe: pandas.DataFrame, target_path: str, split: str) -> None:
    if "levenshtein_distance" in dataframe.columns:
        dataframe = dataframe.drop("levenshtein_distance", axis=1)
    dataframe.to_csv(os.path.join(target_path, split + ".csv"), header=True, index=False)


def name_from_arguments(
    alpha: float,
    beamsize: int,
    filter_distance: int,
    filter_quantile: float,
    max_match_fraction: float,
    replace_minimal: bool,
    replace_satisfied: bool,
    include_satisfied_token: bool,
    parent_dataset: str,
) -> str:
    max_match_fraction_name = "sf" + str(max_match_fraction)
    satisfied_name = "t1" if include_satisfied_token else "t0"
    replace_satisfied_name = "sr1" if replace_satisfied else "sr0"
    replace_minimal_name = "mr1" if replace_minimal else "mr0"
    if filter_distance != 0 and filter_quantile != 0:
        raise ValueError("Only set filter_quantile or filter_distance")
    filter_distance_name = (
        "d" + str(filter_distance)
        if filter_distance != 0
        else ("q" + str(filter_quantile) if filter_quantile != 0 else "d0")
    )
    return (
        parent_dataset
        + "-repair-"
        + satisfied_name
        + "-"
        + replace_satisfied_name
        + "-"
        + replace_minimal_name
        + "-"
        + filter_distance_name
        + "-"
        + max_match_fraction_name
        + "-"
        + "a"
        + str(alpha)
        + "-"
        + "bs"
        + str(beamsize)
    )


def read_from_eval_arguments(
    alpha: float,
    beamsize: int,
    split: str,
    model: str,
    samples: Optional[int] = None,
    from_log: bool = False,
) -> pandas.DataFrame:
    samples_str = "-n" + str(samples) if samples is not None else ""
    folder = split if from_log else "gen"
    path = os.path.join(
        LOCAL_STORAGE_DIR,
        LTL_REP_BUCKET_DIR,
        model,
        "eval",
        folder,
        "a" + str(alpha) + "-bs" + str(beamsize) + samples_str,
    )
    split = "log" if from_log else split
    return load_from_file(path, split)


def get_parent_dataset(parent_model: str) -> str:
    path = os.path.join(LOCAL_STORAGE_DIR, LTL_REP_BUCKET_DIR, parent_model, "args.json")
    with open(path, "r") as parent_arguments_file:
        parent_arguments_buffer = parent_arguments_file.read()
        parent_arguments = json.loads(parent_arguments_buffer)
        return parent_arguments["dataset_name"]


def get_input_path(alpha: int, beamsize: int, parent_model: str) -> str:
    return os.path.join(
        LOCAL_STORAGE_DIR,
        LTL_REP_BUCKET_DIR,
        parent_model,
        "eval",
        "gen",
        "a" + str(alpha) + "-bs" + str(beamsize),
    )


def create_dummy_circuit_split(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    col = pandas.Series(
        "aag 10 5 0 5 0\\n2\\n4\\n6\\n8\\n10\\n2\\n4\\n6\\n8\\n10", index=range(len(dataframe))
    )
    dataframe["repair_circuit"] = col
    return dataframe


def create_dummy_spec_split(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    col1 = pandas.Series("i0", index=range(len(dataframe)))
    col2 = pandas.Series("", index=range(len(dataframe)))
    dataframe["assumptions"] = col2
    dataframe["guarantees"] = col1
    return dataframe


def create_broken_test_split(parent_model: str, target_path: str) -> None:
    alpha = 0.5
    for beamsize in [1, 4, 8, 16]:
        try:
            dataframe = read_from_eval_arguments(
                alpha=alpha,
                beamsize=beamsize,
                split="test",
                model=parent_model,
                samples=1024,
            )
            grp = dataframe.groupby(["guarantees", "assumptions"])
            collected = []
            for _, group in grp:
                filtered = group[(group["status"] == "Satisfied") | (group["status"] == "Match")]
                if filtered.count()[0] == 0:
                    collected.append(group)

            new = pandas.DataFrame([], columns=dataframe.columns)
            new = new.append(collected, ignore_index=True)
            save_split_to_file(
                dataframe=new,
                target_path=target_path,
                split="broken-a" + str(alpha) + "-bs" + str(beamsize),
            )
        except FileNotFoundError:
            print(
                "File not found: alpha: {}, beam size: {}, samples: 1024".format(alpha, beamsize)
            )


def create_and_save_test_splits(
    dataframe: pandas.DataFrame, target_path: str, parent_model: str
) -> None:
    save_split_to_file(
        dataframe=create_dummy_circuit_split(dataframe=dataframe.copy()),
        target_path=target_path,
        split="dummy_circuit",
    )
    save_split_to_file(
        dataframe=create_dummy_spec_split(dataframe=dataframe.copy()),
        target_path=target_path,
        split="dummy_spec",
    )
    create_broken_test_split(target_path=target_path, parent_model=parent_model)


def get_output_path(
    alpha: float,
    beamsize: int,
    filter_distance: int,
    filter_quantile: float,
    max_match_fraction: float,
    replace_minimal: bool,
    replace_satisfied: bool,
    include_satisfied_token: bool,
    parent_dataset: str,
) -> str:
    name = name_from_arguments(
        alpha=alpha,
        beamsize=beamsize,
        max_match_fraction=max_match_fraction,
        filter_distance=filter_distance,
        filter_quantile=filter_quantile,
        replace_satisfied=replace_satisfied,
        replace_minimal=replace_minimal,
        include_satisfied_token=include_satisfied_token,
        parent_dataset=parent_dataset,
    )
    return os.path.join(
        LOCAL_STORAGE_DIR,
        LTL_REP_BUCKET_DIR,
        name,
    )


def create_split(
    split: str,
    alpha: float,
    beamsize: int,
    reference_beamsizes: List[int],
    filter_distance: int,
    filter_quantile: float,
    max_match_fraction: float,
    replace_minimal: bool,
    replace_satisfied: bool,
    include_satisfied_token: bool,
    parent_model: str,
    target_path: str,
) -> Tuple[Dict, Dict, Dict]:
    if filter_distance != 0 and filter_quantile != 0.0:
        raise ValueError("Only set filter_quantile or filter_distance")

    dataframe = read_from_eval_arguments(
        alpha=alpha, beamsize=beamsize, split=split, model=parent_model
    )
    dataframe = calculate_levenshtein(dataframe=dataframe)
    save_plots_to_file(dataframe=dataframe, target_path=target_path, split=split, before=True)
    if replace_satisfied and not replace_minimal:
        dataframe = replace_satisfied_with_match(dataframe=dataframe)
    elif replace_minimal:
        reference_dataframes = []
        for reference_beamsize in reference_beamsizes:
            reference_dataframes.append(
                read_from_eval_arguments(
                    alpha=alpha,
                    beamsize=reference_beamsize,
                    split=split,
                    model=parent_model,
                )
            )
        reference_dataframe = pandas.concat(
            reference_dataframes, ignore_index=True
        ).drop_duplicates()
        dataframe = replace_minimal_distance(
            dataframe=dataframe, dataframe_reference=reference_dataframe
        )
    if filter_distance != 0 and split == "train":
        dataframe = filter_distance_bigger(dataframe=dataframe, distance=filter_distance)
    if split == "train":
        dataframe = filter_matches(dataframe=dataframe, max_match_fraction=max_match_fraction)
        quantile_distance = dataframe.levenshtein_distance.quantile(filter_quantile)
        dataframe = filter_distance_bigger(dataframe=dataframe, distance=quantile_distance)
        dataframe = filter_matches(dataframe=dataframe, max_match_fraction=max_match_fraction)
    metrics_dict = metrics(dataframe=dataframe)
    if split == "train":
        metrics_dict["filter_distance"] = round(quantile_distance)
    count_dict = counts(dataframe=dataframe)
    if include_satisfied_token:
        dataframe = add_satisfied_token(dataframe=dataframe)
    dataframe = dataframe.reset_index(drop=True)
    if split == "test":
        create_and_save_test_splits(
            dataframe=dataframe, target_path=target_path, parent_model=parent_model
        )
    save_plots_to_file(dataframe=dataframe, target_path=target_path, split=split)
    save_split_to_file(dataframe=dataframe, target_path=target_path, split=split)
    return dataframe, metrics_dict, count_dict


def write_counts(count_dict: Dict, target_path: str) -> None:
    jsonString = json.dumps(count_dict, indent=4)
    jsonFile = open(os.path.join(target_path, "counts.json"), "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def write_metadata(
    alpha: float,
    beamsize: int,
    filter_distance: int,
    filter_quantile: float,
    max_match_fraction: float,
    replace_minimal: bool,
    replace_satisfied: bool,
    include_satisfied_token: bool,
    parent_dataset: str,
    parent_model: str,
    target_path: str,
    metrics: Dict,
) -> None:
    if filter_distance != 0 and filter_quantile != 0.0:
        raise ValueError("Only set filter_quantile or filter_distance")

    if filter_distance == 0 and filter_quantile != 0.0:
        filter_distance = metrics["train"]["filter_distance"]

    metadata = {}
    train_frac = metrics["train"]["total_count"] / metrics["total"]["total_count"]
    val_frac = metrics["val"]["total_count"] / metrics["total"]["total_count"]
    test_frac = metrics["test"]["total_count"] / metrics["total"]["total_count"]
    metadata["num_samples"] = metrics["total"]["total_count"]
    metadata["realizable_frac"] = metrics["total"]["realizable_fraction"]
    metadata["train_frac"] = round(train_frac, 2)
    metadata["val_frac"] = round(val_frac, 2)
    metadata["test_frac"] = round(test_frac, 2)
    metadata["satisfied_fraction"] = round(metrics["total"]["satisfied_fraction"], 2)
    metadata["match_fraction"] = round(metrics["total"]["match_fraction"], 2)
    metadata["distance_lower_10_fraction"] = round(
        metrics["total"]["distance_lower_10_fraction"], 2
    )
    metadata["distance_lower_50_fraction"] = round(
        metrics["total"]["distance_lower_50_fraction"], 2
    )
    metadata["distance_broken_mean"] = round(metrics["total"]["distance_broken_mean"], 2)
    metadata["distance_broken_std"] = round(metrics["total"]["distance_broken_std"], 2)
    metadata["distance_broken_median"] = round(metrics["total"]["distance_broken_median"], 2)
    metadata["distance_all_mean"] = round(metrics["total"]["distance_all_mean"], 2)
    metadata["distance_all_std"] = round(metrics["total"]["distance_all_std"], 2)
    metadata["distance_all_median"] = round(metrics["total"]["distance_all_median"], 2)
    metadata["parent_dataset"] = parent_dataset
    metadata["eval_beam_size"] = beamsize
    metadata["parent_model"] = parent_model
    metadata["max_match_fraction"] = max_match_fraction
    metadata["filter_distance"] = filter_distance
    metadata["filter_quantile"] = filter_quantile
    metadata["replace_satisfied"] = replace_satisfied
    metadata["replace_minimal_distance"] = replace_minimal
    metadata["include_satisfied_token"] = include_satisfied_token
    metadata["inputs"] = ["i0", "i1", "i2", "i3", "i4"]
    metadata["outputs"] = ["o0", "o1", "o2", "o3", "o4"]
    metadata["eval_alpha"] = alpha
    jsonString = json.dumps(metadata, indent=4)
    jsonFile = open(os.path.join(target_path, "metadata.json"), "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def write_description(
    alpha: float,
    beamsize: int,
    filter_distance: int,
    filter_quantile: float,
    max_match_fraction: float,
    replace_minimal: bool,
    replace_satisfied: bool,
    include_satisfied_token: bool,
    parent_dataset: str,
    parent_model: str,
    target_path: str,
    name: str,
    column_description: List[str],
    metrics: Dict,
) -> None:

    if filter_distance != 0 and filter_quantile != 0.0:
        raise ValueError("Only set filter_quantile or filter_distance")
    with open(os.path.join(target_path, "description.md"), "w") as file:
        file.write("# Description of Dataset " + name + "\n")

        file.write("\n")
        file.writelines(column_description)
        file.write("\n")
        file.write(
            "Generated from dataset "
            + parent_dataset
            + " with model "
            + parent_model
            + ", with alpha "
            + str(alpha)
            + " and beam size "
            + str(beamsize)
            + ".\n"
        )
        if include_satisfied_token:
            file.write(
                "The target circuit starts with a special letter `v` or `s`, standing for violated or satisfied. It denotes whether the `repair_circuit` violates or satisfies the specification. Using the special token, we force the transformer to learn model checking of the `repair_circuit`.\n"
            )

        if filter_distance == 0 and filter_quantile != 0.0:
            filter_distance = metrics["train"]["filter_distance"]
            file.write(
                "The training split of the dataset only contains samples where the `repair_circuit` lies within a levenshtein distance of "
                + str(filter_distance)
                + " to its target. This coresponds to a quantile of "
                + str(filter_quantile)
                + ". All other samples where filtered out. Validation and testing is performed on the unfiltered dataset\n"
            )

        if filter_distance != 0:
            file.write(
                "The training split of the dataset only contains samples where the `repair_circuit` lies within a levenshtein distance of "
                + str(filter_distance)
                + " to its target. All other samples where filtered out. Validation and testing is performed on the unfiltered dataset\n"
            )
        file.write(
            "The fraction of repair_circuits, which match the target in the training set is not higher than "
            + str(max_match_fraction)
            + ".\n"
        )
        if replace_satisfied and not replace_minimal:
            file.write(
                "To reduce the number of misleading targets, we replaced the target circuit of each satisfied sample with its `repair_circuit`. Samples, where the repair circuit does not satisfy the specification stay as is.\n"
            )
        if replace_minimal:
            file.write(
                "To reduce the number of misleading targets, we replaced the target circuit of each satisfied sample with a circuit that satisfies the specification an is as close to the `repair_circuit` as possible. We search in other beams and in evaluations of other beamsizes to find such targets.\n"
            )


def create_from_eval_artifacts(
    alpha: float,
    beamsize: int,
    splits: List[str] = ["test", "val", "train"],
    parent_model: str = "repair-data-2",
    filter_distance: int = 0,
    filter_quantile: float = 0.0,
    column_description: List[str] = None,
    max_match_fraction: float = 1.0,
    include_satisfied_token: Optional[bool] = None,
    replace_satisfied: Optional[bool] = None,
    replace_minimal: Optional[bool] = None,
    overwrite: bool = False,
    upload: bool = False,
    reference_beamsizes: List[int] = None,
    download: bool = False,
) -> None:

    include_satisfied_token = False if include_satisfied_token is None else include_satisfied_token
    replace_satisfied = False if replace_satisfied is None else replace_satisfied
    replace_minimal = False if replace_minimal is None else replace_minimal

    if reference_beamsizes is None:
        reference_beamsizes = [beamsize]

    if replace_minimal and replace_satisfied:
        print("set either replace minimal or replace satisfied")
        return

    if download:
        data = LTLRepairGenData()
        data.load(parent_model, overwrite)

    if column_description is None:
        column_description = [
            "Columns:\n",
            " - status: Whether the repair_circuit is invalid (Invalid/Error), the repair_circuit violates the specification (Violated) or satisfies the specification (Satisfied)\n",
            " - assumptions: Comma separated list of assumptions of the specification\n",
            " - guarantees: Comma separated list of guarantees of the specification\n",
            " - repair_circuit: A possibly broken circuit(see status) which - if broken - is still close to target circuit\n",
            " - inputs: the input variables\n",
            " - outputs: the output variables\n",
            " - realizable: Whether specification is realizable or target circuit (and repair_circuit) is a counter example\n",
            " - circuit: the target circuit, either a circuit that satisfies assumptions -> guarantees(if realizable=1) or a counter example(if realizable=0).",
        ]
    parent_dataset = get_parent_dataset(parent_model=parent_model)
    output_path = get_output_path(
        alpha=alpha,
        beamsize=beamsize,
        filter_distance=filter_distance,
        filter_quantile=filter_quantile,
        max_match_fraction=max_match_fraction,
        replace_minimal=replace_minimal,
        replace_satisfied=replace_satisfied,
        include_satisfied_token=include_satisfied_token,
        parent_dataset=parent_dataset,
    )
    name = name_from_arguments(
        alpha=alpha,
        beamsize=beamsize,
        filter_distance=filter_distance,
        filter_quantile=filter_quantile,
        max_match_fraction=max_match_fraction,
        replace_minimal=replace_minimal,
        replace_satisfied=replace_satisfied,
        include_satisfied_token=include_satisfied_token,
        parent_dataset=parent_dataset,
    )
    print(output_path)
    if overwrite and os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    metrics_total = {}
    counts_total = {}
    dataframes = []
    for split in splits:
        dataframe, metrics_split, counts_split = create_split(
            split=split,
            alpha=alpha,
            beamsize=beamsize,
            reference_beamsizes=reference_beamsizes,
            filter_distance=filter_distance,
            filter_quantile=filter_quantile,
            max_match_fraction=max_match_fraction,
            replace_minimal=replace_minimal,
            replace_satisfied=replace_satisfied,
            include_satisfied_token=include_satisfied_token,
            parent_model=parent_model,
            target_path=output_path,
        )
        dataframes.append(dataframe)
        metrics_total[split] = metrics_split
        counts_total[split] = counts_split
    dataframe_total = pandas.concat(dataframes, ignore_index=True)
    metrics_total["total"] = metrics(dataframe_total)
    counts_total["total"] = counts(dataframe_total)
    save_plots_to_file(dataframe_total, output_path, "total")
    write_counts(count_dict=counts_total, target_path=output_path)
    write_metadata(
        metrics=metrics_total,
        target_path=output_path,
        alpha=alpha,
        beamsize=beamsize,
        filter_distance=filter_distance,
        filter_quantile=filter_quantile,
        max_match_fraction=max_match_fraction,
        replace_minimal=replace_minimal,
        replace_satisfied=replace_satisfied,
        include_satisfied_token=include_satisfied_token,
        parent_dataset=parent_dataset,
        parent_model=parent_model,
    )
    write_description(
        metrics=metrics_total,
        alpha=alpha,
        beamsize=beamsize,
        filter_distance=filter_distance,
        filter_quantile=filter_quantile,
        max_match_fraction=max_match_fraction,
        replace_minimal=replace_minimal,
        replace_satisfied=replace_satisfied,
        include_satisfied_token=include_satisfied_token,
        parent_dataset=parent_dataset,
        parent_model=parent_model,
        target_path=output_path,
        name=name,
        column_description=column_description,
    )
    if upload:
        data_class = LTLRepairSplitData()
        data_class.upload(name, overwrite)


def create_from_eval_artifacts_multiple(
    parent_model: str = "repair-data-2",
    alphas: List[float] = [0.5],
    beamsizes: List[int] = [1, 3],
    filter_quantile: List[float] = [1.0],
    column_description: Dict[str, str] = None,
    max_match_fraction: List[float] = [0.0, 0.1, 0.2, 1.0],
    include_satisfied_token: Optional[List[bool]] = None,
    replace_satisfied: Optional[List[bool]] = None,
    replace_minimal: Optional[List[bool]] = None,
    overwrite: bool = False,
    upload: bool = False,
    reference_beamsizes: List[int] = [1, 2, 3, 4],
    cores: int = 4,
    download: bool = False,
) -> None:
    include_satisfied_token = (
        [False] if include_satisfied_token is None else include_satisfied_token
    )
    replace_satisfied = [False] if replace_satisfied is None else replace_satisfied
    replace_minimal = [False] if replace_minimal is None else replace_minimal

    def check_argument(argument: List[bool]) -> None:
        if len(argument) > 2 or (argument == [True, True]) or argument == [False, False]:
            raise ValueError(
                "Argument value of " + f"{argument=}".partition("=")[0] + " not allowed."
            )

    check_argument(include_satisfied_token)
    check_argument(replace_satisfied)
    check_argument(replace_minimal)

    if download:
        data = LTLRepairGenData()
        data.load(parent_model, overwrite)

    commands = []
    for alpha in alphas:
        for beamsize in beamsizes:
            for filter_quantile_ in filter_quantile:
                for include_satisfied_token_ in include_satisfied_token:
                    for replace_satisfied_ in replace_satisfied:
                        for replace_minimal_ in replace_minimal:
                            for max_match_fraction_ in max_match_fraction:
                                if not replace_satisfied_ or not replace_minimal_:
                                    commands.append(
                                        "python ml2/ltl/ltl_repair/ltl_repair_data_creation.py --alpha "
                                        + str(alpha)
                                        + " --beamsize "
                                        + str(beamsize)
                                        + " --filter-quantile "
                                        + str(filter_quantile_)
                                        + (
                                            " --replace-satisfied "
                                            if replace_satisfied_
                                            else " --no-replace-satisfied "
                                        )
                                        + (
                                            " --replace-minimal "
                                            if replace_minimal_
                                            else " --no-replace-minimal "
                                        )
                                        + (
                                            " --include-satisfied-token "
                                            if include_satisfied_token_
                                            else " --no-include-satisfied-token "
                                        )
                                        + (" -o " if overwrite else "")
                                        + (" -u " if upload else "")
                                        + " --reference-beamsizes "
                                        + " ".join(map(lambda x: str(x), reference_beamsizes))
                                        + " -p "
                                        + str(parent_model)
                                        + " --max-match-fraction "
                                        + str(max_match_fraction_)
                                    )
    with open("commands.txt", "w") as file:
        for command in commands:
            file.writelines(command + "\n")
    os.system("parallel -j " + str(cores) + " < commands.txt")


# create_from_eval_artifacts_multiple(
#     alphas=[0.5],
#     parent_model="repair-data-2",
#     beamsizes=[1, 2, 3],
#     filter_distance=[None, 10, 50],
#     replace_satisfied=[True, False],
#     replace_minimal=[True, False],
#     overwrite=True,
#     upload=True,
#     reference_beamsizes=[1, 2, 3],
# )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-u", "--upload", action="store_true")
    parser.add_argument("-d", "--download", action="store_true")
    parser.add_argument("--reference-beamsizes", nargs="+", type=int)
    parser.add_argument("-p", "--parent", type=str, default="repair-data-2")

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--alpha-m", nargs="+", type=float)
    parser.add_argument("--beamsize-m", nargs="+", type=int)
    parser.add_argument("--filter-quantile-m", nargs="+", type=float)
    parser.add_argument("--max-match-fraction-m", nargs="+", type=float)
    parser.add_argument("--replace-satisfied-m-all", action="store_true")
    parser.add_argument("--replace-satisfied-m-some", action="store_true")
    parser.add_argument("--replace-satisfied-m-none", action="store_true")
    parser.add_argument("--replace-minimal-m-all", action="store_true")
    parser.add_argument("--replace-minimal-m-some", action="store_true")
    parser.add_argument("--replace-minimal-m-none", action="store_true")
    parser.add_argument("--include-satisfied-token-m-all", action="store_true")
    parser.add_argument("--include-satisfied-token-m-some", action="store_true")
    parser.add_argument("--include-satisfied-token-m-none", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beamsize", type=int, default=3)
    parser.add_argument("--filter-quantile", type=float, default=0.0)
    parser.add_argument("--max-match-fraction", type=float, default=1.0)
    parser.add_argument("--replace-satisfied", action="store_true")
    parser.add_argument("--replace-minimal", action="store_true")
    parser.add_argument("--include-satisfied-token", action="store_true")
    parser.add_argument("--no-replace-satisfied", action="store_true")
    parser.add_argument("--no-replace-minimal", action="store_true")
    parser.add_argument("--no-include-satisfied-token", action="store_true")

    args = parser.parse_args()
    print(args)

    if (
        (args.include_satisfied_token and args.no_include_satisfied_token)
        or (args.replace_satisfied and args.no_replace_satisfied)
        or (args.replace_minimal and args.no_replace_minimal)
    ):
        raise argparse.ArgumentError(
            message="Can't activate and deactivate arguments at the same time", argument=None
        )

    if args.all:
        create_from_eval_artifacts_multiple(
            alphas=args.alpha_m,
            beamsizes=args.beamsize_m,
            filter_quantile=args.filter_quantile_m,
            max_match_fraction=args.max_match_fraction_m,
            replace_satisfied=[True, False]
            if args.replace_satisfied_m_some
            else (
                [True]
                if args.replace_satisfied_m_all
                else ([False] if args.replace_satisfied_m_none else None)
            ),
            replace_minimal=[True, False]
            if args.replace_minimal_m_some
            else (
                [True]
                if args.replace_minimal_m_all
                else ([False] if args.replace_minimal_m_none else None)
            ),
            include_satisfied_token=[True, False]
            if args.include_satisfied_token_m_some
            else (
                [True]
                if args.include_satisfied_token_m_all
                else ([False] if args.include_satisfied_token_m_none else None)
            ),
            parent_model=args.parent,
            overwrite=args.overwrite,
            upload=args.upload,
            cores=args.cores,
            download=args.download,
        )
    else:
        create_from_eval_artifacts(
            alpha=args.alpha,
            beamsize=args.beamsize,
            parent_model=args.parent,
            filter_quantile=args.filter_quantile,
            max_match_fraction=args.max_match_fraction,
            include_satisfied_token=True
            if args.include_satisfied_token
            else (False if args.no_include_satisfied_token else None),
            replace_satisfied=True
            if args.replace_satisfied
            else (False if args.no_replace_satisfied else None),
            replace_minimal=True
            if args.replace_minimal
            else (False if args.no_replace_minimal else None),
            overwrite=args.overwrite,
            upload=args.upload,
            reference_beamsizes=args.reference_beamsizes,
        )


if __name__ == "__main__":
    main()
