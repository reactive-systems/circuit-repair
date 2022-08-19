"""Represents a dataset for training LTL Repair"""

import argparse
from copy import deepcopy
import csv
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from Levenshtein import distance as lev
from tqdm import tqdm


import re
import tensorflow as tf


import pandas as pd

from ... import aiger
from .ltl_repair_data_gen import LTLRepairGenData
from ...data import SupervisedData, SplitSupervisedData
from ..ltl_syn.ltl_syn_data import LTLSynData
from ...data.utils import from_csv_str, to_csv_str
from ...data.stats import stats_from_counts
import plotly.express as px
from plotly.graph_objects import Figure
from ...globals import (
    LTL_REP_ALIASES,
    LTL_REP_BUCKET_DIR,
    LTL_REP_WANDB_PROJECT,
)
from ..ltl_spec import LTLSpec
from ...tools.nuxmv import nuXmv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# reapair circuit statistics keys
I_MAX_VAR_INDEX: str = "MAX VARIABLE INDEX"
I_NUM_INPUTS: str = "NUM INPUTS"
I_NUM_LATCHES: str = "NUM LATCHES"
I_NUM_OUTPUTS: str = "NUM OUTPUTS"
I_NUM_AND_GATES: str = "NUM AND GATES"
# circuit statistics keys
T_MAX_VAR_INDEX: str = "MAX VARIABLE INDEX"
T_NUM_INPUTS: str = "NUM INPUTS"
T_NUM_LATCHES: str = "NUM LATCHES"
T_NUM_OUTPUTS: str = "NUM OUTPUTS"
T_NUM_AND_GATES: str = "NUM AND GATES"


def sample_to_csv_row(sample: Dict) -> List:
    status = to_csv_str(sample["status"])
    assumptions = ",".join(sample["assumptions"])
    guarantees = ",".join(sample["guarantees"])
    repair_circuit = to_csv_str(sample["repair_circuit"])
    inputs = ",".join(sample["inputs"])
    outputs = ",".join(sample["outputs"])
    realizable = sample["realizable"]
    circuit = to_csv_str(sample["circuit"])
    return [status, assumptions, guarantees, repair_circuit, inputs, outputs, realizable, circuit]


def csv_row_to_sample(row: List) -> Dict:
    return {
        "status": from_csv_str(row[0]),
        "assumptions": row[1].split(","),
        "guarantees": row[2].split(","),
        "repair_circuit": from_csv_str(row[3]),
        "inputs": row[4].split(","),
        "outputs": row[5].split(","),
        "realizable": int(row[6]),
        "circuit": from_csv_str(row[7]),
        "hash": from_csv_str(row[8]),
    }


class LTLRepairData(SupervisedData):

    ALIASES = LTL_REP_ALIASES
    BUCKET_DIR = LTL_REP_BUCKET_DIR
    WANDB_PROJECT = LTL_REP_WANDB_PROJECT

    def sample_generator(self):
        for _, row in self.data_frame.iterrows():
            sample = {
                "status": row["status"],
                "assumptions": row["assumptions"].split(",")
                if "assumptions" in row and row["assumptions"]
                else [],  # key in dict check for data that does not contain assumptions
                "guarantees": row["guarantees"].split(",") if row["guarantees"] else [],
                "repair_circuit": row["repair_circuit"],
                "inputs": row["inputs"].split(",") if row["inputs"] else [],
                "outputs": row["outputs"].split(",") if row["outputs"] else [],
                "realizable": row["realizable"],
                "target_circuit": row["circuit"],
                "hash": row["hash"] if "hash" in row else "",
            }
            yield sample

    def generator(self):
        for sample in self.sample_generator():
            yield LTLSpec.from_dict(sample), sample["repair_circuit"], sample["hash"], sample[
                "target_circuit"
            ]

    def input_generator(self):
        for inp, _, _, _ in self.generator():
            yield inp

    def target_generator(self):
        for _, _, _, tar in self.generator():
            yield tar

    def circuit_generator(self):
        for _, circ, _, _ in self.generator():
            yield circ

    def tf_generator(self):
        for inp, circ, _hash, tar in self.generator():
            if not self.input_encoder.encode(inp):
                error = self.input_encoder.error
                self.input_encoder_errors[error] = self.input_encoder_errors.get(error, 0) + 1
                continue
            input_tensor = self.input_encoder.tensor
            if not self.circuit_encoder.encode(circ):
                error = self.circuit_encoder.error
                self.circuit_encoder_errors[error] = self.circuit_encoder_errors.get(error, 0) + 1
                continue
            circuit_tensor = self.circuit_encoder.tensor
            if not self.target_encoder.encode(tar):
                error = self.target_encoder.error
                self.target_encoder_errors[error] = self.target_encoder_errors.get(error, 0) + 1
                continue
            target_tensor = self.target_encoder.tensor
            yield input_tensor, circuit_tensor, target_tensor

    def tf_dataset(self, input_encoder, circuit_encoder, target_encoder):
        self.input_encoder = input_encoder
        self.target_encoder = target_encoder
        self.circuit_encoder = circuit_encoder
        self.input_encoder_errors = {}
        self.target_encoder_errors = {}
        self.circuit_encoder_errors = {}
        output_signature = (
            input_encoder.tensor_spec,
            circuit_encoder.tensor_spec,
            target_encoder.tensor_spec,
        )
        return tf.data.Dataset.from_generator(self.tf_generator, output_signature=output_signature)

    def save_to_path(self, path: str) -> None:
        path = path + ".csv"
        circuit_series = self.data_frame["circuit"]
        repair_circuit_series = self.data_frame["repair_circuit"]
        self.data_frame["circuit"] = circuit_series.str.replace("\n", "\\n")
        self.data_frame["repair_circuit"] = repair_circuit_series.str.replace("\n", "\\n")
        self.data_frame.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
        self.data_frame["circuit"] = circuit_series
        self.data_frame["repair_circuit"] = repair_circuit_series

    @classmethod
    def load_from_path(cls, path: str, keep_timeouts: bool = False):
        """Load a dataset from csv file

        Args:
            path (str): The path of the dataset csv file
            keep_timeouts (bool, optional): Whether to remove timeouts when loading the dataset. Defaults to False.

        Returns:
            LTLRepairData: An object of this class with the dataset from the file.
        """
        logger.info("Load data from file:" + path)
        data_frame = pd.read_csv(
            path,
            converters={
                "circuit": lambda c: str(c).replace("\\n", "\n"),
                "repair_circuit": lambda c: str(c).replace("\\n", "\n"),
            },
            dtype={
                "status": str,
                "assumptions": str,
                "guarantees": str,
                "inputs": str,
                "outputs": str,
                "realizable": int,
                "hash": str,
            },
            keep_default_na=False,
        )
        if not keep_timeouts:
            data_frame = data_frame[data_frame["status"] != "Timeouts"]
        logger.info(
            "======================SAMPLES READ: "
            + str(len(data_frame))
            + " ======================"
        )
        return cls(data_frame)

    def find_minimal_alternative_target(
        self, dataset_reference: Optional["LTLRepairData"] = None, calc_distance_only: bool = False
    ):
        """Finds the minimal alternative target. Searches in dataset_reference for beams which have a the same specification but a different correct solution. Thereby this is an alternative target. If the alternative target has a smaller edit distance to the misprediction, the alternative target is chosen as new target. Adds the column levenshtein_distance containing the edit distance between prediction and target.

        Args:
            dataset_reference (LTLRepairData, optional): A dataset containing correct solutions from multiple beams. If argument not present, we use self for the search of alternative targets
            calc_distance_only (bool, optional):  If this is set, we don't update the target but the levenshtein distance between the prediction and an alternative minimal target. Defaults to False.
        """
        if "levenshtein_distance" in self.data_frame.columns:
            self.data_frame.drop("levenshtein_distance", axis=1, inplace=True)

        def find_minimal_distance(
            row: pd.DataFrame, dataframe_reference: pd.DataFrame
        ) -> Tuple[str, int]:
            """Calculates the minimal distance between a sample in row and the reference dataset.

            Args:
                row (pd.DataFrame): The row of the sample we try to find an alternative target
                dataframe_reference (pd.DataFrame): The reference dataset, in which we search for alternative targets

            Returns:
                Tuple[str, int]: the alternative target and levenshtein distance between prediction and alternative target.
            """
            if "hash" in row.index and "hash" in dataframe_reference.columns:
                match = dataframe_reference["hash"] == row["hash"]
            else:
                guarantees_match = dataframe_reference["guarantees"] == row["guarantees"]
                assumptions_match = dataframe_reference["assumptions"] == row["assumptions"]
                match = assumptions_match & guarantees_match
            same_spec = dataframe_reference[match]
            lv = same_spec.apply(
                lambda x: [x["circuit"], lev(x["circuit"], row["repair_circuit"])],
                axis=1,
                result_type="expand",
            ).reset_index()
            if len(lv) == 0 or (lv.iloc[lv[1].idxmin()][1] > row["levenshtein_distance"]):
                return row["circuit"], row["levenshtein_distance"]
            else:
                return lv.iloc[lv[1].idxmin()][0], lv.iloc[lv[1].idxmin()][1]

        def apply_fn(row: pd.DataFrame, dataframe_reference: pd.DataFrame) -> pd.DataFrame:
            if row["status"] == "Match":
                row["levenshtein_distance"] = 0
                return row
            circuit, lv = find_minimal_distance(row, dataframe_reference)
            if not calc_distance_only:
                row["circuit"] = circuit
            row["levenshtein_distance"] = lv
            return row

        self.treat_satisfied_as_match(calc_distance_only=calc_distance_only)
        dataframe_reference = (
            dataset_reference.data_frame if dataset_reference else self.data_frame
        )

        tqdm.pandas(desc="replacing misleading target with minimal distance target")
        self.data_frame = self.data_frame.progress_apply(
            lambda x: apply_fn(row=x, dataframe_reference=dataframe_reference),
            axis=1,
            result_type="expand",
        )

    def treat_satisfied_as_match(self, calc_distance_only: bool = False):
        """Transform samples which are satisfied into samples which are a match, by replacing the target circuit with the prediction. Adds the column levenshtein_distance containing the edit distance between prediction and target.

        Args:
            calc_distance_only (bool, optional): If this is set, we don't update the target but the levenshtein distance between the prediction and an alternative minimal target. Defaults to False.
        """
        if "levenshtein_distance" in self.data_frame.columns:
            self.data_frame.drop("levenshtein_distance", axis=1, inplace=True)

        def apply_fn(row: pd.Series) -> pd.Series:
            if not calc_distance_only:
                row["circuit"] = (
                    row["repair_circuit"] if row["status"] == "Satisfied" else row["circuit"]
                )
                row["status"] = "Match" if row["status"] == "Satisfied" else row["status"]

            app = pd.Series(
                [
                    0
                    if row["status"] == "Satisfied"
                    else lev(row["repair_circuit"], row["circuit"])
                ],
                index=["levenshtein_distance"],
            )
            return pd.concat([row, app])

        tqdm.pandas(desc="treat satisfied as match")
        self.data_frame = self.data_frame.progress_apply(
            apply_fn,
            axis=1,
            result_type="expand",
        )

    def add_levenshtein(
        self,
        consider_satisfied: bool = False,
        consider_minimal: bool = False,
        dataset_reference: "LTLRepairData" = None,
    ):
        """Adds Levenshtein Distance column to the dataset. The distance is either calculated on
            - the difference between prediction and its target (consider_satisfied and consider_minimal is False)
            - the difference between prediction and its target except when the sample is Satisfied, then 0 (consider_satisfied is True)
            - the difference between prediction ans a minimal target, see find_minimal_alternative_target. (consider_minimal is True)

        Args:
            consider_satisfied (bool, optional): Defaults to False.
            consider_minimal (bool, optional): Defaults to False.
            dataset_reference (LTLRepairData, optional): The dataset reference passed to find_minimal_alternative_target. Defaults to None.
        """
        if "levenshtein_distance" in self.data_frame.columns:
            self.data_frame.drop("levenshtein_distance", axis=1, inplace=True)
        if consider_satisfied:
            self.treat_satisfied_as_match(calc_distance_only=True)
        elif consider_minimal:
            self.find_minimal_alternative_target(
                calc_distance_only=True, dataset_reference=dataset_reference
            )
        else:
            self.data_frame = self.calculate_levenshtein(self.data_frame, set_match=True)

    def filter_matches(self, max_match_fraction: float, alter: bool = False, **params):
        """Removes samples which are matches from the dataset until only max_match_fraction matching samples are left. If alter is set, instead of removing them alter the circuit. The amount of samples removed or altered is calculated based on the percentage of matches, which should be in this dataset after the function is applied. Hence when choosing alter, effectively less matches are altered to match the max_match_fraction, compared to the amount of samples removed, if alter is not set, as the total amount of samples changes.

        Args:
            max_match_fraction (float): How many matching samples should be in the datset after applying this function
            alter (bool, optional): If this is set, don't remove samples, instead, alter the circuits. Defaults to False.
            params: Additional optional arguments for circuit altering

        Raises:
            ValueError: If the dataset would be empty after applying this function
        """

        match = self.data_frame[
            (self.data_frame["status"] == "Match")
            | (self.data_frame["repair_circuit"] == self.data_frame["circuit"])
        ]
        if (len(self.data_frame) == len(match)) and max_match_fraction != 1.0 and not alter:
            raise ValueError(
                "tried filtering with match fraction but dataset contains only matches"
            )
        new_fraction = (
            min(
                (
                    (
                        ((len(self.data_frame) - len(match)) / (1 - max_match_fraction))
                        - (len(self.data_frame) - len(match))
                    )
                    if not alter
                    else ((len(self.data_frame) * max_match_fraction))
                )
                / len(match),
                1,
            )
            if max_match_fraction < 1
            else 1
        )
        new_matches = match.sample(frac=new_fraction)
        remainder = pd.DataFrame()
        if alter:
            remainder = match.copy()
            remainder.drop(new_matches.index, inplace=True)
            remainder["repair_circuit"] = self.alter_circuit_batch(remainder["circuit"], **params)
            remainder["status"] = "Changed"

        self.data_frame = self.data_frame[
            (self.data_frame["status"] != "Match")
            & (self.data_frame["repair_circuit"] != self.data_frame["circuit"])
        ]
        self.data_frame = self.data_frame.append([new_matches, remainder]).reset_index()
        self.data_frame.drop("index", axis=1, inplace=True)

    def filter_distance_bigger(self, max_distance: int, alter: bool = False, **params):
        """Removes samples with a higher edit distance than the given distance. If alter is set, instead of removing them alter the circuit.

        Args:
            distance (int):The maximal edit distance which should be included in the dataset
            alter (bool, optional): If this is set, don't remove samples, instead, alter the circuits. Defaults to False.
            params: Additional optional arguments for circuit altering

        """
        self.add_levenshtein()
        remainder = pd.DataFrame()
        if alter:
            remainder = self.data_frame[
                self.data_frame["levenshtein_distance"] > max_distance
            ].copy()
            remainder["repair_circuit"] = self.alter_circuit_batch(remainder["circuit"], **params)
            remainder["status"] = "Changed"
        self.data_frame = self.data_frame[self.data_frame["levenshtein_distance"] <= max_distance]
        self.data_frame = self.data_frame.append(remainder).reset_index()

    @staticmethod
    def calculate_levenshtein(
        df: pd.DataFrame,
        over: List[Dict] = [
            {
                "between_1": "repair_circuit",
                "between_2": "circuit",
                "res": "levenshtein_distance",
            }
        ],
        set_match: bool = True,
    ) -> pd.DataFrame:
        """Calculates the Levenshtein distance over multiple pairs of columns in a given dataframe.

        Args:
            df (pd.DataFrame): dataframe on which to calculate the distance
            over ( str , optional): _description_. Defaults to [ { "between_1": "repair_circuit", "between_2": "circuit", "res": "levenshtein_distance", } ]. Defines triples of the columns the distance is to be calculated over, including column containing the result.
            set_match (bool, optional): _description_. Defaults to True. Whether to update the "status" column according to the distance.

        Returns:
            pd.DataFrame: An updated dataframe
        """

        def apply(x: pd.DataFrame, i: int, cols: List[str]) -> pd.DataFrame:
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

        df = df.copy()
        for i in range(len(over)):
            cols = df.columns
            if over[i]["between_1"] not in cols or over[i]["between_2"] not in cols:
                raise ValueError("Arguments passed to calculate_levenshtein not consistent")
            df = df.apply(
                lambda x: apply(x, i, cols),
                axis=1,
                result_type="expand",
            )
            cols = cols.append(pd.Index([over[i]["res"]]))
            df.columns = cols
        return df

    def keep_one_beam(self, keep: str, dataset_reference: Optional["LTLRepairData"] = None):
        """Only keep one beam from a dataset that contains multiple beams. Basically removes duplicate samples (assumption + guarantees are equal). If keep is set to best, the best sample is determined based on minimal alternative targets.

        Args:
            keep (str): Either random or best.
            dataset_reference (LTLRepairData, optional): The dataset reference passed to find_minimal_alternative_target. Defaults to None.
        """

        def minimal_distance(group: pd.DataFrame) -> pd.DataFrame:
            group_ = group.reset_index()
            group_["levenshtein_distance"] = group_["levenshtein_distance"].astype(int)
            return group_.loc[group_["levenshtein_distance"].idxmin()]

        old_cols = self.data_frame.columns
        groupby = ["guarantees", "assumptions"]
        if "hash" in old_cols:
            groupby = ["hash"]
        if keep == "random":
            self.data_frame = (
                self.data_frame.groupby(groupby, group_keys=False)
                .apply(lambda x: x.sample())
                .reset_index()
                .drop("index", axis=1)
            )
        elif keep == "best":
            self.add_levenshtein(consider_minimal=True, dataset_reference=dataset_reference)
            self.data_frame = (
                self.data_frame.groupby(groupby, group_keys=False)
                .apply(minimal_distance)
                .drop("levenshtein_distance", axis=1)
            )
            self.data_frame.drop(groupby + ["index"], axis=1).reset_index().reindex(
                columns=old_cols
            )
        elif keep != "all":
            raise ValueError

    @classmethod
    def from_LTLSynData(cls, path_from: str):

        """Uses a LTLSynData dataset split to create a LTLRepairData dataset split. Introduces the necessary columns and creates the repair circuit by copying the target circuit.

        Args:
            path_from (str): The path from which to load the LTLSynData dataset split.
        """

        df = LTLSynData.load_from_path(path_from).data_frame
        df.insert(0, "status", "Changed", True)
        df.insert(3, "repair_circuit", df["circuit"], True)
        return cls(df)

    def alter_circuit(self, **params):

        """Modifies (i.e. breaks) the repair circuit by according to parameters.

        Args:
            params : Pass parameters to Circuit.alter()
        """

        self.data_frame["repair_circuit"] = self.alter_circuit_batch(
            self.data_frame["repair_circuit"], **params
        )

    @staticmethod
    def alter_circuit_batch(circuits_str: List[str], **params) -> List[str]:
        """Batch altering of a list of circuits. Uses Circuit.alter()

        Args:
            circuits_str (List[str]): The input list of circuits
            params: optional Arguments forwarded to alter.

        Returns:
            List[str]: The altered list of circuits
        """
        new_circuits = []
        for circuit_str in tqdm(circuits_str, desc="alter circuits"):
            circuit = aiger.parse(circuit_str)
            circuit.alter(**params)
            new_circuits.append(str(circuit))
        return new_circuits

    def stats(self) -> Dict:
        logger.info("Calculate statistics")
        self.add_levenshtein()
        metrics: Dict[str, Any] = {}
        metrics["num_samples"] = len(self.data_frame)
        metrics["realizable_fraction"] = round(
            len(self.data_frame[self.data_frame["realizable"] == 1]) / len(self.data_frame), 2
        )
        metrics["satisfied_fraction"] = round(
            len(self.data_frame[self.data_frame["status"] == "Satisfied"]) / len(self.data_frame),
            2,
        )
        metrics["match_fraction"] = round(
            len(self.data_frame[self.data_frame["status"] == "Match"]) / len(self.data_frame), 2
        )
        metrics["violated_fraction"] = round(
            len(self.data_frame[self.data_frame["status"] == "Violated"]) / len(self.data_frame), 2
        )
        metrics["changed_fraction"] = round(
            len(self.data_frame[self.data_frame["status"] == "Changed"]) / len(self.data_frame), 2
        )
        metrics["distance_lower_10_fraction"] = round(
            len(self.data_frame[self.data_frame["levenshtein_distance"] < 10])
            / len(self.data_frame),
            2,
        )
        metrics["distance_lower_50_fraction"] = round(
            len(self.data_frame[self.data_frame["levenshtein_distance"] < 50])
            / len(self.data_frame),
            2,
        )
        metrics["distance_all_mean"] = round(self.data_frame["levenshtein_distance"].mean(), 2)
        metrics["distance_all_std"] = round(self.data_frame["levenshtein_distance"].std(), 2)
        metrics["distance_all_median"] = round(self.data_frame["levenshtein_distance"].median(), 2)
        metrics["distance_broken_mean"] = round(
            self.data_frame[self.data_frame["levenshtein_distance"] != 0][
                "levenshtein_distance"
            ].mean(),
            2,
        )
        metrics["distance_broken_std"] = round(
            self.data_frame[self.data_frame["levenshtein_distance"] != 0][
                "levenshtein_distance"
            ].std(),
            2,
        )
        metrics["distance_broken_median"] = round(
            self.data_frame[self.data_frame["levenshtein_distance"] != 0][
                "levenshtein_distance"
            ].median(),
            2,
        )
        return metrics

    def plot_lev(
        self,
        bins: Optional[int] = None,
        range: Optional[List[float]] = None,
        log: bool = False,
        x_label: Optional[str] = None,
        filter_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        dataset_name: str = "None",
    ) -> Figure:
        self.add_levenshtein()
        frame = self.data_frame.copy()
        if filter_range is not None:
            if filter_range[0] is not None:
                frame = frame[frame["levenshtein_distance"] >= filter_range[0]]
            if filter_range[1] is not None:
                frame = frame[frame["levenshtein_distance"] <= filter_range[1]]
        return LTLRepairSplitData.histogram_lev(
            dataframes={dataset_name: frame},
            group_label="dataset",
            x_label=x_label,
            bins=bins,
            range=range,
            log=log,
            legend=False,
        )


class LTLRepairSplitData(SplitSupervisedData):

    ALIASES = LTL_REP_ALIASES
    BUCKET_DIR = LTL_REP_BUCKET_DIR
    WANDB_PROJECT = LTL_REP_WANDB_PROJECT

    def stats(self, model_check: bool = True, sample_size: int = 4000) -> Dict:
        metrics: Dict[str, Any] = {}
        total_basic_splits = 0
        for split in self.split_names:
            metrics[split] = self[split].stats()
            total_basic_splits = metrics[split]["num_samples"] + total_basic_splits
        for split in ["val", "train", "test"]:
            if split in self.split_names:
                metrics[split + "_fraction"] = metrics[split]["num_samples"] / total_basic_splits
        metrics["splits"] = self.split_names
        if model_check:
            for split, v in self.model_check_repair_circuit(
                only_altered=True, samples=sample_size
            ).items():
                metrics[split]["satisfied_in_changed_fraction"] = v
                metrics[split]["satisfied_in_changed_sample_size"] = sample_size
        return metrics

    @staticmethod
    def histogram_lev(
        dataframes: Dict[str, pd.DataFrame],
        group_label: str,
        x_label: Optional[str] = None,
        bins: int = None,
        range: Optional[List[float]] = None,
        log: bool = True,
        legend: bool = True,
    ) -> Figure:
        x_label = (
            x_label
            if x_label
            else "Levenshtein distance between repair circuit and target circuit"
        )
        combined = []
        for filter, dataframe in dataframes.items():
            dataframe_ = dataframe
            dataframe_["mod"] = filter
            combined.append(dataframe_)
        dataframe_combined = pd.concat(combined, axis=0)
        fig = px.histogram(
            dataframe_combined,
            x="levenshtein_distance",
            marginal="box",  # or violin, rug
            barmode="group",
            color="mod",
            nbins=bins,
            log_y=log,
            range_x=range,
            # histnorm="percent",
            histfunc="count",
            title=None,
            labels={
                "mod": group_label,
                "levenshtein_distance": x_label,
            },
            width=619,
            height=300,
            color_discrete_sequence=px.colors.qualitative.G10,
        )
        fig.update_layout(
            margin=dict(l=0, r=10, t=10, b=0),
            showlegend=legend,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    @staticmethod
    def plot_hist_compare(
        compare: List[Tuple[str, str, Optional[Tuple[Optional[int], Optional[int]]]]],
        bins: int = 300,
        range: Optional[List[float]] = None,
        log: bool = True,
        legend: bool = True,
    ) -> Figure:
        dataframes = {}
        for name, split, filter_range in compare:
            ds = LTLRepairSplitData.load(name, overwrite=False)
            frame = ds[split].data_frame.copy()
            if filter_range is not None:
                if filter_range[0] is not None:
                    frame = frame[frame["levenshtein_distance"] >= filter_range[0]]
                if filter_range[1] is not None:
                    frame = frame[frame["levenshtein_distance"] <= filter_range[1]]
            dataframes[name + "(" + split + ")"] = frame
        return LTLRepairSplitData.histogram_lev(
            dataframes=dataframes,
            group_label="dataset",
            bins=bins,
            range=range,
            log=log,
            legend=legend,
        )

    # def stats(self, splits: List[str] = None) -> Dict:
    #     counts: Dict[str, List[int]] = {
    #         T_MAX_VAR_INDEX: [],
    #         T_NUM_INPUTS: [],
    #         T_NUM_LATCHES: [],
    #         T_NUM_OUTPUTS: [],
    #         T_NUM_AND_GATES: [],
    #         I_MAX_VAR_INDEX: [],
    #         I_NUM_INPUTS: [],
    #         I_NUM_LATCHES: [],
    #         I_NUM_OUTPUTS: [],
    #         I_NUM_AND_GATES: [],
    #     }
    #     for _, repair_circuit, circuit in self.generator(splits):
    #         (
    #             num_var_index,
    #             num_inputs,
    #             num_latches,
    #             num_outputs,
    #             num_and_gates,
    #         ) = aiger.header_ints_from_str(circuit)
    #         counts[T_MAX_VAR_INDEX].append(num_var_index)
    #         counts[T_NUM_INPUTS].append(num_inputs)
    #         counts[T_NUM_LATCHES].append(num_latches)
    #         counts[T_NUM_OUTPUTS].append(num_outputs)
    #         counts[T_NUM_AND_GATES].append(num_and_gates)
    #         (
    #             num_var_index,
    #             num_inputs,
    #             num_latches,
    #             num_outputs,
    #             num_and_gates,
    #         ) = aiger.header_ints_from_str(repair_circuit)
    #         counts[I_MAX_VAR_INDEX].append(num_var_index)
    #         counts[I_NUM_INPUTS].append(num_inputs)
    #         counts[I_NUM_LATCHES].append(num_latches)
    #         counts[I_NUM_OUTPUTS].append(num_outputs)
    #         counts[I_NUM_AND_GATES].append(num_and_gates)
    #     return stats_from_counts(counts)

    def plot_stats(self, splits: List[str] = None) -> None:
        raise NotImplementedError
        stats = self.stats(splits)

        def plot_stats(stats: dict, filepath: str, title: str = None):
            file_dir = os.path.dirname(filepath)
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
            fig, ax = plt.subplots()
            max_value = stats["max"]
            min_value = stats["min"]
            bins = stats["bins"]
            ax.bar(range(max_value + 1), bins, color="#3071ff", width=0.7, align="center")
            if title:
                ax.set_title(title)
            ax.set_xlim(min_value - 1, max_value + 1)
            ax.set_ylim(0, max(bins) + 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig(filepath, dpi=fig.dpi, facecolor="white", format="eps")
            if title:
                logging.info("%s statistics plotted to %s", title, filepath)

        filepath = os.path.join(self.stats_path(self.name), "t_max_var_id.eps")
        plot_stats(stats[T_MAX_VAR_INDEX], filepath, "Maximal Variable Index")
        filepath = os.path.join(self.stats_path(self.name), "t_num_inputs.eps")
        plot_stats(stats[T_NUM_INPUTS], filepath, "Number of Inputs")
        filepath = os.path.join(self.stats_path(self.name), "t_max_num_latches.eps")
        plot_stats(stats[T_NUM_LATCHES], filepath, "Number of Latches")
        filepath = os.path.join(self.stats_path(self.name), "t_num_outputs.eps")
        plot_stats(stats[T_NUM_OUTPUTS], filepath, "Number of Outputs")
        filepath = os.path.join(self.stats_path(self.name), "t_num_and_gates.eps")
        plot_stats(stats[T_NUM_AND_GATES], filepath, "Number of AND Gates")
        filepath = os.path.join(self.stats_path(self.name), "i_max_var_id.eps")
        plot_stats(stats[I_MAX_VAR_INDEX], filepath, "Maximal Variable Index")
        filepath = os.path.join(self.stats_path(self.name), "i_num_inputs.eps")
        plot_stats(stats[I_NUM_INPUTS], filepath, "Number of Inputs")
        filepath = os.path.join(self.stats_path(self.name), "i_max_num_latches.eps")
        plot_stats(stats[I_NUM_LATCHES], filepath, "Number of Latches")
        filepath = os.path.join(self.stats_path(self.name), "i_num_outputs.eps")
        plot_stats(stats[I_NUM_OUTPUTS], filepath, "Number of Outputs")
        filepath = os.path.join(self.stats_path(self.name), "i_num_and_gates.eps")
        plot_stats(stats[I_NUM_AND_GATES], filepath, "Number of AND Gates")

    def add_metadata(self):
        self.metadata = {**self.metadata, **self.stats()}

    @classmethod
    def load_from_path(cls, path: str, splits: List[str] = ["train", "val", "test"]):
        if not splits:
            splits = ["train", "val", "test"]

        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
            logger.info("Read in metadata")

        split_dataset = cls(metadata=metadata)
        for split in splits:
            split_path = os.path.join(path, split + ".csv")
            if metadata.get("curriculum", False):
                raise NotImplementedError
            else:
                split_dataset[split] = LTLRepairData.load_from_path(split_path)
        return split_dataset

    @classmethod
    def load_from_LTLRepairGenData(
        cls,
        name: str,
        load_from_model: str,
        load_from_alpha: List[float],
        load_from_beamsize: List[int],
        load_from_num_samples: int = None,
        overwrite: bool = True,
        splits: List[str] = None,
        generate: bool = False,
        calc_stats: bool = True,
        is_reference: bool = False,
    ):
        """Uses LTLRepairGenData Evaluation Results, to create a LTLRepairData dataset.

        Args:
            name (str): The new name of the dataset
            load_from_model (str): The name of the model (base model) of which evaluation the dataset will be created
            load_from_alpha (List[float]): Which alpha was used in the evaluation.
            load_from_beamsize (List[int]): which beamsize was used in the evaluation.
            load_from_num_samples (int, optional): How many samples where used in the Evaluation. Defaults to all samples.
            overwrite (bool, optional): If the base model (including evaluation data) should be downloaded from the server. Defaults to True.
            splits (List[str], optional): The splits in this dataset. Defaults to ["train", "val", "test"].
            generate (bool, optional): Whether the data should be loaded (False) or also generated (True). Defaults to True.
            calc_stats (bool, optional): Whether to automatically calculate the statistics cs for metadata. Might take some time. Defaults to True.
            is_reference (bool, optional): Whether this dataset will be a reference dataset (Containing as most alternative solutions as possible). Defaults to False.

        Raises:
            NotImplementedError: If generate is True

         Returns:
            LTLRepairSplitData: The dataset containing the splits.
        """
        if not splits:
            splits = ["train", "val", "test"]

        if generate:
            raise NotImplementedError

        def get_parent_dataset() -> str:
            path = os.path.join(LTLRepairGenData.local_path(load_from_model), "args.json")
            with open(path, "r") as parent_arguments_file:
                parent_arguments_buffer = parent_arguments_file.read()
                parent_arguments = json.loads(parent_arguments_buffer)
                return parent_arguments["dataset_name"]

        if name in cls.ALIASES:
            name = cls.ALIASES[name]
        LTLRepairGenData.download(load_from_model, overwrite)

        metadata = (
            {
                "parent_model": load_from_model,
                "parent_dataset": get_parent_dataset(),
                "eval_beam_size": str(load_from_beamsize[0]),
                "eval_alpha": load_from_alpha[0],
            }
            if ((len(load_from_beamsize) == 1) and (len(load_from_alpha) == 1))
            else {}
        )
        split_dataset = cls(metadata=metadata)
        num_samples_str = "-n" + str(load_from_num_samples) if load_from_num_samples else ""
        for split in splits:
            df = []
            for bs in load_from_beamsize:
                for a in load_from_alpha:
                    split_path = os.path.join(
                        cls.local_path(load_from_model),
                        "gen",
                        "a" + str(a) + "-bs" + str(bs) + num_samples_str,
                        split + ".csv",
                    )
                    dataset = LTLRepairData.load_from_path(split_path, keep_timeouts=False)
                    df.append(dataset.data_frame)
            dataset.data_frame = pd.concat(df, ignore_index=True).drop_duplicates()
            split_dataset[split] = dataset
        split_dataset.name = name
        inputs, outputs = (
            split_dataset[split_dataset.split_names[0]].data_frame["inputs"][0].split(","),
            split_dataset[split_dataset.split_names[0]].data_frame["outputs"][0].split(","),
        )
        split_dataset.metadata = {
            **metadata,
            "inputs": inputs,
            "outputs": outputs,
        }
        if calc_stats:
            split_dataset.add_metadata()
        if is_reference:
            for split in split_dataset.split_names:
                split_dataset[split].treat_satisfied_as_match()
                split_dataset[split].data_frame = split_dataset[split].data_frame[
                    split_dataset[split].data_frame["levenshtein_distance"] == 0
                ]
        return split_dataset

    @classmethod
    def load_from_LTLSynData(
        cls,
        name: str,
        load_from: str,
        overwrite: bool = True,
        splits: List[str] = None,
        samples: int = None,
        calc_stats: bool = True,
    ):
        """Uses a LTLSynData dataset to create a LTLRepairData dataset. Introduces the necessary columns and creates the repair circuit by copying the target circuit.

        Args:
            name (str): The new name of the dataset
            load_from (str): Where to load the dataset from
            overwrite (bool, optional): Whether to overwrite the original dataset with the cloud version. Defaults to True.
            splits (List[str], optional): The splits in this dataset. Defaults to ["train", "val", "test"].
            samples (int, optional): Set only for debugging purposes. Limits the number of samples to read in. Defaults to None.

        Raises:
            NotImplementedError:If dataset is curriculum

        Returns:
            LTLRepairSplitData: The dataset containing the splits.
        """

        if not splits:
            splits = ["train", "val", "test"]

        if name in cls.ALIASES:
            name = cls.ALIASES[name]
        cls.download(load_from, overwrite)

        metadata_path = os.path.join(cls.local_path(load_from), "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
            logger.info("Read in metadata")
        metadata = {"inputs": metadata["inputs"], "outputs": metadata["outputs"]}
        metadata["parent_dataset"] = load_from
        split_dataset = cls(metadata=metadata)
        for split in splits:
            split_path = os.path.join(cls.local_path(load_from), split + ".csv")
            if metadata.get("curriculum", False):
                raise NotImplementedError
            else:
                split_dataset[split] = LTLRepairData.from_LTLSynData(split_path)
        if samples:
            for split in split_dataset.split_names:
                split_dataset[split].data_frame = split_dataset[split].data_frame[:samples]
        split_dataset.name = name
        if calc_stats:
            split_dataset.add_metadata()
        return split_dataset

    def treat_satisfied_as_match(self, all: bool = False):
        if all:
            for split in self.split_names:
                self[split].treat_satisfied_as_match()
        else:
            self["train"].treat_satisfied_as_match()

    def find_minimal_alternative_target(
        self, reference_dataset: "LTLRepairSplitData" = None, all: bool = False
    ):
        if all:
            for split in self.split_names:
                self[split].find_minimal_alternative_target(
                    dataset_reference=reference_dataset[split] if reference_dataset else None
                )
        else:
            self["train"].find_minimal_alternative_target(
                dataset_reference=reference_dataset["train"] if reference_dataset else None
            )

    def filter_matches(self, max_match_fraction: float, alter: bool = False, **params):
        for split in self.split_names:
            self[split].filter_matches(
                max_match_fraction=max_match_fraction, alter=alter, **params
            )

    def filter_distance(self, max_distance: int, alter: bool = False, **params):
        if max_distance > 0:
            for split in self.split_names:
                self[split].filter_distance_bigger(
                    max_distance=max_distance, alter=alter, **params
                )

    def add_dummy_split(self, mode: str, splits: List[str] = ["val", "test"]):
        for split in splits:
            if split not in self.split_names:
                raise ValueError
            name = "dummy-" + mode + "-" + split
            if name in self.split_names:
                logger.info("Split already exists. Will overwrite split " + name)
            if mode == "circuit":
                new_df = self[split].data_frame.copy()
                new_df[
                    "repair_circuit"
                ] = "aag 10 5 0 5 0\\n2\\n4\\n6\\n8\\n10\\n2\\n4\\n6\\n8\\n10"
                new_df["status"] = "Changed"
                self["dummy-" + mode + "-" + split] = LTLRepairData(new_df)

            elif mode == "spec":
                new_df = self[split].data_frame.copy()
                new_df["assumptions"] = ""
                new_df["guarantees"] = "i0"
                new_df["status"] = "Changed"
                self["dummy-" + mode + "-" + split] = LTLRepairData(new_df)
            else:
                raise ValueError

    def add_broken_split(self, splits: List[str] = ["val", "test"]):
        for split in splits:
            if split not in self.split_names:
                raise ValueError
            self["broken-" + split] = LTLRepairData(
                self[split].data_frame[
                    (self[split].data_frame["status"] != "Match")
                    & (self[split].data_frame["status"] != "Satisfied")
                ]
            )

    def alter_circuit(
        self,
        **params,
    ):
        """Modifies (i.e.) breaks the repair circuit according to chosen parameters

        Args:
            params : Pass parameters to Circuit.alter()
        """
        for split in self.split_names:
            self[split].alter_circuit(**params)

    def tf_dataset(self, input_encoder, circuit_encoder, target_encoder, splits=None):
        """Constructs for each split a tensorflow dataset given an input encoder
        and a target encoder"""
        split_names = splits if splits else self.split_names
        return {
            name: split.tf_dataset(input_encoder, circuit_encoder, target_encoder)
            for name, split in self._splits.items()
            if name in split_names
        }

    @classmethod
    def add_basic_args(cls, parser):
        parser.add_argument("--overwrite-base", action="store_true", dest="overwrite_base")
        parser.add_argument("--overwrite-new", action="store_true", dest="overwrite_new")
        parser.add_argument("--upload", action="store_true", dest="upload")
        parser.add_argument("--name", type=str, dest="name", default=None)
        parser.add_argument("--splits", nargs="*", dest="splits", default=None)

    @classmethod
    def add_alter_basic_args(cls, parser):
        parser.add_argument("--from", type=str, dest="from")

    @classmethod
    def add_sweep_args(cls, parser):
        parser.add_argument("--mode", type=str, dest="mode", default=None)
        parser.add_argument("--sweep-table", type=str, dest="sweep_table", default=None)

    @classmethod
    def add_generate_basic_args(cls, parser):
        parser.add_argument("--generate", action="store_true", dest="generate", default=False)
        parser.add_argument("--parent-model", type=str, dest="load_from_model")
        parser.add_argument("--load-from-alpha", type=str, dest="load_from_alpha")
        parser.add_argument("--load-from-beamsize", type=str, dest="load_from_beamsize")
        parser.add_argument(
            "--load-from-num-samples", type=str, dest="load_from_num_samples", default=None
        )
        parser.add_argument(
            "--reference-beamsizes", nargs="*", dest="reference_beamsizes", default=None
        )
        parser.add_argument("--reference-alphas", nargs="*", dest="reference_alphas", default=None)

    @classmethod
    def add_alter_circuit_args(cls, parser):
        parser.add_argument("--max-changes", type=int, dest="max_changes", default=50)
        parser.add_argument("--min-changes", type=int, dest="min_changes", default=1)
        parser.add_argument("--range-68-changes", type=int, dest="range_68_changes", default=15)
        parser.add_argument("--min-var", type=int, dest="min_var", default=0)
        parser.add_argument("--max-var", type=int, dest="max_var", default=61)
        parser.add_argument("--range-68-var", type=int, dest="range_68_var", default=20)
        parser.add_argument("--fraction-delete", type=float, dest="fraction_delete", default=0.2)

    @classmethod
    def add_alter_generate_args(cls, parser):
        parser.add_argument("--max-distance", type=int, dest="max_distance", default=None)
        parser.add_argument(
            "--max-match-fraction", type=float, dest="max_match_fraction", default=1.0
        )
        parser.add_argument(
            "--remove-or-alter", type=str, dest="remove_or_alter", default="remove"
        )
        parser.add_argument(
            "--replace-satisfied", action="store_true", dest="replace_satisfied", default=False
        )
        parser.add_argument(
            "--replace-minimal", action="store_true", dest="replace_minimal", default=False
        )
        parser.add_argument("--add-broken", action="store_true", dest="add_broken", default=False)
        parser.add_argument("--add-dummy", action="store_true", dest="add_dummy", default=False)

    @classmethod
    def cli(cls):
        parser = argparse.ArgumentParser(description="ML2 Repair Data Creation")
        subparsers = parser.add_subparsers(dest="command", help="")

        alter_parser = subparsers.add_parser(
            "alter", help="Use Synthesis Data to alter the circuit and create Repair Data"
        )
        cls.add_basic_args(alter_parser)
        cls.add_alter_basic_args(alter_parser)
        cls.add_alter_circuit_args(alter_parser)

        generate_parser = subparsers.add_parser(
            "generate", help="Use Evaluation Data to create a Repair Dataset"
        )
        cls.add_basic_args(generate_parser)
        cls.add_generate_basic_args(generate_parser)
        cls.add_alter_generate_args(generate_parser)
        cls.add_alter_circuit_args(generate_parser)

        sweep_parser = subparsers.add_parser(
            "sweep", help="Efficiently generate a batch of different dataset"
        )
        cls.add_basic_args(sweep_parser)
        cls.add_generate_basic_args(sweep_parser)
        cls.add_alter_basic_args(sweep_parser)
        cls.add_sweep_args(sweep_parser)

        args = parser.parse_args()
        args_dict = vars(args)
        command = args_dict.pop("command")

        if not args_dict["name"]:
            raise ValueError
        name = args_dict.pop("name")
        upload = args_dict.pop("upload")
        overwrite_new = args_dict.pop("overwrite_new")
        overwrite_base = args_dict.pop("overwrite_base")
        generate = args_dict.pop("generate")

        if command == "alter":
            dataset = cls.load_from_LTLSynData(
                name=name,
                load_from=args_dict["from"],
                overwrite=overwrite_base,
                splits=args_dict["splits"],
                calc_stats=False,
            )
            dataset.alter_circuit(**args_dict)
            dataset.metadata = {**dataset.metadata, **dataset.stats(), **args_dict}
            dataset.save(
                name=name,
                upload=upload,
                overwrite_local=overwrite_new,
                overwrite_bucket=overwrite_new,
                auto_version=False,
                add_to_wandb=upload,
            )

        elif command == "generate":
            dataset = cls.load_from_LTLRepairGenData(
                name=name,
                load_from_model=args_dict["load_from_model"],
                load_from_alpha=[args_dict["load_from_alpha"]],
                load_from_beamsize=[args_dict["load_from_beamsize"]],
                load_from_num_samples=args_dict["load_from_num_samples"],
                overwrite=overwrite_base,
                splits=args_dict["splits"],
                generate=generate,
                calc_stats=False,
            )
            dataset.process_misleading(**args_dict)
            dataset.process_filter(**args_dict)
            dataset.process_dummy(**args_dict)
            dataset.metadata = {**dataset.metadata, **args_dict, **dataset.stats()}
            dataset.save(
                name=name,
                upload=upload,
                overwrite_local=overwrite_new,
                overwrite_bucket=overwrite_new,
                auto_version=False,
                add_to_wandb=upload,
            )

        elif command == "sweep":
            # sweep arguments
            mode = args_dict.pop("mode")
            sweep_table = pd.read_json(args_dict.pop("sweep_table"), orient="table")

            if "name" not in sweep_table.columns and not upload:
                raise ValueError("auto versioning of sweep only possible if upload is true")

            if mode == "alter":
                dataset = cls.load_from_LTLSynData(
                    name="tmp",
                    load_from=args_dict["from"],
                    overwrite=overwrite_base,
                    splits=args_dict["splits"],
                    calc_stats=False,
                )
                for _, line in sweep_table.iterrows():
                    sweep_args = line.to_dict()
                    dataset_cp = deepcopy(dataset)
                    dataset_cp.name = sweep_args["name"] if "name" in sweep_args.keys() else name
                    dataset_cp.alter_circuit(**sweep_args)
                    dataset_cp.metadata = {
                        **dataset_cp.metadata,
                        **args_dict,
                        **sweep_args,
                        **dataset_cp.stats(),
                    }
                    dataset_cp.save(
                        name=sweep_args["name"] if "name" in sweep_args.keys() else name,
                        upload=upload,
                        overwrite_local=overwrite_new,
                        overwrite_bucket=overwrite_new,
                        auto_version=("name" not in sweep_args.keys() and upload),
                        add_to_wandb=upload,
                    )

            elif mode == "generate":
                args_dict.pop("load_from_alpha", None)
                args_dict.pop("load_from_beamsize", None)
                if "load_from_beamsize" not in sweep_table.columns:
                    sweep_table["load_from_beamsize"] = 1
                if "load_from_alpha" not in sweep_table.columns:
                    sweep_table["load_from_alpha"] = 0.5

                for beam_size, group_beam in sweep_table.groupby(by="load_from_beamsize"):
                    for alpha, group_alpha in group_beam.groupby(by="load_from_alpha"):
                        dataset = cls.load_from_LTLRepairGenData(
                            name=name,
                            load_from_model=args_dict["load_from_model"],
                            load_from_alpha=[alpha],
                            load_from_beamsize=[beam_size],
                            load_from_num_samples=args_dict["load_from_num_samples"],
                            overwrite=overwrite_base,
                            splits=args_dict["splits"],
                            calc_stats=False,
                        )
                        if (
                            "replace_minimal" in group_alpha.columns
                            and group_alpha["replace_minimal"].max()
                        ):
                            dataset_s = deepcopy(dataset)
                            dataset_s.process_misleading(
                                load_from_alpha=alpha,
                                load_from_beamsize=beam_size,
                                replace_minimal=True,
                                replace_satisfied=False,
                                **args_dict,
                            )
                            for _, line in group_alpha[group_alpha["replace_minimal"]].iterrows():
                                sweep_args = line.to_dict()
                                dataset_cp = deepcopy(dataset_s)
                                dataset_cp.name = (
                                    sweep_args["name"] if "name" in sweep_args.keys() else name
                                )
                                dataset_cp.process_filter(**sweep_args)
                                dataset_cp.process_dummy(**sweep_args)
                                dataset_cp.metadata = {
                                    **dataset_cp.metadata,
                                    **args_dict,
                                    **sweep_args,
                                    **dataset_cp.stats(),
                                }
                                dataset_cp.save(
                                    name=sweep_args["name"]
                                    if "name" in sweep_args.keys()
                                    else name,
                                    upload=upload,
                                    overwrite_local=overwrite_new,
                                    overwrite_bucket=overwrite_new,
                                    auto_version=("name" not in sweep_args.keys() and upload),
                                    add_to_wandb=upload,
                                )
                        group_alpha = group_alpha[group_alpha["replace_minimal"] != True]
                        if (
                            "replace_satisfied" in group_alpha.columns
                            and group_alpha["replace_satisfied"].max()
                        ):
                            dataset_m = deepcopy(dataset)
                            dataset_m.process_misleading(
                                load_from_alpha=alpha,
                                load_from_beamsize=beam_size,
                                replace_minimal=False,
                                replace_satisfied=True,
                                **args_dict,
                            )
                            for _, line in group_alpha[
                                group_alpha["replace_satisfied"]
                            ].iterrows():
                                sweep_args = line.to_dict()
                                dataset_cp = deepcopy(dataset_m)
                                dataset_cp.name = (
                                    sweep_args["name"] if "name" in sweep_args.keys() else name
                                )
                                dataset_cp.process_filter(**sweep_args)
                                dataset_cp.process_dummy(**sweep_args)
                                dataset_cp.metadata = {
                                    **dataset_cp.metadata,
                                    **args_dict,
                                    **sweep_args,
                                    **dataset_cp.stats(),
                                }
                                dataset_cp.save(
                                    name=sweep_args["name"]
                                    if "name" in sweep_args.keys()
                                    else name,
                                    upload=upload,
                                    overwrite_local=overwrite_new,
                                    overwrite_bucket=overwrite_new,
                                    auto_version=("name" not in sweep_args.keys() and upload),
                                    add_to_wandb=upload,
                                )
                        group_alpha = group_alpha[group_alpha["replace_satisfied"] != True]

                        for _, line in group_alpha.iterrows():
                            sweep_args = line.to_dict()
                            dataset_cp = deepcopy(dataset)
                            dataset_cp.name = (
                                sweep_args["name"] if "name" in sweep_args.keys() else name
                            )
                            dataset_cp.process_filter(**sweep_args)
                            dataset_cp.process_dummy(**sweep_args)
                            dataset_cp.metadata = {
                                **dataset_cp.metadata,
                                **args_dict,
                                **sweep_args,
                                **dataset_cp.stats(),
                            }
                            dataset_cp.save(
                                name=sweep_args["name"] if "name" in sweep_args.keys() else name,
                                upload=upload,
                                overwrite_local=overwrite_new,
                                overwrite_bucket=overwrite_new,
                                auto_version=("name" not in sweep_args.keys() and upload),
                                add_to_wandb=upload,
                            )

            else:
                raise ValueError

        else:
            raise Exception("Unknown command %s", args.command)

    def process_misleading(
        self,
        load_from_model: str,
        splits: List[str],
        load_from_beamsize: int,
        load_from_alpha: float,
        reference_alphas: List[float] = None,
        reference_beamsizes: List[int] = None,
        replace_minimal: bool = False,
        replace_satisfied: bool = False,
        **_,
    ):

        if replace_minimal and replace_satisfied:
            raise ValueError("set either replace minimal or replace satisfied")
        elif replace_satisfied:
            self.treat_satisfied_as_match(all=True)
        elif replace_minimal:
            if not reference_beamsizes and reference_alphas:
                reference_beamsizes = [load_from_beamsize]
            if not reference_alphas and reference_beamsizes:
                reference_alphas = [load_from_alpha]
            if reference_alphas and reference_beamsizes:
                dataset_reference = LTLRepairSplitData.load_from_LTLRepairGenData(
                    "tmp",
                    load_from_model=load_from_model,
                    load_from_alpha=reference_alphas,
                    load_from_beamsize=reference_beamsizes,
                    calc_stats=False,
                    is_reference=True,
                    splits=splits,
                )
                self.find_minimal_alternative_target(reference_dataset=dataset_reference)
            else:
                self.find_minimal_alternative_target()

    def process_dummy(self, add_broken: bool = False, add_dummy: bool = False, **_):
        if add_broken:
            self.add_broken_split()
        if add_dummy:
            self.add_dummy_split("circuit")
            self.add_dummy_split("spec")

    def process_filter(
        self,
        max_match_fraction: float = 1,
        max_distance: int = None,
        remove_or_alter: str = "remove",
        **params,
    ):
        if remove_or_alter not in ["remove", "alter"]:
            raise ValueError("argument for remove_or_alter not allowed: " + remove_or_alter)

        if max_distance is not None and max_distance > 0:
            self.filter_distance(
                max_distance=max_distance,
                alter=(remove_or_alter == "alter"),
                **params,
            )
        if max_match_fraction is not None and max_match_fraction != 1.0:
            self.filter_matches(
                max_match_fraction=max_match_fraction,
                alter=(remove_or_alter == "alter"),
                **params,
            )

    def model_check_repair_circuit(
        self,
        timeout: float = 10.0,
        set_status_satisfied: bool = False,
        set_status_violated: bool = False,
        only_altered: bool = False,
        samples: Optional[int] = None,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        nuxmv = nuXmv(port=50051)
        splits = self.split_names if splits is None else splits
        results: Dict[str, float] = {}
        for split in self.split_names:
            total_changed: int = 0
            satisfied_changed: int = 0
            counters = {}
            with tqdm(desc=split + "in " + self.name) as pbar:
                if only_altered:
                    df = self[split].data_frame[self[split].data_frame["status"] == "Changed"]
                df = df if (samples is None or samples >= len(df)) else df.sample(n=samples)
                for _, row in df.iterrows():
                    if row["status"] == "Changed":
                        total_changed = total_changed + 1
                    sample = {
                        "assumptions": row["assumptions"].split(",")
                        if "assumptions" in row and row["assumptions"]
                        else [],  # key in dict check for data that does not contain assumptions
                        "guarantees": row["guarantees"].split(",") if row["guarantees"] else [],
                        "inputs": row["inputs"].split(",") if row["inputs"] else [],
                        "outputs": row["outputs"].split(",") if row["outputs"] else [],
                    }
                    result = nuxmv.model_check(
                        LTLSpec.from_dict(sample),
                        row["repair_circuit"] + "\n",
                        realizable=row["realizable"],
                        timeout=timeout,
                    )

                    if row["circuit"] == row["repair_circuit"]:
                        satisfied_changed = (
                            satisfied_changed + 1
                            if row["status"] == "Changed"
                            else satisfied_changed
                        )
                        row["status"] == "Match" if set_status_satisfied else row["status"]
                    elif result.value == "satisfied":
                        satisfied_changed = (
                            satisfied_changed + 1
                            if row["status"] == "Changed"
                            else satisfied_changed
                        )
                        row["status"] == "Satisfied" if set_status_satisfied else row["status"]
                    elif result.value == "violated" and set_status_violated:
                        row["status"] = "Violated"
                    elif result.value == "error" and row["status"] != "Changed":
                        print(sample)
                        row["status"] = "Error"
                    elif result.value == "invalid":
                        row["status"] == "Invalid"

                    counters[result.value] = counters.get(result.value, 0) + 1
                    pbar.update()
                    pbar.set_postfix(counters)
            results[split] = satisfied_changed / total_changed if total_changed != 0 else 0
        return results


def space_dataset(path: str, num_next: int = 2):
    # TODO test function
    split_dataset = LTLRepairSplitData.load_from_path(path)
    fin_regex = re.compile("(F)([A-Za-z])")
    glob_regex = re.compile("(G)([A-Za-z])")
    next_regex = re.compile("(X)([A-Za-z])")
    for split in ["train", "val", "test", "timeouts"]:
        df = split_dataset[split].data_frame
        df["guarantees"] = df["guarantees"].str.replace(fin_regex, r"\g<1> \g<2>")
        df["guarantees"] = df["guarantees"].str.replace(glob_regex, r"\g<1> \g<2>")
        for _ in range(num_next):
            df["guarantees"] = df["guarantees"].str.replace(next_regex, r"\g<1> \g<2>")
    split_dataset.save_to_path(path)


if __name__ == "__main__":
    LTLRepairSplitData.cli()
