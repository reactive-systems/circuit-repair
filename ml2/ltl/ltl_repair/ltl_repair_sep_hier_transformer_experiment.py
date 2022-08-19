"""
Experiment for the separated Hierarchical Transformer. This Experiments mirrows the LTLSynHierTransformerExperiment but uses a separated hierarchical transformer with one local encoder. Uses ltl_repair_data as dataset class.
"""

import argparse
from collections import Counter
import csv
import json
import logging
import os
import random
import shutil
from typing import Dict, List, Optional, Tuple
import numpy as np
import sys
import tensorflow as tf
import pandas
from tqdm import tqdm
import wandb
from ml2.aiger.aiger_encoder import AIGERSequenceEncoder

from ml2.ltl.ltl_repair.ltl_repair_data import LTLRepairData, LTLRepairSplitData
from ml2.ltl.ltl_repair.ltl_repair_data_gen import LTLRepairGenData
from ml2.ltl.ltl_spec.ltl_spec import LTLSpec
from ml2.tools.nuxmv.nuxmv import nuXmv
from ...globals import LTL_REP_BUCKET_DIR, LTL_REP_WANDB_PROJECT

from ... import models
from ...layers import positional_encoding as pe

from ...data import TPEFormat
from ...data import ExprNotation
from ...optimization import lr_schedules
from ..ltl_spec import LTLSpecPropertyEncoder
from ..ltl_syn.ltl_syn_experiment import LTLSynExperiment
import plotly.graph_objects as go


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLRepSepHierTransformerExperiment(LTLSynExperiment):

    BUCKET_DIR = LTL_REP_BUCKET_DIR
    WANDB_PROJECT = LTL_REP_WANDB_PROJECT

    def __init__(
        self,
        ttot_learning: bool = False,
        constant_learning_rate: float = None,
        custom_pos_enc: bool = True,
        d_embed: int = 256,
        d_embed_enc: int = None,
        d_embed_dec: int = None,
        d_ff: int = 1024,
        d_ff_enc_g: int = None,
        d_ff_enc_l1: int = None,
        d_ff_enc_l2: int = None,
        d_ff_dec: int = None,
        dropout: float = 0.0,
        dropout_enc: float = None,
        dropout_dec: float = None,
        ff_activation_enc_g: str = "relu",
        ff_activation_enc_l1: str = "relu",
        ff_activation_enc_l2: str = "relu",
        ff_activation_dec: str = "relu",
        fix_local_embed: bool = False,
        name: str = "hier-transformer",
        num_properties: int = 12,
        num_heads: int = 4,
        num_heads_enc_g: int = None,
        num_heads_enc_l1: int = None,
        num_heads_enc_l2: int = None,
        num_heads_dec: int = None,
        num_layers: int = 8,
        num_layers_enc_g: int = None,
        num_layers_enc_l1: int = None,
        num_layers_enc_l2: int = None,
        num_layers_dec: int = None,
        property_tree_size: int = 25,
        warmup_steps: int = 4000,
        dataset_name: str = "repair-0",
        **kwargs,
    ):
        self.ttot_learning = ttot_learning
        self._attn_model = None
        self.constant_learning_rate = constant_learning_rate
        self.custom_pos_enc = custom_pos_enc
        if not custom_pos_enc:
            raise NotImplementedError
        self.d_embed_enc = d_embed_enc if d_embed_enc else d_embed
        self.d_embed_dec = d_embed_dec if d_embed_dec else d_embed
        self.d_ff_enc_g = d_ff_enc_g if d_ff_enc_g else d_ff
        self.d_ff_enc_l1 = d_ff_enc_l1 if d_ff_enc_l1 else d_ff
        self.d_ff_enc_l2 = d_ff_enc_l2 if d_ff_enc_l2 else d_ff
        self.d_ff_dec = d_ff_dec if d_ff_dec else d_ff
        self.dropout_enc = dropout_enc if dropout_enc else dropout
        self.dropout_dec = dropout_dec if dropout_dec else dropout
        self.ff_activation_enc_g = ff_activation_enc_g
        self.ff_activation_enc_l1 = ff_activation_enc_l1
        self.ff_activation_enc_l2 = ff_activation_enc_l2
        self.ff_activation_dec = ff_activation_dec
        self.fix_local_embed = fix_local_embed
        self.property_tree_size = property_tree_size
        self.num_properties = num_properties
        self.num_heads_enc_g = num_heads_enc_g if num_heads_enc_g else num_heads
        self.num_heads_enc_l1 = num_heads_enc_l1 if num_heads_enc_l1 else num_heads
        self.num_heads_enc_l2 = num_heads_enc_l2 if num_heads_enc_l2 else num_heads
        self.num_heads_dec = num_heads_dec if num_heads_dec else num_heads
        self.num_layers_enc_g = num_layers_enc_g if num_layers_enc_g else num_layers // 2
        self.num_layers_enc_l1 = num_layers_enc_l1 if num_layers_enc_l1 else num_layers // 2
        self.num_layers_enc_l2 = num_layers_enc_l2 if num_layers_enc_l2 else num_layers // 2
        self.num_layers_dec = num_layers_dec if num_layers_dec else num_layers
        self.warmup_steps = warmup_steps
        self._circuit_encoder = None
        if self.d_embed_enc % self.num_heads_enc_g != 0:
            sys.exit(
                f"Encoder embedding dimension {self.d_embed_enc} is "
                "not divisible by the number of attention heads"
                f"{self.num_heads_enc_g}"
            )
        if self.d_embed_enc % self.num_heads_enc_l1 != 0:
            sys.exit(
                f"Encoder embedding dimension {self.d_embed_enc} is "
                "not divisible by the number of attention heads"
                f"{self.num_heads_enc_l1}"
            )
        if self.d_embed_enc % self.num_heads_enc_l2 != 0:
            sys.exit(
                f"Encoder embedding dimension {self.d_embed_enc} is "
                "not divisible by the number of attention heads"
                f"{self.num_heads_enc_l2}"
            )
        if self.d_embed_dec % self.num_heads_dec != 0:
            sys.exit(
                (
                    f"Decoder embedding dimension {self.d_embed_dec} is "
                    "not divisible by the number of attention heads "
                    f"{self.num_heads_dec}"
                )
            )
        super().__init__(name=name, dataset_name=dataset_name, **kwargs)

    @property
    def attn_model(self):
        if not self._attn_model:
            self._attn_model = self.init_model(training=False, attn_weights=True)
            logger.info("Created attention model")
            checkpoint = tf.train.latest_checkpoint(self.local_path(self.name))
            if checkpoint:
                logger.info("Found checkpoint %s", checkpoint)
                self._attn_model.load_weights(checkpoint).expect_partial()
                logger.info("Loaded weights from checkpoint")
        return self._attn_model

    def attn_weights(self, spec, training: bool = False):
        if not self.input_encoder.encode(spec):
            logger.info("Econding error: %s", self.input_encoder.error)
            return None
        formula_tensor, pos_enc_tensor = self.input_encoder.tensor
        # pylint: disable=E1102
        preds, _, enc_attn_local, enc_attn_global, dec_attn = self.attn_model(
            (tf.expand_dims(formula_tensor, axis=0), tf.expand_dims(pos_enc_tensor, axis=0)),
            training=training,
        )
        results = []
        attention_dict_list = []

        if self.fix_local_embed:

            tokens = []
            for a in spec.assumptions:
                tokens.append(a)
            for g in spec.guarantees:
                tokens.append(g)
            token_ids = list(range(len(tokens)))

        else:

            tokens = [
                t
                for local_tokens in self.input_encoder.property_padded_tokens
                for t in local_tokens
            ]
            token_ids = []

            for i, t in enumerate(tokens):
                if t != "<p>":
                    token_ids.append(i)

        for head in range(0, self.num_heads_enc_g):
            layerdict = {}
            for layer in range(1, self.num_layers_enc_g + 1):
                playerdict = {}
                for player_new, player in enumerate(token_ids):
                    attended_player_dict = {}
                    for player_attended_new, player_attended in enumerate(token_ids):
                        att = enc_attn_global[f"layer_{layer}"]["self_attn"][0][head][player][
                            player_attended
                        ].numpy()
                        attended_player_dict[player_attended_new] = str(att)
                    playerdict[player_new] = attended_player_dict
                layerdict[layer] = playerdict
            attention_dict_list.append(layerdict)

        for beam in preds[0]:
            if not self.target_encoder.decode(np.array(beam)):
                logger.info("Decoding error: %s", self.target_encoder.error)
                # return None
            beam_result = {}
            beam_result["circuit"] = self.target_encoder.circuit
            results.append(beam_result)
        return (attention_dict_list, self.num_layers_enc_g, [tokens[i] for i in token_ids])

    @property
    def init_input_encoder(self):
        return LTLSpecPropertyEncoder(
            property_pad=self.property_tree_size,
            num_properties=self.num_properties,
            notation=ExprNotation.INFIX,
            encoded_notation=ExprNotation.PREFIX,
            eos=False,
            tpe_format=TPEFormat.BRANCHDOWN,
            tpe_pad=self.d_embed_enc,
        )

    @property
    def init_target_encoder(self):
        return AIGERSequenceEncoder(
            start=True,
            eos=True,
            pad=self.max_target_length,
            components=self.aiger_order,
            encode_start=False,
            encode_realizable=self.encode_realizable,
            inputs=self.inputs,
            outputs=self.outputs,
            unfold_negations=self.aiger_unfold_negations,
            unfold_latches=self.aiger_unfold_latches,
            include_satisfied_token=False,
        )

    @property
    def init_circuit_encoder(self):
        return AIGERSequenceEncoder(
            start=True,
            eos=True,
            pad=self.max_target_length,
            components=self.aiger_order,
            encode_start=False,
            encode_realizable=self.encode_realizable,
            inputs=self.inputs,
            outputs=self.outputs,
            unfold_negations=self.aiger_unfold_negations,
            unfold_latches=self.aiger_unfold_latches,
            include_satisfied_token=False,
        )

    @property
    def init_learning_rate(self):
        if self.constant_learning_rate:
            return self.constant_learning_rate
        return lr_schedules.TransformerSchedule(self.d_embed_enc, warmup_steps=self.warmup_steps)

    @property
    def init_optimizer(self):
        return tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

    def __create_hyperparams(self):
        args = self.args
        params = {}
        params_sep_local1 = {}
        params_sep_local2 = {}

        params["d_embed_enc"] = args["d_embed_enc"]
        params["d_ff_enc"] = args["d_ff_enc_g"]
        params["dropout_enc"] = args["dropout_enc"]
        params["ff_activation_enc"] = args["ff_activation_enc_g"]
        params["num_heads_enc"] = args["num_heads_enc_g"]
        params["num_layers_enc"] = args["num_layers_enc_g"]
        params["input_pad_id"] = self.input_pad_id
        params["alpha"] = args["alpha"]
        params["beam_size"] = args["beam_size"]
        params["d_embed_dec"] = args["d_embed_dec"]
        params["d_ff_dec"] = args["d_ff_dec"]
        params["dropout_dec"] = args["dropout_dec"]
        params["max_decode_length"] = self.max_target_length
        params["dtype_float"] = args["dtype_float"]
        params["dtype_int"] = args["dtype_int"]
        params["num_heads_dec"] = args["num_heads_dec"]
        params["ff_activation_dec"] = args["ff_activation_dec"]
        params["num_layers_dec"] = args["num_layers_dec"]
        params["target_pad_id"] = self.input_pad_id
        params["target_eos_id"] = self.target_eos_id
        params["target_start_id"] = self.target_start_id
        params["target_vocab_size"] = self.target_vocab_size
        params["fix_local_embed"] = self.fix_local_embed
        params["drop_batch_remainder"] = args["drop_batch_remainder"]

        params_sep_local1["input_vocab_size"] = self.input_vocab_size
        params_sep_local1["d_ff_enc"] = args["d_ff_enc_l1"]
        params_sep_local1["ff_activation_enc"] = args["ff_activation_enc_l1"]
        params_sep_local1["num_heads_enc"] = args["num_heads_enc_l1"]
        params_sep_local1["num_layers_enc"] = args["num_layers_enc_l1"]
        params_sep_local1["input_dimensions"] = (self.num_properties, self.property_tree_size)

        params_sep_local2["input_vocab_size"] = self.target_vocab_size
        params_sep_local2["d_ff_enc"] = args["d_ff_enc_l2"]
        params_sep_local2["ff_activation_enc"] = args["ff_activation_enc_l2"]
        params_sep_local2["num_heads_enc"] = args["num_heads_enc_l2"]
        params_sep_local2["num_layers_enc"] = args["num_layers_enc_l2"]
        params_sep_local2["input_dimensions"] = (1, self.max_target_length)

        params["params_sep_local"] = [params_sep_local1, params_sep_local2]

        return params

    def init_model(self, training: bool = True, attn_weights: bool = False):
        params = self.__create_hyperparams()
        return models.fast_2_separated_hierarchical_transformer.create_model(
            params, training=training, attn_weights=attn_weights
        )

    @property
    def init_dataset(self):
        return LTLRepairSplitData.load(self.dataset_name)

    @property
    def init_tf_dataset(self):
        return self.dataset.tf_dataset(
            self.input_encoder, self.circuit_encoder, self.target_encoder
        )

    def accuracy_beams(self, alpha: float, beamsize: int) -> float:
        pass

    def eval_pipeline(
        self,
        data: List[str],
        base_model: str,
        alpha_base: float,
        beam_base: int,
        alpha_repair: float,
        beam_repair: int,
        samples: int,
        repeats: int,
        keep: str,
        shuffle_data: bool = True,
        reference_alphas: List[float] = None,
        reference_beams: List[int] = None,
    ):
        # init samples,  batch size, alpha and beam_size
        orig_batch_size = self.batch_size
        self.alpha = alpha_repair
        self.beam_size = beam_repair
        self.batch_size = orig_batch_size // beam_repair

        dataset_reference = None
        for _data in data:
            if _data in ["val", "test", "train", "test_fixed"]:
                if reference_alphas or reference_beams:
                    alphas = reference_alphas if reference_alphas else [alpha_base]
                    beams = reference_beams if reference_beams else [beam_base]
                    dataset_reference = LTLRepairSplitData.load_from_LTLRepairGenData(
                        "tmp",
                        splits=data,
                        load_from_model=base_model,
                        load_from_alpha=alphas,
                        load_from_beamsize=beams,
                        overwrite=False,
                        calc_stats=False,
                        is_reference=True,
                    )
            else:
                if keep == "best":
                    raise ValueError()

            logger.info(
                "Evaluating %s set. Pipeline Mode with %d repeats. Base model: %s, base_alpha: %1.1f, base beam size: %d. Repair alpha: %1.1f, repair beam size: %d.",
                _data,
                repeats,
                base_model,
                alpha_base,
                beam_base,
                alpha_repair,
                beam_repair,
            )

            # create folder for summarized results

            dir = self.pipe_dir(
                alpha_base=alpha_base,
                alpha_repair=alpha_repair,
                beam_base=beam_base,
                beam_repair=beam_repair,
                keep=keep,
                data=_data,
                repeats=repeats,
                samples=samples,
            )
            if not os.path.isdir(dir):
                os.makedirs(dir)
            logger.info("Saving summary to %s", dir)

            # evaluate base model
            identifier = "%030x" % random.randrange(16 ** 30)
            base_experiment = LTLRepairGenData.load(base_model)
            base_experiment.eval(
                alphas=[alpha_base],
                beam_sizes=[beam_base],
                data=_data,
                samples=samples,
                shuffle_data=shuffle_data,
                identifier=identifier,
                mode="pipe",
            )
            shutil.move(
                os.path.join(base_experiment.tmp_dir, identifier + ".csv"),
                os.path.join(dir, "0.csv"),
            )
            shutil.move(
                os.path.join(base_experiment.tmp_dir, identifier + "-stats.json"),
                os.path.join(dir, "0-stats.json"),
            )
            samples = samples * beam_base if keep == "all" else samples
            print(
                "======================SAMPLES CALC: "
                + str(samples)
                + " ========================================================"
            )

            if samples * beam_repair < self.batch_size:
                self.batch_size = samples
                steps = 1
            else:
                steps = None
            # steps = samples // self.batch_size

            # evaluate repair model
            tmp_dir = self.eval_directory(mode="pipe")
            for i in range(1, repeats + 1):
                identifier_new = "%030x" % random.randrange(16 ** 30)
                logger.info(
                    "Evaluating iteration %d with alpha %1.1f, beam_size %d and batch_size %d. Using Identifier %s as input and identifier %s as output.",
                    i,
                    self.alpha,
                    self.beam_size,
                    self.batch_size,
                    identifier,
                    identifier_new,
                )
                identifier = identifier_new

                dataset = LTLRepairData.load_from_path(
                    os.path.join(dir, str(i - 1) + ".csv"), keep_timeouts=False
                )
                data_ref = dataset_reference[_data] if dataset_reference else None
                dataset.keep_one_beam(keep=keep, dataset_reference=data_ref)
                generator = dataset.generator()
                self.eval_generator(
                    generator,
                    str(i - 1),
                    steps=steps,
                    training=False,
                    verify=True,
                    mode="pipe",
                    identifier=identifier,
                    includes_target=False
                    if (_data == "timeouts" or _data == "syntcomp" or _data == "jarvis")
                    else True,
                )

                shutil.move(
                    os.path.join(tmp_dir, identifier + ".csv"), os.path.join(dir, str(i) + ".csv")
                )
                shutil.move(
                    os.path.join(tmp_dir, identifier + "-stats.json"),
                    os.path.join(dir, str(i) + "-stats.json"),
                )
                samples = samples * beam_repair if keep == "all" else samples
                print(
                    "======================SAMPLES CALC: "
                    + str(samples)
                    + " ========================================================"
                )
                if samples * beam_repair < self.batch_size:
                    self.batch_size = samples
                    steps = 1
                else:
                    steps = None

            summary = self.create_summary(dir=dir, repeats=repeats)
            self.calc_pipe_stats(
                summary=summary,
                repeats=repeats,
                beam_base=beam_base,
                beam_repair=beam_repair,
                keep=keep,
                data=_data,
                dir=dir,
            )

    @staticmethod
    def create_summary(dir: str, repeats: int):
        # read base model eval
        tables = [pandas.read_csv(os.path.join(dir, "0.csv"))]
        tables[0] = tables[0].rename(
            columns={
                "status": "status_0",
                "repair_circuit": "prediction_0",
                "circuit": "target",
            }
        )
        tables[0]["hash"] = tables[0]["hash"].astype(str)
        tables[0] = tables[0].reset_index().set_index("hash").drop(["index"], axis=1)

        # read repair model eval
        history = ["hash"]
        for i in range(1, repeats + 1):
            history.append("prediction_" + str(i - 1))
            dataframe = pandas.read_csv(os.path.join(dir, str(i) + ".csv"))
            dataframe[history] = dataframe["hash"].str.split(",", i, expand=True)
            dataframe = dataframe.rename(
                columns={
                    "status": "status_" + str(i),
                    "repair_circuit": "prediction_" + str(i),
                    "circuit": "target",
                }
            ).drop(
                ["assumptions", "guarantees", "inputs", "outputs", "realizable", "target"],
                axis=1,
            )
            dataframe["hash"] = dataframe["hash"].astype(str)
            dataframe = dataframe.reset_index().drop(["index"], axis=1).set_index(history)
            tables.append(dataframe)

        join_on = ["hash"]
        index = ["hash", "prediction_0"]
        summary = tables[0]

        for i in range(1, repeats + 1):
            join_on.append("prediction_" + str(i - 1))
            index.append("prediction_" + str(i))
            summary = (
                summary.join(tables[i], on=join_on, how="left").reset_index().set_index(index)
            )
        summary.to_csv(os.path.join(dir, "summary.csv"))
        return summary

    def calc_pipe_stats(
        self,
        summary: pandas.DataFrame,
        repeats: int,
        beam_base: int,
        beam_repair: int,
        keep: str,
        data: str,
        dir: str,
    ) -> None:
        stats = self.pipe_statistics(summary=summary, repeats=repeats)
        with open(
            os.path.join(dir, "summary-stats.json"),
            "w",
        ) as fout:
            json_dumps_str = json.dumps(stats.to_dict(), indent=4)
            fout.write(json_dumps_str)
        self.pipe_plots(save=True, show=False)
        if self.stream_to_wandb:
            self.pipe_log_wandb_stats(
                stats=stats,
                repeats=repeats,
                beam_base=beam_base,
                beam_repair=beam_repair,
                data=data,
                keep=keep,
            )

    def pipe_dir(
        self,
        alpha_base: float,
        alpha_repair: float,
        beam_base: int,
        beam_repair: int,
        keep: str,
        data: str,
        repeats: int,
        samples: int,
    ) -> str:
        arguments: str = f"a{alpha_base}:{alpha_repair}-bs{beam_base}:{beam_repair}-k{keep}-d{data}-r{repeats}-s{samples}"
        dir = os.path.join(self.eval_dir, "pipe-" + arguments)
        return dir

    @staticmethod
    def create_pipe_plots(filenames: List[str]) -> Tuple[go.Figure, go.Figure]:
        """Create two plotly plots summarizing the overall accuracy of pipe evaluations

        Args:
            filenames (List[str]): A list of filenames on which the plots are based

        Returns:
            Tuple[go.Figure, go.Figure]: The two plotly plots
        """
        fig_over = go.Figure()
        fig_each = go.Figure()
        for filename in filenames:
            stats = pandas.read_json(filename).reset_index()
            filename = filename[filename.find("/eval/") + 6 : filename.find("/summary-stats.json")]
            fig_over.add_trace(
                go.Scatter(
                    x=stats["index"],
                    y=stats["semantic overall accuracy"],
                    mode="lines+markers",
                    name="Sem " + filename,
                )
            )
            fig_over.add_trace(
                go.Scatter(
                    x=stats["index"],
                    y=stats["syntactic overall accuracy"],
                    mode="lines+markers",
                    name="Syn " + filename,
                )
            )
            fig_over.add_trace(
                go.Scatter(
                    x=stats["index"],
                    y=stats["encoded overall accuracy"],
                    mode="lines+markers",
                    name="Enc " + filename,
                )
            )
            fig_each.add_trace(
                go.Scatter(
                    x=stats["index"],
                    y=stats["semantic accuracy"],
                    mode="lines+markers",
                    name="Sem " + filename,
                )
            )
            fig_each.add_trace(
                go.Scatter(
                    x=stats["index"],
                    y=stats["syntactic accuracy"],
                    mode="lines+markers",
                    name="Syn " + filename,
                )
            )
            fig_each.add_trace(
                go.Scatter(
                    x=stats["index"],
                    y=stats["encoded accuracy"],
                    mode="lines+markers",
                    name="Enc " + filename,
                )
            )
        # Edit the layout
        fig_over.update_layout(
            title="Pipe Overall Accuracy", xaxis_title="repeats", yaxis_title="accuracy"
        )
        fig_each.update_layout(
            title="Pipe Accuracy", xaxis_title="repeats", yaxis_title="accuracy"
        )
        return fig_over, fig_each

    def pipe_plots(self, save: bool = True, show: bool = False):
        if wandb.run is None and self.stream_to_wandb:
            raise ValueError
        fig_over, fig_each = self.create_pipe_plots(self.pipe_stats_files())
        if self.stream_to_wandb:
            wandb.log({"Pipe Overall Accuracy": fig_over, "Pipe Accuracy": fig_each})
        if show:
            fig_over.show()
            fig_each.show()
        if save:
            fig_over.write_html(os.path.join(self.eval_dir, "pipe_overall_accuracy.html"))
            fig_each.write_html(os.path.join(self.eval_dir, "pipe_accuracy.html"))

    def pipe_stats_files(self) -> List[str]:
        pipe_folders = []
        for root, _, files in os.walk(
            self.eval_dir,
        ):
            if root.find("pipe-") != -1:
                for file in files:
                    if file == "summary-stats.json":
                        pipe_folders.append(os.path.join(root, file))
        return list(set(pipe_folders))

    @staticmethod
    def pipe_statistics(summary: pandas.DataFrame, repeats: int) -> pandas.DataFrame:
        statistics_list: Dict[str, list] = {
            "index": [],
            "semantic accuracy": [],
            "syntactic accuracy": [],
            "encoded accuracy": [],
            "semantic overall accuracy": [],
            "syntactic overall accuracy": [],
            "encoded overall accuracy": [],  # this is the semantic accuracy of all samples that did not throw an encoding error
        }

        for i in range(0, repeats + 1):
            statistics_list["index"].append(
                "base model evaluation"
                if i == 0
                else "repair model evaluation iteration " + str(i)
            )
            statistics_list["semantic accuracy"].append(
                LTLRepSepHierTransformerExperiment.pipe_accuracy(summary, i, False, False)
            )
            statistics_list["syntactic accuracy"].append(
                LTLRepSepHierTransformerExperiment.pipe_accuracy(summary, i, True, False)
            )
            statistics_list["encoded accuracy"].append(
                LTLRepSepHierTransformerExperiment.pipe_accuracy(summary, i, False, True)
            )
            statistics_list["semantic overall accuracy"].append(
                LTLRepSepHierTransformerExperiment.pipe_accumulated_accuracy(
                    summary, i, False, False
                )
            )
            statistics_list["syntactic overall accuracy"].append(
                LTLRepSepHierTransformerExperiment.pipe_accumulated_accuracy(
                    summary, i, True, False
                )
            )
            statistics_list["encoded overall accuracy"].append(
                LTLRepSepHierTransformerExperiment.pipe_accumulated_accuracy(
                    summary, i, False, True
                )
            )

        return pandas.DataFrame(statistics_list).set_index("index")

    @staticmethod
    def pipe_accumulated_accuracy(
        summary: pandas.DataFrame, i: int, syn: bool = False, enc: bool = False
    ) -> float:
        if enc:
            summary_ = summary[
                (summary["status_0"] == "Violated")
                | (summary["status_0"] == "Satisfied")
                | (summary["status_0"] == "Match")
                | (summary["status_0"] == "Invalid")
            ]
        else:
            summary_ = summary
        count = 0
        for _, group in summary_.groupby(["hash"]):
            filter = (
                (group["status_" + str(0)] == "Match")
                if syn
                else (group["status_" + str(0)] == "Match")
                | (group["status_" + str(0)] == "Satisfied")
            )
            for i in range(1, i + 1):
                filter = (
                    (filter | (group["status_" + str(i)] == "Match"))
                    if syn
                    else (
                        filter
                        | (group["status_" + str(i)] == "Match")
                        | (group["status_" + str(i)] == "Satisfied")
                    )
                )
            if len(group[filter]) != 0:
                count = count + 1
        return count / len(summary_.groupby("hash"))

    @staticmethod
    def pipe_accuracy(
        summary: pandas.DataFrame, i: int, syn: bool = False, enc: bool = False
    ) -> float:
        if enc:
            summary_ = summary[
                (summary["status_0"] == "Violated")
                | (summary["status_0"] == "Satisfied")
                | (summary["status_0"] == "Match")
                | (summary["status_0"] == "Invalid")
            ]
        else:
            summary_ = summary
        count = 0
        groupby = ["hash"]
        for j in range(0, i):
            groupby.append("prediction_" + str(j))
        for _, group in summary_[summary_["status_" + str(i)].notna()].groupby(groupby):
            filter = (
                (group["status_" + str(i)] == "Match")
                if syn
                else (group["status_" + str(i)] == "Match")
                | (group["status_" + str(i)] == "Satisfied")
            )
            if len(group[filter]) != 0:
                count = count + 1
        return count / len(summary_[summary_["status_" + str(i)].notna()].groupby(groupby))

    @staticmethod
    def pipe_log_wandb_stats(
        stats: pandas.DataFrame,
        repeats: int,
        beam_base: int,
        beam_repair: int,
        data: str,
        keep: str,
    ) -> None:
        if wandb.run is None:
            raise ValueError
        logdata = {}
        repeat_list = [repeats]
        if repeats > 5:
            repeat_list.append(5)
        if repeats > 10:
            repeat_list.append(10)

        for r in repeat_list:
            logdata[
                data
                + "/pipe/sem_improvement_accuracy/r"
                + str(r)
                + "/bb"
                + str(beam_base)
                + "/br"
                + str(beam_repair)
                + "/k"
                + keep
            ] = (stats["semantic overall accuracy"][-1] - stats["semantic overall accuracy"][0])
            logdata[
                data
                + "/pipe/sem_overall_accuracy/r"
                + str(r)
                + "/bb"
                + str(beam_base)
                + "/br"
                + str(beam_repair)
                + "/k"
                + keep
            ] = stats["semantic overall accuracy"][-1]
            logdata[
                data
                + "/pipe/enc_overall_accuracy/r"
                + str(r)
                + "/bb"
                + str(beam_base)
                + "/br"
                + str(beam_repair)
                + "/k"
                + keep
            ] = stats["encoded overall accuracy"][-1]
            logdata[
                data
                + "/pipe/syn_improvement_accuracy/r"
                + str(r)
                + "/bb"
                + str(beam_base)
                + "/br"
                + str(beam_repair)
                + "/k"
                + keep
            ] = (stats["syntactic overall accuracy"][-1] - stats["syntactic overall accuracy"][0])
            logdata[
                data
                + "/pipe/syn_overall_accuracy/r"
                + str(r)
                + "/bb"
                + str(beam_base)
                + "/br"
                + str(beam_repair)
                + "/k"
                + keep
            ] = stats["syntactic overall accuracy"][-1]
        wandb.log(logdata)
        best_sem = 0
        best_syn = 0
        best_enc = 0
        for k in wandb.run.summary.keys():
            if k.find("val/pipe/sem_overall_accuracy/") != -1 and k.find("/best") == -1:
                best_sem = max(wandb.run.summary[k], best_sem)
            elif k.find("val/pipe/syn_overall_accuracy/") != -1 and k.find("/best") == -1:
                best_syn = max(wandb.run.summary[k], best_syn)
            elif k.find("val/pipe/enc_overall_accuracy/") != -1 and k.find("/best") == -1:
                best_enc = max(wandb.run.summary[k], best_enc)
        wandb.run.summary["val/pipe/sem_overall_accuracy/best"] = best_sem
        wandb.run.summary["val/pipe/syn_overall_accuracy/best"] = best_syn
        wandb.run.summary["val/pipe/enc_overall_accuracy/best"] = best_enc

    def eval(
        self,
        alphas: List[float],
        beam_sizes: List[int],
        nuxmv_port: int = None,
        data: List[str] = None,
        samples: Optional[int] = None,
        shuffle_data: bool = True,
        training: bool = False,
        identifier: str = None,
    ):
        if samples is None:
            samples = 1024
        orig_batch_size = self.batch_size

        if not data:
            data = ["test", "val"]

        if nuxmv_port:
            self._verifier = nuXmv(port=nuxmv_port)

        if shuffle_data:
            self._dataset = None
            self.shuffle_on_load = True

        for alpha in alphas:
            self.alpha = alpha
            for beam_size in beam_sizes:
                self.beam_size = beam_size
                self.batch_size = orig_batch_size // beam_size
                if samples * beam_size < self.batch_size:
                    self.batch_size = samples
                steps = samples // self.batch_size

                if "test" in data:
                    logger.info(
                        "Evaluating testset for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_split(split="test", steps=steps, training=training, verify=True)

                if "dummy_circuit" in data:
                    logger.info(
                        "Evaluating Dummy_Circuit for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_file(
                        file="dummy_circuit", steps=steps, training=training, verify=True
                    )

                if "dummy_spec" in data:
                    logger.info(
                        "Evaluating Dummy_Spec for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_file(file="dummy_spec", steps=steps, training=training, verify=False)

                if "broken" in data:
                    logger.info(
                        "Evaluating Broken for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    file = "broken-a" + str(alpha) + "-bs" + str(beam_size)
                    self.eval_file(file=file, steps=None, training=training, verify=False)

                if "train" in data:
                    logger.info(
                        "Evaluating trainingset for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_split(split="train", steps=steps, training=training, verify=True)

                if "val" in data:
                    logger.info(
                        "Evaluating validationset for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_split(split="val", steps=steps, training=training, verify=True)

                if "pipe-test" in data:
                    logger.info(
                        "Evaluating testset for %d steps with alpha %1.1f, beam_size %d and batch_size %d - Pipeline Mode. Using Identifier %s",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                        identifier if identifier else "NONE",
                    )
                    self.eval_split(
                        split="test",
                        steps=steps,
                        training=training,
                        verify=True,
                        mode="pipe",
                        identifier=identifier,
                    )

                if "pipe-val" in data:
                    logger.info(
                        "Evaluating testset for %d steps with alpha %1.1f, beam_size %d and batch_size %d - Pipeline Mode. Using Identifier %s",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                        identifier if identifier else "NONE",
                    )
                    self.eval_split(
                        split="val",
                        steps=steps,
                        training=training,
                        verify=True,
                        mode="pipe",
                        identifier=identifier,
                    )

                self._eval_model = None

    def eval_split(
        self,
        split: str,
        steps: int = None,
        training: bool = False,
        verify: bool = False,
        mode: str = "eval",
        identifier: str = None,
    ):
        generator = self.dataset.generator(splits=[split])
        self.eval_generator(
            generator=generator,
            name=split,
            includes_target=True,
            steps=steps,
            training=training,
            verify=verify,
            mode=mode,
            identifier=identifier,
        )

    def eval_directory(self, mode: str, name: str = ""):
        if mode not in ["pipe", "eval"]:
            raise ValueError
        if mode == "eval" and not name:
            raise ValueError

        # Save log / generated Data in the following manner:
        # eval -> arguments -> split -> log-no.samples.csv
        # tmp  -> result.csv

        arguments: str = f"a{self.alpha}-bs{self.beam_size}"

        dir = os.path.join(self.eval_dir, arguments, name) if mode == "eval" else self.tmp_dir
        if not os.path.isdir(dir):
            os.makedirs(dir)
        return dir

    def eval_generator(
        self,
        generator,
        name: str,
        includes_target: bool = False,
        steps: int = None,
        training: bool = False,
        verify: bool = False,
        mode: str = "eval",
        identifier: str = None,
    ):

        if mode not in ["pipe", "eval"]:
            raise ValueError
        if not verify:
            raise NotImplementedError

        # Save log / generated Data in the following manner:
        # eval -> arguments -> split -> log-no.samples.csv
        # tmp  -> result.csv

        no_samples: str = f"-n{steps * self.batch_size}" if steps else ""

        dir = self.eval_directory(mode=mode, name=name)
        logfilename = (
            "log" + no_samples if mode == "eval" else (identifier if identifier else "result")
        )
        statsfilename = (
            "stats" + no_samples
            if mode == "eval"
            else ((identifier if identifier else "result") + "-stats")
        )
        log_filepath = os.path.join(dir, logfilename + ".csv")
        stats_filepath = os.path.join(dir, statsfilename + ".json")
        logger.info("Saving files to %s", log_filepath)

        log_file = open(log_filepath, "w")
        fieldnames = (
            [
                "hash",
                "beam",
                "status",
                "problem",
                "repair_circuit",
                "prediction",
                "target",
            ]
            if mode == "eval"
            else [
                "status",
                "assumptions",
                "guarantees",
                "repair_circuit",
                "inputs",
                "outputs",
                "realizable",
                "circuit",
                "hash",
            ]
        )
        file_writer = csv.DictWriter(log_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        file_writer.writeheader()
        counters: Counter = Counter()
        pbar = tqdm(desc="Evaluated samples", unit="sample")
        problem_batch: List[Tuple[LTLSpec, str]] = []
        formula_batch: List[tf.Tensor] = []
        pos_enc_formula_batch: List[tf.Tensor] = []
        repair_circuit_batch: List[tf.Tensor] = []
        pos_enc_repair_circuit_batch: List[tf.Tensor] = []
        target_batch: List[str] = []
        hash_batch: List[str] = []

        def eval_batch():
            nonlocal counters, problem_batch, formula_batch, pos_enc_formula_batch, repair_circuit_batch, pos_enc_repair_circuit_batch, target_batch, hash_batch, pbar
            batch_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    formula_batch,
                    repair_circuit_batch,
                    pos_enc_formula_batch,
                    pos_enc_repair_circuit_batch,
                )
            )
            batch = next(iter(batch_dataset.batch(self.batch_size, drop_remainder=False)))
            predictions, _ = self.eval_model(batch, training=training)  # pylint: disable=E1102
            for i, pred in enumerate(predictions):
                any_beam_satisfied = False
                problem = problem_batch[i][0]
                repair_circuit = problem_batch[i][1]
                target = target_batch[i] if includes_target else ""
                problem_name = problem.name if problem.name else problem.formula_str
                for beam_id, beam in enumerate(pred):
                    row = {
                        "assumptions": problem.assumption_list_str,
                        "guarantees": problem.guarantee_list_str,
                        "problem": problem_name,
                        "beam": beam_id,
                        "repair_circuit_old": repair_circuit.replace("\n", "\\n"),
                        "repair_circuit_new": "",
                        "prediction": "",
                        "inputs": problem.input_str,
                        "outputs": problem.output_str,
                        # True # 'i0 i0' in target
                        "realizable": 1 if target.find("i0 i0") != -1 else 0,
                        "hash": hash_batch[i],
                    }
                    if not self.target_encoder.decode(np.array(beam)):
                        if mode == "eval":
                            row["status"] = f"Decoding Error {self.target_encoder.error}"
                            row["prediction"] = np.array2string(
                                np.array(beam), max_line_width=3 * self.max_target_length
                            )
                        else:
                            row["status"] = "Decoding Error"
                            row["prediction"] = ""
                        row_write = {key: value for key, value in row.items() if key in fieldnames}
                        row_write.update({"repair_circuit": row["repair_circuit_old"]})
                        file_writer.writerow(row_write)
                        counters["decoding_error"] += 1
                        continue
                    realizable = self.target_encoder.realizable  # True # 'i0 i0' in target
                    prediction = self.target_encoder.circuit
                    row["prediction"] = prediction.replace("\n", "\\n")
                    row["repair_circuit_new"] = prediction.replace("\n", "\\n")
                    if includes_target:
                        if prediction == target:
                            row["status"] = "Match"
                            counters["match"] += 1
                    # pylint: disable=E1102
                    result = self.verifier.model_check(problem, prediction + "\n", realizable)
                    counters[result.value] += 1
                    if result.value == "satisfied":
                        any_beam_satisfied = True
                    row["circuit"] = target.replace("\n", "\\n")
                    row["target"] = target.replace("\n", "\\n")
                    if "status" not in row:
                        row["status"] = result.value.capitalize()
                    else:
                        if row["status"] == "Match" and result.value != "satisfied":
                            logger.warning(
                                "Match not satisfied: P:"
                                + row["prediction"]
                                + " T:"
                                + row["target"]
                            )
                    row_write = {key: value for key, value in row.items() if key in fieldnames}
                    if mode == "pipe":
                        row_write["hash"] = row_write["hash"] + "," + row["repair_circuit_old"]
                    row_write.update(
                        {
                            "repair_circuit": row["repair_circuit_old"]
                            if mode == "eval"
                            else row["repair_circuit_new"]
                        }
                    )
                    file_writer.writerow(row_write)
                if any_beam_satisfied:
                    counters["beam_search_satisfied"] += 1
                pbar.update()
                pbar.set_postfix(counters)
            problem_batch = []
            formula_batch = []
            pos_enc_formula_batch = []
            repair_circuit_batch = []
            pos_enc_repair_circuit_batch = []
            target_batch = []
            hash_batch = []
            counters["steps"] += 1

        for sample in generator:
            counters["samples"] += 1
            problem = sample[0]
            problem_name = problem.name if problem.name else problem.formula_str
            repair_circuit = sample[1]
            hash = sample[2]
            target = sample[3] if includes_target else ""
            row = {
                "assumptions": problem.assumption_list_str,
                "guarantees": problem.guarantee_list_str,
                "problem": problem_name,
                "beam": 0,
                "repair_circuit": repair_circuit.replace("\n", "\\n"),
                "repair_circuit_old": repair_circuit.replace("\n", "\\n"),
                "repair_circuit_new": "",
                "prediction": "",
                "inputs": problem.input_str,
                "outputs": problem.output_str,
                # True # 'i0 i0' in target
                "realizable": 1 if target.find("i0 i0") != -1 else 0,
                "target": target.replace("\n", "\\n") if target else "",
                "hash": hash,
                "status": "",
            }
            if not self.input_encoder.encode(problem):
                if mode == "eval":
                    row["status"] = f"Encoding Error Spec{self.input_encoder.error}"
                else:
                    row["status"] = "Encoding Error Spec"
                row_write = {key: value for key, value in row.items() if key in fieldnames}
                file_writer.writerow(row_write)
                counters["encoding_error"] += 1
                pbar.update()
            elif includes_target and not self.target_encoder.encode(target):
                if mode == "eval":
                    row["status"] = f"Target Error {self.target_encoder.error}"
                else:
                    row["status"] = "Encoding Error Target"
                row_write = {key: value for key, value in row.items() if key in fieldnames}
                file_writer.writerow(row_write)
                counters["target_error"] += 1
                pbar.update()
            elif not self.circuit_encoder.encode(repair_circuit):
                if mode == "eval":
                    row["status"] = f"Encoding Error Circuit {self.circuit_encoder.error}"
                else:
                    row["status"] = "Encoding Error Repair"
                row_write = {key: value for key, value in row.items() if key in fieldnames}
                file_writer.writerow(row_write)
                counters["encoding_error"] += 1
                pbar.update()
            else:
                problem_batch.append((problem, repair_circuit))
                formula_tensor, pos_enc_formula_tensor = self.input_encoder.tensor
                formula_batch.append(formula_tensor)
                pos_enc_formula_batch.append(pos_enc_formula_tensor)
                repair_circuit_tensor = tf.expand_dims(self.circuit_encoder.tensor, axis=0)
                pos_enc_repair_circuit_tensor = pe.positional_encoding(
                    self.max_target_length, self.d_embed_enc
                )
                repair_circuit_batch.append(repair_circuit_tensor)
                pos_enc_repair_circuit_batch.append(pos_enc_repair_circuit_tensor)
                if includes_target:
                    target_batch.append(target)
                hash_batch.append(sample[2])

            if counters["samples"] % self.batch_size == 0 and problem_batch:
                eval_batch()
                if steps and counters["steps"] >= steps:
                    break

        if problem_batch:
            eval_batch()

        pbar.close()
        log_file.close()
        stats: Dict = dict(counters)
        stats["sem_accuracy"] = counters["beam_search_satisfied"] / counters["samples"]
        stats["accuracy_encoded"] = counters["beam_search_satisfied"] / (
            counters["samples"] - counters["encoding_error"]
        )
        dataframe = pandas.read_csv(log_filepath)
        dataframe = dataframe.fillna("")
        stats["beams_accuracy"] = len(
            dataframe[
                (dataframe["status"] == "Satisfied") | (dataframe["status"] == "Match")
            ].groupby("hash")
        ) / len(dataframe.groupby("hash"))
        print(log_filepath)
        print(stats["sem_accuracy"])
        print(stats["beams_accuracy"])
        if self.stream_to_wandb:
            if mode == "eval":
                wandb.log(
                    {
                        name + "/sem_accuracy/b" + str(self.beam_size): stats["sem_accuracy"],
                        name + "/beam_accuracy/b" + str(self.beam_size): stats["beams_accuracy"],
                    }
                )
        with open(stats_filepath, "w") as stats_file:
            json.dump(stats, stats_file, indent=4)

    @property
    def circuit_encoder(self):
        if not self._circuit_encoder:
            self._circuit_encoder = self.init_circuit_encoder
            if not self._circuit_encoder.load_vocabulary(self.local_dir):
                logger.info("Building circuit encoder vocabulary")
                self._circuit_encoder.build_vocabulary(self.dataset.circuit_generator())
                self._circuit_encoder.vocabulary_to_file(self.local_dir)
            logger.info("Initialized circuit encoder")
        return self._circuit_encoder

    def prepare_tf_dataset(self, tf_dataset):
        # Keras needs target as input for training for transformers
        # Wrap single input in ragged tensor for sep hierarch transformer
        # TODO an input tensor never should be empty. If so (i.e. no assumptions), a dummy input should be used

        def shape_dataset(input_tensor, circuit_tensor, target_tensor):
            ltl_tensor, tpe_tensor = input_tensor
            circuit_tensor = tf.expand_dims(circuit_tensor, axis=0)
            lpe_tensor = pe.positional_encoding(self.max_target_length, self.d_embed_enc)
            return (
                (ltl_tensor, circuit_tensor, tpe_tensor, lpe_tensor, target_tensor),
                target_tensor,
            )

        return tf_dataset.map(shape_dataset)

    @classmethod
    def cli(cls):
        parser = argparse.ArgumentParser(description="ML2 experiment")
        subparsers = parser.add_subparsers(dest="command", help="")

        train_parser = subparsers.add_parser("train", help="Training")
        cls.add_init_args(train_parser)
        cls.add_train_args(train_parser)

        eval_parser = subparsers.add_parser("eval", help="Evaluation")
        cls.add_eval_args(eval_parser)

        pipe_parser = subparsers.add_parser("pipe", help="Pipeline Evaluation")
        cls.add_pipe_args(pipe_parser)

        tune_parser = subparsers.add_parser("tune", help="Hyperparameter Tuning")
        cls.add_train_args(tune_parser)
        cls.add_tune_args(tune_parser)

        args = parser.parse_args()
        args_dict = vars(args)
        command = args_dict.pop("command")

        if command == "train":
            if args_dict["parent_name"]:
                parent_experiment = cls.load(args_dict["parent_name"])
                parent_args = parent_experiment.args
                # TODO overwrite arguments that were not specified
                parent_args.update(args_dict)
                args_dict = parent_args
            save_to_wandb = args_dict.pop("save_to_wandb")
            stream_to_wandb = args_dict.pop("stream_to_wandb")
            upload = args_dict["upload"]
            experiment = cls(**args_dict)
            experiment.run(stream_to_wandb=stream_to_wandb)
            experiment.save(
                experiment.name,
                auto_version=False,
                upload=upload,
                overwrite_bucket=upload,
                overwrite_local=True,
                add_to_wandb=save_to_wandb,
            )

        elif command == "eval":
            name = args_dict.pop("name")
            experiment = cls.load(name)
            upload = args_dict.pop("upload")
            stream_to_wandb = args_dict.pop("stream_to_wandb")
            wandb_run_id = args_dict.pop("wandb_run_id")
            if stream_to_wandb:
                if experiment.wandb_run_id is None:
                    experiment.wandb_run_id = wandb_run_id
                experiment.eval_wandb_init()
                experiment.stream_to_wandb = True
            experiment.eval(**args_dict)
            if stream_to_wandb:
                wandb.finish()
            if upload:
                cls.upload(f"{experiment.name}/eval", overwrite=True)

        elif command == "pipe":
            name = args_dict.pop("name")
            experiment = cls.load(name)
            upload = args_dict.pop("upload")
            stream_to_wandb = args_dict.pop("stream_to_wandb")
            wandb_run_id = args_dict.pop("wandb_run_id")
            if stream_to_wandb:
                if experiment.wandb_run_id is None:
                    experiment.wandb_run_id = wandb_run_id
                experiment.eval_wandb_init()
                experiment.stream_to_wandb = True
            experiment.eval_pipeline(**args_dict)
            if stream_to_wandb:
                wandb.finish()
            if upload:
                cls.upload(f"{experiment.name}/eval", overwrite=True)

        else:
            raise Exception("Unknown command %s", args.command)

    @classmethod
    def add_pipe_args(cls, parser):
        parser.add_argument("-n", "--name", required=True)
        parser.add_argument("--base-model", required=True)
        parser.add_argument("--no-shuffle-data", action="store_false", dest="shuffle_data")
        parser.add_argument("-u", "--upload", action="store_true")
        parser.add_argument("--wandb-run-id", dest="wandb_run_id", default=None)
        parser.add_argument("--stream-to-wandb", action="store_true")
        parser.add_argument("--alpha-base", type=int, default=0.5)
        parser.add_argument("--beam-base", type=int, default=4)
        parser.add_argument("--alpha-repair", type=int, default=0.5)
        parser.add_argument("--beam-repair", type=int, default=4)
        parser.add_argument("--repeats", type=int, default=2)
        parser.add_argument("--samples", type=int, default=1024)
        parser.add_argument("-d", "--data", nargs="*", default=None)
        parser.add_argument("--keep", default=None)
        parser.add_argument("--reference-alphas", dest="reference_alphas", nargs="*", default=None)
        parser.add_argument("--reference-beams", dest="reference_beams", nargs="*", default=None)

    @classmethod
    def add_init_args(cls, parser):
        super().add_init_args(parser)
        defaults = cls.get_default_args()
        parser.add_argument(
            "--constant-learning-rate", type=float, default=defaults["constant_learning_rate"]
        )
        parser.add_argument("--d-embed", type=int, default=defaults["d_embed"])
        parser.add_argument("--d-embed-enc", type=int, default=defaults["d_embed_enc"])
        parser.add_argument("--d-embed-dec", type=int, default=defaults["d_embed_dec"])
        parser.add_argument("--d-ff", type=int, default=defaults["d_ff"])
        parser.add_argument("--d-ff-enc-g", type=int, default=defaults["d_ff_enc_g"])
        parser.add_argument("--d-ff-enc-l1", type=int, default=defaults["d_ff_enc_l1"])
        parser.add_argument("--d-ff-enc-l2", type=int, default=defaults["d_ff_enc_l2"])
        parser.add_argument("--d-ff-dec", type=int, default=defaults["d_ff_dec"])
        parser.add_argument("--dropout", type=float, default=defaults["dropout"])
        parser.add_argument("--dropout-enc", type=float, default=defaults["dropout_enc"])
        parser.add_argument("--dropout-dec", type=float, default=defaults["dropout_dec"])
        parser.add_argument(
            "--ff-activation-enc-g", type=str, default=defaults["ff_activation_enc_g"]
        )
        parser.add_argument(
            "--ff-activation-enc-l1", type=str, default=defaults["ff_activation_enc_l1"]
        )
        parser.add_argument(
            "--ff-activation-enc-l2", type=str, default=defaults["ff_activation_enc_l2"]
        )
        parser.add_argument("--ff-activation-dec", type=str, default=defaults["ff_activation_dec"])
        parser.add_argument("--fix-local-embed", action="store_true")
        parser.add_argument("--num-heads", type=int, default=defaults["num_heads"])
        parser.add_argument("--num-heads-enc-g", type=int, default=defaults["num_heads_enc_g"])
        parser.add_argument("--num-heads-enc-l1", type=int, default=defaults["num_heads_enc_l1"])
        parser.add_argument("--num-heads-enc-l2", type=int, default=defaults["num_heads_enc_l2"])
        parser.add_argument("--num-heads-dec", type=int, default=defaults["num_heads_dec"])
        parser.add_argument("--num-layers", type=int, default=defaults["num_layers"])
        parser.add_argument("--num-layers-enc-g", type=int, default=defaults["num_layers_enc_g"])
        parser.add_argument("--num-layers-enc-l1", type=int, default=defaults["num_layers_enc_l1"])
        parser.add_argument("--num-layers-enc-l2", type=int, default=defaults["num_layers_enc_l2"])
        parser.add_argument("--num-layers-dec", type=int, default=defaults["num_layers_dec"])
        parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])
        parser.add_argument("--ttot-learning", action="store_true", dest="ttot_learning")

    @classmethod
    def add_tune_args(cls, parser):
        parser.add_argument("--d-embed", nargs="*", default=[64, 256])
        parser.add_argument("--d-ff", nargs="*", default=[64, 256, 1024])
        parser.add_argument("-n", "--name", default="ht-tune")
        parser.add_argument("--num-layers", nargs="*", default=[4, 8])
        parser.add_argument("--num-heads", nargs="*", default=[4, 8])


if __name__ == "__main__":
    LTLRepSepHierTransformerExperiment.cli()
