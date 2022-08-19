"""Repair Data Generation by evaluating a trained LTL-Synthesis Model"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import sys
import tensorflow as tf
import json
import os
import csv
from tqdm import tqdm
from collections import Counter
from ml2.ltl.ltl_spec.ltl_spec_data import LTLSpecData

from ml2.tools.nuxmv.nuxmv import nuXmv

from ... import models

from ...data import TPEFormat
from ...data import ExprNotation
from ...optimization import lr_schedules
from ..ltl_spec import LTLSpecPropertyEncoder
from ..ltl_syn.ltl_syn_experiment import LTLSynExperiment
from .ltl_syn_data import LTLSynSplitData
from ..ltl_spec import LTLSpec
from ...globals import LTL_REP_BUCKET_DIR, LTL_REP_WANDB_PROJECT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LTLRepairGenData(LTLSynExperiment):

    BUCKET_DIR = LTL_REP_BUCKET_DIR
    WANDB_PROJECT = LTL_REP_WANDB_PROJECT

    def __init__(
        self,
        constant_learning_rate: float = None,
        custom_pos_enc: bool = True,
        d_embed: int = 256,
        d_embed_enc: int = None,
        d_embed_dec: int = None,
        d_ff: int = 1024,
        d_ff_enc_d0: int = None,
        d_ff_enc_d1: int = None,
        d_ff_dec: int = None,
        dropout: float = 0.0,
        dropout_enc: float = None,
        dropout_dec: float = None,
        ff_activation_enc_d0: str = "relu",
        ff_activation_enc_d1: str = "relu",
        ff_activation_dec: str = "relu",
        fix_d1_embed: bool = False,
        name: str = "repair-data",
        num_properties: int = 12,
        num_heads: int = 4,
        num_heads_enc_d0: int = None,
        num_heads_enc_d1: int = None,
        num_heads_dec: int = None,
        num_layers: int = 8,
        num_layers_enc_d0: int = None,
        num_layers_enc_d1: int = None,
        num_layers_dec: int = None,
        property_tree_size: int = 25,
        warmup_steps: int = 4000,
        **kwargs,
    ) -> None:
        self.constant_learning_rate = constant_learning_rate
        self.custom_pos_enc = custom_pos_enc
        if not custom_pos_enc:
            raise NotImplementedError
        self.d_embed_enc = d_embed_enc if d_embed_enc else d_embed
        self.d_embed_dec = d_embed_dec if d_embed_dec else d_embed
        self.d_ff_enc_d0 = d_ff_enc_d0 if d_ff_enc_d0 else d_ff
        self.d_ff_enc_d1 = d_ff_enc_d1 if d_ff_enc_d1 else d_ff
        self.d_ff_dec = d_ff_dec if d_ff_dec else d_ff
        self.dropout_enc = dropout_enc if dropout_enc else dropout
        self.dropout_dec = dropout_dec if dropout_dec else dropout
        self.ff_activation_enc_d0 = ff_activation_enc_d0
        self.ff_activation_enc_d1 = ff_activation_enc_d1
        self.ff_activation_dec = ff_activation_dec
        self.fix_d1_embed = fix_d1_embed
        self.property_tree_size = property_tree_size
        self.num_properties = num_properties
        self.num_heads_enc_d0 = num_heads_enc_d0 if num_heads_enc_d0 else num_heads
        self.num_heads_enc_d1 = num_heads_enc_d1 if num_heads_enc_d1 else num_heads
        self.num_heads_dec = num_heads_dec if num_heads_dec else num_heads
        self.num_layers_enc_d0 = num_layers_enc_d0 if num_layers_enc_d0 else num_layers // 2
        self.num_layers_enc_d1 = num_layers_enc_d1 if num_layers_enc_d1 else num_layers // 2
        self.num_layers_dec = num_layers_dec if num_layers_dec else num_layers
        self.warmup_steps = warmup_steps
        if self.d_embed_enc % self.num_heads_enc_d0 != 0:
            sys.exit(
                f"Encoder embedding dimension {self.d_embed_enc} is "
                "not divisible by the number of attention heads"
                f"{self.num_heads_enc_d0}"
            )
        if self.d_embed_enc % self.num_heads_enc_d1 != 0:
            sys.exit(
                f"Encoder embedding dimension {self.d_embed_enc} is "
                "not divisible by the number of attention heads"
                f"{self.num_heads_enc_d1}"
            )
        if self.d_embed_dec % self.num_heads_dec != 0:
            sys.exit(
                (
                    f"Decoder embedding dimension {self.d_embed_dec} is "
                    "not divisible by the number of attention heads "
                    f"{self.num_heads_dec}"
                )
            )
        super().__init__(name=name, **kwargs)

    @property
    def init_input_encoder(self) -> LTLSpecPropertyEncoder:
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
    def init_learning_rate(self) -> Any:
        if self.constant_learning_rate:
            return self.constant_learning_rate
        return lr_schedules.TransformerSchedule(self.d_embed_enc, warmup_steps=self.warmup_steps)

    @property
    def init_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

    def init_model(self, training: bool = True, attn_weights: bool = False) -> tf.keras.Model:
        args = self.args
        args["input_vocab_size"] = self.input_vocab_size
        args["input_eos_id"] = self.input_eos_id
        args["input_pad_id"] = self.input_pad_id
        args["target_vocab_size"] = self.target_vocab_size
        args["target_start_id"] = self.target_start_id
        args["target_eos_id"] = self.target_eos_id
        args["target_pad_id"] = self.target_pad_id
        args["input_length"] = (self.num_properties, self.property_tree_size)
        args["max_decode_length"] = self.max_target_length
        return models.hierarchical_transformer_2d.create_model(
            args, training=training, custom_pos_enc=self.custom_pos_enc, attn_weights=attn_weights
        )

    @property
    def init_dataset(self):
        return LTLSynSplitData.load(self.dataset_name)

    def prepare_tf_dataset(self, tf_dataset: tf.data.Dataset):
        def shape_dataset(input_tensor, target_tensor):
            if self.custom_pos_enc:
                ltl_tensor, tpe_tensor = input_tensor
                return ((ltl_tensor, tpe_tensor, target_tensor), target_tensor)
            return ((input_tensor, target_tensor), target_tensor)

        return tf_dataset.map(shape_dataset)

    def eval(
        self,
        alphas: List[float],
        beam_sizes: List[int],
        nuxmv_port: int = None,
        data: List[str] = None,
        samples: Optional[int] = 1024,
        shuffle_data: bool = True,
        training: bool = False,
        identifier: str = None,
        mode: Optional[str] = None,
    ):
        orig_batch_size = self.batch_size

        if not data:
            data = ["test", "syntcomp", "jarvis", "timeouts"]

        if nuxmv_port:
            self._verifier = nuXmv(port=nuxmv_port)

        if "generate" in data:
            samples = None

        if shuffle_data:
            self._dataset = None
            self.shuffle_on_load = True

        for alpha in alphas:
            self.alpha = alpha
            for beam_size in beam_sizes:
                self.beam_size = beam_size
                self.batch_size = orig_batch_size // beam_size
                if samples:
                    if samples * beam_size < self.batch_size:
                        self.batch_size = samples
                    steps = samples // self.batch_size

                if "test" in data:
                    if mode == "pipe":
                        logger.info(
                            "Evaluating testset for %d steps with alpha %1.1f, beam_size %d and batch_size %d - Pipeline Mode. Using Identifier %s",
                            steps,
                            self.alpha,
                            self.beam_size,
                            self.batch_size,
                            identifier if identifier else "NONE",
                        )
                    else:
                        logger.info(
                            "Evaluating testset for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                            steps,
                            self.alpha,
                            self.beam_size,
                            self.batch_size,
                        )
                    self.eval_split(
                        split="test",
                        steps=steps,
                        training=training,
                        verify=True,
                        identifier=identifier,
                        mode=mode,
                    )
                if "val" in data:
                    if mode == "pipe":
                        logger.info(
                            "Evaluating valset for %d steps with alpha %1.1f, beam_size %d and batch_size %d - Pipeline Mode. Using Identifier %s",
                            steps,
                            self.alpha,
                            self.beam_size,
                            self.batch_size,
                            identifier if identifier else "NONE",
                        )
                    else:
                        logger.info(
                            "Evaluating valset for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                            steps,
                            self.alpha,
                            self.beam_size,
                            self.batch_size,
                        )
                    self.eval_split(
                        split="val",
                        steps=steps,
                        training=training,
                        verify=True,
                        identifier=identifier,
                        mode=mode,
                    )

                if "test_fixed" in data:
                    if mode == "pipe":
                        logger.info(
                            "Evaluating test set with fixed samples for %d steps with alpha %1.1f, beam_size %d and batch_size %d - Pipeline Mode. Using Identifier %s",
                            steps,
                            self.alpha,
                            self.beam_size,
                            self.batch_size,
                            identifier if identifier else "NONE",
                        )
                    else:
                        logger.info(
                            "Evaluating test set with fixed samples for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                            steps,
                            self.alpha,
                            self.beam_size,
                            self.batch_size,
                        )
                    self.eval_split(
                        split="test_fixed",
                        steps=steps,
                        training=training,
                        verify=True,
                        identifier=identifier,
                        mode=mode,
                    )

                if "generate" in data:
                    logger.info(
                        "Generating Repair Data with alpha %1.1f, beam_size %d and batch_size %d",
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    print("generating train data")
                    self.eval_split(split="train", training=training, verify=True, mode="create")
                    print("generating test data")
                    self.eval_split(split="test", training=training, verify=True, mode="create")
                    print("generating val data")
                    self.eval_split(split="val", training=training, verify=True, mode="create")
                    print("constructing dataset")

                if "syntcomp" in data:
                    logger.info(
                        "Evaluating SYNTCOMP 2020 with alpha %1.1f, beam_size %d and batch_size %d",
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_ltl_specs(
                        "sc-0", training=training, identifier=identifier, mode=mode, steps=steps
                    )

                if "jarvis" in data:
                    logger.info(
                        "Evaluating smart home benchmarks with alpha %1.1f, beam_size %d and batch_size %d",
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_ltl_specs(
                        "jarvis-0",
                        training=training,
                        identifier=identifier,
                        mode=mode,
                        steps=steps,
                    )

                if "timeouts" in data:
                    logger.info(
                        "Evaluating timeouts for %d steps with alpha %1.1f, beam_size %d and batch_size %d",
                        steps,
                        self.alpha,
                        self.beam_size,
                        self.batch_size,
                    )
                    self.eval_timeouts(
                        steps=steps, training=training, identifier=identifier, mode=mode
                    )

                self._eval_model = None

    def eval_split(
        self,
        split: str,
        steps: int = None,
        training: bool = False,
        verify: bool = False,
        mode: Optional[str] = None,
        identifier: str = None,
    ):
        if mode is None:
            mode = "eval"
        generator = self.dataset.generator(splits=[split])
        self.eval_generator(
            generator,
            split,
            includes_target=True,
            steps=steps,
            training=training,
            verify=verify,
            mode=mode,
            identifier=identifier,
        )

    def eval_timeouts(
        self,
        steps: int = None,
        training: bool = False,
        mode: Optional[str] = None,
        identifier: str = None,
    ):
        if mode is None:
            mode = "eval"
        timeouts = self.dataset["timeouts"]
        timeouts = [(sample, hash) for sample, _, hash in timeouts.generator()]
        self.eval_generator(
            timeouts,
            "timeouts",
            includes_target=False,
            steps=steps,
            training=training,
            verify=True,
            mode=mode,
            identifier=identifier,
        )

    def eval_ltl_specs(
        self,
        name: str,
        steps: int = None,
        training: bool = False,
        mode: Optional[str] = None,
        identifier: str = None,
    ):
        def spec_filter(spec):
            return spec.num_inputs <= len(self.inputs) and spec.num_outputs <= len(self.outputs)

        if mode is None:
            mode = "eval"

        LTLSpecData.download(name)
        spec_ds = LTLSpecData.from_bosy_files(LTLSpecData.local_path(name), spec_filter)
        spec_ds.rename_aps(self.inputs, self.outputs)
        for spec in spec_ds.dataset:
            spec.inputs = self.inputs
            spec.outputs = self.outputs

        self.eval_generator(
            spec_ds.generator(),
            name,
            includes_target=False,
            steps=steps,
            training=training,
            verify=True,
            mode=mode,
            identifier=identifier,
        )

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
        if mode not in ["create", "pipe", "eval"]:
            raise ValueError
        if not verify:
            raise NotImplementedError

        # Save log / generated Data in the following manner:
        # eval -> arguments -> split -> log-no.samples.csv
        # gen  -> arguments -> split.csv
        # tmp  -> result.csv

        arguments: str = f"a{self.alpha}-bs{self.beam_size}"
        no_samples: str = f"-n{steps * self.batch_size}" if steps else ""
        dir = (
            os.path.join(self.eval_dir, arguments, name)
            if mode == "eval"
            else (os.path.join(self.gen_dir, arguments) if mode == "create" else self.tmp_dir)
        )
        if not os.path.isdir(dir):
            os.makedirs(dir)
        logfilename = (
            "log" + no_samples
            if mode == "eval"
            else (name if mode == "create" else (identifier if identifier else "result"))
        )
        statsfilename = (
            "stats" + no_samples
            if mode == "eval"
            else (
                name + "-stats"
                if mode == "create"
                else (identifier if identifier else "result") + "-stats"
            )
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
        problem_batch: List[LTLSpec] = []
        formula_batch: List[tf.Tensor] = []
        pos_enc_batch: List[tf.Tensor] = []
        target_batch: List[str] = []
        hash_batch: List[str] = []

        def eval_batch():
            nonlocal counters, problem_batch, formula_batch, pos_enc_batch, target_batch, hash_batch, pbar
            batch_dataset = tf.data.Dataset.from_tensor_slices((formula_batch, pos_enc_batch))
            batch = next(iter(batch_dataset.batch(self.batch_size, drop_remainder=False)))
            predictions, _ = self.eval_model(batch, training=training)
            for i, pred in enumerate(predictions):
                any_beam_satisfied = False
                problem: LTLSpec = problem_batch[i]
                target = target_batch[i] if includes_target else ""
                problem_name = problem.name if problem.name else problem.formula_str
                for beam_id, beam in enumerate(pred):
                    row = {
                        "assumptions": problem.assumption_list_str,
                        "guarantees": problem.guarantee_list_str,
                        "problem": problem_name,
                        "beam": beam_id,
                        "repair_circuit": "",
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
                            file_writer.writerow(
                                {key: value for key, value in row.items() if key in fieldnames}
                            )
                        else:
                            row["status"] = "Decoding Error"
                            row["prediction"] = ""
                        counters["decoding_error"] += 1
                        continue
                    realizable = self.target_encoder.realizable  # True # 'i0 i0' in target
                    circuit = self.target_encoder.circuit
                    row["repair_circuit"] = circuit.replace("\n", "\\n")
                    row["prediction"] = circuit.replace("\n", "\\n")
                    if includes_target and circuit == target:
                        row["status"] = "Match"
                        counters["match"] += 1
                    result = self.verifier.model_check(problem, circuit + "\n", realizable)
                    if result.value == "satisfied":
                        any_beam_satisfied = True
                    counters[result.value] += 1
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
                    file_writer.writerow(
                        {key: value for key, value in row.items() if key in fieldnames}
                    )
                if any_beam_satisfied:
                    counters["beam_search_satisfied"] += 1
                pbar.update()
                pbar.set_postfix(counters)
            problem_batch = []
            formula_batch = []
            pos_enc_batch = []
            target_batch = []
            hash_batch = []
            counters["steps"] += 1

        for sample in generator:
            counters["samples"] += 1
            problem: LTLSpec = sample[0]
            problem_name = problem.name if problem.name else problem.formula_str
            target = sample[1] if includes_target else ""
            hash = sample[2] if includes_target else sample[1]
            row = {
                "assumptions": problem.assumption_list_str,
                "guarantees": problem.guarantee_list_str,
                "problem": problem_name,
                "beam": 0,
                "repair_circuit": "",
                "prediction": "",
                "inputs": problem.input_str,
                "outputs": problem.output_str,
                # True # 'i0 i0' in target
                "realizable": 1 if target.find("i0 i0") != -1 else 0,
                "hash": hash,
                "target": target.replace("\n", "\\n") if target else "",
            }
            if not self.input_encoder.encode(problem):
                # input encoder error
                if mode == "eval":
                    row["status"] = f"Encoding Error {self.input_encoder.error}"
                    file_writer.writerow(
                        {key: value for key, value in row.items() if key in fieldnames}
                    )
                else:
                    row["status"] = "Encoding Error"
                    file_writer.writerow(
                        {key: value for key, value in row.items() if key in fieldnames}
                    )
                counters["encoding_error"] += 1
                pbar.update()
            elif includes_target and not self.target_encoder.encode(target):
                # target encoder error
                if mode == "eval":
                    row["status"] = f"Target Error {self.target_encoder.error}"
                    file_writer.writerow(
                        {key: value for key, value in row.items() if key in fieldnames}
                    )
                else:
                    row["status"] = "Target Error"
                    file_writer.writerow(
                        {key: value for key, value in row.items() if key in fieldnames}
                    )
                counters["target_error"] += 1
                pbar.update()
            else:
                problem_batch.append(problem)
                formula_tensor, pos_enc_tensor = self.input_encoder.tensor
                formula_batch.append(formula_tensor)
                pos_enc_batch.append(pos_enc_tensor)
                target_batch.append(sample[1] if includes_target else "")
                hash_batch.append(sample[2] if includes_target else sample[1])

            if counters["samples"] % self.batch_size == 0 and problem_batch:
                eval_batch()
                if steps and counters["steps"] >= steps:
                    break

        if problem_batch:
            eval_batch()

        pbar.close()
        log_file.close()
        stats: Dict = dict(counters)
        stats["accuracy"] = counters["beam_search_satisfied"] / counters["samples"]
        stats["accuracy_encoded"] = counters["beam_search_satisfied"] / (
            counters["samples"] - counters["encoding_error"]
        )
        with open(stats_filepath, "w") as stats_file:
            json.dump(stats, stats_file, indent=4)

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
        parser.add_argument("--d-ff-enc-d0", type=int, default=defaults["d_ff_enc_d0"])
        parser.add_argument("--d-ff-enc-d1", type=int, default=defaults["d_ff_enc_d1"])
        parser.add_argument("--d-ff-dec", type=int, default=defaults["d_ff_dec"])
        parser.add_argument("--dropout", type=float, default=defaults["dropout"])
        parser.add_argument("--dropout-enc", type=float, default=defaults["dropout_enc"])
        parser.add_argument("--dropout-dec", type=float, default=defaults["dropout_dec"])
        parser.add_argument(
            "--ff-activation-enc-d0", type=str, default=defaults["ff_activation_enc_d0"]
        )
        parser.add_argument(
            "--ff-activation-enc-d1", type=str, default=defaults["ff_activation_enc_d1"]
        )
        parser.add_argument("--ff-activation-dec", type=str, default=defaults["ff_activation_dec"])
        parser.add_argument("--fix-d1-embed", action="store_true")
        parser.add_argument("--num-heads", type=int, default=defaults["num_heads"])
        parser.add_argument("--num-heads-enc-d0", type=int, default=defaults["num_heads_enc_d0"])
        parser.add_argument("--num-heads-enc-d1", type=int, default=defaults["num_heads_enc_d1"])
        parser.add_argument("--num-heads-dec", type=int, default=defaults["num_heads_dec"])
        parser.add_argument("--num-layers", type=int, default=defaults["num_layers"])
        parser.add_argument("--num-layers-enc-d0", type=int, default=defaults["num_layers_enc_d0"])
        parser.add_argument("--num-layers-enc-d1", type=int, default=defaults["num_layers_enc_d1"])
        parser.add_argument("--num-layers-dec", type=int, default=defaults["num_layers_dec"])
        parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])

    @classmethod
    def add_tune_args(cls, parser):
        parser.add_argument("--d-embed", nargs="*", default=[64, 256])
        parser.add_argument("--d-ff", nargs="*", default=[64, 256, 1024])
        parser.add_argument("-n", "--name", default="ht-tune")
        parser.add_argument("--num-layers", nargs="*", default=[4, 8])
        parser.add_argument("--num-heads", nargs="*", default=[4, 8])


if __name__ == "__main__":
    LTLRepairGenData.cli()
