"""AIGER circuit class based on https://github.com/mvcisback/py-aiger/blob/main/aiger/parser.py"""

import logging
import re
from typing import Callable, List
from scipy.stats import truncnorm, uniform
import numpy as np

from ml2.ltl.ltl_syn.ltl_syn_data import NUM_INPUTS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Header:
    def __init__(
        self, max_var_id: int, num_inputs: int, num_latches: int, num_outputs: int, num_ands: int
    ):
        self.max_var_id = max_var_id
        self.num_inputs = num_inputs
        self.num_latches = num_latches
        self.num_outputs = num_outputs
        self.num_ands = num_ands

    def __str__(self):
        return (
            f"aag {self.max_var_id} {self.num_inputs} {self.num_latches} "
            f"{self.num_outputs} {self.num_ands}"
        )


class Latch:
    def __init__(self, lit: int, next_lit: int):
        self.lit = lit
        self.next_lit = next_lit

    def __str__(self):
        return f"{self.lit} {self.next_lit}"

    def change(self, index: int, f_change: Callable[[int], int]) -> int:
        if index == 0:
            new_var = f_change(self.lit)
            self.lit = new_var
            return new_var
        elif index == 1:
            new_var = f_change(self.next_lit)
            self.next_lit = new_var
            return new_var
        else:
            raise ValueError


class And:
    def __init__(self, lit: int, arg1: int, arg2: int):
        self.lit = lit
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return f"{self.lit} {self.arg1} {self.arg2}"

    def change(self, index: int, f_change: Callable[[int], int]) -> int:
        if index == 0:
            new_var = f_change(self.lit)
            self.lit = new_var
            return new_var
        elif index == 1:
            new_var = f_change(self.arg1)
            self.arg1 = new_var
            return new_var
        elif index == 2:
            new_var = f_change(self.arg2)
            self.arg2 = new_var
            return new_var
        else:
            raise ValueError


class Symbol:
    def __init__(
        self,
        kind: str,
        idx: int,
        name: str,
    ):
        self.kind = kind
        self.idx = idx
        self.name = name

    def __str__(self):
        return f"{self.kind}{self.idx} {self.name}"


class Circuit:
    def __init__(
        self,
        header=None,
        inputs=None,
        latches=None,
        outputs=None,
        ands=None,
        symbols=None,
        comments=None,
    ):
        self.header: Header = header
        self.inputs: List[int] = inputs if inputs else []
        self.latches: List[Latch] = latches if latches else []
        self.outputs: List[int] = outputs if outputs else []
        self.ands: List[And] = ands if ands else []
        self.symbols = symbols if symbols else []
        self.comments = comments if comments else []

    @property
    def max_var_id(self):
        lit = 0
        if self.inputs:
            lit = max(self.inputs)
        components = self.latches + self.ands
        if components:
            lit = max(lit, max([x.lit for x in components]))
        return lit // 2

    @property
    def num_inputs(self):
        return len(self.inputs)

    @property
    def num_latches(self):
        return len(self.latches)

    @property
    def num_outputs(self):
        return len(self.outputs)

    @property
    def num_ands(self):
        return len(self.ands)

    @property
    def input_var_ids(self):
        return [i // 2 for i in self.inputs]

    @property
    def latch_var_ids(self):
        return [l.lit // 2 for l in self.latches]

    @property
    def output_var_ids(self):
        return [o // 2 for o in self.outputs]

    @property
    def and_var_ids(self):
        return [a.lit // 2 for a in self.ands]

    @property
    def num_possible_deletes(self):
        return self.num_ands + self.num_latches

    @property
    def size(self):
        return self.num_latches + self.num_ands

    @property
    def num_vars(self):
        return self.num_inputs + self.num_outputs + (self.num_ands * 3) + (self.num_latches * 2)

    def change_input(self, index: int, f_change: Callable[[int], int]) -> None:
        if index >= self.num_inputs:
            raise ValueError
        new_var = f_change(self.inputs[index])
        self.inputs[index] = new_var
        self.header.max_var_id = max(new_var // 2, self.header.max_var_id)

    def change_output(self, index: int, f_change: Callable[[int], int]) -> None:
        if index >= self.num_outputs:
            raise ValueError
        new_var = f_change(self.outputs[index])
        self.outputs[index] = new_var
        self.header.max_var_id = max(new_var // 2, self.header.max_var_id)

    def get_latch_by_idx(self, idx):
        for latch in self.latches:
            if latch.lit // 2 == idx:
                return latch
        return None

    def alter_variable(self, min_var: int, max_var: int, range_68_var: int) -> None:
        i = self.uniform(lower=0, upper=self.num_vars)

        def f_change(x: int) -> int:
            return self.sample_new_var(x, min_var=min_var, max_var=max_var, range_68=range_68_var)

        if i < self.num_inputs:  # input
            self.change_input(i, f_change=f_change)
        elif i < (self.num_inputs + (self.num_latches * 2)):  # latch
            i, j = divmod((i - self.num_inputs), 2)
            new_var = self.latches[i].change(j, f_change=f_change)
            self.header.max_var_id = max(new_var // 2, self.header.max_var_id)
        elif i < (self.num_inputs + (self.num_latches * 2) + self.num_outputs):  # output
            i = i - self.num_inputs - (self.num_latches * 2)
            self.change_output(i, f_change=f_change)
        elif i < (
            self.num_inputs
            + (self.num_latches * 2)
            + self.num_outputs
            + (self.num_ands * 3)  # and
        ):
            i, j = divmod((i - self.num_inputs - (self.num_latches * 2) - self.num_outputs), 3)
            new_var = self.ands[i].change(j, f_change=f_change)
            self.header.max_var_id = max(new_var // 2, self.header.max_var_id)
        else:
            raise ValueError

    def remove_line(self) -> None:
        i = self.uniform(lower=0, upper=self.num_possible_deletes)
        if i < (self.num_latches):  # latch
            self.latches = self.latches[:i] + self.latches[(i + 1) :]
            self.header.num_latches = self.num_latches
        elif i < (self.num_latches + self.num_ands):  # and
            i = i - self.num_latches
            self.ands = self.ands[:i] + self.ands[(i + 1) :]
            self.header.num_ands = self.num_ands
        else:
            raise ValueError
        pass

    def __str__(self):
        return "\n".join(
            [
                str(x)
                for x in [
                    self.header,
                    *self.inputs,
                    *self.latches,
                    *self.outputs,
                    *self.ands,
                    *self.symbols,
                    *self.comments,
                ]
            ]
        )

    @staticmethod
    def uniform(lower: int, upper: int) -> int:
        if lower >= upper:
            raise ValueError(
                "Distribution empty: lower bound %d is not smaller than upper bound %d (not included)"
                % (lower, upper)
            )
        while True:
            sample = uniform.rvs(loc=lower - 1, scale=upper + 1)
            var = np.round(sample).astype(int)
            if var >= lower and var <= upper - 1:
                return var

    @staticmethod
    def sample_amount_changes(
        max_changes: int = 50, min_changes: int = 1, range_68: int = 15
    ) -> int:
        if min_changes > max_changes:
            raise ValueError(
                "Distribution empty: no values between min changes %d and mac changes %d"
                % (min_changes, max_changes)
            )
        std = np.ceil(range_68 / 2)
        b = (max_changes + 1 - min_changes) / std
        while True:
            sample = truncnorm.rvs(-1 / std, b, loc=min_changes, scale=std, size=1)
            amount = np.round(sample).astype(int)[0]
            if amount >= min_changes and amount <= max_changes:
                return amount

    @staticmethod
    def sample_new_var(
        old_var: int,
        min_var: int = 0,
        max_var: int = 61,
        range_68: int = 20,
    ) -> int:
        if old_var not in range(min_var, max_var + 1):
            raise ValueError(
                "Distribution empty: no values between min var %d and max var %d except maybe the old variable value %d"
                % (min_var, max_var, old_var)
            )
        range_68 = min(range_68, max_var - min_var)
        std = np.ceil(range_68 / 2)
        a, b = (min_var - 1 - old_var) / std, (max_var + 1 - old_var) / std
        new_var = old_var
        while True:
            sample = truncnorm.rvs(a, b, loc=old_var, scale=std, size=1)
            new_var = np.round(sample).astype(int)[0]
            if new_var != old_var and new_var >= min_var and new_var <= max_var:
                return new_var

    def alter(
        self,
        max_changes: int = 50,
        min_changes: int = 1,
        range_68_changes: int = 15,
        min_var: int = 0,
        max_var: int = 61,
        range_68_var: int = 20,
        fraction_delete: float = 0.2,
        **_,
    ) -> None:
        for _ in range(
            self.sample_amount_changes(
                max_changes=max_changes, min_changes=min_changes, range_68=range_68_changes
            )
        ):
            if self.num_possible_deletes:
                if uniform().rvs() < fraction_delete:
                    self.remove_line()
                else:
                    self.alter_variable(
                        min_var=min_var, max_var=max_var, range_68_var=range_68_var
                    )
            else:
                self.alter_variable(min_var=min_var, max_var=max_var, range_68_var=range_68_var)


HEADER_PATTERN = re.compile(r"aag (\d+) (\d+) (\d+) (\d+) (\d+)")


def parse_header(line, state):
    if state.header:
        return False
    match = HEADER_PATTERN.fullmatch(line)
    if not match:
        message = f"Failed to parse aag header: {line}"
        logger.critical(message)
        raise ValueError(message)

    try:
        ids = [int(idx) for idx in match.groups()]

        if any(x < 0 for x in ids):
            message = "Indicies must be positive"
            logger.critical(message)
            raise ValueError(message)

        max_var_id, num_inputs, num_latches, num_outputs, num_ands = ids
        if num_inputs + num_latches + num_ands > max_var_id:
            message = (
                "Sum of number of inputs, latches and ands is greater than max variable index"
            )
            logger.critical(message + "error suppressed")
            # raise ValueError(message)

        state.header = Header(max_var_id, num_inputs, num_latches, num_outputs, num_ands)

    except ValueError as exc:
        raise ValueError("Failed to parse aag header") from exc
    return True


IO_PATTERN = re.compile(r"(\d+)")


def parse_input(line, state):
    match = IO_PATTERN.fullmatch(line)
    if not match or state.num_inputs >= state.header.num_inputs:
        return False
    lit = int(line)
    state.inputs.append(lit)
    return True


def parse_output(line, state):
    match = IO_PATTERN.fullmatch(line)
    if not match or state.num_outputs >= state.header.num_outputs:
        return False
    lit = int(line)
    state.outputs.append(lit)
    return True


LATCH_PATTERN = re.compile(r"(\d+) (\d+)")


def parse_latch(line, state):
    if state.header.num_latches and state.num_latches >= state.header.num_latches:
        return False

    match = LATCH_PATTERN.fullmatch(line)
    if not match:
        if state.header.num_latches:
            message = f"Expecting a latch: {line}"
            logger.critical(message)
            raise ValueError(message)
        return False

    groups = match.groups()
    lit = int(groups[0])
    next_lit = int(groups[1])

    state.latches.append(Latch(lit, next_lit))
    return True


AND_PATTERN = re.compile(r"(\d+) (\d+) (\d+)")


def parse_and(line, state):
    if state.header.num_ands and state.num_ands >= state.header.num_ands:
        return False

    match = AND_PATTERN.fullmatch(line)
    if not match:
        if state.header.num_ands:
            message = f"Expecting an and: {line}"
            logger.critical(message)
            raise ValueError(message)
        return False

    groups = match.groups()
    lit = int(groups[0])
    arg1 = int(groups[1])
    arg2 = int(groups[2])

    state.ands.append(And(lit, arg1, arg2))
    return True


SYM_PATTERN = re.compile(r"([ilo])(\d+) (.*)")


def parse_symbol(line, state):
    match = SYM_PATTERN.fullmatch(line)
    if not match:
        return False
    kind, idx, name = match.groups()
    state.symbols.append(Symbol(kind, idx, name))
    return True


def parse_comment(line, state):
    if state.comments:
        state.comments.append(line.restrip())
    elif line.rstrip() == "c":
        state.comments = []
    else:
        return False
    return True


DEFAULT_COMPONENTS = ["header", "inputs", "latches", "outputs", "ands", "symbols", "comments"]


def parser_seq(components):
    for component in components:
        yield {
            "header": parse_header,
            "inputs": parse_input,
            "latches": parse_latch,
            "outputs": parse_output,
            "ands": parse_and,
            "symbols": parse_symbol,
            "comments": parse_comment,
        }.get(component)


def parse(circuit: str, components=None, state=None):
    if not components:
        components = DEFAULT_COMPONENTS

    if not state:
        state = Circuit()

    parsers = parser_seq(components)
    parser = next(parsers)

    lines = circuit.split("\n")
    for line in lines:
        while not parser(line, state):
            try:
                parser = next(parsers)
            except StopIteration as exc:
                message = f"Could not parse line: {line}"
                logger.critical(message)
                raise ValueError(message) from exc

    return state


def parse_no_header(circuit: str, num_inputs: int, num_outputs: int, components=None):
    state = Circuit()

    state.header = Header(None, num_inputs, None, num_outputs, None)

    parse(circuit, components, state)

    state.header.max_var_id = state.max_var_id
    state.header.num_latches = state.num_latches
    state.header.num_ands = state.num_ands

    return state
