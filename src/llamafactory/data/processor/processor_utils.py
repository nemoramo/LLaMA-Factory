# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from the examples."""
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        ...


def _sanitize_prompt_pool_weight(value: Any) -> float:
    try:
        weight = float(value)
    except Exception:
        weight = 1.0

    if not math.isfinite(weight) or weight < 0.0:
        return 0.0

    return weight


def select_prompt_pool_top1(pool: list[Any]) -> Any:
    r"""Select the max-weight entry from a prompt pool (tie-break by first occurrence)."""
    if not pool:
        raise ValueError("prompt_pool is empty.")

    best_item = pool[0]
    best_weight = (
        _sanitize_prompt_pool_weight(best_item.get("weight", 1.0))
        if isinstance(best_item, dict)
        else 1.0
    )

    for item in pool[1:]:
        weight = _sanitize_prompt_pool_weight(item.get("weight", 1.0)) if isinstance(item, dict) else 1.0
        if weight > best_weight:
            best_item = item
            best_weight = weight

    return best_item


def append_suffix_to_system(system: str | None, suffix: str) -> str:
    r"""Append a suffix to the system prompt with a newline separator when needed."""
    base = "" if system is None else str(system)
    suffix = "" if suffix is None else str(suffix)
    if not suffix:
        return base
    if not base:
        return suffix

    sep = ""
    if base and not base.endswith(("\n", " ")) and not suffix.startswith(("\n", " ")):
        sep = "\n"
    return f"{base}{sep}{suffix}"


def apply_prompt_pool_top1_to_example(
    prompt: Any,
    response: Any,
    system: Any,
    prompt_pool: Any,
) -> tuple[Any, Any, Any]:
    r"""Apply prompt_pool Top1 entry to (prompt, response, system) for evaluation tokenization.

    - If prompt_pool entry is str/dict, treat it as a suffix and append to system prompt.
    - If entry is dict and contains `completion`, override the assistant target text.
    - If entry is list[dict], treat it as a full prompt replacement and do not touch system.
    """
    if not (isinstance(prompt_pool, list) and len(prompt_pool) > 0):
        return prompt, response, system

    chosen = select_prompt_pool_top1(prompt_pool)

    # Full prompt replacement: do not modify system.
    if isinstance(chosen, list) and all(isinstance(m, dict) for m in chosen):
        return chosen, response, system

    if isinstance(chosen, dict):
        suffix = str(chosen.get("text") or chosen.get("suffix") or chosen.get("content") or "")
    else:
        suffix = str(chosen)

    new_system = system
    if suffix:
        new_system = append_suffix_to_system(system, suffix)

    # Optional target override.
    if isinstance(chosen, dict):
        completion = chosen.get("completion") or chosen.get("response") or chosen.get("output")
        if completion is not None:
            completion_str = str(completion)
            if isinstance(response, list) and len(response) > 0 and isinstance(response[-1], dict):
                new_response = list(response)
                last = dict(new_response[-1])
                last["content"] = completion_str
                new_response[-1] = last
            elif isinstance(response, list):
                new_response = list(response)
                new_response.append({"role": "assistant", "content": completion_str})
            else:
                new_response = [{"role": "assistant", "content": completion_str}]
            return prompt, new_response, new_system

    return prompt, response, new_system


def search_for_fit(numbers: list[int], capacity: int) -> int:
    r"""Find the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""Compute the real sequence length after truncation by the cutoff_len."""
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len
