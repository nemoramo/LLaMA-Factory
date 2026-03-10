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

import os

import pytest

from llamafactory.hparams import get_train_args
from llamafactory.train.test_utils import load_dataset_module


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")
TINY_DATA = os.getenv("TINY_DATA", "llamafactory/tiny-supervised-dataset")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "grpo",
    "do_train": True,
    "finetuning_type": "lora",
    "template": "llama3",
    "dataset": TINY_DATA,
    "dataset_dir": "ONLINE",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
    "report_to": "none",
}


@pytest.mark.runs_on(["cpu", "mps"])
def test_grpo_loads_prompt_only_dataset():
    dataset_module = load_dataset_module(**TRAIN_ARGS)
    train_dataset = dataset_module["train_dataset"]
    sample = train_dataset[0]

    assert "prompt" in sample
    assert "reference_text" in sample
    assert "sample_id" in sample
    assert "input_ids" not in sample
    assert "labels" not in sample
    assert isinstance(sample["prompt"], list)
    assert isinstance(sample["reference_text"], str)


@pytest.mark.runs_on(["cpu", "mps"])
def test_grpo_rejects_tokenized_path():
    with pytest.raises(ValueError, match="tokenized_path"):
        load_dataset_module(tokenized_path="dummy_tokenized_path", **TRAIN_ARGS)


def test_grpo_rejects_packing():
    with pytest.raises(ValueError, match="does not support packing"):
        get_train_args({**TRAIN_ARGS, "packing": True})


def test_grpo_rejects_experimental_funaudiochat_full_colocate_tp():
    with pytest.raises(ValueError, match="still experimental"):
        get_train_args(
            {
                **TRAIN_ARGS,
                "template": "funaudiochat",
                "finetuning_type": "full",
                "grpo_use_vllm": True,
                "grpo_vllm_mode": "colocate",
                "grpo_vllm_tensor_parallel_size": 2,
            }
        )


def test_grpo_allows_experimental_funaudiochat_full_colocate_tp_with_override():
    _, data_args, _, finetuning_args, _ = get_train_args(
        {
            **TRAIN_ARGS,
            "template": "funaudiochat",
            "finetuning_type": "full",
            "grpo_use_vllm": True,
            "grpo_vllm_mode": "colocate",
            "grpo_vllm_tensor_parallel_size": 2,
            "grpo_allow_experimental_funaudiochat_colocate_tp": True,
        }
    )

    assert data_args.template == "funaudiochat"
    assert finetuning_args.grpo_allow_experimental_funaudiochat_colocate_tp is True
