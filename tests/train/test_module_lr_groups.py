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

import math

import pytest
import torch
from transformers import Seq2SeqTrainingArguments

from llamafactory.hparams.finetuning_args import FinetuningArguments
from llamafactory.train.trainer_utils import create_custom_optimizer, create_custom_scheduler


class DummyModuleLRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.continuous_audio_tower = torch.nn.Module()
        self.continuous_audio_tower.proj = torch.nn.Linear(4, 4)
        self.continuous_audio_tower.encoder = torch.nn.Linear(4, 4)
        self.language_model = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)


@pytest.mark.runs_on(["cpu", "npu", "cuda"])
def test_module_lr_groups_optimizer_and_scheduler(tmp_path):
    model = DummyModuleLRModel()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=10,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none",
    )
    finetuning_args = FinetuningArguments(
        module_lr_groups=[
            {
                "name": "proj",
                "patterns": ["continuous_audio_tower.proj"],
                "lr": 3e-5,
                "lr_scheduler_type": "linear",
            },
            {
                "name": "audio",
                "patterns": ["continuous_audio_tower"],
                "lr": 1e-5,
                "lr_scheduler_type": "constant_with_warmup",
            },
        ]
    )

    optimizer = create_custom_optimizer(model, training_args, finetuning_args)
    assert optimizer is not None

    def has_lr(target: float) -> bool:
        return any(math.isclose(pg["lr"], target, rel_tol=0.0, abs_tol=1e-12) for pg in optimizer.param_groups)

    assert has_lr(1e-5)
    assert has_lr(3e-5)
    assert has_lr(1e-4)  # default (unmatched) parameters

    scheduler = create_custom_scheduler(training_args, num_training_steps=10, optimizer=optimizer)
    assert scheduler is not None
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert len(scheduler.lr_lambdas) == len(optimizer.param_groups)


def test_module_lr_groups_conflicts():
    with pytest.raises(ValueError):
        FinetuningArguments(
            module_lr_groups=[{"name": "audio", "patterns": ["continuous_audio_tower"], "lr": 1e-5}],
            loraplus_lr_ratio=16.0,
        )
