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

from types import SimpleNamespace

import pytest

from llamafactory.train.grpo.reward import build_asr_reward_suite


def _build_args():
    return SimpleNamespace(
        grpo_reward_wer_weight=1.0,
        grpo_reward_cer_weight=0.25,
        grpo_empty_penalty=1.0,
        grpo_repeat_penalty=0.2,
    )


def test_grpo_reward_prefers_better_transcript():
    reward_funcs, reward_weights = build_asr_reward_suite(_build_args())
    assert reward_weights == [1.0, 0.0, 0.0, 0.0, 0.0]

    composite_reward, wer_reward, cer_reward, empty_penalty, repeat_penalty = reward_funcs
    reference_text = ["hello world", "hello world", "hello world"]
    completions = [
        [{"role": "assistant", "content": "hello world"}],
        [{"role": "assistant", "content": "hello there"}],
        [{"role": "assistant", "content": ""}],
    ]

    composite_scores = composite_reward(completions=completions, reference_text=reference_text)
    wer_scores = wer_reward(completions=completions, reference_text=reference_text)
    cer_scores = cer_reward(completions=completions, reference_text=reference_text)
    empty_scores = empty_penalty(completions=completions, reference_text=reference_text)

    assert composite_scores[0] > composite_scores[1] > composite_scores[2]
    assert wer_scores[0] > wer_scores[1] >= wer_scores[2]
    assert cer_scores[0] > cer_scores[1] >= cer_scores[2]
    assert empty_scores == [0.0, 0.0, -1.0]


def test_grpo_reward_normalizes_special_tokens_and_penalizes_repetition():
    reward_funcs, _ = build_asr_reward_suite(_build_args())
    composite_reward, _, _, _, repeat_penalty = reward_funcs

    reference_text = ["hello world", "hello world"]
    completions = [
        [{"role": "assistant", "content": "<|audio_bos|> HELLO, world! <|audio_eos|>"}],
        [{"role": "assistant", "content": "hello hello hello hello"}],
    ]

    composite_scores = composite_reward(completions=completions, reference_text=reference_text)
    repeat_scores = repeat_penalty(completions=completions, reference_text=reference_text)

    assert composite_scores[0] == pytest.approx(1.25, rel=1e-6)
    assert composite_scores[0] > composite_scores[1]
    assert repeat_scores[0] == 0.0
    assert repeat_scores[1] < 0.0
