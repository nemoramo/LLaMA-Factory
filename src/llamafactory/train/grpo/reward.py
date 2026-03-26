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

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ...extras.packages import is_jieba_available
from ..metric_utils import compute_error_rate, has_cjk, normalize_text


if is_jieba_available():
    import jieba


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


def _extract_text(item: Any) -> str:
    if item is None:
        return ""

    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        content = item.get("content")
        return str(content) if content is not None else ""

    if isinstance(item, list):
        texts: list[str] = []
        for message in item:
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if content is not None:
                    texts.append(str(content))
        if texts:
            return "\n".join(texts)
        return "\n".join(str(elem) for elem in item if elem is not None)

    return str(item)


def _clean_prediction(text: str) -> str:
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    return normalize_text(text)


def _word_tokens(text: str) -> list[str]:
    if not text:
        return []

    if is_jieba_available() and has_cjk(text):
        return [token for token in jieba.cut(text) if token.strip()]

    return text.split()


def _repeat_ratio(text: str) -> float:
    tokens = _word_tokens(text)
    if len(tokens) <= 1:
        return 0.0

    repeated = sum(1 for idx in range(1, len(tokens)) if tokens[idx] == tokens[idx - 1])
    return float(repeated) / float(len(tokens) - 1)


def _score_components(completion: Any, reference_text: Any) -> dict[str, float]:
    pred_text = _clean_prediction(_extract_text(completion))
    ref_text = _clean_prediction(_extract_text(reference_text))

    wer = compute_error_rate(_word_tokens(ref_text), _word_tokens(pred_text))
    cer = compute_error_rate(list(ref_text), list(pred_text))
    empty = 1.0 if not pred_text else 0.0
    repeat = _repeat_ratio(pred_text)

    return {
        "wer_score": 1.0 - min(max(wer, 0.0), 1.0),
        "cer_score": 1.0 - min(max(cer, 0.0), 1.0),
        "empty_penalty": -empty,
        "repeat_penalty": -repeat,
    }


def build_asr_reward_suite(finetuning_args: Any) -> tuple[list[Callable[..., list[float]]], list[float]]:
    wer_weight = float(finetuning_args.grpo_reward_wer_weight)
    cer_weight = float(finetuning_args.grpo_reward_cer_weight)
    empty_penalty = float(finetuning_args.grpo_empty_penalty)
    repeat_penalty = float(finetuning_args.grpo_repeat_penalty)

    def composite_reward(*, completions, reference_text, **kwargs) -> list[float]:
        rewards: list[float] = []
        for completion, ref in zip(completions, reference_text):
            components = _score_components(completion, ref)
            reward = (
                wer_weight * components["wer_score"]
                + cer_weight * components["cer_score"]
                + empty_penalty * components["empty_penalty"]
                + repeat_penalty * components["repeat_penalty"]
            )
            rewards.append(float(reward))

        return rewards

    def wer_reward(*, completions, reference_text, **kwargs) -> list[float]:
        return [float(_score_components(completion, ref)["wer_score"]) for completion, ref in zip(completions, reference_text)]

    def cer_reward(*, completions, reference_text, **kwargs) -> list[float]:
        return [float(_score_components(completion, ref)["cer_score"]) for completion, ref in zip(completions, reference_text)]

    def empty_output_penalty(*, completions, reference_text, **kwargs) -> list[float]:
        return [
            float(_score_components(completion, ref)["empty_penalty"])
            for completion, ref in zip(completions, reference_text)
        ]

    def repetition_penalty(*, completions, reference_text, **kwargs) -> list[float]:
        return [
            float(_score_components(completion, ref)["repeat_penalty"])
            for completion, ref in zip(completions, reference_text)
        ]

    composite_reward.__name__ = "asr_reward"
    wer_reward.__name__ = "asr_wer_score"
    cer_reward.__name__ = "asr_cer_score"
    empty_output_penalty.__name__ = "asr_empty_penalty"
    repetition_penalty.__name__ = "asr_repeat_penalty"

    reward_funcs = [composite_reward, wer_reward, cer_reward, empty_output_penalty, repetition_penalty]
    reward_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
    return reward_funcs, reward_weights
