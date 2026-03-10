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

import copy
from typing import Any

from ..data_utils import Role
from .dynamic_prompt import DynamicPromptDataset


def extract_reference_text(response_messages: list[dict[str, Any]]) -> str:
    r"""Extract the assistant reference transcript from aligned response messages."""
    assistant_texts: list[str] = []
    for message in response_messages:
        if message.get("role") == Role.ASSISTANT.value:
            content = message.get("content")
            if content is not None:
                assistant_texts.append(str(content))

    if assistant_texts:
        return "\n".join(assistant_texts).strip()

    if response_messages:
        content = response_messages[-1].get("content")
        if content is not None:
            return str(content).strip()

    return ""


class DynamicPromptGRPODataset(DynamicPromptDataset):
    r"""Prompt-only raw dataset for GRPO training on aligned chat/audio data."""

    def __init__(
        self,
        dataset,
        template,
        tokenizer,
        processor,
        data_args,
        *,
        seed: int | None = None,
        enable_prompt_sampling: bool = True,
    ) -> None:
        super().__init__(dataset, template, tokenizer, processor, data_args, seed=seed)
        self.enable_prompt_sampling = enable_prompt_sampling

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.dataset[idx]

        chosen = self._sample_pool_choice(example) if self.enable_prompt_sampling else None
        system = self._build_system_message(example, chosen)
        prompt = self._build_prompt_messages(example, chosen)
        response = self._build_response_messages(example, chosen)

        prompt_messages = copy.deepcopy(prompt)
        if system:
            prompt_messages = [{"role": Role.SYSTEM.value, "content": system}, *prompt_messages]

        item: dict[str, Any] = {
            "prompt": prompt_messages,
            "reference_text": extract_reference_text(response),
            "sample_id": self._get_sample_id(example),
        }

        if example.get("_audios") is not None:
            item["audios"] = example.get("_audios") or []

        if example.get("_images") is not None:
            item["images"] = example.get("_images") or []

        if example.get("_videos") is not None:
            item["videos"] = example.get("_videos") or []

        if example.get("_tools") is not None:
            item["tools"] = example.get("_tools")

        return item
