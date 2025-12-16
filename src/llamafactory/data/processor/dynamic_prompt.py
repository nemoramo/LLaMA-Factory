# Copyright 2025 the LlamaFactory team.
# Additional author: ramos.ma (GitHub: nemoramo).
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
import hashlib
import math
import random
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, get_worker_info

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..data_utils import Role
from .processor_utils import DatasetProcessor
from .supervised import SupervisedDatasetProcessor


logger = logging.get_logger(__name__)


class DynamicPromptDataset(Dataset):
    """Wraps an aligned HF dataset to sample prompts dynamically at access time.

    Expected aligned fields per sample:
    - _prompt_pool: optional list[...] of candidate *suffix prompts*.
      Each entry can be:
        * str: suffix appended to the last user message.
        * dict: {"text": "...", "weight": 0.2, "completion": "..."} for weighted suffix + optional target override.
        * list[dict]: full prompt messages to replace current prompt.
    - _prompt: list[{"role": ..., "content": ...}]  (fallback when no pool)
    - _response: list[{"role": ..., "content": ...}] (kept as-is)
    - _system/_tools/_images/_videos/_audios: optional extras

    Notes:
    - This dataset encodes/tokenizes on-the-fly, which can be CPU-heavy.
    - Not compatible with streaming datasets (needs random access + __len__/__getitem__).
    """

    def __init__(
        self,
        dataset,
        template,
        tokenizer,
        processor,
        data_args,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.data_args = data_args

        # RNG is created lazily and seeded per worker/rank to avoid identical sampling streams.
        self._base_seed = seed
        self._rng: random.Random | None = None
        self._rng_seeded: bool = False

        # Reuse the supervised processor for encoding logic.
        self.encoder = SupervisedDatasetProcessor(
            template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
        )

        # Fail-fast schema check (helps catch "loaded tokenized dataset" mistakes).
        column_names = getattr(dataset, "column_names", None)
        if isinstance(column_names, (list, tuple, set)):
            required = {"_prompt", "_response"}
            colset = set(column_names)
            if "input_ids" in colset and not required.issubset(colset):
                raise ValueError(
                    "DynamicPromptDataset expects an aligned dataset with columns "
                    f"{sorted(required)}. Got tokenized columns {sorted(colset)}. "
                    "This often happens when loading a tokenized dataset (`tokenized_path`)."
                )

            missing = required - colset
            if missing:
                # Support lazily aligned datasets (e.g., HF dataset.with_transform()).
                try:
                    sample = dataset[0]
                    if isinstance(sample, dict) and required.issubset(sample.keys()):
                        missing = set()
                except Exception:
                    pass

            if missing:
                raise ValueError(
                    "DynamicPromptDataset expects an aligned dataset with columns "
                    f"{sorted(required)}, but missing: {sorted(missing)}. "
                    "This often happens when alignment/conversion did not run."
                )

    @staticmethod
    def _stable_hash64(text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_rng(self) -> random.Random:
        """Initialize RNG once per worker process (and mix rank/worker into seed)."""
        if self._rng is None:
            self._rng = random.Random()

        if not self._rng_seeded:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0

            rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    rank = int(torch.distributed.get_rank())
                except Exception:
                    rank = 0

            worker_seed = int(torch.initial_seed())
            base_seed = int(self._base_seed) if self._base_seed is not None else 0
            mixed = (worker_seed + base_seed + rank * 1000 + worker_id) % (2**32)

            self._rng.seed(mixed)
            self._rng_seeded = True

        return self._rng

    @staticmethod
    def _sanitize_weight(w: Any) -> float:
        try:
            fw = float(w)
        except Exception:
            fw = 1.0
        if not math.isfinite(fw) or fw < 0:
            return 0.0
        return fw

    def _choose_from_pool(self, pool: Sequence[Any]) -> Any:
        """Choose one entry from prompt pool.

        Supports weighted sampling when pool entries are dicts like:
        {"text": "...", "weight": 0.2, ...}. If no weights are provided, uses uniform sampling.

        IMPORTANT: returns the original item (dict/list/str), so downstream can interpret schema.
        """
        rng = self._get_rng()
        if len(pool) == 0:
            raise ValueError("prompt_pool is empty.")

        values: list[Any] = []
        weights: list[float] = []
        for item in pool:
            w = 1.0
            if isinstance(item, dict) and "weight" in item:
                w = self._sanitize_weight(item.get("weight"))
            values.append(item)
            weights.append(w)

        if any(w > 0 for w in weights):
            return rng.choices(values, weights=weights, k=1)[0]
        return rng.choice(values)

    def _get_sample_id(self, example: dict[str, Any]) -> str:
        """Best-effort stable id used for deterministic per-sample prompt selection."""
        id_key = getattr(self.data_args, "dynamic_prompt_id_key", None)
        if isinstance(id_key, str) and id_key and id_key in example and example[id_key] is not None:
            return str(example[id_key])

        audios = example.get("_audios")
        if isinstance(audios, list) and len(audios) > 0 and audios[0] is not None:
            return str(audios[0])

        prompt_messages = example.get("_prompt")
        if isinstance(prompt_messages, list):
            for m in reversed(prompt_messages):
                if isinstance(m, dict) and m.get("role") == Role.USER.value:
                    return str(m.get("content", ""))

        return str(prompt_messages) if prompt_messages is not None else ""

    def _choose_from_pool_deterministic(self, pool: Sequence[Any], example: dict[str, Any]) -> Any:
        if len(pool) == 0:
            raise ValueError("prompt_pool is empty.")

        base_seed = int(self._base_seed) if self._base_seed is not None else 0
        sample_id = self._get_sample_id(example)
        h = self._stable_hash64(f"{base_seed}|{sample_id}")

        values: list[Any] = []
        weights: list[float] = []
        for item in pool:
            w = 1.0
            if isinstance(item, dict) and "weight" in item:
                w = self._sanitize_weight(item.get("weight"))
            values.append(item)
            weights.append(w)

        total = sum(weights)
        if total > 0:
            # Map hash to [0, total) deterministically.
            r = (h / float(2**64)) * total
            acc = 0.0
            for v, w in zip(values, weights):
                if w <= 0:
                    continue
                acc += w
                if r < acc:
                    return v
            return values[-1]

        return values[int(h % len(values))]

    def _sample_pool_choice(self, example: dict[str, Any]) -> Any | None:
        pool = example.get("_prompt_pool")
        if not (isinstance(pool, list) and len(pool) > 0):
            return None
        if getattr(self.data_args, "dynamic_prompt_deterministic", False):
            return self._choose_from_pool_deterministic(pool, example)
        return self._choose_from_pool(pool)

    def _build_prompt_messages(self, example: dict[str, Any], chosen: Any | None) -> list[dict[str, Any]]:
        """Build prompt messages, optionally applying a sampled pool entry."""
        prompt_messages = example.get("_prompt") or []
        if not isinstance(prompt_messages, list):
            prompt_messages = []
        prompt_messages = copy.deepcopy(prompt_messages)

        if chosen is None:
            return prompt_messages

        # If pool entry is a full prompt (list[dict]), use it directly.
        if isinstance(chosen, list) and all(isinstance(m, dict) for m in chosen):
            return copy.deepcopy(chosen)

        # If pool entry is a single message dict (OpenAI-style), append it.
        if isinstance(chosen, dict) and ("content" in chosen or "role" in chosen):
            msg = copy.deepcopy(chosen)
            if "role" not in msg:
                msg["role"] = Role.USER.value
            msg.setdefault("content", "")
            prompt_messages.append(msg)
            return prompt_messages

        # Otherwise treat as suffix (string or config dict with "text"/"suffix").
        if isinstance(chosen, dict):
            suffix = str(chosen.get("text") or chosen.get("suffix") or "")
        else:
            suffix = "" if chosen is None else str(chosen)

        if not suffix:
            return prompt_messages

        for m in reversed(prompt_messages):
            if m.get("role") == Role.USER.value:
                base = m.get("content", "")
                sep = ""
                if base and not base.endswith(("\n", " ")) and not suffix.startswith(("\n", " ")):
                    sep = "\n"
                m["content"] = f"{base}{sep}{suffix}" if base else suffix
                break
        else:
            prompt_messages.append({"role": Role.USER.value, "content": suffix})

        return prompt_messages

    def _build_response_messages(self, example: dict[str, Any], chosen: Any | None) -> list[dict[str, Any]]:
        response_messages = example.get("_response") or []
        if not isinstance(response_messages, list):
            response_messages = []
        response_messages = copy.deepcopy(response_messages)

        # If chosen entry provides completion override, apply it.
        if isinstance(chosen, dict):
            completion = (
                chosen.get("completion")
                or chosen.get("response")
                or chosen.get("output")
            )
            if completion is not None:
                completion_str = str(completion)
                if len(response_messages) > 0:
                    response_messages[-1]["content"] = completion_str
                else:
                    response_messages.append({"role": Role.ASSISTANT.value, "content": completion_str})

        return response_messages

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.dataset[idx]

        chosen = self._sample_pool_choice(example)
        prompt = self._build_prompt_messages(example, chosen)
        response = self._build_response_messages(example, chosen)

        input_ids, labels = self.encoder._encode_data_example(
            prompt=prompt,
            response=response,
            system=example.get("_system"),
            tools=example.get("_tools"),
            images=example.get("_images") or [],
            videos=example.get("_videos") or [],
            audios=example.get("_audios") or [],
        )

        item: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

        if example.get("_images") is not None:
            item["images"] = example.get("_images") or []
        if example.get("_videos") is not None:
            item["videos"] = example.get("_videos") or []
        if example.get("_audios") is not None:
            item["audios"] = example.get("_audios") or []

        return item


class DynamicPromptPackedDataset(DynamicPromptDataset):
    """A DynamicPromptDataset variant that packs multiple samples into one sequence.

    This is intended for SFT when you want:
    - `dynamic_prompt_sampling: true` (sample from `_prompt_pool` at access time), and
    - `packing: true` (optionally `neat_packing: true` for strict isolation).

    Implementation notes:
    - Packing is done on-the-fly in `__getitem__`, so it is CPU-heavy.
    - Each packed item always includes the requested index, and then tries to add extra random indices
      until reaching `cutoff_len` capacity. As a result, raw samples may appear multiple times per epoch.
    """

    def __init__(
        self,
        dataset,
        template,
        tokenizer,
        processor,
        data_args,
        seed: int | None = None,
        max_samples_per_pack: int = 8,
        max_trials_per_extra: int = 8,
    ) -> None:
        super().__init__(
            dataset=dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
            seed=seed,
        )
        self.tokenizer = tokenizer
        self.max_samples_per_pack = max(1, int(max_samples_per_pack))
        self.max_trials_per_extra = max(1, int(max_trials_per_extra))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = self._get_rng()

        cutoff_len = int(self.data_args.cutoff_len)
        if cutoff_len <= 0:
            raise ValueError(f"Invalid cutoff_len for packing: {self.data_args.cutoff_len}")
        # Match PackedSupervisedDatasetProcessor behavior: always reserve 1 token for padding.
        target_len = cutoff_len + 1
        capacity = target_len - 1

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id and tokenizer.eos_token_id are both None.")

        packed_input_ids: list[int] = []
        packed_labels: list[int] = []
        packed_position_ids: list[int] = []
        packed_attention_mask: list[int] = []

        packed_images: list[Any] = []
        packed_videos: list[Any] = []
        packed_audios: list[Any] = []
        has_images = False
        has_videos = False
        has_audios = False

        used: set[int] = {int(idx)}

        def add_segment(item: dict[str, Any], seg_id: int) -> None:
            nonlocal has_images, has_videos, has_audios
            seg_input_ids = item["input_ids"]
            seg_labels = item["labels"]
            start = len(packed_input_ids)

            # Safety: never exceed packing capacity.
            # Extra segments are already checked before adding; this primarily protects the first segment.
            if start + len(seg_input_ids) > capacity:
                keep = max(0, capacity - start)
                seg_input_ids = seg_input_ids[:keep]
                seg_labels = seg_labels[:keep]

            seg_len = len(seg_input_ids)
            if seg_len == 0:
                return

            packed_input_ids.extend(seg_input_ids)
            packed_labels.extend(seg_labels)
            if self.data_args.neat_packing:
                # Per-segment position reset for strict isolation.
                packed_position_ids.extend(list(range(seg_len)))
                packed_attention_mask.extend([seg_id] * seg_len)
            else:
                # Continuous positions for normal packing.
                packed_position_ids.extend(list(range(start, start + seg_len)))
                packed_attention_mask.extend([1] * seg_len)

            # Neat packing: mask boundary labels to avoid cross-segment loss contributions.
            if self.data_args.neat_packing:
                if start == 0:
                    packed_labels[0] = IGNORE_INDEX
                else:
                    packed_labels[start] = IGNORE_INDEX

            if "images" in item:
                has_images = True
                packed_images.extend(item.get("images") or [])
            if "videos" in item:
                has_videos = True
                packed_videos.extend(item.get("videos") or [])
            if "audios" in item:
                has_audios = True
                packed_audios.extend(item.get("audios") or [])

        # First segment: always include `idx`.
        add_segment(super().__getitem__(idx), seg_id=1)

        seg_id = 2
        while seg_id <= self.max_samples_per_pack and len(packed_input_ids) < capacity:
            remaining = capacity - len(packed_input_ids)
            if remaining <= 0:
                break

            chosen: dict[str, Any] | None = None
            for _ in range(self.max_trials_per_extra):
                cand = int(rng.randrange(len(self.dataset)))
                if cand in used and len(used) < len(self.dataset):
                    continue
                cand_item = super().__getitem__(cand)
                if len(cand_item["input_ids"]) <= remaining:
                    chosen = cand_item
                    used.add(cand)
                    break

            if chosen is None:
                break

            add_segment(chosen, seg_id=seg_id)
            seg_id += 1

        # Ensure final length is exactly `cutoff_len + 1` and contains at least one pad token.
        if len(packed_input_ids) >= target_len:
            # Reserve the last token for padding.
            packed_input_ids = packed_input_ids[: target_len - 1]
            packed_labels = packed_labels[: target_len - 1]
            packed_position_ids = packed_position_ids[: target_len - 1]
            packed_attention_mask = packed_attention_mask[: target_len - 1]

        pad_length = target_len - len(packed_input_ids)
        if pad_length < 1:
            pad_length = 1  # defensive; should not happen due to the truncation above.

        packed_input_ids.extend([int(pad_token_id)] * pad_length)
        packed_position_ids.extend([0] * pad_length)
        packed_labels.extend([IGNORE_INDEX] * pad_length)
        if self.data_args.neat_packing:
            packed_attention_mask.extend([0] * pad_length)
        else:
            packed_attention_mask.extend([1] * pad_length)

        if not (
            len(packed_input_ids)
            == len(packed_labels)
            == len(packed_position_ids)
            == len(packed_attention_mask)
            == target_len
        ):
            raise ValueError(
                "Packed sample has inconsistent lengths: "
                f"input_ids={len(packed_input_ids)}, labels={len(packed_labels)}, "
                f"position_ids={len(packed_position_ids)}, attention_mask={len(packed_attention_mask)}, "
                f"expected={target_len}"
            )

        item: dict[str, Any] = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "position_ids": packed_position_ids,
            "labels": packed_labels,
        }
        if has_images:
            item["images"] = packed_images
        if has_videos:
            item["videos"] = packed_videos
        if has_audios:
            item["audios"] = packed_audios

        return item


class DynamicPromptProcessor(DatasetProcessor):
    """Placeholder for interface consistency; not used for HF map preprocessing."""

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:  # pragma: no cover
        raise NotImplementedError("DynamicPromptProcessor should not run HF map preprocessing.")

    def print_data_example(self, example: dict[str, list[int]]) -> None:  # pragma: no cover
        logger.info_rank0(f"input_ids:\n{example['input_ids']}")
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        logger.info_rank0(f"labels:\n{valid_labels}")
