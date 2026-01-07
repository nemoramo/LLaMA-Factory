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
import threading
from queue import Empty, Full, Queue
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..data_utils import Role
from .processor_utils import DatasetProcessor, greedy_knapsack
from .supervised import SupervisedDatasetProcessor


logger = logging.get_logger(__name__)


class DynamicPromptDataset(Dataset):
    """Wraps an aligned HF dataset to sample prompts dynamically at access time.

    Expected aligned fields per sample:
    - _prompt_pool: optional list[...] of candidate *suffix prompts*.
      Each entry can be:
        * str: suffix appended to the system prompt.
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

    @staticmethod
    def _append_suffix_to_system(system: str | None, suffix: str) -> str:
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

    def _build_system_message(self, example: dict[str, Any], chosen: Any | None) -> str | None:
        """Build system message, optionally applying a sampled pool entry as a suffix."""
        system = example.get("_system")
        if chosen is None:
            return system

        # If pool entry is a full prompt (list[dict]), do not touch system.
        if isinstance(chosen, list) and all(isinstance(m, dict) for m in chosen):
            return system

        # Otherwise treat as suffix (string or config dict with "text"/"suffix"/"content").
        if isinstance(chosen, dict):
            suffix = str(chosen.get("text") or chosen.get("suffix") or chosen.get("content") or "")
        else:
            suffix = str(chosen)

        if not suffix:
            return system
        return self._append_suffix_to_system(system, suffix)

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

        pool = example.get("_prompt_pool")
        has_pool = isinstance(pool, list) and len(pool) > 0
        deterministic = bool(getattr(self.data_args, "dynamic_prompt_deterministic", False))
        max_tries = int(getattr(self.data_args, "dynamic_prompt_encode_max_tries", 3) or 3)
        max_tries = max(1, max_tries)
        if not has_pool:
            max_tries = 1
        if deterministic:
            max_tries = min(max_tries, 2)

        input_ids: list[int] | None = None
        labels: list[int] | None = None
        last_error: ValueError | None = None
        tried_no_pool = False

        for attempt in range(max_tries):
            if attempt == 0:
                chosen = self._sample_pool_choice(example)
            else:
                if has_pool and not tried_no_pool:
                    chosen = None
                    tried_no_pool = True
                elif deterministic:
                    chosen = None
                else:
                    chosen = self._choose_from_pool(pool) if has_pool else None

            system = self._build_system_message(example, chosen)
            prompt = self._build_prompt_messages(example, chosen)
            response = self._build_response_messages(example, chosen)

            try:
                input_ids, labels = self.encoder._encode_data_example(
                    prompt=prompt,
                    response=response,
                    system=system,
                    tools=example.get("_tools"),
                    images=example.get("_images") or [],
                    videos=example.get("_videos") or [],
                    audios=example.get("_audios") or [],
                )
                last_error = None
                break
            except ValueError as err:
                last_error = err
                continue

        if input_ids is None or labels is None or last_error is not None:
            sample_id = self._get_sample_id(example)
            audios = example.get("_audios") or []
            audio_hint = None
            if isinstance(audios, list) and len(audios) > 0:
                audio_hint = audios[0]
            raise ValueError(
                f"Failed to encode sample idx={idx} id={sample_id!r} audio={audio_hint!r}: {last_error}"
            ) from last_error

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


class DynamicPromptPackedBatchProcessor:
    """Batch processor that does on-the-fly dynamic prompt sampling + buffered knapsack packing.

    Intended usage: `datasets.IterableDataset.map(batched=True, batch_size=buffer_size)`.
    Buffering is only for speed (batching Python/tokenizer work); it doesn't require full-dataset tokenization.
    """

    def __init__(
        self,
        template,
        tokenizer,
        processor,
        data_args,
        *,
        dataset_converter: Any | None = None,
        id_key: str | None = None,
        seed: int | None = None,
        max_samples_per_pack: int = 8,
        shuffle_packs: bool = True,
    ) -> None:
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.max_samples_per_pack = max(1, int(max_samples_per_pack))
        self.shuffle_packs = bool(shuffle_packs)
        self.dataset_converter = dataset_converter
        self.id_key = id_key
        self._buffer_idx = 0
        self._seen_samples = 0
        self._log_every = int(getattr(data_args, "dynamic_prompt_packing_log_interval", 0) or 0)

        self._base_seed = seed
        self._rng: random.Random | None = None
        self._rng_seeded: bool = False

        self.encoder = SupervisedDatasetProcessor(
            template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
        )

    def _get_rng(self) -> random.Random:
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
    def _stable_hash64(text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

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

    @staticmethod
    def _append_suffix_to_system(system: str | None, suffix: str) -> str:
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

    def _build_system_message(self, example: dict[str, Any], chosen: Any | None) -> str | None:
        system = example.get("_system")
        if chosen is None:
            return system
        if isinstance(chosen, list) and all(isinstance(m, dict) for m in chosen):
            return system
        if isinstance(chosen, dict):
            suffix = str(chosen.get("text") or chosen.get("suffix") or chosen.get("content") or "")
        else:
            suffix = str(chosen)
        if not suffix:
            return system
        return self._append_suffix_to_system(system, suffix)

    @staticmethod
    def _build_prompt_messages(example: dict[str, Any], chosen: Any | None) -> list[dict[str, Any]]:
        prompt_messages = example.get("_prompt") or []
        if not isinstance(prompt_messages, list):
            prompt_messages = []
        prompt_messages = copy.deepcopy(prompt_messages)

        if chosen is None:
            return prompt_messages

        if isinstance(chosen, list) and all(isinstance(m, dict) for m in chosen):
            return copy.deepcopy(chosen)

        return prompt_messages

    @staticmethod
    def _build_response_messages(example: dict[str, Any], chosen: Any | None) -> list[dict[str, Any]]:
        response_messages = example.get("_response") or []
        if not isinstance(response_messages, list):
            response_messages = []
        response_messages = copy.deepcopy(response_messages)

        if isinstance(chosen, dict):
            completion = chosen.get("completion") or chosen.get("response") or chosen.get("output")
            if completion is not None:
                completion_str = str(completion)
                if len(response_messages) > 0:
                    response_messages[-1]["content"] = completion_str
                else:
                    response_messages.append({"role": Role.ASSISTANT.value, "content": completion_str})

        return response_messages

    def _pack_segments(
        self,
        segments: list[dict[str, Any]],
        *,
        pad_token_id: int,
        target_len: int,
        capacity: int,
    ) -> dict[str, Any]:
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

        seg_id = 1
        for seg in segments:
            seg_input_ids = seg.get("input_ids") or []
            seg_labels = seg.get("labels") or []
            start = len(packed_input_ids)

            if start >= capacity:
                break

            if start + len(seg_input_ids) > capacity:
                keep = max(0, capacity - start)
                seg_input_ids = seg_input_ids[:keep]
                seg_labels = seg_labels[:keep]

            seg_len = len(seg_input_ids)
            if seg_len == 0:
                continue

            packed_input_ids.extend(seg_input_ids)
            packed_labels.extend(seg_labels)
            if self.data_args.neat_packing:
                packed_position_ids.extend(list(range(seg_len)))
                packed_attention_mask.extend([seg_id] * seg_len)
                packed_labels[start] = IGNORE_INDEX
            else:
                packed_position_ids.extend(list(range(start, start + seg_len)))
                packed_attention_mask.extend([1] * seg_len)

            if "images" in seg:
                has_images = True
                packed_images.extend(seg.get("images") or [])
            if "videos" in seg:
                has_videos = True
                packed_videos.extend(seg.get("videos") or [])
            if "audios" in seg:
                has_audios = True
                packed_audios.extend(seg.get("audios") or [])

            seg_id += 1

        # Ensure there is always at least one padding token (see `target_len = cutoff_len + 1`).
        if len(packed_input_ids) > capacity:
            packed_input_ids = packed_input_ids[:capacity]
            packed_labels = packed_labels[:capacity]
            packed_position_ids = packed_position_ids[:capacity]
            packed_attention_mask = packed_attention_mask[:capacity]

        pad_length = target_len - len(packed_input_ids)
        if pad_length <= 0:
            raise ValueError(
                "Internal error: packed sequence length must be < target_len to keep at least one padding token."
            )

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

    def __call__(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        items, lengths = self.encode_examples(examples)
        packed, _, _ = self.pack_encoded_items(items, lengths, carryover_packs=0)
        return packed

    def encode_examples(self, examples: dict[str, list[Any]]) -> tuple[list[dict[str, Any]], list[int]]:
        """Encode a raw buffer into per-sample segments (input_ids/labels + optional media).

        Returns (items, lengths). Invalid/too-long samples are dropped (with limited warnings).
        """
        if not examples:
            return [], []

        prompt_col = examples.get("_prompt")
        if isinstance(prompt_col, list):
            batch_size = len(prompt_col)
        else:
            first_col = next((v for v in examples.values() if isinstance(v, list)), None)
            batch_size = len(first_col) if isinstance(first_col, list) else 0
        if batch_size == 0:
            return [], []

        for k, v in examples.items():
            if isinstance(v, list) and len(v) != batch_size:
                raise ValueError(
                    "Dynamic prompt packing received inconsistent batch column lengths: "
                    f"len({k})={len(v)} vs batch_size={batch_size}"
                )

        self._buffer_idx += 1
        self._seen_samples += batch_size
        if self._log_every > 0 and self._buffer_idx % self._log_every == 0:
            logger.info_rank0(
                f"Dynamic prompt packing: processed {self._buffer_idx} buffers, "
                f"{self._seen_samples} raw samples (latest buffer_size={batch_size})."
            )

        cutoff_len = int(self.data_args.cutoff_len)
        if cutoff_len <= 0:
            raise ValueError(f"Invalid cutoff_len for packing: {self.data_args.cutoff_len}")

        target_len = cutoff_len + 1
        capacity = target_len - 1

        items: list[dict[str, Any]] = []
        lengths: list[int] = []

        dropped_invalid = 0
        dropped_long = 0
        dropped_encode = 0

        warn_limit = int(getattr(self, "_warn_limit", 5) or 5)
        warned = int(getattr(self, "_warned", 0) or 0)

        for i in range(batch_size):
            row = {k: v[i] for k, v in examples.items()}
            if self.dataset_converter is not None:
                aligned = self.dataset_converter(row)
                if not isinstance(aligned, dict):
                    dropped_invalid += 1
                    continue
                if isinstance(self.id_key, str) and self.id_key and self.id_key in row and row[self.id_key] is not None:
                    aligned[self.id_key] = row[self.id_key]
                example = aligned
            else:
                example = row

            prompt = example.get("_prompt") or []
            response = example.get("_response") or []
            if not (isinstance(prompt, list) and isinstance(response, list)):
                dropped_invalid += 1
                continue
            if len(prompt) % 2 != 1 or len(response) != 1:
                dropped_invalid += 1
                if warned < warn_limit:
                    logger.warning_rank0("Dropped invalid example: {}".format(prompt + response))
                    warned += 1
                continue

            chosen = self._sample_pool_choice(example)
            prompt_messages = self._build_prompt_messages(example, chosen)
            response_messages = self._build_response_messages(example, chosen)
            system = self._build_system_message(example, chosen)

            try:
                input_ids, labels = self.encoder._encode_data_example(
                    prompt=prompt_messages,
                    response=response_messages,
                    system=system,
                    tools=example.get("_tools"),
                    images=example.get("_images") or [],
                    videos=example.get("_videos") or [],
                    audios=example.get("_audios") or [],
                )
            except Exception as err:
                dropped_encode += 1
                if warned < warn_limit:
                    logger.warning_rank0(f"Dropped example due to encode error: {err}")
                    warned += 1
                continue

            l = len(input_ids)
            if l == 0:
                dropped_invalid += 1
                continue
            if l > capacity:
                dropped_long += 1
                if warned < warn_limit:
                    logger.warning_rank0(f"Dropped lengthy example with length {l} > {capacity}.")
                    warned += 1
                continue

            item: dict[str, Any] = {"input_ids": input_ids, "labels": labels}
            if example.get("_images") is not None:
                item["images"] = example.get("_images") or []
            if example.get("_videos") is not None:
                item["videos"] = example.get("_videos") or []
            if example.get("_audios") is not None:
                item["audios"] = example.get("_audios") or []

            items.append(item)
            lengths.append(l)

        self._warned = warned
        self._warn_limit = warn_limit

        # Cumulative stats (per worker process).
        seen_total = int(getattr(self, "_seen_total", 0) or 0) + batch_size
        kept_total = int(getattr(self, "_kept_total", 0) or 0) + len(items)
        dropped_invalid_total = int(getattr(self, "_dropped_invalid_total", 0) or 0) + dropped_invalid
        dropped_long_total = int(getattr(self, "_dropped_long_total", 0) or 0) + dropped_long
        dropped_encode_total = int(getattr(self, "_dropped_encode_total", 0) or 0) + dropped_encode

        self._seen_total = seen_total
        self._kept_total = kept_total
        self._dropped_invalid_total = dropped_invalid_total
        self._dropped_long_total = dropped_long_total
        self._dropped_encode_total = dropped_encode_total

        log_every_seen = int(getattr(self, "_log_every_seen", 200000) or 200000)
        last_log_seen_total = int(getattr(self, "_last_log_seen_total", 0) or 0)
        if log_every_seen > 0 and (seen_total - last_log_seen_total) >= log_every_seen:
            last_log_kept_total = int(getattr(self, "_last_log_kept_total", 0) or 0)
            last_log_dropped_invalid_total = int(getattr(self, "_last_log_dropped_invalid_total", 0) or 0)
            last_log_dropped_long_total = int(getattr(self, "_last_log_dropped_long_total", 0) or 0)
            last_log_dropped_encode_total = int(getattr(self, "_last_log_dropped_encode_total", 0) or 0)

            window_seen = seen_total - last_log_seen_total
            window_kept = kept_total - last_log_kept_total
            window_drop_invalid = dropped_invalid_total - last_log_dropped_invalid_total
            window_drop_long = dropped_long_total - last_log_dropped_long_total
            window_drop_encode = dropped_encode_total - last_log_dropped_encode_total

            total_keep_ratio = kept_total / max(1, seen_total)
            window_keep_ratio = window_kept / max(1, window_seen)

            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0

            logger.info_rank0(
                "Dynamic prompt packing stats (worker={}): total kept={}/{} ({:.2%}), "
                "dropped_invalid={}, dropped_long={}, dropped_encode={}; window kept={}/{} ({:.2%}), "
                "dropped_invalid={}, dropped_long={}, dropped_encode={}.".format(
                    worker_id,
                    kept_total,
                    seen_total,
                    total_keep_ratio,
                    dropped_invalid_total,
                    dropped_long_total,
                    dropped_encode_total,
                    window_kept,
                    window_seen,
                    window_keep_ratio,
                    window_drop_invalid,
                    window_drop_long,
                    window_drop_encode,
                )
            )

            self._last_log_seen_total = seen_total
            self._last_log_kept_total = kept_total
            self._last_log_dropped_invalid_total = dropped_invalid_total
            self._last_log_dropped_long_total = dropped_long_total
            self._last_log_dropped_encode_total = dropped_encode_total

        return items, lengths

    def pack_encoded_items(
        self, items: list[dict[str, Any]], lengths: list[int], *, carryover_packs: int = 0
    ) -> tuple[dict[str, list[Any]], list[dict[str, Any]], list[int]]:
        """Pack already-encoded segments into fixed-length packed sequences.

        Returns (packed, carry_items, carry_lengths). `carryover_packs` holds back a number of lowest-fill packed
        sequences (by sum(lengths)/capacity) and returns their raw segments for cross-buffer mixing.
        """
        if not items:
            return {}, [], []

        if len(items) != len(lengths):
            raise ValueError(f"Dynamic prompt packing internal error: len(items)={len(items)} != len(lengths)={len(lengths)}")

        cutoff_len = int(self.data_args.cutoff_len)
        if cutoff_len <= 0:
            raise ValueError(f"Invalid cutoff_len for packing: {self.data_args.cutoff_len}")

        target_len = cutoff_len + 1
        capacity = target_len - 1

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id and tokenizer.eos_token_id are both None.")

        length2indexes: dict[int, list[int]] = defaultdict(list)
        for i, l in enumerate(lengths):
            if l <= 0 or l > capacity:
                continue
            length2indexes[int(l)].append(i)

        if not length2indexes:
            return {}, [], []

        rng = self._get_rng()
        knapsacks = greedy_knapsack([int(l) for l in lengths if 0 < int(l) <= capacity], capacity)
        if self.shuffle_packs:
            rng.shuffle(knapsacks)

        packs: list[list[int]] = []
        for knapsack in knapsacks:
            if not knapsack:
                continue

            picked: list[int] = []
            for l in knapsack:
                if length2indexes[l]:
                    picked.append(length2indexes[l].pop())

            if not picked:
                continue

            for start in range(0, len(picked), self.max_samples_per_pack):
                pack = picked[start : start + self.max_samples_per_pack]
                if pack:
                    packs.append(pack)

        if not packs:
            return {}, [], []

        carryover_packs = int(carryover_packs or 0)
        keep_pack_ids: set[int] = set()
        if carryover_packs > 0 and len(packs) > carryover_packs:
            fills = [sum(lengths[i] for i in pack) for pack in packs]
            worst = sorted(range(len(packs)), key=lambda j: fills[j])[:carryover_packs]
            keep_pack_ids = set(worst)

        model_inputs: dict[str, list[Any]] = defaultdict(list)
        carry_indices: list[int] = []
        for j, pack in enumerate(packs):
            if j in keep_pack_ids:
                carry_indices.extend(pack)
                continue

            segments = [items[i] for i in pack]
            packed = self._pack_segments(
                segments,
                pad_token_id=int(pad_token_id),
                target_len=target_len,
                capacity=capacity,
            )
            for k, v in packed.items():
                model_inputs[k].append(v)

        carry_items = [items[i] for i in carry_indices] if carry_indices else []
        carry_lengths = [lengths[i] for i in carry_indices] if carry_indices else []
        return dict(model_inputs), carry_items, carry_lengths


def build_dynamic_prompt_packed_iterable_dataset(
    dataset,
    *,
    template,
    tokenizer,
    processor,
    data_args,
    dataset_converter: Any | None = None,
    id_key: str | None = None,
    seed: int | None = None,
    buffer_size: int = 20000,
    max_samples_per_pack: int = 8,
    shuffle_packs: bool = True,
    num_shards: int = 0,
    global_shuffle: bool = True,
    prefetch_buffers: int = 0,
    carryover_packs: int = 0,
):
    """Build a sharded HF IterableDataset for buffered dynamic prompt packing."""
    try:
        from datasets import Dataset
    except Exception as err:  # pragma: no cover
        raise ImportError("Dynamic prompt packing requires the `datasets` package.") from err

    if not isinstance(dataset, Dataset):
        raise ValueError("Dynamic prompt packing requires a HF map-style `Dataset` as input (not streaming).")

    if not isinstance(buffer_size, int) or buffer_size <= 0:
        raise ValueError(f"Invalid buffer_size for dynamic prompt packing: {buffer_size}")

    # Best-effort resource hint: buffering scales CPU/RAM roughly linearly with buffer_size and cutoff_len.
    try:
        cutoff_len = int(getattr(data_args, "cutoff_len", 0) or 0)
        if cutoff_len > 0:
            approx_tokens = buffer_size * cutoff_len
            if approx_tokens >= 5_000_000:
                logger.warning_rank0(
                    f"Dynamic prompt packing: buffer_size({buffer_size}) * cutoff_len({cutoff_len}) "
                    f"≈ {approx_tokens:,} tokens buffered per map call. This can be CPU/RAM heavy. "
                    "Consider reducing `dynamic_prompt_packing_buffer_size` if you hit OOM/slowdowns."
                )
    except Exception:
        pass

    if num_shards <= 0:
        # Power-of-two default tends to play well with common world sizes and dataloader workers.
        num_shards = 1024
        try:
            n = len(dataset)
            if isinstance(n, int) and n > 0:
                num_shards = min(num_shards, n)
        except Exception:
            pass

    if global_shuffle:
        try:
            dataset = dataset.shuffle(seed=int(seed or 0))
        except Exception as err:
            logger.warning_rank0(f"Failed to shuffle dataset for dynamic prompt packing: {err}")

    try:
        iterable_ds = dataset.to_iterable_dataset(num_shards=int(num_shards))
    except Exception as err:
        # HF `datasets` does not always support converting a formatted map-style dataset (e.g. `with_transform`,
        # `with_format`, or selected columns/format kwargs) into an iterable dataset. Clear formatting/transform
        # and retry.
        try:
            unformatted = dataset
            if hasattr(unformatted, "reset_format"):
                try:
                    unformatted.reset_format()
                except Exception:
                    pass
            if hasattr(unformatted, "with_format"):
                try:
                    unformatted = unformatted.with_format(None)
                except Exception:
                    pass

            iterable_ds = unformatted.to_iterable_dataset(num_shards=int(num_shards))
            dataset = unformatted
        except Exception:
            raise ValueError("Dynamic prompt packing requires `dataset.to_iterable_dataset(num_shards=...)`.") from err

    raw_columns = getattr(dataset, "column_names", None)
    remove_columns = list(raw_columns) if isinstance(raw_columns, (list, tuple)) else None

    packer = DynamicPromptPackedBatchProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
        dataset_converter=dataset_converter,
        id_key=id_key,
        seed=seed,
        max_samples_per_pack=max_samples_per_pack,
        shuffle_packs=shuffle_packs,
    )

    prefetch_buffers = int(prefetch_buffers or 0)
    if prefetch_buffers < 0:
        raise ValueError(f"Invalid prefetch_buffers for dynamic prompt packing: {prefetch_buffers}")

    carryover_packs = int(carryover_packs or 0)
    if carryover_packs < 0:
        raise ValueError(f"Invalid carryover_packs for dynamic prompt packing: {carryover_packs}")

    if carryover_packs > 0:
        logger.info_rank0(
            "Dynamic prompt packing: enable cross-buffer carryover (carryover_packs={}). "
            "This may improve packing efficiency but slightly changes sample order.".format(carryover_packs)
        )
        if prefetch_buffers <= 0:
            prefetch_buffers = 1

    if prefetch_buffers > 0:
        logger.info_rank0(
            "Dynamic prompt packing: enable packed-buffer prefetch (prefetch_buffers={}). "
            "This may reduce dataloader stalls but increases CPU/RAM usage.".format(prefetch_buffers)
        )
        return _DynamicPromptPackedPrefetchDataset(
            iterable_ds=iterable_ds,
            packer=packer,
            buffer_size=int(buffer_size),
            prefetch_buffers=prefetch_buffers,
            carryover_packs=carryover_packs,
        )

    packed = iterable_ds.map(packer, batched=True, batch_size=int(buffer_size), remove_columns=remove_columns)
    # Repeat indefinitely to avoid DDP hangs when `max_steps` exceeds one full pass. Training length should be
    # controlled via `max_steps`. Each "cycle" still uses every raw sample at most once (per rank shard).
    return packed.repeat(None)


class _DynamicPromptPackedPrefetchDataset(IterableDataset):
    """Torch IterableDataset that prefetches *packed buffers* in a background thread.

    Why:
      HF IterableDataset.map(batched=True, batch_size=buffer_size) processes an entire buffer before yielding any
      packed example. This can create periodic stalls at buffer boundaries when the dataloader queue is filled by
      items from the current buffer only. Prefetching a small number of packed buffers ahead helps hide these stalls.

    Notes:
      - Prefetch is per dataloader worker process.
      - Increasing `prefetch_buffers` increases CPU/RAM usage (packed samples are large Python lists).
    """

    def __init__(
        self,
        *,
        iterable_ds,
        packer: DynamicPromptPackedBatchProcessor,
        buffer_size: int,
        prefetch_buffers: int,
        carryover_packs: int = 0,
    ) -> None:
        self.iterable_ds = iterable_ds
        self.packer = packer
        self.buffer_size = int(buffer_size)
        self.prefetch_buffers = int(prefetch_buffers)
        self.carryover_packs = int(carryover_packs or 0)

        if self.buffer_size <= 0:
            raise ValueError(f"Invalid buffer_size for dynamic prompt packing: {self.buffer_size}")
        if self.prefetch_buffers <= 0:
            raise ValueError(f"Invalid prefetch_buffers for dynamic prompt packing: {self.prefetch_buffers}")
        if self.carryover_packs < 0:
            raise ValueError(f"Invalid carryover_packs for dynamic prompt packing: {self.carryover_packs}")

    @staticmethod
    def _take_raw_batch(raw_iter, batch_size: int) -> dict[str, list[Any]] | None:
        rows: list[dict[str, Any]] = []
        for _ in range(batch_size):
            try:
                row = next(raw_iter)
            except StopIteration:
                break
            if row is None:
                continue
            if not isinstance(row, dict):
                raise ValueError(f"Dynamic prompt packing expected dict rows, got: {type(row)}")
            rows.append(row)

        if not rows:
            return None

        keys = list(rows[0].keys())
        examples: dict[str, list[Any]] = {k: [] for k in keys}
        for row in rows:
            for k in keys:
                examples[k].append(row.get(k))
        return examples

    @staticmethod
    def _yield_packed_examples(packed: dict[str, list[Any]]):
        if not packed:
            return

        keys = list(packed.keys())
        if not keys:
            return

        first = packed.get(keys[0])
        if not isinstance(first, list) or len(first) == 0:
            return

        n = len(first)
        for k in keys[1:]:
            v = packed.get(k)
            if isinstance(v, list) and len(v) != n:
                raise ValueError(
                    "Dynamic prompt packing produced inconsistent packed column lengths: "
                    f"len({k})={len(v)} vs len({keys[0]})={n}"
                )

        for i in range(n):
            yield {k: packed[k][i] for k in keys}

    @staticmethod
    def _queue_put(q: Queue, item: Any, stop: threading.Event, timeout_s: float = 0.5) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=timeout_s)
                return True
            except Full:
                continue
        return False

    def __iter__(self):
        # Repeat indefinitely. Training length should be controlled via `max_steps`.
        while True:
            raw_iter = iter(self.iterable_ds)
            yield from self._iter_one_pass_with_prefetch(raw_iter)

    def _iter_one_pass_with_prefetch(self, raw_iter):
        sentinel = object()
        q: Queue = Queue(maxsize=self.prefetch_buffers)
        stop = threading.Event()
        error: list[BaseException] = []

        def producer():
            try:
                carry_items: list[dict[str, Any]] = []
                carry_lengths: list[int] = []
                while not stop.is_set():
                    examples = self._take_raw_batch(raw_iter, self.buffer_size)
                    if examples is None:
                        break

                    if self.carryover_packs > 0:
                        items, lengths = self.packer.encode_examples(examples)
                        if items:
                            pool_items = carry_items + items
                            pool_lengths = carry_lengths + lengths
                            packed, carry_items, carry_lengths = self.packer.pack_encoded_items(
                                pool_items, pool_lengths, carryover_packs=self.carryover_packs
                            )
                        else:
                            packed = {}
                    else:
                        packed = self.packer(examples)
                    if not packed:
                        continue

                    if not self._queue_put(q, packed, stop):
                        break

                if self.carryover_packs > 0 and carry_items and not stop.is_set():
                    packed, _, _ = self.packer.pack_encoded_items(carry_items, carry_lengths, carryover_packs=0)
                    if packed:
                        self._queue_put(q, packed, stop)

                self._queue_put(q, sentinel, stop)
            except BaseException as err:  # pragma: no cover
                error.append(err)
                self._queue_put(q, sentinel, stop)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        try:
            while True:
                try:
                    buf = q.get(timeout=0.5)
                except Empty:
                    if not t.is_alive() and q.empty():
                        break
                    continue

                if buf is sentinel:
                    break

                yield from self._yield_packed_examples(buf)

            if error:
                raise error[0]
        finally:
            stop.set()
            t.join(timeout=1.0)


class DynamicPromptProcessor(DatasetProcessor):
    """Placeholder for interface consistency; not used for HF map preprocessing."""

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:  # pragma: no cover
        raise NotImplementedError("DynamicPromptProcessor should not run HF map preprocessing.")

    def print_data_example(self, example: dict[str, list[int]]) -> None:  # pragma: no cover
        logger.info_rank0(f"input_ids:\n{example['input_ids']}")
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        logger.info_rank0(f"labels:\n{valid_labels}")
