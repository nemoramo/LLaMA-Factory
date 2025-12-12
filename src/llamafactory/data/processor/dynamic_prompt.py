from __future__ import annotations

import copy
import math
import random
from typing import Any, Optional, Sequence

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
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.data_args = data_args

        # RNG is created lazily and seeded per worker/rank to avoid identical sampling streams.
        self._base_seed = seed
        self._rng: Optional[random.Random] = None
        self._rng_seeded: bool = False

        # Reuse the supervised processor for encoding logic.
        self.encoder = SupervisedDatasetProcessor(
            template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
        )

        # Fail-fast schema check (helps catch "loaded tokenized dataset" mistakes).
        column_names = getattr(dataset, "column_names", None)
        if isinstance(column_names, (list, tuple, set)):
            required = {"_prompt", "_response"}
            missing = required - set(column_names)
            if missing:
                raise ValueError(
                    "DynamicPromptDataset expects an aligned dataset with columns "
                    f"{sorted(required)}, but missing: {sorted(missing)}. "
                    "This often happens when loading a tokenized dataset (`tokenized_path`) "
                    "or when alignment/conversion did not run."
                )

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

    def _sample_pool_choice(self, example: dict[str, Any]) -> Any | None:
        pool = example.get("_prompt_pool")
        if not (isinstance(pool, list) and len(pool) > 0):
            return None
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


class DynamicPromptProcessor(DatasetProcessor):
    """Placeholder for interface consistency; not used for HF map preprocessing."""

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:  # pragma: no cover
        raise NotImplementedError("DynamicPromptProcessor should not run HF map preprocessing.")

    def print_data_example(self, example: dict[str, list[int]]) -> None:  # pragma: no cover
        logger.info_rank0(f"input_ids:\n{example['input_ids']}")
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        logger.info_rank0(f"labels:\n{valid_labels}")
