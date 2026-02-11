# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: str | None = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    voxtral_chat_template: bool = field(
        default=False,
        metadata={
            "help": (
                "Voxtral only: use chat-style templating (ShareGPT messages) instead of the official "
                "transcription request template. Default is False (transcription template)."
            )
        },
    )
    voxtral_transcription_language: str | None = field(
        default=None,
        metadata={
            "help": (
                "Voxtral only (transcription template): language code used in the `lang: xx [TRANSCRIBE]` prefix "
                "when the dataset does not provide a per-sample language."
            )
        },
    )
    dataset: str | None = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    eval_dataset: str | None = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    media_dir: str | None = field(
        default=None,
        metadata={"help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over", "interleave_once"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling/sampling w.o. replacement)."},
    )
    interleave_probs: str | None = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: int | None = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the validation set, should be an integer or a float in range `[0,1)`."},
    )
    eval_on_each_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to evaluate on each dataset separately."},
    )
    packing: bool | None = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    tool_format: str | None = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples."},
    )
    default_system: str | None = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    enable_thinking: bool | None = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    tokenized_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={"help": "Whether or not to use a shared file system for the datasets."},
    )
    dynamic_prompt_sampling: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable dynamic prompt sampling at training time (SFT only). "
                "If a sample provides `prompt_pool`, one entry is chosen randomly per access and "
                "appended to the system prompt; during evaluation tokenization, the top-1 (max-weight) entry is used "
                "for deterministic eval loss."
            )
        },
    )

    dynamic_prompt_prompt_pool_uniform_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "When `dynamic_prompt_sampling` is enabled and a sample provides `prompt_pool`, "
                "ignore per-entry `weight` fields and sample uniformly (1/N). "
                "This avoids legacy skew (e.g. weights like 0.63/0.07/0.27/0.03). "
                "Set to false to respect the per-entry `weight` values."
            )
        },
    )


    dynamic_prompt_packing: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable on-the-fly buffered packing (encode+pack) at training time (SFT only). "
                "This uses the dynamic prompt packing pipeline (IterableDataset + buffered knapsack packing) even when "
                "`dynamic_prompt_sampling` is disabled. Requires `packing=true` and does not support streaming."
            )
        },
    )

    dynamic_prompt_lazy_align: bool = field(
        default=True,
        metadata={
            "help": (
                "When `dynamic_prompt_sampling` is enabled (SFT only), align/convert the training dataset "
                "on-the-fly via `Dataset.with_transform` instead of running a full `map` conversion up-front. "
                "Disabling this may increase startup time but can improve training throughput."
            )
        },
    )

    dynamic_prompt_deterministic: bool = field(
        default=False,
        metadata={
            "help": (
                "Use deterministic per-sample prompt sampling based on a stable hash. "
                "Useful for reproducibility and more stable resume when using dynamic prompt sampling. "
                "Note: deterministic mapping does not vary across epochs."
            )
        },
    )

    dynamic_prompt_id_key: str | None = field(
        default=None,
        metadata={
            "help": (
                "Field name to use as a stable per-sample id for deterministic dynamic prompt sampling. "
                "If not set, will try to use the first audio path in `_audios`, then fallback to the last user message."
            )
        },
    )

    dynamic_prompt_packing_global_shuffle: bool = field(
        default=True,
        metadata={
            "help": (
                "When using on-the-fly buffered packing (`dynamic_prompt_packing` or `dynamic_prompt_sampling` + `packing`), "
                "shuffle the map-style dataset once (with `seed`) before converting it to an iterable dataset for "
                "buffered encode+pack."
            )
        },
    )

    dynamic_prompt_packing_num_shards: int = field(
        default=0,
        metadata={
            "help": (
                "Number of shards used by `Dataset.to_iterable_dataset(num_shards=...)` for dynamic prompt packing. "
                "A larger value enables efficient per-rank/per-worker sharding. Use 0 for auto."
            )
        },
    )

    dynamic_prompt_packing_buffer_size: int = field(
        default=20000,
        metadata={
            "help": (
                "Buffer size for dynamic prompt packing (`dynamic_prompt_packing` or `dynamic_prompt_sampling` + `packing`). "
                "The dataset encodes/tokenizes in buffered batches and packs each buffer with greedy knapsack. "
                "Larger values improve packing efficiency but increase CPU/memory usage."
            )
        },
    )

    dynamic_prompt_packing_log_interval: int = field(
        default=0,
        metadata={"help": ("Log dynamic prompt packing progress every N buffers. Set to 0 to disable.")},
    )

    dynamic_prompt_packing_max_samples_per_pack: int = field(
        default=8,
        metadata={
            "help": (
                "Upper bound on how many raw samples are concatenated into a single packed sequence when using "
                "dynamic prompt packing."
            )
        },
    )

    dynamic_prompt_packing_shuffle: bool = field(
        default=True,
        metadata={"help": "Shuffle packed sequences inside each buffer for dynamic prompt packing."},
    )

    dynamic_prompt_packing_prefetch_buffers: int = field(
        default=2,
        metadata={
            "help": (
                "Prefetch N *packed buffers* ahead (per dataloader worker) when using dynamic prompt packing. "
                "This helps hide stalls at buffer boundaries, but increases CPU/RAM usage. "
                "Set to 0 to disable."
            )
        },
    )

    dynamic_prompt_packing_carryover_packs: int = field(
        default=0,
        metadata={
            "help": (
                "Hold back N lowest-fill packed sequences from each buffer and carry their raw segments to the next "
                "buffer, so packing can mix across buffer boundaries. This can improve packing efficiency when "
                "buffer_size is small, at the cost of slightly changing sample order. Set to 0 to disable."
            )
        },
    )

    sharded_dataset_backend: Literal["off", "polars_parquet_shards"] = field(
        default="off",
        metadata={
            "help": (
                "Enable sharded dataset backend to avoid building a full HF map-style index for huge JSONL. "
                "Currently supported: 'polars_parquet_shards' (manifest.json produced by scripts/shard_jsonl_to_parquet.py). "
                "Default is 'off'."
            )
        },
    )

    sharded_manifest_path: str | None = field(
        default=None,
        metadata={"help": "Path to shard manifest.json for sharded dataset backend."},
    )

    sharded_input_aligned: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether the sharded parquet rows already contain aligned columns like `_prompt`/`_response`. "
                "If False (default), loader will run on-the-fly alignment using the dataset config converter."
            )
        },
    )

    sharded_shuffle_shards: bool = field(
        default=True,
        metadata={"help": "Shuffle shard order per cycle when using sharded parquet backend."},
    )

    sharded_row_shuffle_buffer: int = field(
        default=0,
        metadata={
            "help": (
                "Row-level shuffle buffer size inside each shard for sharded parquet backend. "
                "Set to 0 to disable."
            )
        },
    )

    sharded_row_group_shuffle: bool = field(
        default=False,
        metadata={
            "help": (
                "Shuffle parquet row-groups (or row-group blocks) within each shard for better dataset mixing. "
                "This is typically cheaper than large row-level shuffle buffers."
            )
        },
    )

    sharded_row_group_shuffle_block_size: int = field(
        default=0,
        metadata={
            "help": (
                "When `sharded_row_group_shuffle` is enabled, shuffle coarse blocks of N row-groups. "
                "Set to 0 for full row-group shuffle, 1+ for block shuffle, or -1 for auto (aim ~1 output RecordBatch per block)."
            )
        },
    )

    sharded_parquet_batch_rows: int = field(
        default=8192,
        metadata={"help": "PyArrow parquet iter_batches batch_size (rows) for sharded parquet backend."},
    )

    sharded_prefetch_next_shard: bool = field(
        default=True,
        metadata={
            "help": (
                "Prefetch the *next* parquet shard in a background thread (per dataloader worker) when using "
                "the sharded parquet backend. Helps hide stalls at shard boundaries in DDP, at the cost of "
                "extra CPU/IO and a small amount of extra RAM."
            )
        },
    )

    sharded_prefetch_queue_batches: int = field(
        default=1,
        metadata={
            "help": (
                "Max number of parquet RecordBatches to prefetch ahead for the *next* shard (per dataloader worker). "
                "Set to 0 to disable next-shard prefetch. Keep small (1-4) to avoid RAM blowups."
            )
        },
    )

    sharded_prefetch_log: bool = field(
        default=False,
        metadata={"help": "Log shard prefetch events (rank0) for debugging sharded parquet stalls."},
    )

    sharded_resume_mode: Literal["off", "shard_boundary"] = field(
        default="off",
        metadata={
            "help": (
                "Resume mode for the sharded parquet backend. "
                "'shard_boundary' enables coarse resume at shard boundaries by persisting per-rank/worker shard cursors. "
                "This avoids re-reading completed shards after restart, but may repeat some data within the last shard."
            )
        },
    )

    sharded_resume_state_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Directory to persist sharded-resume state (JSON files). "
                "If None, defaults to `<output_dir>/shard_resume_state`."
            )
        },
    )

    sharded_resume_prefer_checkpoint: bool = field(
        default=True,
        metadata={
            "help": (
                "When resuming from a checkpoint, prefer loading shard-resume state from "
                "`<checkpoint_dir>/shard_resume_state/` if present (otherwise fall back to the output_dir state)."
            )
        },
    )

    sharded_resume_log: bool = field(
        default=False,
        metadata={"help": "Log shard-resume load/save events (rank0) for debugging."},
    )

    log_audio_epochs: bool = field(
        default=False,
        metadata={
            "help": (
                "Log data-progress in audio-hours/epochs based on per-sample audio duration (seconds). "
                "Also writes progress metadata into each checkpoint. "
                "Supports jsonl `duration`-like keys and the common segment filename pattern "
                "`*_segXXXX_start-end.wav` when duration keys are missing."
            )
        },
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

        if self.media_dir is None:
            self.media_dir = self.dataset_dir

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError("`interleave_probs` is only valid for interleaved mixing.")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
                raise ValueError("The length of dataset and interleave probs should be identical.")

            if self.eval_dataset is not None and len(self.eval_dataset) != len(self.interleave_probs):
                raise ValueError("The length of eval dataset and interleave probs should be identical.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.streaming and getattr(self, "dynamic_prompt_packing", False):
            raise ValueError("`dynamic_prompt_packing` does not support `streaming`.")

        if getattr(self, "sharded_dataset_backend", "off") != "off":
            if not isinstance(getattr(self, "sharded_manifest_path", None), str) or not self.sharded_manifest_path:
                raise ValueError("`sharded_manifest_path` is required when `sharded_dataset_backend` is enabled.")
            if self.streaming:
                raise ValueError("`sharded_dataset_backend` does not support `streaming`.")
            if self.val_size > 1e-6:
                raise ValueError("`sharded_dataset_backend` does not support `val_size` splitting; set `val_size=0`.")
        else:
            if getattr(self, "sharded_resume_mode", "off") != "off":
                raise ValueError("`sharded_resume_mode` requires enabling `sharded_dataset_backend`.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

        if self.neat_packing:
            self.packing = True

        if getattr(self, "dynamic_prompt_packing", False):
            self.packing = True

        if self.packing:
            self.cutoff_len -= 1  # avoid pad_to_multiple_of, needs improve

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
