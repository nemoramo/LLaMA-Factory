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
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset, get_dataset_converter
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
    VoxtralPackedSupervisedDatasetProcessor,
    VoxtralSupervisedDatasetProcessor,
    VoxtralUnsupervisedDatasetProcessor,
)
from .processor.dynamic_prompt import (
    DynamicPromptDataset,
    build_dynamic_prompt_packed_iterable_dataset,
    build_dynamic_prompt_packed_iterable_dataset_from_iterable,
)
from .sharded_reader import ShardedParquetIterableDataset


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# Hugging Face Datasets cache compatibility
#
# Older `datasets` versions serialized list features with `_type: "List"` in
# cached `dataset_info.json` and Arrow schema metadata. Newer versions (e.g.
# datasets>=3) dropped the `List` feature type in favor of `Sequence`/`LargeList`.
# When users upgrade `datasets` without rebuilding caches, loading datasets can
# fail with:
#   ValueError: Feature type 'List' not found.
#
# We patch the Datasets feature deserializer to keep existing caches readable.
# In legacy caches, list features were serialized as `_type: "List"` (with a
# nested `feature`), whereas newer versions use either:
#   - `Sequence(feature=...)` for lists of scalars, or
#   - Python list shorthand (`[{...}]`) for lists of structs.
# The compatibility shim below converts legacy `List` to the appropriate modern
# representation based on the inner `feature` type.
# ---------------------------------------------------------------------------
try:
    from datasets.features import features as _ds_features  # type: ignore

    if hasattr(_ds_features, "generate_from_dict") and hasattr(_ds_features, "_FEATURE_TYPES"):
        _orig_generate_from_dict = _ds_features.generate_from_dict  # type: ignore[attr-defined]

        def _generate_from_dict_compat(obj: Any):  # type: ignore[no-redef]
            if isinstance(obj, list):
                return [_generate_from_dict_compat(v) for v in obj]

            if not isinstance(obj, dict) or "_type" not in obj or isinstance(obj.get("_type"), dict):
                if isinstance(obj, dict):
                    return {k: _generate_from_dict_compat(v) for k, v in obj.items()}
                return obj

            if obj.get("_type") == "List":
                obj = dict(obj)
                obj.pop("_type", None)
                feature = _generate_from_dict_compat(obj.get("feature"))
                if isinstance(feature, dict):
                    return [feature]

                return _ds_features.Sequence(feature=feature, length=obj.get("length", -1))  # type: ignore[attr-defined]

            return _orig_generate_from_dict(obj)

        _ds_features.generate_from_dict = _generate_from_dict_compat  # type: ignore[attr-defined]
except Exception:
    pass


class _LazyAlignTransform:
    """A picklable callable transform for HF `Dataset.with_transform(...)`.

    Must correctly handle:
      - single example access: dict of scalars/structures
      - batched access: dict of lists/arrays (e.g. slicing)

    Notes:
      - `datasets.Dataset.with_transform` usually calls the transform with batched inputs (even for single index),
        but we keep this robust to single-example dicts to avoid subtle format-dependent bugs.
    """

    def __init__(self, dataset_converter: Any, id_key: Optional[str]) -> None:
        self.dataset_converter = dataset_converter
        self.id_key = id_key

    @staticmethod
    def _looks_like_chat_message_dict(x: Any) -> bool:
        """Detect common chat message dict variants to avoid mis-classifying messages as a batch."""
        if not isinstance(x, dict):
            return False

        # OpenAI style: {"role": "...", "content": "..."}
        if "role" in x and "content" in x:
            return True

        # ShareGPT style: {"from": "...", "value": "..."}
        if "from" in x and "value" in x:
            return True

        # Other common variants (best-effort): {"speaker": "...", "text"/"content": "..."}
        if "speaker" in x and ("text" in x or "content" in x):
            return True

        # e.g. {"type": "...", "text": "..."}
        if "type" in x and "text" in x:
            return True

        return False

    @staticmethod
    def _is_batched_examples(examples: dict[str, Any]) -> tuple[bool, int]:
        """Heuristically detect dict-of-lists/arrays batch input.

        Consider it batched if all values are list/tuple/np.ndarray with the same length.
        This avoids mis-detecting single examples that contain list fields (e.g. sharegpt conversations),
        as long as there exists at least one non-list column (common case).
        """
        batch_size: Optional[int] = None
        for v in examples.values():
            # Guard: list[{"role","content"}] at the top-level is likely a single-example messages list.
            # This protects edge cases like {"messages": [{"role":..., "content":...}, ...]} being misinterpreted as a batch.
            if isinstance(v, (list, tuple)) and len(v) > 0 and _LazyAlignTransform._looks_like_chat_message_dict(v[0]):
                return False, 1

            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    return False, 1
                n = len(v)
            elif isinstance(v, (list, tuple)):
                n = len(v)
            else:
                return False, 1

            if batch_size is None:
                batch_size = n
            elif n != batch_size:
                return False, 1

        return True, int(batch_size or 0)

    def __call__(self, examples: dict[str, Any]) -> dict[str, Any]:
        if not examples:
            return {}

        is_batched, batch_size = self._is_batched_examples(examples)

        # -------- single example --------
        if not is_batched:
            aligned = self.dataset_converter(examples)
            if isinstance(self.id_key, str) and self.id_key and self.id_key in examples:
                aligned[self.id_key] = examples[self.id_key]
            return aligned

        # -------- batched examples --------
        outputs: dict[str, list[Any]] = {}

        for i in range(batch_size):
            row = {k: (v[i] if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in examples.items()}
            aligned = self.dataset_converter(row)
            for k, val in aligned.items():
                outputs.setdefault(k, []).append(val)

        # Preserve requested id_key if present in original examples.
        if isinstance(self.id_key, str) and self.id_key and self.id_key in examples:
            v = examples[self.id_key]
            outputs[self.id_key] = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v] * batch_size

        return outputs


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    lazy_align: bool = False,
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    if lazy_align:
        if data_args.streaming:
            raise ValueError("Lazy alignment is incompatible with `streaming`. Enable eager alignment instead.")
        if dataset_attr.formatting not in ["alpaca", "sharegpt", "openai"]:
            raise ValueError(f"Lazy alignment does not support formatting: {dataset_attr.formatting}.")

        # Dynamic prompt packing has its own on-the-fly dataset conversion path via `dataset_converter`,
        # so we can skip the expensive full-dataset alignment (`dataset.map`) for large JSONLs.
        if (
            getattr(data_args, "dynamic_prompt_sampling", False) or getattr(data_args, "dynamic_prompt_packing", False)
        ) and data_args.packing:
            logger.info_rank0(
                f"Dynamic prompt packing enabled: skip alignment for dataset {dataset_attr}; "
                "conversion will run on-the-fly during training."
            )
            return dataset

        dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
        transform = _LazyAlignTransform(dataset_converter=dataset_converter, id_key=data_args.dynamic_prompt_id_key)
        dataset = dataset.with_transform(transform, output_all_columns=False)
        logger.info_rank0(f"Lazy-aligned dataset {dataset_attr}: conversion will run on-the-fly.")
        return dataset

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    lazy_align: bool = False,
    return_dict: bool = False,
) -> Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    if (
        lazy_align
        and len(dataset_names) > 1
        and not (
            (
                getattr(data_args, "dynamic_prompt_sampling", False)
                or getattr(data_args, "dynamic_prompt_packing", False)
            )
            and data_args.packing
            and stage == "sft"
        )
    ):
        raise ValueError("Lazy alignment currently supports a single dataset.")

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(
            dataset_attr, model_args, data_args, training_args, lazy_align=lazy_align
        )

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    is_voxtral = processor is not None and processor.__class__.__name__ == "VoxtralProcessor"
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing and data_args.neat_packing:  # hack datasets to have int32 attention mask
            from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

            def __init__(self, data, **kwargs):
                return TypedSequence.__init__(
                    self,
                    data,
                    type=kwargs.pop("type", None),
                    try_type=kwargs.pop("try_type", None),
                    optimized_int_type=kwargs.pop("optimized_int_type", None),
                )

            OptimizedTypedSequence.__init__ = __init__

        if is_voxtral:
            dataset_processor_class = (
                VoxtralPackedSupervisedDatasetProcessor if data_args.packing else VoxtralSupervisedDatasetProcessor
            )
        elif data_args.packing:
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = VoxtralUnsupervisedDatasetProcessor if is_voxtral else UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Union["Dataset", "IterableDataset"] | None,
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Union["Dataset", "IterableDataset"] | None:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    if (
        (data_args.dynamic_prompt_sampling or getattr(data_args, "dynamic_prompt_packing", False))
        and stage == "sft"
        and not is_eval
    ):
        if data_args.streaming:
            raise ValueError("On-the-fly dynamic prompt (sampling/packing) does not support streaming datasets.")
        logger.info_rank0("On-the-fly dynamic prompt enabled: skip tokenization for training dataset.")
        return dataset

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    sharded_backend = str(getattr(data_args, "sharded_dataset_backend", "off") or "off").strip()
    sharded_manifest_path = getattr(data_args, "sharded_manifest_path", None)
    use_sharded_train = (
        stage == "sft"
        and not data_args.streaming
        and bool(getattr(data_args, "packing", False))
        and (data_args.dynamic_prompt_sampling or getattr(data_args, "dynamic_prompt_packing", False))
        and sharded_backend != "off"
        and isinstance(sharded_manifest_path, str)
        and sharded_manifest_path.strip()
    )

    if use_sharded_train:
        if float(getattr(data_args, "val_size", 0.0) or 0.0) > 0:
            raise ValueError(
                "Sharded parquet backend does not support `val_size` splitting. "
                "Set `val_size=0` and provide `eval_dataset` explicitly (or disable sharded backend)."
            )

        if getattr(data_args, "dynamic_prompt_packing", False) and not data_args.packing:
            raise ValueError("`dynamic_prompt_packing` requires `packing=true` (SFT only).")

        max_samples_per_pack = int(getattr(data_args, "dynamic_prompt_packing_max_samples_per_pack", 8) or 8)
        buffer_size = int(getattr(data_args, "dynamic_prompt_packing_buffer_size", 20000) or 20000)
        shuffle_packs = bool(getattr(data_args, "dynamic_prompt_packing_shuffle", True))
        prefetch_buffers = int(getattr(data_args, "dynamic_prompt_packing_prefetch_buffers", 0) or 0)
        carryover_packs = int(getattr(data_args, "dynamic_prompt_packing_carryover_packs", 0) or 0)

        max_steps = int(getattr(training_args, "max_steps", 0) or 0)
        if max_steps <= 0:
            raise ValueError(
                "Sharded parquet backend uses an IterableDataset without `__len__`; please set `max_steps` > 0 "
                "(and optionally `num_train_epochs=1`)."
            )

        # Prefer per-rank dataloading for iterable datasets. Without this, Accelerate may default to
        # `dispatch_batches=True` (rank0 iterates + broadcast), causing cross-rank duplication.
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size > 1 and hasattr(training_args, "accelerator_config"):
            cfg = training_args.accelerator_config
            if cfg is None:
                logger.warning_rank0(
                    "Sharded parquet backend: `training_args.accelerator_config` is None; cannot auto-set "
                    "`dispatch_batches`. If you observe cross-rank duplication, consider setting "
                    "`dispatch_batches: false` in your Accelerate config."
                )
            elif isinstance(cfg, dict):
                if cfg.get("dispatch_batches", None) is None:
                    cfg["dispatch_batches"] = False
                    logger.info_rank0(
                        "Sharded parquet backend: set `dispatch_batches: false` to enable per-rank sharded dataloading."
                    )
                elif cfg.get("dispatch_batches", False):
                    logger.warning_rank0(
                        "Sharded parquet backend: detected `dispatch_batches: true`. This may cause rank0-only "
                        "iteration + broadcast, leading to cross-rank duplication. Consider setting "
                        "`dispatch_batches: false`."
                    )
            else:
                if getattr(cfg, "dispatch_batches", None) is None:
                    cfg.dispatch_batches = False
                    logger.info_rank0(
                        "Sharded parquet backend: set `dispatch_batches: false` to enable per-rank sharded dataloading."
                    )
                elif getattr(cfg, "dispatch_batches", False):
                    logger.warning_rank0(
                        "Sharded parquet backend: detected `dispatch_batches: true`. This may cause rank0-only "
                        "iteration + broadcast, leading to cross-rank duplication. Consider setting "
                        "`dispatch_batches: false`."
                    )

        # Load eval dataset (train dataset is read from sharded parquet).
        with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
            eval_dataset = _get_merged_dataset(
                data_args.eval_dataset,
                model_args,
                data_args,
                training_args,
                stage,
                lazy_align=False,
                return_dict=data_args.eval_on_each_dataset,
            )

        dataset_module = {}
        with training_args.main_process_first(
            desc="pre-process dataset", local=(not data_args.data_shared_file_system)
        ):
            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    eval_out: dict[str, Any] = {}
                    for key, ds in eval_dataset.items():
                        eval_out[key] = _get_preprocessed_dataset(
                            ds, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                        )
                    dataset_module["eval_dataset"] = eval_out
                else:
                    dataset_module["eval_dataset"] = _get_preprocessed_dataset(
                        eval_dataset,
                        data_args,
                        training_args,
                        stage,
                        template,
                        tokenizer,
                        processor,
                        is_eval=True,
                    )

            input_aligned = bool(getattr(data_args, "sharded_input_aligned", False))
            dataset_converter = None
            id_key = None

            if not input_aligned:
                dataset_names = data_args.dataset or []
                if len(dataset_names) == 0:
                    raise ValueError("Sharded parquet backend: `dataset` is empty.")

                dataset_attrs = get_dataset_list(dataset_names, data_args.dataset_dir)
                if not dataset_attrs:
                    raise ValueError("Sharded parquet backend: failed to resolve dataset list.")

                # Pick a "base" conversion schema for on-the-fly alignment.
                dataset_attr = dataset_attrs[0]
                best_score = 0
                for cand in dataset_attrs:
                    score = 0
                    for field in ("tools", "images", "videos", "audios"):
                        if getattr(cand, field, None):
                            score += 1
                    if score > best_score:
                        dataset_attr = cand
                        best_score = score

                def _modality_compatible(base_val: Any, other_val: Any) -> bool:
                    if base_val is None:
                        return other_val is None
                    return other_val is None or other_val == base_val

                for other in dataset_attrs[1:]:
                    if (
                        other.formatting != dataset_attr.formatting
                        or other.messages != dataset_attr.messages
                        or other.system != dataset_attr.system
                        or not _modality_compatible(dataset_attr.tools, other.tools)
                        or not _modality_compatible(dataset_attr.images, other.images)
                        or not _modality_compatible(dataset_attr.videos, other.videos)
                        or not _modality_compatible(dataset_attr.audios, other.audios)
                        or other.role_tag != dataset_attr.role_tag
                        or other.content_tag != dataset_attr.content_tag
                        or other.user_tag != dataset_attr.user_tag
                        or other.assistant_tag != dataset_attr.assistant_tag
                        or other.observation_tag != dataset_attr.observation_tag
                        or other.function_tag != dataset_attr.function_tag
                        or other.system_tag != dataset_attr.system_tag
                    ):
                        raise ValueError(
                            "Sharded parquet backend with on-the-fly alignment requires all mixed datasets "
                            "to share the same `formatting` and column/tag mapping. "
                            f"Got mismatch between {dataset_attr.dataset_name} and {other.dataset_name}."
                        )

                dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
                id_key = data_args.dynamic_prompt_id_key
                logger.info_rank0(
                    "Sharded parquet backend: dataset is not aligned; will convert to `_prompt/_response/...` on-the-fly."
                )
            else:
                logger.info_rank0("Sharded parquet backend: input is aligned (_prompt/_response present).")

            shuffle_shards = bool(getattr(data_args, "sharded_shuffle_shards", True))
            row_shuffle_buffer = int(getattr(data_args, "sharded_row_shuffle_buffer", 0) or 0)
            row_group_shuffle = bool(getattr(data_args, "sharded_row_group_shuffle", False))
            row_group_shuffle_block_size = int(getattr(data_args, "sharded_row_group_shuffle_block_size", 0) or 0)
            parquet_batch_rows = int(getattr(data_args, "sharded_parquet_batch_rows", 8192) or 8192)
            prefetch_next_shard = bool(getattr(data_args, "sharded_prefetch_next_shard", True))
            prefetch_queue_batches = int(getattr(data_args, "sharded_prefetch_queue_batches", 1) or 0)
            prefetch_log = bool(getattr(data_args, "sharded_prefetch_log", False))
            resume_mode = str(getattr(data_args, "sharded_resume_mode", "off") or "off")
            resume_state_dir = getattr(data_args, "sharded_resume_state_dir", None)
            resume_prefer_checkpoint = bool(getattr(data_args, "sharded_resume_prefer_checkpoint", True))
            resume_log = bool(getattr(data_args, "sharded_resume_log", False))
            resume_from_checkpoint = getattr(training_args, "resume_from_checkpoint", None)
            if resume_mode != "off":
                resolved_state_dir = (
                    str(resume_state_dir)
                    if isinstance(resume_state_dir, str) and resume_state_dir
                    else os.path.join(str(getattr(training_args, "output_dir", "") or ""), "shard_resume_state")
                )
                if resolved_state_dir:
                    os.environ.setdefault("LLAMAFACTORY_SHARDED_RESUME_STATE_DIR", resolved_state_dir)

            try:
                import pyarrow.parquet  # type: ignore  # noqa: F401
            except Exception as err:  # noqa: BLE001
                raise ImportError(
                    "Sharded parquet backend requires `pyarrow`. "
                    "Install it in the training environment (not user-site when PYTHONNOUSERSITE=1), e.g. "
                    "`pip install pyarrow` or `conda install -c conda-forge pyarrow`."
                ) from err

            raw_train_ds = ShardedParquetIterableDataset(
                manifest_path=str(sharded_manifest_path),
                seed=int(training_args.seed),
                shuffle_shards=shuffle_shards,
                row_shuffle_buffer=row_shuffle_buffer,
                row_group_shuffle=row_group_shuffle,
                row_group_shuffle_block_size=row_group_shuffle_block_size,
                parquet_batch_rows=parquet_batch_rows,
                prefetch_next_shard=prefetch_next_shard,
                prefetch_queue_batches=prefetch_queue_batches,
                prefetch_log=prefetch_log,
                resume_mode=resume_mode,
                resume_state_dir=resume_state_dir,
                resume_prefer_checkpoint=resume_prefer_checkpoint,
                resume_log=resume_log,
                output_dir=str(getattr(training_args, "output_dir", "") or ""),
                resume_from_checkpoint=str(resume_from_checkpoint) if isinstance(resume_from_checkpoint, str) else None,
            )

            dataset_module["train_dataset"] = build_dynamic_prompt_packed_iterable_dataset_from_iterable(
                raw_train_ds,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
                dataset_converter=dataset_converter,
                id_key=id_key,
                seed=training_args.seed,
                buffer_size=buffer_size,
                max_samples_per_pack=max_samples_per_pack,
                shuffle_packs=shuffle_packs,
                prefetch_buffers=prefetch_buffers,
                carryover_packs=carryover_packs,
            )

            logger.info_rank0("Wrapped train dataset with sharded parquet reader + buffered knapsack packing.")
            logger.info_rank0(
                "Note: on-the-fly packing changes epoch semantics (raw samples/tokens per step vary); "
                "prefer controlling training budget via `max_steps`."
            )

        return dataset_module

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        lazy_align_train = (
            (data_args.dynamic_prompt_sampling or getattr(data_args, "dynamic_prompt_packing", False))
            and (getattr(data_args, "dynamic_prompt_lazy_align", True) or bool(data_args.packing))
            and stage == "sft"
        )
        dataset = _get_merged_dataset(
            data_args.dataset, model_args, data_args, training_args, stage, lazy_align=lazy_align_train
        )
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            lazy_align=False,
            return_dict=data_args.eval_on_each_dataset,
        )

    dataset_module = None
    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        # move front to make sure eval_dataset(if contain or split) can preprocessed appropriately
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)

        if "train" in train_dict:
            train_dict["train"] = _get_preprocessed_dataset(
                train_dict["train"], data_args, training_args, stage, template, tokenizer, processor, is_eval=False
            )

        for key in eval_dict:
            eval_dict[key] = _get_preprocessed_dataset(
                eval_dict[key], data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        # Combine train and eval dictionaries
        dataset_dict = DatasetDict({**train_dict, **eval_dict})

        if (
            data_args.tokenized_path is not None
            and not data_args.dynamic_prompt_sampling
            and not getattr(data_args, "dynamic_prompt_packing", False)
        ):  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        dataset_module = get_dataset_module(dataset_dict)
        if (
            (data_args.dynamic_prompt_sampling or getattr(data_args, "dynamic_prompt_packing", False))
            and stage == "sft"
            and not data_args.streaming
        ):
            train_ds = dataset_module.get("train_dataset")
            if train_ds is not None:
                if getattr(data_args, "dynamic_prompt_packing", False) and not data_args.packing:
                    raise ValueError("`dynamic_prompt_packing` requires `packing=true` (SFT only).")

                if data_args.packing:
                    max_samples_per_pack = int(
                        getattr(data_args, "dynamic_prompt_packing_max_samples_per_pack", 8) or 8
                    )
                    max_steps = int(getattr(training_args, "max_steps", 0) or 0)
                    if max_steps <= 0:
                        raise ValueError(
                            "Dynamic prompt packing uses an IterableDataset without `__len__`; please set `max_steps` > 0 "
                            "(and optionally `num_train_epochs=1`)."
                        )

                    # Prefer per-rank dataloading for dynamic prompt packing. Without this, Accelerate may default to
                    # `dispatch_batches=True` for iterable datasets, which iterates only on rank0 and broadcasts.
                    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
                    if world_size > 1 and hasattr(training_args, "accelerator_config"):
                        cfg = training_args.accelerator_config
                        if cfg is None:
                            logger.warning_rank0(
                                "Dynamic prompt packing: `training_args.accelerator_config` is None; cannot auto-set "
                                "`dispatch_batches`. If you observe cross-rank duplication, consider setting "
                                "`dispatch_batches: false` in your Accelerate config."
                            )
                        elif isinstance(cfg, dict):
                            if cfg.get("dispatch_batches", None) is None:
                                cfg["dispatch_batches"] = False
                                logger.info_rank0(
                                    "Dynamic prompt packing: set `dispatch_batches: false` to enable per-rank sharded dataloading."
                                )
                            elif cfg.get("dispatch_batches", False):
                                logger.warning_rank0(
                                    "Dynamic prompt packing: detected `dispatch_batches: true`. This may cause rank0-only "
                                    "iteration + broadcast, leading to cross-rank duplication. Consider setting "
                                    "`dispatch_batches: false`."
                                )
                        else:
                            if getattr(cfg, "dispatch_batches", None) is None:
                                cfg.dispatch_batches = False
                                logger.info_rank0(
                                    "Dynamic prompt packing: set `dispatch_batches: false` to enable per-rank sharded dataloading."
                                )
                            elif getattr(cfg, "dispatch_batches", False):
                                logger.warning_rank0(
                                    "Dynamic prompt packing: detected `dispatch_batches: true`. This may cause rank0-only "
                                    "iteration + broadcast, leading to cross-rank duplication. Consider setting "
                                    "`dispatch_batches: false`."
                                )

                    buffer_size = int(getattr(data_args, "dynamic_prompt_packing_buffer_size", 20000) or 20000)
                    shuffle_packs = bool(getattr(data_args, "dynamic_prompt_packing_shuffle", True))
                    prefetch_buffers = int(getattr(data_args, "dynamic_prompt_packing_prefetch_buffers", 0) or 0)
                    carryover_packs = int(getattr(data_args, "dynamic_prompt_packing_carryover_packs", 0) or 0)

                    num_shards = int(getattr(data_args, "dynamic_prompt_packing_num_shards", 0) or 0)
                    if num_shards <= 0:
                        # Ensure `num_shards` is large enough for Accelerate to shard by data sources
                        # when `dispatch_batches=False` (avoids cross-rank duplication).
                        num_workers = int(getattr(training_args, "dataloader_num_workers", 0) or 0)
                        base = max(1, world_size * max(1, num_workers))
                        num_shards = base * 8
                        if world_size > 1:
                            num_shards = ((num_shards + world_size - 1) // world_size) * world_size
                        # Cap num_shards by dataset size when possible to avoid excessive empty shards on small datasets.
                        try:
                            n = len(train_ds)
                            if isinstance(n, int) and n > 0:
                                num_shards = min(num_shards, n)
                                if world_size > 1 and n >= world_size:
                                    num_shards = max(world_size, (num_shards // world_size) * world_size)
                        except Exception:
                            pass
                        # HuggingFace `Dataset.to_iterable_dataset(num_shards=...)` can be extremely slow when
                        # `num_shards` is very large on huge concatenated datasets. Cap it to a reasonable default.
                        if num_shards > 2048:
                            num_shards = 2048
                            if world_size > 1:
                                num_shards = max(world_size, (num_shards // world_size) * world_size)

                    global_shuffle = bool(getattr(data_args, "dynamic_prompt_packing_global_shuffle", True))
                    dataset_converter = None
                    id_key = None
                    col_names = getattr(train_ds, "column_names", None)
                    is_aligned = (
                        isinstance(col_names, (list, tuple)) and "_prompt" in col_names and "_response" in col_names
                    )
                    if not is_aligned:
                        dataset_names = data_args.dataset or []
                        if len(dataset_names) == 0:
                            raise ValueError("Dynamic prompt packing: `dataset` is empty.")

                        dataset_attrs = get_dataset_list(dataset_names, data_args.dataset_dir)
                        # Pick a "base" conversion schema for on-the-fly alignment.
                        #
                        # NOTE: On-the-fly alignment only supports a single `dataset_converter`, so all mixed datasets
                        # must be compatible with that schema. We allow some datasets to *omit* optional modality
                        # columns (e.g. text-only dataset mixed with audio dataset) as long as they otherwise share the
                        # same formatting + message/tag mapping, and they do not use a different column name for the
                        # same modality (which would silently drop data).
                        dataset_attr = dataset_attrs[0]
                        best_score = 0
                        for cand in dataset_attrs:
                            score = 0
                            for field in ("tools", "images", "videos", "audios"):
                                if getattr(cand, field, None):
                                    score += 1
                            if score > best_score:
                                dataset_attr = cand
                                best_score = score

                        def _modality_compatible(base_val: Any, other_val: Any) -> bool:
                            if base_val is None:
                                return other_val is None
                            return other_val is None or other_val == base_val

                        # When mixing multiple datasets, dynamic prompt packing can still run alignment on-the-fly
                        # as long as their conversion schema is identical (same formatting + column/tag mapping).
                        for other in dataset_attrs[1:]:
                            if (
                                other.formatting != dataset_attr.formatting
                                or other.messages != dataset_attr.messages
                                or other.system != dataset_attr.system
                                or not _modality_compatible(dataset_attr.tools, other.tools)
                                or not _modality_compatible(dataset_attr.images, other.images)
                                or not _modality_compatible(dataset_attr.videos, other.videos)
                                or not _modality_compatible(dataset_attr.audios, other.audios)
                                or other.role_tag != dataset_attr.role_tag
                                or other.content_tag != dataset_attr.content_tag
                                or other.user_tag != dataset_attr.user_tag
                                or other.assistant_tag != dataset_attr.assistant_tag
                                or other.observation_tag != dataset_attr.observation_tag
                                or other.function_tag != dataset_attr.function_tag
                                or other.system_tag != dataset_attr.system_tag
                            ):
                                raise ValueError(
                                    "Dynamic prompt packing with on-the-fly alignment requires all mixed datasets "
                                    "to share the same `formatting` and column/tag mapping. "
                                    f"Got mismatch between {dataset_attr.dataset_name} and {other.dataset_name}."
                                )

                        dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
                        id_key = data_args.dynamic_prompt_id_key
                        logger.info_rank0(
                            "Dynamic prompt packing: dataset is not aligned; will convert to `_prompt/_response/...` on-the-fly."
                        )

                    dataset_module["train_dataset"] = build_dynamic_prompt_packed_iterable_dataset(
                        train_ds,
                        template=template,
                        tokenizer=tokenizer,
                        processor=processor,
                        data_args=data_args,
                        dataset_converter=dataset_converter,
                        id_key=id_key,
                        seed=training_args.seed,
                        buffer_size=buffer_size,
                        max_samples_per_pack=max_samples_per_pack,
                        shuffle_packs=shuffle_packs,
                        num_shards=num_shards,
                        global_shuffle=global_shuffle,
                        prefetch_buffers=prefetch_buffers,
                        carryover_packs=carryover_packs,
                    )
                    logger.info_rank0(
                        "Wrapped train dataset with buffered knapsack packing (on-the-fly encode + on-the-fly pack)."
                    )

                    logger.info_rank0(
                        "Note: on-the-fly packing changes epoch semantics (raw samples/tokens per step vary); "
                        "prefer controlling training budget via `max_steps`."
                    )
                else:
                    dataset_module["train_dataset"] = DynamicPromptDataset(
                        train_ds,
                        template=template,
                        tokenizer=tokenizer,
                        processor=processor,
                        data_args=data_args,
                        seed=training_args.seed,
                    )
                    logger.info_rank0(
                        "Wrapped train dataset with DynamicPromptDataset for on-the-fly prompt sampling."
                    )

    # NOTE:
    # `training_args.main_process_first` ensures non-rank0 processes wait until rank0 reaches the end of the context,
    # but rank0 does not necessarily wait for other ranks to finish the body before proceeding. This can lead to
    # collective ops timeouts when rank0 starts model init/training while other ranks are still preparing datasets.
    #
    # Synchronize here so all ranks finish dataset loading/preprocessing/wrapping before leaving `get_dataset()`.
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
            if dist.get_backend() == "nccl":
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()
    except Exception as err:
        logger.warning_rank0(f"Failed to synchronize after dataset preprocessing: {err}")

    return dataset_module
