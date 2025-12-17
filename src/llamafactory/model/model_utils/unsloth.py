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

import inspect
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.misc import get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    r"""Best-effort filter kwargs to match function signature.

    Unsloth occasionally changes parameter names across versions (e.g. `max_seq_len` vs `max_seq_length`).
    We keep this helper minimal and robust to avoid hard pinning on a single Unsloth release.
    """
    try:
        sig = inspect.signature(func)
    except Exception:
        return kwargs

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _rename_kwargs_for_signature(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Rename a few common Unsloth kwargs based on the target signature."""
    try:
        params = inspect.signature(func).parameters
    except Exception:
        return kwargs

    out = dict(kwargs)
    if "max_seq_length" in out and "max_seq_length" not in params and "max_seq_len" in params:
        out["max_seq_len"] = out.pop("max_seq_length")
    return out


def _get_unsloth_loaders(config: "PretrainedConfig") -> list[Any]:
    r"""Return candidate Unsloth loader classes in preferred order for a model config."""
    import unsloth  # type: ignore

    FastLanguageModel = getattr(unsloth, "FastLanguageModel", None)
    FastModel = getattr(unsloth, "FastModel", None)

    model_type = getattr(config, "model_type", None)
    loaders: list[Any] = []

    # Gemma 3 / Gemma 3n multimodal models are served by `FastModel` in upstream notebooks.
    if FastModel is not None and model_type in ("gemma3", "gemma3n"):
        loaders.append(FastModel)

    if FastLanguageModel is not None:
        loaders.append(FastLanguageModel)

    if FastModel is not None and FastModel not in loaders:
        loaders.append(FastModel)

    return loaders


def _get_unsloth_kwargs(
    config: "PretrainedConfig",
    model_name_or_path: str,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> dict[str, Any]:
    return {
        "model_name": model_name_or_path,
        "max_seq_length": model_args.model_max_length or 4096,
        "dtype": model_args.compute_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "full_finetuning": finetuning_args.finetuning_type == "full",
        "device_map": {"": get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
        "fix_tokenizer": False,
        "trust_remote_code": model_args.trust_remote_code,
        "use_gradient_checkpointing": "unsloth",
    }


def load_unsloth_pretrained_model(
    config: "PretrainedConfig", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["PreTrainedModel"]:
    r"""Optionally load pretrained model with unsloth. Used in training."""
    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.model_name_or_path, model_args, finetuning_args)
    last_error: Exception | None = None
    for loader_cls in _get_unsloth_loaders(config):
        try:
            call_kwargs = _rename_kwargs_for_signature(loader_cls.from_pretrained, unsloth_kwargs)
            call_kwargs = _filter_kwargs(loader_cls.from_pretrained, call_kwargs)
            model, _ = loader_cls.from_pretrained(**call_kwargs)
            logger.info_rank0(f"Loaded model with Unsloth backend: {loader_cls.__name__}.")
            return model
        except NotImplementedError as e:
            last_error = e
            continue

    logger.warning_rank0(
        "Unsloth does not support model type {} (or missing FastModel for multimodal). Error: {}".format(
            getattr(config, "model_type", None), last_error
        )
    )
    model_args.use_unsloth = False
    return None


def get_unsloth_peft_model(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    peft_kwargs: dict[str, Any],
) -> "PreTrainedModel":
    r"""Get the peft model for the pretrained model with unsloth. Used in training."""
    import unsloth  # type: ignore

    FastLanguageModel = getattr(unsloth, "FastLanguageModel", None)
    FastModel = getattr(unsloth, "FastModel", None)

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    max_seq_length = model_args.model_max_length or 4096
    if FastModel is not None and model_type in ("gemma3", "gemma3n"):
        # Match upstream Gemma3n notebook defaults as closely as possible.
        extra_kwargs: dict[str, Any] = {
            "model": model,
            "finetune_vision_layers": not finetuning_args.freeze_vision_tower,
            "finetune_language_layers": not finetuning_args.freeze_language_model,
            "finetune_attention_modules": True,
            "finetune_mlp_modules": True,
            "max_seq_length": max_seq_length,
            "use_gradient_checkpointing": "unsloth",
        }
        merged = {**peft_kwargs, **extra_kwargs}
        merged = _rename_kwargs_for_signature(FastModel.get_peft_model, merged)
        merged = _filter_kwargs(FastModel.get_peft_model, merged)
        return FastModel.get_peft_model(**merged)

    if FastLanguageModel is None:
        raise ImportError("Unsloth `FastLanguageModel` not found; please upgrade/downgrade unsloth.")

    unsloth_peft_kwargs = {
        "model": model,
        "max_seq_length": max_seq_length,
        "use_gradient_checkpointing": "unsloth",
    }
    merged = {**peft_kwargs, **unsloth_peft_kwargs}
    merged = _rename_kwargs_for_signature(FastLanguageModel.get_peft_model, merged)
    merged = _filter_kwargs(FastLanguageModel.get_peft_model, merged)
    return FastLanguageModel.get_peft_model(**merged)


def load_unsloth_peft_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Load peft model with unsloth. Used in both training and inference."""
    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.adapter_name_or_path[0], model_args, finetuning_args)
    if not is_trainable:
        unsloth_kwargs["use_gradient_checkpointing"] = False

    last_error: Exception | None = None
    for loader_cls in _get_unsloth_loaders(config):
        try:
            call_kwargs = _rename_kwargs_for_signature(loader_cls.from_pretrained, unsloth_kwargs)
            call_kwargs = _filter_kwargs(loader_cls.from_pretrained, call_kwargs)
            model, _ = loader_cls.from_pretrained(**call_kwargs)
            if not is_trainable and hasattr(loader_cls, "for_inference"):
                loader_cls.for_inference(model)
            logger.info_rank0(f"Loaded adapter with Unsloth backend: {loader_cls.__name__}.")
            return model
        except NotImplementedError as e:
            last_error = e
            continue

    raise ValueError(
        "Unsloth does not support model type {} (or missing FastModel for multimodal). Error: {}".format(
            getattr(config, "model_type", None), last_error
        )
    )
