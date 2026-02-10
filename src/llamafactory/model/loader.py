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
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from ..extras.packages import is_torch_version_greater_than
from .adapter import init_adapter
from .model_utils.ktransformers import load_kt_pretrained_model
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _try_register_funaudiochat() -> bool:
    """Best-effort registration for FunAudioChat HF classes.

    FunAudioChat checkpoints may not ship with `auto_map`, so `trust_remote_code` alone is insufficient.
    """
    try:
        from .funaudiochat.register import register_funaudiochat

        register_funaudiochat()
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Failed to register FunAudioChat: {e}.")
        return False


def _maybe_register_funaudiochat_from_error(err: Exception) -> bool:
    return "funaudiochat" in str(err).lower() and _try_register_funaudiochat()


def _try_register_qwen3_asr() -> bool:
    """Best-effort registration for Qwen3-ASR HF classes.

    We vendor a minimal Qwen3-ASR implementation under `llamafactory.model.qwen3_asr` and register its
    Auto* mappings on demand (typically when upstream transformers does not yet recognize Qwen3-ASR
    checkpoints).
    """
    try:
        from .qwen3_asr.register import register_qwen3_asr

        register_qwen3_asr()
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Failed to register Qwen3-ASR: {e}.")
        return False


def _maybe_register_qwen3_asr_from_error(err: Exception) -> bool:
    msg = str(err).lower()
    return ("qwen3_asr" in msg or "qwen3-asr" in msg) and _try_register_qwen3_asr()


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    cfg = None
    try:
        cfg = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    except Exception as e:  # noqa: BLE001
        if _maybe_register_funaudiochat_from_error(e):
            cfg = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        elif _maybe_register_qwen3_asr_from_error(e):
            cfg = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        else:
            logger.debug(f"Failed to load config before tokenizer: {e}.")

    tokenizer_extra_kwargs: dict[str, Any] = {}
    processor_extra_kwargs: dict[str, Any] = {}
    is_voxtral = getattr(cfg, "model_type", None) == "voxtral"
    if is_voxtral:
        try:
            from mistral_common.protocol.instruct.validator import ValidationMode

            tokenizer_extra_kwargs["mode"] = ValidationMode.finetuning
            processor_extra_kwargs["mode"] = ValidationMode.finetuning
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "Voxtral requires `mistral-common` to be installed. Install it with `pip install mistral-common`."
            ) from e

    try:
        if is_voxtral:
            init_kwargs = {k: v for k, v in init_kwargs.items() if k != "trust_remote_code"}
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                padding_side="right",
                **tokenizer_extra_kwargs,
                **init_kwargs,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=model_args.use_fast_tokenizer,
                split_special_tokens=model_args.split_special_tokens,
                padding_side="right",
                **tokenizer_extra_kwargs,
                **init_kwargs,
            )
    except ValueError:  # try another one
        if is_voxtral:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **tokenizer_extra_kwargs,
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        if is_voxtral:
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                **processor_extra_kwargs,
                **init_kwargs,
            )
        else:
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                use_fast=model_args.use_fast_tokenizer,
                **processor_extra_kwargs,
                **init_kwargs,
            )
    except ValueError:  # try another one
        try:
            if is_voxtral:
                raise
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                use_fast=not model_args.use_fast_tokenizer,
                **processor_extra_kwargs,
                **init_kwargs,
            )
        except Exception as e:  # noqa: BLE001
            if _maybe_register_funaudiochat_from_error(e) or _maybe_register_qwen3_asr_from_error(e):
                processor = AutoProcessor.from_pretrained(
                    model_args.model_name_or_path,
                    use_fast=not model_args.use_fast_tokenizer,
                    **processor_extra_kwargs,
                    **init_kwargs,
                )
            else:
                raise
    except Exception as e:
        if _maybe_register_funaudiochat_from_error(e) or _maybe_register_qwen3_asr_from_error(e):
            try:
                processor = AutoProcessor.from_pretrained(
                    model_args.model_name_or_path,
                    use_fast=model_args.use_fast_tokenizer,
                    **processor_extra_kwargs,
                    **init_kwargs,
                )
            except Exception as err:  # noqa: BLE001
                logger.info_rank0(f"Failed to load processor: {err}.")
                processor = None
        else:
            logger.info_rank0(f"Failed to load processor: {e}.")
            processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        # Some checkpoints (notably FunAudioChat) do not ship a proper processor `auto_map`,
        # so AutoProcessor may silently fall back to returning a tokenizer/feature-extractor.
        # Detect this case via config and retry after registration.
        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        except Exception as e:  # noqa: BLE001
            if _maybe_register_funaudiochat_from_error(e):
                cfg = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)

        if getattr(cfg, "model_type", None) == "funaudiochat":
            _try_register_funaudiochat()
            try:
                processor = AutoProcessor.from_pretrained(
                    model_args.model_name_or_path,
                    use_fast=model_args.use_fast_tokenizer,
                    **init_kwargs,
                )
            except ValueError:
                processor = AutoProcessor.from_pretrained(
                    model_args.model_name_or_path,
                    use_fast=not model_args.use_fast_tokenizer,
                    **init_kwargs,
                )
            except Exception as e:  # noqa: BLE001
                logger.info_rank0(f"Failed to reload processor after FunAudioChat registration: {e}.")

        if processor is not None and "Processor" not in processor.__class__.__name__:
            logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
            processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    try:
        return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    except Exception as e:  # noqa: BLE001
        if _maybe_register_funaudiochat_from_error(e) or _maybe_register_qwen3_asr_from_error(e):
            return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        raise


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    require_logits = (finetuning_args.stage not in ["pt", "sft"]) or getattr(
        finetuning_args, "use_chunked_ce_loss", False
    )
    apply_liger_kernel(config, model_args, is_trainable, require_logits=require_logits)

    model = None
    lazy_load = False
    if model_args.use_kt:
        from ktransformers.sft.monkey_patch_torch_module import install_patch

        install_patch()
        model = load_kt_pretrained_model(config, model_args)
    elif model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        init_kwargs["torch_dtype"] = "auto"

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif isinstance(getattr(config, "model_type", None), str) and getattr(config, "model_type").startswith(
                "qwen3_asr"
            ):
                load_class = AutoModel
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                try:
                    model = load_class.from_pretrained(**init_kwargs)
                except Exception as e:  # noqa: BLE001
                    # Qwen3-ASR models may require local registration via `qwen-asr` package or third_party submodule.
                    if isinstance(getattr(config, "model_type", None), str) and getattr(config, "model_type").startswith(
                        "qwen3_asr"
                    ) and _try_register_qwen3_asr():
                        model = load_class.from_pretrained(**init_kwargs)
                    else:
                        raise
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    # Conv3D is not recommended when using torch 2.9.x
    if is_torch_version_greater_than("2.9.0") and not is_torch_version_greater_than("2.10.0"):
        if any(isinstance(m, torch.nn.Conv3d) for m in model.modules()):
            raise ValueError(
                "Unsupported torch version detected: torch 2.9.x with Conv3D. "
                "This combination is known to cause severe performance regression. "
                "Please downgrade torch to <2.9 or remove Conv3D. "
                "See https://github.com/pytorch/pytorch/issues/166122"
            )

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    # Borrowing the kernel plugins ability of v1 to temporarily apply the NPU fusion operator to v0,
    # it is turned off by default, and can be discarded after the transition period ends.
    if model_args.use_v1_kernels and is_trainable:
        logger.warning_rank0(
            "You are try to using future feature about kernels, please note that this feature "
            "is not supported for all models. If get any error, please disable this feature, or report the issue."
        )
        from ..v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = apply_default_kernels(model, include_kernels=model_args.use_v1_kernels)

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
