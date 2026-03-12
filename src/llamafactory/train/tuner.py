# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
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

import json
import os
import shutil
import sys
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback, PreTrainedModel
from transformers.utils import cached_file

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import find_available_port, get_device_name, get_torch_device, infer_optim_dtype
from ..extras.packages import is_mcore_adapter_available, is_ray_available, is_transformers_version_greater_than
from ..hparams import RayArguments, get_infer_args, get_ray_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from .trainer_utils import (
    get_placement_group,
    get_ray_head_node_ip,
    get_ray_remote_config_for_worker,
    get_swanlab_callback,
    sort_placement_group_by_node_ip,
)


ray: Any | None = None
if is_ray_available():
    import ray as ray_module

    ray = ray_module


if TYPE_CHECKING:
    from transformers import TrainerCallback

    from ..hparams import TrainingArguments


logger = logging.get_logger(__name__)


def _require_ray() -> Any:
    if not is_ray_available():
        raise ImportError("ray is not installed. Please install it with `pip install ray` or disable Ray training.")

    return ray


def _save_training_command(args: Any, training_args: "TrainingArguments") -> None:
    """Save the current training command into output_dir for reproducibility."""
    if not getattr(training_args, "should_save", False):
        return

    try:
        os.makedirs(training_args.output_dir, exist_ok=True)
        cmd_path = os.path.join(training_args.output_dir, "training_command.txt")
        command_line = " ".join(sys.argv)
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write(f"# Saved at {datetime.utcnow().isoformat()}Z\n")
            f.write(command_line + "\n")
        logger.info_rank0(f"Training command saved to {cmd_path}.")
    except Exception as e:  # noqa: BLE001
        logger.warning_rank0(f"Failed to save training command: {e}.")


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _load_source_json(
    model_name_or_path: str | None, filename: str, model_revision: str, cache_dir: str | None, token: str | None
) -> Optional[dict[str, Any]]:
    if not model_name_or_path:
        return None

    local_path = os.path.join(model_name_or_path, filename)
    if os.path.exists(local_path):
        return _read_json(local_path)

    try:
        resolved_path = cached_file(
            model_name_or_path,
            filename,
            revision=model_revision,
            cache_dir=cache_dir,
            token=token,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Failed to resolve `{filename}` from `{model_name_or_path}`: {e}.")
        return None

    if resolved_path and os.path.exists(resolved_path):
        return _read_json(resolved_path)

    return None


def _normalize_exported_tokenizer_config(model_args: Any) -> None:
    tokenizer_config_path = os.path.join(model_args.export_dir, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return

    tokenizer_config = _read_json(tokenizer_config_path)
    source_tokenizer_config = _load_source_json(
        model_args.model_name_or_path,
        "tokenizer_config.json",
        model_args.model_revision,
        model_args.cache_dir,
        model_args.hf_hub_token,
    )
    source_tokenizer_config = source_tokenizer_config or {}

    changed = False

    source_tokenizer_class = source_tokenizer_config.get("tokenizer_class")
    if tokenizer_config.get("tokenizer_class") == "TokenizersBackend" and isinstance(source_tokenizer_class, str):
        tokenizer_config["tokenizer_class"] = source_tokenizer_class
        changed = True

    additional_special_tokens = tokenizer_config.get("additional_special_tokens")
    merged_additional_special_tokens: list[str] = []
    if isinstance(source_tokenizer_config.get("additional_special_tokens"), list):
        merged_additional_special_tokens.extend(source_tokenizer_config["additional_special_tokens"])
    if isinstance(additional_special_tokens, list):
        merged_additional_special_tokens.extend(additional_special_tokens)

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        merged_additional_special_tokens.extend(extra_special_tokens)
        tokenizer_config.pop("extra_special_tokens", None)
        changed = True

    if isinstance(getattr(model_args, "add_special_tokens", None), list):
        merged_additional_special_tokens.extend(model_args.add_special_tokens)

    merged_additional_special_tokens = list(dict.fromkeys(merged_additional_special_tokens))
    if merged_additional_special_tokens and merged_additional_special_tokens != additional_special_tokens:
        tokenizer_config["additional_special_tokens"] = merged_additional_special_tokens
        changed = True

    if not isinstance(tokenizer_config.get("extra_special_tokens"), dict):
        source_extra_special_tokens = source_tokenizer_config.get("extra_special_tokens")
        if isinstance(source_extra_special_tokens, dict) and source_extra_special_tokens:
            tokenizer_config["extra_special_tokens"] = source_extra_special_tokens
            changed = True

    if changed:
        _write_json(tokenizer_config_path, tokenizer_config)
        logger.info_rank0(f"Normalized tokenizer config for export at {tokenizer_config_path}.")


def _save_processor_sidecar_configs(processor: Any, export_dir: str) -> None:
    for attr_name in ("image_processor", "video_processor"):
        attr = getattr(processor, attr_name, None)
        if attr is None or not hasattr(attr, "save_pretrained"):
            continue

        try:
            attr.save_pretrained(export_dir)
            logger.info_rank0(f"Saved `{attr_name}` sidecar config to {export_dir}.")
        except Exception as e:  # noqa: BLE001
            logger.warning_rank0(f"Cannot save `{attr_name}` sidecar config, please copy it manually: {e}.")


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    raw_callbacks = config.get("callbacks")
    callbacks: list[Any] = raw_callbacks if isinstance(raw_callbacks, list) else []
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    _save_training_command(args, training_args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
        if not is_mcore_adapter_available():
            raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")
        if finetuning_args.stage == "pt":
            from .mca import run_pt as run_pt_mca

            run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            from .mca import run_sft as run_sft_mca

            run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "dpo":
            from .mca import run_dpo as run_dpo_mca

            run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)

    elif finetuning_args.stage == "pt":
        from .pt import run_pt

        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        from .sft import run_sft

        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        from .rm import run_rm

        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        from .ppo import run_ppo

        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        from .dpo import run_dpo

        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        from .kto import run_kto

        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    if is_ray_available() and _require_ray().is_initialized():
        return  # if ray is intialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(
    args: Optional[dict[str, Any] | list[str]] = None, callbacks: Optional[list["TrainerCallback"]] = None
) -> None:
    parsed_args = read_args(args)
    if isinstance(parsed_args, list) and ("-h" in parsed_args or "--help" in parsed_args):
        get_train_args(parsed_args)

    ray_args = get_ray_args(parsed_args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        _ray_training_function(ray_args, config={"args": parsed_args, "callbacks": callbacks})
    else:
        _training_function(config={"args": parsed_args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:  # if infer_dtype is auto, try using half precision first
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    # Preserve tied embeddings on export when LoRA training materializes
    # `embed_tokens` and `lm_head` as separate trainable modules.
    # For Qwen3/Qwen3.5 endpointing this is important: exporting an untied
    # pair can change the next-token distribution in vLLM even when the model
    # should conceptually remain tied.
    try:
        if getattr(model.config, "tie_word_embeddings", False):
            input_emb = model.get_input_embeddings()
            output_emb = model.get_output_embeddings()
            if (
                input_emb is not None
                and output_emb is not None
                and hasattr(input_emb, "weight")
                and hasattr(output_emb, "weight")
                and input_emb.weight.data_ptr() != output_emb.weight.data_ptr()
            ):
                if input_emb.weight.shape != output_emb.weight.shape:
                    raise ValueError(
                        "Cannot retie input/output embeddings before export because their shapes differ: "
                        f"{tuple(input_emb.weight.shape)} vs {tuple(output_emb.weight.shape)}."
                    )

                logger.warning_rank0(
                    "Detected untied input/output embeddings but `tie_word_embeddings=True` in config; "
                    "copying `lm_head` weights into input embeddings and re-tying before export."
                )
                with torch.no_grad():
                    input_emb.weight.copy_(output_emb.weight)

                model.tie_weights()

                if input_emb.weight.data_ptr() != output_emb.weight.data_ptr():
                    logger.warning_rank0(
                        "`model.tie_weights()` did not restore shared storage for input/output embeddings; "
                        "export will proceed with copied values but separate tensors."
                    )
    except Exception as e:
        logger.warning_rank0(f"Failed to validate tie_word_embeddings before export: {e}.")

    # Prepare save arguments (safe_serialization removed in transformers v5.0.0)
    save_kwargs: dict[str, object] = {
        "save_directory": model_args.export_dir,
        "max_shard_size": f"{model_args.export_size}GB",
    }
    if not is_transformers_version_greater_than("5.0.0"):
        save_kwargs["safe_serialization"] = not model_args.export_legacy_format

    model.save_pretrained(**save_kwargs)
    if model_args.export_hub_model_id is not None:
        # Prepare push arguments (safe_serialization removed in transformers v5.0.0)
        push_kwargs: dict[str, object] = {
            "max_shard_size": f"{model_args.export_size}GB",
        }
        if not is_transformers_version_greater_than("5.0.0"):
            push_kwargs["safe_serialization"] = not model_args.export_legacy_format

        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            **push_kwargs,
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            if model_args.model_name_or_path is None:
                raise ValueError("Please provide `model_name_or_path`.")
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            _save_processor_sidecar_configs(processor, model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        _normalize_exported_tokenizer_config(model_args)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")

    ollama_modelfile = os.path.join(model_args.export_dir, "Modelfile")
    with open(ollama_modelfile, "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
        logger.info_rank0(f"Ollama modelfile saved in {ollama_modelfile}")


class Worker:
    def __init__(self):
        self._setup_env_visible_devices()

        local_rank = os.environ.get("LOCAL_RANK", "0")
        get_torch_device().set_device(int(local_rank))

    def _setup_env_visible_devices(self) -> None:
        RAY_NOSET_VISIBLE_DEVICES_LIST = [
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        ]
        is_ray_noset_visible_devices = any(os.environ.get(env_var, None) for env_var in RAY_NOSET_VISIBLE_DEVICES_LIST)
        if is_ray_noset_visible_devices:
            device_name = get_device_name().upper()
            ray_module = _require_ray()
            local_rank = str(ray_module.get_runtime_context().get_accelerator_ids()[device_name][0])
            os.environ["LOCAL_RANK"] = local_rank
        else:
            os.environ["LOCAL_RANK"] = "0"

    def _training_function(self, config: dict[str, Any]) -> None:
        _training_function(config)


def _ray_training_function(ray_args: "RayArguments", config: dict[str, Any]) -> None:
    ray_module = _require_ray()
    num_workers = ray_args.ray_num_workers
    master_addr = ray_args.master_addr
    master_port = ray_args.master_port
    logger.info(f"Using ray.remote mode with {num_workers} workers for distributed training.")

    # initialize ray
    if not ray_module.is_initialized():
        if ray_args.ray_init_kwargs is not None:
            if not isinstance(ray_args.ray_init_kwargs, Mapping):
                raise ValueError("`ray_init_kwargs` must be a mapping or a JSON object string.")

            ray_module.init(**dict(ray_args.ray_init_kwargs))
        else:
            ray_module.init()

    # verify resources
    device_name = get_device_name().upper()
    total_devices = int(ray_module.cluster_resources().get(device_name, 0))
    if num_workers > total_devices:
        raise ValueError(
            f"The number of devices in the Ray cluster ({total_devices}) should be greater than num_workers ({num_workers})."
        )

    # verify master_addr
    if master_addr is None:
        master_addr = get_ray_head_node_ip()
        logger.info(f"`master_addr` is not specified, using head node ip: {master_addr}.")
    else:
        nodes = [node["NodeManagerAddress"] for node in ray_module.nodes() if node["Alive"]]
        if master_addr not in nodes:
            raise ValueError(f"The `master_addr` ({master_addr}) is not in Ray cluster or not alive ")

    # create placementgroup for resource management
    pg, bundle = get_placement_group(total_devices)
    ray_module.get(pg.ready())
    logger.info(f"Create placement group with {num_workers} bundles: {bundle}")

    # get sorted_bundle_indices
    sorted_bundle_indices = sort_placement_group_by_node_ip(pg, master_addr)

    # get master port
    if master_port is None:
        master_port = find_available_port()
        logger.info(f"`master_port` is not specified, using available port: {master_port}.")
    master_port = str(master_port)

    # backing up environment variables
    current_env = dict(os.environ.items())

    # launch workers
    RayWorker = ray_module.remote(Worker)
    workers: list[Any] = []
    for rank in range(num_workers):
        remote_config = get_ray_remote_config_for_worker(
            placement_group=pg,
            bundle_idx=sorted_bundle_indices[rank],
            rank=rank,
            world_size=num_workers,
            master_addr=master_addr,
            master_port=master_port,
            env=current_env,
        )
        worker = RayWorker.options(**remote_config).remote()
        workers.append(worker)

    ray_module.get([worker._training_function.remote(config=config) for worker in workers])
    ray_module.shutdown()
