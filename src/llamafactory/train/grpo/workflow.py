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

from dataclasses import fields
from typing import TYPE_CHECKING, Any, Optional

from trl import GRPOConfig

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras import logging
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .reward import build_asr_reward_suite
from .trainer import CustomFunAudioChatGRPOTrainer


if TYPE_CHECKING:
    from transformers import TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


def _maybe_patch_funaudiochat_full_fsdp(
    training_args: "TrainingArguments", finetuning_args: "FinetuningArguments", model_type: Optional[str]
) -> None:
    if model_type != "funaudiochat" or finetuning_args.finetuning_type != "full" or not training_args.fsdp:
        return

    fsdp_config = dict(training_args.fsdp_config or {})
    if finetuning_args.funaudiochat_freeze_audio_tower:
        frozen_audio_regex = "continuous_audio_tower|audio_tower|audio_invert_tower"
        existing_ignored_modules = fsdp_config.get("ignored_modules")
        if isinstance(existing_ignored_modules, str) and existing_ignored_modules:
            if frozen_audio_regex not in existing_ignored_modules:
                fsdp_config["ignored_modules"] = f"(?:{existing_ignored_modules})|(?:{frozen_audio_regex})"
        elif existing_ignored_modules is None:
            fsdp_config["ignored_modules"] = frozen_audio_regex

    if finetuning_args.freeze_multi_modal_projector:
        frozen_projector_regex = "multi_modal_projector"
        existing_ignored_modules = fsdp_config.get("ignored_modules")
        if isinstance(existing_ignored_modules, str) and existing_ignored_modules:
            if frozen_projector_regex not in existing_ignored_modules:
                fsdp_config["ignored_modules"] = f"(?:{existing_ignored_modules})|(?:{frozen_projector_regex})"
        elif existing_ignored_modules is None:
            fsdp_config["ignored_modules"] = frozen_projector_regex

    fsdp_config.setdefault("use_orig_params", True)
    if (
        "transformer_layer_cls_to_wrap" not in fsdp_config
        and getattr(training_args, "fsdp_transformer_layer_cls_to_wrap", None) is not None
    ):
        fsdp_config["transformer_layer_cls_to_wrap"] = training_args.fsdp_transformer_layer_cls_to_wrap

    training_args.fsdp_config = fsdp_config
    logger.info_rank0(
        "Patched FSDP config for full FunAudioChat GRPO: "
        f"use_orig_params={fsdp_config.get('use_orig_params')}, "
        f"ignored_modules={fsdp_config.get('ignored_modules')!r}."
    )


def _build_grpo_config(
    training_args: "TrainingArguments", data_args: "DataArguments", finetuning_args: "FinetuningArguments"
) -> GRPOConfig:
    valid_fields = {field.name for field in fields(GRPOConfig)}
    training_dict = training_args.to_dict()
    config_kwargs = {key: value for key, value in training_dict.items() if key in valid_fields}
    config_kwargs.update(
        remove_unused_columns=False,
        max_prompt_length=data_args.cutoff_len,
        num_generations=finetuning_args.grpo_num_generations,
        max_completion_length=finetuning_args.grpo_max_completion_length,
        beta=finetuning_args.grpo_beta,
        temperature=finetuning_args.grpo_temperature,
        top_p=finetuning_args.grpo_top_p,
        top_k=finetuning_args.grpo_top_k,
        min_p=finetuning_args.grpo_min_p,
        repetition_penalty=finetuning_args.grpo_repetition_penalty,
        num_iterations=finetuning_args.grpo_num_iterations,
        epsilon=finetuning_args.grpo_epsilon,
        epsilon_high=finetuning_args.grpo_epsilon_high,
        delta=finetuning_args.grpo_delta,
        scale_rewards=finetuning_args.grpo_scale_rewards,
        loss_type=finetuning_args.grpo_loss_type,
        mask_truncated_completions=finetuning_args.grpo_mask_truncated_completions,
        sync_ref_model=finetuning_args.grpo_sync_ref_model,
        ref_model_mixup_alpha=finetuning_args.grpo_ref_model_mixup_alpha,
        ref_model_sync_steps=finetuning_args.grpo_ref_model_sync_steps,
        top_entropy_quantile=finetuning_args.grpo_top_entropy_quantile,
        generation_batch_size=finetuning_args.grpo_generation_batch_size,
        steps_per_generation=finetuning_args.grpo_steps_per_generation,
        generation_kwargs=finetuning_args.grpo_generation_kwargs,
        use_vllm=finetuning_args.grpo_use_vllm,
        use_transformers_paged=finetuning_args.grpo_use_transformers_paged,
        vllm_mode=finetuning_args.grpo_vllm_mode,
        vllm_gpu_memory_utilization=finetuning_args.grpo_vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=finetuning_args.grpo_vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=finetuning_args.grpo_vllm_enable_sleep_mode,
        vllm_guided_decoding_regex=finetuning_args.grpo_vllm_guided_decoding_regex,
        vllm_server_base_url=finetuning_args.grpo_vllm_server_base_url,
        vllm_server_host=finetuning_args.grpo_vllm_server_host,
        vllm_server_port=finetuning_args.grpo_vllm_server_port,
        vllm_server_timeout=finetuning_args.grpo_vllm_server_timeout,
        vllm_importance_sampling_correction=finetuning_args.grpo_vllm_importance_sampling_correction,
        vllm_importance_sampling_cap=finetuning_args.grpo_vllm_importance_sampling_cap,
        disable_dropout=finetuning_args.grpo_disable_dropout,
        ds3_gather_for_generation=finetuning_args.grpo_ds3_gather_for_generation,
        shuffle_dataset=finetuning_args.grpo_shuffle_dataset,
        log_completions=finetuning_args.grpo_log_completions,
        num_completions_to_print=finetuning_args.grpo_num_completions_to_print,
        wandb_log_unique_prompts=finetuning_args.grpo_wandb_log_unique_prompts,
    )
    return GRPOConfig(**config_kwargs)


def run_grpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    if data_args.template != "funaudiochat":
        raise ValueError("GRPO ASR stage currently only supports `template: funaudiochat`.")

    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="grpo", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model.config, "model_type", None) != "funaudiochat":
        raise ValueError("GRPO ASR stage currently only supports FunAudioChat checkpoints.")

    _maybe_patch_funaudiochat_full_fsdp(training_args, finetuning_args, getattr(model.config, "model_type", None))

    if finetuning_args.grpo_use_vllm and finetuning_args.grpo_vllm_mode != "colocate":
        raise NotImplementedError(
            "FunAudioChat GRPO currently supports only `grpo_vllm_mode: colocate`."
        )

    if (
        finetuning_args.finetuning_type == "full"
        and finetuning_args.grpo_use_vllm
        and finetuning_args.grpo_vllm_mode == "colocate"
        and finetuning_args.grpo_beta != 0.0
    ):
        logger.warning_rank0(
            "Full-parameter FunAudioChat GRPO with colocated vLLM and `grpo_beta != 0` will instantiate a separate "
            "reference model and is often memory-heavy. Prefer ZeRO-3 and consider `grpo_beta: 0.0` as the "
            "practical starting point."
        )

    reward_funcs, reward_weights = build_asr_reward_suite(finetuning_args)
    grpo_config = _build_grpo_config(training_args, data_args, finetuning_args)
    grpo_config.reward_weights = reward_weights

    trainer = CustomFunAudioChatGRPOTrainer(
        model=model,
        model_args=model_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
        template=template,
        tokenizer=tokenizer,
        processor=tokenizer_module["processor"],
        args=grpo_config,
        reward_funcs=reward_funcs,
        callbacks=callbacks,
        **dataset_module,
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss", "reward"])
