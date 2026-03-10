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

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union


@dataclass
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: int | None = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: float | None = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class OFTArguments:
    r"""Arguments pertaining to the OFT training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the OFT fine-tuning."},
    )
    oft_rank: int = field(
        default=0,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_block_size: int = field(
        default=32,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply OFT. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO, KTO and GRPO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )
    grpo_num_generations: int = field(
        default=4,
        metadata={"help": "Number of sampled completions per prompt in GRPO training."},
    )
    grpo_max_completion_length: int = field(
        default=256,
        metadata={"help": "Maximum number of generated completion tokens in GRPO training."},
    )
    grpo_beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient in GRPO training."},
    )
    grpo_temperature: float = field(
        default=0.8,
        metadata={"help": "Sampling temperature for GRPO rollout generation."},
    )
    grpo_top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p sampling parameter for GRPO rollout generation."},
    )
    grpo_top_k: int | None = field(
        default=None,
        metadata={"help": "Top-k sampling parameter for GRPO rollout generation."},
    )
    grpo_min_p: float | None = field(
        default=None,
        metadata={"help": "Minimum token probability threshold for GRPO rollout generation."},
    )
    grpo_repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty used during GRPO rollout generation."},
    )
    grpo_num_iterations: int = field(
        default=1,
        metadata={"help": "Number of policy update iterations per sampled GRPO batch."},
    )
    grpo_epsilon: float = field(
        default=0.2,
        metadata={"help": "Lower clipping coefficient used by the GRPO objective."},
    )
    grpo_epsilon_high: float | None = field(
        default=None,
        metadata={"help": "Upper clipping coefficient used by the GRPO objective."},
    )
    grpo_delta: float | None = field(
        default=None,
        metadata={"help": "Optional upper cap for two-sided GRPO clipping."},
    )
    grpo_scale_rewards: Literal["group", "batch", "none"] = field(
        default="group",
        metadata={"help": "Reward normalization strategy used by GRPO."},
    )
    grpo_loss_type: Literal["grpo", "bnpo", "dr_grpo", "dapo"] = field(
        default="dapo",
        metadata={"help": "Loss variant used by GRPO."},
    )
    grpo_mask_truncated_completions: bool = field(
        default=True,
        metadata={"help": "Whether truncated completions are masked out from the GRPO loss."},
    )
    grpo_sync_ref_model: bool = field(
        default=False,
        metadata={"help": "Whether to periodically sync the GRPO reference model."},
    )
    grpo_ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={"help": "Reference model mixup coefficient used when syncing GRPO reference weights."},
    )
    grpo_ref_model_sync_steps: int = field(
        default=512,
        metadata={"help": "Number of steps between GRPO reference model syncs."},
    )
    grpo_top_entropy_quantile: float = field(
        default=1.0,
        metadata={"help": "Top-entropy token quantile kept by the GRPO loss."},
    )
    grpo_generation_batch_size: int | None = field(
        default=None,
        metadata={"help": "Optional explicit generation batch size for GRPO."},
    )
    grpo_steps_per_generation: int | None = field(
        default=None,
        metadata={"help": "Optional number of optimizer steps that reuse the same GRPO rollout batch."},
    )
    grpo_generation_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={"help": "Optional extra generation kwargs forwarded to GRPO rollout generation."},
    )
    grpo_use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use vLLM-backed rollout generation in GRPO."},
    )
    grpo_use_transformers_paged: bool = field(
        default=False,
        metadata={"help": "Whether to use transformers paged generation in GRPO."},
    )
    grpo_vllm_mode: Literal["server", "colocate"] = field(
        default="server",
        metadata={"help": "vLLM mode used by GRPO when `grpo_use_vllm=true`."},
    )
    grpo_vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={"help": "vLLM GPU memory utilization used by GRPO in colocate mode."},
    )
    grpo_vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "vLLM tensor parallel size used by GRPO in colocate mode."},
    )
    grpo_allow_experimental_funaudiochat_colocate_tp: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow the experimental FunAudioChat GRPO path that combines "
                "`finetuning_type=full`, `grpo_use_vllm=true`, `grpo_vllm_mode=colocate`, and "
                "`grpo_vllm_tensor_parallel_size>1`. This path is fail-fast by default because it "
                "still depends on external vLLM fixes and has unresolved hang reports."
            )
        },
    )
    grpo_vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={"help": "Whether GRPO should enable vLLM sleep mode in colocate mode."},
    )
    grpo_vllm_guided_decoding_regex: str | None = field(
        default=None,
        metadata={"help": "Optional guided decoding regex for GRPO vLLM generation."},
    )
    grpo_vllm_server_base_url: str | None = field(
        default=None,
        metadata={"help": "Optional GRPO vLLM server base URL."},
    )
    grpo_vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "GRPO vLLM server host when using server mode."},
    )
    grpo_vllm_server_port: int = field(
        default=8000,
        metadata={"help": "GRPO vLLM server port when using server mode."},
    )
    grpo_vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "GRPO vLLM server connection timeout in seconds."},
    )
    grpo_vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={"help": "Whether to apply truncated importance sampling when using vLLM in GRPO."},
    )
    grpo_vllm_importance_sampling_cap: float = field(
        default=2.0,
        metadata={"help": "Upper bound for truncated importance sampling in GRPO."},
    )
    grpo_disable_dropout: bool = field(
        default=False,
        metadata={"help": "Whether to disable dropout when running GRPO."},
    )
    grpo_ds3_gather_for_generation: bool = field(
        default=True,
        metadata={"help": "Whether to gather ZeRO-3 weights for GRPO generation."},
    )
    grpo_shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle prompts before sampling GRPO rollouts."},
    )
    grpo_log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to print / log sampled GRPO completions."},
    )
    grpo_num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Optional cap on the number of logged GRPO completions."},
    )
    grpo_wandb_log_unique_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to deduplicate prompts in GRPO wandb completion logs."},
    )
    grpo_reward_wer_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of the normalized WER reward term in GRPO ASR training."},
    )
    grpo_reward_cer_weight: float = field(
        default=0.25,
        metadata={"help": "Weight of the normalized CER reward term in GRPO ASR training."},
    )
    grpo_empty_penalty: float = field(
        default=1.0,
        metadata={"help": "Penalty applied to empty GRPO ASR completions."},
    )
    grpo_repeat_penalty: float = field(
        default=0.2,
        metadata={"help": "Penalty weight applied to repetition-heavy GRPO ASR completions."},
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: int | None = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Literal["ascending", "descending", "random", "fixed"] | None = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: int | None = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: str | None = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: str | None = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: str | None = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: str | None = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: str | None = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FinetuningArguments(
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto", "grpo"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_mca: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use MCA (Megatron Core Adapter) training. "
                "Controlled by USE_MCA environment variable."
            )
        },
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    module_lr_groups: Optional[Union[list[dict[str, Any]], str]] = field(
        default=None,
        metadata={
            "help": (
                "Optional per-module learning rate and scheduler overrides. "
                "Provide a YAML list of dicts (or a JSON list string). "
                "Each item must include: `patterns` (list[str]) and `lr` (float). "
                "Optional keys: `name`, `lr_scheduler_type`, `warmup_ratio`, `warmup_steps`, `lr_scheduler_kwargs`."
            )
        },
    )
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    use_chunked_ce_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use chunked cross entropy loss (torchtune-style, computes CE in token chunks "
                "to reduce peak memory from fp32 upcasting)."
            )
        },
    )
    chunked_ce_num_chunks: int = field(
        default=8,
        metadata={"help": "Number of token chunks used by chunked cross entropy loss."},
    )
    chunked_ce_upcast_logits: bool = field(
        default=True,
        metadata={"help": "Whether to upcast logits to fp32 per chunk before computing cross entropy."},
    )
    use_asft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the ASFT loss."},
    )
    asft_alpha: float = field(
        default=0.1,
        metadata={"help": "The alpha parameter for ASFT loss to control the power of adaptive weight."},
    )
    use_eaft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the EAFT loss."},
    )
    eaft_alpha: float = field(
        default=1.0,
        metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    funaudiochat_full_audio_tuning: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to fully fine-tune FunAudioChat audio encoder + adapter (continuous_audio_tower + audio_tower) "
                "while keeping the language model in LoRA/OFT."
            )
        },
    )
    funaudiochat_freeze_audio_tower: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to freeze FunAudioChat audio modules (continuous_audio_tower/audio_tower/audio_invert_tower) "
                "for LLM-only tuning."
            )
        },
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    compute_endpointing_metrics: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to compute speech endpointing label metrics on the first supervised tag token, "
                "including merged metrics where `<UNADDRESSED>` is treated as `<EOU>`."
            )
        },
    )
    compute_wer_cer: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute WER and CER for generation-based evaluation."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    early_stopping_steps: int | None = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )
    eval_num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use for evaluation."},
    )
    eval_loss_on_full_dataset: bool = field(
        default=True,
        metadata={
            "help": (
                "When `predict_with_generate` is enabled and `eval_num_samples` is set, "
                "compute eval loss on the full evaluation dataset while computing generative metrics "
                "(WER/CER/ROUGE/BLEU) on the sampled subset."
            )
        },
    )
    eval_max_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of new tokens to generate for evaluation."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: list[str] | None = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        self.oft_target: list[str] = split_arg(self.oft_target)
        self.additional_target: list[str] | None = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        if isinstance(self.module_lr_groups, str) and self.module_lr_groups.strip().startswith("["):
            self.module_lr_groups = json.loads(self.module_lr_groups)
        if isinstance(self.grpo_generation_kwargs, str) and self.grpo_generation_kwargs.strip().startswith("{"):
            self.grpo_generation_kwargs = json.loads(self.grpo_generation_kwargs)

        if self.module_lr_groups is not None:
            if not isinstance(self.module_lr_groups, list):
                raise ValueError("`module_lr_groups` must be a list of dicts (or a JSON list string).")

            normalized_groups: list[dict[str, Any]] = []
            for idx, raw_group in enumerate(self.module_lr_groups):
                if not isinstance(raw_group, dict):
                    raise ValueError(f"`module_lr_groups[{idx}]` must be a dict, got {type(raw_group)}.")

                name = str(raw_group.get("name") or f"group{idx}")

                patterns = raw_group.get("patterns", None)
                if patterns is None:
                    raise ValueError(f"`module_lr_groups[{idx}].patterns` is required.")
                if isinstance(patterns, str):
                    patterns = [item.strip() for item in patterns.split(",") if item.strip()]
                if (
                    not isinstance(patterns, list)
                    or not patterns
                    or any(not isinstance(p, str) or not p for p in patterns)
                ):
                    raise ValueError(f"`module_lr_groups[{idx}].patterns` must be a non-empty list of strings.")

                lr = raw_group.get("lr", None)
                if lr is None:
                    raise ValueError(f"`module_lr_groups[{idx}].lr` is required.")
                lr = float(lr)
                if lr <= 0:
                    raise ValueError(f"`module_lr_groups[{idx}].lr` must be > 0, got {lr}.")

                lr_scheduler_type = raw_group.get("lr_scheduler_type", None)
                if lr_scheduler_type is not None:
                    lr_scheduler_type = str(lr_scheduler_type)

                warmup_ratio = raw_group.get("warmup_ratio", None)
                if warmup_ratio is not None:
                    warmup_ratio = float(warmup_ratio)
                    if warmup_ratio < 0 or warmup_ratio > 1:
                        raise ValueError(
                            f"`module_lr_groups[{idx}].warmup_ratio` must be in [0, 1], got {warmup_ratio}."
                        )

                warmup_steps = raw_group.get("warmup_steps", None)
                if warmup_steps is not None:
                    warmup_steps = int(warmup_steps)
                    if warmup_steps < 0:
                        raise ValueError(f"`module_lr_groups[{idx}].warmup_steps` must be >= 0, got {warmup_steps}.")

                lr_scheduler_kwargs = raw_group.get("lr_scheduler_kwargs", None)
                if isinstance(lr_scheduler_kwargs, str) and lr_scheduler_kwargs.strip().startswith("{"):
                    lr_scheduler_kwargs = json.loads(lr_scheduler_kwargs)
                if lr_scheduler_kwargs is not None and not isinstance(lr_scheduler_kwargs, dict):
                    raise ValueError(
                        f"`module_lr_groups[{idx}].lr_scheduler_kwargs` must be a dict (or JSON dict string)."
                    )

                normalized_groups.append(
                    {
                        "name": name,
                        "patterns": patterns,
                        "lr": lr,
                        "lr_scheduler_type": lr_scheduler_type,
                        "warmup_ratio": warmup_ratio,
                        "warmup_steps": warmup_steps,
                        "lr_scheduler_kwargs": lr_scheduler_kwargs,
                    }
                )

            self.module_lr_groups = normalized_groups
            if len(self.module_lr_groups) == 0:
                raise ValueError("`module_lr_groups` must contain at least one group.")

        assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
            raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")

        if self.stage == "grpo" and self.grpo_num_generations < 2:
            raise ValueError("GRPO requires at least 2 generations per prompt.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.module_lr_groups is not None and (
            self.use_galore
            or self.use_apollo
            or self.use_badam
            or self.loraplus_lr_ratio is not None
            or self.use_adam_mini
            or self.use_muon
        ):
            raise ValueError(
                "`module_lr_groups` cannot be combined with GaLore/APOLLO/BAdam/LoRA+/Adam-mini/Muon optimizers."
            )

        if self.pissa_init and (self.stage in ["ppo", "kto", "grpo"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
