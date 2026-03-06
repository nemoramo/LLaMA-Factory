# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/modeling_llava.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import transformers
import transformers.models
from transformers.activations import ACT2FN

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than


if TYPE_CHECKING:
    from transformers import LlavaConfig, PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)
transformers_logger = transformers.utils.logging.get_logger(__name__)


@dataclass
class CompositeModel:
    model_type: str
    projector_key: str
    vision_model_keys: list[str]
    language_model_keys: list[str]
    lora_conflict_keys: list[str]

    def get_projector(self, module: "torch.nn.Module") -> "torch.nn.Module":
        for key in self.projector_key.split("."):
            module = getattr(module, key)

        return module


COMPOSITE_MODELS: dict[str, "CompositeModel"] = {}


def _register_composite_model(
    model_type: str,
    projector_key: Optional[str] = None,
    vision_model_keys: Optional[list[str]] = None,
    language_model_keys: Optional[list[str]] = None,
    lora_conflict_keys: Optional[list[str]] = None,
):
    r"""Register a new composite model.

    Args:
        model_type: model type
        projector_key: multi_modal_projector
        vision_model_keys: vision_tower
        language_model_keys: language_model
        lora_conflict_keys: None

    """
    COMPOSITE_MODELS[model_type] = CompositeModel(
        model_type=model_type,
        projector_key=projector_key or "multi_modal_projector",
        vision_model_keys=vision_model_keys or ["vision_tower"],
        language_model_keys=language_model_keys or ["language_model", "lm_head"],
        lora_conflict_keys=lora_conflict_keys or [],
    )


class LlavaMultiModalProjectorForYiVL(torch.nn.Module):
    def __init__(self, config: "LlavaConfig") -> None:
        super().__init__()

        self.config = config
        if config is None:
            return

        self.linear_1 = torch.nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_2 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.linear_3 = torch.nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_4 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]

    def forward(self, image_features: "torch.Tensor") -> "torch.Tensor":
        hidden_states = self.linear_1(image_features)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        hidden_states = self.linear_4(hidden_states)
        if hidden_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.linear_1.weight.dtype

            transformers_logger.warning_once("The hidden states seems to be silently casted in float32.")
            hidden_states = hidden_states.to(target_dtype)

        return hidden_states


class LlavaMultiModalProjectorForYiVLForVLLM(LlavaMultiModalProjectorForYiVL):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, projector_hidden_act: str) -> None:
        super().__init__(config=None)

        self.linear_1 = torch.nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.linear_2 = torch.nn.LayerNorm(text_hidden_size, bias=True)
        self.linear_3 = torch.nn.Linear(text_hidden_size, text_hidden_size, bias=True)
        self.linear_4 = torch.nn.LayerNorm(text_hidden_size, bias=True)
        self.act = ACT2FN[projector_hidden_act]


def autocast_projector_dtype(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""Cast projector output to half precision for fine-tuning quantized VLMs."""

    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)

    if getattr(model, "quantization_method", None):
        model_type = getattr(model.config, "model_type", None)
        if model_type in COMPOSITE_MODELS:
            mm_projector = COMPOSITE_MODELS[model_type].get_projector(model)
        else:
            return

        logger.info_rank0(f"Casting multimodal projector outputs in {model_args.compute_dtype}.")
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)


def cast_gemma3n_audio_outputs(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""Casts audio tower outputs to model dtype for Gemma 3n."""
    if getattr(model.config, "model_type", None) != "gemma3n":
        return

    def _audio_tower_forward_post_hook(
        module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        if isinstance(output, torch.Tensor):
            return output.to(model_args.compute_dtype)
        elif isinstance(output, tuple):
            return tuple(
                t.to(model_args.compute_dtype) if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                for t in output
            )
        return output

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model

    # Check for audio_tower in common locations
    audio_tower = None
    if hasattr(base_model, "audio_tower"):
        audio_tower = base_model.audio_tower
    elif hasattr(base_model, "model") and hasattr(base_model.model, "audio_tower"):
        audio_tower = base_model.model.audio_tower

    if audio_tower is not None:
        logger.info_rank0(f"Casting audio tower outputs in {model_args.compute_dtype}.")
        audio_tower.register_forward_hook(_audio_tower_forward_post_hook)


def patch_gemma3n_audio_token_mask() -> None:
    r"""Patch Gemma3n audio token mask to be range-based.

    Upstream Gemma3nModel uses `audio_mask = input_ids >= embed_audio.vocab_offset`, which assumes the vocabulary ends
    at the last audio token. If users add new tokens (e.g., special tags) beyond the original vocabulary, those token
    ids will be mis-classified as audio tokens and cause out-of-range indexing in `embed_audio`.

    This patch makes the mask range-based:
        audio_offset <= input_ids < audio_offset + audio_vocab_size

    It is behavior-preserving for the original (un-resized) vocab, and enables safe vocab resizing.
    """
    try:
        from transformers.models.gemma3n import modeling_gemma3n
    except Exception:  # noqa: BLE001
        return

    Gemma3nModel = getattr(modeling_gemma3n, "Gemma3nModel", None)
    if Gemma3nModel is None:
        return

    if getattr(Gemma3nModel, "_llamafactory_audio_token_mask_patched", False):
        return

    can_return_tuple = getattr(modeling_gemma3n, "can_return_tuple", None)
    if can_return_tuple is None:
        return

    Gemma3nModelOutputWithPast = getattr(modeling_gemma3n, "Gemma3nModelOutputWithPast", None)
    Cache = getattr(modeling_gemma3n, "Cache", None)
    if Gemma3nModelOutputWithPast is None or Cache is None:
        return

    @can_return_tuple
    def _patched_forward(  # type: ignore[override]
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
        input_features: Optional[torch.FloatTensor] = None,  # audio inputs
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], "Cache"]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **lm_kwargs,
    ) -> "Gemma3nModelOutputWithPast":
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # Prepare per-layer inputs from inputs_ids
            per_layer_inputs_mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
            per_layer_inputs_tokens = torch.where(per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids))
            per_layer_inputs = self.language_model.get_per_layer_inputs(per_layer_inputs_tokens)

            # Handle vision tokens (>= embed_vision.vocab_offset and < embed_audio.vocab_offset)
            vision_mask = torch.logical_and(
                input_ids >= self.embed_vision.vocab_offset, input_ids < self.embed_audio.vocab_offset
            )
            dummy_vision_token_id = self.embed_vision.vocab_offset + self.embed_vision.vocab_size - 1
            vision_input_ids = torch.where(vision_mask, input_ids, dummy_vision_token_id).to(inputs_embeds.device)
            vision_embeds = self.embed_vision(input_ids=vision_input_ids)
            expanded_vision_mask = vision_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(expanded_vision_mask, vision_embeds, inputs_embeds)

            # Handle audio tokens (>= embed_audio.vocab_offset and < vocab_offset + vocab_size)
            audio_offset = self.embed_audio.vocab_offset
            audio_vocab_size = self.embed_audio.vocab_size
            audio_mask = torch.logical_and(input_ids >= audio_offset, input_ids < (audio_offset + audio_vocab_size))
            dummy_audio_token_id = audio_offset + audio_vocab_size - 1
            audio_input_ids = torch.where(audio_mask, input_ids, dummy_audio_token_id).to(inputs_embeds.device)
            audio_embeds = self.embed_audio(input_ids=audio_input_ids)
            expanded_audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(expanded_audio_mask, audio_embeds, inputs_embeds)
        else:
            per_layer_inputs = None

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Merge text and audio
        if input_features is not None and input_features_mask is not None:
            audio_features, audio_mask = self.get_audio_features(input_features, ~input_features_mask)

            # The Gemma3nProcessor expects all audio will be 30s in length and inserts 188 audio soft tokens into the
            # text to account for this. However, the audio preprocessing and encoder do not gurarantee they will
            # produce 188 soft tokens; they will produce at most that many tokens, but they may produce fewer tokens
            # depending on the length of the longest audio input in the batch. When we encounter this situation, we pad
            # the audio feature out to 188 soft tokens with the emebedding of the last token in the embed_audio vocab.
            audio_padding_toks = torch.tensor([[self.vocab_size - 1]], dtype=torch.long, device=audio_features.device)
            audio_padding_embs = self.embed_audio(input_ids=audio_padding_toks)
            audio_features = torch.where(audio_mask.unsqueeze(-1), audio_padding_embs, audio_features)

            audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape
            extra_padding_tokens = self.config.audio_soft_tokens_per_image - audio_seq_len
            extra_padding_features = audio_padding_embs.expand(audio_batch_size, extra_padding_tokens, audio_embed_dim)

            audio_features = torch.cat((audio_features, extra_padding_features), dim=1)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self.language_model(
            input_ids=None,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        return Gemma3nModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
        )

    Gemma3nModel.forward = _patched_forward  # type: ignore[assignment]
    setattr(Gemma3nModel, "_llamafactory_audio_token_mask_patched", True)
    logger.info_rank0("Patched Gemma3nModel audio_mask to be range-based (safe vocab resizing).")


def patch_gemma3n_config_vocab_size() -> None:
    r"""Patch Gemma3nConfig to expose a `vocab_size` attribute for PEFT compatibility.

    PEFT checks whether the vocabulary was resized by comparing:
        model.config.vocab_size  vs  model.config.__class__.from_pretrained(model_id).vocab_size

    However, upstream `Gemma3nConfig` does not define `vocab_size`; the value lives in `text_config.vocab_size`.
    This causes checkpoint saving to crash when `save_embedding_layers="auto"` in PEFT.

    We provide `vocab_size` as a property that mirrors `text_config.vocab_size` and supports assignment so that
    LLaMA-Factory's embedding-resize logic can keep the config consistent.
    """
    try:
        from transformers.models.gemma3n import configuration_gemma3n
    except Exception:  # noqa: BLE001
        return

    Gemma3nConfig = getattr(configuration_gemma3n, "Gemma3nConfig", None)
    if Gemma3nConfig is None:
        return

    if getattr(Gemma3nConfig, "_llamafactory_vocab_size_patched", False):
        return

    if hasattr(Gemma3nConfig, "vocab_size"):
        # Upstream may add it in the future.
        setattr(Gemma3nConfig, "_llamafactory_vocab_size_patched", True)
        return

    def _get_vocab_size(self) -> Optional[int]:
        text_cfg = getattr(self, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            try:
                return int(getattr(text_cfg, "vocab_size"))
            except Exception:  # noqa: BLE001
                return getattr(text_cfg, "vocab_size")
        return getattr(self, "_llamafactory_vocab_size", None)

    def _set_vocab_size(self, value) -> None:
        try:
            value_int = int(value)
        except Exception:  # noqa: BLE001
            value_int = value

        text_cfg = getattr(self, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            try:
                setattr(text_cfg, "vocab_size", value_int)
            except Exception:  # noqa: BLE001
                pass

        setattr(self, "_llamafactory_vocab_size", value_int)

    Gemma3nConfig.vocab_size = property(_get_vocab_size, _set_vocab_size)  # type: ignore[attr-defined]
    setattr(Gemma3nConfig, "_llamafactory_vocab_size_patched", True)
    logger.info_rank0("Patched Gemma3nConfig.vocab_size for PEFT checkpoint saving.")


def configure_visual_model(config: "PretrainedConfig") -> None:
    r"""Patch VLMs before loading them."""
    if getattr(config, "text_config", None) and not getattr(config, "hidden_size", None):
        # required for ds zero3 and valuehead models
        setattr(config, "hidden_size", getattr(config.text_config, "hidden_size", None))

    if getattr(config, "is_yi_vl_derived_model", None):
        logger.info_rank0("Detected Yi-VL model, applying projector patch.")
        transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVL


def get_forbidden_modules(config: "PretrainedConfig", finetuning_args: "FinetuningArguments") -> set[str]:
    r"""Freeze vision tower and language model for VLM full/freeze tuning."""
    model_type = getattr(config, "model_type", None)
    forbidden_modules = set()
    if model_type in COMPOSITE_MODELS:
        if finetuning_args.freeze_vision_tower:
            vision_model_keys = COMPOSITE_MODELS[model_type].vision_model_keys
            logger.info_rank0(f"Set vision model not trainable: {vision_model_keys}.")
            forbidden_modules.update(vision_model_keys)

        if finetuning_args.freeze_multi_modal_projector:
            projector_key = COMPOSITE_MODELS[model_type].projector_key
            logger.info_rank0(f"Set multi model projector not trainable: {projector_key}.")
            forbidden_modules.add(projector_key)

        if finetuning_args.freeze_language_model:
            language_model_keys = COMPOSITE_MODELS[model_type].language_model_keys
            logger.info_rank0(f"Set language model not trainable: {language_model_keys}.")
            forbidden_modules.update(language_model_keys)

        if model_type == "funaudiochat" and getattr(finetuning_args, "funaudiochat_freeze_audio_tower", False):
            audio_keys = ["continuous_audio_tower", "audio_tower", "audio_invert_tower"]
            logger.info_rank0(f"FunAudioChat: set audio modules not trainable: {audio_keys}.")
            forbidden_modules.update(audio_keys)

    return forbidden_modules


def patch_target_modules(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments", target_modules: list[str]
) -> list[str]:
    r"""Freeze vision tower for VLM LoRA tuning."""
    model_type = getattr(model.config, "model_type", None)
    if model_type in COMPOSITE_MODELS:
        forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
        forbidden_modules.update(COMPOSITE_MODELS[model_type].lora_conflict_keys)
        module_names = []
        for name, _ in model.named_modules():
            if any(target_module in name for target_module in target_modules) and not any(
                forbidden_module in name for forbidden_module in forbidden_modules
            ):
                module_names.append(name)

        return module_names
    else:
        return target_modules


_register_composite_model(
    model_type="dots_ocr",
    projector_key="vision_tower.merger",
    vision_model_keys=["vision_tower"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["merger"],
)


_register_composite_model(
    model_type="gemma3",
)


_register_composite_model(
    model_type="gemma3n",
    vision_model_keys=["vision_tower", "audio_tower"],
    lora_conflict_keys=["timm_model", "subsample_conv_projection"],
)

_register_composite_model(
    model_type="funaudiochat",
    vision_model_keys=["audio_invert_tower"],
    language_model_keys=["language_model", "lm_head"],
)

_register_composite_model(
    model_type="voxtral",
    projector_key="multi_modal_projector",
    vision_model_keys=["audio_tower"],
    language_model_keys=["language_model"],
)


# copied from qwen2vl
_register_composite_model(
    model_type="glm4v",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="glm4v_moe",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="glm_ocr",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="internvl",
)

_register_composite_model(
    model_type="interns1",
)

_register_composite_model(
    model_type="Keye",
    projector_key="mlp_AR",
    vision_model_keys=["visual.vision_model.patch_embedding", "visual.vision_model.encoder"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embedding"],
)


_register_composite_model(
    model_type="kimi_vl",
)


_register_composite_model(
    model_type="llama4",
    vision_model_keys=["vision_model"],
)


_register_composite_model(
    model_type="llava",
)


_register_composite_model(
    model_type="llava_next",
)


_register_composite_model(
    model_type="llava_next_video",
)


_register_composite_model(
    model_type="minicpmv",
    projector_key="resampler",
    vision_model_keys=["vpm"],
    language_model_keys=["llm"],
)


_register_composite_model(
    model_type="minicpmo",
    projector_key="resampler",
    vision_model_keys=["vpm", "apm", "audio_avg_pooler", "audio_projection_layer", "tts"],
    language_model_keys=["llm"],
    lora_conflict_keys=["audio_projection_layer"],
)


_register_composite_model(
    model_type="mistral3",
    projector_key="model.multi_modal_projector",
)


_register_composite_model(
    model_type="mllama",
    vision_model_keys=["vision_model"],
)


_register_composite_model(
    model_type="paligemma",
)


_register_composite_model(
    model_type="qwen2_audio",
    vision_model_keys=["audio_tower"],
)


_register_composite_model(
    model_type="qwen2_5_omni_thinker",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks", "audio_tower"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen2_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen2_5_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_vl_moe",
    projector_key="visual.merger",
    vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_omni_moe_thinker",
    projector_key="visual.merger",
    vision_model_keys=[
        "visual.pos_embed",
        "visual.patch_embed",
        "visual.blocks",
        "visual.deepstack_merger_list",
        "audio_tower",
    ],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_5",
    projector_key="model.visual.merger",
    vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_5_moe",
    projector_key="model.visual.merger",
    vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="video_llava",
)
