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

"""Voxtral data processor.

Author: yufeng.ma
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ...extras import logging
from ...extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX
from .processor_utils import DatasetProcessor, apply_prompt_pool_top1_to_example, greedy_knapsack


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)

_SEGMENT_DURATION_RE = re.compile(r"_seg\\d+_(\\d+(?:\\.\\d+)?)-(\\d+(?:\\.\\d+)?)\\.wav$")


def _get_audio_num_chunks(audio_path: str, target_sampling_rate: int, chunk_samples: int) -> int:
    """Returns the number of Voxtral 30s chunks for the given audio file."""
    audio_path = audio_path.strip()
    if audio_path.startswith("file://"):
        audio_path = audio_path[7:]

    match = _SEGMENT_DURATION_RE.search(audio_path)
    if match is not None:
        start, end = float(match.group(1)), float(match.group(2))
        duration = max(0.0, end - start)
        resampled = int(math.ceil(duration * float(target_sampling_rate)))
        return max(1, int(math.ceil(resampled / float(chunk_samples))))

    try:
        import soundfile as sf

        info = sf.info(audio_path)
        frames = int(getattr(info, "frames", 0) or 0)
        samplerate = int(getattr(info, "samplerate", 0) or 0)
        if frames <= 0 or samplerate <= 0:
            return 1

        # ceil(frames/samplerate / (chunk_samples/target_sr))
        num = frames * int(target_sampling_rate)
        den = samplerate * int(chunk_samples)
        return max(1, (num + den - 1) // den)
    except Exception:  # noqa: BLE001
        pass

    # Fallback (slower): decode+resample.
    try:
        from transformers.audio_utils import load_audio

        wav = load_audio(audio_path, sampling_rate=target_sampling_rate)
        samples = int(getattr(wav, "shape", [0])[0] if hasattr(wav, "shape") else len(wav))
        return max(1, (samples + chunk_samples - 1) // chunk_samples)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to infer Voxtral chunk count for: {audio_path}") from e


def _get_voxtral_model_id(processor: Any, tokenizer: Any) -> str:
    for obj in (processor, tokenizer):
        for attr in ("name_or_path", "_name_or_path"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


@dataclass(frozen=True)
class VoxtralTranscriptionTemplate:
    prefix_ids: tuple[int, ...]  # includes <s> [INST] [BEGIN_AUDIO]
    audio_token_id: int
    end_inst_id: int
    lang_id: int
    colon_id: int
    transcribe_id: int
    audio_tokens_per_chunk: int
    chunk_samples: int
    target_sampling_rate: int

    def build_prompt_ids(self, num_chunks: int, language_ids: list[int]) -> list[int]:
        if num_chunks <= 0:
            raise ValueError(f"Voxtral num_chunks must be positive, got {num_chunks}.")

        prompt_ids = list(self.prefix_ids)
        prompt_ids += [self.audio_token_id] * (self.audio_tokens_per_chunk * num_chunks)
        prompt_ids += [self.end_inst_id, self.lang_id, self.colon_id]
        prompt_ids += list(language_ids)
        prompt_ids += [self.transcribe_id]
        return prompt_ids


_TRANSCRIPTION_TEMPLATE_CACHE: dict[tuple[str, int, int], VoxtralTranscriptionTemplate] = {}


def _infer_audio_token_id(input_ids: list[int]) -> int:
    """Infer Voxtral `audio_token_id` from a transcription request by finding the longest repeated run."""
    if not input_ids:
        raise ValueError("Empty Voxtral input_ids; cannot infer audio_token_id.")

    best_token = input_ids[0]
    best_len = 1
    cur_token = input_ids[0]
    cur_len = 1
    for token in input_ids[1:]:
        if token == cur_token:
            cur_len += 1
            continue
        if cur_len > best_len:
            best_len = cur_len
            best_token = cur_token
        cur_token = token
        cur_len = 1
    if cur_len > best_len:
        best_len = cur_len
        best_token = cur_token

    # Voxtral uses hundreds of repeated audio placeholders per chunk; be conservative here.
    if best_len < 32:
        raise ValueError("Failed to infer Voxtral audio_token_id (no sufficiently long repeated token run found).")
    return int(best_token)


def _get_voxtral_transcription_template(processor: Any, tokenizer: Any, template: Any) -> VoxtralTranscriptionTemplate:
    if processor is None:
        raise ValueError("VoxtralProcessor is required for Voxtral transcription template.")

    model_id = _get_voxtral_model_id(processor, tokenizer) or getattr(tokenizer, "__class__", type(tokenizer)).__name__
    chunk_samples = int(getattr(getattr(template, "mm_plugin", None), "pad_to_multiple_of", 480000))
    target_sampling_rate = int(getattr(processor, "audio_sampling_rate", 16000) or 16000)
    cache_key = (model_id, chunk_samples, target_sampling_rate)
    cached = _TRANSCRIPTION_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dummy_audio = np.zeros(target_sampling_rate, dtype=np.float32)  # ~1s
    inputs = processor.apply_transcription_request(
        language="en",
        audio=dummy_audio,
        model_id=_get_voxtral_model_id(processor, tokenizer) or "dummy",
        sampling_rate=target_sampling_rate,
        format=["wav"],
        common_kwargs={"return_tensors": "pt", "return_dict": True, "tokenize": True},
    )

    input_ids = inputs["input_ids"][0].tolist()
    audio_token_id = _infer_audio_token_id(input_ids)
    input_features = inputs.get("input_features", None)
    num_chunks = int(getattr(input_features, "shape", [1])[0] if input_features is not None else 1)

    first_audio = input_ids.index(audio_token_id)
    last_audio = len(input_ids) - 1 - input_ids[::-1].index(audio_token_id)
    audio_count = last_audio - first_audio + 1
    if audio_count % num_chunks != 0:
        raise ValueError(
            f"Unexpected Voxtral audio token count {audio_count} for {num_chunks} chunks; cannot derive template."
        )
    audio_tokens_per_chunk = audio_count // num_chunks

    suffix = input_ids[last_audio + 1 :]
    if len(suffix) < 4:
        raise ValueError("Unexpected Voxtral transcription suffix; cannot derive template.")

    tmpl = VoxtralTranscriptionTemplate(
        prefix_ids=tuple(input_ids[:first_audio]),
        audio_token_id=audio_token_id,
        end_inst_id=int(suffix[0]),
        lang_id=int(suffix[1]),
        colon_id=int(suffix[2]),
        transcribe_id=int(suffix[-1]),
        audio_tokens_per_chunk=int(audio_tokens_per_chunk),
        chunk_samples=chunk_samples,
        target_sampling_rate=target_sampling_rate,
    )

    _TRANSCRIPTION_TEMPLATE_CACHE[cache_key] = tmpl
    return tmpl


def _get_voxtral_example_language(examples: dict[str, list[Any]], index: int, data_args: Any) -> str:
    for key in ("language", "lang", "_language", "_lang"):
        if key not in examples:
            continue
        value = examples[key][index]
        if isinstance(value, str) and value.strip():
            return value.strip()

    fallback = getattr(data_args, "voxtral_transcription_language", None)
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()

    raise ValueError(
        "Voxtral transcription template requires a language. "
        "Set `voxtral_transcription_language` or provide a per-sample `lang`/`language` column."
    )


def _maybe_parse_audio_json(audio: str) -> dict[str, Any] | None:
    audio = audio.strip()
    if not (audio.startswith("{") and audio.endswith("}")):
        return None
    try:
        obj = json.loads(audio)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _normalize_audio_path(audio: AudioInput) -> str:
    if not isinstance(audio, str):
        raise TypeError(f"Voxtral audio input must be a file path string, got {type(audio)}.")

    obj = _maybe_parse_audio_json(audio)
    if obj is None:
        return audio

    path = obj.get("path") or obj.get("wav_path") or obj.get("audio_path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Invalid Voxtral audio json: {audio}")
    return path


def _content_with_audios(text: str, audio_iter: Any) -> str | list[dict[str, str]]:
    if AUDIO_PLACEHOLDER not in text:
        return text

    parts = text.split(AUDIO_PLACEHOLDER)
    chunks: list[dict[str, str]] = []
    for i, part in enumerate(parts):
        if part:
            chunks.append({"type": "text", "text": part})
        if i == len(parts) - 1:
            continue

        try:
            audio = next(audio_iter)
        except StopIteration as e:
            raise ValueError("Audio placeholders exceed the number of provided audios.") from e

        chunks.append({"type": "audio", "path": _normalize_audio_path(audio)})

    return chunks


def _build_voxtral_conversation(
    prompt: list[dict[str, str]],
    response: list[dict[str, str]],
    system: str | None,
    audios: list[AudioInput],
) -> list[dict[str, Any]]:
    audio_iter = iter(audios or [])
    conversation: list[dict[str, Any]] = []
    if system:
        conversation.append({"role": "system", "content": str(system)})

    for message in prompt + response:
        role = message.get("role")
        if role not in ("user", "assistant", "system", "tool"):
            raise ValueError(f"Unsupported role for Voxtral: {role}")

        content = str(message.get("content") or "")
        content_out = _content_with_audios(content, audio_iter)
        conversation.append({"role": role, "content": content_out})

    # Fail fast on extra audios to prevent silent misalignment.
    try:
        next(audio_iter)
    except StopIteration:
        return conversation
    raise ValueError("The number of audios does not match the number of audio placeholders in messages.")


def _find_assistant_start(
    tokenizer: PreTrainedTokenizer,
    input_ids: list[int],
    assistant_text: str,
    eos_suffix_ids: list[int],
) -> int:
    assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    if not eos_suffix_ids:
        eos_suffix_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    suffix_len = len(eos_suffix_ids)
    assistant_len = len(assistant_ids)
    if suffix_len and input_ids[-suffix_len:] != eos_suffix_ids:
        raise ValueError("Unexpected Voxtral sequence suffix; cannot locate assistant span.")

    start = len(input_ids) - suffix_len - assistant_len
    if start >= 0 and input_ids[start : start + assistant_len] == assistant_ids:
        return start

    # Fallback: search from the end for a match right before the EOS suffix.
    limit = len(input_ids) - suffix_len
    for i in range(limit - assistant_len, -1, -1):
        if (
            input_ids[i : i + assistant_len] == assistant_ids
            and input_ids[i + assistant_len : limit] == eos_suffix_ids
        ):
            return i

    raise ValueError("Failed to locate assistant content span in Voxtral tokenized sequence.")


def _get_default_eos_suffix_ids(tokenizer: PreTrainedTokenizer) -> list[int]:
    # Best-effort: infer EOS suffix ids by probing a minimal chat.
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=True,
    )
    y_ids = tokenizer.encode("y", add_special_tokens=False)
    for i in range(len(dummy) - len(y_ids), -1, -1):
        if dummy[i : i + len(y_ids)] == y_ids:
            suffix = dummy[i + len(y_ids) :]
            if suffix:
                return list(suffix)
            break

    if tokenizer.eos_token_id is None:
        return []
    return [tokenizer.eos_token_id]


@dataclass
class VoxtralSupervisedDatasetProcessor(DatasetProcessor):
    _eos_suffix_ids: list[int] = field(default_factory=list, init=False, repr=False)

    def _get_eos_suffix_ids(self) -> list[int]:
        if not self._eos_suffix_ids:
            self._eos_suffix_ids = _get_default_eos_suffix_ids(self.tokenizer)
        return self._eos_suffix_ids

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: str | None,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        language: str | None = None,
    ) -> tuple[list[int], list[int]]:
        if len(response) != 1:
            raise ValueError("Voxtral supervised processor expects exactly one assistant response.")

        use_chat_template = bool(getattr(self.data_args, "voxtral_chat_template", False))
        if use_chat_template:
            conversation = _build_voxtral_conversation(prompt, response, system, audios)
            input_ids: list[int] = self.tokenizer.apply_chat_template(conversation, tokenize=True)

            if self.data_args.train_on_prompt:
                labels = list(input_ids)
            else:
                eos_suffix_ids = self._get_eos_suffix_ids()
                assistant_text = str(response[-1].get("content") or "")
                prompt_len = _find_assistant_start(self.tokenizer, input_ids, assistant_text, eos_suffix_ids)
                labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
        else:
            if language is None:
                raise ValueError("Missing Voxtral transcription language.")
            if len(audios) != 1:
                raise ValueError(
                    f"Voxtral transcription template expects exactly one audio, got {len(audios)}. "
                    "Use `voxtral_chat_template: true` for multi-audio chat-style datasets."
                )

            tmpl = _get_voxtral_transcription_template(self.processor, self.tokenizer, self.template)
            audio_path = _normalize_audio_path(audios[0])
            num_chunks = _get_audio_num_chunks(audio_path, tmpl.target_sampling_rate, tmpl.chunk_samples)
            language_ids = self.tokenizer.encode(language, add_special_tokens=False)
            prompt_ids = tmpl.build_prompt_ids(num_chunks=num_chunks, language_ids=language_ids)

            cutoff_len = int(self.data_args.cutoff_len)
            if cutoff_len > 0 and len(prompt_ids) > cutoff_len:
                raise ValueError(
                    f"Voxtral audio prompt length {len(prompt_ids)} exceeds cutoff_len={cutoff_len}. "
                    "Please increase cutoff_len or filter long audios."
                )

            assistant_text = str(response[-1].get("content") or "")
            response_ids = self.tokenizer.encode(assistant_text, add_special_tokens=False)
            eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

            input_ids = prompt_ids + response_ids + eos_ids
            if self.data_args.train_on_prompt:
                labels = list(input_ids)
            else:
                labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids + eos_ids

        cutoff_len = int(self.data_args.cutoff_len)
        if cutoff_len > 0 and len(input_ids) > cutoff_len:
            if use_chat_template and audios:
                tmpl = _get_voxtral_transcription_template(self.processor, self.tokenizer, self.template)
                if tmpl.audio_token_id in input_ids:
                    last_audio = len(input_ids) - 1 - input_ids[::-1].index(tmpl.audio_token_id)
                    if cutoff_len <= last_audio:
                        raise ValueError(
                            f"Voxtral truncation would drop audio placeholders (last_audio_idx={last_audio}, "
                            f"cutoff_len={cutoff_len}). Please increase cutoff_len."
                        )
            input_ids = input_ids[:cutoff_len]
            labels = labels[:cutoff_len]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        has_pool = getattr(self.data_args, "dynamic_prompt_sampling", False) and "_prompt_pool" in examples
        use_chat_template = bool(getattr(self.data_args, "voxtral_chat_template", False))
        for i in range(len(examples["_prompt"])):
            prompt = examples["_prompt"][i]
            response = examples["_response"][i]
            system = examples["_system"][i]
            if has_pool:
                prompt, response, system = apply_prompt_pool_top1_to_example(
                    prompt, response, system, examples["_prompt_pool"][i]
                )

            if len(prompt) % 2 != 1 or len(response) != 1:
                logger.warning_rank0(f"Dropped invalid example: {prompt + response}")
                continue

            language = None
            if not use_chat_template:
                language = _get_voxtral_example_language(examples, i, self.data_args)

            try:
                input_ids, labels = self._encode_data_example(
                    prompt=prompt,
                    response=response,
                    system=system,
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                    language=language,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning_rank0(f"Dropped Voxtral example due to encoding error: {e}")
                continue
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class VoxtralPackedSupervisedDatasetProcessor(VoxtralSupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        has_pool = getattr(self.data_args, "dynamic_prompt_sampling", False) and "_prompt_pool" in examples
        use_chat_template = bool(getattr(self.data_args, "voxtral_chat_template", False))
        for i in range(len(examples["_prompt"])):
            prompt = examples["_prompt"][i]
            response = examples["_response"][i]
            system = examples["_system"][i]
            if has_pool:
                prompt, response, system = apply_prompt_pool_top1_to_example(
                    prompt, response, system, examples["_prompt_pool"][i]
                )

            if len(prompt) % 2 != 1 or len(response) != 1:
                logger.warning_rank0(f"Dropped invalid example: {prompt + response}")
                continue

            language = None
            if not use_chat_template:
                language = _get_voxtral_example_language(examples, i, self.data_args)

            try:
                input_ids, labels = self._encode_data_example(
                    prompt=prompt,
                    response=response,
                    system=system,
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                    language=language,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning_rank0(f"Dropped Voxtral example due to encoding error: {e}")
                continue
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            # Neat packing: mask boundary labels to avoid cross-segment loss contributions.
            if self.data_args.neat_packing and packed_labels:
                packed_labels[0] = IGNORE_INDEX
                for j in range(1, len(packed_labels)):
                    if packed_attention_masks[j] != packed_attention_masks[j - 1] and packed_attention_masks[j] != 0:
                        packed_labels[j] = IGNORE_INDEX

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs


@dataclass
class VoxtralUnsupervisedDatasetProcessor(DatasetProcessor):
    _eos_suffix_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _test_tokenizer: PreTrainedTokenizer | None = field(default=None, init=False, repr=False)

    def _get_eos_suffix_ids(self) -> list[int]:
        if not self._eos_suffix_ids:
            self._eos_suffix_ids = _get_default_eos_suffix_ids(self.tokenizer)
        return self._eos_suffix_ids

    def _get_test_tokenizer(self) -> PreTrainedTokenizer:
        if self._test_tokenizer is not None:
            return self._test_tokenizer

        try:
            from mistral_common.protocol.instruct.validator import ValidationMode
            from transformers import AutoTokenizer

            self._test_tokenizer = AutoTokenizer.from_pretrained(
                getattr(self.tokenizer, "name_or_path", None) or getattr(self.tokenizer, "_name_or_path", None),
                mode=ValidationMode.test,
                padding_side=getattr(self.tokenizer, "padding_side", "right"),
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Failed to load a test-mode tokenizer for Voxtral prompt-only generation.") from e

        return self._test_tokenizer

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: str | None,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        language: str | None = None,
    ) -> tuple[list[int], list[int]]:
        eos_suffix_ids = self._get_eos_suffix_ids()
        use_chat_template = bool(getattr(self.data_args, "voxtral_chat_template", False))

        if use_chat_template:
            if len(response) == 1:
                conversation = _build_voxtral_conversation(prompt, response, system, audios)
                full_ids: list[int] = self.tokenizer.apply_chat_template(conversation, tokenize=True)
                assistant_text = str(response[-1].get("content") or "")
                prompt_len = _find_assistant_start(self.tokenizer, full_ids, assistant_text, eos_suffix_ids)
                input_ids = full_ids[:prompt_len]
                labels = full_ids[prompt_len:]
            else:
                # Prompt-only inference: use test-mode tokenizer (expects last role user/tool).
                conversation = _build_voxtral_conversation(prompt, [], system, audios)
                input_ids = self._get_test_tokenizer().apply_chat_template(conversation, tokenize=True)
                labels = []
        else:
            if language is None:
                raise ValueError("Missing Voxtral transcription language.")
            if len(audios) != 1:
                raise ValueError(
                    f"Voxtral transcription template expects exactly one audio, got {len(audios)}. "
                    "Use `voxtral_chat_template: true` for multi-audio chat-style datasets."
                )

            tmpl = _get_voxtral_transcription_template(self.processor, self.tokenizer, self.template)
            audio_path = _normalize_audio_path(audios[0])
            num_chunks = _get_audio_num_chunks(audio_path, tmpl.target_sampling_rate, tmpl.chunk_samples)
            language_ids = self.tokenizer.encode(language, add_special_tokens=False)
            input_ids = tmpl.build_prompt_ids(num_chunks=num_chunks, language_ids=language_ids)

            if len(response) == 1:
                assistant_text = str(response[-1].get("content") or "")
                labels = self.tokenizer.encode(assistant_text, add_special_tokens=False)
                if self.tokenizer.eos_token_id is not None:
                    labels += [self.tokenizer.eos_token_id]
            else:
                labels = []

        cutoff_len = int(self.data_args.cutoff_len)
        if cutoff_len > 0:
            # Voxtral cannot truncate `input_ids` when audio placeholders are present (it would desync with `input_features`).
            if len(input_ids) > cutoff_len:
                raise ValueError(
                    f"Voxtral prompt length {len(input_ids)} exceeds cutoff_len={cutoff_len}. "
                    "Please increase cutoff_len or filter long audios."
                )

            max_target_len = cutoff_len - len(input_ids)
            if max_target_len < 0:
                max_target_len = 0
            labels = labels[:max_target_len]
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        has_pool = getattr(self.data_args, "dynamic_prompt_sampling", False) and "_prompt_pool" in examples
        use_chat_template = bool(getattr(self.data_args, "voxtral_chat_template", False))
        for i in range(len(examples["_prompt"])):
            prompt = examples["_prompt"][i]
            response = examples["_response"][i]
            system = examples["_system"][i]
            if has_pool:
                prompt, response, system = apply_prompt_pool_top1_to_example(
                    prompt, response, system, examples["_prompt_pool"][i]
                )

            if len(prompt) % 2 != 1:
                logger.warning_rank0(f"Dropped invalid example: {prompt + response}")
                continue

            language = None
            if not use_chat_template:
                language = _get_voxtral_example_language(examples, i, self.data_args)

            try:
                input_ids, labels = self._encode_data_example(
                    prompt=prompt,
                    response=response,
                    system=system,
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                    language=language,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning_rank0(f"Dropped Voxtral example due to encoding error: {e}")
                continue
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(example["labels"], skip_special_tokens=False)))
