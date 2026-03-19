# Copyright 2025 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
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
import random
import time
from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import get_worker_info
from transformers import DataCollatorForSeq2Seq

from ..extras import logging
from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER
from ..extras.misc import is_env_enabled
from ..extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image

try:
    import torchaudio.functional as AF  # type: ignore
except Exception:  # noqa: BLE001
    AF = None


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


logger = logging.get_logger(__name__)


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""Expand 2d attention mask to 4d attention mask.

    Expand the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    handle packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    # Create a non-padding mask.
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    # Create indices for comparison.
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    # Create a lower triangular mask.
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images, videos and audios.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None
    audio_specaugment: bool = False
    audio_specaugment_mask_param: float = 0.1
    audio_specaugment_num_masks: int = 2
    audio_specaugment_fill_value: float = 0.0
    _specaug_rng: Optional[random.Random] = field(default=None, init=False, repr=False)
    _specaug_rng_seeded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            self.get_rope_func = self.model.get_rope_index  # transformers < 4.52.0 or qwen2.5 omni
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index  # transformers >= 4.52.0
        else:
            self.get_rope_func = None

    def _get_specaug_rng(self) -> random.Random:
        if self._specaug_rng is None:
            self._specaug_rng = random.Random()

        if not self._specaug_rng_seeded:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0

            rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    rank = int(torch.distributed.get_rank())
                except Exception:
                    rank = 0

            worker_seed = int(torch.initial_seed())
            mixed = (worker_seed + rank * 1000 + worker_id) % (2**32)
            self._specaug_rng.seed(mixed)
            self._specaug_rng_seeded = True

        return self._specaug_rng

    def _apply_audio_specaugment(self, audio: np.ndarray) -> np.ndarray:
        if not self.audio_specaugment:
            return audio

        if self.audio_specaugment_mask_param <= 0 or self.audio_specaugment_num_masks <= 0:
            return audio

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        length = int(arr.shape[0])
        if length <= 1:
            return arr.astype(np.float32, copy=False)

        out = arr.astype(np.float32, copy=True)
        max_len = int(length * float(self.audio_specaugment_mask_param))
        if max_len <= 0:
            return out

        rng = self._get_specaug_rng()
        if AF is None:
            for _ in range(int(self.audio_specaugment_num_masks)):
                mask_len = rng.randint(0, max_len)
                if mask_len <= 0:
                    continue
                start = rng.randrange(0, max(1, length - mask_len + 1))
                out[start : start + mask_len] = float(self.audio_specaugment_fill_value)
            return out

        x = torch.from_numpy(out).unsqueeze(0)  # [1, T]
        for _ in range(int(self.audio_specaugment_num_masks)):
            seed = rng.randrange(0, 2**31 - 1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                x = AF.mask_along_axis(
                    x,
                    mask_param=max_len,
                    mask_value=float(self.audio_specaugment_fill_value),
                    axis=1,
                )
        return x.squeeze(0).cpu().numpy().astype(np.float32)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        perf_enabled = is_env_enabled("LLAMAFACTORY_PERF_LOG")
        dl_perf_enabled = perf_enabled and is_env_enabled("LLAMAFACTORY_DATALOADER_PERF_LOG")
        preserve_audio_meta = is_env_enabled("LLAMAFACTORY_PRESERVE_AUDIO_META")

        t_collate0 = time.perf_counter() if dl_perf_enabled else 0.0
        perf_batch: dict[str, Any] = {}

        dp_encode_sum = 0.0
        dp_pack_sum = 0.0
        dp_total_sum = 0.0
        dp_encode_n = 0
        dp_pack_n = 0
        dp_total_n = 0

        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        batch_audio_durations_meta: list[float] = []
        for feature in features:
            v = feature.pop("perf_dp_encode_ms", None)
            if v is not None:
                try:
                    dp_encode_sum += float(v)
                    dp_encode_n += 1
                except Exception:
                    pass
            v = feature.pop("perf_dp_pack_ms", None)
            if v is not None:
                try:
                    dp_pack_sum += float(v)
                    dp_pack_n += 1
                except Exception:
                    pass
            v = feature.pop("perf_dp_total_ms", None)
            if v is not None:
                try:
                    dp_total_sum += float(v)
                    dp_total_n += 1
                except Exception:
                    pass

            d = feature.pop("audio_duration_sec", None)
            try:
                batch_audio_durations_meta.append(float(d) if d is not None else 0.0)
            except Exception:  # noqa: BLE001
                batch_audio_durations_meta.append(0.0)
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])

        if perf_enabled:
            if dp_encode_n > 0:
                perf_batch["perf_dl_dp_encode_ms"] = dp_encode_sum / float(dp_encode_n)
                perf_batch["perf_dl_dp_encode_n"] = int(dp_encode_n)
            if dp_pack_n > 0:
                perf_batch["perf_dl_dp_pack_ms"] = dp_pack_sum / float(dp_pack_n)
                perf_batch["perf_dl_dp_pack_n"] = int(dp_pack_n)
            if dp_total_n > 0:
                perf_batch["perf_dl_dp_total_ms"] = dp_total_sum / float(dp_total_n)
                perf_batch["perf_dl_dp_total_n"] = int(dp_total_n)

        # NOTE:
        # Some workflows (e.g. prompt packing) pre-compute `position_ids` and produce fixed-length sequences.
        # In such cases, appending dummy multimodal tokens would (1) exceed the model's max length and
        # (2) create ragged `position_ids`, which breaks `tokenizer.pad`.
        allow_fake_mm = not any("position_ids" in f for f in features)

        fake_input_ids = []
        if (
            allow_fake_mm
            and self.template.mm_plugin.image_token is not None
            and sum(batch_imglens) == 0
            and sum(batch_vidlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, fake_images, [], [], self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, fake_images, [], [], self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_images = fake_images
            batch_imglens[0] = 1

        if (
            allow_fake_mm and self.template.mm_plugin.audio_token is not None and sum(batch_audlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": AUDIO_PLACEHOLDER}]
            fake_audios = [np.zeros(1600)]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, [], [], fake_audios, self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, [], [], fake_audios, self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_audios = fake_audios
            batch_audlens[0] = 1

        if len(fake_input_ids) != 0:
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]

            if "position_ids" in features[0]:
                pos = features[0]["position_ids"]
                pad = [0] * len(fake_input_ids)
                if torch.is_tensor(pos):
                    pad_tensor = torch.zeros(len(fake_input_ids), dtype=pos.dtype, device=pos.device)
                    if self.tokenizer.padding_side == "right":
                        pos = torch.cat([pos, pad_tensor], dim=-1)
                    else:
                        pos = torch.cat([pad_tensor, pos], dim=-1)
                elif isinstance(pos, list):
                    if self.tokenizer.padding_side == "right":
                        pos = pos + pad
                    else:
                        pos = pad + pos
                features[0]["position_ids"] = pos

            batch_input_ids[0] = features[0]["input_ids"]

        if self.audio_specaugment and len(batch_audios) != 0:
            t_spec0 = time.perf_counter() if dl_perf_enabled else 0.0
            if self.processor is None:
                raise ValueError("Processor is required for audio SpecAugment.")
            if self.template is None:
                raise ValueError("Template is required for audio SpecAugment.")

            audio_sampling_rate = getattr(self.processor, "audio_sampling_rate", 16000)
            max_retries = int(os.getenv("LLAMAFACTORY_AUDIO_LOAD_RETRIES", "1"))
            retry_sleep_sec = float(os.getenv("LLAMAFACTORY_AUDIO_LOAD_RETRY_SLEEP", "0.2"))
            log_limit = int(os.getenv("LLAMAFACTORY_AUDIO_LOAD_ERROR_LOG_LIMIT", "20"))
            logged = int(getattr(self, "_audio_specaug_error_logged", 0))
            suppressed = bool(getattr(self, "_audio_specaug_error_suppressed", False))

            augmented_audios: list[Any] = []
            for audio in batch_audios:
                # Best-effort SpecAugment: if waveform loading fails, keep the original audio and let the
                # model/plugin handle retries + loss masking later.
                if isinstance(audio, np.ndarray):
                    augmented = self._apply_audio_specaugment(audio)
                    if preserve_audio_meta:
                        augmented_audios.append({"raw": audio, "array": augmented})
                    else:
                        augmented_audios.append(augmented)
                    continue

                last_error: Exception | None = None
                y: np.ndarray | None = None
                for attempt in range(max(0, max_retries) + 1):
                    try:
                        y, _ = self.template.mm_plugin._load_single_audio(audio, float(audio_sampling_rate))
                        last_error = None
                        break
                    except Exception as e:  # noqa: BLE001
                        last_error = e
                        is_not_found = isinstance(e, FileNotFoundError) or (
                            isinstance(e, OSError) and getattr(e, "errno", None) == 2
                        )
                        if is_not_found or attempt >= max(0, max_retries):
                            break
                        time.sleep(retry_sleep_sec * float(attempt + 1))

                if last_error is not None or y is None:
                    augmented_audios.append(audio)
                    if logged < log_limit:
                        logger.warning_rank0(
                            "SpecAugment skipped due to audio load error (audio=%r): %s",
                            audio,
                            repr(last_error),
                        )
                        logged += 1
                    elif not suppressed:
                        logger.warning_rank0(
                            "Too many SpecAugment audio load errors (%d+); suppressing further logs.",
                            log_limit,
                        )
                        suppressed = True
                    continue

                augmented = self._apply_audio_specaugment(y)
                if preserve_audio_meta:
                    augmented_audios.append({"raw": audio, "array": augmented})
                else:
                    augmented_audios.append(augmented)

            setattr(self, "_audio_specaug_error_logged", logged)
            setattr(self, "_audio_specaug_error_suppressed", suppressed)
            batch_audios = augmented_audios
            if dl_perf_enabled:
                perf_batch["perf_dl_specaug_ms"] = (time.perf_counter() - t_spec0) * 1000.0
                perf_batch["perf_dl_specaug_n"] = 1

        t_mm0 = time.perf_counter() if dl_perf_enabled else 0.0
        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
            self.processor,
        )
        if dl_perf_enabled:
            perf_batch["perf_dl_mm_inputs_ms"] = (time.perf_counter() - t_mm0) * 1000.0
            perf_batch["perf_dl_mm_inputs_n"] = 1

        # Pull mm_plugin internal perf (if any) into collator-level metrics and ensure they won't reach model forward.
        for perf_key, dl_key in (
            ("perf_mm_audio_load_ms", "perf_dl_audio_load_ms"),
            ("perf_mm_speech_tokenizer_ms", "perf_dl_speech_tokenizer_ms"),
            ("perf_mm_feature_extractor_ms", "perf_dl_feature_extractor_ms"),
        ):
            v = mm_inputs.pop(perf_key, None)
            if v is None:
                continue
            try:
                perf_batch[dl_key] = float(v)
                perf_batch[dl_key.replace("_ms", "_n")] = 1
            except Exception:
                continue

        # Reconstruct per-sample audio durations from mm_plugin's per-audio durations when available.
        # This makes `audio_hours` robust even when dataset metadata doesn't carry duration info.
        audio_duration_sec_by_audio = mm_inputs.pop("audio_duration_sec_by_audio", None)
        batch_audio_durations = batch_audio_durations_meta
        if audio_duration_sec_by_audio is not None and batch_audlens:
            by_audio_list = None
            try:
                if torch.is_tensor(audio_duration_sec_by_audio):
                    by_audio_list = audio_duration_sec_by_audio.to(dtype=torch.float32, device="cpu").tolist()
                elif isinstance(audio_duration_sec_by_audio, (list, tuple)):
                    by_audio_list = [float(x) if x is not None else 0.0 for x in audio_duration_sec_by_audio]
            except Exception:  # noqa: BLE001
                by_audio_list = None

            expected = int(sum(int(x) for x in batch_audlens)) if batch_audlens else 0
            if by_audio_list is not None and expected > 0:
                if len(by_audio_list) != expected:
                    # Best-effort: align to expected flattened audio count.
                    if len(by_audio_list) < expected:
                        by_audio_list = by_audio_list + [0.0] * (expected - len(by_audio_list))
                    else:
                        by_audio_list = by_audio_list[:expected]

                mm_per_sample: list[float] = []
                off = 0
                for n in batch_audlens:
                    nn = int(n)
                    if nn <= 0:
                        mm_per_sample.append(0.0)
                        continue
                    s = 0.0
                    for x in by_audio_list[off : off + nn]:
                        try:
                            s += float(x)
                        except Exception:
                            continue
                    mm_per_sample.append(float(s))
                    off += nn

                if len(mm_per_sample) == len(batch_audio_durations_meta):
                    merged: list[float] = []
                    for meta, mm_dur in zip(batch_audio_durations_meta, mm_per_sample):
                        merged.append(float(meta) if float(meta) > 0 else float(mm_dur))
                    batch_audio_durations = merged

        audio_feature_load_fail = mm_inputs.pop("feature_load_fail_mask", None)
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: dict[str, torch.Tensor] = super().__call__(features)

        if audio_feature_load_fail is not None:
            try:
                fail_list = (
                    audio_feature_load_fail.to(dtype=torch.bool, device="cpu").tolist()
                    if torch.is_tensor(audio_feature_load_fail)
                    else list(audio_feature_load_fail)
                )
            except Exception:
                fail_list = None

            if fail_list and any(bool(x) for x in fail_list):
                # Identify which packed segments to mask. For neat packing, `attention_mask` contains segment ids (1..N).
                # We mask the entire segment's labels when its audio waveform cannot be loaded after retries.
                labels = features.get("labels")
                input_ids = features.get("input_ids")
                attn = features.get("attention_mask")
                if torch.is_tensor(labels) and torch.is_tensor(input_ids) and torch.is_tensor(attn) and labels.ndim == 2:
                    expected_audios = int(sum(int(x) for x in batch_audlens)) if batch_audlens else 0
                    if expected_audios and len(fail_list) != expected_audios:
                        logger.warning_rank0(
                            "Audio load-fail mask length mismatch: got %d, expected %d. "
                            "Falling back to masking whole samples that contain any failed audio.",
                            len(fail_list),
                            expected_audios,
                        )
                        fail_list = (fail_list + [False] * expected_audios)[:expected_audios]

                    audio_token_id = None
                    if self.model is not None and getattr(self.model, "config", None) is not None:
                        audio_token_id = getattr(self.model.config, "audio_token_index", None)
                    if audio_token_id is None and self.template.mm_plugin.audio_token is not None:
                        try:
                            audio_token_id = int(self.tokenizer.convert_tokens_to_ids(self.template.mm_plugin.audio_token))
                        except Exception:
                            audio_token_id = None

                    use_segment_ids = attn.dtype != torch.bool
                    try:
                        use_segment_ids = use_segment_ids and int(attn.max().item()) > 1
                    except Exception:
                        use_segment_ids = False

                    speech_attn = mm_inputs.get("speech_attention_mask", None)
                    audio_group_size = 5
                    if self.processor is not None:
                        audio_group_size = int(getattr(self.processor, "audio_group_size", 5) or 5)

                    audio_seqlens: list[int] | None = None
                    if torch.is_tensor(speech_attn) and speech_attn.ndim == 2 and audio_group_size > 0:
                        lengths = speech_attn.to(dtype=torch.long, device="cpu").sum(-1).tolist()
                        audio_seqlens = [int((int(l) + audio_group_size - 1) // audio_group_size) for l in lengths]
                        if expected_audios and len(audio_seqlens) != expected_audios:
                            logger.warning_rank0(
                                "Audio seqlen list length mismatch: got %d, expected %d. "
                                "Falling back to masking whole samples that contain any failed audio.",
                                len(audio_seqlens),
                                expected_audios,
                            )
                            audio_seqlens = (audio_seqlens + [1] * expected_audios)[:expected_audios]

                    # Walk through each batch item and mask labels for segments whose audio failed.
                    audio_offset = 0
                    bsz = int(labels.shape[0])
                    for b in range(bsz):
                        audlen = int(batch_audlens[b]) if b < len(batch_audlens) else 0
                        if audlen <= 0:
                            continue

                        sub_fail = fail_list[audio_offset : audio_offset + audlen]
                        if not any(bool(x) for x in sub_fail):
                            audio_offset += audlen
                            continue

                        # Fallback: if we can't map to segment ids, ignore the whole packed sample.
                        if not use_segment_ids or audio_token_id is None or audio_seqlens is None:
                            labels[b].fill_(IGNORE_INDEX)
                            audio_offset += audlen
                            continue

                        seqlens_slice = audio_seqlens[audio_offset : audio_offset + audlen]
                        audio_offset += audlen

                        audio_pos = (input_ids[b] == int(audio_token_id)).nonzero(as_tuple=False).flatten().tolist()
                        pos_cursor = 0
                        seg_ids: set[int] = set()
                        mapping_failed = False
                        for j in range(audlen):
                            n = int(seqlens_slice[j]) if j < len(seqlens_slice) else 0
                            if n < 0:
                                n = 0
                            positions = audio_pos[pos_cursor : pos_cursor + n]
                            pos_cursor += n

                            if not bool(sub_fail[j]):
                                continue

                            seg_id = None
                            for p in positions:
                                v = int(attn[b, p].item())
                                if v != 0:
                                    seg_id = v
                                    break
                            if seg_id is None:
                                mapping_failed = True
                                break
                            seg_ids.add(int(seg_id))

                        if mapping_failed or len(seg_ids) == 0:
                            labels[b].fill_(IGNORE_INDEX)
                            continue

                        for seg_id in seg_ids:
                            labels[b].masked_fill_(attn[b] == int(seg_id), IGNORE_INDEX)

                    features["labels"] = labels

        if self.get_rope_func is not None:
            rope_index_kwargs = {
                "input_ids": features["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": (features["attention_mask"] >= 1).float(),
            }
            if "mm_token_type_ids" in inspect.signature(self.get_rope_func).parameters:
                image_token_id = getattr(self.model.config, "image_token_id", None)
                video_token_id = getattr(self.model.config, "video_token_id", None)
                if image_token_id is not None or video_token_id is not None:
                    mm_token_type_ids = torch.zeros_like(features["input_ids"])
                    if image_token_id is not None:
                        mm_token_type_ids[features["input_ids"] == image_token_id] = 1
                    if video_token_id is not None:
                        mm_token_type_ids[features["input_ids"] == video_token_id] = 2
                    rope_index_kwargs["mm_token_type_ids"] = mm_token_type_ids
            if "second_per_grid_ts" in mm_inputs:  # for qwen2vl
                rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
            elif "video_second_per_grid" in mm_inputs:  # for qwen2.5 omni
                rope_index_kwargs["second_per_grids"] = mm_inputs.get("video_second_per_grid")

            if getattr(self.model.config, "model_type", None) in ["qwen2_5_omni_thinker", "qwen3_omni_moe_thinker"]:
                rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
                feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
                if feature_attention_mask is not None:  # FIXME: need to get video image lengths
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    rope_index_kwargs["audio_seqlens"] = audio_feature_lengths  # prepare for input

                features["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
                features["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(
                    dim=-1
                ).unsqueeze(-1)
            else:  # for qwen vl
                features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)

        if (
            self.model is not None
            and getattr(self.model.config, "model_type", None)
            in [
                "glm4v",
                "glm_ocr",
                "Keye",
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen2_5_omni_thinker",
                "qwen3_omni_moe_thinker",
                "qwen3_5",
                "qwen3_vl",
                "qwen3_vl_moe",
            ]
            and ("position_ids" not in features or features["position_ids"].dim() != 3)
        ):
            raise ValueError(f"{self.model.config.model_type} requires 3D position ids for mrope.")

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)

        audio_duration_tensor = None
        if batch_audio_durations:
            audio_duration_tensor = torch.tensor(batch_audio_durations, dtype=torch.float32)

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            out = {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}
            # NOTE: Keep audio_duration_sec at top-level so it won't be forwarded into the model by wrappers.
            if audio_duration_tensor is not None:
                out["audio_duration_sec"] = audio_duration_tensor
            if dl_perf_enabled:
                perf_batch["perf_dl_collate_ms"] = (time.perf_counter() - t_collate0) * 1000.0
                perf_batch["perf_dl_collate_n"] = 1
            out.update(perf_batch)
            return out

        if audio_duration_tensor is not None:
            features["audio_duration_sec"] = audio_duration_tensor
        if dl_perf_enabled:
            perf_batch["perf_dl_collate_ms"] = (time.perf_counter() - t_collate0) * 1000.0
            perf_batch["perf_dl_collate_n"] = 1

        # Qwen3-ASR: keep alignment metadata as tensors so Accelerate can concatenate microbatches.
        # Keep packed Qwen3-ASR audio transport trainer-safe: Accelerate may slice top-level tensors using the text
        # batch size, so reshape audio-major features into a sample-major container before returning the batch.
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        plugin_name = type(getattr(self.template, "mm_plugin", None)).__name__
        is_qwen3_asr = (isinstance(model_type, str) and model_type.startswith("qwen3_asr")) or (
            plugin_name == "Qwen3ASRPlugin"
        )
        if is_qwen3_asr:
            if (
                torch.is_tensor(features.get("input_features"))
                and torch.is_tensor(features.get("feature_attention_mask"))
                and features["input_features"].ndim == 3
                and features["feature_attention_mask"].ndim == 2
                and features["input_features"].shape[0] == int(sum(batch_audlens))
                and len(batch_audlens) > 0
            ):
                flat_input_features = features["input_features"]
                flat_feature_attention_mask = features["feature_attention_mask"]
                batch_size = len(batch_audlens)
                max_audios_per_sample = max(int(x) for x in batch_audlens)
                _, num_mel_bins, max_feature_len = flat_input_features.shape
                packed_input_features = flat_input_features.new_zeros(
                    (batch_size, max_audios_per_sample, num_mel_bins, max_feature_len)
                )
                packed_feature_attention_mask = flat_feature_attention_mask.new_zeros(
                    (batch_size, max_audios_per_sample, max_feature_len)
                )

                offset = 0
                for i, audlen in enumerate(batch_audlens):
                    audlen = int(audlen)
                    if audlen <= 0:
                        continue

                    next_offset = offset + audlen
                    packed_input_features[i, :audlen] = flat_input_features[offset:next_offset]
                    packed_feature_attention_mask[i, :audlen] = flat_feature_attention_mask[offset:next_offset]
                    offset = next_offset

                if offset != int(flat_input_features.shape[0]):
                    raise ValueError(
                        "Qwen3-ASR packed audio collation produced inconsistent audio grouping: "
                        f"consumed={offset}, total={int(flat_input_features.shape[0])}, "
                        f"batch_audlens={batch_audlens!r}"
                    )

                features["input_features"] = packed_input_features
                features["feature_attention_mask"] = packed_feature_attention_mask

            features["qwen3_asr_audios_per_sample"] = torch.tensor(batch_audlens, dtype=torch.long, device="cpu")
            audio_token = getattr(self.template.mm_plugin, "audio_token", None)
            if isinstance(audio_token, str) and audio_token:
                try:
                    audio_token_id = self.tokenizer.convert_tokens_to_ids(audio_token)
                    if audio_token_id is not None and torch.is_tensor(features.get("input_ids")):
                        counts = (features["input_ids"] == int(audio_token_id)).sum(dim=-1).to(dtype=torch.long, device="cpu")
                        features["qwen3_asr_audio_token_count"] = counts
                except Exception:  # noqa: BLE001
                    pass

        features.update(perf_batch)
        return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for 4d attention mask."""

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        for key, value in features.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)
            elif isinstance(value, list):
                features[key] = [
                    v.to(self.compute_dtype) if torch.is_tensor(v) and torch.is_floating_point(v) else v for v in value
                ]

        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for pairwise data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        r"""Pad batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                    "audios": feature["audios"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for KTO data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "cross_attention_mask" in kl_batch:  # for mllama inputs
            batch["kl_cross_attention_mask"] = kl_batch["cross_attention_mask"]

        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch
