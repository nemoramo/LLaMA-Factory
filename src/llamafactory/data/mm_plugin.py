# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/processing_llava.py
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

import inspect
import importlib
import json
import math
import os
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Optional, Sequence, TypeGuard, TypedDict, Union, cast
from urllib.parse import urlparse

import numpy as np
import torch
from transformers.image_utils import get_image_size, is_valid_image, to_numpy_array
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)
from typing_extensions import NotRequired, override

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras import logging
from ..extras.misc import is_env_enabled
from ..extras.packages import is_pillow_available, is_pyav_available, is_transformers_version_greater_than
from ..extras.storage_uri import maybe_map_mount_to_tos_uri


# Optional S3/TOS support. Only used when audio paths start with s3:// or tos:// (or when mapping fuse mounts).
try:
    import boto3  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    boto3 = None

try:
    from botocore.config import Config as _BotoConfig  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    _BotoConfig = None

# Optional pydub support. Preferred backend for audio loading when available.
try:
    from pydub import AudioSegment  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    AudioSegment = None

# Optional pydub mediainfo support. Uses ffprobe to fetch metadata without decoding.
try:
    from pydub.utils import mediainfo as pydub_mediainfo  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    pydub_mediainfo = None

# Optional librosa support. Fallback backend for audio loading when pydub is unavailable.
try:
    import librosa  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    librosa = None

# Optional audioread support. Helps load non-soundfile formats (e.g. mp3) without
# relying on librosa's deprecated internal audioread loader.
try:
    import audioread  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    audioread = None

# Optional soundfile support. Fast path for wav/flac/ogg-style local audio.
try:
    import soundfile as soundfile  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    soundfile = None


_S3_CLIENT_CACHE: tuple[int, Any] | None = None
_TOS_CLIENT_CACHE: tuple[int, Any] | None = None


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        v = os.environ.get(name)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def _parse_int_env(
    name: str,
    *,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw = os.environ.get(str(name))
    if raw is None:
        v = int(default)
    else:
        s = str(raw).strip()
        if s == "":
            v = int(default)
        else:
            try:
                v = int(s)
            except Exception:
                logger.warning_rank0("Invalid %s=%r (expected int); using default=%d.", str(name), raw, int(default))
                v = int(default)

    if min_value is not None and v < int(min_value):
        logger.warning_rank0("%s=%d is too small; clamping to %d.", str(name), int(v), int(min_value))
        v = int(min_value)
    if max_value is not None and v > int(max_value):
        logger.warning_rank0("%s=%d is too large; clamping to %d.", str(name), int(v), int(max_value))
        v = int(max_value)
    return int(v)


def _ensure_http_scheme(endpoint: str) -> str:
    ep = (endpoint or "").strip()
    if ep.startswith("http://") or ep.startswith("https://"):
        return ep
    return f"https://{ep}"


def _get_s3_client():
    global _S3_CLIENT_CACHE
    if boto3 is None:
        raise ImportError(
            "Loading audio from S3/TOS requires `boto3`. Please install it in your environment, e.g. `pip install boto3`."
        )
    pid = os.getpid()
    if _S3_CLIENT_CACHE is not None and _S3_CLIENT_CACHE[0] == pid:
        return _S3_CLIENT_CACHE[1]

    cfg = None
    if _BotoConfig is not None:
        # Avoid a tiny default pool when many dataloader workers fetch in parallel.
        max_conn = _parse_int_env(
            "LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS",
            default=64,
            min_value=8,
            max_value=1024,
        )
        cfg = _BotoConfig(max_pool_connections=int(max_conn))

    client = boto3.client("s3", config=cfg) if cfg is not None else boto3.client("s3")
    _S3_CLIENT_CACHE = (pid, client)
    return client


def _get_tos_client():
    global _TOS_CLIENT_CACHE
    if boto3 is None:
        raise ImportError(
            "Loading audio from S3/TOS requires `boto3`. Please install it in your environment, e.g. `pip install boto3`."
        )
    pid = os.getpid()
    if _TOS_CLIENT_CACHE is not None and _TOS_CLIENT_CACHE[0] == pid:
        return _TOS_CLIENT_CACHE[1]

    ak = _env_first("TOS_ACCESS_KEY_ID", "TOS_AK", "AWS_ACCESS_KEY_ID")
    sk = _env_first("TOS_SECRET_ACCESS_KEY", "TOS_SK", "AWS_SECRET_ACCESS_KEY")
    token = _env_first("TOS_SESSION_TOKEN", "AWS_SESSION_TOKEN")
    endpoint = _env_first("TOS_ENDPOINT", "TOS_ENDPOINT_URL")
    region = _env_first("TOS_REGION", "AWS_DEFAULT_REGION")
    addressing_style = (_env_first("TOS_ADDRESSING_STYLE", default="virtual") or "virtual").strip().lower()
    if addressing_style not in ("virtual", "path"):
        addressing_style = "virtual"

    if not ak or not sk or not endpoint or not region:
        raise RuntimeError(
            "TOS access requires env vars: TOS_ACCESS_KEY_ID/TOS_SECRET_ACCESS_KEY/TOS_ENDPOINT/TOS_REGION "
            "(or their AWS_* fallbacks)."
        )

    endpoint = _ensure_http_scheme(endpoint)

    cfg = None
    if _BotoConfig is not None:
        max_conn = _parse_int_env(
            "LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS",
            default=64,
            min_value=8,
            max_value=1024,
        )
        cfg = _BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": addressing_style},
            max_pool_connections=int(max_conn),
        )

    sess = boto3.session.Session(
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        aws_session_token=token,
        region_name=region,
    )
    client = sess.client("s3", endpoint_url=endpoint, config=cfg) if cfg is not None else sess.client("s3", endpoint_url=endpoint)
    _TOS_CLIENT_CACHE = (pid, client)
    return client


class _MissingPILImage:
    pass


Image: Any | None = None
ImageObject = _MissingPILImage
if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject

av = None
if is_pyav_available():
    import av


from transformers.image_utils import make_flat_list_of_images


def _load_make_batched_videos():
    for module_name in ("transformers.video_utils", "transformers.image_utils"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        func = getattr(module, "make_batched_videos", None)
        if callable(func):
            return func

    raise ImportError("Cannot find `make_batched_videos` in transformers.video_utils or transformers.image_utils.")


make_batched_videos = _load_make_batched_videos()


if TYPE_CHECKING:
    from av.stream import Stream
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.image_processing_utils import BaseImageProcessor
    from transformers.video_processing_utils import BaseVideoProcessor

    class EncodedImage(TypedDict):
        path: str | None
        bytes: bytes | None

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, Any]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]
    AudioInput = Union[str, BinaryIO, NDArray, dict[str, Any]]

    class RegularizedImageOutput(TypedDict):
        images: list[Any]

    class RegularizedVideoOutput(TypedDict):
        videos: list[list[Any]]
        durations: list[float]
        fps_per_video: NotRequired[list[float]]

    class RegularizedAudioOutput(TypedDict):
        audios: list[NDArray]
        sampling_rates: list[float]
        load_fail_mask: NotRequired[list[bool]]

    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

        def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
            ...


logger = logging.get_logger(__name__)


def _get_paligemma_token_type_ids(imglens: list[int], seqlens: list[int], processor: MMProcessor) -> list[list[int]]:
    r"""Get paligemma token type ids for computing loss.

    It is slightly different with the original token type ids where the prompt part is 0.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * processor.image_seq_length
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


def _get_gemma3_token_type_ids(batch_ids: list[list[int]], processor: MMProcessor):
    r"""Get gemma3 token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    image_token_id: int = getattr(processor, "image_token_id")
    batch_token_type_ids = []
    for token_ids in batch_ids:
        token_ids = np.array(token_ids)
        token_type_ids = np.zeros_like(token_ids)
        token_type_ids[token_ids == image_token_id] = 1
        batch_token_type_ids.append(token_type_ids.tolist())

    return batch_token_type_ids


def _make_batched_images(images: Sequence[Any], imglens: list[int]) -> list[list[Any]]:
    r"""Make nested list of images."""
    remaining_images = list(images)
    batch_images = []
    for imglen in imglens:
        batch_images.append(remaining_images[:imglen])
        remaining_images = remaining_images[imglen:]

    return batch_images


def _check_video_is_nested_images(video: VideoInput) -> bool:
    r"""Check if the video is nested images."""
    return isinstance(video, list) and all(
        _is_path_like(frame) or isinstance(frame, (bytes, dict, ImageObject)) or _is_file_like(frame) for frame in video
    )


def _is_file_like(obj: object) -> TypeGuard[BinaryIO]:
    return callable(getattr(obj, "read", None))


def _is_path_like(obj: object) -> TypeGuard[str | os.PathLike[str]]:
    return isinstance(obj, (str, os.PathLike))


def _seek_to_start(obj: object) -> None:
    seek = getattr(obj, "seek", None)
    if not callable(seek):
        return

    try:
        seek(0, 0)
    except TypeError:
        try:
            seek(0)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def _require_pillow() -> None:
    if Image is None:
        raise ImportError("Image processing requires Pillow. Please install it in your environment.")


def _require_pyav() -> None:
    if av is None:
        raise ImportError("Video processing requires PyAV. Please install it in your environment.")


def _decode_video_frame(frame: object) -> Any:
    to_image = getattr(frame, "to_image", None)
    if not callable(to_image):
        raise TypeError(f"Invalid video frame type: {type(frame)}")

    return to_image()


@dataclass
class MMPluginMixin:
    image_token: str | None
    video_token: str | None
    audio_token: str | None
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor: MMProcessor | None,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(
            processor, "video_processor", getattr(processor, "image_processor", None)
        )
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)

        # Check if processor is actually needed based on inputs and model capabilities.
        # This allows models with multimodal capabilities to run in text-only mode
        # when no actual multimodal inputs are provided.
        requires_image_processor = len(images) != 0 and self.image_token is not None
        requires_video_processor = len(videos) != 0 and self.video_token is not None
        requires_audio_processor = len(audios) != 0 and self.audio_token is not None
        requires_processor = requires_image_processor or requires_video_processor or requires_audio_processor

        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )

        if requires_processor and processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        if requires_image_processor and image_processor is None:
            raise ValueError("Image processor was not found, please check and update your model file.")

        if requires_video_processor and video_processor is None:
            raise ValueError("Video processor was not found, please check and update your model file.")

        if requires_audio_processor and feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        for message in messages:
            num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
            num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
            num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
            )

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )

    def _preprocess_image(
        self, image: Any, image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> Any:
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_indices(
        self, video_stream: Stream, video_fps: float, video_maxlen: int, **kwargs
    ) -> list[int]:
        r"""Compute video sample indices according to fps."""
        total_frames = video_stream.frames
        if total_frames == 0:  # infinite video
            return [int(idx) for idx in np.asarray(np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)).reshape(-1)]

        duration = video_stream.duration
        time_base = video_stream.time_base
        if duration is None or time_base is None:
            sample_frames = min(total_frames, video_maxlen)
        else:
            sample_frames = max(1, math.floor(float(duration * time_base) * video_fps))

        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return [int(idx) for idx in np.asarray(np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)).reshape(-1)]

    def _regularize_images(self, images: Sequence[Any], **kwargs) -> RegularizedImageOutput:
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        _require_pillow()
        assert Image is not None
        results: list[Any] = []
        for image in images:
            if _is_path_like(image):
                image = Image.open(image)
            elif _is_file_like(image):
                _seek_to_start(image)
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image_path = image["path"]
                    if image_path is None:
                        raise ValueError("Encoded image input must contain either `bytes` or `path`.")
                    image = Image.open(image_path)

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

    def _regularize_videos(self, videos: list[VideoInput], **kwargs) -> RegularizedVideoOutput:
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results: list[list[Any]] = []
        durations: list[float] = []
        for video in videos:
            frames: list[Any] = []
            if _check_video_is_nested_images(video):
                assert isinstance(video, list)
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not (
                        (_is_path_like(frame) and os.path.exists(frame)) or _is_file_like(frame)
                    ):
                        raise ValueError("Invalid image found in video frames.")
                frame_inputs = cast(list[Any], video)
                frames = self._regularize_images(frame_inputs, **kwargs)["images"]
                durations.append(len(frames) / kwargs.get("video_fps", 2.0))
            else:
                _require_pyav()
                assert av is not None
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(_decode_video_frame(frame))

                if video_stream.duration is None:
                    durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                else:
                    time_base = video_stream.time_base
                    if time_base is None:
                        durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                    else:
                        durations.append(float(video_stream.duration * time_base))

                frames = self._regularize_images(frames, **kwargs)["images"]

            results.append(frames)

        return {"videos": results, "durations": durations}

    def _load_audio_with_pydub(
        self,
        src: str | BinaryIO | BytesIO,
        sampling_rate: float,
    ) -> tuple[NDArray, float]:
        r"""Load audio with pydub + ffmpeg and return mono float32 waveform."""
        if AudioSegment is None:
            raise ImportError(
                "Loading audio requires `pydub`. Please install it in your environment, e.g. `pip install pydub`."
            )

        if isinstance(src, BytesIO) or _is_file_like(src):
            _seek_to_start(src)

        segment = AudioSegment.from_file(src)

        target_sr = int(sampling_rate) if sampling_rate is not None else segment.frame_rate
        if segment.frame_rate != target_sr:
            segment = segment.set_frame_rate(target_sr)

        samples = np.array(segment.get_array_of_samples())
        if segment.channels > 1:
            samples = samples.reshape(-1, segment.channels).mean(axis=1)

        sample_width = max(int(segment.sample_width), 1)
        max_val = float(1 << (8 * sample_width - 1))
        if max_val <= 0:
            max_val = 1.0

        waveform = (samples.astype(np.float32) / max_val).astype(np.float32)
        return waveform, float(segment.frame_rate)

    def _load_audio_with_audioread(
        self,
        path: str,
        sampling_rate: float,
    ) -> tuple[NDArray, float]:
        r"""Load audio with audioread and return mono float32 waveform."""
        if audioread is None:
            raise ImportError(
                "Loading audio requires `audioread`. Please install it in your environment, e.g. `pip install audioread`."
            )

        with audioread.audio_open(path) as f:
            sr = int(getattr(f, "samplerate", 0) or 0)
            channels = int(getattr(f, "channels", 0) or 0)
            if sr <= 0 or channels <= 0:
                raise ValueError(f"Invalid audio metadata from audioread: sr={sr} channels={channels}")

            raw = bytearray()
            for buf in f:
                raw.extend(buf)

        if len(raw) == 0:
            raise ValueError("Empty audio data from audioread.")

        samples = np.frombuffer(raw, dtype=np.dtype("<i2"))
        if samples.size == 0:
            raise ValueError("Empty decoded samples from audioread.")

        if channels > 1:
            usable = samples.size - (samples.size % channels)
            if usable <= 0:
                raise ValueError(f"Invalid decoded audio length: n={samples.size} channels={channels}")
            samples = samples[:usable].reshape(-1, channels).mean(axis=1)

        waveform = (samples.astype(np.float32) / 32768.0).astype(np.float32)

        target_sr = int(sampling_rate) if sampling_rate is not None else sr
        if sr != target_sr:
            if librosa is None:
                raise ImportError("Resampling requires `librosa`.")
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr).astype(np.float32)
            sr = target_sr

        return waveform, float(sr)

    def _load_single_audio(
        self,
        audio: AudioInput,
        sampling_rate: float,
    ) -> tuple[NDArray, float]:
        """Normalize a single audio input to (np.ndarray, sr).

        Supports numpy arrays, file-like objects, local paths and s3:// / tos:// URIs.
        """
        # Dict wrapper support (used by optional audio SpecAugment meta preservation).
        if isinstance(audio, dict):
            arr = audio.get("array")
            if isinstance(arr, np.ndarray):
                return arr, sampling_rate

            raw_bytes = audio.get("bytes")
            if isinstance(raw_bytes, (bytes, bytearray)):
                audio = BytesIO(bytes(raw_bytes))
            else:
                path = audio.get("path") or audio.get("wav_path") or audio.get("audio_path") or audio.get("uri")
                if isinstance(path, (str, os.PathLike)) and os.fspath(path):
                    audio = os.fsdecode(path)
                else:
                    raw = audio.get("raw")
                    if raw is not None:
                        audio = raw

        # Already decoded array
        if isinstance(audio, np.ndarray):
            return audio, sampling_rate

        # Binary/file-like
        if isinstance(audio, BytesIO) or _is_file_like(audio):
            _seek_to_start(audio)

            if AudioSegment is not None:
                try:
                    return self._load_audio_with_pydub(audio, sampling_rate)
                except Exception:  # noqa: BLE001
                    _seek_to_start(audio)

            if librosa is not None:
                y, sr = librosa.load(audio, sr=sampling_rate)
                return y, sr

            raise ImportError(
                "Neither `pydub` nor `librosa` is available for loading audio. "
                "Please install at least one of them, e.g. `pip install pydub`."
            )

        # String or os.PathLike path / URI
        if isinstance(audio, (str, os.PathLike)):
            path = os.fsdecode(audio)
            mapped = maybe_map_mount_to_tos_uri(path)
            if mapped is not None:
                path = mapped

            # S3/TOS URI: s3://bucket/key or tos://bucket/key
            if path.startswith(("s3://", "tos://")):
                parsed = urlparse(path)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")

                s3_client = _get_tos_client() if path.startswith("tos://") else _get_s3_client()
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                data = obj["Body"].read()
                bio = BytesIO(data)

                if AudioSegment is not None:
                    try:
                        return self._load_audio_with_pydub(bio, sampling_rate)
                    except Exception:  # noqa: BLE001
                        _seek_to_start(bio)

                if librosa is not None:
                    y, sr = librosa.load(bio, sr=sampling_rate)
                    return y, sr

                raise ImportError(
                    "Neither `pydub` nor `librosa` is available for loading audio from S3. "
                    "Please install at least one of them, e.g. `pip install pydub`."
                )

            # Local path – prefer soundfile (fast for wav/flac/ogg), then pydub, then audioread, then librosa.
            if soundfile is not None:
                try:
                    data, sr = soundfile.read(path, dtype="float32", always_2d=True)
                    if not isinstance(sr, (int, float)) or sr <= 0:
                        raise ValueError(f"Invalid sampling rate from soundfile: {sr!r}")
                    num_channels = int(data.shape[-1]) if data.ndim > 1 else 1
                    waveform = data.mean(axis=1).astype(np.float32) if num_channels > 1 else data[:, 0].astype(np.float32)
                    target_sr = int(sampling_rate) if sampling_rate is not None else int(sr)
                    if int(sr) != int(target_sr):
                        if librosa is None:
                            raise ImportError("Resampling requires `librosa`.")
                        waveform = librosa.resample(waveform, orig_sr=int(sr), target_sr=int(target_sr)).astype(
                            np.float32
                        )
                        sr = target_sr
                    return waveform, float(sr)
                except Exception:  # noqa: BLE001
                    pass

            if AudioSegment is not None:
                try:
                    return self._load_audio_with_pydub(path, sampling_rate)
                except Exception:  # noqa: BLE001
                    pass

            if audioread is not None:
                try:
                    return self._load_audio_with_audioread(path, sampling_rate)
                except Exception:  # noqa: BLE001
                    pass

            if librosa is not None:
                y, sr = librosa.load(path, sr=sampling_rate)
                return y, sr

            raise ImportError(
                "Neither `pydub` nor `librosa` is available for loading local audio files. "
                "Please install at least one of them, e.g. `pip install pydub`."
            )

        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _regularize_audios(
        self,
        audios: list[AudioInput],
        sampling_rate: float,
        **kwargs,
    ) -> RegularizedAudioOutput:
        r"""Regularizes audios to avoid error.

        Including reading and resampling.
        """
        results: list[NDArray] = []
        sampling_rates: list[float] = []
        load_fail_mask: list[bool] = []

        max_retries = int(os.getenv("LLAMAFACTORY_AUDIO_LOAD_RETRIES", "1"))
        retry_sleep_sec = float(os.getenv("LLAMAFACTORY_AUDIO_LOAD_RETRY_SLEEP", "0.2"))
        log_limit = int(os.getenv("LLAMAFACTORY_AUDIO_LOAD_ERROR_LOG_LIMIT", "20"))
        logged = int(getattr(self, "_mm_audio_load_error_logged", 0))
        suppressed = bool(getattr(self, "_mm_audio_load_error_suppressed", False))

        for audio in audios:
            last_error: Exception | None = None
            y: NDArray | None = None
            sr: float | None = None
            for attempt in range(max(0, max_retries) + 1):
                try:
                    y, sr = self._load_single_audio(audio, sampling_rate)
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

            if last_error is not None or y is None or sr is None:
                # Use a dummy waveform so feature extraction can proceed and the collator can
                # mask labels for the affected segments via `feature_load_fail_mask`.
                results.append(np.zeros(0, dtype=np.float32))
                sampling_rates.append(float(sampling_rate))
                load_fail_mask.append(True)
                if logged < log_limit:
                    logger.warning_rank0(
                        "Audio load error; using dummy waveform (audio=%r): %s",
                        audio,
                        repr(last_error),
                    )
                    logged += 1
                elif not suppressed:
                    logger.warning_rank0(
                        "Too many audio load errors (%d+); suppressing further logs.",
                        log_limit,
                    )
                    suppressed = True
                continue

            results.append(y)
            sampling_rates.append(float(sr))
            load_fail_mask.append(False)

        setattr(self, "_mm_audio_load_error_logged", logged)
        setattr(self, "_mm_audio_load_error_suppressed", suppressed)

        return {"audios": results, "sampling_rates": sampling_rates, "load_fail_mask": load_fail_mask}

    def _get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor,
        imglens: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""Process visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                            where num_patches == torch.prod(image_grid_thw)

        Returns: (mllama)
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

        """
        mm_inputs = {}
        if len(images) != 0:
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            image_processor_kwargs = {}
            if getattr(processor, "image_do_pan_and_scan", False):  # gemma3 image processor
                image_processor_kwargs.update(
                    {
                        "do_pan_and_scan": True,
                        "pan_and_scan_min_crop_size": 256,
                        "pan_and_scan_max_num_crops": 4,
                        "pan_and_scan_min_ratio_to_activate": 1.2,
                    }
                )

            mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))

        if len(videos) != 0:
            video_processor: BaseImageProcessor = getattr(
                processor, "video_processor", getattr(processor, "image_processor", None)
            )
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pt"))

        if len(audios) != 0:
            feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
            audio_sampling_rate = getattr(processor, "audio_sampling_rate", 16000)
            audio_padding = getattr(processor, "audio_padding", "max_length")
            audio_out = self._regularize_audios(
                audios,
                sampling_rate=audio_sampling_rate,
            )
            audios = audio_out["audios"]
            feature_load_fail_mask = audio_out.get("load_fail_mask", None)
            if feature_load_fail_mask is None:
                feature_load_fail_mask = [False] * len(audios)
            elif len(feature_load_fail_mask) != len(audios):
                feature_load_fail_mask = (list(feature_load_fail_mask) + [False] * len(audios))[: len(audios)]
            min_samples = int(getattr(feature_extractor, "n_fft", 400) or 400)
            if min_samples > 0:
                audios = [np.pad(a, (0, max(0, min_samples - a.shape[0])), mode="constant") for a in audios]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=audio_sampling_rate,
                    return_attention_mask=True,
                    padding=audio_padding,
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)  # prevent conflicts
            mm_inputs["feature_load_fail_mask"] = torch.tensor(feature_load_fail_mask, dtype=torch.bool)

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: list[int] | None,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        tokenizer: PreTrainedTokenizer,
        processor: MMProcessor | None,
    ) -> tuple[list[int], list[int] | None]:
        r"""Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        r"""Build batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            audios: a list of audio inputs, shape (num_audios,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            audlens: number of audios in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos

        """
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class ErnieVLPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        image_idx, video_idx = 0, 0
        for message in messages:
            content = message["content"]
            image_token = self.image_token or "<|IMAGE_PLACEHOLDER|>"
            video_token = self.video_token or "<|VIDEO_PLACEHOLDER|>"
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[image_idx].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"Picture {image_idx + 1}:<|IMAGE_START|>{image_token * image_seqlen}<|IMAGE_END|>",
                    1,
                )
                image_idx += 1
            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[video_idx].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"Video {video_idx + 1}:<|VIDEO_START|>{video_token * video_seqlen}<|VIDEO_END|>",
                    1,
                )
                video_idx += 1
            message["content"] = content
        return messages


@dataclass
class Gemma3Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_messages(messages, images, videos, audios)

        # Detect if we have actual multimodal inputs or placeholders.
        # This allows Gemma3 to work in text-only mode when no multimodal content is present.
        has_mm_inputs = len(images) != 0 or len(videos) != 0 or len(audios) != 0
        has_mm_placeholder = any(
            (IMAGE_PLACEHOLDER in m["content"])
            or (VIDEO_PLACEHOLDER in m["content"])
            or (AUDIO_PLACEHOLDER in m["content"])
            for m in messages
        )

        if processor is None:
            # Allow text-only usage with Gemma3 templates (no multimodal processor available).
            # This is useful when training/using Gemma3 for text-only tasks.
            if not has_mm_inputs and not has_mm_placeholder:
                return messages

            raise ValueError(
                "Multimodal inputs/placeholders were found but processor is missing. "
                "Please check and update your model file."
            )

        self._validate_input(processor, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        boi_token: str = getattr(processor, "boi_token")
        full_image_sequence: str = getattr(processor, "full_image_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token

        do_pan_and_scan: bool = getattr(processor, "image_do_pan_and_scan", False)
        if do_pan_and_scan:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if do_pan_and_scan:
                    image_placeholder_str = (
                        "Here is the original image {{image}} and here are some crops to help you see better "
                        + " ".join(["{{image}}"] * mm_inputs["num_crops"][0][num_image_tokens])
                    )
                else:
                    image_placeholder_str = "{{image}}"

                content = content.replace(IMAGE_PLACEHOLDER, image_placeholder_str, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", image_str)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        # Handle text-only mode: when no multimodal processor exists and no multimodal inputs are provided,
        # return token_type_ids with all zeros for proper loss computation during training.
        # This enables Gemma3 to be used for text-only tasks without requiring a multimodal processor.
        if processor is None and len(images) == 0 and len(videos) == 0 and len(audios) == 0:
            return {"token_type_ids": [[0] * len(token_ids) for token_ids in batch_ids]}

        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("num_crops", None)
        mm_inputs["token_type_ids"] = _get_gemma3_token_type_ids(batch_ids, processor)
        return mm_inputs


class Gemma3nPlugin(Gemma3Plugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        boi_token: str = getattr(processor, "boi_token")
        boa_token: str = getattr(processor, "boa_token")
        full_image_sequence: str = getattr(processor, "full_image_sequence")
        full_audio_sequence: str = getattr(processor, "full_audio_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token
        audio_str = full_audio_sequence if self.expand_mm_tokens else boa_token

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, image_str, 1)

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, audio_str, 1)

            message["content"] = content

        return messages


@dataclass
class InternVLPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: ProcessorMixin,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        image_processor_kwargs = {}
        if getattr(processor, "crop_to_patches", False):
            image_processor_kwargs.update(
                {
                    "crop_to_patches": True,
                    "max_patches": 12,
                    "min_patches": 1,
                }
            )

        mm_inputs = {}
        image_video_patches = []

        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 1024 * 1024),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]

        if len(images) != 0:
            images = make_flat_list_of_images(images)
            image_inputs = image_processor(images=images, return_tensors="pt", **image_processor_kwargs)
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)

        if len(videos) != 0:
            videos = make_batched_videos(videos)
            num_frames_per_video = [len(video) for video in videos]
            patch_indices = np.cumsum(num_frames_per_video)
            image_processor_kwargs["crop_to_patches"] = False
            video_inputs = image_processor(images=videos, return_tensors="pt", **image_processor_kwargs)
            video_num_patches = video_inputs.pop("num_patches")
            video_pixel_values = video_inputs.pop("pixel_values")
            video_num_patches_indices = np.cumsum(video_num_patches)

        # NOT SUPPORT IMAGE VIDEO INTERLEAVED
        if len(images) != 0 and image_pixel_values is not None:
            for i in range(len(images)):
                start_index = image_num_patches_indices[i - 1] if i > 0 else 0
                end_index = image_num_patches_indices[i]
                image_video_patches.append(image_pixel_values[start_index:end_index])

        if len(videos) != 0 and video_pixel_values is not None:
            patch_indices_with_prefix = [0] + list(patch_indices)
            for i in range(len(videos)):
                current_patch_index = patch_indices_with_prefix[i]
                end_patch_index = patch_indices_with_prefix[i + 1]
                start_index = video_num_patches_indices[current_patch_index - 1] if i > 0 else 0
                end_index = video_num_patches_indices[end_patch_index - 1]
                image_video_patches.append(video_pixel_values[start_index:end_index])

        if len(images) != 0 or len(videos) != 0:
            mm_inputs["pixel_values"] = torch.cat(image_video_patches, dim=0)

        if len(images) != 0:
            mm_inputs.update({"image_num_patches": image_num_patches})

        if len(videos) != 0:
            mm_inputs.update({"video_patch_indices": patch_indices})
            mm_inputs.update({"video_num_patches": video_num_patches})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: ProcessorMixin | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        image_seqlen = getattr(processor, "image_seq_length") if self.expand_mm_tokens else 1
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        image_pixel_patch_list = mm_inputs.get("image_num_patches")  # pathes of images
        video_num_patches = mm_inputs.get("video_num_patches")  # all patches for frames of videos
        video_patch_indices = mm_inputs.get("video_patch_indices")  # num frames of per video

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<img>{'<IMG_CONTEXT>' * image_seqlen * image_pixel_patch_list[num_image_tokens]}</img>",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                current_patch_index = video_patch_indices[num_video_tokens - 1] if num_video_tokens > 0 else 0
                end_patch_index = video_patch_indices[num_video_tokens]
                num_patches = list(video_num_patches[current_patch_index:end_patch_index])
                video_replaced_prompt = "\n".join(
                    f"Frame{i + 1}: <img>{'<IMG_CONTEXT>' * image_seqlen * num_patches[i]}</img>"
                    for i in range(len(num_patches))
                )
                content = content.replace(VIDEO_PLACEHOLDER, video_replaced_prompt, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: ProcessorMixin | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("image_num_patches", None)
        mm_inputs.pop("video_patch_indices", None)
        mm_inputs.pop("video_num_patches", None)
        return mm_inputs


class KimiVLPlugin(BasePlugin):
    @override
    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_hws = mm_inputs.get("image_grid_hws", [])
        else:
            image_grid_hws = [None] * len(images)

        num_image_tokens = 0
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        merge_length = math.prod(image_processor.merge_kernel_size)
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_hws[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|media_start|>image<|media_content|>{self.image_token * image_seqlen}<|media_end|>",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Llama4Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_height, image_width = mm_inputs["pixel_values"][0].shape[-2:]
                num_patches_per_chunk = int(
                    (image_height // processor.patch_size)
                    * (image_width // processor.patch_size)
                    // processor.downsample_ratio
                )
                aspect_ratios = mm_inputs.pop("aspect_ratios")

        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            if self.expand_mm_tokens:
                placeholder_count = content.count(IMAGE_PLACEHOLDER)
                prompt_splits = content.split(IMAGE_PLACEHOLDER)
                new_content = []
                for local_image_index, split_part in enumerate(prompt_splits):
                    new_content.append(split_part)
                    if local_image_index < placeholder_count:
                        tokens_for_this_image = processor._prompt_split_image(
                            aspect_ratios[num_image_tokens], num_patches_per_chunk
                        )
                        num_image_tokens += 1
                        new_content.append(tokens_for_this_image)

                content = "".join(new_content)
            else:
                content = content.replace(IMAGE_PLACEHOLDER, self.image_token)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("aspect_ratios", None)
        return mm_inputs


@dataclass
class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen = 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if self.expand_mm_tokens:
            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim
                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
                video_seqlen = image_seqlen // 4 * num_frames  # divide by 4 needed for avg pooling layer
        else:
            video_seqlen = 1

        for message in messages:
            content = message["content"]
            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


@dataclass
class MiniCPMVPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if "valid_image_nums_ls" in kwargs:
                valid_image_nums_ls = kwargs["valid_image_nums_ls"]
                new_images = []
                idx = 0
                for valid_image_nums in valid_image_nums_ls:
                    new_images.append(images[idx : idx + valid_image_nums])
                    idx += valid_image_nums

                images = new_images

            image_inputs = image_processor(
                images, do_pad=True, max_slice_nums=image_processor.max_slice_nums, return_tensors="pt"
            )
            mm_inputs.update(image_inputs)

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            video_inputs = image_processor(videos, do_pad=True, max_slice_nums=2, return_tensors="pt")
            mm_inputs.update(video_inputs)

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            if "valid_audio_nums_ls" in kwargs:
                valid_audio_nums_ls = kwargs["valid_audio_nums_ls"]
                audios_ls = []
                idx = 0
                for valid_audio_nums in valid_audio_nums_ls:
                    audios_ls.append(audios[idx : idx + valid_audio_nums])
                    idx += valid_audio_nums
            else:
                audios_ls = [audios]

            audio_features, audio_feature_lens, audio_phs = processor.audio_feature_extract(
                audios_ls,
                chunk_input=True,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )
            audio_feature_lens = [torch.tensor(audio_feature_len) for audio_feature_len in audio_feature_lens]
            mm_inputs.update({"audio_features": audio_features, "audio_feature_lens": audio_feature_lens})
            if kwargs.get("ret_phs", False):
                mm_inputs.update({"audio_phs": audio_phs})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs, audio_inputs = {}, {}
        if len(images) != 0 and len(videos) != 0:
            raise ValueError("MiniCPM-V model does not support input images and videos at the same time.")

        if len(videos) != 0:
            max_slice_nums = 2
            use_image_id = False
            mm_inputs = self._get_mm_inputs([], videos, [], processor)
        else:
            max_slice_nums = image_processor.max_slice_nums
            use_image_id = image_processor.use_image_id

        for i, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = len(mm_inputs["pixel_values"][num_video_tokens]) if self.expand_mm_tokens else 1
                content = content.replace(VIDEO_PLACEHOLDER, "{{image}}" * video_seqlen, 1)
                num_video_tokens += 1

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, "{{audio}}", 1)
                num_audio_tokens += 1

            message["content"] = content.replace("{{image}}", "(<image>./</image>)").replace(
                "{{audio}}", "(<audio>./</audio>)"
            )

        if len(images):
            mm_inputs = self._get_mm_inputs(images, [], [], processor)

        if len(audios):
            audio_inputs = self._get_mm_inputs([], [], audios, processor, ret_phs=True)

        if self.expand_mm_tokens and mm_inputs:
            pattern = "(<image>./</image>)"
            image_sizes = mm_inputs["image_sizes"]
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                image_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(image_tags)):
                    final_text = (
                        final_text
                        + text_chunks[i]
                        + image_processor.get_slice_image_placeholder(
                            image_sizes[0][idx], idx, max_slice_nums, use_image_id
                        )
                    )
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        if self.expand_mm_tokens and audio_inputs:
            pattern = "(<audio>./</audio>)"
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                audio_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(audio_tags)):
                    audio_placeholder = audio_inputs["audio_phs"][0][idx]
                    final_text = final_text + text_chunks[i] + audio_placeholder
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        # image bound
        image_bounds_list = []
        valid_image_nums_ls = []
        for i, input_ids in enumerate(batch_ids):
            input_ids_ = torch.tensor(input_ids)
            start_cond = (input_ids_ == processor.tokenizer.im_start_id) | (
                input_ids_ == processor.tokenizer.slice_start_id
            )
            end_cond = (input_ids_ == processor.tokenizer.im_end_id) | (input_ids_ == processor.tokenizer.slice_end_id)
            image_start_tokens = torch.where(start_cond)[0]
            image_start_tokens += 1
            image_end_tokens = torch.where(end_cond)[0]
            valid_image_nums_ls.append(imglens[i])
            image_bounds = torch.hstack(
                [
                    image_start_tokens.unsqueeze(-1),
                    image_end_tokens.unsqueeze(-1),
                ]
            )
            image_bounds_list.append(image_bounds)

        mm_inputs = self._get_mm_inputs(images, videos, [], processor, valid_image_nums_ls=valid_image_nums_ls)
        if "tgt_sizes" not in mm_inputs:
            dummy_data = [torch.empty(0) for _ in range(len(batch_ids))]
            mm_inputs.update({"tgt_sizes": dummy_data, "pixel_values": dummy_data, "image_sizes": dummy_data})

        mm_inputs.update({"image_bound": image_bounds_list})

        if len(audios) > 0:
            # audio bound
            audio_bounds_ls = []
            spk_bounds_ls = []
            valid_audio_nums_ls = []

            for input_ids, audiolen in zip(batch_ids, audlens):
                input_ids_ = torch.tensor(input_ids)
                audio_start_idx = torch.where(input_ids_ == processor.tokenizer.audio_start_id)[0]
                audio_end_idx = torch.where(input_ids_ == processor.tokenizer.audio_end_id)[0]
                assert len(audio_start_idx) == len(audio_end_idx)
                audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])
                audio_bounds_ls.append(audio_bounds)
                valid_audio_nums_ls.append(audiolen)

                spk_start_idx = torch.where(input_ids_ == processor.tokenizer.spk_start_id)[0]
                spk_end_idx = torch.where(input_ids_ == processor.tokenizer.spk_end_id)[0]
                assert len(spk_start_idx) == len(spk_end_idx)
                spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
                spk_bounds_ls.append(spk_bounds)

            audio_inputs = self._get_mm_inputs([], [], audios, processor, valid_audio_nums_ls=valid_audio_nums_ls)
            mm_inputs.update(audio_inputs)
            mm_inputs.update({"audio_bounds": audio_bounds_ls, "spk_bounds": spk_bounds_ls})

        return mm_inputs


@dataclass
class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor, imglens)
        if mm_inputs:
            num_tiles = mm_inputs.pop("num_tiles")
            image_token_id: int = getattr(processor, "image_token_id")
            max_image_tiles: int = getattr(processor.image_processor, "max_image_tiles")
            cross_attention_token_mask = [
                get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
            ]
            mm_inputs["cross_attention_mask"] = torch.from_numpy(
                convert_sparse_cross_attention_mask_to_dense(
                    cross_attention_token_mask,
                    num_tiles=num_tiles,
                    max_num_tiles=max_image_tiles,
                    length=max(len(input_ids) for input_ids in batch_ids),
                )
            )  # shape: (batch_size, length, max_num_images, max_num_tiles)

        return mm_inputs


@dataclass
class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "", 1)
                num_image_tokens += 1

            message["content"] = content

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: list[int],
        labels: list[int] | None,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        tokenizer: PreTrainedTokenizer,
        processor: MMProcessor | None,
    ) -> tuple[list[int], list[int] | None]:
        self._validate_input(processor, images, videos, audios)
        num_images = len(images)
        image_seqlen = processor.image_seq_length if self.expand_mm_tokens else 0  # skip mm token
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * num_images * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * num_images * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        seqlens = [len(input_ids) for input_ids in batch_ids]
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


@dataclass
class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                # BC for transformers < 4.49.0
                if isinstance(mm_inputs["image_sizes"], list):
                    image_sizes = iter(mm_inputs["image_sizes"][0])
                else:
                    image_sizes = iter(mm_inputs["image_sizes"].tolist())

                image_break_token: str = getattr(processor, "image_break_token")
                image_end_token: str = getattr(processor, "image_end_token")

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    patch_size = processor.patch_size * getattr(processor, "spatial_merge_size", 1)
                    height, width = next(image_sizes)
                    num_height_tokens = height // patch_size
                    num_width_tokens = width // patch_size
                    replace_tokens = [[self.image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]  # flatten list
                    replace_tokens[-1] = image_end_token
                    replace_str = "".join(replace_tokens)
                else:
                    replace_str = self.image_token

                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        # ref to this commit https://github.com/huggingface/transformers/pull/35122
        # after transformers 4.49.0, the `image_sizes` is mandatory as an input parameter for Pixtral VisionEncoder forwarding.
        # it can be passed into `LlavaConditionalGeneration` as a parameter.
        if not is_transformers_version_greater_than("4.49.0"):
            mm_inputs.pop("image_sizes", None)
        return mm_inputs


@dataclass
class Qwen2AudioPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        bos_token: str = getattr(processor, "audio_bos_token")
        eos_token: str = getattr(processor, "audio_eos_token")
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs([], [], audios, processor)
            if "feature_attention_mask" in mm_inputs:
                audio_lengths = mm_inputs["feature_attention_mask"].sum(-1).tolist()

        for message in messages:
            content = message["content"]
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    audio_seqlen = (input_length - 2) // 2 + 1
                else:
                    audio_seqlen = 1

                content = content.replace(
                    AUDIO_PLACEHOLDER, f"{bos_token}{self.audio_token * audio_seqlen}{eos_token}", 1
                )

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)

@dataclass
class Qwen3ASRPlugin(BasePlugin):
    """Multimodal plugin for Qwen3-ASR.

    Qwen3-ASR expands each `<audio>` placeholder into:
      <audio_bos_token> + <audio_token> * N + <audio_eos_token>

    where `N` is computed from `feature_attention_mask` via the same length rule used by
    `qwen_asr.core.transformers_backend.processing_qwen3_asr._get_feat_extract_output_lengths`.
    """

    @staticmethod
    def _get_audio_token_length(feature_length: int) -> int:
        # Mirrors Qwen3ASRProcessor's `_get_feat_extract_output_lengths`.
        input_lengths_leave = feature_length % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_length // 100) * 13
        return int(output_lengths)

    def _try_get_feature_length(
        self,
        audio: AudioInput,
        sampling_rate: float,
        hop_length: int | None,
        min_samples: int | None = None,
    ) -> int | None:
        if hop_length is None or hop_length <= 0:
            return None

        if isinstance(audio, np.ndarray):
            n_samples = int(audio.shape[0])
            if isinstance(min_samples, int) and min_samples > 0:
                n_samples = max(n_samples, int(min_samples))
            return max(1, n_samples // int(hop_length))

        if isinstance(audio, str):
            obj = None
            audio_str = audio.strip()
            if audio_str.startswith("{") and audio_str.endswith("}"):
                try:
                    obj = json.loads(audio_str)
                    if not isinstance(obj, dict):
                        obj = None
                except Exception:  # noqa: BLE001
                    obj = None

            path = None
            duration_sec = None
            has_offset = False
            if obj is not None:
                path = obj.get("path") or obj.get("wav_path") or obj.get("audio_path")
                has_offset = any(
                    k in obj
                    for k in (
                        "offset_sec",
                        "offset_secs",
                        "offset_seconds",
                        "offset",
                        "start_sec",
                        "start_secs",
                        "start_seconds",
                        "start_time",
                        "start",
                    )
                )

                for k in ("duration_sec", "duration_secs", "duration_seconds", "duration"):
                    if k not in obj:
                        continue
                    try:
                        d = float(obj.get(k))
                        if math.isfinite(d) and d >= 0:
                            duration_sec = d
                            break
                    except Exception:  # noqa: BLE001
                        continue

                if duration_sec is None:
                    for k in ("duration_ms", "duration_msec"):
                        if k not in obj:
                            continue
                        try:
                            d_ms = float(obj.get(k))
                            d = d_ms / 1000.0
                            if math.isfinite(d) and d >= 0:
                                duration_sec = d
                                break
                        except Exception:  # noqa: BLE001
                            continue
            else:
                path = audio

            if duration_sec is not None:
                n_samples = int(round(float(duration_sec) * float(sampling_rate)))
                if isinstance(min_samples, int) and min_samples > 0:
                    n_samples = max(n_samples, int(min_samples))
                return max(1, n_samples // int(hop_length))

            # Segment-based audio without explicit duration: do not probe file-level metadata.
            if has_offset:
                return None

            if not isinstance(path, str) or path == "":
                return None
            if path.startswith("file://"):
                path = path[7:]
            if path.startswith("s3://"):
                return None

            # Prefer header probing to avoid feature extraction.
            if soundfile is not None:
                try:
                    info = soundfile.info(path)
                    frames = getattr(info, "frames", None)
                    sr = getattr(info, "samplerate", None)
                    if isinstance(frames, int) and isinstance(sr, int) and frames >= 0 and sr > 0:
                        n_samples = int(round(float(frames) * float(sampling_rate) / float(sr)))
                        if isinstance(min_samples, int) and min_samples > 0:
                            n_samples = max(n_samples, int(min_samples))
                        return max(1, n_samples // int(hop_length))
                except Exception:  # noqa: BLE001
                    pass

            if pydub_mediainfo is not None:
                try:
                    meta = pydub_mediainfo(path)
                    if isinstance(meta, dict):
                        dur = meta.get("duration")
                        if dur is not None:
                            d = float(dur)
                            if math.isfinite(d) and d >= 0:
                                n_samples = int(round(d * float(sampling_rate)))
                                if isinstance(min_samples, int) and min_samples > 0:
                                    n_samples = max(n_samples, int(min_samples))
                                return max(1, n_samples // int(hop_length))
                except Exception:  # noqa: BLE001
                    pass

            return None

        return None

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        bos_token: str = getattr(processor, "audio_bos_token")
        eos_token: str = getattr(processor, "audio_eos_token")
        messages = deepcopy(messages)
        audio_lengths: list[int] = []
        if self.expand_mm_tokens:
            feature_extractor = getattr(processor, "feature_extractor", None)
            sampling_rate = float(getattr(processor, "audio_sampling_rate", 16000))
            hop_length = getattr(feature_extractor, "hop_length", None)
            hop_length = int(hop_length) if isinstance(hop_length, int) and hop_length > 0 else None
            min_samples = int(getattr(feature_extractor, "n_fft", 400) or 400)

            unknown_audios: list[AudioInput] = []
            unknown_positions: list[int] = []
            for idx, audio in enumerate(audios):
                feature_length = self._try_get_feature_length(
                    audio, sampling_rate, hop_length, min_samples=min_samples
                )
                if feature_length is None:
                    audio_lengths.append(-1)
                    unknown_audios.append(audio)
                    unknown_positions.append(idx)
                else:
                    audio_lengths.append(int(feature_length))

            # Fallback to full feature extraction only for audios we cannot probe cheaply.
            if unknown_audios:
                mm_inputs = self._get_mm_inputs([], [], unknown_audios, processor)
                if "feature_attention_mask" in mm_inputs:
                    fallback_lengths = mm_inputs["feature_attention_mask"].sum(-1).tolist()
                    for idx, feature_length in zip(unknown_positions, fallback_lengths):
                        audio_lengths[idx] = int(feature_length)

        for message in messages:
            content = message["content"]
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens and audio_lengths:
                    feature_length = int(audio_lengths.pop(0))
                    if feature_length > 0:
                        audio_seqlen = max(1, self._get_audio_token_length(feature_length))
                    else:
                        audio_seqlen = 1
                else:
                    audio_seqlen = 1

                content = content.replace(
                    AUDIO_PLACEHOLDER,
                    f"{bos_token}{self.audio_token * audio_seqlen}{eos_token}",
                    1,
                )

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class VoxtralPlugin(BasePlugin):
    r"""Voxtral multimodal plugin.

    Voxtral expects `input_features` to be stacked along the batch dimension by 30-second chunks.
    We therefore cannot batch feature extraction across audios with padding-to-longest (it would over-pad
    shorter audios and change the number of chunks). Instead, we extract features per audio and then
    concatenate chunks in-order.
    """

    max_source_positions: int = 3000  # Whisper mel frames per 30s.
    pad_to_multiple_of: int = 480000  # 30s at 16kHz.

    def _validate_input(  # type: ignore[override]
        self,
        processor: MMProcessor | None,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
    ) -> None:
        # Voxtral always supports audio; do not gate on `audio_token` presence.
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )
        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0:
            if processor is None:
                raise ValueError("Processor was not found, please check and update your model file.")
            if getattr(processor, "feature_extractor", None) is None:
                raise ValueError("Audio feature extractor was not found, please check and update your model file.")

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        if processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")
        if len(audios) == 0:
            return {}

        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        if feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")

        audio_sampling_rate = int(getattr(processor, "audio_sampling_rate", 16000))
        min_samples = int(getattr(feature_extractor, "n_fft", 400) or 400)

        input_features_list: list[torch.Tensor] = []
        for audio in audios:
            # Normalize `file://` URIs to local paths.
            if isinstance(audio, str) and audio.startswith("file://"):
                audio = audio[7:]

            wav, _ = self._load_single_audio(audio, float(audio_sampling_rate))
            if min_samples > 0:
                wav = np.pad(wav, (0, max(0, min_samples - wav.shape[0])), mode="constant")

            wav_inputs = feature_extractor(
                wav,
                sampling_rate=audio_sampling_rate,
                padding=True,
                truncation=False,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=False,
                return_tensors="pt",
            )
            feats: torch.Tensor = wav_inputs["input_features"]  # (1, 128, T)
            if feats.ndim != 3:
                raise ValueError(f"Unexpected Voxtral input_features shape: {tuple(feats.shape)}")

            # Safety: pad mel frames to a multiple of `max_source_positions` before chunking.
            if feats.shape[-1] % self.max_source_positions != 0:
                pad_len = self.max_source_positions - (feats.shape[-1] % self.max_source_positions)
                feats = torch.nn.functional.pad(feats, (0, pad_len))

            chunked = feats[0].reshape(feats.shape[1], -1, self.max_source_positions).transpose(0, 1)
            input_features_list.append(chunked)

        if len(input_features_list) == 0:
            return {}

        return {"input_features": torch.cat(input_features_list, dim=0)}


@dataclass
class FunAudioChatPlugin(BasePlugin):
    r"""FunAudioChat multimodal plugin (S2T-friendly).

    - Expands each audio placeholder into a variable number of `<|AUDIO|>` tokens according to speech token length.
    - Builds `speech_ids/speech_attention_mask` (discrete tokens) and `input_features/feature_attention_mask`
      (continuous waveform features) for FunAudioChatForConditionalGeneration.
    """

    token_fps: int = 25  # FunAudioChat uses 25Hz discrete frames.
    _segment_duration_re = re.compile(r"_seg\d+_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\.wav$")

    _duration_cache_max_size: int = 20000

    def _get_duration_cache(self) -> tuple[dict[str, float], threading.Lock]:
        cache = getattr(self, "_audio_duration_sec_cache", None)
        lock = getattr(self, "_audio_duration_sec_cache_lock", None)
        if cache is None or lock is None:
            cache = {}
            lock = threading.Lock()
            setattr(self, "_audio_duration_sec_cache", cache)
            setattr(self, "_audio_duration_sec_cache_lock", lock)
        return cache, lock

    def _cache_get_duration_sec(self, path: str) -> float | None:
        cache, lock = self._get_duration_cache()
        with lock:
            return cache.get(path)

    def _cache_set_duration_sec(self, path: str, duration_sec: float) -> None:
        if not (isinstance(duration_sec, (int, float)) and math.isfinite(duration_sec) and duration_sec >= 0):
            return
        cache, lock = self._get_duration_cache()
        with lock:
            if len(cache) >= int(self._duration_cache_max_size):
                cache.clear()
            cache[path] = float(duration_sec)

    def _probe_duration_sec(self, path: str) -> float | None:
        if not isinstance(path, str) or path == "":
            return None
        if path.startswith("file://"):
            path = path[7:]

        # Avoid probing non-local URIs here.
        if path.startswith("s3://"):
            return None

        # Fast path: soundfile header (wav/flac/ogg/etc).
        if soundfile is not None:
            try:
                info = soundfile.info(path)
                frames = getattr(info, "frames", None)
                sr = getattr(info, "samplerate", None)
                if isinstance(frames, int) and isinstance(sr, int) and frames >= 0 and sr > 0:
                    return float(frames) / float(sr)
            except Exception:  # noqa: BLE001
                pass

        # Fallback: ffprobe via pydub.utils.mediainfo (works for mp3/m4a/...)
        if pydub_mediainfo is not None:
            try:
                meta = pydub_mediainfo(path)
                if isinstance(meta, dict):
                    dur = meta.get("duration")
                    if dur is not None:
                        d = float(dur)
                        if math.isfinite(d) and d >= 0:
                            return d
            except Exception:  # noqa: BLE001
                pass

        return None

    def _get_audio_duration_sec(self, path: str) -> float | None:
        cached = self._cache_get_duration_sec(path)
        if cached is not None:
            return cached
        dur = self._probe_duration_sec(path)
        if dur is not None:
            self._cache_set_duration_sec(path, dur)
        return dur

    def _parse_audio_json(self, audio: str) -> dict | None:
        audio = audio.strip()
        if not (audio.startswith("{") and audio.endswith("}")):
            return None
        try:
            obj = json.loads(audio)
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def _extract_audio_fields(self, audio: AudioInput) -> tuple[str | None, str | None, float | None, int | None]:
        r"""Return (path, token_str, duration_sec, num_frames) if available.

        Note: Some datasets store per-utterance `offset`+`duration` for long audio files (e.g., MGB2).
        In that case, `duration_sec` is segment-level, so we must avoid caching it for the full path.
        """
        # Dict wrapper (e.g., SpecAugment preserves raw metadata while supplying decoded waveform array).
        if isinstance(audio, dict):
            path = None
            token = None
            duration_sec = None
            num_frames = None

            p = audio.get("path") or audio.get("wav_path") or audio.get("audio_path")
            if isinstance(p, str) and p:
                path = p

            t = audio.get("token")
            if isinstance(t, str) and t:
                token = t

            for k in ("duration_sec", "duration_secs", "duration_seconds", "duration"):
                if k not in audio or audio.get(k) is None:
                    continue
                try:
                    d = float(audio.get(k))
                    if math.isfinite(d) and d >= 0:
                        duration_sec = d
                        break
                except Exception:  # noqa: BLE001
                    continue

            if duration_sec is None:
                for k in ("duration_ms", "duration_msec"):
                    if k not in audio or audio.get(k) is None:
                        continue
                    try:
                        d_ms = float(audio.get(k))
                        d = d_ms / 1000.0
                        if math.isfinite(d) and d >= 0:
                            duration_sec = d
                            break
                    except Exception:  # noqa: BLE001
                        continue

            for k in ("num_frames", "num_frames_25hz"):
                if k not in audio or audio.get(k) is None:
                    continue
                try:
                    nf = int(audio.get(k))
                    if nf >= 0:
                        num_frames = nf
                        break
                except Exception:  # noqa: BLE001
                    continue

            raw = audio.get("raw")
            if isinstance(raw, str):
                p2, t2, d2, nf2 = self._extract_audio_fields(raw)
                if path is None:
                    path = p2
                if token is None:
                    token = t2
                if duration_sec is None:
                    duration_sec = d2
                if num_frames is None:
                    num_frames = nf2

            return path, token, duration_sec, num_frames

        if not isinstance(audio, str):
            return None, None, None, None

        obj = self._parse_audio_json(audio)
        if obj is None:
            return audio, None, None, None  # treat as plain path

        path = obj.get("path") or obj.get("wav_path") or obj.get("audio_path")
        token = obj.get("token")

        offset_sec = None
        for k in (
            "offset_sec",
            "offset_secs",
            "offset_seconds",
            "offset",
            "start_sec",
            "start_secs",
            "start_seconds",
            "start_time",
            "start",
        ):
            if k not in obj:
                continue
            try:
                o = float(obj.get(k))
                if math.isfinite(o) and o >= 0:
                    offset_sec = o
                    break
            except Exception:  # noqa: BLE001
                continue

        duration_sec = None
        for k in ("duration_sec", "duration_secs", "duration_seconds", "duration"):
            if k not in obj:
                continue
            try:
                d = float(obj.get(k))
                if math.isfinite(d) and d >= 0:
                    duration_sec = d
                    break
            except Exception:  # noqa: BLE001
                continue

        if duration_sec is None:
            for k in ("duration_ms", "duration_msec"):
                if k not in obj:
                    continue
                try:
                    d_ms = float(obj.get(k))
                    d = d_ms / 1000.0
                    if math.isfinite(d) and d >= 0:
                        duration_sec = d
                        break
                except Exception:  # noqa: BLE001
                    continue

        num_frames = None
        for k in ("num_frames", "num_frames_25hz"):
            if k not in obj:
                continue
            try:
                nf = int(obj.get(k))
                if nf >= 0:
                    num_frames = nf
                    break
            except Exception:  # noqa: BLE001
                continue

        # Only cache full-file durations. For segment-based audio items, caching would be incorrect.
        if duration_sec is not None and path and offset_sec is None:
            self._cache_set_duration_sec(str(path), duration_sec)

        return path or None, token or None, duration_sec, num_frames

    @override
    def _load_single_audio(self, audio: AudioInput, sampling_rate: float) -> tuple[NDArray, float]:
        # Dict wrapper support (e.g., SpecAugment preserves raw metadata alongside decoded waveform).
        if isinstance(audio, dict):
            arr = audio.get("array")
            if isinstance(arr, np.ndarray):
                return arr, float(sampling_rate)

            raw_bytes = audio.get("bytes")
            if isinstance(raw_bytes, (bytes, bytearray)):
                audio = BytesIO(bytes(raw_bytes))
            else:
                path = audio.get("path") or audio.get("wav_path") or audio.get("audio_path") or audio.get("uri")
                if isinstance(path, (str, os.PathLike)) and os.fspath(path):
                    audio = path
                else:
                    raw = audio.get("raw")
                    if raw is not None:
                        audio = raw

        # Support JSON-encoded audio items (from FunAudioChat dataset format).
        if isinstance(audio, str):
            obj = self._parse_audio_json(audio)
            if obj is None:
                path = audio
                offset_sec = None
                duration_sec = None
            else:
                path = obj.get("path") or obj.get("wav_path") or obj.get("audio_path") or ""
                if path.startswith("file://"):
                    path = path[7:]

                offset_sec = None
                for k in (
                    "offset_sec",
                    "offset_secs",
                    "offset_seconds",
                    "offset",
                    "start_sec",
                    "start_secs",
                    "start_seconds",
                    "start_time",
                    "start",
                ):
                    if k not in obj:
                        continue
                    try:
                        o = float(obj.get(k))
                        if math.isfinite(o) and o >= 0:
                            offset_sec = o
                            break
                    except Exception:  # noqa: BLE001
                        continue

                duration_sec = None
                for k in ("duration_sec", "duration_secs", "duration_seconds", "duration"):
                    if k not in obj:
                        continue
                    try:
                        d = float(obj.get(k))
                        if math.isfinite(d) and d >= 0:
                            duration_sec = d
                            break
                    except Exception:  # noqa: BLE001
                        continue

            if not isinstance(path, str) or path == "":
                return np.zeros(0, dtype=np.float32), float(sampling_rate)

            # Segment-based loading (offset + duration) for local WAV files.
            if offset_sec is not None and duration_sec is not None and path.lower().endswith(".wav") and path.startswith("s3://") is False:
                import wave

                try:
                    with wave.open(path, "rb") as wf:
                        sr = int(wf.getframerate())
                        nch = int(wf.getnchannels())
                        sampwidth = int(wf.getsampwidth())
                        total_frames = int(wf.getnframes())

                        start_frame = max(0, int(float(offset_sec) * float(sr)))
                        num_frames = max(0, int(float(duration_sec) * float(sr)))
                        if start_frame >= total_frames or num_frames <= 0:
                            return np.zeros(0, dtype=np.float32), float(sampling_rate)

                        wf.setpos(min(start_frame, total_frames))
                        raw = wf.readframes(min(num_frames, max(0, total_frames - start_frame)))

                    if sampwidth == 1:
                        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                        samples = (samples - 128.0) / 128.0
                    elif sampwidth == 2:
                        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                        samples = samples / float(1 << 15)
                    elif sampwidth == 3:
                        u8 = np.frombuffer(raw, dtype=np.uint8)
                        if len(u8) % 3 != 0:
                            u8 = u8[: len(u8) - (len(u8) % 3)]
                        u8 = u8.reshape(-1, 3)
                        vals = (
                            (u8[:, 0].astype(np.int32))
                            | (u8[:, 1].astype(np.int32) << 8)
                            | (u8[:, 2].astype(np.int32) << 16)
                        )
                        sign = vals & 0x800000
                        vals = vals - (sign << 1)
                        samples = vals.astype(np.float32) / float(1 << 23)
                    elif sampwidth == 4:
                        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
                        samples = samples / float(1 << 31)
                    else:
                        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

                    if nch > 1:
                        samples = samples.reshape(-1, nch).mean(axis=1).astype(np.float32)
                    else:
                        samples = samples.astype(np.float32)

                    target_sr = int(sampling_rate) if sampling_rate is not None else sr
                    if int(sr) != int(target_sr):
                        if librosa is None:
                            raise ImportError("Resampling requires `librosa`.")
                        samples = librosa.resample(samples, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
                        sr = target_sr

                    return samples, float(sr)
                except Exception:  # noqa: BLE001
                    # Fall back to the default loader on failure.
                    pass

            audio = path
        return super()._load_single_audio(audio, sampling_rate)

    def _build_speech_strings(
        self, audios: list[AudioInput], processor: MMProcessor
    ) -> tuple[list[str], list[AudioInput], list[bool], list[float]]:
        audio_sampling_rate = getattr(processor, "audio_sampling_rate", 16000)
        audio_pad_token: str = getattr(processor, "audio_pad_token", "<|audio_pad|>")

        speech: list[str] = []
        feature_audios: list[AudioInput] = []
        feature_exist_mask: list[bool] = []
        audio_duration_sec_by_audio: list[float] = []

        for audio in audios:
            if isinstance(audio, str):
                path, token, duration_sec, num_frames = self._extract_audio_fields(audio)
                if token is not None and token != "":
                    speech_str = token
                    if num_frames is None and audio_pad_token:
                        try:
                            n = int(str(token).count(str(audio_pad_token)))
                            if n > 0:
                                num_frames = n
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    # Prefer inferring duration from file name to avoid extra audio I/O.
                    if num_frames is None and duration_sec is not None:
                        num_frames = int(float(duration_sec) * float(self.token_fps))

                    if num_frames is None and path:
                        m = self._segment_duration_re.search(path)
                        if m is not None:
                            try:
                                start = float(m.group(1))
                                end = float(m.group(2))
                                duration = max(0.0, end - start)
                                num_frames = int(duration * float(self.token_fps))
                            except Exception:  # noqa: BLE001
                                num_frames = None

                    if num_frames is None:
                        # Fallback: infer duration from metadata (no waveform decode) when possible.
                        duration = None
                        if path:
                            duration = self._get_audio_duration_sec(path)
                        if duration is not None:
                            num_frames = int(float(duration) * float(self.token_fps))
                        else:
                            # Last resort: decode waveform to infer duration.
                            wav, _ = self._load_single_audio(path or audio, float(audio_sampling_rate))
                            num_frames = int((float(wav.shape[0]) / float(audio_sampling_rate)) * float(self.token_fps))

                    speech_str = audio_pad_token * max(1, int(num_frames))

                try:
                    if duration_sec is not None:
                        audio_duration_sec_by_audio.append(float(duration_sec))
                    elif num_frames is not None:
                        audio_duration_sec_by_audio.append(float(num_frames) / float(self.token_fps))
                    else:
                        audio_duration_sec_by_audio.append(0.0)
                except Exception:  # noqa: BLE001
                    audio_duration_sec_by_audio.append(0.0)

                speech.append(speech_str)
                if path is not None and path != "":
                    # Keep the original string so JSON-encoded audio metadata (e.g., offset) is preserved.
                    feature_audios.append(audio)
                    feature_exist_mask.append(True)
                else:
                    feature_exist_mask.append(False)

            elif isinstance(audio, dict):
                # Dict wrapper: prefer raw metadata (duration/num_frames/token) for speech length,
                # while keeping the dict itself for continuous feature extraction (may contain `array`).
                path, token, duration_sec, num_frames = self._extract_audio_fields(audio)
                if token is not None and token != "":
                    speech_str = token
                    if num_frames is None and audio_pad_token:
                        try:
                            n = int(str(token).count(str(audio_pad_token)))
                            if n > 0:
                                num_frames = n
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    if num_frames is None and duration_sec is not None:
                        num_frames = int(float(duration_sec) * float(self.token_fps))

                    if num_frames is None and path:
                        m = self._segment_duration_re.search(path)
                        if m is not None:
                            try:
                                start = float(m.group(1))
                                end = float(m.group(2))
                                duration = max(0.0, end - start)
                                num_frames = int(duration * float(self.token_fps))
                            except Exception:  # noqa: BLE001
                                num_frames = None

                    if num_frames is None:
                        duration = None
                        if path:
                            duration = self._get_audio_duration_sec(path)
                        if duration is not None:
                            num_frames = int(float(duration) * float(self.token_fps))
                        else:
                            wav, _ = self._load_single_audio(audio, float(audio_sampling_rate))
                            num_frames = int((float(wav.shape[0]) / float(audio_sampling_rate)) * float(self.token_fps))

                    speech_str = audio_pad_token * max(1, int(num_frames))

                try:
                    if duration_sec is not None:
                        audio_duration_sec_by_audio.append(float(duration_sec))
                    elif num_frames is not None:
                        audio_duration_sec_by_audio.append(float(num_frames) / float(self.token_fps))
                    else:
                        audio_duration_sec_by_audio.append(0.0)
                except Exception:  # noqa: BLE001
                    audio_duration_sec_by_audio.append(0.0)

                speech.append(speech_str)
                has_waveform = isinstance(audio.get("array"), np.ndarray) or isinstance(
                    audio.get("bytes"), (bytes, bytearray)
                )
                if has_waveform or (path is not None and path != "") or isinstance(audio.get("raw"), (str, os.PathLike)):
                    feature_audios.append(audio)
                    feature_exist_mask.append(True)
                else:
                    feature_exist_mask.append(False)

            else:
                # NDArray / file-like
                wav, _ = self._load_single_audio(audio, float(audio_sampling_rate))
                num_frames = int((float(wav.shape[0]) / float(audio_sampling_rate)) * float(self.token_fps))
                speech.append(audio_pad_token * max(1, num_frames))
                try:
                    audio_duration_sec_by_audio.append(float(wav.shape[0]) / float(audio_sampling_rate))
                except Exception:  # noqa: BLE001
                    audio_duration_sec_by_audio.append(float(num_frames) / float(self.token_fps))
                feature_audios.append(audio)
                feature_exist_mask.append(True)

        return speech, feature_audios, feature_exist_mask, audio_duration_sec_by_audio

    def _get_speech_lengths(self, speech: list[str], processor: MMProcessor) -> list[int]:
        speech_tokenizer = getattr(processor, "speech_tokenizer", None)
        if speech_tokenizer is None:
            # Fallback: treat each speech string as a single token.
            return [1] * len(speech)

        audio_group_size = getattr(processor, "audio_group_size", 5)
        speech_inputs = speech_tokenizer(
            speech,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            pad_to_multiple_of=audio_group_size,
            return_tensors="pt",
        )
        return speech_inputs["attention_mask"].sum(-1).tolist()

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        if processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        bos_token: str = getattr(processor, "audio_bos_token", "<|audio_bos|>")
        eos_token: str = getattr(processor, "audio_eos_token", "<|audio_eos|>")
        audio_group_size: int = getattr(processor, "audio_group_size", 5)

        placeholders = list(
            dict.fromkeys(  # preserve order, remove duplicates
                [
                    AUDIO_PLACEHOLDER,
                    f"{bos_token}{self.audio_token}{eos_token}",
                ]
            )
        )

        messages = deepcopy(messages)
        speech, _, _, _ = self._build_speech_strings(audios, processor)
        speech_lengths = self._get_speech_lengths(speech, processor)

        def _find_next_placeholder(content: str, start: int) -> tuple[int, str] | None:
            best_ph: str | None = None
            best_idx: int | None = None
            for ph in placeholders:
                idx = content.find(ph, start)
                if idx == -1:
                    continue
                if best_idx is None or idx < best_idx:
                    best_idx, best_ph = idx, ph
            if best_idx is None or best_ph is None:
                return None
            return best_idx, best_ph

        for message in messages:
            content = message["content"]
            search_start = 0
            while True:
                found = _find_next_placeholder(content, search_start)
                if found is None:
                    break
                idx, ph = found
                if len(speech_lengths) == 0:
                    raise ValueError("Audio placeholders exceed the number of provided audios.")

                speech_length = int(speech_lengths.pop(0))
                audio_seqlen = (speech_length + (audio_group_size - 1)) // audio_group_size
                replacement = f"{bos_token}{(self.audio_token or '') * int(max(1, audio_seqlen))}{eos_token}"
                content = content[:idx] + replacement + content[idx + len(ph) :]
                search_start = idx + len(replacement)

            message["content"] = content

        if len(speech_lengths) != 0:
            raise ValueError("The number of audios does not match the number of audio placeholders in messages.")

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: MMProcessor | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        if processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        dl_perf_enabled = is_env_enabled("LLAMAFACTORY_PERF_LOG") and is_env_enabled("LLAMAFACTORY_DATALOADER_PERF_LOG")
        mm_inputs: dict[str, list[int] | torch.Tensor] = {}

        speech, feature_audios, feature_exist_mask, audio_duration_sec_by_audio = self._build_speech_strings(audios, processor)
        # We may downgrade some items to "no continuous features" if waveform loading fails.
        feature_exist_mask = list(feature_exist_mask)
        feature_load_fail_mask: list[bool] = [False] * len(feature_exist_mask)
        audio_group_size = getattr(processor, "audio_group_size", 5)
        speech_tokenizer = getattr(processor, "speech_tokenizer", None)
        if speech_tokenizer is None:
            raise ValueError("Speech tokenizer was not found, please check and update your model file.")

        t_speech0 = time.perf_counter() if dl_perf_enabled else 0.0
        speech_inputs = speech_tokenizer(
            speech,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            pad_to_multiple_of=audio_group_size,
            return_tensors="pt",
        )
        if dl_perf_enabled:
            mm_inputs["perf_mm_speech_tokenizer_ms"] = (time.perf_counter() - t_speech0) * 1000.0
        mm_inputs["speech_ids"] = speech_inputs.pop("input_ids")
        mm_inputs["speech_attention_mask"] = speech_inputs.pop("attention_mask")
        if audio_duration_sec_by_audio:
            # Duration aligned with input `audios` (flattened across batch), so collator can reconstruct
            # per-sample durations even when dataset metadata is missing.
            mm_inputs["audio_duration_sec_by_audio"] = torch.tensor(audio_duration_sec_by_audio, dtype=torch.float32)

        # Continuous waveform features are only computed for audios with an available waveform.
        if len(feature_audios) != 0:
            feature_extractor = getattr(processor, "feature_extractor", None)
            if feature_extractor is None:
                raise ValueError("Audio feature extractor was not found, please check and update your model file.")

            audio_sampling_rate = getattr(processor, "audio_sampling_rate", 16000)
            audio_padding = getattr(processor, "audio_padding", "max_length")

            max_retries = int(os.getenv("LLAMAFACTORY_AUDIO_LOAD_RETRIES", "1"))
            retry_sleep_sec = float(os.getenv("LLAMAFACTORY_AUDIO_LOAD_RETRY_SLEEP", "0.2"))
            log_limit = int(os.getenv("LLAMAFACTORY_AUDIO_LOAD_ERROR_LOG_LIMIT", "20"))
            logged = int(getattr(self, "_audio_load_error_logged", 0))
            suppressed = bool(getattr(self, "_audio_load_error_suppressed", False))

            t_load0 = time.perf_counter() if dl_perf_enabled else 0.0
            wavs: list[NDArray] = []
            # Map `feature_audios` back to their sample indices.
            true_indices = [i for i, m in enumerate(feature_exist_mask) if m]
            for idx, audio in zip(true_indices, feature_audios):
                last_error: Exception | None = None
                for attempt in range(max(0, max_retries) + 1):
                    try:
                        wav, _ = self._load_single_audio(audio, float(audio_sampling_rate))
                        wavs.append(wav)
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

                if last_error is not None:
                    feature_exist_mask[idx] = False
                    feature_load_fail_mask[idx] = True
                    if logged < log_limit:
                        logger.warning_rank0(
                            "Skip continuous audio features due to load error (idx=%d, audio=%r): %s",
                            idx,
                            audio,
                            repr(last_error),
                        )
                        logged += 1
                    elif not suppressed:
                        logger.warning_rank0(
                            "Too many audio load errors (%d+); suppressing further logs.",
                            log_limit,
                        )
                        suppressed = True

            setattr(self, "_audio_load_error_logged", logged)
            setattr(self, "_audio_load_error_suppressed", suppressed)
            if dl_perf_enabled:
                mm_inputs["perf_mm_audio_load_ms"] = (time.perf_counter() - t_load0) * 1000.0

            # Only run the feature extractor when at least one waveform is available.
            min_samples = int(getattr(feature_extractor, "n_fft", 400) or 400)
            if len(wavs) != 0:
                if min_samples > 0:
                    wavs = [np.pad(w, (0, max(0, min_samples - w.shape[0])), mode="constant") for w in wavs]
                t_fx0 = time.perf_counter() if dl_perf_enabled else 0.0
                wav_inputs = feature_extractor(
                    wavs,
                    sampling_rate=audio_sampling_rate,
                    return_attention_mask=True,
                    padding=audio_padding,
                    return_tensors="pt",
                )
                if dl_perf_enabled:
                    mm_inputs["perf_mm_feature_extractor_ms"] = (time.perf_counter() - t_fx0) * 1000.0
                mm_inputs.update(wav_inputs)
                mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)

            # Align `feature_exist_mask` device with feature tensors.
            device = None
            fam = mm_inputs.get("feature_attention_mask", None)
            if torch.is_tensor(fam):
                device = fam.device
            mm_inputs["feature_exist_mask"] = torch.tensor(feature_exist_mask, dtype=torch.bool, device=device)
            mm_inputs["feature_load_fail_mask"] = torch.tensor(feature_load_fail_mask, dtype=torch.bool, device=device)
        else:
            mm_inputs["feature_exist_mask"] = torch.tensor(feature_exist_mask, dtype=torch.bool)
            mm_inputs["feature_load_fail_mask"] = torch.tensor(feature_load_fail_mask, dtype=torch.bool)

        return mm_inputs


@dataclass
class Qwen2VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    @override
    def _preprocess_image(self, image: ImageObject, **kwargs) -> ImageObject:
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))

        return image

    @override
    def _regularize_videos(self, videos: list[VideoInput], **kwargs) -> RegularizedVideoOutput:
        results: list[list[Any]] = []
        fps_per_video: list[float] = []
        durations: list[float] = []
        for video in videos:
            frames: list[Any] = []
            if _check_video_is_nested_images(video):
                assert isinstance(video, list)
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not (
                        (_is_path_like(frame) and os.path.exists(frame)) or _is_file_like(frame)
                    ):
                        raise ValueError("Invalid image found in video frames.")

                frame_inputs = cast(list[Any], video)
                frames = self._regularize_images(frame_inputs, **kwargs)["images"]
                fps_per_video.append(kwargs.get("video_fps", 2.0))
                durations.append(len(frames) / kwargs.get("video_fps", 2.0))
            else:
                _require_pyav()
                assert av is not None
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(_decode_video_frame(frame))

                if video_stream.duration is None:
                    fps_per_video.append(kwargs.get("video_fps", 2.0))
                    durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                else:
                    time_base = video_stream.time_base
                    if time_base is None:
                        fps_per_video.append(kwargs.get("video_fps", 2.0))
                        durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                    else:
                        fps_per_video.append(len(sample_indices) / float(video_stream.duration * time_base))
                        durations.append(float(video_stream.duration * time_base))

                frames = self._regularize_images(frames, **kwargs)["images"]

            if len(frames) % 2 != 0:
                frames.append(frames[-1])

            results.append(frames)

        return {"videos": results, "fps_per_video": fps_per_video, "durations": durations}

    @override
    def _get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor,
    ) -> dict[str, torch.Tensor]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseVideoProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(video_processor(videos=video_data["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Qwen3VLPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor,
    ) -> dict[str, torch.Tensor]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_metadata = [
                {"fps": getattr(processor, "video_fps", 24.0), "duration": duration, "total_num_frames": len(video)}
                for video, duration in zip(videos["videos"], videos["durations"])
            ]
            mm_inputs.update(
                video_processor(
                    videos=videos["videos"],
                    video_metadata=video_metadata,
                    fps=getattr(processor, "video_fps", 2.0),
                    return_metadata=True,
                )
            )
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in videos["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        video_processor: BaseImageProcessor = getattr(processor, "video_processor")

        image_merge_length: int = getattr(image_processor, "merge_size") ** 2
        video_merge_length: int = getattr(video_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0  # hard code for now
            video_metadata = mm_inputs.get("video_metadata", {})

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0
            timestamps = [0]

        for idx, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod() // image_merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    metadata = video_metadata[idx]
                    timestamps = processor._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        video_processor.merge_size,
                    )
                    video_structure = ""
                    for frame_index in range(num_frames):
                        video_seqlen = (
                            video_grid_thw[num_video_tokens][1:].prod() // video_merge_length
                            if self.expand_mm_tokens
                            else 1
                        )
                        timestamp_sec = timestamps[frame_index]
                        frame_structure = (
                            f"<{timestamp_sec:.1f} seconds>"
                            f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}"
                        )
                        video_structure += frame_structure
                else:
                    video_structure = f"{self.vision_bos_token}{self.video_token}{self.vision_eos_token}"

                content = content.replace(VIDEO_PLACEHOLDER, video_structure, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class GLM4VPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor,
    ) -> dict[str, torch.Tensor]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            # prepare video metadata
            video_metadata = [
                {"fps": 2, "duration": duration, "total_frames": len(video)}
                for video, duration in zip(video_data["videos"], video_data["durations"])
            ]
            mm_inputs.update(video_processor(images=None, videos=video_data["videos"], video_metadata=video_metadata))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0  # hard code for now
            timestamps = mm_inputs.get("timestamps", [])

            if hasattr(timestamps, "tolist"):
                timestamps = timestamps.tolist()

            if not timestamps:
                timestamps_list = []
            elif isinstance(timestamps[0], list):
                timestamps_list = timestamps[0]
            else:
                timestamps_list = timestamps

            unique_timestamps = timestamps_list.copy()
            selected_timestamps = unique_timestamps[:num_frames]
            while len(selected_timestamps) < num_frames:
                selected_timestamps.append(selected_timestamps[-1] if selected_timestamps else 0)

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0
            selected_timestamps = [0]

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|begin_of_image|>{self.image_token * image_seqlen}<|end_of_image|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_structure = ""
                for frame_index in range(num_frames):
                    video_seqlen = (
                        video_grid_thw[num_video_tokens][1:].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    timestamp_sec = selected_timestamps[frame_index]
                    frame_structure = (
                        f"<|begin_of_image|>{self.image_token * video_seqlen}<|end_of_image|>{timestamp_sec}"
                    )
                    video_structure += frame_structure

                if not self.expand_mm_tokens:
                    video_structure = self.video_token

                content = content.replace(VIDEO_PLACEHOLDER, f"<|begin_of_video|>{video_structure}<|end_of_video|>", 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: ProcessorMixin | None,
    ) -> dict[str, list[int] | torch.Tensor]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("timestamps", None)
        return mm_inputs


@dataclass
class Qwen2OmniPlugin(Qwen2VLPlugin):
    audio_bos_token: str = "<|audio_start|>"
    audio_eos_token: str = "<|audio_end|>"

    @override
    def _get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor,
    ) -> dict[str, torch.Tensor]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseVideoProcessor = getattr(processor, "video_processor", None)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(video_processor(videos=video_dict["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            mm_inputs["video_second_per_grid"] = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
            )

        if len(audios) != 0:
            audio_sampling_rate = getattr(processor, "audio_sampling_rate", 16000)
            audio_padding = getattr(processor, "audio_padding", "max_length")
            audios = self._regularize_audios(
                audios,
                sampling_rate=audio_sampling_rate,
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=audio_sampling_rate,
                    return_attention_mask=True,
                    padding=audio_padding,
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

        merge_length = processor.image_processor.merge_size**2
        use_audio_in_video = getattr(processor, "use_audio_in_video", False)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            if "feature_attention_mask" in mm_inputs:
                if processor.__class__.__name__ == "Qwen3OmniMoeProcessor":  # for qwen3omni
                    input_lengths = mm_inputs["feature_attention_mask"].sum(-1)
                    input_lengths_leave = input_lengths % 100
                    feature_lengths = (input_lengths_leave - 1) // 2 + 1
                    audio_lengths = ((feature_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
                else:
                    input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                    audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            mm_inputs = {}
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            if (
                use_audio_in_video and len(audios) and len(videos)
            ):  # if use the audio of video # deal video token and audio token togather
                if len(videos) != len(audios):
                    raise ValueError(
                        f"Number of videos ({len(videos)}) must match number of audios ({len(audios)}) when using audio in video."
                    )

                while VIDEO_PLACEHOLDER in content:
                    video_pos = content.find(VIDEO_PLACEHOLDER)
                    audio_pos = content.find(AUDIO_PLACEHOLDER, video_pos)
                    if audio_pos == -1 or audio_pos < video_pos:
                        raise ValueError(
                            f"Each {VIDEO_PLACEHOLDER} must be followed by an {AUDIO_PLACEHOLDER} when using audio in video."
                        )

                    audio_t_index = torch.arange(audio_lengths[num_audio_tokens])
                    video_t_index = (
                        torch.arange(video_grid_thw[num_video_tokens][0])
                        .view(-1, 1, 1)
                        .expand(
                            -1,
                            video_grid_thw[num_video_tokens][1] // image_processor.merge_size,
                            video_grid_thw[num_video_tokens][2] // image_processor.merge_size,
                        )
                        .flatten()
                        * mm_inputs["video_second_per_grid"][num_video_tokens]
                        * 25  # FIXME hardcode of position_id_per_seconds=25
                    ).long()
                    t_ntoken_per_chunk = 50  # FIXME hardcode: [25 * 2]
                    video_chunk_indices = processor.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                    audio_chunk_indices = processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                    placeholder_string = ""
                    placeholder_string += self.vision_bos_token + self.audio_bos_token
                    for j in range(max(len(video_chunk_indices), len(audio_chunk_indices))):
                        video_chunk_index = video_chunk_indices[j] if j < len(video_chunk_indices) else None
                        audio_chunk_index = audio_chunk_indices[j] if j < len(audio_chunk_indices) else None
                        if video_chunk_index is not None:
                            placeholder_string += self.video_token * (video_chunk_index[1] - video_chunk_index[0])

                        if audio_chunk_index is not None:
                            placeholder_string += self.audio_token * (audio_chunk_index[1] - audio_chunk_index[0])

                    placeholder_string += self.audio_eos_token + self.vision_eos_token
                    content = content.replace(VIDEO_PLACEHOLDER, placeholder_string, 1)
                    content = content.replace(AUDIO_PLACEHOLDER, "", 1)
                    num_audio_tokens += 1
                    num_video_tokens += 1
            else:
                while AUDIO_PLACEHOLDER in content:
                    audio_seqlen = audio_lengths[num_audio_tokens] if self.expand_mm_tokens else 1
                    content = content.replace(
                        AUDIO_PLACEHOLDER,
                        f"{self.audio_bos_token}{self.audio_token * audio_seqlen}{self.audio_eos_token}",
                        1,
                    )
                    num_audio_tokens += 1

                while VIDEO_PLACEHOLDER in content:
                    video_seqlen = (
                        video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    content = content.replace(
                        VIDEO_PLACEHOLDER,
                        f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                        1,
                    )
                    num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor | None,
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        num_frames = 0
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values_images" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values_images"][0]))
                num_frames = 1

            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs["pixel_values_videos"][0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim

            if "pixel_values_images" in mm_inputs or "pixel_values_videos" in mm_inputs:
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                video_seqlen = image_seqlen * num_frames
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen, video_seqlen = 1, 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)
                num_video_tokens += 1

            content = content.replace("{{image}}", self.image_token)
            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


@dataclass
class LFMVLPlugin(BasePlugin):
    r"""Plugin for LFM2.5-VL vision-language models.

    LFM2.5-VL uses dynamic image token counts based on image resolution.
    The image processor returns spatial_shapes tensor with [height, width] grid dimensions.
    Token count per image = (spatial_h * spatial_w) / (downsample_factor^2)
    """

    @override
    def _get_mm_inputs(
        self,
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: MMProcessor,
    ) -> dict[str, torch.Tensor]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))
        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list[ImageInput],
        videos: list[VideoInput],
        audios: list[AudioInput],
        processor: Optional[MMProcessor],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        downsample_factor: int = getattr(image_processor, "downsample_factor", 2)

        if self.expand_mm_tokens and len(images) > 0:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            spatial_shapes = mm_inputs.get("spatial_shapes", [])
        else:
            spatial_shapes = []

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens and len(spatial_shapes) > num_image_tokens:
                    h, w = spatial_shapes[num_image_tokens].tolist()
                    image_seqlen = (h * w) // (downsample_factor * downsample_factor)
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class YoutuVLPlugin(BasePlugin):
    r"""Plugin for Youtu-VL vision-language models."""

    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        for message in messages:
            content = message["content"]
            content = content.replace(
                IMAGE_PLACEHOLDER, f"{self.vision_bos_token}{self.image_token}{self.vision_eos_token}"
            )
            content = content.replace(
                VIDEO_PLACEHOLDER, f"{self.vision_bos_token}{self.video_token}{self.vision_eos_token}"
            )

            message["content"] = content

        return messages


PLUGINS = {
    "base": BasePlugin,
    "ernie_vl": ErnieVLPlugin,
    "gemma3": Gemma3Plugin,
    "glm4v": GLM4VPlugin,
    "gemma3n": Gemma3nPlugin,
    "intern_vl": InternVLPlugin,
    "kimi_vl": KimiVLPlugin,
    "llama4": Llama4Plugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "lfm2_vl": LFMVLPlugin,
    "minicpm_v": MiniCPMVPlugin,
    "mllama": MllamaPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "funaudiochat": FunAudioChatPlugin,
    "qwen2_audio": Qwen2AudioPlugin,
    "qwen3_asr": Qwen3ASRPlugin,
    "voxtral": VoxtralPlugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "video_llava": VideoLlavaPlugin,
    "youtu_vl": YoutuVLPlugin,
}


def register_mm_plugin(name: str, plugin_class: type[BasePlugin]) -> None:
    r"""Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: str | None = None,
    video_token: str | None = None,
    audio_token: str | None = None,
    **kwargs,
) -> BasePlugin:
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token, **kwargs)
