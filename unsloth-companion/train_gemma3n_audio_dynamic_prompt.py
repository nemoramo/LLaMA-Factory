#!/usr/bin/env python3
"""Standalone Unsloth Gemma-3n Audio SFT with dynamic prompt_pool sampling.

This script is intentionally *not* wired into LLaMA-Factory's Trainer / dataset pipeline.
It consumes the ShareGPT-Audio jsonl produced by LLaMA-Factory converters:
  - messages: [{"role": "user"/"assistant", "content": "... <audio> ..."}, ...]
  - audios: ["/abs/path.wav"]
  - prompt_pool (optional): [{"text": "...", "completion": "...", "weight": 0.2}, ...]

Dynamic prompt behavior:
  - one prompt_pool entry is sampled per example at collate time
  - entry.text is appended to the system prompt
  - entry.completion overwrites the assistant target text

Eval behavior:
  - prompt_pool uses the max-weight (Top1) entry (no sampling)

Limitations:
  - only 1 audio per example is supported (len(audios) == 1).
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import math
import os
import random
import re
import shutil
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore

try:
    from torch.utils.data import get_worker_info  # type: ignore
except Exception:
    get_worker_info = None  # type: ignore

try:
    from unsloth import FastModel  # type: ignore
except Exception:
    FastModel = None  # type: ignore

try:
    from transformers import TrainerCallback  # type: ignore
    from transformers import Trainer  # type: ignore
except Exception:
    TrainerCallback = object  # type: ignore
    Trainer = None  # type: ignore

try:
    from trl import SFTConfig, SFTTrainer  # type: ignore
except Exception:
    SFTConfig = None  # type: ignore
    SFTTrainer = None  # type: ignore


try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None

try:
    import librosa  # type: ignore
except Exception:
    librosa = None

try:
    import torchaudio  # type: ignore
    import torchaudio.functional as AF  # type: ignore
except Exception:
    torchaudio = None
    AF = None


DEFAULT_SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."


def _sanitize_weight(w: Any) -> float:
    try:
        fw = float(w)
    except Exception:
        fw = 1.0
    if not math.isfinite(fw) or fw < 0:
        return 0.0
    return fw


def _top1_from_pool(pool: Sequence[Any]) -> Any:
    if len(pool) == 0:
        raise ValueError("prompt_pool is empty.")

    def _key(item: Any) -> float:
        if isinstance(item, dict) and "weight" in item:
            return _sanitize_weight(item.get("weight"))
        return 1.0

    return max(pool, key=_key)


def _stable_hash64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _choose_from_pool(pool: Sequence[Any], rng: random.Random) -> Any:
    if len(pool) == 0:
        raise ValueError("prompt_pool is empty.")

    values: list[Any] = []
    weights: list[float] = []
    for item in pool:
        w = 1.0
        if isinstance(item, dict) and "weight" in item:
            w = _sanitize_weight(item.get("weight"))
        values.append(item)
        weights.append(w)

    if any(w > 0 for w in weights):
        return rng.choices(values, weights=weights, k=1)[0]
    return rng.choice(values)


def _get_sample_id(example: dict[str, Any], id_key: Optional[str]) -> str:
    if isinstance(id_key, str) and id_key and id_key in example and example[id_key] is not None:
        return str(example[id_key])
    audios = example.get("audios")
    if isinstance(audios, list) and len(audios) > 0 and audios[0] is not None:
        return str(audios[0])
    messages = example.get("messages")
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
    return str(messages) if messages is not None else ""


def _choose_from_pool_deterministic(
    pool: Sequence[Any], example: dict[str, Any], seed: int, id_key: Optional[str]
) -> Any:
    if len(pool) == 0:
        raise ValueError("prompt_pool is empty.")

    sample_id = _get_sample_id(example, id_key=id_key)
    h = _stable_hash64(f"{int(seed)}|{sample_id}")

    values: list[Any] = []
    weights: list[float] = []
    for item in pool:
        w = 1.0
        if isinstance(item, dict) and "weight" in item:
            w = _sanitize_weight(item.get("weight"))
        values.append(item)
        weights.append(w)

    total = sum(weights)
    if total > 0:
        r = (h / float(2**64)) * total
        acc = 0.0
        for v, w in zip(values, weights):
            if w <= 0:
                continue
            acc += w
            if r < acc:
                return v
        return values[-1]

    return values[int(h % len(values))]


def _ensure_system_message(messages: list[dict[str, Any]], default_system_prompt: str) -> dict[str, Any]:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            return m
    sys_msg = {"role": "system", "content": str(default_system_prompt or "")}
    messages.insert(0, sys_msg)
    return sys_msg


def _append_suffix_to_system(messages: list[dict[str, Any]], suffix: str) -> None:
    if not suffix:
        return

    sys_msg: Optional[dict[str, Any]] = None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            sys_msg = m
            break
    if sys_msg is None:
        sys_msg = {"role": "system", "content": ""}
        messages.insert(0, sys_msg)

    content = sys_msg.get("content", "")
    if isinstance(content, str):
        base = content
        sep = " " if base and not base.endswith((" ", "\n")) and not suffix.startswith((" ", "\n")) else ""
        sys_msg["content"] = f"{base}{sep}{suffix}" if base else suffix
        return

    if isinstance(content, list):
        # Best-effort support for multimodal-style content lists.
        for item in reversed(content):
            if isinstance(item, dict) and "text" in item:
                base = "" if item.get("text") is None else str(item.get("text"))
                sep = " " if base and not base.endswith((" ", "\n")) and not suffix.startswith((" ", "\n")) else ""
                item["text"] = f"{base}{sep}{suffix}" if base else suffix
                sys_msg["content"] = content
                return
        content.append({"type": "text", "text": suffix})
        sys_msg["content"] = content
        return

    base = "" if content is None else str(content)
    sep = " " if base and not base.endswith((" ", "\n")) and not suffix.startswith((" ", "\n")) else ""
    sys_msg["content"] = f"{base}{sep}{suffix}" if base else suffix


def _override_last_assistant(messages: list[dict[str, Any]], completion: Optional[str]) -> None:
    if completion is None:
        return
    completion_str = str(completion)
    for m in reversed(messages):
        if m.get("role") == "assistant":
            m["content"] = completion_str
            return
    messages.append({"role": "assistant", "content": completion_str})


def _load_audio_array(path: str, target_sr: int) -> "list[float]":
    """Load mono audio and resample to target_sr if needed. Returns float32 list."""
    if not path:
        raise ValueError("Empty audio path.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    audio: Any = None
    sr: Optional[int] = None

    # Try soundfile first (fast, common).
    if sf is not None:
        try:
            audio, sr = sf.read(path, always_2d=False)
        except Exception:
            audio = None
            sr = None

    # Fallback: torchaudio / torchcodec (avoids librosa's audioread deprecation warnings).
    if audio is None or sr is None:
        if torchaudio is not None and torch is not None:
            try:
                wav, sr = torchaudio.load(path)  # float32, channels_first=True
                if wav.ndim == 2:
                    wav = wav.mean(dim=0)
                elif wav.ndim != 1:
                    wav = wav.reshape(-1)
                audio = wav.cpu().numpy()
            except Exception:
                audio = None
                sr = None

    # Last resort: librosa (handles many formats; may rely on audioread which is deprecated in librosa>=0.10).
    if audio is None or sr is None:
        if librosa is None:
            raise RuntimeError(f"Failed to load audio: {path}. Install `soundfile`, `torchaudio` or `librosa`.")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"PySoundFile failed\. Trying audioread instead\.",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"librosa\.core\.audio\.__audioread_load.*Deprecated.*",
                    category=FutureWarning,
                )
                audio, sr = librosa.load(path, sr=None, mono=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio: {path}. Install `soundfile`, `torchaudio` or `librosa`. Original error: {e}"
            ) from e

    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert to float32 in [-1, 1] best-effort.
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(1.0, float(info.max))
    else:
        audio = audio.astype(np.float32)

    if sr is None:
        raise RuntimeError(f"Audio sample rate unknown: {path}")

    if int(sr) != int(target_sr):
        resampled = False

        # Prefer torchaudio resample if available (keeps librosa optional).
        if AF is not None and torch is not None:
            try:
                wav = torch.from_numpy(audio).float().unsqueeze(0)  # [1, T]
                wav = AF.resample(wav, int(sr), int(target_sr))
                audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
                sr = int(target_sr)
                resampled = True
            except Exception:
                resampled = False

        # Fallback: librosa.resample.
        if (not resampled) and librosa is not None:
            try:
                audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
                sr = int(target_sr)
                resampled = True
            except Exception:
                resampled = False

        if not resampled:
            raise RuntimeError(f"Need `torchaudio` or `librosa` to resample {path} from {sr} -> {target_sr}.")

    if audio.size == 0:
        raise RuntimeError(f"Empty audio after decoding: {path}")

    return audio.tolist()


class SpecAugment:
    """Waveform-level time masking (SpecAugment-style) using torchaudio.functional.mask_along_axis."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        mask_param: float = 0.1,
        num_masks: int = 2,
        fill_value: float = 0.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.mask_param = float(mask_param)
        self.num_masks = int(num_masks)
        self.fill_value = float(fill_value)

    def __call__(self, audio: "list[float]", *, rng: random.Random) -> "list[float]":
        if (not self.enabled) or self.mask_param <= 0 or self.num_masks <= 0:
            return audio

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        length = int(arr.shape[0])
        if length <= 1:
            return arr.astype(np.float32, copy=False).tolist()

        out = arr.astype(np.float32, copy=True)
        max_len = int(length * float(self.mask_param))
        if max_len <= 0:
            return out.tolist()

        if AF is None:
            for _ in range(self.num_masks):
                mask_len = rng.randint(0, max_len)
                if mask_len <= 0:
                    continue
                start = rng.randrange(0, max(1, length - mask_len + 1))
                out[start : start + mask_len] = float(self.fill_value)
            return out.tolist()

        x = torch.from_numpy(out).unsqueeze(0)  # [1, T]
        for _ in range(self.num_masks):
            seed = rng.randrange(0, 2**31 - 1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                x = AF.mask_along_axis(
                    x,
                    mask_param=max_len,
                    mask_value=float(self.fill_value),
                    axis=1,
                )
        return x.squeeze(0).cpu().numpy().astype(np.float32).tolist()


def _extract_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages up to (but excluding) the first assistant message."""
    prompt_messages: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            break
        if isinstance(m, dict):
            prompt_messages.append(m)
    return prompt_messages


def _extract_reference_text(messages: list[dict[str, Any]]) -> Optional[str]:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            return None if content is None else str(content)
    return None


def _build_generation_prompt_text(
    processor: Any,
    example: dict[str, Any],
    *,
    dataset_audio_placeholder: str,
    dynamic_prompt: bool,
) -> str:
    messages = example.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError("Each example must have `messages` as list[dict].")
    messages = copy.deepcopy(messages)

    prompt_messages = _extract_prompt_messages(messages)

    _ensure_system_message(prompt_messages, default_system_prompt=DEFAULT_SYSTEM_PROMPT)

    if dynamic_prompt:
        pool = example.get("prompt_pool")
        if isinstance(pool, list) and len(pool) > 0:
            chosen = _top1_from_pool(pool)
            if isinstance(chosen, dict):
                suffix = str(chosen.get("text") or chosen.get("suffix") or "")
            else:
                suffix = "" if chosen is None else str(chosen)
            _append_suffix_to_system(prompt_messages, suffix=suffix)

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True).strip()

    audio_token = getattr(processor, "audio_token", None) or getattr(processor.tokenizer, "audio_token", None)
    if not isinstance(audio_token, str) or not audio_token:
        raise ValueError("Cannot determine Gemma3n audio_token from processor/tokenizer.")

    return prompt_text.replace(dataset_audio_placeholder, audio_token)


def _get_dist_rank_world_size() -> tuple[int, int]:
    rank = 0
    world_size = 1
    if torch is None:
        return rank, world_size
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = int(torch.distributed.get_world_size())
            rank = int(torch.distributed.get_rank())
    except Exception:
        pass
    return rank, world_size


def _select_eval_generation_indices(*, dataset_len: int, num_samples: int, seed: int) -> list[int]:
    n = int(dataset_len)
    if n <= 0 or int(num_samples) <= 0:
        return []
    k = min(int(num_samples), n)
    rng = random.Random(int(seed))
    indices = list(range(n))
    return rng.sample(indices, k) if n > k else indices


def _generate_samples_for_indices(
    *,
    model: Any,
    processor: Any,
    dataset: Any,
    indices: Sequence[int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    dataset_audio_placeholder: str,
    dynamic_prompt: bool,
) -> list[dict[str, Any]]:
    if dataset is None or not indices:
        return []

    # Best-effort device resolution.
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_sr = (
        getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        or getattr(processor, "audio_sampling_rate", None)
        or 16000
    )
    target_sr = int(target_sr)

    model_was_training = getattr(model, "training", False)
    model.eval()

    def _short(x: Optional[str], limit: int = 200) -> str:
        if x is None:
            return ""
        s = str(x).replace("\n", "\\n")
        return s if len(s) <= limit else s[: limit - 3] + "..."

    results: list[dict[str, Any]] = []
    rank, _ = _get_dist_rank_world_size()
    try:
        torch.manual_seed(int(seed) + int(rank))
    except Exception:
        pass

    with torch.no_grad():
        for idx in indices:
            ex = dataset[int(idx)]

            prompt_text = _build_generation_prompt_text(
                processor,
                ex,
                dataset_audio_placeholder=dataset_audio_placeholder,
                dynamic_prompt=dynamic_prompt,
            )

            audio_paths = ex.get("audios")
            if not (isinstance(audio_paths, list) and len(audio_paths) == 1 and isinstance(audio_paths[0], str)):
                raise ValueError("Generation currently supports exactly 1 audio per example (audios=[path]).")

            audio_arr = _load_audio_array(audio_paths[0], target_sr=target_sr)
            inputs = processor(text=[prompt_text], audio=[audio_arr], return_tensors="pt", padding=True)
            for k_in, v_in in list(inputs.items()):
                if torch.is_tensor(v_in):
                    inputs[k_in] = v_in.to(device)

            prompt_len = int(inputs["input_ids"].shape[1])
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
            completion_ids = gen_ids[0, prompt_len:]
            pred = processor.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            ref: Optional[str] = None
            if dynamic_prompt:
                pool = ex.get("prompt_pool")
                if isinstance(pool, list) and len(pool) > 0:
                    chosen = _top1_from_pool(pool)
                    if isinstance(chosen, dict) and chosen.get("completion") is not None:
                        ref = str(chosen.get("completion"))
            if ref is None:
                ref = _extract_reference_text(ex.get("messages") or [])

            results.append(
                {
                    "idx": int(idx),
                    "rank": int(rank),
                    "pred": _short(pred),
                    "ref": _short(ref),
                }
            )

    if model_was_training:
        model.train()
    return results


def _distributed_generate_and_print_samples(
    *,
    model: Any,
    processor: Any,
    dataset: Any,
    num_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    step_label: str,
    dataset_audio_placeholder: str,
    dynamic_prompt: bool,
) -> None:
    if dataset is None or int(num_samples) <= 0:
        return

    rank, world_size = _get_dist_rank_world_size()

    indices = _select_eval_generation_indices(dataset_len=len(dataset), num_samples=int(num_samples), seed=int(seed))
    if not indices:
        return

    local_indices = indices if world_size <= 1 else indices[int(rank) :: int(world_size)]
    local_results = _generate_samples_for_indices(
        model=model,
        processor=processor,
        dataset=dataset,
        indices=local_indices,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        seed=int(seed),
        dataset_audio_placeholder=dataset_audio_placeholder,
        dynamic_prompt=dynamic_prompt,
    )

    is_dist = False
    try:
        is_dist = bool(
            torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1
        )
    except Exception:
        is_dist = False

    if is_dist:
        gathered: list[list[dict[str, Any]]] = [[] for _ in range(int(world_size))]
        torch.distributed.all_gather_object(gathered, local_results)
        if int(rank) != 0:
            return
        merged = [x for sub in gathered for x in sub]
    else:
        if int(rank) != 0:
            return
        merged = list(local_results)

    position = {int(idx): pos for pos, idx in enumerate(indices)}
    merged.sort(key=lambda r: position.get(int(r.get("idx", -1)), 1_000_000_000))

    print(f"[eval-gen] step={step_label} samples={min(int(num_samples), len(dataset))} world_size={int(world_size)}")
    for r in merged:
        idx = int(r.get("idx", -1))
        r_rank = int(r.get("rank", 0))
        pred = str(r.get("pred", ""))
        ref = str(r.get("ref", ""))
        rank_suffix = f" rank={r_rank}" if int(world_size) > 1 else ""
        print(f"[eval-gen] idx={idx}{rank_suffix} pred={pred} | ref={ref}")


@dataclass
class DynamicPromptAudioCollator:
    processor: Any
    seed: int
    dynamic_prompt: bool = True
    dynamic_prompt_deterministic: bool = False
    dynamic_prompt_id_key: Optional[str] = None
    dataset_audio_placeholder: str = "<audio>"

    specaug_enabled: bool = True
    specaug_mask_param: float = 0.1
    specaug_num_masks: int = 2
    specaug_fill_value: float = 0.0
    specaug_train_only: bool = True

    _rng: Optional[random.Random] = None
    _rng_seeded: bool = False

    def _get_rng(self) -> random.Random:
        if self._rng is None:
            self._rng = random.Random()
        if not self._rng_seeded:
            # Seed per worker/rank to avoid identical sampling streams.
            worker_id = 0
            rank = 0
            worker_seed = 0
            try:
                wi = get_worker_info()
                worker_id = wi.id if wi else 0
                worker_seed = int(torch.initial_seed())
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    try:
                        rank = int(torch.distributed.get_rank())
                    except Exception:
                        rank = 0
            except Exception:
                pass

            mixed = (int(self.seed) + worker_seed + rank * 1000 + worker_id) % (2**32)
            self._rng.seed(mixed)
            self._rng_seeded = True
        return self._rng

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        prompt_texts: list[str] = []
        audios: list[list[float]] = []

        audio_placeholder = self.dataset_audio_placeholder
        audio_token = getattr(self.processor, "audio_token", None) or getattr(
            self.processor.tokenizer, "audio_token", None
        )
        if not isinstance(audio_token, str) or not audio_token:
            raise ValueError("Cannot determine Gemma3n audio_token from processor/tokenizer.")

        full_audio_sequence = getattr(self.processor, "full_audio_sequence", None)
        if not isinstance(full_audio_sequence, str) or not full_audio_sequence:
            raise ValueError("Cannot determine Gemma3n full_audio_sequence from processor.")

        target_sr = (
            getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            or getattr(self.processor, "audio_sampling_rate", None)
            or 16000
        )
        target_sr = int(target_sr)

        rng = self._get_rng()
        is_eval_batch = any(bool(ex.get("__is_eval__", False)) for ex in examples)
        apply_specaug = bool(self.specaug_enabled) and (not bool(self.specaug_train_only) or not is_eval_batch)
        specaug = SpecAugment(
            enabled=bool(self.specaug_enabled),
            mask_param=float(self.specaug_mask_param),
            num_masks=int(self.specaug_num_masks),
            fill_value=float(self.specaug_fill_value),
        )

        for ex in examples:
            messages = ex.get("messages")
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                raise ValueError("Each example must have `messages` as list[dict].")
            messages = copy.deepcopy(messages)

            _ensure_system_message(messages, default_system_prompt=DEFAULT_SYSTEM_PROMPT)

            if self.dynamic_prompt:
                pool = ex.get("prompt_pool")
                if isinstance(pool, list) and len(pool) > 0:
                    if bool(ex.get("__is_eval__", False)):
                        chosen = _top1_from_pool(pool)
                    elif self.dynamic_prompt_deterministic:
                        chosen = _choose_from_pool_deterministic(pool, ex, seed=self.seed, id_key=self.dynamic_prompt_id_key)
                    else:
                        chosen = _choose_from_pool(pool, rng)
                    if isinstance(chosen, dict):
                        suffix = str(chosen.get("text") or chosen.get("suffix") or "")
                        completion = chosen.get("completion")
                    else:
                        suffix = str(chosen) if chosen is not None else ""
                        completion = None
                    _append_suffix_to_system(messages, suffix=suffix)
                    _override_last_assistant(messages, completion=completion)

            # Build full text.
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ).strip()
            full_text = full_text.replace(audio_placeholder, audio_token)
            texts.append(full_text)

            # Build prompt-only text (for masking prompt tokens in loss).
            prompt_messages: list[dict[str, Any]] = []
            for m in messages:
                if m.get("role") == "assistant":
                    break
                prompt_messages.append(m)
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            ).strip()
            prompt_text = prompt_text.replace(audio_placeholder, audio_token)
            prompt_texts.append(prompt_text)

            audio_paths = ex.get("audios")
            if not (isinstance(audio_paths, list) and len(audio_paths) == 1 and isinstance(audio_paths[0], str)):
                raise ValueError("This script currently supports exactly 1 audio per example (audios=[path]).")
            audio_arr = _load_audio_array(audio_paths[0], target_sr=target_sr)
            if apply_specaug:
                audio_arr = specaug(audio_arr, rng=rng)
            audios.append(audio_arr)

        # Tokenize full inputs (includes audio feature extraction + token_type_ids).
        batch = self.processor(text=texts, audio=audios, return_tensors="pt", padding=True)

        # Compute prompt lengths WITHOUT re-running the audio feature extractor.
        # We replicate the processor's text-side expansion for the audio placeholder.
        prompt_expanded = [t.replace(audio_token, full_audio_sequence) for t in prompt_texts]
        prompt_enc = self.processor.tokenizer(prompt_expanded, return_tensors="pt", padding=True)
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1).tolist()

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Mask prompt tokens (train on assistant only).
        for i, pl in enumerate(prompt_lens):
            labels[i, : int(pl)] = -100

        # Mask padding + special multimodal tokens.
        tok = self.processor.tokenizer
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == int(pad_id)] = -100
        for attr in (
            "image_token_id",
            "audio_token_id",
            "boi_token_id",
            "eoi_token_id",
            "boa_token_id",
            "eoa_token_id",
        ):
            if hasattr(tok, attr):
                labels[labels == int(getattr(tok, attr))] = -100

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--model_name", type=str, default="unsloth/gemma-3n-E4B-it")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hf_token", type=str, default=None)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,post,linear_start,linear_end,embedding_projection",
        help="Comma-separated target module names for Unsloth LoRA injection.",
    )
    p.add_argument(
        "--finetune_vision_layers", action="store_true", help="Enable vision/audio branch finetuning in Unsloth."
    )

    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument(
        "--save_top_k",
        type=int,
        default=0,
        help=(
            "Keep K checkpoints (0 disables). "
            "With eval enabled: keep best K by lowest eval_loss (requires eval_steps == save_steps). "
            "With eval disabled: keep latest K by step."
        ),
    )
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--optim", type=str, default="adamw_8bit")
    p.add_argument("--dataloader_num_workers", type=int, default=10)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument(
        "--trl_logit_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to keep TRL SFTTrainer's extra metrics (entropy / mean_token_accuracy). "
            "These require full logits and can cause OOM with large vocab models; default: off."
        ),
    )

    # Eval loop (Transformers v4.56 uses `eval_strategy` instead of `evaluation_strategy`)
    p.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=500)

    # Dynamic prompt
    p.add_argument("--dynamic_prompt", action="store_true", help="Enable prompt_pool sampling (dynamic prompt).")
    p.add_argument(
        "--dynamic_prompt_deterministic",
        action="store_true",
        help="Deterministic per-sample prompt_pool choice (stable across epochs).",
    )
    p.add_argument("--dynamic_prompt_id_key", type=str, default=None)
    p.add_argument("--dataset_audio_placeholder", type=str, default="<audio>")

    # SpecAugment
    p.add_argument("--specaug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--specaug_mask_param", type=float, default=0.1)
    p.add_argument("--specaug_num_masks", type=int, default=2)
    p.add_argument("--specaug_fill_value", type=float, default=0.0)
    p.add_argument("--specaug_train_only", action=argparse.BooleanOptionalAction, default=True)

    # Print generations during eval / after training
    p.add_argument("--eval_generate_samples", type=int, default=0, help="Print N generations from eval set (0 disables).")
    p.add_argument("--eval_generate_max_new_tokens", type=int, default=128)
    p.add_argument("--eval_generate_do_sample", action="store_true")
    p.add_argument("--eval_generate_temperature", type=float, default=1.0)
    p.add_argument("--eval_generate_top_p", type=float, default=0.95)
    p.add_argument("--eval_generate_top_k", type=int, default=64)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    missing: list[str] = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if load_dataset is None:
        missing.append("datasets")
    if TrainerCallback is object:
        missing.append("transformers")
    if FastModel is None:
        missing.append("unsloth")
    if SFTTrainer is None or SFTConfig is None:
        missing.append("trl")
    if missing:
        raise RuntimeError(f"Missing required packages: {', '.join(sorted(set(missing)))}")

    # TRL's SFTTrainer computes extra metrics (entropy / mean_token_accuracy) from `outputs.logits`.
    # Gemma-3n has a huge vocab (e.g. 262400), so materializing full logits can easily OOM.
    # Default: disable these metrics and let Unsloth use its fused CE loss path (no logits).
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1" if bool(args.trl_logit_metrics) else "0"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Quantized (4-bit/8-bit) models cannot be moved across devices by Accelerate.
    # Under torchrun/DDP, each rank must load the model directly onto its local GPU.
    if bool(args.load_in_4bit) and torch.cuda.is_available():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except Exception:
            local_rank = 0
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass

    def _is_world_process_zero_from_args(args_tr: Any) -> bool:
        try:
            should_save = getattr(args_tr, "should_save", None)
            if should_save is not None:
                return bool(should_save)
        except Exception:
            pass
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_rank()) == 0
        except Exception:
            pass
        return True

    model, processor = FastModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,  # auto
        max_seq_length=int(args.max_seq_length),
        load_in_4bit=bool(args.load_in_4bit),
        device_map={"": torch.cuda.current_device()} if bool(args.load_in_4bit) and torch.cuda.is_available() else None,
        full_finetuning=False,
        token=args.hf_token,
    )

    target_modules = [m.strip() for m in str(args.target_modules).split(",") if m.strip()]
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=bool(args.finetune_vision_layers),
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        random_state=int(args.seed),
        use_rslora=False,
        loftq_config=None,
        target_modules=target_modules,
    )

    trainable_params = sum(int(p.numel()) for p in model.parameters() if getattr(p, "requires_grad", False))
    if trainable_params <= 0:
        raise RuntimeError(
            "No trainable parameters found after LoRA injection. "
            "Check `--target_modules` and that Unsloth PEFT injection succeeded."
        )

    train_dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_jsonl, split="train") if args.eval_jsonl else None

    if str(args.eval_strategy) != "no" and eval_dataset is None:
        raise ValueError("`--eval_strategy` requires `--eval_jsonl`.")

    if eval_dataset is not None:
        eval_dataset = eval_dataset.add_column("__is_eval__", np.ones(len(eval_dataset), dtype=np.bool_))

    # `--save_top_k`:
    # - with eval enabled: keep best K checkpoints by lowest eval_loss
    # - with eval disabled: keep latest K checkpoints by step
    if int(args.save_top_k) > 0 and eval_dataset is None and str(args.eval_strategy) != "no":
        raise ValueError("`--save_top_k` with eval requires `--eval_jsonl`.")

    collator = DynamicPromptAudioCollator(
        processor=processor,
        seed=int(args.seed),
        dynamic_prompt=bool(args.dynamic_prompt),
        dynamic_prompt_deterministic=bool(args.dynamic_prompt_deterministic),
        dynamic_prompt_id_key=args.dynamic_prompt_id_key,
        dataset_audio_placeholder=args.dataset_audio_placeholder,
        specaug_enabled=bool(args.specaug),
        specaug_mask_param=float(args.specaug_mask_param),
        specaug_num_masks=int(args.specaug_num_masks),
        specaug_fill_value=float(args.specaug_fill_value),
        specaug_train_only=bool(args.specaug_train_only),
    )

    class _PrintSamplesCallback(TrainerCallback):
        def on_evaluate(self, args_tr, state, control, **kwargs):  # noqa: ANN001
            if eval_dataset is None or int(args.eval_generate_samples) <= 0:
                return control
            try:
                _distributed_generate_and_print_samples(
                    model=kwargs.get("model", model),
                    processor=processor,
                    dataset=eval_dataset,
                    num_samples=int(args.eval_generate_samples),
                    max_new_tokens=int(args.eval_generate_max_new_tokens),
                    do_sample=bool(args.eval_generate_do_sample),
                    temperature=float(args.eval_generate_temperature),
                    top_p=float(args.eval_generate_top_p),
                    top_k=int(args.eval_generate_top_k),
                    seed=int(args.seed) + int(state.global_step),
                    step_label=str(int(state.global_step)),
                    dataset_audio_placeholder=args.dataset_audio_placeholder,
                    dynamic_prompt=bool(args.dynamic_prompt),
                )
            except Exception as e:
                msg = str(e).lower()
                is_oom = "out of memory" in msg or "cuda error" in msg and "memory" in msg
                if is_oom and torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    print(f"[eval-gen] skipped due to OOM at step={int(state.global_step)}: {e}")
                else:
                    raise
            return control

    class _TopKCheckpointCallback(TrainerCallback):
        def __init__(self, k: int) -> None:
            self.k = int(k)
            self._loss_by_step: dict[int, float] = {}
            self._kept: dict[str, float] = {}

        def on_evaluate(self, args_tr, state, control, metrics=None, **kwargs):  # noqa: ANN001
            if not _is_world_process_zero_from_args(args_tr):
                return control
            if metrics is None:
                return control
            loss = metrics.get("eval_loss")
            if loss is None:
                return control
            try:
                loss_f = float(loss)
            except Exception:
                return control
            if not math.isfinite(loss_f):
                return control
            self._loss_by_step[int(state.global_step)] = loss_f
            return control

        def on_save(self, args_tr, state, control, **kwargs):  # noqa: ANN001
            if not _is_world_process_zero_from_args(args_tr):
                return control
            if self.k <= 0:
                return control
            step = int(state.global_step)
            loss = self._loss_by_step.get(step)
            if loss is None:
                return control

            ckpt_dir = kwargs.get("checkpoint_folder") or os.path.join(str(args_tr.output_dir), f"checkpoint-{step}")
            if isinstance(ckpt_dir, str) and os.path.isdir(ckpt_dir):
                self._kept[ckpt_dir] = loss

            keep_sorted = sorted(self._kept.items(), key=lambda kv: kv[1])[: self.k]
            keep_paths = {p for p, _ in keep_sorted}

            for path in glob.glob(os.path.join(str(args_tr.output_dir), "checkpoint-*")):
                if os.path.isdir(path) and path not in keep_paths:
                    shutil.rmtree(path, ignore_errors=True)
                    self._kept.pop(path, None)

            self._kept = {p: self._kept[p] for p in keep_paths if p in self._kept}
            return control

    class _KeepLastKCheckpointsCallback(TrainerCallback):
        def __init__(self, k: int) -> None:
            self.k = int(k)

        def on_save(self, args_tr, state, control, **kwargs):  # noqa: ANN001
            if not _is_world_process_zero_from_args(args_tr):
                return control
            if self.k <= 0:
                return control

            out_dir = str(args_tr.output_dir)
            checkpoints: list[tuple[int, str]] = []
            for path in glob.glob(os.path.join(out_dir, "checkpoint-*")):
                if not os.path.isdir(path):
                    continue
                base = os.path.basename(path)
                m = re.search(r"checkpoint-(\\d+)$", base)
                step = int(m.group(1)) if m else -1
                checkpoints.append((step, path))

            checkpoints.sort(key=lambda x: x[0], reverse=True)
            keep_paths = {p for _, p in checkpoints[: self.k]}
            for _, path in checkpoints[self.k :]:
                if path not in keep_paths:
                    shutil.rmtree(path, ignore_errors=True)

            return control

    callbacks_list: list[TrainerCallback] = []
    if eval_dataset is not None and int(args.eval_generate_samples) > 0:
        callbacks_list.append(_PrintSamplesCallback())

    save_top_k = int(args.save_top_k)
    use_topk_by_loss = bool(eval_dataset is not None and str(args.eval_strategy) != "no")
    if save_top_k > 0:
        if use_topk_by_loss:
            if str(args.eval_strategy) == "steps" and int(args.eval_steps) != int(args.save_steps):
                raise ValueError("`--save_top_k` requires `--eval_steps == --save_steps` so each checkpoint has eval_loss.")
            callbacks_list.append(_TopKCheckpointCallback(save_top_k))
        else:
            callbacks_list.append(_KeepLastKCheckpointsCallback(save_top_k))
    callbacks = callbacks_list if callbacks_list else None

    if bool(args.trl_logit_metrics):
        TrainerCls = SFTTrainer
    else:
        if Trainer is None:
            raise RuntimeError("`transformers.Trainer` is required.")

        class _SFTTrainerNoLogitMetrics(SFTTrainer):  # type: ignore[misc]
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001
                inputs["use_cache"] = False
                return Trainer.compute_loss(
                    self,
                    model,
                    inputs,
                    return_outputs=bool(return_outputs),
                    num_items_in_batch=num_items_in_batch,
                )

        TrainerCls = _SFTTrainerNoLogitMetrics

    trainer = TrainerCls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,  # keep TRL in text-mode; collator handles audio features
        data_collator=collator,
        callbacks=callbacks,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            per_device_eval_batch_size=int(args.per_device_eval_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            num_train_epochs=float(args.num_train_epochs),
            learning_rate=float(args.learning_rate),
            warmup_ratio=float(args.warmup_ratio),
            weight_decay=float(args.weight_decay),
            lr_scheduler_type=str(args.lr_scheduler_type),
            logging_steps=int(args.logging_steps),
            eval_strategy=str(args.eval_strategy),
            eval_steps=int(args.eval_steps),
            save_strategy="steps" if str(args.eval_strategy) == "no" else str(args.eval_strategy),
            save_steps=int(args.save_steps),
            optim=str(args.optim),
            seed=int(args.seed),
            report_to=str(args.report_to),
            dataloader_num_workers=int(args.dataloader_num_workers),
            remove_unused_columns=False,
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            prediction_loss_only=not bool(args.trl_logit_metrics),
            ddp_find_unused_parameters=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )

    trainer.train()
    # If no periodic eval is configured, still allow printing a few generations after training.
    if eval_dataset is not None and int(args.eval_generate_samples) > 0 and str(args.eval_strategy) == "no":
        _distributed_generate_and_print_samples(
            model=model,
            processor=processor,
            dataset=eval_dataset,
            num_samples=int(args.eval_generate_samples),
            max_new_tokens=int(args.eval_generate_max_new_tokens),
            do_sample=bool(args.eval_generate_do_sample),
            temperature=float(args.eval_generate_temperature),
            top_p=float(args.eval_generate_top_p),
            top_k=int(args.eval_generate_top_k),
            seed=int(args.seed),
            step_label="final",
            dataset_audio_placeholder=args.dataset_audio_placeholder,
            dynamic_prompt=bool(args.dynamic_prompt),
        )

    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        try:
            processor.save_pretrained(args.output_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
