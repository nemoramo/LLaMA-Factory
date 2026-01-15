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

r"""Convert an ASR ShareGPT JSONL into a FunAudioChat-friendly S2T JSONL.

Author: yufeng.ma

This script is intended for datasets like:

Input (jsonl, each line):
  {
    "messages": [...],
    "audios": ["/abs/path/to.wav"],
    "prompt_pool": [
      {"text": "...normalized...", "completion": "<norm>", "weight": 0.63},
      {"text": "...verbatim...", "completion": "<raw>", "weight": 0.27},
      ...
    ]
  }

Output (jsonl, each line):
  {
    "system": "You are asked to generate text tokens.",
    "messages": [
      {"role": "user", "content": "Transcribe ... {audio}"},
      {"role": "assistant", "content": "<normalized_text>"}
    ],
    "audio": [
      "{\"path\": \"/abs/path/to.wav\", \"token\": \"<|audio_pad|><|audio_pad|>...\", \"text\": \"<normalized_text>\", \"ref_text\": \"<original_text>\"}"
    ],
    "text": "<normalized_text>",
    "original_text": "<original_text>"
  }

Notes:
- This is **S2T-only**: one input audio, one text target.
- We ignore the original `messages` prompt and rebuild messages from `prompt_pool`.
- Speech tokens are pad-only, derived from WAV duration (25Hz), so no CosyVoice is required.
"""

import argparse
import json
import math
import os
import re
import wave
from pathlib import Path
from typing import Any, Optional


DEFAULT_S2T_PROMPT = "You are asked to generate text tokens."
DEFAULT_AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"
DEFAULT_AUDIO_PAD_TOKEN = "<|audio_pad|>"
DEFAULT_TOKEN_FPS = 25.0
DEFAULT_LOG_EVERY = 100_000

SEGMENT_PATH_DURATION_RE = re.compile(r"_seg\d+_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\.wav$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input ShareGPT ASR jsonl path.")
    p.add_argument("--output", type=str, required=True, help="Output FunAudioChat S2T jsonl path.")
    p.add_argument(
        "--input-audios-key",
        type=str,
        default="audios",
        help="Input JSON field name for audio list (e.g. audios/audio).",
    )
    p.add_argument(
        "--output-audios-key",
        type=str,
        default="audio",
        help="Output JSON field name for audio list (FunAudioChat uses `audio`).",
    )
    p.add_argument(
        "--system",
        type=str,
        default=DEFAULT_S2T_PROMPT,
        help="System prompt string (S2T mode).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Transcribe the audio. Output only the text: {audio}",
        help="User prompt template. Use `{audio}` placeholder for audio template insertion.",
    )
    p.add_argument(
        "--audio-template",
        type=str,
        default=DEFAULT_AUDIO_TEMPLATE,
        help="FunAudioChat audio placeholder template.",
    )
    p.add_argument(
        "--audio-pad-token",
        type=str,
        default=DEFAULT_AUDIO_PAD_TOKEN,
        help="Token used to build a pad-only speech token string.",
    )
    p.add_argument(
        "--token-fps",
        type=float,
        default=DEFAULT_TOKEN_FPS,
        help="Discrete token FPS for pad-only token generation (default 25Hz).",
    )
    p.add_argument(
        "--no-token",
        action="store_true",
        help="Do not precompute token strings; omit `token` in output audio items.",
    )
    p.add_argument(
        "--drop-prompt-pool",
        action="store_true",
        help="Drop `prompt_pool` from output (default keeps it, enabling prompt-style switching).",
    )
    p.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="If a WAV path is missing/unreadable, keep it but omit token instead of failing.",
    )
    p.add_argument(
        "--keep-realpath",
        action="store_true",
        help="Store `path=os.path.realpath(...)` in output (slower due to filesystem calls).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Convert at most N samples (debug).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Print progress every N input lines.",
    )
    return p.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _safe_duration_sec(value: Any) -> Optional[float]:
    try:
        d = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(d) or d < 0:
        return None
    return d


def _pick_from_prompt_pool(prompt_pool: Any, keywords: tuple[str, ...]) -> Optional[str]:
    if not isinstance(prompt_pool, list):
        return None

    best_completion = None
    best_weight = -1.0
    for item in prompt_pool:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).lower()
        if not any(k in text for k in keywords):
            continue
        completion = item.get("completion")
        if not isinstance(completion, str):
            continue
        weight = _safe_float(item.get("weight", 0.0), default=0.0)
        if weight > best_weight:
            best_weight = weight
            best_completion = completion

    return best_completion


def _extract_assistant_text(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            return m["content"]
    return None


def _wav_num_frames(path: str, *, token_fps: float) -> int:
    with wave.open(path, "rb") as wf:
        nframes = wf.getnframes()
        framerate = wf.getframerate()
    if framerate <= 0:
        return 0
    return int((float(nframes) / float(framerate)) * float(token_fps))


def _filename_num_frames(path: str, *, token_fps: float) -> Optional[int]:
    """Fast path: infer duration from `*_segXXXX_<start>-<end>.wav`."""
    m = SEGMENT_PATH_DURATION_RE.search(path)
    if not m:
        return None

    start = _safe_float(m.group(1), default=0.0)
    end = _safe_float(m.group(2), default=0.0)
    duration = max(0.0, end - start)
    return int(duration * float(token_fps))


def _build_pad_token_str(
    audio_path: str,
    *,
    audio_pad_token: str,
    token_fps: float,
    skip_missing_audio: bool,
) -> Optional[str]:
    try:
        n = _filename_num_frames(audio_path, token_fps=token_fps)
        if n is None:
            n = _wav_num_frames(audio_path, token_fps=token_fps)
        n = max(1, int(n))
        return audio_pad_token * n
    except Exception:  # noqa: BLE001
        if skip_missing_audio:
            return None
        raise


def _render_user_prompt(prompt_template: str, *, audio_template: str, num_audios: int) -> str:
    audio_block = audio_template * int(num_audios)
    if "{audio}" in prompt_template:
        return prompt_template.replace("{audio}", audio_block)
    return prompt_template


def convert_file(
    *,
    input_path: str,
    output_path: str,
    input_audios_key: str,
    output_audios_key: str,
    system_prompt: str,
    prompt_template: str,
    audio_template: str,
    audio_pad_token: str,
    token_fps: float,
    no_token: bool,
    drop_prompt_pool: bool,
    skip_missing_audio: bool,
    keep_realpath: bool,
    max_samples: Optional[int],
    log_every: int,
) -> None:
    ensure_parent_dir(output_path)

    n_in = 0
    n_out = 0
    n_missing_token = 0

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            if log_every and n_in % int(log_every) == 0:
                print(f"[..] processed {n_in} lines, wrote {n_out} samples, missing_token={n_missing_token}")
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Skip unparsable line {n_in}: {e}")
                continue

            audios = obj.get(input_audios_key)
            if not isinstance(audios, list) or len(audios) == 0 or not isinstance(audios[0], str):
                continue

            # S2T-only: we expect exactly one input audio per sample.
            if len(audios) != 1:
                raise ValueError(f"Expect 1 audio per sample, but got {len(audios)} at line {n_in}.")

            audio_path = audios[0]
            duration_value = obj.get("audio_duration")
            if duration_value is None:
                duration_value = obj.get("duration")
            duration_sec = _safe_duration_sec(duration_value)
            if duration_sec is None and isinstance(obj.get("durations"), list) and len(obj.get("durations")) == 1:
                duration_sec = _safe_duration_sec(obj["durations"][0])
            prompt_pool = obj.get("prompt_pool")

            normalized_text = _pick_from_prompt_pool(prompt_pool, keywords=("normalized", "lowercased"))
            original_text = _pick_from_prompt_pool(prompt_pool, keywords=("verbatim", "preserving", "punctuation"))
            if normalized_text is None:
                normalized_text = _extract_assistant_text(obj.get("messages"))
            if normalized_text is None:
                continue
            if original_text is None:
                original_text = normalized_text

            user_content = _render_user_prompt(prompt_template, audio_template=audio_template, num_audios=len(audios))
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": normalized_text},
            ]

            out_path = os.path.realpath(audio_path) if keep_realpath else audio_path
            audio_item: dict[str, Any] = {
                "path": out_path,
                "text": normalized_text,
                "ref_text": original_text,
            }
            if duration_sec is not None:
                audio_item["duration"] = duration_sec
            if not no_token:
                token_str = _build_pad_token_str(
                    audio_path,
                    audio_pad_token=audio_pad_token,
                    token_fps=token_fps,
                    skip_missing_audio=skip_missing_audio,
                )
                if token_str is not None:
                    audio_item["token"] = token_str
                else:
                    n_missing_token += 1

            out = {
                "system": system_prompt,
                "messages": messages,
                output_audios_key: [json.dumps(audio_item, ensure_ascii=False, sort_keys=True)],
                "text": normalized_text,
                "original_text": original_text,
            }
            if not drop_prompt_pool and isinstance(prompt_pool, list) and len(prompt_pool) > 0:
                out["prompt_pool"] = prompt_pool
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

            if max_samples is not None and n_out >= max_samples:
                break

    print(f"[OK] Read {n_in} lines, wrote {n_out} samples -> {output_path}")


def main() -> None:
    args = parse_args()
    convert_file(
        input_path=args.input,
        output_path=args.output,
        input_audios_key=args.input_audios_key,
        output_audios_key=args.output_audios_key,
        system_prompt=args.system,
        prompt_template=args.prompt,
        audio_template=args.audio_template,
        audio_pad_token=args.audio_pad_token,
        token_fps=float(args.token_fps),
        no_token=bool(args.no_token),
        drop_prompt_pool=bool(args.drop_prompt_pool),
        skip_missing_audio=bool(args.skip_missing_audio),
        keep_realpath=bool(args.keep_realpath),
        max_samples=args.max_samples,
        log_every=int(args.log_every),
    )


if __name__ == "__main__":
    main()
