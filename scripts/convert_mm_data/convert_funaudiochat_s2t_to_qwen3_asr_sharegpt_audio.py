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

r"""Convert a FunAudioChat S2T JSONL into Qwen3-ASR-ready ShareGPT-audio JSONL.

This is intended for datasets like:

Input (jsonl, each line):
  {
    "system": "...",
    "messages": [...],
    "audio": [
      "{\"path\": \"/abs/path.wav\", \"text\": \"...\", \"ref_text\": \"...\"}"
    ],
    "prompt_pool": [...],
    ...
  }

Output (jsonl, each line):
  {
    "system": "...",
    "messages": [...],               # audio placeholders normalized to "<audio>"
    "audios": ["/abs/path.wav"],
    "prompt_pool": [...]
  }

Notes:
- This script is streaming and can handle very large JSONL files.
- It normalizes FunAudioChat's "<|audio_bos|><|AUDIO|><|audio_eos|>" placeholder
  to the default "<audio>" placeholder expected by LLaMA-Factory multimodal plugins.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_INPUT_AUDIO_KEY = "audio"
DEFAULT_OUTPUT_AUDIOS_KEY = "audios"
DEFAULT_INPUT_AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"
DEFAULT_OUTPUT_AUDIO_PLACEHOLDER = "<audio>"
DEFAULT_LOG_EVERY = 100_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input FunAudioChat S2T jsonl path.")
    p.add_argument("--output", type=str, required=True, help="Output ShareGPT-audio jsonl path.")
    p.add_argument(
        "--input-audios-key",
        type=str,
        default=DEFAULT_INPUT_AUDIO_KEY,
        help="Input JSON field name for audio list (FunAudioChat default: audio).",
    )
    p.add_argument(
        "--output-audios-key",
        type=str,
        default=DEFAULT_OUTPUT_AUDIOS_KEY,
        help="Output JSON field name for audio path list (default: audios).",
    )
    p.add_argument(
        "--input-audio-placeholder",
        type=str,
        default=DEFAULT_INPUT_AUDIO_PLACEHOLDER,
        help="Audio placeholder string in input messages to be normalized.",
    )
    p.add_argument(
        "--output-audio-placeholder",
        type=str,
        default=DEFAULT_OUTPUT_AUDIO_PLACEHOLDER,
        help="Audio placeholder string to write into output messages.",
    )
    p.add_argument(
        "--keep-system",
        action="store_true",
        help="Keep the top-level `system` field in output (default: drop if missing).",
    )
    p.add_argument(
        "--keep-prompt-pool",
        action="store_true",
        help="Keep the `prompt_pool` field in output when present.",
    )
    p.add_argument(
        "--keep-duration",
        action="store_true",
        help="Keep `duration` (seconds) field in output when present.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Convert at most N samples (for debugging).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Log progress every N lines.",
    )
    return p.parse_args()


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _maybe_parse_audio_item(audio_item: Any) -> Optional[str]:
    if isinstance(audio_item, str):
        s = audio_item.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
            except Exception:  # noqa: BLE001
                obj = None
            if isinstance(obj, dict):
                path = obj.get("path") or obj.get("wav_path") or obj.get("audio_path")
                if isinstance(path, str) and path:
                    return path[7:] if path.startswith("file://") else path
                return None
        return s if s else None
    if isinstance(audio_item, dict):
        path = audio_item.get("path") or audio_item.get("wav_path") or audio_item.get("audio_path")
        if isinstance(path, str) and path:
            return path[7:] if path.startswith("file://") else path
    return None


def _normalize_messages(
    messages: Any, input_ph: str, output_ph: str
) -> Optional[list[dict[str, str]]]:
    if not isinstance(messages, list):
        return None
    normed: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        if input_ph and input_ph in content:
            content = content.replace(input_ph, output_ph)
        # Also normalize bare "<|AUDIO|>" if present.
        if "<|AUDIO|>" in content and output_ph != "<|AUDIO|>":
            content = content.replace("<|AUDIO|>", output_ph)
        normed.append({"role": role, "content": content})
    return normed


def main() -> None:
    args = parse_args()
    _ensure_parent(args.output)

    total = 0
    written = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    with open(args.input, "r", encoding="utf-8") as r, open(args.output, "w", encoding="utf-8") as w:
        for line in r:
            if args.max_samples is not None and written >= args.max_samples:
                break

            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue

            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                errors += 1
                continue

            messages = _normalize_messages(
                obj.get("messages"), args.input_audio_placeholder, args.output_audio_placeholder
            )
            if messages is None:
                skipped += 1
                continue

            audio_items = obj.get(args.input_audios_key) or obj.get("audios") or []
            if not isinstance(audio_items, list) or len(audio_items) == 0:
                skipped += 1
                continue

            audio_paths: list[str] = []
            for item in audio_items:
                path = _maybe_parse_audio_item(item)
                if path:
                    audio_paths.append(path)

            if len(audio_paths) == 0:
                skipped += 1
                continue

            out: dict[str, Any] = {"messages": messages, args.output_audios_key: audio_paths}
            if args.keep_system and isinstance(obj.get("system"), str):
                out["system"] = obj["system"]
            if args.keep_prompt_pool and isinstance(obj.get("prompt_pool"), list):
                out["prompt_pool"] = obj["prompt_pool"]
            if args.keep_duration and ("duration" in obj):
                out["duration"] = obj.get("duration")

            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

            if args.log_every and written % int(args.log_every) == 0:
                dt = max(1e-6, time.time() - t0)
                sys.stderr.write(
                    f"[convert] total={total} written={written} skipped={skipped} errors={errors} "
                    f"rate={written/dt:.1f} lines/s\n"
                )
                sys.stderr.flush()

    dt = max(1e-6, time.time() - t0)
    sys.stderr.write(
        f"[convert] DONE total={total} written={written} skipped={skipped} errors={errors} "
        f"elapsed={dt:.1f}s rate={written/dt:.1f} lines/s\n"
    )
    sys.stderr.flush()


if __name__ == "__main__":
    main()
