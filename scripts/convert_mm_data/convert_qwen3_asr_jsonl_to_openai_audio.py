#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e


def _build_messages(user_text: str, target: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": target},
    ]


def main() -> None:
    p = argparse.ArgumentParser(
        prog="convert_qwen3_asr_jsonl_to_openai_audio.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input Qwen3-ASR finetuning JSONL (fields: audio, text, optional prompt)")
    p.add_argument("--output", required=True, help="Output OpenAI messages+audios JSONL for LLaMA-Factory")
    p.add_argument("--audio-key", default="audio", help="Input field name for audio path")
    p.add_argument("--text-key", default="text", help="Input field name for transcript/target text")
    p.add_argument("--prompt-key", default="prompt", help="Optional input field name for system prompt")
    p.add_argument("--duration-key", default="duration", help="Optional input field name for audio duration (seconds)")
    p.add_argument(
        "--user-text",
        default="<audio>",
        help="User message content (must contain the `<audio>` placeholder).",
    )
    p.add_argument("--max-samples", type=int, default=0, help="Convert at most N samples (0 = all).")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "<audio>" not in args.user_text:
        raise ValueError("--user-text must contain `<audio>` placeholder.")

    n_out = 0
    with out_path.open("w", encoding="utf-8") as out:
        for line_no, ex in _jsonl_iter(in_path):
            if args.max_samples and n_out >= args.max_samples:
                break

            audio = ex.get(args.audio_key)
            target = ex.get(args.text_key)
            system = ex.get(args.prompt_key, "") or ""

            if not isinstance(audio, str) or not audio:
                raise ValueError(f"Missing/invalid `{args.audio_key}` at line {line_no}.")
            if not isinstance(target, str) or not target:
                raise ValueError(f"Missing/invalid `{args.text_key}` at line {line_no}.")

            row: dict[str, Any] = {
                "system": system,
                "messages": _build_messages(args.user_text, target),
                "audios": [audio],
            }

            duration = ex.get(args.duration_key, None)
            if isinstance(duration, (int, float)) and duration > 0:
                row[args.duration_key] = float(duration)

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[OK] Converted {n_out} samples -> {out_path}")


if __name__ == "__main__":
    main()

