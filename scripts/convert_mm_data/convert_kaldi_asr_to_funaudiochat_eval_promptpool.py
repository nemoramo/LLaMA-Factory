#!/usr/bin/env python3
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

"""Convert a Kaldi-style ASR testset (wav.scp + text) into FunAudioChat S2T eval jsonl with prompt_pool.

Output format matches existing promptpool eval datasets under dataset_dir:
{
  "system": "...",
  "messages": [{"role":"user","content":"... <audio>"}, {"role":"assistant","content":"..."}],
  "audios": ["/abs/path/to.wav"],
  "prompt_pool": [{"text":"...", "completion":"...", "weight":1.0}]
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _read_kaldi_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # kaldi uses whitespace/tab separated "<utt> <value...>"
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad kaldi line (expect 2 fields): {path} :: {line[:200]}")
            utt, value = parts
            mapping[utt] = value.strip()
    return mapping


def _try_normalize_text(text: str, lang: str) -> str:
    """Best-effort normalization aligned with eval_asr_wer_cer.py --lang-normalize."""

    # Prefer the user's local speech_related_tools if available.
    try:
        speech_tools = Path("~/projects/speech_related_tools").expanduser()
        if speech_tools.exists():
            sys.path.insert(0, str(speech_tools))
        from data.normalize_data import normalize_text  # type: ignore

        return normalize_text(text, lang=lang)
    except Exception:
        # Fallback: lowercase + strip punctuation (keep apostrophes) + collapse spaces.
        s = text.lower()
        s = re.sub(r"[^\w\s']+", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s


def _write_jsonl(
    wav_scp: Path,
    text_file: Path,
    output: Path,
    *,
    lang: str,
    prompt_pool_text: str,
    normalize_completion: bool,
    check_audio_exists: bool,
    max_samples: int,
) -> None:
    wav_map = _read_kaldi_map(wav_scp)
    txt_map = _read_kaldi_map(text_file)

    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    missing_audio = 0
    missing_text = 0

    with output.open("w", encoding="utf-8") as out:
        for utt, audio_path in wav_map.items():
            ref = txt_map.get(utt)
            if ref is None:
                missing_text += 1
                continue
            completion = _try_normalize_text(ref, lang=lang) if normalize_completion else ref.strip()
            if check_audio_exists and not Path(audio_path).exists():
                missing_audio += 1
                continue

            item = {
                "system": "You are asked to generate text tokens.",
                "messages": [
                    {"role": "user", "content": "Transcribe the audio. Output only the text: <audio>"},
                    {"role": "assistant", "content": completion},
                ],
                "audios": [audio_path],
                "prompt_pool": [{"text": prompt_pool_text, "completion": completion, "weight": 1.0}],
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
            if max_samples and n >= max_samples:
                break

    stats = {
        "wav_scp": str(wav_scp),
        "text": str(text_file),
        "output": str(output),
        "lang": lang,
        "normalize_completion": normalize_completion,
        "prompt_pool_text": prompt_pool_text,
        "written": n,
        "missing_text": missing_text,
        "missing_audio": missing_audio,
    }
    stats_path = output.with_suffix(output.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--wav-scp", required=True, help="Path to wav.scp")
    p.add_argument("--text", required=True, help="Path to text")
    p.add_argument("--output", required=True, help="Output jsonl path")
    p.add_argument("--lang", default="english", help="Language name for normalization and lang hint")
    p.add_argument(
        "--prompt-pool-text",
        required=True,
        help="The prompt_pool['text'] to attach (e.g. normalized/verbatim + optional language line).",
    )
    p.add_argument("--normalize-completion", action="store_true", help="Normalize reference text for completion.")
    p.add_argument("--check-audio-exists", action="store_true", help="Skip samples whose audio file is missing.")
    p.add_argument("--max-samples", type=int, default=0, help="Debug: truncate to first N samples in wav.scp order.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _write_jsonl(
        Path(args.wav_scp).expanduser(),
        Path(args.text).expanduser(),
        Path(args.output).expanduser(),
        lang=args.lang,
        prompt_pool_text=args.prompt_pool_text,
        normalize_completion=args.normalize_completion,
        check_audio_exists=args.check_audio_exists,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

