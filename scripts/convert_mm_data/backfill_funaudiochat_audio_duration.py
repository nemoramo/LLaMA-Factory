#!/usr/bin/env python
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

"""Backfill `duration` (seconds) into FunAudioChat `audio` items.

This is useful for large datasets where audio paths are mp3/m4a/... and we
cannot infer duration from `_segXXXX_<start>-<end>.wav` filenames.

Input (FunAudioChat S2T jsonl, each line):
  {
    ...,
    "audio": [
      "{\"path\": \"/abs/path/to.mp3\", \"text\": \"...\", \"ref_text\": \"...\"}"
    ],
    ...
  }

Output:
  audio item JSON strings will be updated with `"duration": <float seconds>`.

Duration inference order:
  1) `_segXXXX_<start>-<end>.wav` filename
  2) `soundfile.info()` (fast header, wav/flac/ogg/...)
  3) `ffprobe` (metadata only; works for mp3/m4a/...)

Notes:
  - This script does NOT restart any training; use the generated jsonl in the next run.
  - By default, we skip items that already contain `token` to avoid extra work.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional


try:
    import soundfile as soundfile  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    soundfile = None


SEGMENT_PATH_DURATION_RE = re.compile(r"_seg\d+_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\.wav$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input FunAudioChat jsonl path.")
    p.add_argument("--output", type=str, required=True, help="Output FunAudioChat jsonl path.")
    p.add_argument(
        "--audio-key",
        type=str,
        default=None,
        help="Audio field name in samples (default: auto-detect `audio` then `audios`).",
    )
    p.add_argument(
        "--skip-if-token-present",
        action="store_true",
        help="Skip backfilling when audio item already has a non-empty `token`.",
    )
    p.add_argument(
        "--cache-size",
        type=int,
        default=20000,
        help="Max in-memory duration cache size (path -> duration_sec).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=100000,
        help="Print progress every N lines.",
    )
    return p.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_duration_sec(value: Any) -> Optional[float]:
    try:
        d = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(d) or d < 0:
        return None
    return d


def _infer_duration_from_filename(path: str) -> Optional[float]:
    m = SEGMENT_PATH_DURATION_RE.search(path)
    if not m:
        return None
    try:
        start = float(m.group(1))
        end = float(m.group(2))
        d = max(0.0, end - start)
    except Exception:  # noqa: BLE001
        return None
    return d


def _probe_duration_sec(path: str) -> Optional[float]:
    if not isinstance(path, str) or path == "":
        return None
    if path.startswith("file://"):
        path = path[7:]
    if path.startswith("s3://"):
        return None
    if not os.path.exists(path):
        return None

    d = _infer_duration_from_filename(path)
    if d is not None:
        return d

    if soundfile is not None:
        try:
            info = soundfile.info(path)
            frames = getattr(info, "frames", None)
            sr = getattr(info, "samplerate", None)
            if isinstance(frames, int) and isinstance(sr, int) and frames >= 0 and sr > 0:
                return float(frames) / float(sr)
        except Exception:  # noqa: BLE001
            pass

    # ffprobe metadata (no decode)
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            stderr=subprocess.DEVNULL,
        )
        obj = json.loads(out.decode("utf-8", errors="ignore") or "{}")
        fmt = obj.get("format") if isinstance(obj, dict) else None
        if isinstance(fmt, dict):
            return _safe_duration_sec(fmt.get("duration"))
    except Exception:  # noqa: BLE001
        return None

    return None


def _parse_audio_item(audio_item: Any) -> tuple[Optional[dict], Optional[str]]:
    if not isinstance(audio_item, str):
        return None, None
    s = audio_item.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None, None
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        return None, None
    if not isinstance(obj, dict):
        return None, None
    path = obj.get("path") or obj.get("wav_path") or obj.get("audio_path")
    return obj, (str(path) if isinstance(path, str) else None)


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.output)

    cache: dict[str, float] = {}
    cache_size = max(0, int(args.cache_size))

    n_in = 0
    n_out = 0
    n_changed = 0
    n_audio_items = 0
    n_audio_changed = 0
    n_audio_skipped_token = 0
    n_audio_missing_duration = 0

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.log_every and n_in % int(args.log_every) == 0:
                print(
                    f"[..] lines={n_in} wrote={n_out} changed={n_changed} "
                    f"audio_items={n_audio_items} audio_changed={n_audio_changed} "
                    f"skipped_token={n_audio_skipped_token} missing_dur={n_audio_missing_duration}"
                )

            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(obj, dict):
                continue

            audio_key = args.audio_key
            if audio_key is None:
                audio_key = "audio" if isinstance(obj.get("audio"), list) else ("audios" if isinstance(obj.get("audios"), list) else None)
            if audio_key is None or not isinstance(obj.get(audio_key), list):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1
                continue

            updated = False
            audio_list = obj.get(audio_key)
            assert isinstance(audio_list, list)
            new_audio_list: list[Any] = []

            for audio_item in audio_list:
                n_audio_items += 1
                audio_obj, path = _parse_audio_item(audio_item)
                if audio_obj is None or path is None:
                    new_audio_list.append(audio_item)
                    continue

                # Already has duration
                if any(k in audio_obj for k in ("duration", "duration_sec", "duration_secs", "duration_seconds", "duration_ms", "num_frames")):
                    new_audio_list.append(audio_item)
                    continue

                if args.skip_if_token_present and isinstance(audio_obj.get("token"), str) and audio_obj.get("token"):
                    n_audio_skipped_token += 1
                    new_audio_list.append(audio_item)
                    continue

                duration_sec = cache.get(path)
                if duration_sec is None:
                    duration_sec = _probe_duration_sec(path)
                    if duration_sec is not None and cache_size > 0:
                        if len(cache) >= cache_size:
                            cache.clear()
                        cache[path] = duration_sec

                if duration_sec is None:
                    n_audio_missing_duration += 1
                    new_audio_list.append(audio_item)
                    continue

                audio_obj["duration"] = float(duration_sec)
                new_audio_list.append(json.dumps(audio_obj, ensure_ascii=False, sort_keys=True))
                updated = True
                n_audio_changed += 1

            if updated:
                obj[audio_key] = new_audio_list
                n_changed += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        f"[OK] lines_in={n_in} lines_out={n_out} lines_changed={n_changed} "
        f"audio_items={n_audio_items} audio_changed={n_audio_changed} "
        f"skipped_token={n_audio_skipped_token} missing_duration={n_audio_missing_duration} -> {args.output}"
    )


if __name__ == "__main__":
    main()
