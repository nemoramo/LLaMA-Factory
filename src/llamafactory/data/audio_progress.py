from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from ..extras import logging
from .parser import get_dataset_list


logger = logging.get_logger(__name__)

_NO_AUDIO_EPSILON_SEC = 1e-6  # keep >0 so older trainers can mark cache "ready" even with text-only files


@dataclass(frozen=True)
class AudioDurationCacheEntry:
    path: str
    size: int
    mtime: int
    md5: str | None
    num_lines: int
    duration_sec: float
    has_audio: bool = True
    scan_mode: str | None = None


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return None
    if v < 0 or v != v or v == float("inf") or v == float("-inf"):
        return None
    return v


def _extract_duration_sec(obj: dict[str, Any]) -> float | None:
    """Extract per-sample duration in seconds from a JSON line.

    Assumes the dataset provides a duration field (user data contract), but keeps
    a few common aliases for robustness.
    """
    for k in ("duration", "duration_sec", "audio_duration", "duration_seconds"):
        if k in obj:
            d = _safe_float(obj.get(k))
            if d is not None:
                return d

    # Some datasets store duration in milliseconds.
    for k in ("duration_ms", "duration_msec"):
        if k in obj:
            d_ms = _safe_float(obj.get(k))
            if d_ms is not None:
                return float(d_ms) / 1000.0

    def _extract_from_audio_obj(inner: dict[str, Any]) -> float | None:
        # Prefer explicit duration fields.
        for k in ("duration", "duration_sec", "audio_duration", "duration_seconds"):
            if k in inner:
                d = _safe_float(inner.get(k))
                if d is not None:
                    return d
        for k in ("duration_ms", "duration_msec"):
            if k in inner:
                d_ms = _safe_float(inner.get(k))
                if d_ms is not None:
                    return float(d_ms) / 1000.0

        # Fallback: infer from segment offsets when duration is absent.
        start = None
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
            if k in inner:
                start = _safe_float(inner.get(k))
                if start is not None:
                    break

        end = None
        for k in ("end_sec", "end_secs", "end_seconds", "end_time", "end"):
            if k in inner:
                end = _safe_float(inner.get(k))
                if end is not None:
                    break

        if start is not None and end is not None:
            d = float(end) - float(start)
            return float(d) if d > 0 else None

        # Some pipelines nest offset metadata under `audio_offset`.
        nested = inner.get("audio_offset")
        if isinstance(nested, dict):
            return _extract_from_audio_obj(nested)

        return None

    # Some FunAudioChat-style datasets store segment-level info in `audio`/`audios` items,
    # often as JSON-encoded strings. Sum all audio segments in the row.
    for k in ("audio", "audios"):
        audios = obj.get(k)
        if not isinstance(audios, list) or not audios:
            continue
        total = 0.0
        found = False
        for a in audios:
            inner: dict[str, Any] | None = None
            if isinstance(a, dict):
                inner = a
            elif isinstance(a, str):
                s = a.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        inner_obj = json.loads(s)
                        inner = inner_obj if isinstance(inner_obj, dict) else None
                    except Exception:  # noqa: BLE001
                        inner = None
            if not isinstance(inner, dict):
                continue
            d = _extract_from_audio_obj(inner)
            if d is None:
                continue
            if d > 0:
                total += float(d)
                found = True
        if found and total > 0:
            return float(total)
    return None


_SEGMENT_DURATION_RE_BYTES = re.compile(rb"_seg\d+_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\.wav")


def _parse_json_number_from_bytes(data: bytes) -> float | None:
    """Parse a JSON number (or quoted number) from bytes."""
    if not data:
        return None
    data = data.lstrip()
    if not data:
        return None
    if data[:1] == b'"':
        end = data.find(b'"', 1)
        if end <= 1:
            return None
        try:
            return _safe_float(data[1:end].decode("utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001
            return None

    end = 0
    for i, b in enumerate(data):
        if (48 <= b <= 57) or b in (43, 45, 46, 69, 101):  # 0-9 + - . E e
            end = i + 1
            continue
        break
    if end == 0:
        return None
    try:
        return _safe_float(data[:end].decode("utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return None


def _extract_duration_sec_from_json_bytes(line: bytes) -> float | None:
    def _sum_for_key(key: bytes) -> tuple[float, bool]:
        total = 0.0
        found = False
        start = 0
        while True:
            pos = line.find(key, start)
            if pos < 0:
                break
            tail = line[pos + len(key) :]
            colon = tail.find(b":")
            if colon >= 0:
                v = _parse_json_number_from_bytes(tail[colon + 1 :])
                if v is not None:
                    total += float(v)
                    found = True
            start = pos + len(key)
        return float(total), bool(found)

    # Common key aliases (keep in sync with `_extract_duration_sec`).
    # Support both regular JSON objects and JSON-encoded strings inside the line
    # (e.g. `"audio": ["{\"duration\": 6.14, ...}"]`).
    unescaped_keys = (b'"duration"', b'"duration_sec"', b'"audio_duration"', b'"duration_seconds"')
    escaped_keys = (b'\\"duration\\"', b'\\"duration_sec\\"', b'\\"audio_duration\\"', b'\\"duration_seconds\\"')
    unescaped_ms_keys = (b'"duration_ms"', b'"duration_msec"')
    escaped_ms_keys = (b'\\"duration_ms\\"', b'\\"duration_msec\\"')

    total = 0.0
    found_any = False
    for key in unescaped_keys:
        s, found = _sum_for_key(key)
        if found:
            total += float(s)
            found_any = True
    if found_any:
        return float(total)

    total = 0.0
    found_any = False
    for key in escaped_keys:
        s, found = _sum_for_key(key)
        if found:
            total += float(s)
            found_any = True
    if found_any:
        return float(total)

    total_ms = 0.0
    found_any = False
    for key in unescaped_ms_keys:
        s, found = _sum_for_key(key)
        if found:
            total_ms += float(s)
            found_any = True
    if found_any:
        return float(total_ms) / 1000.0

    total_ms = 0.0
    found_any = False
    for key in escaped_ms_keys:
        s, found = _sum_for_key(key)
        if found:
            total_ms += float(s)
            found_any = True
    if found_any:
        return float(total_ms) / 1000.0

    # Fallback: infer from segment file names (e.g. *_seg0001_18.59-23.09.wav) embedded in the line.
    total = 0.0
    found = False
    for m in _SEGMENT_DURATION_RE_BYTES.finditer(line):
        try:
            start = float(m.group(1).decode("utf-8", errors="ignore"))
            end = float(m.group(2).decode("utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001
            continue
        d = max(0.0, end - start)
        if d > 0:
            total += float(d)
            found = True
    if found:
        return float(total)

    return None


def _extract_has_audio_from_json_bytes(line: bytes) -> bool:
    # Fast heuristic: our FunAudioChat-style audio datasets have an `audio`/`audios` field.
    return (b'"audio"' in line) or (b'"audios"' in line)


def _estimate_total_duration_from_audio_samples(
    *,
    path: str,
    file_size: int,
    sample_lines: list[bytes],
    sample_raw_lens: list[int],
    target_audio_files: int = 256,
) -> tuple[float, int] | None:
    if file_size <= 0 or not sample_raw_lens:
        return None
    avg_line_bytes = float(sum(sample_raw_lens)) / float(len(sample_raw_lens))
    if not (avg_line_bytes > 0):
        return None
    est_lines = int(round(float(file_size) / avg_line_bytes))
    if est_lines <= 0:
        return None

    def _wav_duration_sec(wav_path: str) -> float | None:
        try:
            st = os.stat(wav_path)
            file_size = int(st.st_size)
            if file_size <= 0:
                return None
            with open(wav_path, "rb") as f:
                header = f.read(12)
                if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                    return None

                sample_rate: int | None = None
                block_align: int | None = None
                data_offset: int | None = None
                data_size: int | None = None

                while True:
                    chunk = f.read(8)
                    if len(chunk) < 8:
                        break
                    cid = chunk[:4]
                    size = int.from_bytes(chunk[4:8], "little", signed=False)
                    if cid == b"fmt ":
                        fmt = f.read(size)
                        if len(fmt) >= 16:
                            channels = int.from_bytes(fmt[2:4], "little", signed=False)
                            sample_rate = int.from_bytes(fmt[4:8], "little", signed=False)
                            block_align = int.from_bytes(fmt[12:14], "little", signed=False)
                            bits_per_sample = int.from_bytes(fmt[14:16], "little", signed=False)
                            if (not block_align) and channels and bits_per_sample:
                                block_align = int(channels) * int((bits_per_sample + 7) // 8)
                    elif cid == b"data":
                        data_offset = int(f.tell())
                        data_size = int(size)
                        break
                    else:
                        f.seek(size, os.SEEK_CUR)
                    if size % 2 == 1:
                        f.seek(1, os.SEEK_CUR)

            if not sample_rate or not block_align or data_offset is None:
                return None
            if data_size is None or data_size <= 0 or (data_offset + data_size) > file_size:
                data_size = max(0, file_size - data_offset)
            if data_size <= 0:
                return None
            frames = float(data_size) / float(block_align)
            d = float(frames) / float(sample_rate)
        except Exception:  # noqa: BLE001
            return None
        return d if d > 0 else None

    total_sample_sec = 0.0
    sample_lines_used = 0
    audio_files_used = 0

    for raw in sample_lines:
        if audio_files_used >= int(target_audio_files):
            break
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(obj, dict):
            continue
        audios = obj.get("audio") or obj.get("audios")
        if not isinstance(audios, list) or not audios:
            continue

        line_sec = 0.0
        any_ok = False
        for a in audios:
            inner: dict[str, Any] | None = None
            if isinstance(a, str):
                try:
                    inner_obj = json.loads(a)
                    inner = inner_obj if isinstance(inner_obj, dict) else None
                except Exception:  # noqa: BLE001
                    inner = None
            elif isinstance(a, dict):
                inner = a

            if not isinstance(inner, dict):
                continue

            d = None
            for k in ("duration", "duration_sec", "audio_duration", "duration_seconds"):
                if k in inner:
                    d = _safe_float(inner.get(k))
                    break
            if d is not None and d > 0:
                line_sec += float(d)
                any_ok = True
                continue

            d_ms = None
            for k in ("duration_ms", "duration_msec"):
                if k in inner:
                    d_ms = _safe_float(inner.get(k))
                    break
            if d_ms is not None and d_ms > 0:
                line_sec += float(d_ms) / 1000.0
                any_ok = True
                continue

            start = None
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
                if k in inner:
                    start = _safe_float(inner.get(k))
                    if start is not None:
                        break

            end = None
            for k in ("end_sec", "end_secs", "end_seconds", "end_time", "end"):
                if k in inner:
                    end = _safe_float(inner.get(k))
                    if end is not None:
                        break

            if start is not None and end is not None:
                seg_d = float(end) - float(start)
                if seg_d > 0:
                    line_sec += float(seg_d)
                    any_ok = True
                    continue

            p = inner.get("path")
            if not isinstance(p, str) or not p:
                continue
            wav_d = _wav_duration_sec(p)
            if wav_d is None:
                continue
            line_sec += float(wav_d)
            audio_files_used += 1
            any_ok = True

        if any_ok and line_sec > 0:
            total_sample_sec += float(line_sec)
            sample_lines_used += 1

    if sample_lines_used <= 0:
        return None
    avg_sec_per_line = float(total_sample_sec) / float(sample_lines_used)
    est_total_sec = float(avg_sec_per_line) * float(est_lines)
    if not (est_total_sec > 0):
        return None
    logger.info_rank0(
        "Audio duration cache: estimated %s (sample_lines=%d, sample_audios=%d, est_lines=%d, est_hours=%.2f)",
        path,
        sample_lines_used,
        audio_files_used,
        est_lines,
        est_total_sec / 3600.0,
    )
    return float(est_total_sec), int(est_lines)


def _read_jsonl_duration_and_md5(
    path: str, *, file_size: int, max_bytes_per_sec: float | None = None
) -> tuple[float, int, str | None, bool, str]:
    md5 = hashlib.md5()
    total_sec = 0.0
    num_lines = 0
    bytes_seen = 0
    start = time.time()
    has_audio: bool | None = None
    found_any_duration = False
    sample_lines: list[bytes] = []
    sample_raw_lens: list[int] = []
    probe_lines = 2048

    with open(path, "rb") as f:
        for raw in f:
            raw_len = len(raw)
            if max_bytes_per_sec is not None and max_bytes_per_sec > 0:
                bytes_seen += raw_len
                if bytes_seen >= 8 * 1024 * 1024:  # 8 MiB
                    elapsed = max(1e-6, time.time() - start)
                    target = bytes_seen / float(max_bytes_per_sec)
                    if target > elapsed:
                        time.sleep(min(target - elapsed, 5.0))
                    bytes_seen = 0
                    start = time.time()

            line = raw.strip()
            if not line:
                continue
            if has_audio is None:
                has_audio = _extract_has_audio_from_json_bytes(line)

            num_lines += 1
            if len(sample_lines) < probe_lines:
                sample_lines.append(line)
                sample_raw_lens.append(raw_len)

            if not has_audio:
                if len(sample_lines) >= probe_lines:
                    break
                continue

            d = _extract_duration_sec_from_json_bytes(line)
            if d is None:
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    obj = None
                if isinstance(obj, dict):
                    d = _extract_duration_sec(obj)
            if d is not None:
                total_sec += float(d)
                found_any_duration = True

            if num_lines >= probe_lines and not found_any_duration:
                est = _estimate_total_duration_from_audio_samples(
                    path=path,
                    file_size=int(file_size),
                    sample_lines=sample_lines,
                    sample_raw_lens=sample_raw_lens,
                )
                if est is not None:
                    est_total_sec, est_lines = est
                    return float(est_total_sec), int(est_lines), None, True, "estimated_audio_sample"
                # Estimation failed (e.g. missing wav headers). Avoid reading the entire file again.
                avg_line_bytes = float(sum(sample_raw_lens)) / float(len(sample_raw_lens)) if sample_raw_lens else 0.0
                est_lines = int(round(float(file_size) / avg_line_bytes)) if avg_line_bytes > 0 else int(num_lines)
                est_lines = max(int(num_lines), int(est_lines))
                return 0.0, int(est_lines), None, True, "missing_duration"

            md5.update(raw)

    if has_audio is None:
        has_audio = False
    if not has_audio:
        return float(_NO_AUDIO_EPSILON_SEC), int(num_lines), None, False, "no_audio"

    return float(total_sec), int(num_lines), md5.hexdigest(), True, "duration_keys_or_seg_filename"


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _expand_dataset_path(path: str) -> list[str]:
    """Expand a dataset path into a list of regular files.

    The file loader supports `file_name` pointing to a directory of shard files.
    This helper returns a stable, sorted list of regular files under the directory.
    """
    try:
        if os.path.isdir(path):
            try:
                names = sorted(os.listdir(path))
            except OSError:
                return []
            out: list[str] = []
            for name in names:
                p = os.path.join(path, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    out.append(p)
            return out
    except OSError:
        return []
    return [path]


def compute_total_audio_duration_sec(
    *,
    dataset_dir: str,
    dataset_names: list[str],
    cache_path: str,
    max_bytes_per_sec: float | None = None,
) -> tuple[float, dict[str, AudioDurationCacheEntry]]:
    """Compute total audio duration (seconds) across file-based jsonl datasets.

    Uses a cache keyed by (size, mtime) to avoid rescanning unchanged files. When
    a file changes, we rescan and update md5 + duration.
    """
    prev: dict[str, Any] | None = None
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:  # noqa: BLE001
            prev = None

    prev_version = prev.get("version") if isinstance(prev, dict) else None
    if prev_version != 2:
        prev = None

    prev_files = (prev or {}).get("files") if isinstance(prev, dict) else None
    if not isinstance(prev_files, dict):
        prev_files = {}

    dataset_attrs = get_dataset_list(dataset_names, dataset_dir)
    files: list[str] = []
    for attr in dataset_attrs:
        if attr.load_from != "file":
            logger.warning_rank0(
                "Audio duration cache: skip non-file dataset %s (load_from=%s).", attr.dataset_name, attr.load_from
            )
            continue
        names = attr.dataset_name
        if isinstance(names, (list, tuple)):
            parts = [str(x) for x in names if x]
        else:
            parts = [str(names)] if names else []
        # dataset_name is the file_name in DATA_CONFIG (dataset_info.json) and can be a directory of shards.
        for part in parts:
            root = os.path.join(dataset_dir, part)
            files.extend(_expand_dataset_path(root))

    seen: set[str] = set()
    files = [p for p in files if p and not (p in seen or seen.add(p))]  # preserve order + de-dup

    entries: dict[str, AudioDurationCacheEntry] = {}
    total_sec = 0.0

    for path in files:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            logger.warning_rank0("Audio duration cache: file not found: %s", path)
            continue
        except IsADirectoryError:
            logger.warning_rank0("Audio duration cache: skip directory path: %s", path)
            continue
        except OSError as err:
            logger.warning_rank0("Audio duration cache: failed to stat %s: %s", path, err)
            continue

        if not stat.S_ISREG(st.st_mode):
            logger.warning_rank0("Audio duration cache: skip non-regular file: %s", path)
            continue

        size = int(st.st_size)
        mtime = int(st.st_mtime)
        prev_entry = prev_files.get(path)
        reuse = False
        dur = 0.0
        num_lines = 0
        md5: str | None = None
        has_audio = True
        scan_mode: str | None = None
        if isinstance(prev_entry, dict) and int(prev_entry.get("size", -1)) == size and int(prev_entry.get("mtime", -2)) == mtime:
            prev_dur = _safe_float(prev_entry.get("duration_sec"))
            if prev_dur is not None:
                dur = float(prev_dur)
                num_lines = int(prev_entry.get("num_lines", 0) or 0)
                # Avoid reusing legacy incomplete entries (e.g. old parser couldn't find durations).
                reuse = not (size > 0 and dur <= 0 and num_lines <= 0)
                if reuse:
                    md5_val = prev_entry.get("md5")
                    md5 = str(md5_val) if isinstance(md5_val, str) and md5_val else None
                    ha = prev_entry.get("has_audio")
                    has_audio = bool(ha) if ha is not None else True
                    sm = prev_entry.get("scan_mode")
                    scan_mode = str(sm) if isinstance(sm, str) and sm else None

        if not reuse:
            start = time.time()
            dur, num_lines, md5, has_audio, scan_mode = _read_jsonl_duration_and_md5(
                path, file_size=size, max_bytes_per_sec=max_bytes_per_sec
            )
            logger.info_rank0(
                "Audio duration cache: scanned %s (lines=%d, hours=%.2f) in %.1fs",
                path,
                num_lines,
                dur / 3600.0,
                time.time() - start,
            )

        entry = AudioDurationCacheEntry(
            path=path,
            size=size,
            mtime=mtime,
            md5=md5,
            num_lines=num_lines,
            duration_sec=float(dur),
            has_audio=bool(has_audio),
            scan_mode=scan_mode,
        )
        entries[path] = entry
        # Exclude text-only datasets from total audio-hours; they keep a tiny epsilon duration only for compatibility.
        if has_audio and float(dur) > 0:
            total_sec += float(dur)

    payload = {
        "version": 2,
        "generated_at": int(time.time()),
        "dataset_dir": dataset_dir,
        "datasets": list(dataset_names),
        "total_duration_sec": float(total_sec),
        "scan_method": "duration_keys_or_seg_filename",
        "files": {p: entry.__dict__ for p, entry in entries.items()},
    }
    try:
        _atomic_write_json(cache_path, payload)
    except Exception as err:  # noqa: BLE001
        logger.warning_rank0("Audio duration cache: failed to write %s: %s", cache_path, err)

    return float(total_sec), entries


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def get_audio_duration_files(*, dataset_dir: str, dataset_names: list[str]) -> list[str]:
    """Return file-based dataset paths (expanded to shard files) for audio duration scan."""
    dataset_attrs = get_dataset_list(dataset_names, dataset_dir)
    files: list[str] = []
    for attr in dataset_attrs:
        if attr.load_from != "file":
            logger.warning_rank0(
                "Audio duration cache: skip non-file dataset %s (load_from=%s).", attr.dataset_name, attr.load_from
            )
            continue
        names = attr.dataset_name
        if isinstance(names, (list, tuple)):
            parts = [str(x) for x in names if x]
        else:
            parts = [str(names)] if names else []
        for part in parts:
            root = os.path.join(dataset_dir, part)
            files.extend(_expand_dataset_path(root))

    seen: set[str] = set()
    return [p for p in files if p and not (p in seen or seen.add(p))]  # preserve order + de-dup


def get_audio_duration_file_fingerprint(paths: list[str]) -> dict[str, tuple[int, int]]:
    """Return {path: (size, mtime)} for regular files."""
    out: dict[str, tuple[int, int]] = {}
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        out[path] = (int(st.st_size), int(st.st_mtime))
    return out


def get_cached_total_audio_duration_sec(cache_path: str) -> float | None:
    obj = _read_json(cache_path)
    if not isinstance(obj, dict):
        return None
    return _safe_float(obj.get("total_duration_sec"))


def is_audio_duration_cache_complete(
    *,
    cache_path: str,
    dataset_dir: str,
    dataset_names: list[str],
    expected_files: dict[str, tuple[int, int]],
) -> bool:
    obj = _read_json(cache_path)
    if not isinstance(obj, dict):
        return False

    cached_dir = obj.get("dataset_dir")
    cached_datasets = obj.get("datasets")
    if cached_dir is not None and str(cached_dir) != str(dataset_dir):
        return False
    if isinstance(cached_datasets, list):
        cached = [str(x) for x in cached_datasets]
        expected = [str(x) for x in dataset_names]
        if sorted(cached) != sorted(expected):
            return False

    files = obj.get("files")
    if not isinstance(files, dict):
        return False

    for path, (size, mtime) in expected_files.items():
        entry = files.get(path)
        if not isinstance(entry, dict):
            return False
        try:
            if int(entry.get("size", -1)) != int(size) or int(entry.get("mtime", -1)) != int(mtime):
                return False
        except Exception:
            return False
        dur = _safe_float(entry.get("duration_sec"))
        if dur is None:
            return False
        has_audio = entry.get("has_audio")
        has_audio = bool(has_audio) if has_audio is not None else True
        if int(size) > 0 and has_audio and not (dur > 0):
            return False

    return True


def maybe_launch_audio_duration_scan(
    *,
    dataset_dir: str,
    dataset_names: list[str],
    cache_path: str,
    log_path: str | None = None,
    max_mb_per_sec: float | None = None,
) -> bool:
    """Launch a detached background scan to build `audio_duration_cache.json`.

    Returns True if a new scan is launched; False otherwise.
    """
    lock_path = f"{cache_path}.scan.lock"
    lock = _read_json(lock_path) or {}
    pid = lock.get("pid")
    try:
        pid_int = int(pid) if pid is not None else 0
    except Exception:
        pid_int = 0
    if pid_int and _pid_alive(pid_int):
        return False
    if os.path.isfile(lock_path) and (not pid_int or not _pid_alive(pid_int)):
        try:
            os.remove(lock_path)
        except Exception:
            pass

    if log_path is None:
        log_path = os.path.join(os.path.dirname(cache_path) or ".", "audio_duration_scan.log")

    cmd = [
        sys.executable,
        "-m",
        "llamafactory.data.audio_progress_scan",
        "--dataset-dir",
        str(dataset_dir),
        "--dataset-names",
        ",".join(str(x) for x in dataset_names),
        "--cache-path",
        str(cache_path),
        "--lock-path",
        str(lock_path),
    ]
    if max_mb_per_sec is not None and float(max_mb_per_sec) > 0:
        cmd.extend(["--max-mb-per-sec", str(float(max_mb_per_sec))])

    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except Exception:
        pass

    try:
        log_f = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    except Exception:
        log_f = subprocess.DEVNULL

    try:
        subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_f,
            stderr=log_f,
            env=os.environ.copy(),
            start_new_session=True,
        )
    except Exception as err:  # noqa: BLE001
        logger.warning_rank0("Failed to launch audio duration scan: %s", err)
        try:
            if hasattr(log_f, "close"):
                log_f.close()
        except Exception:
            pass
        return False

    try:
        if hasattr(log_f, "close"):
            log_f.close()
    except Exception:
        pass

    logger.info_rank0("Audio duration scan launched in background: %s", " ".join(cmd))
    return True
