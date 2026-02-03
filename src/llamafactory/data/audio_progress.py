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


@dataclass(frozen=True)
class AudioDurationCacheEntry:
    path: str
    size: int
    mtime: int
    md5: str | None
    num_lines: int
    duration_sec: float


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
    # Common key aliases (keep in sync with `_extract_duration_sec`).
    for key in (b'"duration"', b'"duration_sec"', b'"audio_duration"', b'"duration_seconds"'):
        pos = line.find(key)
        if pos < 0:
            continue
        tail = line[pos + len(key) :]
        colon = tail.find(b":")
        if colon < 0:
            continue
        return _parse_json_number_from_bytes(tail[colon + 1 :])

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


def _read_jsonl_duration_and_md5(path: str, *, max_bytes_per_sec: float | None = None) -> tuple[float, int, str]:
    md5 = hashlib.md5()
    total_sec = 0.0
    num_lines = 0
    bytes_seen = 0
    start = time.time()

    with open(path, "rb") as f:
        for raw in f:
            md5.update(raw)
            if max_bytes_per_sec is not None and max_bytes_per_sec > 0:
                bytes_seen += len(raw)
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
            d = _extract_duration_sec_from_json_bytes(line)
            if d is None:
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                d = _extract_duration_sec(obj)
                if d is None:
                    continue
            total_sec += float(d)
            num_lines += 1

    return float(total_sec), int(num_lines), md5.hexdigest()


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
        # dataset_name is the file_name in DATA_CONFIG (dataset_info.json) and can be a directory of shards.
        root = os.path.join(dataset_dir, str(attr.dataset_name))
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
        if (
            isinstance(prev_entry, dict)
            and int(prev_entry.get("size", -1)) == size
            and int(prev_entry.get("mtime", -2)) == mtime
            and _safe_float(prev_entry.get("duration_sec")) is not None
        ):
            dur = float(prev_entry["duration_sec"])
            num_lines = int(prev_entry.get("num_lines", 0) or 0)
            md5 = prev_entry.get("md5")
            md5 = str(md5) if isinstance(md5, str) and md5 else None
        else:
            start = time.time()
            dur, num_lines, md5 = _read_jsonl_duration_and_md5(path, max_bytes_per_sec=max_bytes_per_sec)
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
        )
        entries[path] = entry
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
        root = os.path.join(dataset_dir, str(attr.dataset_name))
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
    if isinstance(cached_datasets, list) and [str(x) for x in cached_datasets] != [str(x) for x in dataset_names]:
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
        if _safe_float(entry.get("duration_sec")) is None:
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
