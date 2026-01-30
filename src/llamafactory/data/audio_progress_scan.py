from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from ..extras import logging
from .audio_progress import _atomic_write_json, _pid_alive, compute_total_audio_duration_sec


logger = logging.get_logger(__name__)


def _set_low_priority() -> None:
    try:
        os.nice(19)
    except Exception:
        pass

    try:
        import shutil
        import subprocess

        ionice = shutil.which("ionice")
        if ionice:
            subprocess.run([ionice, "-c2", "-n7", "-p", str(os.getpid())], check=False)  # noqa: S603,S607
    except Exception:
        pass


def _write_lock(lock_path: str, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["pid"] = int(os.getpid())
    payload["updated_at"] = int(time.time())
    try:
        _atomic_write_json(lock_path, payload)
    except Exception:
        pass


def _cleanup_lock(lock_path: str) -> None:
    try:
        obj = None
        if os.path.isfile(lock_path):
            with open(lock_path, encoding="utf-8") as f:
                obj = json.load(f)
        pid = None
        if isinstance(obj, dict):
            pid = obj.get("pid")
        try:
            pid_int = int(pid) if pid is not None else 0
        except Exception:
            pid_int = 0
        if pid_int and pid_int != os.getpid() and _pid_alive(pid_int):
            return
    except Exception:
        pass

    try:
        os.remove(lock_path)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Background scanner for total audio duration (jsonl duration sum).")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-names", required=True, help="Comma-separated dataset keys (from dataset_info.json).")
    parser.add_argument("--cache-path", required=True, help="Output cache path (audio_duration_cache.json).")
    parser.add_argument("--lock-path", required=True, help="Lock file path (cache_path.scan.lock).")
    parser.add_argument("--max-mb-per-sec", type=float, default=0.0, help="Optional scan throttle (MB/s).")
    args = parser.parse_args(argv)

    dataset_dir = str(args.dataset_dir)
    dataset_names = [s.strip() for s in str(args.dataset_names).split(",") if s.strip()]
    cache_path = str(args.cache_path)
    lock_path = str(args.lock_path)

    _set_low_priority()
    _write_lock(
        lock_path,
        {
            "started_at": int(time.time()),
            "dataset_dir": dataset_dir,
            "datasets": dataset_names,
            "cache_path": cache_path,
            "max_mb_per_sec": float(args.max_mb_per_sec or 0.0),
        },
    )

    try:
        max_bytes_per_sec = None
        if float(args.max_mb_per_sec or 0.0) > 0:
            max_bytes_per_sec = float(args.max_mb_per_sec) * 1024.0 * 1024.0
        total, _ = compute_total_audio_duration_sec(
            dataset_dir=dataset_dir,
            dataset_names=dataset_names,
            cache_path=cache_path,
            max_bytes_per_sec=max_bytes_per_sec,
        )
        logger.info_rank0("Audio duration scan done: total_hours=%.2f", float(total) / 3600.0)
        return 0
    except Exception as err:  # noqa: BLE001
        logger.warning_rank0("Audio duration scan failed: %s", err)
        return 1
    finally:
        _cleanup_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

