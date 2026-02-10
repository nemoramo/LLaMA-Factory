#!/usr/bin/env python3
"""
Rebuild `audio_duration_cache.json` for a running output_dir by computing total audio
duration from a sharded Parquet manifest (polars backend).

Why:
  - Some JSONL duration scanners fail to parse durations when they are stored inside
    JSON-encoded strings (e.g. `"audio": ["{\"duration\": 6.14, ...}"]`), causing
    `duration_sec=0` and the trainer refusing to mark the cache as "ready".
  - For the sharded Parquet backend, computing total audio hours from Parquet shards
    is both faster and matches the actual training data source.

This script intentionally avoids importing `llamafactory.*` to stay runnable even if
the local Python env has a broken Transformers stack.

It updates:
  - `total_duration_sec`: computed from shards (sum of `"duration": <float>` in audio JSON strings)
  - `files[*].duration_sec`: ensures >0 for all non-empty files (sets epsilon for missing)
  - `generated_at` and `scan_method`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any


# IMPORTANT: use `\s` (whitespace) rather than `\\s` (a literal backslash + 's').
_DURATION_RE = re.compile(r"\"duration\"\s*:\s*([0-9]+(?:\.[0-9]+)?)")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(_read_text(path))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _guess_manifest_path(run_dir: Path) -> Path | None:
    # Prefer the explicit training command file.
    cmd_path = run_dir / "training_command.txt"
    if cmd_path.is_file():
        m = re.search(r"\\bsharded_manifest_path=([^\\s]+)", _read_text(cmd_path))
        if m:
            p = m.group(1).strip().strip("'\"")
            return Path(p)

    # Fallback: config yaml often contains a `sharded_manifest_path:` line.
    cfg_path = run_dir / "config_base.yaml"
    if cfg_path.is_file():
        m = re.search(r"^\\s*sharded_manifest_path\\s*:\\s*(\\S+)\\s*$", _read_text(cfg_path), flags=re.M)
        if m:
            p = m.group(1).strip().strip("'\"")
            return Path(p)

    return None


def _manifest_parquet_files(manifest_path: Path) -> list[Path]:
    obj = _read_json(manifest_path)
    shards = obj.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Invalid manifest: missing `shards` list in {manifest_path}")

    out: list[Path] = []
    manifest_dir = manifest_path.parent
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        files = shard.get("files")
        if isinstance(files, str):
            files = [files]
        if not isinstance(files, list):
            continue
        for rel in files:
            if not isinstance(rel, str) or not rel:
                continue
            p = Path(rel)
            if not p.is_absolute():
                p = (manifest_dir / p).resolve()
            out.append(p)

    # De-dup while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _dataset_expected_files_from_dataset_info(dataset_dir: Path, dataset_names: list[str]) -> list[str]:
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        return []
    try:
        info = _read_json(info_path)
    except Exception:
        return []

    out: list[Path] = []
    for name in dataset_names:
        entry = info.get(name)
        if not isinstance(entry, dict):
            continue
        file_name = entry.get("file_name")
        if isinstance(file_name, str):
            parts = [file_name]
        elif isinstance(file_name, list):
            parts = [str(x) for x in file_name if isinstance(x, str) and x]
        else:
            parts = []

        for part in parts:
            root = (dataset_dir / part).resolve()
            try:
                st = os.stat(root)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                out.append(root)
                continue
            if stat.S_ISDIR(st.st_mode):
                try:
                    names = sorted(os.listdir(root))
                except OSError:
                    continue
                for fn in names:
                    p = root / fn
                    try:
                        st2 = os.stat(p)
                    except OSError:
                        continue
                    if stat.S_ISREG(st2.st_mode):
                        out.append(p)

    # De-dup while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return [str(p) for p in uniq]


def _parquet_total_duration_sec(
    parquet_files: list[Path],
    *,
    log_every: int = 10,
    limit_shards: int | None = None,
) -> tuple[float, int]:
    import polars as pl

    files = parquet_files if limit_shards is None else parquet_files[: int(limit_shards)]
    if not files:
        return 0.0, 0

    # Duration per row = sum(duration in each audio JSON string).
    row_dur = (
        pl.col("audio")
        .list.eval(pl.element().str.extract(_DURATION_RE.pattern, 1).cast(pl.Float64, strict=False))
        .list.sum()
        .fill_null(0.0)
    )

    total_sec = 0.0
    total_rows = 0
    t0 = time.time()
    n = len(files)

    for i, path in enumerate(files, start=1):
        # `scan_parquet(..., columns=...)` is not supported in some Polars versions;
        # projection pushdown will still ensure only `audio` is read.
        lf = pl.scan_parquet(str(path))
        res = lf.select(row_dur.sum().alias("duration_sec"), pl.len().alias("rows")).collect(streaming=True)
        shard_sec = float(res["duration_sec"][0] or 0.0)
        shard_rows = int(res["rows"][0] or 0)

        total_sec += shard_sec
        total_rows += shard_rows

        if i == 1 or i == n or (log_every > 0 and i % int(log_every) == 0):
            elapsed = max(1e-6, time.time() - t0)
            rate = float(i) / elapsed
            eta_sec = float(n - i) / max(1e-9, rate)
            print(
                f"[shards {i:4d}/{n}] +{shard_sec/3600.0:8.2f}h "
                f"total={total_sec/3600.0:10.2f}h rows={total_rows:,} "
                f"eta={eta_sec/60.0:6.1f}m file={path.name}",
                flush=True,
            )

    return float(total_sec), int(total_rows)


def _touch_up_cache_entries(
    cache_obj: dict[str, Any],
    *,
    epsilon_sec: float,
    expected_paths: list[str] | None = None,
) -> None:
    files = cache_obj.get("files")
    if not isinstance(files, dict):
        files = {}
        cache_obj["files"] = files

    if expected_paths:
        for path_str in expected_paths:
            if not isinstance(path_str, str) or not path_str or path_str in files:
                continue
            try:
                st = os.stat(path_str)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            files[path_str] = {
                "path": path_str,
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
                "md5": None,
                "num_lines": 0,
                "duration_sec": float(epsilon_sec) if int(st.st_size) > 0 else 0.0,
                "has_audio": False if "/text_data/" in path_str else True,
            }

    for path_str, entry in list(files.items()):
        if not isinstance(path_str, str) or not isinstance(entry, dict):
            continue

        entry["path"] = path_str
        try:
            st = os.stat(path_str)
        except OSError:
            # Keep existing fingerprint if the file is missing; the trainer only fingerprints existing files.
            st = None

        if st is not None:
            entry["size"] = int(st.st_size)
            entry["mtime"] = int(st.st_mtime)

        try:
            size = int(entry.get("size", 0) or 0)
        except Exception:
            size = 0
        try:
            dur = float(entry.get("duration_sec") or 0.0)
        except Exception:
            dur = 0.0

        # Compatibility: older trainers require dur>0 for every non-empty file.
        if size > 0 and not (dur > 0):
            entry["duration_sec"] = float(epsilon_sec)

        # Add `has_audio` for newer trainer validation logic (safe to include; ignored by older trainers).
        if "has_audio" not in entry:
            entry["has_audio"] = False if "/text_data/" in path_str else True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Trainer output_dir containing audio_duration_cache.json")
    ap.add_argument(
        "--manifest-path",
        default="",
        help="Sharded manifest.json path. If omitted, try to parse from training_command.txt/config_base.yaml.",
    )
    ap.add_argument("--log-every", type=int, default=10, help="Print progress every N shards (0 disables).")
    ap.add_argument("--limit-shards", type=int, default=0, help="Debug: only scan first N shards (0=all).")
    ap.add_argument("--epsilon-sec", type=float, default=1e-6, help="Fallback duration for non-empty files with 0.")
    ap.add_argument("--dry-run", action="store_true", help="Compute total hours but do not write cache.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cache_path = run_dir / "audio_duration_cache.json"
    if not cache_path.is_file():
        raise FileNotFoundError(str(cache_path))

    manifest_path = Path(args.manifest_path).expanduser().resolve() if args.manifest_path else None
    if manifest_path is None:
        manifest_path = _guess_manifest_path(run_dir)
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError("Cannot find sharded manifest.json (pass --manifest-path).")

    parquet_files = _manifest_parquet_files(manifest_path)
    if not parquet_files:
        raise ValueError(f"No parquet shard files found in manifest: {manifest_path}")

    limit = int(args.limit_shards) if int(args.limit_shards) > 0 else None
    total_sec, total_rows = _parquet_total_duration_sec(parquet_files, log_every=int(args.log_every), limit_shards=limit)
    print(f"[done] total_hours={total_sec/3600.0:.2f} rows={total_rows:,} shards={len(parquet_files)}", flush=True)

    if args.dry_run:
        return 0

    cache_obj = _read_json(cache_path)
    cache_obj["version"] = cache_obj.get("version") or 2
    cache_obj["generated_at"] = int(time.time())
    cache_obj["scan_method"] = "parquet_audio_regex"
    cache_obj["total_duration_sec"] = float(total_sec)
    dataset_dir = cache_obj.get("dataset_dir")
    dataset_names = cache_obj.get("datasets")
    expected_paths = None
    if isinstance(dataset_dir, str) and isinstance(dataset_names, list):
        expected_paths = _dataset_expected_files_from_dataset_info(Path(dataset_dir), [str(x) for x in dataset_names])
    _touch_up_cache_entries(cache_obj, epsilon_sec=float(args.epsilon_sec), expected_paths=expected_paths)

    _atomic_write_json(cache_path, cache_obj)
    print(f"[write] {cache_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
