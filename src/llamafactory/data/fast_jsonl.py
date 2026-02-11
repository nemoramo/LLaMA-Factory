# Copyright 2026 the LlamaFactory team.
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

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class FastJsonlToParquetArgs:
    enabled: bool
    min_total_bytes: int
    cache_dir: str
    compression: str
    force_rebuild: bool = False
    infer_schema_length: int | None = None


def _try_import_polars():
    try:
        import polars as pl  # type: ignore

        return pl
    except Exception:
        return None


def _file_fingerprint(path: str) -> dict[str, Any]:
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def _hash_key(parts: Iterable[str]) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _atomic_write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def maybe_convert_jsonl_files_to_parquet(
    *,
    data_files: list[str],
    args: FastJsonlToParquetArgs,
    logger: Any,
) -> Optional[list[str]]:
    """Convert local jsonl shards to parquet once and reuse in subsequent runs.

    Returns:
      - list of parquet files if conversion is used (cache hit or rebuilt)
      - None if conversion is disabled or failed (caller should fallback to HF json loader)
    """
    if not args.enabled:
        return None

    jsonl_files = [p for p in data_files if os.path.splitext(p)[-1].lower() in (".jsonl",)]
    if len(jsonl_files) != len(data_files):
        # Keep behavior simple: only accelerate pure-jsonl datasets.
        return None

    total_bytes = 0
    fps: list[dict[str, Any]] = []
    for p in jsonl_files:
        try:
            fp = _file_fingerprint(p)
        except OSError:
            return None
        fps.append(fp)
        total_bytes += int(fp["size"])

    if total_bytes < int(args.min_total_bytes):
        return None

    os.makedirs(args.cache_dir, exist_ok=True)

    key_parts = [f'{x["path"]}:{x["size"]}:{x["mtime_ns"]}' for x in fps]
    key = _hash_key(sorted(key_parts))
    out_dir = os.path.join(args.cache_dir, f"jsonl_parquet_{key[:16]}")
    done_path = os.path.join(out_dir, "done.json")
    lock_path = os.path.join(out_dir, ".build.lock")

    def _parquet_name(src_fp: dict[str, Any]) -> str:
        base = os.path.basename(str(src_fp["path"]))
        base = base[:-5] if base.endswith(".jsonl") else base
        fph = _hash_key([f'{src_fp["path"]}:{src_fp["size"]}:{src_fp["mtime_ns"]}'])[:10]
        return os.path.join(out_dir, f"{base}.{fph}.parquet")

    parquet_files = [_parquet_name(fp) for fp in fps]

    # Cache hit (and not forced)
    if (not args.force_rebuild) and os.path.isfile(done_path) and all(os.path.isfile(p) for p in parquet_files):
        logger.info_rank0(
            "Fast JSONL: cache hit -> parquet (%d shards, total=%.2f GB): %s",
            len(parquet_files),
            float(total_bytes) / (1024.0**3),
            out_dir,
        )
        return parquet_files

    os.makedirs(out_dir, exist_ok=True)

    # Acquire a coarse lock for this dataset key so concurrent runs don't rebuild.
    lock_fd = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(
            lock_fd,
            f"pid={os.getpid()} host={socket.gethostname()} time={int(time.time())}\n".encode("utf-8"),
        )
    except FileExistsError:
        lock_fd = None

    if lock_fd is None:
        # Someone else is building. Wait for done.json (or timeout).
        logger.info_rank0("Fast JSONL: waiting for parquet build lock: %s", out_dir)
        deadline = time.time() + 24 * 3600
        while time.time() < deadline:
            if os.path.isfile(done_path) and all(os.path.isfile(p) for p in parquet_files):
                logger.info_rank0("Fast JSONL: lock released, using parquet cache: %s", out_dir)
                return parquet_files
            time.sleep(2.0)
        logger.warning_rank0("Fast JSONL: timed out waiting for parquet cache; falling back to HF json loader.")
        return None

    try:
        pl = _try_import_polars()
        if pl is None:
            logger.warning_rank0("Fast JSONL: polars is not available; falling back to HF json loader.")
            return None

        t0 = time.time()
        logger.info_rank0(
            "Fast JSONL: building parquet cache (%d shards, total=%.2f GB): %s",
            len(jsonl_files),
            float(total_bytes) / (1024.0**3),
            out_dir,
        )

        infer_len = args.infer_schema_length
        for fp, src_path, out_path in zip(fps, jsonl_files, parquet_files):
            if (not args.force_rebuild) and os.path.isfile(out_path):
                continue

            scan_kwargs = {}
            if infer_len is not None:
                scan_kwargs["infer_schema_length"] = int(infer_len)

            # Polars -> Parquet (columnar) for faster future loads.
            lf = pl.scan_ndjson(src_path, **scan_kwargs)
            lf.sink_parquet(out_path, compression=args.compression, statistics=True)

        meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "total_bytes": int(total_bytes),
            "num_shards": int(len(jsonl_files)),
            "compression": str(args.compression),
            "polars_version": getattr(pl, "__version__", None),
            "sources": fps,
            "parquet_files": parquet_files,
        }
        _atomic_write_json(done_path, meta)
        logger.info_rank0("Fast JSONL: parquet cache ready in %.2fs: %s", time.time() - t0, out_dir)
        return parquet_files
    except Exception as err:  # noqa: BLE001
        logger.warning_rank0("Fast JSONL: failed to build parquet cache (%s); falling back to HF json loader.", err)
        return None
    finally:
        try:
            if lock_fd is not None:
                os.close(lock_fd)
        except Exception:
            pass
        # Best-effort: remove lock so other jobs can proceed even if we failed.
        try:
            if os.path.exists(lock_path):
                os.unlink(lock_path)
        except Exception:
            pass

