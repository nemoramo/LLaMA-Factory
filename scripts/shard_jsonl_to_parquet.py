#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import threading
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from queue import Empty as QueueEmpty

# Optional rich progress bar
try:
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


@dataclass(frozen=True)
class InputFingerprint:
    path: str
    size: int
    mtime_ns: int


def _fingerprint(path: str) -> InputFingerprint:
    st = os.stat(path)
    return InputFingerprint(
        path=os.path.abspath(path),
        size=int(st.st_size),
        mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    )


def _iter_inputs(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        p = os.path.expanduser(str(item))
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                fp = os.path.join(p, name)
                if os.path.isfile(fp):
                    out.append(fp)
        else:
            out.append(p)

    # Keep behavior explicit: only jsonl/ndjson are supported.
    out2: list[str] = []
    for p in out:
        ext = os.path.splitext(p)[-1].lower()
        if ext in (".jsonl", ".ndjson"):
            out2.append(p)
    return out2


def _stable_shard_id(key: str, *, seed: int, num_shards: int) -> int:
    """Fast stable hash for sharding (deterministic across processes/runs)."""
    if num_shards <= 0:
        return 0
    b = key.encode("utf-8", errors="ignore")
    h = zlib.crc32(b, seed & 0xFFFFFFFF) & 0xFFFFFFFF
    return int(h % int(num_shards))


def _split_files_by_size(files: list[str], num_workers: int) -> list[list[str]]:
    """Greedy bin-pack by file size to balance workers."""
    if num_workers <= 1:
        return [list(files)]

    sizes: list[tuple[int, str]] = []
    for f in files:
        try:
            sizes.append((int(os.stat(f).st_size), f))
        except OSError:
            sizes.append((0, f))
    sizes.sort(reverse=True)

    bins = [(0, i) for i in range(num_workers)]  # (total_size, worker_id)
    assignments: list[list[str]] = [[] for _ in range(num_workers)]
    for sz, f in sizes:
        bins.sort()
        total, wid = bins[0]
        assignments[wid].append(f)
        bins[0] = (total + sz, wid)

    return [a for a in assignments if a]


def _require_polars():
    try:
        import polars as pl  # type: ignore

        return pl
    except Exception as e:  # noqa: BLE001
        print(
            "polars is required for shard_jsonl_to_parquet.py. Install it in your env, e.g.\n"
            "  pip install -U polars\n"
            "or\n"
            "  conda install -c conda-forge polars\n"
            f"import error: {e!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _write_json(path: str, obj: object) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _collect_output_shards(out_dir: str) -> list[dict]:
    shards_dir = os.path.join(out_dir, "shards")
    files = sorted(glob.glob(os.path.join(shards_dir, "part-*.parquet")))
    by_shard: dict[int, list[str]] = {}
    by_shard_bytes: dict[int, int] = {}
    for fp in files:
        base = os.path.basename(fp)
        # part-00012.parquet or part-00012.001.parquet
        stem = base[len("part-") :]
        shard_str = stem.split(".", 1)[0]
        try:
            shard_id = int(shard_str)
        except Exception:
            continue
        by_shard.setdefault(shard_id, []).append(os.path.relpath(fp, out_dir))
        try:
            by_shard_bytes[shard_id] = by_shard_bytes.get(shard_id, 0) + int(os.stat(fp).st_size)
        except OSError:
            pass

    shards: list[dict] = []
    for shard_id in sorted(by_shard.keys()):
        shards.append(
            {
                "shard_id": int(shard_id),
                "files": by_shard[shard_id],
                "total_bytes": int(by_shard_bytes.get(shard_id, 0)),
            }
        )
    return shards


def _format_bytes(n: int) -> str:
    n = int(n)
    if n < 0:
        return f"{n}B"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    x = float(n)
    u = 0
    while x >= 1024.0 and u < len(units) - 1:
        x /= 1024.0
        u += 1
    if u == 0:
        return f"{int(x)}{units[u]}"
    return f"{x:.2f}{units[u]}"


def _read_proc_rss_kb() -> int | None:
    # Linux-only, best-effort; keeps logs self-contained without extra deps.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:  # noqa: BLE001
        return None
    return None


def _scan_shards_progress(shards_dir: str) -> dict[str, int]:
    shard_ids: set[int] = set()
    base = 0
    extra = 0
    total = 0
    zero = 0
    bytes_total = 0

    try:
        it = os.scandir(shards_dir)
    except FileNotFoundError:
        return {"shard_groups": 0, "base": 0, "extra": 0, "total": 0, "zero": 0, "bytes_total": 0}

    with it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if not (name.startswith("part-") and name.endswith(".parquet")):
                continue

            total += 1
            try:
                st = entry.stat()
                size = int(st.st_size)
            except OSError:
                size = 0
            bytes_total += size
            if size == 0:
                zero += 1

            core = name[len("part-") : -len(".parquet")]
            if "." in core:
                shard_str, part_str = core.split(".", 1)
                part = int(part_str) if part_str.isdigit() else 1
            else:
                shard_str = core
                part = 0

            if shard_str.isdigit():
                shard_id = int(shard_str)
                shard_ids.add(shard_id)
                if part == 0:
                    base += 1
                else:
                    extra += 1

    return {
        "shard_groups": len(shard_ids),
        "base": base,
        "extra": extra,
        "total": total,
        "zero": zero,
        "bytes_total": bytes_total,
    }


def _progress_bar(done: int, total: int, width: int) -> str:
    width = max(5, int(width))
    total = int(total)
    done = max(0, int(done))
    if total <= 0:
        return "[" + ("?" * width) + "]"
    done = min(done, total)
    filled = int(round(width * (done / total)))
    filled = min(width, max(0, filled))
    return "[" + ("#" * filled) + ("." * (width - filled)) + "]"


def _parse_approximate_bytes_per_file(value: str) -> int | str | None:
    s = str(value or "").strip().lower()
    if s in ("", "auto"):
        return "auto"
    if s in ("none", "null", "no", "0"):
        return None
    try:
        n = int(s)
    except ValueError as e:
        raise SystemExit(f"--approximate-bytes-per-file must be int/auto/none, got {value!r}") from e
    if n <= 0:
        raise SystemExit(f"--approximate-bytes-per-file must be > 0, got {n}")
    return n


def _iter_ndjson_objects(
    inputs: list[str],
    *,
    ignore_errors: bool,
    compute_shard: bool = False,
    seed: int = 42,
    num_shards: int = 1,
    id_col: str | None = None,
) -> Iterable[tuple[dict | None, int]]:
    try:
        import orjson  # type: ignore
    except Exception:  # noqa: BLE001
        orjson = None

    for path in inputs:
        path_abs = os.path.abspath(path)
        with open(path, "rb") as f:
            for row_idx, line in enumerate(f):
                raw_len = len(line)
                line = line.strip()
                if not line:
                    yield None, raw_len
                    continue
                try:
                    obj = orjson.loads(line) if orjson is not None else json.loads(line)  # type: ignore[arg-type]
                except Exception:
                    if ignore_errors:
                        yield None, raw_len
                        continue
                    raise
                if not isinstance(obj, dict):
                    if ignore_errors:
                        yield None, raw_len
                        continue
                    raise ValueError(f"Expected ndjson object (dict), got {type(obj)!r} in {path_abs}")
                if compute_shard:
                    if id_col and (id_col in obj) and (obj.get(id_col) is not None):
                        v = obj.get(id_col)
                        if isinstance(v, str):
                            key = v
                        else:
                            try:
                                key = json.dumps(v, ensure_ascii=False)
                            except Exception:
                                key = str(v)
                    else:
                        key = f"{path_abs}:{row_idx}"
                    obj["__shard"] = _stable_shard_id(key, seed=int(seed), num_shards=int(num_shards))
                else:
                    obj["__source_path"] = path_abs
                    obj["__row_index"] = int(row_idx)
                yield obj, raw_len


def _partition_chunk_worker(
    args: tuple,
) -> dict:
    """Worker function for parallel chunk processing.
    
    Returns a dict mapping shard_id -> list of temp file paths written.
    """
    (chunk_idx, rows, num_shards, seed, id_col, compression, statistics, 
     infer_schema_length, schema_obj, schema_cols, tmp_dir, progress_queue) = args
    
    if not rows:
        if progress_queue is not None:
            progress_queue.put(("chunk_done", chunk_idx, 0))
        return {}
    
    import polars as pl
    schema = dict(schema_obj) if not isinstance(schema_obj, dict) else schema_obj
    
    df = pl.from_dicts(rows, infer_schema_length=int(infer_schema_length))
    
    if "__shard" in df.columns:
        df = df.with_columns(pl.col("__shard").cast(pl.UInt32, strict=False))
    else:
        use_id_col = id_col and (id_col in df.columns)
        if use_id_col:
            key_expr = pl.col(id_col).cast(pl.Utf8, strict=False)
        else:
            key_expr = pl.concat_str([pl.col("__source_path"), pl.col("__row_index").cast(pl.Utf8)], separator=":")
        df = df.with_columns((key_expr.hash(seed=int(seed)) % int(num_shards)).cast(pl.UInt32).alias("__shard"))
    to_drop = [c for c in ("__source_path", "__row_index") if c in df.columns]
    if to_drop:
        df = df.drop(to_drop)
    
    # Normalize chunk schema
    for name, dtype in schema.items():
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(name))
    df = df.cast(schema, strict=False)
    df = df.select(["__shard"] + list(schema_cols))
    
    # Write each shard partition to separate temp files
    parts = df.partition_by("__shard", maintain_order=True, include_key=False, as_dict=True)
    result = {}
    
    for k, sub in parts.items():
        shard_id = int(k[0])
        if shard_id < 0 or shard_id >= int(num_shards):
            continue
        
        tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx:06d}_shard_{shard_id:05d}.parquet")
        sub.write_parquet(
            tmp_path,
            compression=compression,
            statistics=statistics,
        )
        result.setdefault(shard_id, []).append(tmp_path)
    
    # Report progress via queue
    if progress_queue is not None:
        progress_queue.put(("chunk_done", chunk_idx, len(rows)))
    
    return result


def _partition_write_chunk(
    pl,
    rows: list[dict],
    *,
    num_shards: int,
    seed: int,
    shards_dir: str,
    id_col: str | None,
    compression: str,
    statistics: bool,
    infer_schema_length: int,
    schema,
    schema_cols: list[str],
    arrow_schema,
    pq,
    writers: list,
) -> None:
    if not rows:
        return
    df = pl.from_dicts(rows, infer_schema_length=int(infer_schema_length))

    if "__shard" in df.columns:
        df = df.with_columns(pl.col("__shard").cast(pl.UInt32, strict=False))
    else:
        use_id_col = id_col and (id_col in df.columns)
        if use_id_col:
            key_expr = pl.col(id_col).cast(pl.Utf8, strict=False)
        else:
            key_expr = pl.concat_str([pl.col("__source_path"), pl.col("__row_index").cast(pl.Utf8)], separator=":")
        df = df.with_columns((key_expr.hash(seed=int(seed)) % int(num_shards)).cast(pl.UInt32).alias("__shard"))
    to_drop = [c for c in ("__source_path", "__row_index") if c in df.columns]
    if to_drop:
        df = df.drop(to_drop)

    # Normalize chunk schema so all shards share the same parquet schema.
    for name, dtype in schema.items():
        if name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(name))
    df = df.cast(schema, strict=False)
    df = df.select(["__shard"] + list(schema_cols))

    parts = df.partition_by("__shard", maintain_order=True, include_key=False, as_dict=True)
    for k, sub in parts.items():
        shard_id = int(k[0])
        if shard_id < 0 or shard_id >= int(num_shards):
            continue
        w = writers[shard_id]
        if w is None:
            out_path = os.path.join(shards_dir, f"part-{shard_id:05d}.parquet")
            w = pq.ParquetWriter(
                out_path,
                arrow_schema,
                compression=str(compression),
                write_statistics=bool(statistics),
            )
            writers[shard_id] = w
        w.write_table(sub.to_arrow())


def _merge_shard_files(
    shard_files: list[str],
    out_path: str,
    arrow_schema,
    pq,
    compression: str,
    statistics: bool,
) -> None:
    """Merge multiple parquet files into one.

    Important: do not concatenate all tables in-memory; stream one file at a time to avoid OOM.
    """
    if not shard_files:
        return
    
    if len(shard_files) == 1:
        os.replace(shard_files[0], out_path)
        return

    # Stream into a single output writer (one input file at a time).
    w = pq.ParquetWriter(
        out_path,
        arrow_schema,
        compression=str(compression),
        write_statistics=bool(statistics),
    )
    try:
        for f in sorted(shard_files):
            try:
                t = pq.read_table(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read temp parquet during merge: {f}") from e
            if t.schema != arrow_schema:
                try:
                    t = t.cast(arrow_schema, safe=False)
                except Exception as e:
                    raise RuntimeError(f"Temp parquet schema mismatch (cannot cast) for: {f}") from e
            w.write_table(t)
    finally:
        try:
            w.close()
        except Exception:
            pass
    
    # Cleanup temp files
    for f in shard_files:
        try:
            os.remove(f)
        except Exception:
            pass


def _file_parallel_worker(args: tuple) -> dict:
    (
        worker_id,
        files,
        num_shards,
        seed,
        id_col,
        compression,
        statistics,
        infer_schema_length,
        schema_obj,
        schema_cols,
        out_worker_dir,
        ignore_errors,
        chunk_rows,
        progress_queue,
        report_bytes,
    ) = args

    # Avoid oversubscribing CPU: we parallelize via processes, so keep polars single-threaded in workers.
    os.environ.setdefault("POLARS_MAX_THREADS", "1")

    import polars as pl
    import pyarrow.parquet as pq  # type: ignore
    schema = dict(schema_obj) if not isinstance(schema_obj, dict) else schema_obj
    empty_df = pl.DataFrame(schema=schema)
    arrow_schema = empty_df.to_arrow().schema

    shards_dir = os.path.join(out_worker_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    writers: list = [None for _ in range(int(num_shards))]

    bytes_read = 0
    rows_written = 0
    last_report = 0
    buf: list[dict] = []

    for obj, nbytes in _iter_ndjson_objects(
        list(files),
        ignore_errors=bool(ignore_errors),
        compute_shard=True,
        seed=int(seed),
        num_shards=int(num_shards),
        id_col=id_col,
    ):
        bytes_read += int(nbytes)
        if obj is None:
            continue
        buf.append(obj)
        if len(buf) < int(chunk_rows):
            continue
        _partition_write_chunk(
            pl,
            buf,
            num_shards=int(num_shards),
            seed=int(seed),
            shards_dir=shards_dir,
            id_col=id_col,
            compression=str(compression),
            statistics=bool(statistics),
            infer_schema_length=int(infer_schema_length),
            schema=schema,
            schema_cols=list(schema_cols),
            arrow_schema=arrow_schema,
            pq=pq,
            writers=writers,
        )
        rows_written += len(buf)
        buf.clear()

        if progress_queue is not None and (bytes_read - last_report) >= int(report_bytes):
            progress_queue.put((int(worker_id), int(bytes_read), int(rows_written)))
            last_report = bytes_read

    if buf:
        _partition_write_chunk(
            pl,
            buf,
            num_shards=int(num_shards),
            seed=int(seed),
            shards_dir=shards_dir,
            id_col=id_col,
            compression=str(compression),
            statistics=bool(statistics),
            infer_schema_length=int(infer_schema_length),
            schema=schema,
            schema_cols=list(schema_cols),
            arrow_schema=arrow_schema,
            pq=pq,
            writers=writers,
        )
        rows_written += len(buf)
        buf.clear()

    for w in writers:
        if w is not None:
            try:
                w.close()
            except Exception:
                pass

    if progress_queue is not None:
        progress_queue.put((int(worker_id), int(bytes_read), int(rows_written), "done"))

    shard_files = sorted(glob.glob(os.path.join(shards_dir, "part-*.parquet")))
    return {
        "worker_id": int(worker_id),
        "files": int(len(list(files))),
        "bytes_read": int(bytes_read),
        "rows_written": int(rows_written),
        "shard_files": shard_files,
        "out_worker_dir": out_worker_dir,
    }


def _cmd_build_file_parallel(args: argparse.Namespace, *, pl, inputs: list[str], out_dir: str, shards_dir: str) -> int:
    num_shards = int(args.num_shards)
    seed = int(args.seed)
    infer_len = int(args.infer_schema_length)
    compression = str(args.compression)
    statistics = not bool(args.no_statistics)
    chunk_rows = int(args.chunk_rows)
    if chunk_rows <= 0:
        raise SystemExit(f"--chunk-rows must be > 0, got {chunk_rows}")

    num_workers = int(args.num_workers) if hasattr(args, "num_workers") and args.num_workers else (os.cpu_count() or 8)
    num_workers = max(1, min(num_workers, len(inputs)))

    id_col = str(args.id_col or "").strip() or None

    # Import inside so importing LLaMA-Factory doesn't hard-require pyarrow.
    import pyarrow.parquet as pq  # type: ignore

    # Infer a stable global schema so each shard parquet file is consistent across workers.
    lf_schema = pl.scan_ndjson(
        inputs,
        infer_schema_length=int(infer_len),
        batch_size=int(args.batch_size),
        low_memory=bool(args.low_memory),
        ignore_errors=bool(args.ignore_errors),
    )
    schema = lf_schema.collect_schema()
    schema_cols = list(schema.keys())
    empty_df = pl.DataFrame(schema=schema)
    arrow_schema = empty_df.to_arrow().schema
    empty_table = empty_df.to_arrow()

    # Temp dir for per-worker shard outputs (limited fanout: num_workers * num_shards files).
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="shard_jsonl_file_parallel_", dir=out_dir)

    assignments = _split_files_by_size(inputs, num_workers)
    num_workers_eff = len(assignments)

    bytes_total = sum(int(os.stat(p).st_size) for p in inputs)
    progress_interval = float(args.progress_interval_sec)
    progress_width = int(args.progress_bar_width)

    # Progress aggregation.
    # Use a Manager queue so it can be pickled and sent to spawned processes.
    mp_ctx = mp.get_context("spawn")
    mgr = Manager() if progress_interval > 0 else None
    progress_queue = mgr.Queue() if mgr is not None else None
    bytes_by_worker = [0 for _ in range(num_workers_eff)]
    rows_by_worker = [0 for _ in range(num_workers_eff)]
    done_by_worker = [False for _ in range(num_workers_eff)]
    stop_evt = threading.Event()
    t0 = time.time()

    def drain_queue() -> None:
        if progress_queue is None:
            return
        while True:
            try:
                msg = progress_queue.get_nowait()
            except QueueEmpty:
                break
            if not isinstance(msg, tuple) or len(msg) < 3:
                continue
            wid = int(msg[0])
            if 0 <= wid < num_workers_eff:
                bytes_by_worker[wid] = int(msg[1])
                rows_by_worker[wid] = int(msg[2])
                if len(msg) >= 4 and msg[3] == "done":
                    done_by_worker[wid] = True

    def emit_progress(tag: str) -> None:
        drain_queue()
        bytes_read = sum(bytes_by_worker)
        rows = sum(rows_by_worker)
        done = sum(1 for x in done_by_worker if x)
        bar = _progress_bar(int(bytes_read), int(bytes_total), progress_width)
        rss_kb = _read_proc_rss_kb()
        rss = "?" if rss_kb is None else f"{rss_kb / 1024.0:.1f}MiB"
        elapsed = time.time() - t0
        pct = 0.0 if bytes_total <= 0 else (100.0 * float(bytes_read) / float(bytes_total))
        bps = 0.0 if elapsed <= 0 else (float(bytes_read) / float(elapsed))
        eta = None if bps <= 0.0 else max(0.0, (float(bytes_total) - float(bytes_read)) / bps)
        eta_s = "?" if eta is None else f"{eta:.0f}s"
        if eta is not None and eta >= 3600:
            eta_s = f"{eta / 3600.0:.2f}h"
        elif eta is not None and eta >= 60:
            eta_s = f"{eta / 60.0:.1f}m"
        print(
            f"[{tag}] {bar} {pct:5.1f}% bytes={_format_bytes(bytes_read)}/{_format_bytes(bytes_total)} "
            f"rows={rows} workers_done={done}/{num_workers_eff} rss={rss} elapsed={elapsed:.0f}s eta={eta_s}",
            file=sys.stderr,
            flush=True,
        )

    def progress_worker() -> None:
        emit_progress("PROGRESS")
        while not stop_evt.wait(progress_interval):
            emit_progress("PROGRESS")

    prog_thread: threading.Thread | None = None
    if progress_interval > 0:
        prog_thread = threading.Thread(target=progress_worker, daemon=True)
        prog_thread.start()

    report_bytes = int(getattr(args, "file_parallel_report_bytes", 512 * 1024 * 1024))
    report_bytes = max(64 * 1024 * 1024, report_bytes)

    # Stage1: parallel parse + write per-worker shard files.
    worker_results: list[dict] = []
    try:
        with ProcessPoolExecutor(max_workers=num_workers_eff, mp_context=mp_ctx) as ex:
            futs = []
            for wid, flist in enumerate(assignments):
                out_worker_dir = os.path.join(tmp_dir, f"worker_{wid:03d}")
                futs.append(
                    ex.submit(
                        _file_parallel_worker,
                        (
                            wid,
                            flist,
                            num_shards,
                            seed,
                            id_col,
                            compression,
                            statistics,
                            infer_len,
                            schema,
                            schema_cols,
                            out_worker_dir,
                            bool(args.ignore_errors),
                            chunk_rows,
                            progress_queue,
                            report_bytes,
                        ),
                    )
                )
            for fut in as_completed(futs):
                worker_results.append(fut.result())
    finally:
        stop_evt.set()
        if prog_thread is not None:
            prog_thread.join(timeout=max(1.0, progress_interval + 1.0))
        drain_queue()

    emit_progress("STAGE1_DONE")

    # Stage2: merge worker shard files into final shard files.
    shard_temp_files: dict[int, list[str]] = {i: [] for i in range(num_shards)}
    for r in worker_results:
        for fp in r.get("shard_files", []):
            base = os.path.basename(fp)
            if not base.startswith("part-") or not base.endswith(".parquet"):
                continue
            core = base[len("part-") : -len(".parquet")]
            if "." in core:
                core = core.split(".", 1)[0]
            if not core.isdigit():
                continue
            shard_id = int(core)
            if 0 <= shard_id < num_shards:
                shard_temp_files[shard_id].append(fp)

    print(f"[INFO] merging worker outputs into {num_shards} shards...", file=sys.stderr, flush=True)
    for shard_id in range(num_shards):
        files = shard_temp_files.get(shard_id, [])
        out_path = os.path.join(shards_dir, f"part-{shard_id:05d}.parquet")
        if files:
            _merge_shard_files(files, out_path, arrow_schema, pq, compression, statistics)
        else:
            pq.write_table(
                empty_table,
                out_path,
                compression=compression,
                write_statistics=statistics,
            )
        if shard_id % max(1, num_shards // 10) == 0 or shard_id % 10 == 0:
            print(f"[MERGE] {shard_id + 1}/{num_shards}", file=sys.stderr, flush=True)

    # Cleanup temp dir
    try:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    elapsed = time.time() - t0
    emit_progress("DONE")

    fps = [_fingerprint(p) for p in inputs]
    build_meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "seed": int(seed),
        "num_shards": int(num_shards),
        "writer": "chunked_file_parallel",
        "chunk_rows": int(chunk_rows),
        "num_workers": int(num_workers_eff),
        "schema_cols": int(len(schema_cols)),
        "inputs": [asdict(x) for x in fps],
        "polars_version": getattr(pl, "__version__", None),
        "compression": str(compression),
        "statistics": bool(statistics),
        "infer_schema_length": infer_len,
        "batch_size": int(args.batch_size),
        "low_memory": bool(args.low_memory),
        "ignore_errors": bool(args.ignore_errors),
        "elapsed_sec": float(elapsed),
    }
    _write_json(os.path.join(out_dir, "build_meta.json"), build_meta)

    shards_existing = _collect_output_shards(out_dir)
    by_id = {int(x["shard_id"]): x for x in shards_existing}
    missing = 0
    shards: list[dict] = []
    for shard_id in range(num_shards):
        s = by_id.get(shard_id)
        if s is None:
            missing += 1
            shards.append({"shard_id": int(shard_id), "files": [], "total_bytes": 0})
        else:
            shards.append(s)
    if missing:
        print(f"[WARN] {missing}/{num_shards} shard_ids have no output files.", file=sys.stderr, flush=True)
    manifest = {
        "version": 1,
        "created_at": build_meta["created_at"],
        "seed": int(seed),
        "num_shards": int(num_shards),
        "backend": "chunked_file_parallel",
        "shards": shards,
    }
    _write_json(os.path.join(out_dir, "manifest.json"), manifest)

    total_files = sum(len(x.get("files", [])) for x in shards)
    print(
        f"OK: wrote {len(shards)} shard groups ({total_files} parquet files) to {out_dir} in {elapsed:.2f}s",
        file=sys.stderr,
    )
    return 0


def _cmd_build_chunked(args: argparse.Namespace, *, pl, inputs: list[str], out_dir: str, shards_dir: str) -> int:
    num_shards = int(args.num_shards)
    seed = int(args.seed)
    infer_len = int(args.infer_schema_length)
    compression = str(args.compression)
    statistics = not bool(args.no_statistics)
    chunk_rows = int(args.chunk_rows)
    if chunk_rows <= 0:
        raise SystemExit(f"--chunk-rows must be > 0, got {chunk_rows}")
    
    num_workers = int(args.num_workers) if hasattr(args, "num_workers") and args.num_workers else 1
    if num_workers < 1:
        num_workers = 1

    id_col = str(args.id_col or "").strip() or None
    # Import inside so importing LLaMA-Factory doesn't hard-require pyarrow.
    import pyarrow.parquet as pq  # type: ignore

    # Infer a stable global schema so each shard parquet file is consistent across chunks.
    # (Otherwise, a shard that starts with "text-only" rows could miss audio columns forever.)
    lf_schema = pl.scan_ndjson(
        inputs,
        infer_schema_length=int(infer_len),
        batch_size=int(args.batch_size),
        low_memory=bool(args.low_memory),
        ignore_errors=bool(args.ignore_errors),
    )
    schema = lf_schema.collect_schema()
    schema_cols = list(schema.keys())
    empty_df = pl.DataFrame(schema=schema)
    arrow_schema = empty_df.to_arrow().schema
    empty_table = empty_df.to_arrow()
    
    tmp_dir = None
    shard_temp_files: dict[int, list[str]] | None = None
    if num_workers > 1:
        # Temp directory for parallel chunk outputs.
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="shard_jsonl_tmp_", dir=out_dir)
        shard_temp_files = {i: [] for i in range(num_shards)}

    t0 = time.time()
    last_progress = 0.0
    rows_total = 0
    chunk_idx = 0  # processed chunk count
    bytes_total = sum(int(os.stat(p).st_size) for p in inputs)
    bytes_read = 0
    objs_seen = 0
    next_chunk_id = 0  # unique id for chunk/temp naming
    executor: ProcessPoolExecutor | None = None
    if num_workers > 1:
        # Keep a single executor alive for the entire build to avoid repeatedly spawning processes.
        executor = ProcessPoolExecutor(max_workers=num_workers)

    def emit_progress(tag: str) -> None:
        nonlocal last_progress
        st = _scan_shards_progress(shards_dir)
        rss_kb = _read_proc_rss_kb()
        bar = _progress_bar(int(bytes_read), int(bytes_total), int(args.progress_bar_width))
        rss = "?" if rss_kb is None else f"{rss_kb / 1024.0:.1f}MiB"
        elapsed = time.time() - t0
        pct = 0.0 if bytes_total <= 0 else (100.0 * float(bytes_read) / float(bytes_total))
        bps = 0.0 if elapsed <= 0 else (float(bytes_read) / float(elapsed))
        eta = None if bps <= 0.0 else max(0.0, (float(bytes_total) - float(bytes_read)) / bps)
        eta_s = "?" if eta is None else f"{eta:.0f}s"
        if eta is not None and eta >= 3600:
            eta_s = f"{eta / 3600.0:.2f}h"
        elif eta is not None and eta >= 60:
            eta_s = f"{eta / 60.0:.1f}m"
        worker_info = f" workers={num_workers}" if num_workers > 1 else ""
        print(
            f"[{tag}] {bar} {pct:5.1f}% bytes={_format_bytes(bytes_read)}/{_format_bytes(bytes_total)} "
            f"rows={rows_total} seen={objs_seen} chunks={chunk_idx}{worker_info} "
            f"shard_groups={st['shard_groups']}/{num_shards} "
            f"base={st['base']} extra={st['extra']} files={st['total']} zeroB={st['zero']} "
            f"size={_format_bytes(st['bytes_total'])} rss={rss} elapsed={elapsed:.0f}s eta={eta_s}",
            file=sys.stderr,
            flush=True,
        )
        last_progress = time.time()
    
    def process_pending_chunks(pending_chunks: list[tuple[int, list[dict]]], progress=None, task_id=None, use_rich=False) -> None:
        """Process pending chunks using multiprocessing (num_workers > 1).

        Writes per-chunk/per-shard temp parquet files into tmp_dir and records them in shard_temp_files.
        """
        nonlocal rows_total, chunk_idx
        if not pending_chunks:
            return
        assert num_workers > 1
        assert tmp_dir is not None
        assert shard_temp_files is not None
        assert executor is not None

        work_args = [
            (idx, rows, num_shards, seed, id_col, compression, statistics, infer_len, schema, schema_cols, tmp_dir, None)
            for idx, rows in pending_chunks
        ]
        rows_in_batch = 0
        for arg in work_args:
            rows_in_batch += len(arg[1])

        futures = [executor.submit(_partition_chunk_worker, arg) for arg in work_args]
        for f in as_completed(futures):
            try:
                result = f.result()
            except Exception as e:
                raise RuntimeError(f"Chunk processing failed (num_workers={num_workers})") from e
            for shard_id, files in result.items():
                shard_temp_files[shard_id].extend(files)

        rows_total += rows_in_batch
        chunk_idx += len(pending_chunks)
        pending_chunks.clear()

    print(
        "[INFO] shard_jsonl_to_parquet build(chunked):"
        f" inputs={len(inputs)}"
        f" output={out_dir}"
        f" num_shards={num_shards}"
        f" seed={seed}"
        f" compression={compression}"
        f" statistics={statistics}"
        f" infer_schema_length={infer_len}"
        f" chunk_rows={chunk_rows}"
        f" num_workers={num_workers}"
        f" low_memory={bool(args.low_memory)}"
        f" ignore_errors={bool(args.ignore_errors)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[INFO] inputs_total_bytes={_format_bytes(bytes_total)}",
        file=sys.stderr,
        flush=True,
    )

    progress_interval = float(args.progress_interval_sec)
    buf: list[dict] = []
    pending_chunks: list[tuple[int, list[dict]]] = []

    # Determine if we can use rich progress bar.
    use_rich = bool(getattr(args, "use_rich", False)) and _HAS_RICH and num_workers > 1
    
    if num_workers == 1:
        # True streaming single-process writer.
        emit_progress("START")
        writers: list[object | None] = [None for _ in range(num_shards)]
        try:
            for obj, nbytes in _iter_ndjson_objects(inputs, ignore_errors=bool(args.ignore_errors)):
                bytes_read += int(nbytes)
                if obj is None:
                    continue
                objs_seen += 1
                buf.append(obj)
                if len(buf) < chunk_rows:
                    continue
                _partition_write_chunk(
                    pl,
                    buf,
                    num_shards=num_shards,
                    seed=seed,
                    shards_dir=shards_dir,
                    id_col=id_col,
                    compression=compression,
                    statistics=statistics,
                    infer_schema_length=infer_len,
                    schema=schema,
                    schema_cols=schema_cols,
                    arrow_schema=arrow_schema,
                    pq=pq,
                    writers=writers,
                )
                rows_total += len(buf)
                chunk_idx += 1
                buf.clear()

                if progress_interval > 0 and (time.time() - last_progress) >= progress_interval:
                    emit_progress("PROGRESS")

            if buf:
                _partition_write_chunk(
                    pl,
                    buf,
                    num_shards=num_shards,
                    seed=seed,
                    shards_dir=shards_dir,
                    id_col=id_col,
                    compression=compression,
                    statistics=statistics,
                    infer_schema_length=infer_len,
                    schema=schema,
                    schema_cols=schema_cols,
                    arrow_schema=arrow_schema,
                    pq=pq,
                    writers=writers,
                )
                rows_total += len(buf)
                chunk_idx += 1
                buf.clear()
        finally:
            for w in writers:
                if w is not None:
                    try:
                        w.close()
                    except Exception:
                        pass

        # Ensure empty shards are created.
        for shard_id in range(num_shards):
            out_path = os.path.join(shards_dir, f"part-{shard_id:05d}.parquet")
            if not os.path.exists(out_path):
                pq.write_table(
                    empty_table,
                    out_path,
                    compression=compression,
                    write_statistics=statistics,
                )
    elif use_rich:
        # Rich progress bar version
        console = Console(stderr=True)
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            TextColumn("• {task.fields[rows]} rows • {task.fields[chunks]} chunks"),
            console=console,
            refresh_per_second=4,
        ) as progress:
            # Main processing task
            task = progress.add_task(
                "[cyan]Processing JSONL...",
                total=None,  # Unknown total initially
                rows=0,
                chunks=0,
            )
            
            try:
                for obj, nbytes in _iter_ndjson_objects(
                    inputs,
                    ignore_errors=bool(args.ignore_errors),
                    compute_shard=True,
                    seed=seed,
                    num_shards=num_shards,
                    id_col=id_col,
                ):
                    bytes_read += int(nbytes)
                    if obj is None:
                        continue
                    objs_seen += 1
                    buf.append(obj)
                    if len(buf) < chunk_rows:
                        continue
                    pending_chunks.append((next_chunk_id, buf))
                    next_chunk_id += 1
                    buf = []
                    
                    # Process when we have enough chunks for all workers
                    if len(pending_chunks) >= num_workers * 2:
                        process_pending_chunks(pending_chunks, progress, task, use_rich=True)
                        progress.update(task, chunks=chunk_idx, rows=rows_total)
                
                if buf:
                    pending_chunks.append((next_chunk_id, buf))
                    next_chunk_id += 1
                    buf = []
                
                # Process remaining chunks
                if pending_chunks:
                    process_pending_chunks(pending_chunks, progress, task, use_rich=True)
                    progress.update(task, chunks=chunk_idx, rows=rows_total)
                
                progress.update(task, description="[green]Processing done")
                
                # Merge phase
                if num_workers > 1:
                    total_temp_files = sum(len(v) for v in shard_temp_files.values())
                    merge_task = progress.add_task(
                        "[yellow]Merging shards...",
                        total=num_shards,
                        rows=total_temp_files,
                        chunks=0,
                    )
                    
                    for shard_id in range(num_shards):
                        files = shard_temp_files.get(shard_id, []) if shard_temp_files is not None else []
                        out_path = os.path.join(shards_dir, f"part-{shard_id:05d}.parquet")
                        if files:
                            _merge_shard_files(files, out_path, arrow_schema, pq, compression, statistics)
                        else:
                            pq.write_table(
                                empty_table,
                                out_path,
                                compression=compression,
                                write_statistics=statistics,
                            )
                        progress.update(merge_task, advance=1)
                    
                    progress.update(merge_task, description="[green]Merging done")
                    
                    # Cleanup temp dir
                    if tmp_dir is not None:
                        try:
                            import shutil
                            shutil.rmtree(tmp_dir)
                        except Exception:
                            pass
            except Exception:
                # Cleanup on error
                if tmp_dir is not None:
                    try:
                        import shutil
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass
                raise
    else:
        # Original text-based progress
        emit_progress("START")
        
        try:
            for obj, nbytes in _iter_ndjson_objects(
                inputs,
                ignore_errors=bool(args.ignore_errors),
                compute_shard=True,
                seed=seed,
                num_shards=num_shards,
                id_col=id_col,
            ):
                bytes_read += int(nbytes)
                if obj is None:
                    continue
                objs_seen += 1
                buf.append(obj)
                if len(buf) < chunk_rows:
                    continue
                pending_chunks.append((next_chunk_id, buf))
                next_chunk_id += 1
                buf = []

                # Process when we have enough chunks for all workers
                if len(pending_chunks) >= num_workers * 2:
                    process_pending_chunks(pending_chunks)

                if progress_interval > 0 and (time.time() - last_progress) >= progress_interval:
                    emit_progress("PROGRESS")

            if buf:
                pending_chunks.append((next_chunk_id, buf))
                next_chunk_id += 1
                buf = []
            
            # Process remaining chunks
            if pending_chunks:
                process_pending_chunks(pending_chunks)

            # Merge temp files or handle single-worker case
            if num_workers > 1:
                total_temp_files = sum(len(v) for v in shard_temp_files.values())
                print(f"[INFO] Merging {total_temp_files} temp files into {num_shards} shards...", 
                      file=sys.stderr, flush=True)
                for shard_id in range(num_shards):
                    files = shard_temp_files.get(shard_id, []) if shard_temp_files is not None else []
                    out_path = os.path.join(shards_dir, f"part-{shard_id:05d}.parquet")
                    if files:
                        _merge_shard_files(files, out_path, arrow_schema, pq, compression, statistics)
                    else:
                        # Empty shard
                        pq.write_table(
                            empty_table,
                            out_path,
                            compression=compression,
                            write_statistics=statistics,
                        )
                    # Progress every 10% or every 10 shards
                    if shard_id % max(1, num_shards // 10) == 0 or shard_id % 10 == 0:
                        emit_progress(f"MERGE {shard_id + 1}/{num_shards}")
                
                # Cleanup temp dir
                if tmp_dir is not None:
                    try:
                        import shutil
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass

        except Exception:
            # Cleanup on error
            if tmp_dir is not None:
                try:
                    import shutil
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            raise

    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
        executor = None

    elapsed = time.time() - t0
    if not use_rich:
        emit_progress("DONE")

    fps = [_fingerprint(p) for p in inputs]
    build_meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "seed": int(seed),
        "num_shards": int(num_shards),
        "writer": "chunked_parquetwriter",
        "chunk_rows": int(chunk_rows),
        "num_workers": int(num_workers),
        "schema_cols": int(len(schema_cols)),
        "inputs": [asdict(x) for x in fps],
        "polars_version": getattr(pl, "__version__", None),
        "compression": str(compression),
        "statistics": bool(statistics),
        "infer_schema_length": infer_len,
        "batch_size": int(args.batch_size),
        "low_memory": bool(args.low_memory),
        "ignore_errors": bool(args.ignore_errors),
        "elapsed_sec": float(elapsed),
        "rows_total": int(rows_total),
    }
    _write_json(os.path.join(out_dir, "build_meta.json"), build_meta)

    shards_existing = _collect_output_shards(out_dir)
    by_id = {int(x["shard_id"]): x for x in shards_existing}
    missing = 0
    shards: list[dict] = []
    for shard_id in range(num_shards):
        s = by_id.get(shard_id)
        if s is None:
            missing += 1
            shards.append({"shard_id": int(shard_id), "files": [], "total_bytes": 0})
        else:
            shards.append(s)
    if missing:
        print(f"[WARN] {missing}/{num_shards} shard_ids have no output files.", file=sys.stderr, flush=True)
    manifest = {
        "version": 1,
        "created_at": build_meta["created_at"],
        "seed": int(seed),
        "num_shards": int(num_shards),
        "backend": "chunked_parquetwriter",
        "shards": shards,
    }
    _write_json(os.path.join(out_dir, "manifest.json"), manifest)

    total_files = sum(len(x.get("files", [])) for x in shards)
    print(
        f"OK: wrote {len(shards)} shard groups ({total_files} parquet files) to {out_dir} in {elapsed:.2f}s",
        file=sys.stderr,
        flush=True,
    )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    pl = _require_polars()
    inputs = _iter_inputs(list(args.input or []))
    if not inputs:
        raise SystemExit("No .jsonl/.ndjson inputs found.")

    out_dir = os.path.abspath(os.path.expanduser(str(args.output)))
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    w = str(getattr(args, "writer", "partitionby"))
    if w == "chunked":
        return _cmd_build_chunked(args, pl=pl, inputs=inputs, out_dir=out_dir, shards_dir=shards_dir)
    if w == "file_parallel":
        return _cmd_build_file_parallel(args, pl=pl, inputs=inputs, out_dir=out_dir, shards_dir=shards_dir)

    num_shards = int(args.num_shards)
    if num_shards <= 0:
        raise SystemExit("--num-shards must be > 0")

    seed = int(args.seed)
    infer_len = int(args.infer_schema_length)
    batch_size = int(args.batch_size)
    compression = str(args.compression)
    statistics = not bool(args.no_statistics)
    approximate_bytes_per_file = _parse_approximate_bytes_per_file(str(args.approximate_bytes_per_file))
    max_rows_per_file = None if args.max_rows_per_file is None else int(args.max_rows_per_file)

    print(
        "[INFO] shard_jsonl_to_parquet build:"
        f" inputs={len(inputs)}"
        f" output={out_dir}"
        f" num_shards={num_shards}"
        f" seed={seed}"
        f" compression={compression}"
        f" statistics={statistics}"
        f" approx_bytes_per_file={approximate_bytes_per_file}"
        f" max_rows_per_file={max_rows_per_file}"
        f" infer_schema_length={infer_len}"
        f" batch_size={batch_size}"
        f" low_memory={bool(args.low_memory)}"
        f" ignore_errors={bool(args.ignore_errors)}",
        file=sys.stderr,
        flush=True,
    )

    # Scan all inputs as one lazy frame, but shard assignment is deterministic per row.
    lf = pl.scan_ndjson(
        inputs,
        infer_schema_length=infer_len,
        batch_size=batch_size,
        low_memory=bool(args.low_memory),
        ignore_errors=bool(args.ignore_errors),
        include_file_paths="__source_path",
        row_index_name="__row_index",
    )

    schema = {}
    try:
        schema = dict(lf.schema)
    except Exception:  # noqa: BLE001
        schema = {}

    id_col = str(args.id_col or "").strip() or None
    if id_col and id_col not in schema:
        print(f"[WARN] id_col={id_col!r} not in inferred schema; falling back to file+row index.", file=sys.stderr)
        id_col = None

    if id_col:
        key_expr = pl.col(id_col).cast(pl.Utf8, strict=False)
    else:
        key_expr = pl.concat_str([pl.col("__source_path"), pl.col("__row_index").cast(pl.Utf8)], separator=":")

    lf = lf.with_columns((key_expr.hash(seed=seed) % num_shards).cast(pl.UInt32).alias("__shard"))
    # Keep outputs clean; `__shard` itself can be excluded from data files via include_key=False.
    lf = lf.drop(["__source_path", "__row_index"])

    from polars.io.partition import FileProviderArgs, PartitionBy

    def file_path_provider(a: FileProviderArgs):
        shard_id = int(a.partition_keys["__shard"][0])
        part = int(a.index_in_partition)
        if part == 0:
            name = f"part-{shard_id:05d}.parquet"
        else:
            name = f"part-{shard_id:05d}.{part:03d}.parquet"
        return os.path.join(shards_dir, name)

    pb = PartitionBy(
        shards_dir,
        key="__shard",
        include_key=False,
        file_path_provider=file_path_provider,
        max_rows_per_file=max_rows_per_file,
        approximate_bytes_per_file=approximate_bytes_per_file,
    )

    t0 = time.time()
    progress_interval = float(args.progress_interval_sec)
    progress_width = int(args.progress_bar_width)
    stop_evt = threading.Event()

    def emit_progress(tag: str) -> None:
        st = _scan_shards_progress(shards_dir)
        rss_kb = _read_proc_rss_kb()
        bar = _progress_bar(int(st["shard_groups"]), num_shards, progress_width)
        rss = "?" if rss_kb is None else f"{rss_kb / 1024.0:.1f}MiB"
        elapsed = time.time() - t0
        print(
            f"[{tag}] {bar} shard_groups={st['shard_groups']}/{num_shards} "
            f"base={st['base']} extra={st['extra']} files={st['total']} zeroB={st['zero']} "
            f"size={_format_bytes(st['bytes_total'])} rss={rss} elapsed={elapsed:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    def progress_worker() -> None:
        emit_progress("PROGRESS")
        while not stop_evt.wait(progress_interval):
            emit_progress("PROGRESS")

    prog_thread: threading.Thread | None = None
    if progress_interval > 0:
        prog_thread = threading.Thread(target=progress_worker, daemon=True)
        prog_thread.start()

    try:
        lf.sink_parquet(
            pb,
            compression=compression,
            statistics=bool(statistics),
            mkdir=True,
        )
    finally:
        stop_evt.set()
        if prog_thread is not None:
            prog_thread.join(timeout=max(1.0, progress_interval + 1.0))

    elapsed = time.time() - t0

    fps = [_fingerprint(p) for p in inputs]
    build_meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "seed": int(seed),
        "num_shards": int(num_shards),
        "inputs": [asdict(x) for x in fps],
        "polars_version": getattr(pl, "__version__", None),
        "compression": str(compression),
        "statistics": bool(statistics),
        "approximate_bytes_per_file": approximate_bytes_per_file,
        "max_rows_per_file": max_rows_per_file,
        "infer_schema_length": infer_len,
        "batch_size": batch_size,
        "low_memory": bool(args.low_memory),
        "ignore_errors": bool(args.ignore_errors),
        "elapsed_sec": float(elapsed),
    }
    _write_json(os.path.join(out_dir, "build_meta.json"), build_meta)

    shards_existing = _collect_output_shards(out_dir)
    by_id = {int(x["shard_id"]): x for x in shards_existing}
    missing = 0
    shards: list[dict] = []
    for shard_id in range(num_shards):
        s = by_id.get(shard_id)
        if s is None:
            missing += 1
            shards.append({"shard_id": int(shard_id), "files": [], "total_bytes": 0})
        else:
            shards.append(s)
    if missing:
        print(f"[WARN] {missing}/{num_shards} shard_ids have no output files.", file=sys.stderr, flush=True)
    manifest = {
        "version": 1,
        "created_at": build_meta["created_at"],
        "seed": int(seed),
        "num_shards": int(num_shards),
        "backend": "polars_partitionby_parquet",
        "shards": shards,
    }
    _write_json(os.path.join(out_dir, "manifest.json"), manifest)

    total_files = sum(len(x.get("files", [])) for x in shards)
    emit_progress("DONE")
    print(
        f"OK: wrote {len(shards)} shard groups ({total_files} parquet files) to {out_dir} in {elapsed:.2f}s",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Shard JSONL/NDJSON into Parquet shards using Polars.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build parquet shards + manifest.json.")
    b.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input .jsonl/.ndjson file or directory. Can be repeated.",
    )
    b.add_argument("--output", required=True, help="Output directory (will create shards/ + manifest.json).")
    b.add_argument("--num-shards", type=int, required=True, help="Number of shard buckets.")
    b.add_argument("--seed", type=int, default=42, help="Seed for shard assignment hash.")
    b.add_argument(
        "--id-col",
        default="",
        help="Optional id column for shard assignment; otherwise uses source_path+row_index.",
    )
    b.add_argument(
        "--writer",
        choices=["partitionby", "chunked", "file_parallel"],
        default="partitionby",
        help="Shard writer implementation. 'partitionby' may use high memory for huge datasets. "
        "Use 'chunked' (single-process streaming) or 'file_parallel' (multi-process by files + merge) for very large inputs.",
    )
    b.add_argument(
        "--chunk-rows",
        type=int,
        default=200000,
        help="Only for --writer=chunked: number of ndjson rows per in-memory chunk.",
    )
    b.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Only for --writer=chunked/--writer=file_parallel: number of worker processes (default: 1).",
    )
    b.add_argument(
        "--use-rich",
        action="store_true",
        help="Use rich library for beautiful progress bars (only for --writer=chunked with --num-workers > 1).",
    )
    b.add_argument("--compression", default="zstd", help="Parquet compression codec (e.g. zstd/snappy).")
    b.add_argument(
        "--infer-schema-length",
        type=int,
        default=100,
        help="Polars scan_ndjson infer_schema_length (default: 100). Avoid None for huge datasets.",
    )
    b.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Polars scan_ndjson batch_size (default: 1024).",
    )
    b.add_argument("--low-memory", action="store_true", help="Enable polars low_memory mode.")
    b.add_argument("--ignore-errors", action="store_true", help="Ignore errors during ndjson scanning (best-effort).")
    b.add_argument("--no-statistics", action="store_true", help="Disable parquet statistics (lower CPU/memory).")
    b.add_argument(
        "--approximate-bytes-per-file",
        default="auto",
        help="PartitionBy approximate_bytes_per_file: auto/int/none (controls file splitting).",
    )
    b.add_argument("--max-rows-per-file", type=int, default=None, help="PartitionBy max_rows_per_file.")
    b.add_argument(
        "--progress-interval-sec",
        type=float,
        default=30.0,
        help="Emit periodic progress while writing shards; set 0 to disable.",
    )
    b.add_argument("--progress-bar-width", type=int, default=30, help="ASCII progress bar width.")
    b.add_argument(
        "--file-parallel-report-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Only for --writer=file_parallel: progress report interval per worker (bytes).",
    )
    b.set_defaults(func=cmd_build)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
