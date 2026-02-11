#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ShardSpec:
    shard_id: int
    files: list[str]


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _write_json(path: str, obj: object) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _format_sec(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s/60.0:.1f}m"
    return f"{s/3600.0:.2f}h"


def _format_bytes(n: int) -> str:
    n = int(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    x = float(n)
    u = 0
    while x >= 1024.0 and u < len(units) - 1:
        x /= 1024.0
        u += 1
    if u == 0:
        return f"{int(x)}{units[u]}"
    return f"{x:.2f}{units[u]}"


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    width = max(5, int(width))
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(width * (done / total)))
    filled = min(width, max(0, filled))
    return "[" + ("#" * filled) + ("." * (width - filled)) + "]"


def _load_manifest(path: str) -> tuple[str, list[ShardSpec], dict[str, Any]]:
    path = os.path.abspath(os.path.expanduser(path))
    base_dir = os.path.dirname(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    shards: list[ShardSpec] = []
    for item in obj.get("shards", []) or []:
        try:
            shard_id = int(item.get("shard_id", 0))
        except Exception:
            continue
        files = item.get("files", [])
        if not isinstance(files, list):
            continue
        files2 = [str(x) for x in files if x]
        if not files2:
            continue
        shards.append(ShardSpec(shard_id=shard_id, files=files2))

    if not shards:
        raise SystemExit(f"No shards found in manifest: {path}")

    return base_dir, sorted(shards, key=lambda s: int(s.shard_id)), obj


def _resolve_files(base_dir: str, files: list[str]) -> list[str]:
    out: list[str] = []
    for fp in files:
        fp = str(fp)
        if os.path.isabs(fp):
            out.append(fp)
        else:
            out.append(os.path.join(base_dir, fp))
    return out


def _shuffle_one_shard(args: tuple) -> dict[str, Any]:
    (
        shard_id,
        input_files,
        out_path,
        seed,
        buffer_row_groups,
        compression,
        statistics,
        use_threads,
    ) = args

    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq

    t0 = time.time()

    # Resolve schema from the first file.
    if not input_files:
        raise RuntimeError(f"shard {shard_id}: no input files")
    pf0 = pq.ParquetFile(input_files[0])
    schema = pf0.schema_arrow

    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = pq.ParquetWriter(
        tmp_path,
        schema,
        compression=str(compression),
        write_statistics=bool(statistics),
    )

    rng = random.Random((int(seed) ^ (int(shard_id) * 1000003)) & 0xFFFFFFFF)
    buffer_n = int(buffer_row_groups or 0)
    if buffer_n < 0:
        raise ValueError("buffer_row_groups must be >= 0")

    buf: list[Any] = []
    total_row_groups = 0
    total_rows = 0
    in_bytes = 0

    try:
        for fp in input_files:
            try:
                in_bytes += int(os.stat(fp).st_size)
            except OSError:
                pass
            pf = pq.ParquetFile(fp)
            if pf.schema_arrow != schema:
                raise RuntimeError(f"shard {shard_id}: schema mismatch in {fp}")
            num_rg = int(pf.num_row_groups)
            total_row_groups += num_rg

            if buffer_n <= 1 or num_rg <= 1:
                for i in range(num_rg):
                    t = pf.read_row_group(i, use_threads=bool(use_threads))
                    total_rows += int(t.num_rows)
                    writer.write_table(t)
                continue

            # Streaming shuffle: keep N row-groups in RAM, emit one per incoming row-group.
            n = min(int(buffer_n), num_rg)
            # Fill.
            for i in range(n):
                t = pf.read_row_group(i, use_threads=bool(use_threads))
                buf.append(t)
            # Process remaining.
            for i in range(n, num_rg):
                t = pf.read_row_group(i, use_threads=bool(use_threads))
                j = rng.randrange(n)
                out_t = buf[j]
                buf[j] = t
                total_rows += int(out_t.num_rows)
                writer.write_table(out_t)

            # Drain.
            rng.shuffle(buf)
            for t in buf:
                total_rows += int(t.num_rows)
                writer.write_table(t)
            buf.clear()

        writer.close()
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

    try:
        out_bytes = int(os.stat(out_path).st_size)
    except OSError:
        out_bytes = 0
    elapsed = time.time() - t0
    return {
        "shard_id": int(shard_id),
        "input_files": [str(x) for x in input_files],
        "in_bytes": int(in_bytes),
        "out_bytes": int(out_bytes),
        "row_groups": int(total_row_groups),
        "rows": int(total_rows),
        "elapsed_sec": float(elapsed),
    }


def cmd_build(args: argparse.Namespace) -> int:
    in_manifest = os.path.abspath(os.path.expanduser(str(args.input_manifest)))
    out_dir = os.path.abspath(os.path.expanduser(str(args.output)))
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"Output dir not empty: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)

    base_dir, shards, src_manifest_obj = _load_manifest(in_manifest)
    num_shards = int(src_manifest_obj.get("num_shards", len(shards)) or len(shards))

    # Keep count stable.
    shards_sorted = sorted(shards, key=lambda s: int(s.shard_id))
    if len(shards_sorted) != int(num_shards):
        print(
            f"[WARN] input manifest has {len(shards_sorted)} shard groups but num_shards={num_shards}.",
            file=sys.stderr,
            flush=True,
        )

    compression = str(args.compression)
    statistics = not bool(args.no_statistics)
    seed = int(args.seed)
    buffer_row_groups = int(args.buffer_row_groups)
    if buffer_row_groups < 0:
        raise SystemExit("--buffer-row-groups must be >= 0")

    num_workers = int(args.num_workers or 0)
    if num_workers <= 0:
        num_workers = min(16, max(1, os.cpu_count() or 8))

    print(
        "[INFO] shuffle parquet shards:\n"
        f"  input_manifest={in_manifest}\n"
        f"  output={out_dir}\n"
        f"  shards={len(shards_sorted)} (num_shards={num_shards})\n"
        f"  seed={seed}\n"
        f"  buffer_row_groups={buffer_row_groups} (0/1 disables)\n"
        f"  compression={compression} statistics={statistics}\n"
        f"  workers={num_workers} use_threads={bool(args.use_threads)}\n",
        file=sys.stderr,
        flush=True,
    )

    t0 = time.time()
    results: list[dict[str, Any]] = []

    work: list[tuple] = []
    for shard in shards_sorted:
        in_files = _resolve_files(base_dir, shard.files)
        out_path = os.path.join(out_dir, "shards", f"part-{int(shard.shard_id):05d}.parquet")
        work.append(
            (
                int(shard.shard_id),
                in_files,
                out_path,
                seed,
                buffer_row_groups,
                compression,
                statistics,
                bool(args.use_threads),
            )
        )

    done = 0
    total = len(work)
    last_print = 0.0
    mp_ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as ex:
        futs = [ex.submit(_shuffle_one_shard, w) for w in work]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            now = time.time()
            if args.progress_interval_sec <= 0:
                continue
            if (now - last_print) >= float(args.progress_interval_sec) or done == total:
                elapsed = now - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                print(
                    f"[INFO] {_progress_bar(done, total, width=int(args.progress_bar_width))} "
                    f"{done}/{total} elapsed={_format_sec(elapsed)} eta={_format_sec(eta)} "
                    f"last_shard={int(r.get('shard_id', -1)):05d} "
                    f"last_time={_format_sec(float(r.get('elapsed_sec', 0.0) or 0.0))}",
                    file=sys.stderr,
                    flush=True,
                )
                last_print = now

    results.sort(key=lambda x: int(x.get("shard_id", 0)))
    elapsed_total = time.time() - t0

    # Write output manifest.
    shards_out: list[dict[str, Any]] = []
    total_bytes = 0
    for r in results:
        shard_id = int(r.get("shard_id", 0))
        out_rel = os.path.relpath(os.path.join(out_dir, "shards", f"part-{shard_id:05d}.parquet"), out_dir)
        out_bytes = int(r.get("out_bytes", 0) or 0)
        total_bytes += out_bytes
        shards_out.append({"shard_id": shard_id, "files": [out_rel], "total_bytes": out_bytes})

    out_manifest = {
        "version": 1,
        "created_at": _now_str(),
        "seed": seed,
        "num_shards": int(num_shards),
        "backend": "shuffle_existing_parquet_row_groups",
        "shards": shards_out,
    }
    _write_json(os.path.join(out_dir, "manifest.json"), out_manifest)

    build_meta = {
        "created_at": out_manifest["created_at"],
        "backend": out_manifest["backend"],
        "input_manifest": in_manifest,
        "seed": seed,
        "buffer_row_groups": buffer_row_groups,
        "compression": compression,
        "statistics": statistics,
        "workers": num_workers,
        "use_threads": bool(args.use_threads),
        "elapsed_sec": float(elapsed_total),
        "bytes_total": int(total_bytes),
        "shards": len(shards_out),
        "per_shard": results,
    }
    _write_json(os.path.join(out_dir, "build_meta.json"), build_meta)

    print(
        f"[OK] wrote shuffled shards={len(shards_out)} size={_format_bytes(total_bytes)} "
        f"to {out_dir} in {_format_sec(elapsed_total)}",
        file=sys.stderr,
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create a shuffled version of existing parquet shards (by row-groups).")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Shuffle existing parquet shards and write a new manifest.json.")
    b.add_argument("--input-manifest", required=True, help="Input shard manifest.json.")
    b.add_argument("--output", required=True, help="Output directory (will create shards/ + manifest.json).")
    b.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    b.add_argument(
        "--buffer-row-groups",
        type=int,
        default=256,
        help="Streaming shuffle buffer size in row-groups per input file; 0/1 disables shuffling.",
    )
    b.add_argument("--num-workers", type=int, default=0, help="Number of processes (default: min(16,cpu_count)).")
    b.add_argument("--use-threads", action="store_true", help="Use pyarrow multi-threaded decoding per process.")
    b.add_argument("--compression", default="zstd", help="Output parquet compression (zstd/snappy).")
    b.add_argument("--no-statistics", action="store_true", help="Disable parquet statistics.")
    b.add_argument(
        "--progress-interval-sec",
        type=float,
        default=30.0,
        help="Progress print interval; set 0 to disable.",
    )
    b.add_argument("--progress-bar-width", type=int, default=30, help="ASCII progress bar width.")
    b.set_defaults(func=cmd_build)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

