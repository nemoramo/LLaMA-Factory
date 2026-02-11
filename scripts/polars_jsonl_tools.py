#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable


def _iter_inputs(path: str) -> Iterable[str]:
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            p = os.path.join(path, name)
            if os.path.isfile(p):
                yield p
        return
    yield path


def _require_polars():
    try:
        import polars as pl  # type: ignore

        return pl
    except Exception as e:  # noqa: BLE001
        print(
            "polars is not installed. Install it in your training env, e.g.\n"
            "  pip install -U polars\n"
            "or\n"
            "  conda install -c conda-forge polars\n"
            f"import error: {e!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def cmd_stats(args: argparse.Namespace) -> int:
    pl = _require_polars()

    duration_cols = [c.strip() for c in str(args.duration_cols).split(",") if c.strip()]
    if not duration_cols:
        duration_cols = [
            "duration",
            "duration_sec",
            "audio_duration",
            "duration_seconds",
            "duration_ms",
            "duration_msec",
        ]

    for p in _iter_inputs(str(args.input)):
        lf = pl.scan_ndjson(p)
        out = {"path": p}

        try:
            out["rows"] = int(lf.select(pl.len()).collect().item())
        except Exception:  # noqa: BLE001
            out["rows"] = None

        schema = {}
        try:
            schema = dict(lf.schema)
        except Exception:  # noqa: BLE001
            schema = {}

        dur_col = next((c for c in duration_cols if c in schema), None)
        if dur_col is not None:
            try:
                dur = lf.select(pl.col(dur_col).cast(pl.Float64, strict=False).sum()).collect().item()
                dur = float(dur) if dur is not None else 0.0
                if dur_col.endswith("_ms") or dur_col.endswith("_msec"):
                    dur = dur / 1000.0
                out["duration_sec_sum"] = dur
                out["duration_hours_sum"] = dur / 3600.0
                out["duration_col"] = dur_col
            except Exception:  # noqa: BLE001
                out["duration_sec_sum"] = None
                out["duration_hours_sum"] = None
                out["duration_col"] = dur_col
        else:
            out["duration_sec_sum"] = None
            out["duration_hours_sum"] = None
            out["duration_col"] = None

        print(out)

    return 0


def cmd_to_parquet(args: argparse.Namespace) -> int:
    pl = _require_polars()

    input_path = str(args.input)
    output_path = str(args.output)
    compression = str(args.compression)

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for p in _iter_inputs(input_path):
            name = os.path.basename(p)
            if name.endswith(".jsonl"):
                name = name[: -len(".jsonl")]
            elif name.endswith(".ndjson"):
                name = name[: -len(".ndjson")]
            out_file = os.path.join(output_path, name + ".parquet")
            _convert_one(pl, p, out_file, compression=compression)
        return 0

    _convert_one(pl, input_path, output_path, compression=compression)
    return 0


def _convert_one(pl, input_path: str, output_path: str, *, compression: str) -> None:
    lf = pl.scan_ndjson(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    sink = getattr(lf, "sink_parquet", None)
    if callable(sink):
        sink(output_path, compression=compression)
        return

    # Older Polars: fallback to streaming collect (may use more RAM than sink_parquet).
    df = lf.collect(streaming=True)
    df.write_parquet(output_path, compression=compression)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fast JSONL/NDJSON tools using Polars (optional).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats", help="Print simple stats for jsonl/ndjson.")
    p_stats.add_argument("--input", required=True, help="File path or directory of .jsonl shards.")
    p_stats.add_argument(
        "--duration-cols",
        default="",
        help="Comma-separated duration columns to try first (default: common duration keys).",
    )
    p_stats.set_defaults(func=cmd_stats)

    p_parq = sub.add_parser("to-parquet", help="Convert jsonl/ndjson to parquet.")
    p_parq.add_argument("--input", required=True, help="File path or directory of .jsonl shards.")
    p_parq.add_argument("--output", required=True, help="Output parquet path, or output directory for shard dir.")
    p_parq.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "lz4", "uncompressed"],
        help="Parquet compression codec.",
    )
    p_parq.set_defaults(func=cmd_to_parquet)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

