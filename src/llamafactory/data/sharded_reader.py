from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Iterator

from torch.utils.data import IterableDataset, get_worker_info

from ..extras import logging


logger = logging.get_logger(__name__)


SHARD_BOUNDARY_KEY = "__llamafactory_shard_boundary__"


@dataclass(frozen=True)
class ShardSpec:
    shard_id: int
    files: list[str]
    total_bytes: int = 0


def load_shard_manifest(manifest_path: str) -> tuple[str, list[ShardSpec]]:
    manifest_path = os.path.abspath(os.path.expanduser(str(manifest_path)))
    base_dir = os.path.dirname(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    shards_obj = obj.get("shards", [])
    shards: list[ShardSpec] = []
    for item in shards_obj:
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
        total_bytes = 0
        try:
            total_bytes = int(item.get("total_bytes", 0) or 0)
        except Exception:
            total_bytes = 0
        shards.append(ShardSpec(shard_id=shard_id, files=files2, total_bytes=total_bytes))

    if not shards:
        raise ValueError(f"No shards found in manifest: {manifest_path}")
    return base_dir, shards


def _get_rank_world_size() -> tuple[int, int]:
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
            world_size = int(torch.distributed.get_world_size())
            if world_size > 0 and rank >= 0:
                return rank, world_size
    except Exception:
        pass

    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)) or str(default))
        except Exception:
            return default

    # Prefer global rank/world_size. Fallback to local rank for single-node runs.
    world_size = _int_env("WORLD_SIZE", 1)
    rank = _int_env("RANK", _int_env("LOCAL_RANK", 0))
    if world_size <= 0:
        world_size = 1
    if rank < 0:
        rank = 0
    return rank, world_size


def _get_worker_info() -> tuple[int, int]:
    info = get_worker_info()
    if info is None:
        return 0, 1
    return int(info.id), int(info.num_workers)


def _partition_items(items: list[Any], index: int, modulo: int) -> list[Any]:
    if modulo <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if (i % modulo) == index]


def _iter_parquet_rows(
    path: str,
    *,
    batch_rows: int,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    # Import inside so importing LLaMA-Factory doesn't hard-require pyarrow.
    import pyarrow.parquet as pq  # type: ignore

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=int(batch_rows), columns=columns):
        # RecordBatch -> list[dict]. Bounded by batch_size.
        for row in batch.to_pylist():
            if row is None:
                continue
            if not isinstance(row, dict):
                continue
            yield row


def _buffer_shuffle(
    it: Iterator[dict[str, Any]],
    *,
    buf_size: int,
    rng: random.Random,
) -> Iterator[dict[str, Any]]:
    buf: list[dict[str, Any]] = []
    for row in it:
        buf.append(row)
        if len(buf) >= buf_size:
            rng.shuffle(buf)
            yield from buf
            buf.clear()
    if buf:
        rng.shuffle(buf)
        yield from buf


class ShardedParquetIterableDataset(IterableDataset):
    """Iterable dataset that yields dict rows from parquet shards with rank/worker sharding.

    This dataset is designed for on-the-fly dynamic prompt packing. It yields a shard-boundary marker
    (`{SHARD_BOUNDARY_KEY: True}`) between shards so pack buffers can be flushed at shard boundaries.
    """

    def __init__(
        self,
        *,
        manifest_path: str,
        seed: int,
        shuffle_shards: bool = True,
        row_shuffle_buffer: int = 0,
        parquet_batch_rows: int = 8192,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.row_shuffle_buffer = int(row_shuffle_buffer or 0)
        self.parquet_batch_rows = int(parquet_batch_rows or 8192)
        self.columns = columns

        if self.parquet_batch_rows <= 0:
            raise ValueError(f"Invalid parquet_batch_rows: {self.parquet_batch_rows}")
        if self.row_shuffle_buffer < 0:
            raise ValueError(f"Invalid row_shuffle_buffer: {self.row_shuffle_buffer}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        base_dir, shards = load_shard_manifest(self.manifest_path)
        rank, world_size = _get_rank_world_size()
        worker_id, num_workers = _get_worker_info()

        # 1) Partition shards across ranks, then across dataloader workers.
        shards_sorted = sorted(shards, key=lambda s: int(s.shard_id))
        shards_rank = _partition_items(shards_sorted, rank, world_size)
        shards_worker = _partition_items(shards_rank, worker_id, num_workers)
        if not shards_worker:
            logger.warning_rank0(
                "ShardedParquetIterableDataset: no shards assigned (rank=%d/%d worker=%d/%d).",
                rank,
                world_size,
                worker_id,
                num_workers,
            )
            return

        # 2) Repeat indefinitely; training length is controlled by max_steps.
        cycle_idx = 0
        while True:
            shard_order = list(shards_worker)
            if self.shuffle_shards and len(shard_order) > 1:
                random.Random(self.seed + cycle_idx).shuffle(shard_order)

            for shard in shard_order:
                # Shard files are stored relative to manifest base dir.
                files = [os.path.join(base_dir, fp) for fp in shard.files]
                row_it = _iter_shard_rows(files, batch_rows=self.parquet_batch_rows, columns=self.columns)
                if self.row_shuffle_buffer > 1:
                    rng = random.Random((self.seed + cycle_idx) ^ (int(shard.shard_id) * 1000003))
                    row_it = _buffer_shuffle(row_it, buf_size=self.row_shuffle_buffer, rng=rng)
                yield from row_it

                # Shard boundary marker (consumed by dynamic prompt packing bufferizer).
                yield {SHARD_BOUNDARY_KEY: True}

            cycle_idx += 1


def _iter_shard_rows(
    files: list[str],
    *,
    batch_rows: int,
    columns: list[str] | None,
) -> Iterator[dict[str, Any]]:
    for fp in files:
        if not os.path.isfile(fp):
            continue
        yield from _iter_parquet_rows(fp, batch_rows=int(batch_rows), columns=columns)
