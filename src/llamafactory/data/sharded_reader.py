from __future__ import annotations

import json
import os
import random
import threading
import time
import zlib
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Iterator
from urllib.parse import urlparse

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


_PYARROW_S3FS_CACHE: dict[tuple[int, str], Any] = {}


def _ensure_http_scheme(endpoint: str) -> str:
    ep = (endpoint or "").strip()
    if ep.startswith("http://") or ep.startswith("https://"):
        return ep
    return f"https://{ep}"


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        v = os.environ.get(name)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def _maybe_map_mount_to_tos_uri(path: str) -> str | None:
    # Only map when explicitly enabled; otherwise treat mount paths as local FS.
    raw_flag = (os.environ.get("LLAMAFACTORY_TOS_SDK_FOR_MOUNT") or "").strip().lower()
    if raw_flag not in ("1", "true", "yes", "y", "on"):
        return None

    # Optional override: "mount_prefix:bucket,mount_prefix2:bucket2"
    raw_map = (os.environ.get("LLAMAFACTORY_TOS_MOUNT_MAP") or "").strip()
    if raw_map:
        pairs: list[tuple[str, str]] = []
        for part in raw_map.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            mp, bucket = part.split(":", 1)
            mp = mp.strip()
            bucket = bucket.strip()
            if not mp or not bucket:
                continue
            pairs.append((os.path.normpath(mp), bucket))
    else:
        pairs = [
            (os.path.normpath("/mnt/asr-audio-data"), "asr-audio-data"),
            (os.path.normpath("/mnt/tts-data-tos"), "tts-data-tos"),
        ]

    p_norm = os.path.normpath(path)
    for mp_norm, bucket in pairs:
        if p_norm == mp_norm or p_norm.startswith(mp_norm + os.sep):
            rel = p_norm[len(mp_norm) :].lstrip(os.sep).replace(os.sep, "/")
            return f"tos://{bucket}/{rel}"
    return None


def _get_pyarrow_s3_filesystem(mode: str):
    mode = str(mode or "").strip().lower()
    if mode not in ("s3", "tos"):
        raise ValueError(f"Invalid mode: {mode!r}")

    cache_key = (os.getpid(), mode)
    fs = _PYARROW_S3FS_CACHE.get(cache_key)
    if fs is not None:
        return fs

    import pyarrow.fs as pafs  # type: ignore

    if mode == "s3":
        fs = pafs.S3FileSystem()
        _PYARROW_S3FS_CACHE[cache_key] = fs
        return fs

    ak = _env_first("TOS_ACCESS_KEY_ID", "TOS_AK", "AWS_ACCESS_KEY_ID")
    sk = _env_first("TOS_SECRET_ACCESS_KEY", "TOS_SK", "AWS_SECRET_ACCESS_KEY")
    token = _env_first("TOS_SESSION_TOKEN", "AWS_SESSION_TOKEN")
    endpoint = _env_first("TOS_ENDPOINT", "TOS_ENDPOINT_URL")
    region = _env_first("TOS_REGION", "AWS_DEFAULT_REGION")
    addressing_style = (_env_first("TOS_ADDRESSING_STYLE", default="virtual") or "virtual").strip().lower()
    if addressing_style not in ("virtual", "path"):
        addressing_style = "virtual"

    if not ak or not sk or not endpoint or not region:
        raise RuntimeError(
            "TOS access requires env vars: TOS_ACCESS_KEY_ID/TOS_SECRET_ACCESS_KEY/TOS_ENDPOINT/TOS_REGION "
            "(or their AWS_* fallbacks)."
        )

    endpoint = _ensure_http_scheme(endpoint)
    parsed = urlparse(endpoint)
    scheme = parsed.scheme or "https"
    endpoint_override = parsed.netloc or parsed.path
    endpoint_override = endpoint_override.strip("/")
    if not endpoint_override:
        raise RuntimeError("Invalid TOS_ENDPOINT.")

    # If endpoint_override is set, pyarrow defaults to path-style unless force_virtual_addressing=True.
    force_virtual = addressing_style == "virtual"
    fs = pafs.S3FileSystem(
        access_key=ak,
        secret_key=sk,
        session_token=token,
        region=region,
        scheme=scheme,
        endpoint_override=endpoint_override,
        force_virtual_addressing=force_virtual,
    )
    _PYARROW_S3FS_CACHE[cache_key] = fs
    return fs


def _iter_parquet_rows(
    path: str,
    *,
    batch_rows: int,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    for batch in _iter_parquet_record_batches(path, batch_rows=batch_rows, columns=columns):
        # RecordBatch -> list[dict]. Bounded by batch_size.
        for row in batch.to_pylist():
            if row is None:
                continue
            if not isinstance(row, dict):
                continue
            yield row


def _iter_parquet_record_batches(
    path: str,
    *,
    batch_rows: int,
    columns: list[str] | None = None,
    row_group_shuffle: bool = False,
    row_group_shuffle_block_size: int = 0,
    row_group_shuffle_seed: int = 0,
):
    # Import inside so importing LLaMA-Factory doesn't hard-require pyarrow.
    import pyarrow.parquet as pq  # type: ignore

    if isinstance(path, str):
        mapped = _maybe_map_mount_to_tos_uri(path)
        if mapped is not None:
            path = mapped

    if isinstance(path, str) and path.startswith(("s3://", "tos://")):
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        fs = _get_pyarrow_s3_filesystem("tos" if path.startswith("tos://") else "s3")
        pf = pq.ParquetFile(f"{bucket}/{key}", filesystem=fs)
    else:
        pf = pq.ParquetFile(path)

    row_groups = None
    if bool(row_group_shuffle) and int(getattr(pf, "num_row_groups", 0) or 0) > 1:
        num_row_groups = int(pf.num_row_groups)
        block_size = int(row_group_shuffle_block_size or 0)
        if block_size == -1:
            # Auto: shuffle coarse row-group blocks so each block is ~1 output RecordBatch.
            # This improves mixing without causing extremely random IO patterns.
            try:
                rows_per_rg = int(getattr(pf.metadata.row_group(0), "num_rows", 0) or 0)  # type: ignore[union-attr]
            except Exception:
                rows_per_rg = 0
            if rows_per_rg <= 0:
                rows_per_rg = max(1, int(getattr(pf.metadata, "num_rows", 0) or 0) // max(1, num_row_groups))  # type: ignore[union-attr]
            block_size = max(1, int(batch_rows) // max(1, int(rows_per_rg)))
        elif block_size <= 0:
            # Default: fully shuffle row-groups.
            block_size = 1

        blocks = [list(range(i, min(i + block_size, num_row_groups))) for i in range(0, num_row_groups, block_size)]
        random.Random(int(row_group_shuffle_seed) & 0xFFFFFFFF).shuffle(blocks)
        row_groups = [rg for block in blocks for rg in block]

    for batch in pf.iter_batches(batch_size=int(batch_rows), columns=columns, row_groups=row_groups):
        yield batch


def _iter_rows_from_record_batches(batch_it, *, on_batch=None) -> Iterator[dict[str, Any]]:
    for batch in batch_it:
        if batch is None:
            continue
        if on_batch is not None:
            on_batch(batch_it, batch)
        for row in batch.to_pylist():
            if row is None:
                continue
            if not isinstance(row, dict):
                continue
            yield row


def _iter_shard_record_batches(
    files: list[str],
    *,
    batch_rows: int,
    columns: list[str] | None,
    row_group_shuffle: bool,
    row_group_shuffle_block_size: int,
    row_group_shuffle_seed: int,
):
    files2 = list(files)
    if bool(row_group_shuffle) and len(files2) > 1:
        random.Random(int(row_group_shuffle_seed) & 0xFFFFFFFF).shuffle(files2)

    for fp in files2:
        seed = int(row_group_shuffle_seed) ^ int(zlib.crc32(str(fp).encode("utf-8")) & 0xFFFFFFFF)
        if isinstance(fp, str) and fp.startswith(("s3://", "tos://")):
            yield from _iter_parquet_record_batches(
                fp,
                batch_rows=int(batch_rows),
                columns=columns,
                row_group_shuffle=bool(row_group_shuffle),
                row_group_shuffle_block_size=int(row_group_shuffle_block_size),
                row_group_shuffle_seed=int(seed),
            )
            continue
        if not os.path.isfile(fp):
            continue
        yield from _iter_parquet_record_batches(
            fp,
            batch_rows=int(batch_rows),
            columns=columns,
            row_group_shuffle=bool(row_group_shuffle),
            row_group_shuffle_block_size=int(row_group_shuffle_block_size),
            row_group_shuffle_seed=int(seed),
        )


class _PrefetchIterator:
    def __init__(self, src_it, *, maxsize: int, name: str) -> None:
        self._src_it = src_it
        self._maxsize = int(maxsize)
        self._name = str(name)

        self._q: Queue = Queue(maxsize=self._maxsize)
        self._stop = threading.Event()
        self._sentinel = object()
        self._err: BaseException | None = None
        self._thread: threading.Thread | None = None

        self.started_at: float | None = None
        self.first_put_at: float | None = None
        self.put_count: int = 0
        self.get_count: int = 0
        self.total_get_wait_s: float = 0.0
        self.last_get_wait_s: float = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return

        self.started_at = time.time()

        def producer():
            try:
                for item in self._src_it:
                    if self._stop.is_set():
                        break
                    # Use a relatively long timeout to avoid tight timeout/retry loops when the consumer
                    # processes each RecordBatch slowly (queue stays full for seconds). We still keep
                    # a timeout (instead of blocking forever) so `close()` can stop the thread.
                    put_timeout_s = 5.0
                    while not self._stop.is_set():
                        try:
                            self._q.put(item, timeout=put_timeout_s)
                            break
                        except Full:
                            continue
                    if self.first_put_at is None:
                        self.first_put_at = time.time()
                    self.put_count += 1
            except BaseException as err:  # pragma: no cover
                self._err = err
            finally:
                # Always signal consumer termination.
                sentinel_timeout_s = 5.0
                while True:
                    try:
                        self._q.put(self._sentinel, timeout=sentinel_timeout_s)
                        break
                    except Full:
                        if self._stop.is_set():
                            break
                        continue

        self._thread = threading.Thread(target=producer, name=f"prefetch:{self._name}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        self.start()
        t0 = time.time()
        while True:
            if self._stop.is_set() and self._q.empty():
                raise StopIteration
            try:
                item = self._q.get(timeout=0.5)
                break
            except Empty:
                continue

        dt = time.time() - t0
        self.last_get_wait_s = float(dt)
        self.total_get_wait_s += float(dt)

        if item is self._sentinel:
            if self._err is not None:
                raise self._err
            raise StopIteration

        self.get_count += 1
        return item


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
        row_group_shuffle: bool = False,
        row_group_shuffle_block_size: int = 0,
        parquet_batch_rows: int = 8192,
        prefetch_next_shard: bool = True,
        prefetch_queue_batches: int = 1,
        prefetch_log: bool = False,
        resume_mode: str = "off",
        resume_state_dir: str | None = None,
        resume_prefer_checkpoint: bool = True,
        resume_log: bool = False,
        output_dir: str = "",
        resume_from_checkpoint: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.row_shuffle_buffer = int(row_shuffle_buffer or 0)
        self.row_group_shuffle = bool(row_group_shuffle)
        self.row_group_shuffle_block_size = int(row_group_shuffle_block_size or 0)
        self.parquet_batch_rows = int(parquet_batch_rows or 8192)
        self.prefetch_next_shard = bool(prefetch_next_shard)
        self.prefetch_queue_batches = int(prefetch_queue_batches or 0)
        self.prefetch_log = bool(prefetch_log)
        self.resume_mode = str(resume_mode or "off")
        self.resume_state_dir = str(resume_state_dir) if isinstance(resume_state_dir, str) and resume_state_dir else None
        self.resume_prefer_checkpoint = bool(resume_prefer_checkpoint)
        self.resume_log = bool(resume_log)
        self.output_dir = str(output_dir or "")
        self.resume_from_checkpoint = (
            str(resume_from_checkpoint) if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint else None
        )
        self.columns = columns

        if self.parquet_batch_rows <= 0:
            raise ValueError(f"Invalid parquet_batch_rows: {self.parquet_batch_rows}")
        if self.row_shuffle_buffer < 0:
            raise ValueError(f"Invalid row_shuffle_buffer: {self.row_shuffle_buffer}")
        if self.row_group_shuffle_block_size < -1:
            raise ValueError(f"Invalid row_group_shuffle_block_size: {self.row_group_shuffle_block_size}")
        if self.prefetch_queue_batches < 0:
            raise ValueError(f"Invalid prefetch_queue_batches: {self.prefetch_queue_batches}")
        if self.resume_mode not in ("off", "shard_boundary"):
            raise ValueError(f"Invalid resume_mode: {self.resume_mode}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        base_dir, shards = load_shard_manifest(self.manifest_path)
        rank, world_size = _get_rank_world_size()
        worker_id, num_workers = _get_worker_info()

        prefetch_log_ranks: set[int] = {0}
        if self.prefetch_log:
            raw = (os.environ.get("LLAMAFACTORY_SHARDED_PREFETCH_LOG_RANKS") or "").strip()
            if raw:
                if raw.lower() in ("all", "*"):
                    prefetch_log_ranks = set(range(max(1, int(world_size))))
                else:
                    parsed: set[int] = set()
                    for part in raw.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            parsed.add(int(part))
                        except Exception:
                            continue
                    if parsed:
                        prefetch_log_ranks = parsed
        prefetch_log_this_rank = bool(self.prefetch_log) and (int(rank) in prefetch_log_ranks)

        if self.row_group_shuffle:
            logger.info_rank0(
                "ShardedParquetIterableDataset: row-group shuffle enabled (block_size=%d).",
                int(self.row_group_shuffle_block_size),
            )

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
        resume_enabled = self.resume_mode == "shard_boundary"

        state_write_dir: str | None = None
        if resume_enabled:
            if self.resume_state_dir is not None:
                state_write_dir = os.path.abspath(os.path.expanduser(self.resume_state_dir))
            elif self.output_dir:
                state_write_dir = os.path.join(os.path.abspath(os.path.expanduser(self.output_dir)), "shard_resume_state")
            if state_write_dir:
                os.makedirs(state_write_dir, exist_ok=True)

        state_read_dir: str | None = state_write_dir
        if resume_enabled and self.resume_prefer_checkpoint and self.resume_from_checkpoint:
            ckpt_dir = os.path.abspath(os.path.expanduser(self.resume_from_checkpoint))
            ckpt_state = os.path.join(ckpt_dir, "shard_resume_state")
            if os.path.isdir(ckpt_state):
                state_read_dir = ckpt_state

        state_file_name = f"rank{rank:05d}_worker{worker_id:03d}.json"
        state_write_path = os.path.join(state_write_dir, state_file_name) if state_write_dir else None
        state_read_path = os.path.join(state_read_dir, state_file_name) if state_read_dir else None

        manifest_abs = os.path.abspath(os.path.expanduser(self.manifest_path))
        try:
            manifest_mtime_ns = int(getattr(os.stat(manifest_abs), "st_mtime_ns", 0) or 0)
        except OSError:
            manifest_mtime_ns = 0

        shards_worker_ids = [int(s.shard_id) for s in shards_worker]

        def _atomic_write_json(path: str, obj: dict[str, Any]) -> None:
            if not path:
                return
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, path)

        def _save_resume_state(*, cycle_idx: int, next_shard_index: int, shard_order_ids: list[int] | None) -> None:
            if not resume_enabled or not state_write_path:
                return
            payload: dict[str, Any] = {
                "version": 1,
                "manifest_path": manifest_abs,
                "manifest_mtime_ns": int(manifest_mtime_ns),
                "seed": int(self.seed),
                "shuffle_shards": bool(self.shuffle_shards),
                "rank": int(rank),
                "world_size": int(world_size),
                "worker_id": int(worker_id),
                "num_workers": int(num_workers),
                "cycle_idx": int(cycle_idx),
                "next_shard_index": int(next_shard_index),
                "shards_worker_ids": list(shards_worker_ids),
                "updated_at_unix_s": float(time.time()),
            }
            if shard_order_ids is not None:
                payload["shard_order_ids"] = [int(x) for x in shard_order_ids]
            try:
                _atomic_write_json(state_write_path, payload)
            except Exception as err:  # noqa: BLE001
                if self.resume_log:
                    logger.warning_rank0("ShardedParquetIterableDataset: failed to write resume state: %s", err)

        cycle_idx = 0
        next_shard_index = 0
        forced_order_cycle: int | None = None
        forced_order_ids: list[int] | None = None
        if resume_enabled and state_read_path and os.path.isfile(state_read_path):
            try:
                with open(state_read_path, encoding="utf-8") as f:
                    st = json.load(f)
            except Exception:
                st = None
            if isinstance(st, dict):
                ok = True
                if str(st.get("manifest_path") or "") != manifest_abs:
                    ok = False
                if int(st.get("seed") or 0) != int(self.seed):
                    ok = False
                if bool(st.get("shuffle_shards", True)) != bool(self.shuffle_shards):
                    ok = False
                if int(st.get("world_size") or 0) != int(world_size):
                    ok = False
                if int(st.get("rank") or -1) != int(rank):
                    ok = False
                if int(st.get("num_workers") or 0) != int(num_workers):
                    ok = False
                if int(st.get("worker_id") or -1) != int(worker_id):
                    ok = False
                ids = st.get("shards_worker_ids")
                if not isinstance(ids, list):
                    ok = False
                else:
                    try:
                        ids2 = [int(x) for x in ids]
                    except Exception:
                        ids2 = []
                    if ids2 != shards_worker_ids:
                        ok = False

                if ok:
                    try:
                        cycle_idx = max(0, int(st.get("cycle_idx") or 0))
                    except Exception:
                        cycle_idx = 0
                    try:
                        next_shard_index = max(0, int(st.get("next_shard_index") or 0))
                    except Exception:
                        next_shard_index = 0
                    order_ids = st.get("shard_order_ids")
                    if isinstance(order_ids, list) and order_ids:
                        try:
                            forced_order_ids = [int(x) for x in order_ids]
                            forced_order_cycle = int(cycle_idx)
                        except Exception:
                            forced_order_ids = None
                            forced_order_cycle = None
                    if self.resume_log:
                        logger.info_rank0(
                            "ShardedParquetIterableDataset: loaded resume state (rank=%d/%d worker=%d/%d cycle=%d next=%d from=%s).",
                            rank,
                            world_size,
                            worker_id,
                            num_workers,
                            cycle_idx,
                            next_shard_index,
                            os.path.dirname(state_read_path),
                        )
                elif self.resume_log:
                    logger.warning_rank0(
                        "ShardedParquetIterableDataset: resume state mismatch; ignoring (rank=%d/%d worker=%d/%d path=%s).",
                        rank,
                        world_size,
                        worker_id,
                        num_workers,
                        state_read_path,
                    )

        carry_prefetch = None
        carry_prefetch_shard_id = None
        while True:
            shard_by_id = {int(s.shard_id): s for s in shards_worker}
            shard_order: list[ShardSpec] = list(shards_worker)
            shard_order_ids: list[int] | None = None
            if forced_order_ids is not None and forced_order_cycle == int(cycle_idx):
                ids = [int(x) for x in forced_order_ids]
                if len(ids) == len(shard_order) and set(ids) == set(shard_by_id.keys()):
                    shard_order = [shard_by_id[i] for i in ids]
                    shard_order_ids = ids
                else:
                    forced_order_ids = None
                    forced_order_cycle = None

            if shard_order_ids is None:
                if self.shuffle_shards and len(shard_order) > 1:
                    random.Random(self.seed + cycle_idx).shuffle(shard_order)
                shard_order_ids = [int(s.shard_id) for s in shard_order]

            shard_order_next = list(shards_worker)
            if self.shuffle_shards and len(shard_order_next) > 1:
                random.Random(self.seed + cycle_idx + 1).shuffle(shard_order_next)

            start_pos = int(next_shard_index) if resume_enabled else 0
            if start_pos < 0:
                start_pos = 0
            if start_pos >= len(shard_order):
                cycle_idx += 1
                next_shard_index = 0
                forced_order_ids = None
                forced_order_cycle = None
                _save_resume_state(cycle_idx=cycle_idx, next_shard_index=next_shard_index, shard_order_ids=None)
                continue

            remaining_shards = max(0, len(shard_order) - start_pos)
            # Prefetch can help even when resuming at the very last shard of a cycle:
            # in that case, prefetch the next cycle's first shard to avoid a boundary stall.
            prefetch_enabled = self.prefetch_next_shard and self.prefetch_queue_batches > 0 and (
                remaining_shards > 1 or bool(shard_order_next)
            )

            def _files_for(shard: ShardSpec) -> list[str]:
                out: list[str] = []
                for fp in shard.files:
                    fp = str(fp)
                    if fp.startswith(("s3://", "tos://")):
                        out.append(fp)
                    elif os.path.isabs(fp):
                        out.append(fp)
                    else:
                        out.append(os.path.join(base_dir, fp))
                return out

            def _row_group_seed(shard: ShardSpec) -> int:
                # Keep this independent of cycle_idx so prefetch-carry between cycles stays consistent.
                return (
                    int(self.seed) ^ (int(shard.shard_id) * 1000003) ^ (int(rank) * 9176) ^ int(worker_id)
                ) & 0xFFFFFFFF

            def _make_row_it(shard: ShardSpec, batch_it) -> Iterator[dict[str, Any]]:
                seen_batches = 0

                def _on_batch(it, _batch) -> None:
                    nonlocal seen_batches
                    if not prefetch_log_this_rank:
                        return
                    if isinstance(it, _PrefetchIterator) and seen_batches == 0:
                        ready_s = None
                        if it.started_at is not None and it.first_put_at is not None:
                            ready_s = max(0.0, float(it.first_put_at - it.started_at))
                        logger.info(
                            "ShardedParquetIterableDataset: shard enter (rank=%d/%d worker=%d/%d cycle=%d shard=%d "
                            "prefetch_ready_s=%s first_get_wait_s=%.3f q_put=%d q_get=%d).",
                            rank,
                            world_size,
                            worker_id,
                            num_workers,
                            cycle_idx,
                            int(shard.shard_id),
                            "?" if ready_s is None else f"{ready_s:.3f}",
                            float(getattr(it, "last_get_wait_s", 0.0) or 0.0),
                            int(it.put_count),
                            int(it.get_count),
                        )
                    seen_batches += 1

                row_it = _iter_rows_from_record_batches(batch_it, on_batch=_on_batch if prefetch_log_this_rank else None)
                if self.row_shuffle_buffer > 1:
                    rng = random.Random((self.seed + cycle_idx) ^ (int(shard.shard_id) * 1000003))
                    row_it = _buffer_shuffle(row_it, buf_size=self.row_shuffle_buffer, rng=rng)
                return row_it

            def _start_prefetch(shard: ShardSpec | None) -> _PrefetchIterator | None:
                if not prefetch_enabled or shard is None:
                    return None
                name = f"rank{rank}-w{worker_id}-c{cycle_idx}-s{int(shard.shard_id)}"
                it = _iter_shard_record_batches(
                    _files_for(shard),
                    batch_rows=self.parquet_batch_rows,
                    columns=self.columns,
                    row_group_shuffle=bool(self.row_group_shuffle),
                    row_group_shuffle_block_size=int(self.row_group_shuffle_block_size),
                    row_group_shuffle_seed=int(_row_group_seed(shard)),
                )
                pre = _PrefetchIterator(it, maxsize=self.prefetch_queue_batches, name=name)
                pre.start()
                if prefetch_log_this_rank:
                    logger.info(
                        "ShardedParquetIterableDataset: prefetch start (rank=%d/%d worker=%d/%d cycle=%d shard=%d q=%d).",
                        rank,
                        world_size,
                        worker_id,
                        num_workers,
                        cycle_idx,
                        int(shard.shard_id),
                        int(self.prefetch_queue_batches),
                    )
                return pre

            # Prime current shard iterator (may reuse prefetched first shard from prior cycle).
            first_shard = shard_order[start_pos]
            if start_pos == 0 and carry_prefetch is not None and carry_prefetch_shard_id == int(first_shard.shard_id):
                current_batch_it = carry_prefetch
                if prefetch_log_this_rank:
                    logger.info(
                        "ShardedParquetIterableDataset: reuse prefetched first shard (rank=%d/%d worker=%d/%d cycle=%d shard=%d).",
                        rank,
                        world_size,
                        worker_id,
                        num_workers,
                        cycle_idx,
                        int(first_shard.shard_id),
                    )
            else:
                current_batch_it = _iter_shard_record_batches(
                    _files_for(first_shard),
                    batch_rows=self.parquet_batch_rows,
                    columns=self.columns,
                    row_group_shuffle=bool(self.row_group_shuffle),
                    row_group_shuffle_block_size=int(self.row_group_shuffle_block_size),
                    row_group_shuffle_seed=int(_row_group_seed(first_shard)),
                )
            carry_prefetch = None
            carry_prefetch_shard_id = None

            prefetch_seed_shard = None
            if (start_pos + 1) < len(shard_order):
                prefetch_seed_shard = shard_order[start_pos + 1]
            elif shard_order_next:
                prefetch_seed_shard = shard_order_next[0]

            next_prefetch = _start_prefetch(prefetch_seed_shard)

            for pos in range(start_pos, len(shard_order)):
                shard = shard_order[pos]
                row_it = _make_row_it(shard, current_batch_it)
                yield from row_it

                # Shard boundary marker (consumed by dynamic prompt packing bufferizer).
                yield {SHARD_BOUNDARY_KEY: True}

                if resume_enabled:
                    next_shard_index = pos + 1
                    _save_resume_state(cycle_idx=cycle_idx, next_shard_index=next_shard_index, shard_order_ids=shard_order_ids)

                if prefetch_log_this_rank and next_prefetch is not None and next_prefetch.first_put_at is None:
                    logger.warning(
                        "ShardedParquetIterableDataset: next shard prefetch not ready at boundary "
                        "(rank=%d/%d worker=%d/%d cycle=%d shard=%d next_q=%d).",
                        rank,
                        world_size,
                        worker_id,
                        num_workers,
                        cycle_idx,
                        int(shard.shard_id),
                        int(getattr(next_prefetch._q, "qsize", lambda: -1)()),
                    )

                if isinstance(current_batch_it, _PrefetchIterator):
                    # Ensure background thread exits promptly after consumption.
                    current_batch_it.close()

                # Advance to next shard.
                if pos + 1 >= len(shard_order):
                    # End of cycle; carry prefetcher for next cycle first shard (if any).
                    if shard_order_next:
                        carry_prefetch = next_prefetch
                        carry_prefetch_shard_id = int(shard_order_next[0].shard_id)
                    else:
                        if next_prefetch is not None:
                            next_prefetch.close()
                    cycle_idx += 1
                    next_shard_index = 0
                    forced_order_ids = None
                    forced_order_cycle = None
                    _save_resume_state(cycle_idx=cycle_idx, next_shard_index=next_shard_index, shard_order_ids=None)
                    break

                next_shard = shard_order[pos + 1]
                if next_prefetch is not None:
                    current_batch_it = next_prefetch
                else:
                    current_batch_it = _iter_shard_record_batches(
                        _files_for(next_shard),
                        batch_rows=self.parquet_batch_rows,
                        columns=self.columns,
                        row_group_shuffle=bool(self.row_group_shuffle),
                        row_group_shuffle_block_size=int(self.row_group_shuffle_block_size),
                        row_group_shuffle_seed=int(_row_group_seed(next_shard)),
                    )

                # Prefetch lookahead: the shard after next, or next cycle's first shard.
                lookahead_shard = None
                if pos + 2 < len(shard_order):
                    lookahead_shard = shard_order[pos + 2]
                elif shard_order_next:
                    lookahead_shard = shard_order_next[0]
                next_prefetch = _start_prefetch(lookahead_shard)


def _iter_shard_rows(
    files: list[str],
    *,
    batch_rows: int,
    columns: list[str] | None,
) -> Iterator[dict[str, Any]]:
    for fp in files:
        if isinstance(fp, str) and fp.startswith(("s3://", "tos://")):
            yield from _iter_parquet_rows(fp, batch_rows=int(batch_rows), columns=columns)
            continue
        if not os.path.isfile(fp):
            continue
        yield from _iter_parquet_rows(fp, batch_rows=int(batch_rows), columns=columns)
