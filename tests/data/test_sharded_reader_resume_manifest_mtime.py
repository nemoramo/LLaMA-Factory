import json
import os
import tempfile
import time


def _write_parquet(path: str, rows: list[dict]):
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _first_data_row(ds):
    from llamafactory.data.sharded_reader import SHARD_BOUNDARY_KEY

    for row in ds:
        if isinstance(row, dict) and row.get(SHARD_BOUNDARY_KEY, False):
            continue
        return row
    return None


def test_sharded_resume_ignores_state_when_manifest_mtime_mismatches():
    from llamafactory.data.sharded_reader import ShardedParquetIterableDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        shards_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shards_dir, exist_ok=True)
        p0 = os.path.join(shards_dir, "part-00000.parquet")
        p1 = os.path.join(shards_dir, "part-00001.parquet")
        _write_parquet(p0, [{"id": "s0"}])
        _write_parquet(p1, [{"id": "s1"}])

        manifest = {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "seed": 42,
            "num_shards": 2,
            "backend": "test",
            "shards": [
                {"shard_id": 0, "files": ["shards/part-00000.parquet"], "total_bytes": int(os.stat(p0).st_size)},
                {"shard_id": 1, "files": ["shards/part-00001.parquet"], "total_bytes": int(os.stat(p1).st_size)},
            ],
        }
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

        st = os.stat(manifest_path)
        manifest_mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))

        state_dir = os.path.join(tmpdir, "shard_resume_state")
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, "rank00000_worker000.json")
        state = {
            "version": 1,
            "manifest_path": os.path.abspath(manifest_path),
            "manifest_mtime_ns": int(manifest_mtime_ns) - 1,  # mismatch
            "seed": 42,
            "shuffle_shards": False,
            "rank": 0,
            "world_size": 1,
            "worker_id": 0,
            "num_workers": 1,
            "cycle_idx": 0,
            "next_shard_index": 1,  # would start from shard 1 if accepted
            "shards_worker_ids": [0, 1],
            "updated_at_unix_s": float(time.time()),
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

        old_rank = os.environ.get("RANK")
        old_world_size = os.environ.get("WORLD_SIZE")
        try:
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            ds = ShardedParquetIterableDataset(
                manifest_path=manifest_path,
                seed=42,
                shuffle_shards=False,
                row_shuffle_buffer=0,
                row_group_shuffle=False,
                row_group_shuffle_block_size=0,
                parquet_batch_rows=1,
                prefetch_next_shard=False,
                prefetch_queue_batches=0,
                prefetch_log=False,
                resume_mode="shard_boundary",
                resume_state_dir=None,
                resume_prefer_checkpoint=True,
                resume_log=True,
                output_dir=tmpdir,
                resume_from_checkpoint=None,
            )

            row = _first_data_row(iter(ds))
            assert isinstance(row, dict)
            assert row.get("id") == "s0"
        finally:
            if old_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = old_rank
            if old_world_size is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = old_world_size


def test_sharded_resume_accepts_state_when_manifest_mtime_matches():
    from llamafactory.data.sharded_reader import ShardedParquetIterableDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        shards_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shards_dir, exist_ok=True)
        p0 = os.path.join(shards_dir, "part-00000.parquet")
        p1 = os.path.join(shards_dir, "part-00001.parquet")
        _write_parquet(p0, [{"id": "s0"}])
        _write_parquet(p1, [{"id": "s1"}])

        manifest = {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "seed": 42,
            "num_shards": 2,
            "backend": "test",
            "shards": [
                {"shard_id": 0, "files": ["shards/part-00000.parquet"], "total_bytes": int(os.stat(p0).st_size)},
                {"shard_id": 1, "files": ["shards/part-00001.parquet"], "total_bytes": int(os.stat(p1).st_size)},
            ],
        }
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

        st = os.stat(manifest_path)
        manifest_mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))

        state_dir = os.path.join(tmpdir, "shard_resume_state")
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, "rank00000_worker000.json")
        state = {
            "version": 1,
            "manifest_path": os.path.abspath(manifest_path),
            "manifest_mtime_ns": int(manifest_mtime_ns),  # match
            "seed": 42,
            "shuffle_shards": False,
            "rank": 0,
            "world_size": 1,
            "worker_id": 0,
            "num_workers": 1,
            "cycle_idx": 0,
            "next_shard_index": 1,
            "shards_worker_ids": [0, 1],
            "updated_at_unix_s": float(time.time()),
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

        old_rank = os.environ.get("RANK")
        old_world_size = os.environ.get("WORLD_SIZE")
        try:
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            ds = ShardedParquetIterableDataset(
                manifest_path=manifest_path,
                seed=42,
                shuffle_shards=False,
                row_shuffle_buffer=0,
                row_group_shuffle=False,
                row_group_shuffle_block_size=0,
                parquet_batch_rows=1,
                prefetch_next_shard=False,
                prefetch_queue_batches=0,
                prefetch_log=False,
                resume_mode="shard_boundary",
                resume_state_dir=None,
                resume_prefer_checkpoint=True,
                resume_log=True,
                output_dir=tmpdir,
                resume_from_checkpoint=None,
            )

            row = _first_data_row(iter(ds))
            assert isinstance(row, dict)
            assert row.get("id") == "s1"
        finally:
            if old_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = old_rank
            if old_world_size is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = old_world_size
