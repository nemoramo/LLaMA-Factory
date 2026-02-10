# Sharded Parquet: Cross-Shard Prefetch (DDP Stall Fix)

## Problem

When training from Parquet shards, shard boundaries can create visible throughput jitter:

- `ShardedParquetIterableDataset` finishes a shard and then synchronously opens/reads the next shard.
- Dynamic prompt packing has a bounded prefetch queue; if the queue drains near a shard boundary, all ranks can stall.
- In DDP, the slowest rank blocks collective ops, so shard-switch stalls amplify into global step-time spikes.

## Solution

Enable **next-shard prefetch** inside `ShardedParquetIterableDataset`:

- Each dataloader worker starts a background thread to prefetch **RecordBatches** from the *next* shard.
- The queue is bounded by `sharded_prefetch_queue_batches` (keep it small).
- Shard boundary marker is preserved: `{ "__llamafactory_shard_boundary__": True }`.
  Dynamic prompt packing still flushes at shard boundaries; prefetch only overlaps IO/decompression for the next shard.

Implementation:

- Code: `src/llamafactory/data/sharded_reader.py`
- Wiring: `src/llamafactory/data/loader.py` reads `DataArguments` and passes knobs into the dataset.
- Args: `src/llamafactory/hparams/data_args.py`

## Knobs

### Sharded backend knobs (DataArguments)

- `sharded_prefetch_next_shard` (bool, default `true`):
  Enable background prefetch for the next shard.
- `sharded_prefetch_queue_batches` (int, default `1`):
  Queue size in number of Parquet RecordBatches (per dataloader worker).
  Recommended range: `1-4`. Keep small to avoid RAM spikes.
- `sharded_prefetch_log` (bool, default `false`):
  Enable rank0 log lines to diagnose shard prefetch behavior.

Existing related knobs:

- `sharded_parquet_batch_rows`:
  Parquet `iter_batches(batch_size=...)`. If rows are large, reduce this first.
- `sharded_row_shuffle_buffer`:
  Adds per-shard row shuffle buffering (more RAM).

### Dataloader knobs (TrainingArguments)

- `dataloader_num_workers`, `dataloader_prefetch_factor`, `dataloader_persistent_workers`

Rule of thumb:

- If `num_workers` is high and each row is large, the *product* of:
  `num_workers * dataloader_prefetch_factor * (packing buffers)` can be the dominant CPU/RAM consumer.
- Next-shard prefetch is intentionally small (RecordBatches, queue size 1 by default), but it still adds extra IO and
  some additional in-memory Arrow buffers.

## Observability

When `sharded_prefetch_log=true`, rank0 will print:

- `prefetch start ...` when the worker starts prefetching a shard.
- `reuse prefetched first shard ...` at cycle boundary (prefetch across epoch/cycle boundary).
- `shard enter ...` when consuming a prefetched shard, including:
  `prefetch_ready_s` and `first_get_wait_s`.
- Warning if prefetch is not ready at shard boundary (queue empty and `first_put_at` not set yet).

## Notes / Semantics

- Shard distribution is done **inside** `ShardedParquetIterableDataset`:
  shards are partitioned by `(rank, world_size)` and then by `(worker_id, num_workers)`.
  Do not apply another `DistributedSampler` on top of this iterable dataset.
- If `shards_per_rank < num_workers`, some workers will have no shards and will exit early.
  In that case, reduce `dataloader_num_workers` to avoid idle workers.

