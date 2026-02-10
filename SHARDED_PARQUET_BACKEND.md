# Sharded Parquet Backend (Polars Shards): Prefetch + Shard-Boundary Resume

This worktree uses a sharded Parquet backend for very large datasets (e.g. FunAudioChat S2T),
where building a full HuggingFace map-style JSONL index is too slow or too memory-hungry.

Two key runtime features are documented here:

- **Cross-shard prefetch**: reduce DDP step-time spikes at shard boundaries.
- **Shard-boundary resume (cycle-aware)**: after a restart, skip whole shards already completed by each
  `(rank, dataloader_worker)` to avoid expensive re-reading/skipping.

The backend name is `polars_parquet_shards` (manifest produced offline; runtime reading is via PyArrow).

## 1) Cross-Shard Prefetch (DDP Stall Fix)

### Problem

When training from Parquet shards, shard boundaries can create visible throughput jitter:

- `ShardedParquetIterableDataset` finishes a shard and then synchronously opens/reads the next shard.
- Dynamic prompt packing has a bounded prefetch queue; if the queue drains near a shard boundary, all ranks can stall.
- In DDP, the slowest rank blocks collective ops, so shard-switch stalls amplify into global step-time spikes.

### Solution

Enable **next-shard prefetch** inside `ShardedParquetIterableDataset`:

- Each dataloader worker starts a background thread to prefetch **RecordBatches** from the *next* shard.
- The queue is bounded by `sharded_prefetch_queue_batches` (keep it small).
- Shard boundary marker is preserved: `{ "__llamafactory_shard_boundary__": True }`.
  Dynamic prompt packing still flushes at shard boundaries; prefetch only overlaps IO/decompression for the next shard.

Implementation:

- Code: `src/llamafactory/data/sharded_reader.py`
- Wiring: `src/llamafactory/data/loader.py`
- Args: `src/llamafactory/hparams/data_args.py`

### Knobs

Sharded backend knobs (DataArguments):

- `sharded_prefetch_next_shard` (bool, default `true`): enable background prefetch for the next shard.
- `sharded_prefetch_queue_batches` (int, default `1`): queue size in number of Parquet RecordBatches (per dataloader worker).
  Recommended range: `1-4`. Keep small to avoid RAM spikes.
- `sharded_prefetch_log` (bool, default `false`): enable rank0 log lines to diagnose shard prefetch behavior.

Existing related knobs:

- `sharded_parquet_batch_rows`: Parquet `iter_batches(batch_size=...)`. If rows are large, reduce this first.
- `sharded_row_shuffle_buffer`: adds per-shard row shuffle buffering (more RAM).

Dataloader knobs (TrainingArguments):

- `dataloader_num_workers`, `dataloader_prefetch_factor`, `dataloader_persistent_workers`

Rule of thumb:

- If `num_workers` is high and each row is large, the *product* of:
  `num_workers * dataloader_prefetch_factor * (packing buffers)` can dominate CPU/RAM.
- Next-shard prefetch is intentionally small (RecordBatches, queue size 1 by default), but still adds extra IO and
  some additional in-memory Arrow buffers.

### Observability

When `sharded_prefetch_log=true`, rank0 will print:

- `prefetch start ...` when the worker starts prefetching a shard.
- `reuse prefetched first shard ...` at cycle boundary (prefetch across epoch/cycle boundary).
- `shard enter ...` when consuming a prefetched shard, including:
  `prefetch_ready_s` and `first_get_wait_s`.
- Warning if prefetch is not ready at shard boundary (queue empty and `first_put_at` not set yet).

## 2) Shard-Boundary Resume (Cycle-Aware)

This is a **coarse** resume mechanism for the sharded parquet backend (`ShardedParquetIterableDataset`).

### What Problem This Solves

When training exits and restarts (OOM, watchdog restart, node issues), resuming
from a checkpoint typically requires **skipping already-consumed data**.

For an `IterableDataset` with multi-worker prefetch + on-the-fly packing, exact
row-level resume is expensive and brittle. With very large datasets, naive skipping
can still require large amounts of parquet scanning and can cause OOM/slow restarts.

This feature enables **resume at shard boundaries**:

- After restart, each `(rank, dataloader_worker)` skips **whole shards already completed** and continues from the next shard.
- It is **cycle-aware**: a "cycle" here is one full assignment of all shards for that rank/worker. When a cycle finishes,
  it increments `cycle_idx` and starts the next cycle with a new deterministic shuffle order.

### How It Works

State files:

- Default write path: `<output_dir>/shard_resume_state/`
- File name: `rank00000_worker000.json` (rank/worker are zero-padded)

Each state contains:

- `cycle_idx`: which shard cycle we are in (0-based)
- `next_shard_index`: index into this cycle's shard order (0..len(shards))
- `shard_order_ids`: optional; the shard-id order for the current cycle
- `shards_worker_ids`: assigned shard IDs for this rank/worker (compat check)
- `manifest_path`, `seed`, `world_size`, `num_workers`, ...

Loading state on resume:

1. Prefer `<resume_from_checkpoint>/shard_resume_state/` when available (`sharded_resume_prefer_checkpoint=true`).
2. Otherwise fall back to `<output_dir>/shard_resume_state/`.
3. Validate compatibility (manifest path, seed, world_size, num_workers, etc). On mismatch, ignore state and start from
   cycle 0 / shard 0.

Checkpoint snapshots (important):

- To reduce risk that the resume state is "ahead" due to dataloader prefetch, the trainer snapshots the current
  resume-state directory into each checkpoint:
  - `<checkpoint_dir>/shard_resume_state/`
- Snapshot is done by `CustomSeq2SeqTrainer._save_checkpoint()`.

### Configuration

DataArguments:

- `sharded_resume_mode`: `off` (default) or `shard_boundary`
- `sharded_resume_state_dir`: custom write dir (default: `<output_dir>/shard_resume_state`)
- `sharded_resume_prefer_checkpoint`: prefer checkpoint snapshot on resume (default: `true`)
- `sharded_resume_log`: rank0 logging for debugging (default: `false`)

Environment (trainer snapshot):

- `LLAMAFACTORY_SHARDED_RESUME_STATE_DIR`

If set, trainer snapshots from that directory; otherwise snapshots from `<output_dir>/shard_resume_state`.
The sharded loader sets this env var automatically when `sharded_resume_mode != off`.

Watchdog script:

- `scripts/monitor_funaudiochat_s2t_training.sh` supports:
  - `SHARDED_RESUME_MODE=off|shard_boundary`
  - `SHARDED_RESUME_STATE_DIR=...` (optional)
  - `SHARDED_RESUME_PREFER_CHECKPOINT=true|false`
  - `SHARDED_RESUME_LOG=true|false`

## 3) Semantics / Operational Notes

- Shard distribution is done **inside** `ShardedParquetIterableDataset`: shards are partitioned by `(rank, world_size)`
  and then by `(worker_id, num_workers)`.
  Do not apply another `DistributedSampler` on top of this iterable dataset.
- If `shards_per_rank < num_workers`, some workers will have no shards and will exit early.
  In that case, reduce `dataloader_num_workers` to avoid idle workers.
- Shard-boundary resume is **coarse**: it guarantees "resume from the next shard". If a crash happens mid-shard,
  the next run may repeat some examples within that shard.
- For best behavior, pair shard-boundary resume with `ignore_data_skip=true` so the Trainer does not perform additional
  skipping that could effectively "double skip".

## 4) Quick Start (Watchdog)

Example (only the sharded-related knobs shown):

```bash
export SHARDED_DATASET_BACKEND=polars_parquet_shards
export SHARDED_MANIFEST_PATH=/data2/.../manifest.json

export SHARDED_PREFETCH_NEXT_SHARD=true
export SHARDED_PREFETCH_QUEUE_BATCHES=1
export SHARDED_PREFETCH_LOG=false

export SHARDED_RESUME_MODE=shard_boundary
export SHARDED_RESUME_PREFER_CHECKPOINT=true
export SHARDED_RESUME_LOG=false

./scripts/monitor_funaudiochat_s2t_training.sh
```

