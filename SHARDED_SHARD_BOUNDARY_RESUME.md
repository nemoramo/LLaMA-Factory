# Sharded Parquet: Shard-Boundary Resume (Cycle-Aware)

This worktree adds a **coarse resume** mechanism for the sharded parquet backend
(`ShardedParquetIterableDataset`) to avoid expensive re-reading/skipping of huge
datasets after restarts.

It is designed for the FunAudioChat S2T sharded pipeline with dynamic prompt
packing (IterableDataset + shard-boundary flush markers).

## What Problem This Solves

When training exits and restarts (OOM, watchdog restart, node issues), resuming
from a checkpoint typically requires **skipping already-consumed data**.

For an `IterableDataset` with multi-worker prefetch + on-the-fly packing, exact
row-level resume is expensive and brittle. With very large datasets (e.g.
hundreds of thousands of audio-hours), naive skipping can still require large
amounts of JSONL/parquet scanning and can cause OOM/slow restarts.

This feature enables **resume at shard boundaries**:

- If a restart happens, each `(rank, dataloader_worker)` can skip **whole shards
  already completed**, and continue from the next shard.
- It is **cycle-aware** (a "cycle" here is one full assignment of all shards
  for that rank/worker). When a cycle finishes, it increments `cycle_idx` and
  starts the next cycle with a new deterministic shuffle order.

## How It Works

### State Files

Each dataloader worker persists a small JSON cursor at shard boundaries:

- Default write path: `<output_dir>/shard_resume_state/`
- File name: `rank00000_worker000.json` (rank/worker are zero-padded)

Each state contains:

- `cycle_idx`: which shard cycle we are in (0-based)
- `next_shard_index`: index into this cycle's shard order (0..len(shards))
- `shard_order_ids`: optional; the shard-id order for the current cycle
- `shards_worker_ids`: assigned shard IDs for this rank/worker (compat check)
- `manifest_path`, `seed`, `world_size`, `num_workers`, ...

### Loading State On Resume

If enabled, the dataset loads state at iterator start:

1. Prefer `<resume_from_checkpoint>/shard_resume_state/` when available
   (`sharded_resume_prefer_checkpoint=true`).
2. Otherwise fall back to `<output_dir>/shard_resume_state/`.
3. Validate compatibility (manifest path, seed, world_size, num_workers, etc).
   On mismatch, state is ignored and iteration starts from cycle 0 / shard 0.

### Checkpoint Snapshots (Important)

To reduce the risk of resuming from a state that is "ahead" due to dataloader
prefetch, the trainer snapshots the current resume-state directory into each
checkpoint:

- `<checkpoint_dir>/shard_resume_state/`

This snapshot is done by `CustomSeq2SeqTrainer._save_checkpoint()`.

## Configuration

### DataArguments

- `sharded_resume_mode`: `off` (default) or `shard_boundary`
- `sharded_resume_state_dir`: custom write dir (default: `<output_dir>/shard_resume_state`)
- `sharded_resume_prefer_checkpoint`: prefer checkpoint snapshot on resume (default: `true`)
- `sharded_resume_log`: rank0 logging for debugging (default: `false`)

### Environment (Trainer Snapshot)

The trainer uses:

- `LLAMAFACTORY_SHARDED_RESUME_STATE_DIR`

If set, it snapshots from that directory; otherwise it snapshots from
`<output_dir>/shard_resume_state`.

The sharded loader sets this env var automatically when `sharded_resume_mode != off`.

### Watchdog Script

`scripts/monitor_funaudiochat_s2t_training.sh` supports:

- `SHARDED_RESUME_MODE=off|shard_boundary`
- `SHARDED_RESUME_STATE_DIR=...` (optional)
- `SHARDED_RESUME_PREFER_CHECKPOINT=true|false`
- `SHARDED_RESUME_LOG=true|false`

## Operational Notes / Limitations

- This is **coarse** resume: it only guarantees "resume from the next shard".
  If a crash happens mid-shard, the next run will re-read the current shard and
  may repeat some examples within that shard.
- For best behavior, pair this with `ignore_data_skip=true` so HF Trainer does
  not perform additional skipping that could "double skip".
- `world_size` and `dataloader_num_workers` should stay the same across resume.
  If they change, the assigned shard set changes; state is ignored.
- The shard shuffle order is deterministic by `(seed + cycle_idx)` and can be
  persisted via `shard_order_ids` for extra safety.

## Where The Code Lives

- Dataset + state logic:
  - `src/llamafactory/data/sharded_reader.py`
- Loader wiring (passes args, sets env var):
  - `src/llamafactory/data/loader.py`
- Checkpoint snapshot:
  - `src/llamafactory/train/sft/trainer.py`

