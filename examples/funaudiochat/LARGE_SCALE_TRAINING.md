# Large-Scale FunAudioChat S2T Training (Parquet Shards)

This guide targets **very large** FunAudioChat S2T runs (e.g. 100k+ audio hours, 10^8+ rows),
where building a full HuggingFace map-style JSONL index is slow and/or can OOM with high
`dataloader_num_workers` + prefetch + dynamic prompt packing.

The approach is:

1. **Offline**: shard JSONL/NDJSON into **N Parquet shard files** + `manifest.json`.
2. **Online**: use the **sharded parquet backend** (rank/worker shard partitioning + optional cross-shard prefetch).
3. Keep **neat packing + dynamic prompt packing** (optional) for training efficiency.

If you are debugging stalls near shard boundaries, also read:
`SHARDED_PARQUET_BACKEND.md`.

## 0) Terminology

- **Shard**: one bucket id `0..num_shards-1` written to `shards/part-00000.parquet` ... `part-xxxxx.parquet`.
- **`.001.parquet`**: not a new shard id. It's a *file split* within the same shard id when using `--writer=partitionby`
  with `--approximate-bytes-per-file auto` (or `--max-rows-per-file`).

## 1) Choosing `num_shards`

Constraints:

- Shards are partitioned **inside** the iterable dataset:
  1. split shards by `(rank, world_size)`, then
  2. split again by `(worker_id, num_workers)`.
- If `num_shards < world_size * dataloader_num_workers`, some dataloader workers can be assigned zero shards and will exit early.

Practical rule of thumb:

- Start with `num_shards >= world_size * dataloader_num_workers * 4`.
- For 8 GPUs and `dataloader_num_workers=8`, this suggests `num_shards >= 256`.
- For FunAudioChat v4 scale, `num_shards=384` is a common, stable tradeoff (smaller per-shard working set, enough rotation).

## 2) Offline: Build Parquet Shards + `manifest.json`

Output structure:

```text
<out_dir>/
  manifest.json
  build_meta.json
  shards/
    part-00000.parquet
    part-00001.parquet
    ...
```

### 2.1 Prereqs

- Use the same Python env as training if possible.
- Required packages: `polars`, `pyarrow`.
- For consistent behavior with watchdog runs, prefer running with `PYTHONNOUSERSITE=1` to avoid `~/.local` shadowing.

### 2.2 Recommended command (fast + bounded memory)

For huge datasets, prefer `--writer=file_parallel`:

```bash
conda run -n llamafactory env PYTHONNOUSERSITE=1 python \
  scripts/shard_jsonl_to_parquet.py build \
  --writer file_parallel \
  --num-workers 16 \
  --chunk-rows 100000 \
  --num-shards 384 \
  --seed 42 \
  --compression zstd \
  --no-statistics \
  --ignore-errors \
  --progress-interval-sec 60 \
  --file-parallel-report-bytes $((1024*1024*1024)) \
  --output /data2/mayufeng/manifests/llama_data_shards/<NEW_OUT_DIR> \
  --input /path/to/train.jsonl
```

Notes:

- `--writer=partitionby` can grow RSS massively before producing output on very large inputs; avoid for v4-scale data.
- `--writer=chunked` is simplest and stable but usually slower (single-process JSON parsing).
- If you want **exactly N shard files** (no `.001` splits), use `file_parallel` or `chunked`.
  If you must use `partitionby`, set `--approximate-bytes-per-file none` and keep `--max-rows-per-file` unset.

### 2.3 Runtime expectation (rough)

Wall time depends on IO + CPU JSON parsing + compression:

- If your old `chunked` run took ~2.5 hours on the same node, `file_parallel` often improves by ~1.3x-2.5x.
- If you are IO-bound (network/fuse mount), speedup will be smaller.

### 2.4 Sanity checks

```bash
python -c "import json; print(json.load(open('/data2/.../manifest.json'))['num_shards'])"
ls /data2/.../shards/part-*.parquet | wc -l
```

## 3) Online: Train From Parquet Shards

### 3.1 Watchdog entrypoint (recommended)

Use the sharded watchdog script:

- `scripts/monitor_funaudiochat_s2t_training.sh`

It will save:

- `training_command.txt`
- `config_base.yaml`

under your `output_dir`.

### 3.2 Required dataset knobs (DataArguments)

```yaml
sharded_dataset_backend: polars_parquet_shards
sharded_manifest_path: /data2/.../manifest.json
```

Recommended defaults for large-scale runs:

```yaml
sharded_shuffle_shards: true
sharded_row_group_shuffle: true
sharded_parquet_batch_rows: 8192

# Cross-shard prefetch (DDP stall fix)
sharded_prefetch_next_shard: true
sharded_prefetch_queue_batches: 1  # try 2-4 if boundary stalls persist
sharded_prefetch_log: false        # set true only when diagnosing

# Coarse resume (optional; useful with watchdog restarts)
sharded_resume_mode: shard_boundary
sharded_resume_prefer_checkpoint: true
sharded_resume_log: false

# Audio progress (optional; can be expensive to pre-scan durations)
log_audio_epochs: true
```

### 3.3 Packing knobs (recommended)

```yaml
packing: true
neat_packing: true

dynamic_prompt_packing: true
dynamic_prompt_packing_buffer_size: 2048
dynamic_prompt_packing_prefetch_buffers: 2
dynamic_prompt_packing_carryover_packs: 2
```

Notes:

- Larger `dynamic_prompt_packing_prefetch_buffers` reduces stalls but increases CPU/RAM and can trigger OOM with high workers.
- `dynamic_prompt_packing_carryover_packs>0` helps packing across buffer boundaries with minimal order changes.

### 3.4 DataLoader knobs (TrainingArguments)

Start conservative, then scale:

```yaml
dataloader_num_workers: 6
dataloader_prefetch_factor: 4
dataloader_persistent_workers: true
```

If you see RAM pressure / worker OOM, reduce:

- `dataloader_num_workers`
- `dataloader_prefetch_factor`
- `dynamic_prompt_packing_prefetch_buffers`
- `sharded_prefetch_queue_batches`
- `sharded_parquet_batch_rows`

## 4) TOS / Fuse Mount Stall Mitigation (Optional)

High concurrency (DDP ranks x dataloader workers) can make fuse mounts jittery.
This worktree supports mapping mount paths to `tos://...` URIs and reading via SDK.

Enable mount-path mapping:

```bash
export LLAMAFACTORY_TOS_SDK_FOR_MOUNT=1
export LLAMAFACTORY_TOS_MOUNT_MAP="/mnt/asr-audio-data:asr-audio-data,/mnt/tts-data-tos:tts-data-tos"
```

If you use **symlinked** prefixes like `/data2/... -> /mnt/...`, add those prefixes to `LLAMAFACTORY_TOS_MOUNT_MAP` too.

Credentials must be provided via environment (often sourced from `.env`, which should stay gitignored).

Connection pool tuning (per-process):

```bash
export LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS=32
export LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS=32
```

## 5) Troubleshooting: shard boundary stalls

Symptoms:

- `s/it` spikes periodically.
- DDP ranks become imbalanced at shard switches.

Actions:

1. Enable `sharded_prefetch_next_shard=true` and keep `sharded_prefetch_queue_batches` small (`1-4`).
2. Reduce `sharded_parquet_batch_rows` (large rows can make each RecordBatch heavy).
3. If IO is remote/fuse, enable TOS SDK mapping (Section 4).
4. If CPU is saturated, reduce `dataloader_num_workers` and/or `dataloader_prefetch_factor`.
