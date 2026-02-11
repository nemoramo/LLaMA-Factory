# FunAudioChat S2T 超大规模训练指南（Parquet Shard）

本指南面向 **超大规模** FunAudioChat S2T 训练（例如 10 万+小时、上亿行 JSONL），此时直接用 HuggingFace
map-style JSONL（`load_dataset("json")`）构建索引往往会非常慢，且在 `dataloader_num_workers`、prefetch、
dynamic prompt packing buffer/prefetch 较高时容易出现内存持续增长甚至 OOM。

推荐方案：

1. **离线**：把 JSONL/NDJSON 分桶写成 **N 个 Parquet shard** + `manifest.json`（一次构建，多次复用）。
2. **在线训练**：启用 **sharded parquet backend**（按 rank/worker 分 shard + 可选跨 shard 预读取）。
3. 保持 **neat packing + dynamic prompt packing**（可选）来提升有效 token 吞吐。

若你在 shard 切换处观察到明显卡顿，建议同时阅读：
`SHARDED_PARQUET_BACKEND.md`。

## 0) 术语说明

- **Shard**：分桶 id `0..num_shards-1`，输出文件 `shards/part-00000.parquet` ... `part-xxxxx.parquet`。
- **`.001.parquet`**：不是额外 shard，而是同一个 shard id 在 `--writer=partitionby` 里因为文件拆分产生的后续分片文件。

## 1) `num_shards` 怎么选

约束：

- shard 分配在 dataset 内部完成：
  1. 先按 `(rank, world_size)` 分，
  2. 再按 `(worker_id, num_workers)` 分。
- 若 `num_shards < world_size * dataloader_num_workers`，可能会出现部分 dataloader worker 分不到 shard，提前退出。

经验法则（建议先这样起步）：

- `num_shards >= world_size * dataloader_num_workers * 4`
- 例如 8 卡、`dataloader_num_workers=8`，建议 `num_shards >= 256`
- v4 级别通常用 `num_shards=384` 是一个比较稳的折中（单 shard 工作集小、轮换足够）

## 2) 离线：构建 Parquet Shards + `manifest.json`

输出目录结构：

```text
<out_dir>/
  manifest.json
  build_meta.json
  shards/
    part-00000.parquet
    part-00001.parquet
    ...
```

### 2.1 依赖与环境

- 建议使用同一套训练环境。
- 依赖：`polars`、`pyarrow`。
- 为了与 watchdog 一致并避免 `~/.local` 包干扰，建议加 `PYTHONNOUSERSITE=1`。

### 2.2 推荐命令（更快且内存更可控）

超大数据优先用 `--writer=file_parallel`（多进程按文件解析 + 统一 merge）：

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

补充说明：

- `--writer=partitionby` 在 v4 级别数据上可能会先把内存撑到很高才开始出文件，不建议。
- `--writer=chunked` 更简单但通常更慢（单进程解析 JSON）。
- 若你必须“严格只有 N 个文件（无 `.001`）”，优先用 `file_parallel` 或 `chunked`。
  若一定要用 `partitionby`，设置 `--approximate-bytes-per-file none`，并且不要设置 `--max-rows-per-file`。

### 2.3 时间预估（粗略）

耗时取决于 IO、CPU 解析 JSON、压缩：

- 如果你之前 `chunked` 全量 v4 用时 ~2.5 小时，同机 `file_parallel` 常见能提升到 ~1.3x-2.5x。
- 若 IO（尤其 fuse/network）是瓶颈，则提速会变小。

### 2.4 快速检查

```bash
python -c "import json; print(json.load(open('/data2/.../manifest.json'))['num_shards'])"
ls /data2/.../shards/part-*.parquet | wc -l
```

## 3) 在线：使用 shard 训练

### 3.1 推荐入口（watchdog）

使用 watchdog 脚本：

- `scripts/monitor_funaudiochat_s2t_training.sh`

它会在 `output_dir` 下自动保存：

- `training_command.txt`
- `config_base.yaml`

### 3.2 必要数据参数（DataArguments）

```yaml
sharded_dataset_backend: polars_parquet_shards
sharded_manifest_path: /data2/.../manifest.json
```

超大规模推荐默认值：

```yaml
sharded_shuffle_shards: true
sharded_row_group_shuffle: true
sharded_parquet_batch_rows: 8192

# 跨 shard 预读（降低 shard 切换卡顿）
sharded_prefetch_next_shard: true
sharded_prefetch_queue_batches: 1  # 仍卡就试 2-4
sharded_prefetch_log: false        # 仅排障时打开

# 粗粒度 resume（可选；watchdog 重启更稳）
sharded_resume_mode: shard_boundary
sharded_resume_prefer_checkpoint: true
sharded_resume_log: false

# 音频进度（可选；超大数据 pre-scan 可能较慢）
log_audio_epochs: true
```

### 3.3 Packing（建议）

```yaml
packing: true
neat_packing: true

dynamic_prompt_packing: true
dynamic_prompt_packing_buffer_size: 2048
dynamic_prompt_packing_prefetch_buffers: 2
dynamic_prompt_packing_carryover_packs: 2
```

注意：

- `dynamic_prompt_packing_prefetch_buffers` 越大越不易卡顿，但更吃 CPU/RAM，worker 多时更容易 OOM。
- `dynamic_prompt_packing_carryover_packs>0` 有助于跨 buffer 打包，提高 pack 效率且不会静默丢数据。

### 3.4 DataLoader（建议起步参数）

先保守起步，再逐步放大：

```yaml
dataloader_num_workers: 6
dataloader_prefetch_factor: 4
dataloader_persistent_workers: true
```

如果出现 RAM 压力 / worker OOM，优先降：

- `dataloader_num_workers`
- `dataloader_prefetch_factor`
- `dynamic_prompt_packing_prefetch_buffers`
- `sharded_prefetch_queue_batches`
- `sharded_parquet_batch_rows`

## 4) TOS / fuse mount 卡顿规避（可选）

高并发（DDP ranks x dataloader workers）下 fuse mount 可能会抖动明显。本 worktree 支持把 mount 路径映射成
`tos://...` 并通过 SDK 读取。

开启映射：

```bash
export LLAMAFACTORY_TOS_SDK_FOR_MOUNT=1
export LLAMAFACTORY_TOS_MOUNT_MAP="/mnt/asr-audio-data:asr-audio-data,/mnt/tts-data-tos:tts-data-tos"
```

如果你使用了 **软连接** 前缀（例如 `/data2/... -> /mnt/...`），也需要把该前缀加入 `LLAMAFACTORY_TOS_MOUNT_MAP`。

AK/SK 等凭证建议通过环境变量注入（通常由 `.env` source 得到；`.env` 应保持 gitignore）。

连接池（每进程）可适当增大：

```bash
export LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS=32
export LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS=32
```

## 5) 排障：shard 切换时卡顿

常见现象：

- `s/it` 周期性抖动
- DDP ranks 利用率不均，在 shard 切换处更明显

建议动作：

1. 确认 `sharded_prefetch_next_shard=true`，并把 `sharded_prefetch_queue_batches` 控制在 `1-4`。
2. 下调 `sharded_parquet_batch_rows`（单个 RecordBatch 太大时切换更慢）。
3. IO 在 fuse/network 时，优先开启 TOS SDK 映射（见第 4 节）。
4. CPU 吃满时，下调 `dataloader_num_workers` / `dataloader_prefetch_factor`。
