# QPS Qwen3 Pressure Test (vLLM)

这个目录提供一套 **本地优先** 的 QPS 压测流程，默认适用于 Linux GPU 机器上的 Docker + vLLM。

当前方案压测的是 **vLLM OpenAI 兼容接口**（`/v1/completions`），不是 gRPC 封装层。

如果你仍然需要在 SageMaker Studio 中运行，也可以切换到兼容模式：`RUN_ENV=sagemaker`。

## Step by step

最短本地路径：

1. 进入目录并构建 bench 镜像

```bash
cd deploy/qps_qwen3_pressure_test
bash scripts/build_bench_image.sh
```

2. 准备 bench 数据

```bash
MODEL_PATH=/abs/path/to/model

python3 scripts/prepare_endpointing_bench.py \
  --input /abs/path/to/test.jsonl \
  --output data/bench_endpointing_512.jsonl \
  --tokenizer "${MODEL_PATH}" \
  --trust-remote-code \
  --max-prompt-len 512
```

如果宿主机没有安装 `transformers`，可以直接复用 bench 镜像来生成数据：

```bash
MODEL_PATH=/abs/path/to/model

docker run --rm --entrypoint python3 --network host \
  -v "$(pwd):/work" \
  -v "${MODEL_PATH}:/model:ro" \
  vllm-bench:latest \
  /work/scripts/prepare_endpointing_bench.py \
    --input /work/your_input.jsonl \
    --output /work/data/bench_endpointing_512.jsonl \
    --tokenizer /model \
    --trust-remote-code \
    --max-prompt-len 512
```

3. 启动本地 vLLM

```bash
MODEL_PATH=/abs/path/to/model \
GPU_IDS=device=0 \
bash scripts/run_server.sh start \
  --served-model-name endpointing-qwen3-0.6b-ft

bash scripts/run_server.sh wait-ready
```

4. 跑 warm 压测

```bash
MODEL_PATH=/abs/path/to/model \
bash scripts/run_bench.sh warm \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft
```

5. 汇总结果

```bash
python3 scripts/summarize_results.py
```

如果你要看“这台机器在 `p99 <= 60ms` 条件下最多能扛多少 `req/s`”，建议单独跑一组 sweep：

```bash
MODEL_PATH=/abs/path/to/model \
RESULTS_DIR="$(pwd)/results/p99_60_sweep" \
bash scripts/run_bench.sh sweep \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft \
  --rates 200,250,300,350,400,500,600,700,800

python3 scripts/summarize_results.py \
  --results-dir "$(pwd)/results/p99_60_sweep" \
  --latency-metric p99 \
  --latency-threshold-ms 60
```

## 目录结构

```text
deploy/qps_qwen3_pressure_test/
  README.md
  Dockerfile.bench
  scripts/
    build_bench_image.sh
    prepare_endpointing_bench.py
    run_server.sh
    run_bench.sh
    summarize_results.py
  data/
  results/
```

## 本地环境要求

- Linux GPU 主机
- Docker 可用
- 能运行 `vllm/vllm-openai`
- 有本地模型目录和 tokenizer 目录

说明：

- 本地模式默认使用 `--network host`，因此更适合 Linux。
- 如果 `TOKENIZER_PATH` 不单独指定，默认会直接使用 `MODEL_PATH`。
- 结果判定仍然沿用现有口径：重点看 `request_throughput`、`request_goodput`、`p50/p90/p99 e2el` 和 `failed`。

## 输入数据格式

bench 数据准备脚本要求输入与训练数据同形态的 JSONL，每行一个 JSON 对象，至少满足以下之一：

- `context`: 对话上下文文本
- `messages`: 消息列表，脚本会回溯最后一条 user message 作为 context

可选字段：

- `lang`
- `label`
- `dialogue_id`
- `turn`

示例：

```json
{"context": "hello there", "lang": "en", "label": "<EOU>", "dialogue_id": "123", "turn": 1}
```

或：

```json
{"messages": [{"role": "user", "content": "hello there"}], "lang": "en", "label": "<EOU>"}
```

## 本地 Quickstart

先进入目录：

```bash
cd deploy/qps_qwen3_pressure_test
```

### 1. 构建 bench 镜像

```bash
bash scripts/build_bench_image.sh
```

### 2. 生成 bench 数据集

```bash
MODEL_PATH=/abs/path/to/model

python3 scripts/prepare_endpointing_bench.py \
  --input /abs/path/to/test.jsonl \
  --output data/bench_endpointing_512.jsonl \
  --tokenizer "${MODEL_PATH}" \
  --trust-remote-code \
  --max-prompt-len 512
```

如果宿主机没有 `transformers`，可以改用容器执行：

```bash
MODEL_PATH=/abs/path/to/model

docker run --rm --entrypoint python3 --network host \
  -v "$(pwd):/work" \
  -v "${MODEL_PATH}:/model:ro" \
  vllm-bench:latest \
  /work/scripts/prepare_endpointing_bench.py \
    --input /work/your_input.jsonl \
    --output /work/data/bench_endpointing_512.jsonl \
    --tokenizer /model \
    --trust-remote-code \
    --max-prompt-len 512
```

### 3. 启动本地 vLLM 服务

单卡示例：

```bash
MODEL_PATH=/abs/path/to/model \
GPU_IDS=device=0 \
bash scripts/run_server.sh start \
  --served-model-name endpointing-qwen3-0.6b-ft
```

查看状态与健康检查：

```bash
bash scripts/run_server.sh status
bash scripts/run_server.sh health
bash scripts/run_server.sh wait-ready
```

### 4. 执行压测

warm steady-state：

```bash
MODEL_PATH=/abs/path/to/model \
bash scripts/run_bench.sh warm \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft
```

burst：

```bash
MODEL_PATH=/abs/path/to/model \
bash scripts/run_bench.sh burst \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft
```

cold start：

```bash
MODEL_PATH=/abs/path/to/model \
bash scripts/run_server.sh restart \
  --served-model-name endpointing-qwen3-0.6b-ft

MODEL_PATH=/abs/path/to/model \
bash scripts/run_bench.sh cold \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft
```

sweep（找 `p90 <= 60ms` 的最大 QPS）：

```bash
MODEL_PATH=/abs/path/to/model \
bash scripts/run_bench.sh sweep \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft \
  --rates 200,250,300,350,400,500
```

### 5. 汇总结果

```bash
python3 scripts/summarize_results.py
```

如果目标是找“`p99 <= 60ms` 时的最大 `req/s`”，推荐单独用一个结果目录跑 sweep，再按 `p99` 阈值汇总：

```bash
MODEL_PATH=/abs/path/to/model \
RESULTS_DIR="$(pwd)/results/p99_60_sweep" \
bash scripts/run_bench.sh sweep \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft \
  --rates 200,250,300,350,400,500,600,700,800

python3 scripts/summarize_results.py \
  --results-dir "$(pwd)/results/p99_60_sweep" \
  --latency-metric p99 \
  --latency-threshold-ms 60
```

脚本会输出一行类似：

```text
Best (p99<=60.0ms): scenario=sweep_c32_r500.json, req/s=498.7, p99=58.3ms, goodput=498.7
```

默认结果目录：

- `deploy/qps_qwen3_pressure_test/results`

默认文件命名：

- `warm_c32_r800.json`
- `burst_c32_r800_b03.json`
- `cold_c16_r200.json`
- `sweep_c32_rXXX.json`

## 常用可调参数

### 服务侧

- `RUN_ENV=local|sagemaker`，默认 `local`
- `MODEL_PATH`
- `GPU_IDS`，例如 `all` 或 `device=0`
- `VLLM_HOST`，默认 `0.0.0.0`
- `VLLM_PORT`，默认 `8000`
- `SERVED_MODEL_NAME`

### bench 侧

- `DATASET`
- `RESULTS_DIR`
- `TOKENIZER_PATH`，默认跟随 `MODEL_PATH`
- `BENCH_HOST`，本地模式默认 `127.0.0.1`；`RUN_ENV=sagemaker` 时默认改为服务容器名 `vllm-endpointing`
- `SERVER_CONTAINER_NAME`，用于 bench 容器在 `RUN_ENV=sagemaker` 下解析 vLLM 服务，默认 `vllm-endpointing`
- `BENCH_PORT`，默认 `8000`
- `REQUEST_RATE`
- `MAX_CONCURRENCY`
- `NUM_PROMPTS`
- `NUM_WARMUPS`
- `BURSTINESS`
- `RATES`

示例：把 warm 改成较小的本地 smoke：

```bash
MODEL_PATH=/abs/path/to/model \
REQUEST_RATE=100 \
MAX_CONCURRENCY=8 \
NUM_PROMPTS=500 \
NUM_WARMUPS=50 \
bash scripts/run_bench.sh warm \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft
```

## 已有服务场景

如果 vLLM 已经在本地 `127.0.0.1:${BENCH_PORT}` 跑起来了，可以直接跳过 `run_server.sh`，只跑 bench：

```bash
TOKENIZER_PATH=/abs/path/to/model \
BENCH_HOST=127.0.0.1 \
BENCH_PORT=8000 \
bash scripts/run_bench.sh warm \
  --dataset "$(pwd)/data/bench_endpointing_512.jsonl" \
  --served-model-name endpointing-qwen3-0.6b-ft
```

## SageMaker 兼容模式

如果要继续在 SageMaker Studio 中运行：

```bash
export RUN_ENV=sagemaker
```

兼容模式下：

- `build_bench_image.sh` 默认使用 `--network sagemaker`
- `run_server.sh` 和 `run_bench.sh` 默认使用 `--network sagemaker`
- `run_bench.sh` 默认会把请求发到同网络中的 `vllm-endpointing:8000`
- `run_server.sh health` / `wait-ready` 会自动探测 `vllm-endpointing` 在 `sagemaker` 网络内的容器 IP
- 你仍然需要自行满足 SageMaker 对 Docker 路径、网络和权限的约束

推荐同时显式传这些路径，而不是依赖默认值：

- `MODEL_PATH`
- `TOKENIZER_PATH`
- `DATASET`
- `RESULTS_DIR`

## 停止服务

```bash
bash scripts/run_server.sh stop
```

## 常见问题

### 1. 本地为什么默认用 `--network host`？

因为 bench 容器和 vLLM 容器都可以直接复用宿主机网络，最简单，且不需要再单独处理端口映射和容器名解析。

### 2. 为什么 bench 还要 `TOKENIZER_PATH`？

`vllm bench serve` 需要 tokenizer 来处理 prompt/token 长度统计。默认会直接复用 `MODEL_PATH`。

### 3. 如果三个 special token 约束输出异常怎么办？

这套 QPS 工具只负责压测 `/v1/completions` 的延迟和吞吐，不负责额外校验模型 export 质量。  
如果你怀疑 endpointing 标签行为异常，先用对应的离线评估脚本检查导出模型和 prompt。
