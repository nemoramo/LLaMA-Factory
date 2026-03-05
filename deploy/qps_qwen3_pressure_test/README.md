# QPS Qwen3 Pressure Test (vLLM)

⚠️ **重要提示：本压测工具必须在 AWS SageMaker Studio 环境中运行**，依赖 SageMaker 特定的 Docker 网络配置和本地路径结构。

这个目录提供一套可直接执行的压测全流程，适配 SageMaker Studio 环境。

## 1. 目录结构

```text
qps_qwen3_pressure_test/
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

## 2. 环境要求

### 2.1 运行环境
- **必须在 AWS SageMaker Studio 实例中运行**（本工具依赖 SageMaker 特定的 Docker 网络和存储配置）
- 需要 GPU 实例（用于运行 vLLM 服务）
- 需要 Docker 权限

### 2.2 测试数据要求
- **必须使用与训练数据相同格式的测试数据**（见 3.2 节详细格式说明）
- 确保测试数据已上传到 SageMaker 环境，或可从 S3 下载到本地路径

### 2.3 SageMaker Studio 特殊配置

- Docker 必须使用 `--network sagemaker`。
- 不能使用 `-p 8000:8000` 端口映射。
- 不建议使用 `--ipc=host`（本环境通常会报权限错误）。
- `-v` 绑定路径必须在 `/home/sagemaker-user` 下，并且路径中不要包含 symlink。

## 3. 一键流程（推荐顺序）

在终端执行：

```bash
cd /home/sagemaker-user/qps_qwen3_pressure_test
```

### 3.1 构建 bench 镜像（补齐 `pandas`）

```bash
bash scripts/build_bench_image.sh
```

### 3.2 准备 bench 数据集（离线套 chat template）

⚠️ **输入数据格式要求**：必须使用与训练数据相同格式的测试数据，每行一个 JSON 对象，包含以下字段：
- `context`: 对话上下文文本（字符串），或
- `messages`: 消息列表（会自动提取最后一条 user 消息作为 context）
- `lang`: 语言代码（如 "en", "zh" 等，可选，默认 "en"）
- `label`: 标签（可选）
- `dialogue_id`, `turn`: 对话标识（可选）

**示例输入格式**（与训练数据格式一致）：
```json
{"context": "用户说的话...", "lang": "zh", "label": "<EOU>", "dialogue_id": "123", "turn": 1}
```

或 messages 格式：
```json
{"messages": [{"role": "user", "content": "用户说的话..."}], "lang": "zh", "label": "<EOU>"}
```

```bash
python3 scripts/prepare_endpointing_bench.py \
  --input /home/sagemaker-user/test.jsonl \
  --output /home/sagemaker-user/qps_qwen3_pressure_test/data/bench_endpointing_512.jsonl \
  --tokenizer /home/sagemaker-user/1.0.2 \
  --trust-remote-code \
  --max-prompt-len 512
```

### 3.3 启动 vLLM 服务

```bash
bash scripts/run_server.sh start \
  --model-path /home/sagemaker-user/1.0.2 \
  --served-model-name endpointing-qwen3-0.6b-ft
```

查看状态和健康检查：

```bash
bash scripts/run_server.sh status
bash scripts/run_server.sh health
```

### 3.4 执行压测

warm steady-state：

```bash
bash scripts/run_bench.sh warm \
  --dataset /home/sagemaker-user/qps_qwen3_pressure_test/data/bench_endpointing_512.jsonl \
  --served-model-name endpointing-qwen3-0.6b-ft
```

burst（`burstiness=0.3`）：

```bash
bash scripts/run_bench.sh burst \
  --dataset /home/sagemaker-user/qps_qwen3_pressure_test/data/bench_endpointing_512.jsonl \
  --served-model-name endpointing-qwen3-0.6b-ft
```

cold start（先重启服务，再跑 cold）：

```bash
bash scripts/run_server.sh restart \
  --model-path /home/sagemaker-user/1.0.2 \
  --served-model-name endpointing-qwen3-0.6b-ft

bash scripts/run_bench.sh cold \
  --dataset /home/sagemaker-user/qps_qwen3_pressure_test/data/bench_endpointing_512.jsonl \
  --served-model-name endpointing-qwen3-0.6b-ft
```

sweep（用于找 `p90 <= 60ms` 的最大 QPS）：

```bash
bash scripts/run_bench.sh sweep \
  --dataset /home/sagemaker-user/qps_qwen3_pressure_test/data/bench_endpointing_512.jsonl \
  --served-model-name endpointing-qwen3-0.6b-ft \
  --rates 200,250,300,350,400,500
```

### 3.5 汇总结果

```bash
python3 scripts/summarize_results.py \
  --results-dir /home/sagemaker-user/qps_qwen3_pressure_test/results
```

## 4. 输出文件

- 压测 json 输出在：`/home/sagemaker-user/qps_qwen3_pressure_test/results`
- 每个场景一个文件：
  - `warm_c32_r800.json`
  - `burst_c32_r800_b03.json`
  - `cold_c16_r200.json`
  - `sweep_c32_rXXX.json`

## 5. 停止服务

```bash
bash scripts/run_server.sh stop
```

---

## 6. 迁移到其他环境（非 SageMaker）

如需在普通 GPU 服务器上运行，需要修改以下配置：

| 文件 | 修改项 |
|------|--------|
| `scripts/build_bench_image.sh` | 移除 `--network sagemaker` |
| `scripts/run_server.sh` | 1. 移除 `--network sagemaker`<br>2. 添加 `-p 8000:8000` 端口映射 |
| `scripts/run_bench.sh` | 修改 `--host 127.0.0.1` 为实际服务 IP |
| `scripts/summarize_results.py` | 修改默认 `--results-dir` 路径 |
| `README.md` 中的示例命令 | 将所有 `/home/sagemaker-user/` 替换为实际路径 |
