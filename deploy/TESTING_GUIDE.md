# Speech Endpointing 模型测试指南

本文档介绍如何使用 vLLM 部署并测试训练完成的 Speech Endpointing 模型。

默认假设：
- 训练阶段用 `compute_endpointing_metrics: true`
- best checkpoint 由 `eval_label_acc`（3-way label accuracy）选择
- 部署后既关心 `treat_unaddressed_as_eou=false` 的 3-way 指标，也关心 `treat_unaddressed_as_eou=true` 的 2-way merge 指标
- 文中的 `${PROJECT_ROOT}`、`${MODEL_DIR}` 等均为占位符，请替换成你的本地路径

## 测试流程概览

```
训练完成的模型 → vLLM gRPC 服务部署 → WebUI 测试/评估
```

---

## 第一步：准备模型

确保你已完成训练并导出合并后的模型：

```bash
# 导出 LoRA 模型（如果还没做）
llamafactory-cli export examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_export.yaml \
  model_name_or_path=Qwen/Qwen3.5-0.8B-Base \
  template=qwen3_5_nothink \
  adapter_name_or_path=/path/to/your/checkpoint-XXXX \
  export_dir=/path/to/exported_model
```

对于 `Qwen/Qwen3.5-0.8B-Base`，这里必须显式覆盖 `template=qwen3_5_nothink`，否则会沿用通用 export YAML 里的 `qwen3_nothink`。

**检查模型文件：**
```bash
ls /path/to/exported_model/
# 应包含：config.json, model.safetensors, tokenizer.json 等
```

---

## 第二步：启动 vLLM gRPC 服务

### 2.1 进入部署目录

```bash
cd ${PROJECT_ROOT}/deploy/vllm_endpointing_grpc/
```

### 2.2 构建 Docker 镜像（首次）

```bash
docker build -t vllm-endpointing-grpc:latest .
```

### 2.3 运行服务

**单卡模式（推荐）：**

```bash
MODEL_DIR=/path/to/your/exported_model

docker run --rm -it --ipc=host --gpus '"device=0"' \
  -p 50051:50051 \
  -p 30000:30000 \
  -v ${MODEL_DIR}:/models/model:ro \
  -e PORT=50051 \
  -e VLLM_CMD="vllm serve /models/model --host 0.0.0.0 --port 30000 --gpu-memory-utilization 0.52" \
  -e VLLM_BASE_URL="http://127.0.0.1:30000/v1" \
  -e LOGIT_BIAS_VALUE=100 \
  vllm-endpointing-grpc:latest
```

**参数说明：**
- `--gpus '"device"'`: 指定 GPU，多卡可改为 `"device=0,1"`
- `-p 50051:50051`: gRPC 服务端口
- `-p 30000:30000`: vLLM HTTP 端口（可选）
- `LOGIT_BIAS_VALUE=100`: 强制模型只输出三个标签 token

### 2.4 验证服务启动

看到类似日志表示启动成功：
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:50051 (Press CTRL+C to quit)
```

---

## 第三步：测试模型

### 方式一：WebUI 可视化测试（推荐）

#### 3.1.1 进入 WebUI 目录

```bash
cd ${PROJECT_ROOT}/deploy/endpointing_webui/
```

#### 3.1.2 安装依赖

```bash
pip install -r requirements.txt
```

#### 3.1.3 启动 WebUI

```bash
python app.py
```

启动后访问：`http://127.0.0.1:7860`

#### 3.1.4 连接 gRPC 服务

1. 在 WebUI 中输入 **Host**: `127.0.0.1`，**Port**: `50051`
2. 点击 **🔗 Connect**
3. 看到 `✅ Connected successfully` 表示连接成功

#### 3.1.5 单条测试

1. 输入 **ASR Text**: 例如 "hello how are you"
2. 选择 **Language**: `en-US`
3. 调整 **EOU Threshold**: `0.6`（默认）
4. 点击 **🚀 Predict**
5. 查看返回的 label、confidence 和三个概率值

#### 3.1.6 批量测试

1. 切换到 **📊 Batch Processing** 标签
2. 上传 Excel/CSV 文件（包含 `asr_text` 列）
3. 配置列映射
4. 点击 **🚀 Run Inference**
5. 导出结果

**Excel 格式示例：**

| asr_text | lang |
|----------|------|
| hello how are you | en-US |
| 你好吗 | zh-CN |
| thank you | en-US |

---

### 方式二：命令行快速测试

#### 3.2.1 使用 grpcurl

```bash
# 安装 grpcurl（如果还没有）
# brew install grpcurl  # macOS
# apt install grpcurl   # Ubuntu

# 测试单条
cd ${PROJECT_ROOT}/deploy/vllm_endpointing_grpc/

grpcurl -plaintext \
  -proto endpointing.proto \
  -d '{
    "request_id":"test-1",
    "lang":"en-US",
    "asr":{"text":"hello how are you"},
    "options":{
      "treat_unaddressed_as_eou":true,
      "eou_threshold":0.6
    }
  }' \
  127.0.0.1:50051 endpointing.v1.EndpointingService/Predict
```

**预期返回：**
```json
{
  "request_id": "test-1",
  "label": "<EOU>",
  "confidence": 0.9917,
  "model": "endpointing-judge-v1",
  "latency_ms": 15,
  "p_eou": 0.9917,
  "p_cont_user": 0.0083,
  "p_unaddressed": 0.0000
}
```

#### 3.2.2 使用 Python 脚本

```bash
cd ${PROJECT_ROOT}/deploy/vllm_endpointing_grpc/

python3 - <<'PY'
import grpc
import endpointing_pb2
import endpointing_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:50051")
stub = endpointing_pb2_grpc.EndpointingServiceStub(channel)

req = endpointing_pb2.EndpointingRequest(
    request_id="test-1",
    lang="en-US",
    asr=endpointing_pb2.Asr(text="hello how are you"),
    options=endpointing_pb2.Options(
        treat_unaddressed_as_eou=True, 
        eou_threshold=0.6
    ),
)
resp = stub.Predict(req, timeout=30)
print(f"Label: {resp.label}")
print(f"Confidence: {resp.confidence:.4f}")
print(f"P(EOU): {resp.p_eou:.4f}")
print(f"P(CONT): {resp.p_cont_user:.4f}")
print(f"P(UNADDRESSED): {resp.p_unaddressed:.4f}")
PY
```

---

## 第四步：批量评估（离线 / 服务）

### 4.1 直接评估本地 HF / merged 模型

```bash
cd ${PROJECT_ROOT}/examples/speech_endpointing/

python eval_hf_endpointing.py \
  --base-model /path/to/your/exported_model \
  --dataset /path/to/your/test.jsonl \
  --out-dir /path/to/eval_hf_out
```

输出：
- `/path/to/eval_hf_out/pred.jsonl`
- `/path/to/eval_hf_out/summary.json`

说明：
- `summary.json` 会同时输出：
  - `tag_eval`：3-way 指标（等价于 `treat_unaddressed_as_eou=false`）
  - `tag_eval_merge_unad_as_eou`：2-way merge 指标（等价于 `treat_unaddressed_as_eou=true`）
  - `export_prompt_probe`：导出模型健康检查
- `export_prompt_probe` 会对一个固定 endpointing prompt 做 next-token probe：
  - 正常情况下 `<EOU>` / `<CONT_USER>` / `<UNADDRESSED>` 应该占据 full-vocab top3
  - 如果没有进入 top3，先优先排查 export 链路，而不是先怀疑模型效果：
    - 先检查导出后的 `config.json` 里 `tie_word_embeddings` 是否和实际权重一致，尤其是 Qwen3 / Qwen3.5
    - 检查 `embed_tokens` 和 `lm_head` 在 merge 后是否仍然保持了正确的 tied-embedding 关系
    - 再检查 `llamafactory-cli export` 是否用了正确 `template`
    - 再检查导出时是否保留了 `add_special_tokens` 和 `resize_vocab`
    - 确认当前评估目录是否真的是刚导出的 merged model
    - 确认评估时 prompt 格式是否和训练 / 部署保持一致
- 适合快速检查导出模型是否还能保持“单标签 token 分类”行为

### 4.2 评估已部署的 OpenAI-compatible 服务

如果你想同时拿到：
- `treat_unaddressed_as_eou=false` 的 3-way tag KPI
- `treat_unaddressed_as_eou=true` 的 merge KPI

使用 `eval_sglang_endpointing.py` 对 vLLM HTTP 端口评估：

```bash
cd ${PROJECT_ROOT}/examples/speech_endpointing/

python eval_sglang_endpointing.py \
  --input /path/to/your/test.jsonl \
  --base-url http://127.0.0.1:30000 \
  --model endpointing-judge-v1 \
  --out-dir /path/to/sglang_eval_out
```

输出：
- `pred_<model>_<run_id>.jsonl`
- `summary_<model>_<run_id>.json`

其中 summary 里最重要的两个字段是：
- `tag_eval`
  - 等价于 `treat_unaddressed_as_eou=false`
  - 包含 `accuracy`、`per_label`、`kpi.FAR_unad/Interrupt/Delay/Missed`
- `tag_eval_merge_unad_as_eou`
  - 等价于 `treat_unaddressed_as_eou=true`
  - 包含 `accuracy`、`per_label`、`kpi.Interrupt/Delay`

如果容器没有暴露 `30000`，记得在 `docker run` 时保留：

```bash
-p 30000:30000
```

---

## 关键概念说明

### 三个标签含义

| 标签 | 含义 | 系统动作 |
|------|------|----------|
| `<EOU>` | End of Utterance（用户说完了）| 系统应该回复 |
| `<CONT_USER>` | Continue User（用户未说完）| 继续等待输入 |
| `<UNADDRESSED>` | Unaddressed（非对系统说）| 忽略该语音 |

### 重要参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `eou_threshold` | EOU 决策阈值 | 0.5-0.7 |
| `treat_unaddressed_as_eou` | 将 UNADDRESSED 视为 EOU | true（生产环境） |
| `logit_bias` | 强制输出三个标签之一 | 100 |
| `eval_label_acc` | 训练内 best checkpoint 指标 | 推荐 |

### 概率计算逻辑

1. vLLM 返回 `top_logprobs`（top 20 token 的对数概率）
2. 提取三个标签 token 的概率，在三个标签内归一化
3. 如果 `treat_unaddressed_as_eou=true`：
   - `P(EOU) += P(UNADDRESSED)`
   - `P(UNADDRESSED) = 0`
   - 重新归一化
4. 如果 `P(EOU) < eou_threshold`，返回 `<CONT_USER>`

### 训练内指标 vs 部署指标

- 训练内 `eval_label_acc`：3-way 标签精度，用于选 best checkpoint
- 训练内 `eval_merged_label_acc`：把 `<UNADDRESSED>` 合并到 `<EOU>` 后的 2-way 精度
- 部署时 `treat_unaddressed_as_eou=true`：更接近线上最终决策
- 因此建议同时看 3-way 与 merge 两套结果，不要只看单一 accuracy

---

## 常见问题

### Q: 服务启动失败，提示 "CUDA out of memory"
A: 调整 `--gpu-memory-utilization` 参数，例如改为 `0.45` 或 `0.35`

### Q: WebUI 连接失败
A: 检查：
1. gRPC 服务是否已启动（`docker ps`）
2. 端口是否正确（默认 50051）
3. 防火墙是否放行

### Q: 模型输出不是三个标签之一
A: 确保：
1. 模型已正确导出并包含特殊 token
2. `LOGIT_BIAS_VALUE` 设置为 100
3. tokenizer 包含 `<EOU>`, `<CONT_USER>`, `<UNADDRESSED>`

### Q: 如何停止服务
A: 在运行 docker 的终端按 `Ctrl+C`，或执行：
```bash
docker stop $(docker ps -q --filter ancestor=vllm-endpointing-grpc:latest)
```

---

## 目录索引

| 目录 | 用途 |
|------|------|
| `deploy/vllm_endpointing_grpc/` | vLLM gRPC 服务部署 |
| `deploy/endpointing_webui/` | WebUI 测试工具 |
| `examples/speech_endpointing/` | 训练配置和评估脚本 |

---

## 相关文件

- `deploy/vllm_endpointing_grpc/README.md` - vLLM 服务详细文档
- `deploy/endpointing_webui/README.md` - WebUI 详细文档
- `examples/speech_endpointing/eval_hf_endpointing.py` - 批量评估脚本
- `examples/speech_endpointing/eval_sglang_endpointing.py` - 服务评估脚本（同时输出 merge / unmerged KPI）
