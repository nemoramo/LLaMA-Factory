# vLLM Speech Endpointing gRPC 服务（推荐，单卡）

该目录是一个**可独立拷贝**的部署示例：基于 `vllm/vllm-openai` 镜像，在 vLLM OpenAI 兼容接口（`/v1/chat/completions`）外封装一层 **gRPC 服务**，对外提供标准 RPC：

- `endpointing.v1.EndpointingService/Predict`

协议由 `endpointing.proto` 维护。

相对 SGLang，该实现默认更适合我们当前的 **speech endpointing** 业务：单轮、`max_tokens=1`、并对三个 special token 使用 `logit_bias` 强约束输出。

---

## 目录结构

- `Dockerfile`：镜像构建（base = `vllm/vllm-openai`，已清空 ENTRYPOINT）
- `vllm_endpointing_grpc_service.py`：gRPC 服务（可选同容器拉起 vLLM）
- `endpointing.proto`：gRPC 协议
- `endpointing_pb2.py`、`endpointing_pb2_grpc.py`：Python stubs（已生成）

---

## 构建镜像

在本目录执行：

```bash
docker build -t vllm-endpointing-grpc:latest .
```

指定 vLLM base 镜像 tag（可选）：

```bash
docker build --build-arg VLLM_IMAGE=vllm/vllm-openai:<TAG> -t vllm-endpointing-grpc:<TAG> .
```

---

## 运行（推荐：单容器模式，同容器拉起 vLLM）

默认单卡：容器只暴露一张 GPU（通过 `docker run --gpus '"device=0"'` 控制），因此不需要 TP/PP 等配置（README 不包含 TP）。

```bash
MODEL_DIR=/abs/path/to/model

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

说明：

- gRPC 监听端口：`50051`
- vLLM 监听端口：`30000`
- `LOGIT_BIAS_VALUE` 默认 100：会自动从 `/models/model` 的 tokenizer 计算 `<EOU>/<CONT_USER>/<UNADDRESSED>` 三个 token id 并构造 `logit_bias`

---

## 运行（分离模式：连接已有 vLLM）

如果 vLLM 已经在 `http://<HOST>:30000/v1` 提供服务：

```bash
docker run --rm -it \
  -p 50051:50051 \
  -e PORT=50051 \
  -e VLLM_BASE_URL="http://<HOST>:30000/v1" \
  -e VLLM_API_KEY="EMPTY" \
  -e LOGIT_BIAS_JSON='{"151665":100,"151666":100,"151667":100}' \
  vllm-endpointing-grpc:latest
```

> 分离模式下容器里可能没有模型文件，无法自动加载 tokenizer，所以需要显式提供 `LOGIT_BIAS_JSON`（token id -> bias）。

---

## 调用示例

### 1) grpcurl（推荐）

本机安装 `grpcurl` 后：

```bash
grpcurl -plaintext \
  -proto endpointing.proto \
  -d '{"request_id":"test-1","lang":"en-US","asr":{"text":"hello"},"options":{"treat_unaddressed_as_eou":true,"eou_threshold":0.6}}' \
  127.0.0.1:50051 endpointing.v1.EndpointingService/Predict
```

### 2) 没有 grpcurl：用 Python 调用（宿主机）

```bash
python3 - <<'PY'
import grpc
import endpointing_pb2, endpointing_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:50051")
stub = endpointing_pb2_grpc.EndpointingServiceStub(channel)

req = endpointing_pb2.EndpointingRequest(
    request_id="test-1",
    lang="en-US",
    asr=endpointing_pb2.Asr(text="hello"),
    options=endpointing_pb2.Options(treat_unaddressed_as_eou=True, eou_threshold=0.6),
)
resp = stub.Predict(req, timeout=30)
print(resp)
PY
```

---

## 返回值与判决规则（重要）

响应结构见 `endpointing.proto` 的 `EndpointingResponse`：

- `label`：`"<EOU>" | "<CONT_USER>" | "<UNADDRESSED>"`
- `confidence`：只在三个标签内归一化后的概率
- `latency_ms`：外层服务耗时（ms）
- `p_eou / p_cont_user / p_unaddressed`：三类标签内归一化后的概率（和为 1）

判决规则：

1) 外层服务向 vLLM 请求 `logprobs/top_logprobs`，仅在 `{<EOU>, <CONT_USER>, <UNADDRESSED>}` 三个标签内做归一化（保证三者概率和为 1）。
2) 若 `treat_unaddressed_as_eou=true`：
   - `P(<EOU>) += P(<UNADDRESSED>)`
   - `P(<UNADDRESSED>) = 0`
   - 再在 3 个标签内重新归一化。
3) 若 `P(<EOU>) < eou_threshold`，即使 `<EOU>` 概率最大，也会返回 `<CONT_USER>`（降低误判为结束的概率）。

输出约束（关键）：

- 每次请求都会携带 `logit_bias`，对 `<EOU>/<CONT_USER>/<UNADDRESSED>` 三个 token 施加 `+100` 的强偏置，尽可能保证输出只在三者内。

`meta` 字段会被接收但不会参与推理，也不会转发给模型。

---

## 指标（KPI）计算

`treat_unaddressed_as_eou=true` 场景下的 **延迟率/打断率** 计算公式见：`metrics.md`。

---

## 配置项（环境变量）

- `PORT`（默认 `50051`）
- `HOST`（默认 `0.0.0.0`）
- `VLLM_BASE_URL`（默认 `http://127.0.0.1:30000/v1`）
- `VLLM_API_KEY`（默认 `EMPTY`）
- `VLLM_MODEL`（默认：自动从 `/v1/models` 取第一个）
- `MODEL_ALIAS`（默认 `endpointing-judge-v1`）
- `EOU_THRESHOLD`（默认 `0.6`）
- `TOP_LOGPROBS`（默认 `20`）
- `TIMEOUT_S`（默认 `30`）
- `READY_TIMEOUT_S`（默认 `120`）
- `VLLM_CMD`（默认空；设置后外层服务会在容器内拉起 vLLM）
- `TOKENIZER_DIR`（默认空；若挂载了 `/models/model` 则会自动使用它）
- `LOGIT_BIAS_VALUE`（默认 `100`）
- `LOGIT_BIAS_JSON`（默认空；提供后优先使用）

---

## 常见问题

### 1) 为什么要 `logit_bias`？

endpointing 是一个 **三分类**任务，我们希望模型输出**严格为** `<EOU>/<CONT_USER>/<UNADDRESSED>` 之一。
`logit_bias=+100` 能大幅降低输出其它 token 的概率，并且便于从 `top_logprobs` 里稳定抽取三类概率。

### 2) 分离模式怎么拿到 token id？

在能访问模型 tokenizer 的环境执行：

```bash
python3 - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/abs/path/to/model", trust_remote_code=True)
for t in ["<EOU>", "<CONT_USER>", "<UNADDRESSED>"]:
    ids = tok.encode(t, add_special_tokens=False)
    print(t, ids)
PY
```

将输出填入 `LOGIT_BIAS_JSON`（key 必须是字符串）。

---

## 已验证的兼容性陷阱

在 `Qwen3-0.6B` speech endpointing 的实际部署测试里，已经遇到过下面这类版本错位问题：

- merge / export 环境使用 `transformers==5.2.0`
- `vllm-endpointing-grpc` 基于 `vllm 0.12.0`
- base image 自带 `transformers 4.57.x`

现象：

- `config.json` 可以正常读取
- 但 `AutoTokenizer.from_pretrained()` 可能在部署镜像里失败，报错类似：
  - `'list' object has no attribute 'keys'`

原因：

- `transformers 5.2.0` 导出的 `tokenizer_config.json` 里，`extra_special_tokens` 可能是 `list`
- 旧一点的 deployment-side `transformers` 在读取这个字段时不兼容

另外，**不要简单把当前 `vllm 0.12.0` 镜像里的 `transformers` 直接升级到 `5.2.0`**。我们已经验证过，这样虽然能修复 tokenizer 加载，但会导致：

- `from vllm import LLM`
- 报 `ImportError: cannot import name 'ALLOWED_LAYER_TYPES'`

也就是：

- 老 `transformers`：tokenizer 读不了
- 直接升到 `5.2.0`：vLLM 自己 import 不了

推荐处理方式：

1. 优先升级到一个**原生支持更高 transformers 版本**的 vLLM base image，再构建 `vllm-endpointing-grpc`
2. 如果短期不能换 vLLM base，就在 export 后增加一个 tokenizer config 兼容层，把 `tokenizer_config.json` 规范化到 deployment 镜像可接受的格式

### 当前仓库内已验证可工作的组合

在当前仓库里，我们已经用 `Qwen3-0.6B` 的 speech endpointing merged 模型做过一轮 smoke，下面这组配置可以把服务成功拉起：

- `VLLM_IMAGE=vllm/vllm-openai:v0.17.0`
- deployment 镜像内使用 `protobuf==5.29.6`
- deployment 镜像设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
- merged 模型目录里的 `tokenizer_config.json` 需要满足：
  - `additional_special_tokens` 是 `list`
  - 不要保留 `extra_special_tokens=list` 这种 `transformers 5.2.0` 导出后可能出现的形式

如果你看到下面两类报错，基本可以直接按上面的方向排查：

- `Descriptors cannot be created directly`
  - 这是旧 `pb2` 生成代码和较新 protobuf runtime 的兼容性问题
- `AttributeError: 'list' object has no attribute 'keys'`
  - 这是 deployment-side tokenizer 读取 `extra_special_tokens=list` 时的兼容性问题

## 2026-03-12 补充：`Qwen/Qwen3.5-0.8B-Base` 的 raw `llamafactory-cli export` 现状

- 已实测：当前主线里的 `llamafactory-cli export examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_export.yaml` 配合 `template=qwen3_5_nothink`，已经可以把 `Qwen3.5-0.8B-Base` speech endpointing LoRA checkpoint 导出成可被本目录 docker 加载的目录。
- 当前 export 后处理已经补上：
- 规范化 `tokenizer_config.json`，避免 `tokenizer_class = "TokenizersBackend"` 和缺失 `extra_special_tokens` / `additional_special_tokens` 的兼容性问题
- 自动保存 `preprocessor_config.json` / `video_preprocessor_config.json`，避免 `Qwen3_5ForConditionalGeneration` 在 vLLM 多模态初始化阶段缺 processor sidecar
- export 前会检查 `tie_word_embeddings`；如果 LoRA merge 后 `embed_tokens` / `lm_head` 已经分叉，但 config 仍然要求 tied，会先把 `lm_head` 拷回 input embeddings 再 re-tie 保存
- 2026-03-12 的最终 smoke 里，`Qwen3.5-0.8B-Base` 的 raw CLI export 产物已经在 `vllm-endpointing-grpc:v0.17.0` 上拉起：
- GPU：`5`
- `gpu_memory_utilization=0.42`
- `gRPC Predict` 返回：`label=<EOU>`, `confidence=0.9975`, `latency_ms=289`
- 同一轮 no-bias top-logprobs 检查里：
- `/v1/chat/completions`: `<EOU>` / `<UNADDRESSED>` / `<CONT_USER>` 位于 top1/top2/top3
- `/v1/completions`: `<EOU>` / `<CONT_USER>` / `<UNADDRESSED>` 位于 top1/top2/top3
- 这次也顺手验证了一个运行时边界：
- 对 `Qwen3.5-0.8B-Base` 这种多模态模型，`gpu_memory_utilization=0.15` 太低，vLLM 会在 KV cache 初始化阶段报 `No available memory for the cache blocks`
- 因此当前推荐把 `Qwen3.5` 的部署参数与 `Qwen3-0.6B` 分开看待，不要直接复用更小模型的 `gpu_memory_utilization`
- `gpu_memory_utilization` 还会受当时 GPU 5 上其它进程占用影响；如果同卡已有大进程，`0.52` 这类更激进的值可能因为启动时可用显存不足而失败
- 另外，当前 pinned 的 `vllm/vllm-openai:v0.17.0` 已不再接受旧启动参数 `--disable-log-requests`，deploy 脚本或手工启动命令需要一起更新。
- 这次复测也说明，之前 `checkpoint-1323` 的 no-bias top-k 异常首先应该归因到 export 产物里的 `tie_word_embeddings` 问题，而不是直接归因到 checkpoint 质量。
