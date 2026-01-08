# SGLang 端点检测 gRPC 服务（可独立拷贝目录）

该目录是一个**可独立拷贝**的部署示例：基于 `lmsysorg/sglang` 镜像，在 SGLang OpenAI 兼容接口（`/v1/chat/completions`）外封装一层 **gRPC 服务**，对外提供标准 RPC：

- `endpointing.v1.EndpointingService/Predict`

协议由 `endpointing.proto` 维护。

---

## 目录结构

- `Dockerfile`：镜像构建（base = `lmsysorg/sglang`）
- `sglang_endpointing_grpc_service.py`：gRPC 服务（可选同容器拉起 SGLang）
- `endpointing.proto`：gRPC 协议
- `endpointing_pb2.py`、`endpointing_pb2_grpc.py`：Python stubs（已生成）

---

## 构建镜像

在本目录执行：

```bash
docker build -t sglang-endpointing-grpc:latest .
```

指定 SGLang base 镜像 tag（可选）：

```bash
docker build --build-arg SGLANG_IMAGE=lmsysorg/sglang:<TAG> -t sglang-endpointing-grpc:<TAG> .
```

---

## 运行（推荐：单容器模式，同容器拉起 SGLang）

默认单卡：容器只暴露一张 GPU（通过 `docker run --gpus '"device=0"'` 控制），因此不需要额外配置并行相关参数。

```bash
MODEL_DIR=/abs/path/to/model

docker run --rm -it --ipc=host --gpus '"device=0"' \
  -p 50051:50051 \
  -p 30000:30000 \
  -v ${MODEL_DIR}:/models/model:ro \
  -e PORT=50051 \
  -e MODEL_ALIAS=endpointing-judge-v1 \
  -e EOU_THRESHOLD=0.6 \
  -e TOP_LOGPROBS=20 \
  -e SGLANG_CMD="python3 -m sglang.launch_server --model-path /models/model --host 0.0.0.0 --port 30000" \
  sglang-endpointing-grpc:latest
```

说明：

- gRPC 监听端口：`50051`
- SGLang 监听端口：`30000`
- `SGLANG_CMD` 为空则不会在容器内拉起 SGLang（见下一节分离模式）

---

## 运行（分离模式：连接已有 SGLang）

如果 SGLang 已经在 `http://<HOST>:30000/v1` 提供服务：

```bash
docker run --rm -it \
  -p 50051:50051 \
  -e PORT=50051 \
  -e SGLANG_BASE_URL="http://<HOST>:30000/v1" \
  -e SGLANG_API_KEY="EMPTY" \
  sglang-endpointing-grpc:latest
```

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

### 2) 没有 grpcurl：用容器内 Python 调用

```bash
docker exec -i <container_name> python3 - <<'PY'
import grpc
import endpointing_pb2, endpointing_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:50051")
stub = endpointing_pb2_grpc.EndpointingServiceStub(channel)

req = endpointing_pb2.EndpointingRequest(
    request_id="test-1",
    lang="en-US",
    asr=endpointing_pb2.Asr(text="hello"),
    history=[
        endpointing_pb2.HistoryTurn(role=endpointing_pb2.USER, text="turn on do not disturb mode"),
        endpointing_pb2.HistoryTurn(role=endpointing_pb2.ASSISTANT, text="okay do not disturb mode is now on"),
    ],
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

判决规则：

1) 外层服务向 SGLang 请求 `logprobs/top_logprobs`，仅在 `{<EOU>, <CONT_USER>, <UNADDRESSED>}` 三个标签内做归一化（保证三者概率和为 1）。
2) 若 `treat_unaddressed_as_eou=true`：
   - `P(<EOU>) += P(<UNADDRESSED>)`
   - `P(<UNADDRESSED>) = 0`
   - 再在 3 个标签内重新归一化。
3) 若 `P(<EOU>) < eou_threshold`，即使 `<EOU>` 概率最大，也会返回 `<CONT_USER>`（降低误判为结束的概率）。

`meta` 字段会被接收但不会参与推理，也不会转发给模型。

---

## 配置项（环境变量）

- `PORT`（默认 `50051`）
- `HOST`（默认 `0.0.0.0`）
- `SGLANG_BASE_URL`（默认 `http://127.0.0.1:30000/v1`）
- `SGLANG_API_KEY`（默认 `EMPTY`）
- `SGLANG_MODEL`（默认：自动从 `/v1/models` 取第一个）
- `MODEL_ALIAS`（默认 `endpointing-judge-v1`）
- `EOU_THRESHOLD`（默认 `0.6`）
- `TOP_LOGPROBS`（默认 `20`）
- `TIMEOUT_S`（默认 `10`）
- `READY_TIMEOUT_S`（默认 `120`）
- `SGLANG_CMD`（默认空；设置后外层服务会在容器内拉起 SGLang）
