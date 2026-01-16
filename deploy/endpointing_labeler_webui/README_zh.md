# Speech Endpointing 标注/评测 GUI（text-generation-webui 扩展）

该目录用于给标注测试人员提供一个**可视化评测工具**，通过 gRPC 调用 Endpointing 服务，展示 **三类归一化概率**，支持 **批量评测、人工标注、导出 Excel/JSONL**。

## 后端服务说明

本工具通过 gRPC 连接到 Endpointing 服务。推荐使用 **vLLM** 作为后端服务：

- **vLLM 服务**（推荐）：位于 `deploy/vllm_endpointing_grpc/`
  - 基于 vLLM OpenAI 兼容接口封装
  - 单卡部署，性能更优
  - 支持 `logit_bias` 强约束输出
  - 默认监听端口：`50051`

详细部署说明请参考 [deploy/vllm_endpointing_grpc/README.md](../vllm_endpointing_grpc/README.md)

## 目录结构

- `extension/endpointing_tool/`：WebUI 扩展（script + gRPC stubs）
- `config/config.json`：默认配置样例
- `samples/sample.jsonl`：JSONL 输入样例
- `requirements_extra.txt`：扩展所需额外依赖
- `windows/Start_Endpointing_GUI.bat`：Windows 一键启动模板

## 运行方式（开发/内网）

1. 准备 text-generation-webui（建议以 submodule 方式固定版本）
2. 拷贝扩展到 webui：

```bash
cp -r deploy/endpointing_labeler_webui/extension/endpointing_tool \
  <text-generation-webui>/extensions/endpointing_tool
```

3. 放置默认配置（可选）：

```bash
mkdir -p <text-generation-webui>/user_data/endpointing_tool
cp deploy/endpointing_labeler_webui/config/config.json \
  <text-generation-webui>/user_data/endpointing_tool/config.json
```

4. 启动 WebUI（只加载本扩展）：

```bash
cd <text-generation-webui>
python server.py --extensions endpointing_tool
```

打开页面后在 “Speech Endpointing” Tab 里使用。
## 完整使用流程

### 1. 启动 vLLM Endpointing 服务

```bash
cd deploy/vllm_endpointing_grpc

# 构建镜像
docker build -t vllm-endpointing-grpc:latest .

# 启动服务（假设模型在 /path/to/model）
MODEL_DIR=/path/to/model
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

### 2. 启动标注 WebUI

按照上面的"运行方式"部分启动 text-generation-webui。

### 3. 配置连接

在 WebUI 的 "Speech Endpointing" Tab 中配置 gRPC 地址：`127.0.0.1:50051`
## JSONL 输入格式

每行一个样本：

```json
{"request_id":"id","session_id":"s","lang":"en-US","asr":{"text":"hello"},"history":[{"role":"user","text":"..."}],"options":{"treat_unaddressed_as_eou":true,"eou_threshold":0.6}}
```

- `history.role` 仅支持 `user/assistant`
- `options.eou_threshold` 可选
- 允许带上 `human_label` 与 `note`（导入后可继续编辑）

## Excel 导出字段

- `history_text`：给标注人员查看的可读对话
- `history_json`：机器可回放 JSON
- 预测字段：`pred_label/p_eou/p_cont_user/p_unaddressed/confidence/latency_ms/model/error`
- 标注字段：`human_label/note`

## Windows 离线一键包（建议做法）

> 目标：标注机器无需联网，解压即用

### A. 在有网机器上构建离线包

1. 准备 webui 源码：

```bash
git clone https://github.com/oobabooga/text-generation-webui
```

2. 复制扩展：

```bash
cp -r deploy/endpointing_labeler_webui/extension/endpointing_tool \
  text-generation-webui/extensions/endpointing_tool
```

3. 生成 `portable_env`（推荐 Python venv 方式）：

```bash
cd text-generation-webui
python -m venv portable_env
portable_env\Scripts\pip install -r requirements/portable/requirements.txt
portable_env\Scripts\pip install -r ..\deploy\endpointing_labeler_webui\requirements_extra.txt
```

> 如果需要离线轮子，可先在联网机上 `pip download -d wheelhouse -r requirements/portable/requirements.txt -r requirements_extra.txt`，再在离线机用 `pip install --no-index --find-links=wheelhouse ...`。

4. 放置默认配置（可选）：

```bash
mkdir user_data\endpointing_tool
copy ..\deploy\endpointing_labeler_webui\config\config.json user_data\endpointing_tool\config.json
```

5. 添加一键启动脚本：

```bash
copy ..\deploy\endpointing_labeler_webui\windows\Start_Endpointing_GUI.bat .
```

6. 将整个 `text-generation-webui` 目录打包成 zip，发给标注人员。

### B. 在离线 Windows10/11 上使用

- 解压 zip
- 双击 `Start_Endpointing_GUI.bat`
- 浏览器打开后，在 Tab 中直接使用

## gRPC 依赖

扩展依赖：

- `grpcio`
- `protobuf==3.20.3`
- `pandas` / `openpyxl`（导出 Excel 用）

这些依赖已汇总在 `requirements_extra.txt`。
