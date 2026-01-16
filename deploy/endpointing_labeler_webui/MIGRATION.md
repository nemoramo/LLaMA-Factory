# 从 SGLang 迁移到 vLLM

## 变更说明

本项目已从 SGLang 后端迁移到 vLLM 后端，以获得更好的性能和单卡部署支持。

## 主要变化

1. **后端服务**
   - 旧：`deploy/sglang_endpointing_grpc/` (已删除)
   - 新：`deploy/vllm_endpointing_grpc/`

2. **服务特性**
   - vLLM 基于 OpenAI 兼容接口封装
   - 更适合单卡部署场景
   - 支持 `logit_bias` 强约束输出
   - 更好的性能和资源利用率

3. **WebUI 标注工具**
   - 无需修改，通过 gRPC 协议与后端通信
   - 默认端口保持不变：`50051`
   - 配置文件兼容，无需更新

## 迁移步骤

### 如果你之前使用 SGLang 服务

1. 停止旧的 SGLang 服务
2. 按照 [vllm_endpointing_grpc/README.md](../vllm_endpointing_grpc/README.md) 部署新的 vLLM 服务
3. 标注 WebUI 无需修改，可直接连接新服务

### 部署 vLLM 服务

```bash
cd deploy/vllm_endpointing_grpc

# 构建镜像
docker build -t vllm-endpointing-grpc:latest .

# 启动服务
MODEL_DIR=/path/to/your/model
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

### 验证服务

标注 WebUI 连接后，如果能正常预测并看到三类概率值（EOU/CONT_USER/UNADDRESSED），说明迁移成功。

## 优势

- **性能提升**：vLLM 针对推理场景优化，吞吐量更高
- **单卡友好**：不需要 tensor parallelism，单卡即可高效运行
- **资源控制**：更细粒度的 GPU 内存控制（`--gpu-memory-utilization`）
- **社区活跃**：vLLM 生态更完善，更新更频繁
- **标准接口**：兼容 OpenAI API，易于集成其他工具

## 注意事项

- 确保模型的 tokenizer 中包含 `<EOU>`, `<CONT_USER>`, `<UNADDRESSED>` 三个 special tokens
- 如果使用分离模式部署，需要手动指定 token IDs 的 `LOGIT_BIAS_JSON`
- vLLM 版本建议使用最新的 stable release
