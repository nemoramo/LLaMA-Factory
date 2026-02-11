# FunAudioChat (S2T) 在 LLaMA-Factory 中的集成

本项目内置了对 **FunAudioChat** 检查点（`model_type=funaudiochat`）的集成，支持 **仅 S2T（语音到文本）** 的微调。

- 模板：`template: funaudiochat`
- MM 插件：将音频占位符扩展为可变长度的 `<|AUDIO|>` token，并提供：
  - `speech_ids` / `speech_attention_mask`（离散 25Hz 帧，无需 CosyVoice）
  - `input_features` / `feature_attention_mask`（连续波形特征）
  - `feature_exist_mask`（哪些音频有波形特征）
- 训练时会自动禁用语音生成/损失（`disable_speech=True`）。

## 数据集格式（ShareGPT）

使用 ShareGPT 风格的数据集，包含一个 `audios` 列（通过 `dataset_info.json` 映射到你的 JSONL 字段）。

- `messages[*].content` 应包含一个音频占位符。推荐格式（FunAudio 风格）：
  - `<|audio_bos|><|AUDIO|><|audio_eos|>`
- `audios` 字段是一个列表，其长度与消息中的占位符数量匹配。

## 数据转换（标准化 vs 原始）

如果你的 ASR 数据集同时包含标准化文本（小写/无标点）和原始文本（大小写/带标点），
请参阅 `examples/funaudiochat/DATA_CONVERSION_zh.md` 了解实用的转换工作流程，以及如何通过
`prompt_pool` + `dynamic_prompt_sampling: true` 保留两个目标。

每个音频项可以是：

1) 普通路径字符串：`"/abs/path/to.wav"`
2) JSON 字符串（FunAudio 兼容）：

```json
{"path": "/abs/path/to.wav", "token": "<|audio_pad|><|audio_pad|>..."}
```

如果省略/留空 `token`，插件需要得到 25Hz 帧数来构建仅由 pad 组成的 token 序列：

- 推荐（更快，适用于 mp3/m4a/...）：在 JSON 里补充 `duration`（单位：秒）。
- 兜底：从 `_segXXXX_<start>-<end>.wav` 文件名推断，其次读取音频元信息（ffprobe），最后才会解码波形。

## 示例配置

参见 `examples/funaudiochat/funaudiochat_s2t_sft_full.yaml`。

## 批量评测（prompt_pool + normalized WER/WERE）

如果你的评测数据使用了 `prompt_pool`（例如 `*_norm_text_promptpool_*`），并且希望评测时带上
**language 提示**、使用 **normalized prompt**（与训练对齐），可以直接用脚本：

```bash
conda activate llamafactory
python scripts/eval_funaudiochat_s2t_promptpool.py \
  --model /path/to/checkpoint-XXXXX \
  --base-model FunAudioLLM/Fun-Audio-Chat-8B \
  --gpus 6,7
```

输出默认写到 `--out-root`（默认：`/data2/mayufeng/llamafactory_eval/funaudiochat`），包含：

- `generated_predictions.jsonl`（prompt/predict/label）
- `normalized_wer_were_eval.json`（调用 `~/projects/speech_related_tools/evaluate/eval_asr_wer_cer.py`）
- `summary.json`（各测试集路径与指标汇总）

## 打包训练（参考）

长期实验建议通过 watchdog 脚本启动：`scripts/monitor_funaudiochat_s2t_training.sh`。

- 训练进程退出（OOM / 断连 / 崩溃）会自动重启
- 使用 `overwrite_output_dir=false` 时，LLaMA-Factory 会从 `output_dir` 的最新 checkpoint 自动恢复
- 脚本会在 `output_dir` 下保存复现信息：
  - `training_command.txt`（完整命令行）
  - `config_base.yaml`（启动时使用的基础 YAML 配置副本）

### 超大数据集（Parquet shard）

如果训练 JSONL 极大（例如 10 万+小时），建议启用 **sharded parquet backend**
来避免构建全量 HF map-style JSONL 索引，并降低高 worker/prefetch 下的 OOM 风险。

- 指南：`examples/funaudiochat/LARGE_SCALE_TRAINING.zh-CN.md`
- 后端细节（跨 shard 预读 + shard-boundary resume）：`SHARDED_PARQUET_BACKEND.md`

### neat packing 参考命令

```bash
export OUTPUT_DIR="/path/to/llamafactory_saves/funaudiochat/s2t_lora_neatpack_run"
# 可选：从已有 LoRA adapter checkpoint 初始化。
export INIT_ADAPTER_NAME_OR_PATH="/path/to/prev_adapter/checkpoint-XXXXX"

GPUS=0,1,2,3,4,5 NPROC_PER_NODE=6 \
PACKING=true NEAT_PACKING=true \
DYNAMIC_PROMPT_LAZY_ALIGN=true DYNAMIC_PROMPT_PACKING_BUFFER_SIZE=2048 \
DYNAMIC_PROMPT_PACKING_PREFETCH_BUFFERS=4 DYNAMIC_PROMPT_PACKING_CARRYOVER_PACKS=2 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_STEPS=60000 EVAL_STEPS=2000 SAVE_STEPS=2000 \
DATALOADER_NUM_WORKERS=6 PREPROCESSING_NUM_WORKERS=32 DATALOADER_PREFETCH_FACTOR=4 \
EVAL_MAX_NEW_TOKENS=512 \
bash scripts/monitor_funaudiochat_s2t_training.sh
```

注意：
- 开启 packing 后，“epoch” 的语义可能不等价于完整数据集遍历，建议用 `MAX_STEPS` 控制训练时长。
- 如果 `output_dir` 里已经有 checkpoint，脚本会忽略 `INIT_ADAPTER_NAME_OR_PATH`，直接从最新 checkpoint 恢复。
- 动态 prompt 打包相关参数（可选）：
  - `DYNAMIC_PROMPT_PACKING_PREFETCH_BUFFERS`（`dynamic_prompt_packing_prefetch_buffers`）：每个 dataloader worker 预取 N 个 *packed buffer*，减少 buffer 切换卡顿（更吃 CPU/RAM）。
  - `DYNAMIC_PROMPT_PACKING_CARRYOVER_PACKS`（`dynamic_prompt_packing_carryover_packs`）：把每个 buffer 中最“空”的 N 个 pack 对应的 raw segments 留到下一个 buffer 再一起 pack，实现跨 buffer 混合（更高打包效率，但样本顺序会有轻微变化）。设为 `0` 可关闭。

## Attention 实现（推荐：`fa2`）

LLaMA-Factory 通过 `flash_attn` 参数支持 3 种 FunAudioChat 的 attention 实现：

- `fa2`：Transformers `flash_attention_2`（需要 `flash-attn` 包）
- `sdpa`：torch SDPA（`torch.nn.functional.scaled_dot_product_attention`）
- `disabled`：eager attention

**推荐：** 当可用时使用 `flash_attn: fa2`（最快）。如果无法安装，使用 `sdpa`。

### 基准测试（单 GPU）

这些数据来自在 **NVIDIA H20 (sm90)**、**单 GPU**、**LoRA + bf16**、
**仅 S2T**、`per_device_train_batch_size=23`、`max_steps=100`、`audio_padding=max_length`
条件下进行的受控本地基准测试，使用固定长度约 8.9 秒的音频样本重复（因此填充和内存是稳定的）。

| `flash_attn` | 训练速度（步/秒） | 峰值显存 |
| --- | ---: | ---: |
| `fa2` | **0.358** | 45,250 MiB |
| `sdpa` | 0.189 | **36,965 MiB** |
| `disabled` | 0.112 | 68,109 MiB |

注意事项：
- 速度和内存会根据音频持续时间分布、截止长度、梯度检查点设置等而变化。
- `fa2` 在我们的测试中最快；`sdpa` 可能更节省内存，具体取决于你的工作负载。

### 基准测试（neat packing vs 不打包）

在 **2× NVIDIA H20** 上进行的 **25 分钟**计时运行中，我们比较了 FunAudioChat S2T 的
**neat packing** 与 **不打包**，使用相同的训练数据集，并跟踪 **有效 token** 数
（非忽略标签，不包括音频 token）：

- `neat_packing=true` + 动态 prompt 打包（`per_device_train_batch_size=2`）：约 **1.00M** 有效 token/GPU/25分钟（≈**669 token/秒/GPU**）
- `packing=false`（`per_device_train_batch_size=8`）：约 **0.71M** 有效 token/GPU/25分钟（≈**475 token/秒/GPU**）

总体而言，在相同的实际时间窗口内，neat packing 提供的有效 token 约多 **1.41×**（日志中的稳态吞吐量约高 **1.77×**）。

### 安装 FlashAttention-2 (`fa2`)

#### 1) 快速安装（建议先尝试）

在你的训练 Python 环境中：

```bash
python -m pip install -U --no-build-isolation flash-attn
```

如果你的 GPU 是 H20/H100 (sm90)，可以通过限制构建架构来加速编译：

```bash
export FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS=8
export NVCC_THREADS=4
python -m pip install -U --no-build-isolation flash-attn
```

验证：

```bash
python -c "from transformers.utils import is_flash_attn_2_available; print(is_flash_attn_2_available())"
```

你还应该在训练日志中看到：
`Using FlashAttention-2 for faster training and inference.`

#### 2) 如果 pip 构建失败：先构建 wheel 再安装

这是一个稳健的回退方案，可以避免一些 PEP517 边缘情况：

```bash
set -euo pipefail
mkdir -p /tmp/flash_attn_build && cd /tmp/flash_attn_build
python -m pip download --no-deps --no-binary :all: flash-attn==2.8.3
tar -xzf flash_attn-2.8.3.tar.gz
cd flash_attn-2.8.3

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS=8
export NVCC_THREADS=4

python setup.py bdist_wheel -v
python -m pip install --no-deps dist/flash_attn-*.whl
```
