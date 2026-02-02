# 在 LLaMA-Factory 中微调 Qwen3-ASR

本仓库提供 **Qwen3-ASR**（`model_type=qwen3_asr`）的训练集成示例，例如：

- `Qwen/Qwen3-ASR-0.6B`
- `Qwen/Qwen3-ASR-1.7B`

## 依赖说明

Qwen3-ASR 可能尚未合入上游 `transformers`。LLaMA-Factory 支持两种加载方式：

1) 安装官方包：

```bash
python -m pip install -U qwen-asr
```

2) 或使用本仓库 vendored 的 submodule（推荐，便于复现）：

```bash
git submodule update --init --recursive third_party/qwen3-asr
```

## Template / 音频占位符

- Template：`template: qwen3_asr`
- 音频占位符：在 `messages[*].content` 中使用 `<audio>`
- 音频文件：提供 `audios` 列表，长度需与 `<audio>` 占位符数量一致

## 数据格式（OpenAI messages + audios）

JSONL 每行示例：

```json
{
  "system": "",
  "messages": [
    { "role": "user", "content": "<audio>" },
    { "role": "assistant", "content": "language English<asr_text>This is a test sentence." }
  ],
  "audios": ["/abs/path/to.wav"]
}
```

建议遵循官方推荐的目标文本格式：

- `language English<asr_text>...`
- `language None<asr_text>...`（未知语种时）

## On-the-fly Packing

支持：

- `packing: true`
- `dynamic_prompt_packing: true`（训练时做 buffered packing）

示例配置见：`examples/qwen3_asr/qwen3_asr_sft_full.yaml`。

## FlashAttention 2（FA2）

在 YAML 中设置 `flash_attn: fa2`，并使用 `bf16: true`（或 `fp16`），同时确保环境中已安装 FlashAttention 2。

## 将官方 JSONL 转成 LLaMA-Factory 格式

如果你的数据是 Qwen3-ASR 官方 finetuning JSONL（字段：`audio`, `text`, 可选 `prompt`），可用脚本转换为
LLaMA-Factory 的 OpenAI+audio JSONL：

```bash
python scripts/convert_mm_data/convert_qwen3_asr_jsonl_to_openai_audio.py \
  --input /path/to/train.jsonl \
  --output /path/to/train_openai_audio.jsonl
```

## 示例配置

- 全参微调：`examples/qwen3_asr/qwen3_asr_sft_full.yaml`
- LoRA 微调：`examples/qwen3_asr/qwen3_asr_sft_lora.yaml`

