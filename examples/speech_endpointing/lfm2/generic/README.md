# LFM2.5 Speech Endpointing 训练配置说明

## 概述

本目录包含使用 LFM2.5-1.2B-Instruct 训练语音端点检测 (Speech Endpointing) 模型的通用 YAML 配置文件。

## 核心特性

- **Neat Packing**: 启用样本打包，同时确保不会跨样本计算损失
- **LoRA**: 低秩适配微调，减少训练参数
- **FlashAttention-2**: 加速注意力计算，减少显存占用

## 配置文件说明

### 1. 训练配置: `lfm2_5_speech_endpointing_lora_neat_packing_fa2.yaml`

#### 关键参数说明

**模型设置:**
- `model_name_or_path`: LiquidAI/LFM2.5-1.2B-Instruct
- `template`: lfm2 - LFM2.5 模板
- `flash_attn: fa2`: 启用 FlashAttention-2
- `add_special_tokens`: 添加 3 个端点检测标签 token
  - `<EOU>`: End of Utterance - 用户发言结束
  - `<CONT_USER>`: 用户可能继续发言
  - `<UNADDRESSED>`: 非对助手的语音
- `resize_vocab: true`: 调整词表大小以容纳新 token

**LoRA 设置:**
- `lora_rank: 16`: LoRA 秩
- `lora_alpha: 32`: LoRA alpha (缩放因子 = alpha/rank = 2)
- `lora_target`: 目标模块 (所有线性层)

**数据集设置:**
- `dataset_dir: /path/to/your/dataset_dir`: 数据集目录路径（包含 dataset_info.json）
- `packing: true`: 启用样本打包
- `neat_packing: true`: 启用"整洁打包"，避免跨样本损失
- `cutoff_len: 512`: 最大序列长度

**训练设置:**
- 有效批大小 = 16 * 4 = 64
- 学习率: 2.0e-4
- 训练周期: 2.0 epochs

**评估设置:**
- `compute_accuracy: true`: 计算 token 级别准确率
- `predict_with_generate: false`: 不使用生成模式评估
- `max_new_tokens: 1`: 只生成一个 token (分类任务)

### 2. 导出配置: `lfm2_5_speech_endpointing_lora_export.yaml`

用于将训练好的 LoRA adapter 合并到基础模型中，生成可部署的完整模型。

**关键参数:**
- `adapter_name_or_path`: 训练好的 LoRA checkpoint 路径
- `export_device: cpu`: 在 CPU 上执行合并（避免显存不足）
- `export_size: 2`: 分片大小 (GB)

## 使用方式

```bash
# 训练
llamafactory-cli train lfm2_5_speech_endpointing_lora_neat_packing_fa2.yaml

# 导出
llamafactory-cli export lfm2_5_speech_endpointing_lora_export.yaml
```

## 注意事项

1. **显存需求**: 1.2B 模型建议单卡 40GB+ GPU
2. **数据集路径**: 请确保 `dataset_dir` 和 `dataset` 配置正确指向你的数据文件
3. **特殊 Token**: 导出时必须使用与训练时相同的 `add_special_tokens` 配置
4. **Neat Packing**: 这是分类任务的关键，确保损失只计算在正确的样本上
