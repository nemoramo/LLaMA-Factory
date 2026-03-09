# Qwen3 Speech Endpointing 训练配置说明

## 概述

本目录包含使用 Qwen3 (0.6B / 1.7B) 以及 `Qwen/Qwen3.5-0.8B-Base` 训练语音端点检测 (Speech Endpointing) 模型的通用 YAML 配置文件。

## 核心特性

- **Neat Packing**: 启用样本打包，同时确保不会跨样本计算损失
- **LoRA**: 低秩适配微调，减少训练参数
- **FlashAttention-2**: 加速注意力计算，减少显存占用

## 配置文件说明

### 1. 训练配置: `qwen3_speech_endpointing_lora_neat_packing_fa2.yaml`

当前 Git 默认跟踪的 Qwen3 配置只有 `generic` 目录。
因此 `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B` 和 `Qwen/Qwen3.5-0.8B-Base`
都统一基于这个通用 YAML，通过 overrides 覆盖模型相关字段。

#### 关键参数说明

**模型设置:**
- `model_name_or_path`: 基础模型路径 (`Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B` 或 `Qwen/Qwen3.5-0.8B-Base`)
- `template`: 默认 `qwen3_nothink`
- 如果底模换成 `Qwen/Qwen3.5-0.8B-Base`，需要额外覆盖 `template=qwen3_5_nothink`
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

对于 **0.6B** 模型:
```yaml
per_device_train_batch_size: 32
gradient_accumulation_steps: 2
# 有效批大小 = 32 * 2 = 64
```

对于 **1.7B** 模型:
```yaml
per_device_train_batch_size: 16
gradient_accumulation_steps: 4
# 有效批大小 = 16 * 4 = 64
```

**评估设置:**
- `compute_endpointing_metrics: true`: 计算 endpointing 的 label-level 指标
- 会额外输出 `label_acc`、`label_macro_f1`、`label_far_unad`、`label_interrupt`、`label_delay`、`label_missed`
- 同时输出 `merged_label_*` 指标，对应 `treat_unaddressed_as_eou=true`
- 仍会保留 legacy `accuracy`（token-level）供对照
- `metric_for_best_model: eval_label_acc`: 用 3-way label accuracy 选择 best checkpoint
- `predict_with_generate: false`: 不使用生成模式评估
- `max_new_tokens: 1`: 只生成一个 token (分类任务)

### 2. 导出配置: `qwen3_speech_endpointing_lora_export.yaml`

用于将训练好的 LoRA adapter 合并到基础模型中，生成可部署的完整模型。

**关键参数:**
- `adapter_name_or_path`: 训练好的 LoRA checkpoint 路径
- `export_device: cpu`: 在 CPU 上执行合并（避免显存不足）
- `export_size: 2`: 分片大小 (GB)

## 代码实现要点

### 1. Template 定义 (`src/llamafactory/data/template.py`)

```python
register_template(
    name="qwen3_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)
```

### 2. 数据格式

训练数据使用 sharegpt 格式，每条记录包含:

```json
{
  "messages": [
    {"role": "system", "content": "You are a turn-taking judge..."},
    {"role": "user", "content": "[CONTEXT]\nUser: hello\n[/CONTEXT]"},
    {"role": "assistant", "content": "<EOU>"}
  ],
  "label": "<EOU>"
}
```

### 3. 训练与导出参考命令

以下命令默认在仓库根目录执行。

#### 通用训练命令（Qwen3-0.6B / Qwen3-1.7B）

```bash
llamafactory-cli train examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml \
  model_name_or_path=Qwen/Qwen3-0.6B \
  dataset_dir=/path/to/your/dataset_dir \
  dataset=speech_endpointing_train \
  eval_dataset=speech_endpointing_valid \
  output_dir=/path/to/output/qwen3_0_6b_lora_neatpacking
```

#### Qwen3.5-0.8B-Base 训练参考命令（generic YAML + overrides）

```bash
llamafactory-cli train examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml \
  model_name_or_path=Qwen/Qwen3.5-0.8B-Base \
  template=qwen3_5_nothink \
  dataset_dir=/path/to/your/dataset_dir \
  dataset=speech_endpointing_train \
  eval_dataset=speech_endpointing_valid \
  output_dir=/path/to/output/qwen3_5_0_8b_base_lora_neatpacking \
  overwrite_output_dir=false
```

如果当前环境缺 `tensorboard` 或 `matplotlib`，在训练命令后追加：

```bash
report_to=none plot_loss=false
```

#### Qwen3-0.6B 导出参考命令

```bash
llamafactory-cli export examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_export.yaml \
  model_name_or_path=Qwen/Qwen3-0.6B \
  adapter_name_or_path=/path/to/output/qwen3_0_6b_lora_neatpacking/checkpoint-xxx \
  export_dir=/path/to/exported/qwen3_0_6b_endpointing
```

#### Qwen3.5-0.8B-Base 导出参考命令

当前导出继续复用通用 export YAML，并覆盖底模与模板：

```bash
llamafactory-cli export examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_export.yaml \
  model_name_or_path=Qwen/Qwen3.5-0.8B-Base \
  template=qwen3_5_nothink \
  adapter_name_or_path=/path/to/output/qwen3_5_0_8b_base_lora_neatpacking/checkpoint-xxx \
  export_dir=/path/to/exported/qwen3_5_0_8b_base_endpointing
```

#### 训练产物里的指标查看

训练过程中，每次保存的 checkpoint 目录下都会带一个 `trainer_state.json`，可以直接查看该次 eval 的结构化指标，而不必只盯着 `tail -f` 日志：

```text
${OUTPUT_DIR}/checkpoint-200/trainer_state.json
${OUTPUT_DIR}/checkpoint-400/trainer_state.json
...
```

其中：
- `eval_label_acc`：3-way label accuracy，对应 `treat_unaddressed_as_eou=false`
- `eval_merged_label_acc`：把 `<UNADDRESSED>` 合并到 `<EOU>` 后的 accuracy，对应 `treat_unaddressed_as_eou=true`
- `best_metric` / `best_model_checkpoint`：当前 best checkpoint 的主指标和值

例如：

```bash
python - <<'PY'
import json
p = "/path/to/output_dir/checkpoint-400/trainer_state.json"
d = json.load(open(p))
print("best_metric:", d.get("best_metric"))
print("best_model_checkpoint:", d.get("best_model_checkpoint"))
for item in d.get("log_history", []):
    if "eval_label_acc" in item:
        print(item["step"], item["eval_label_acc"], item["eval_merged_label_acc"], item["eval_loss"])
PY
```

如果需要持续看结构化日志，也可以直接查看 `${OUTPUT_DIR}/trainer_log.jsonl`。

## 不同模型大小的配置差异

| 参数 | 0.6B | 1.7B |
|------|------|------|
| `per_device_train_batch_size` | 32 | 16 |
| `per_device_eval_batch_size` | 16 | 8 |
| `gradient_accumulation_steps` | 2 | 4 |
| 有效批大小 | 64 | 64 |

其余所有参数保持相同。

## 注意事项

1. **显存需求**: 
   - 0.6B 模型可在单卡 24GB GPU 上训练
   - 1.7B 模型建议单卡 40GB+ GPU

2. **数据集路径**: 请确保 `dataset_dir` 和 `dataset` 配置正确指向你的数据文件

3. **特殊 Token**: 导出时必须使用与训练时相同的 `add_special_tokens` 配置

4. **Neat Packing**: 这是分类任务的关键，确保损失只计算在正确的样本上
