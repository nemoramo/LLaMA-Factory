# Speech Endpointing（三分类标签）

本目录提供多套 **Speech Endpointing / Turn-taking（三分类标签）** 的训练与导出配置（YAML），已按 **模型** 分组索引（见下）。

这个 recipe 用 **LLaMA-Factory + LoRA(SFT)** 把"端点检测/轮次接续（turn-taking）"做成 **下一 token 三分类**：
模型对输入对话只输出 **一个标签 token**（`<EOU>` / `<CONT_USER>` / `<UNADDRESSED>`）。

本方案使用 **新增 special tokens + resize vocab**（不复用 unused token），适合需要模型直接输出上述 3 个字符串标签的场景。

---

## 0. 配置索引（按模型）

### Qwen3 (Instruct)

- Qwen3-0.6B / Qwen3-1.7B
  - 通用训练配置（neat packing + LoRA + FlashAttention-2）：`examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml`
  - 通用导出配置（merge LoRA）：`examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_export.yaml`

### Qwen2.5 (Instruct)

- Qwen2.5-3B
  - 训练（neat packing）：`examples/speech_endpointing/qwen2_5/3b/qwen2_5_3b_speech_endpointing_lora_neat_packing.yaml`
  - 导出（merge LoRA）：`examples/speech_endpointing/qwen2_5/3b/qwen2_5_3b_speech_endpointing_lora_export.yaml`
- Qwen2.5-0.5B
  - 训练（neat packing）：`examples/speech_endpointing/qwen2_5/0_5b/qwen2_5_0_5b_speech_endpointing_lora_neat_packing.yaml`
  - 导出（merge LoRA）：`examples/speech_endpointing/qwen2_5/0_5b/qwen2_5_0_5b_speech_endpointing_lora_export.yaml`

### LFM2

- LFM2.5-1.2B
  - 通用训练配置（neat packing + LoRA + FlashAttention-2）：`examples/speech_endpointing/lfm2/generic/lfm2_5_speech_endpointing_lora_neat_packing_fa2.yaml`
  - 通用导出配置（merge LoRA）：`examples/speech_endpointing/lfm2/generic/lfm2_5_speech_endpointing_lora_export.yaml`

---

## 1. 核心思路

- 在 tokenizer 中新增 3 个 special tokens：`<EOU>`, `<CONT_USER>`, `<UNADDRESSED>`
- 训练时打开 `resize_vocab: true`，触发 embedding/lm_head 扩表与初始化
- 数据中 assistant 回复 **严格等于一个标签 token**（不要附带解释文本、标点、空格或换行）
- 推理时固定 `max_new_tokens=1`、`temperature=0`，让模型只吐 1 个标签

> 注意：量化模型（4bit/8bit）通常无法 resize embedding；本 recipe 默认不启用量化。

---

## 2. 数据格式（OpenAI messages JSONL）

推荐使用 OpenAI messages：每行一个 JSON（JSONL），包含 `messages` 数组，元素形如：

```json
{"messages":[
  {"role":"system","content":"You are a turn-taking judge... Output exactly one tag."},
  {"role":"user","content":"[CONTEXT]\nUser: ...\n[/CONTEXT]"},
  {"role":"assistant","content":"<EOU>"}
]}
```

要求：
- `role` 只用 `system/user/assistant`
- assistant 的 `content` 必须是 **`<EOU>` / `<CONT_USER>` / `<UNADDRESSED>` 之一**，且只包含这个 token

仓库内提供一个最小样例：`examples/speech_endpointing/speech_endpointing_sample.jsonl`

---

## 3. 从 TorchTune manifest 转换（可选）

如果你当前数据是 TorchTune 的 `*.manifest`（JSONL），通常每行会带有 `label/context/messages/meta` 等字段。
LLaMA-Factory 的 neat packing 不要求"预先 packing"，只要样本能被解析成 OpenAI messages 即可。

本目录提供一个 stdlib-only 的转换脚本：

```bash
python examples/speech_endpointing/convert_torchtune_manifest.py \
  --input /path/to/train_set.manifest \
  --output /path/to/speech_endpointing_train.jsonl \
  --print-dataset-info --dataset-key speech_endpointing_train
```

验证集同理：

```bash
python examples/speech_endpointing/convert_torchtune_manifest.py \
  --input /path/to/test_set.manifest \
  --output /path/to/speech_endpointing_eval.jsonl \
  --print-dataset-info --dataset-key speech_endpointing_eval
```

> 常见文件名可能是 `train_set.manifest/test_set.manifest` 或 `trainset.manifest/testset.manifest`，按实际路径传入即可。

---

## 4. 配置 dataset_info.json

假设你的数据目录是 `${DATASET_DIR}`（里面有 `dataset_info.json` 和数据文件），新增两条数据集定义，例如：

```json
{
  "speech_endpointing_train": {
    "file_name": "speech_endpointing_train.jsonl",
    "formatting": "openai",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system",
      "observation_tag": "observation",
      "function_tag": "function"
    }
  },
  "speech_endpointing_eval": {
    "file_name": "speech_endpointing_eval.jsonl",
    "formatting": "openai",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system",
      "observation_tag": "observation",
      "function_tag": "function"
    }
  }
}
```

然后把你的训练/验证文件放到同目录下：
- `${DATASET_DIR}/speech_endpointing_train.jsonl`
- `${DATASET_DIR}/speech_endpointing_eval.jsonl`

---

## 5. 训练（LoRA + resize vocab）

编辑配置：`examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml`

你需要至少修改：
- `model_name_or_path`: 选择 `Qwen/Qwen3-0.6B` 或 `Qwen/Qwen3-1.7B`
- `dataset_dir`
- `dataset` / `eval_dataset`
- `output_dir`

启动训练（支持 Hydra overrides：在 YAML 路径后追加 `key=value` 覆盖配置）：

```bash
llamafactory-cli train examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml \
  model_name_or_path=Qwen/Qwen3-0.6B \
  dataset_dir=/path/to/your/dataset_dir \
  output_dir=/path/to/output_dir
```

也可以用本目录的统一入口脚本（单卡）：

```bash
GPU_ID=0 CFG=examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml \
  bash examples/speech_endpointing/run.sh train \
    model_name_or_path=Qwen/Qwen3-0.6B \
    dataset_dir=/path/to/your/dataset_dir \
    output_dir=/path/to/output_dir
```

例如：复现"单卡 + 频繁评估 + 训练 3 epoch（续训）"这类变体，不需要单独维护 YAML，只要覆盖少量字段：

```bash
llamafactory-cli train examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml \
  model_name_or_path=Qwen/Qwen3-0.6B \
  dataset_dir=/path/to/your/dataset_dir \
  output_dir=/path/to/output_dir \
  save_steps=200 eval_steps=200 \
  overwrite_output_dir=false \
  num_train_epochs=3.0
```

如需关闭 packing / neat packing（例如遇到版本兼容问题）：

```bash
llamafactory-cli train examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml \
  packing=false neat_packing=false
```

导出（merge LoRA）同理，用 export 配置并覆盖输出路径即可：

```bash
llamafactory-cli export examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_export.yaml \
  model_name_or_path=Qwen/Qwen3-0.6B \
  adapter_name_or_path=/path/to/lora_adapter_dir \
  export_dir=/path/to/export_dir
```

### 不同模型大小的 batch size 配置

| 模型 | per_device_train_batch_size | gradient_accumulation_steps | 有效 batch size |
|------|----------------------------|----------------------------|----------------|
| Qwen3-0.6B | 32 | 2 | 64 |
| Qwen3-1.7B | 16 | 4 | 64 |

### 关于新增 token 的训练（重要）

- 本 recipe 通过 `add_special_tokens` + `resize_vocab` 扩表。
- LoRA 场景下，如果你 **额外设置了** `additional_target`（例如多模态 projector），请确保把 embedding/lm_head 也包含进去；否则新增 token 可能学不动。

---

## 6. Neat Packing（LLaMA-Factory 内置）

neat packing 是 LLaMA-Factory 在数据预处理阶段做的"无跨样本 attention 的 packing"，适合大量短样本（endpointing 很典型）。

在训练配置里打开：

- `packing: true`
- `neat_packing: true`

本目录提供已开启 neat packing 的配置示例：
- `examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml`
- `examples/speech_endpointing/qwen2_5/3b/qwen2_5_3b_speech_endpointing_lora_neat_packing.yaml`

注意：
- neat packing 只能在 **SFT** 使用（仓库会做检查）。
- 部分 transformers 版本与 neat packing 存在兼容性限制；如果启动时报错，请按报错提示调整环境版本或先关闭 neat packing。

---

## 7. 推理建议

endpointing 需要"只生成 1 个标签"：
- `do_sample: false`
- `temperature: 0.0`
- `max_new_tokens: 1`

如果你用自己的推理脚本（Transformers），注意 decode 时可能需要 `skip_special_tokens=False` 才能看到 `<EOU>` 这种标签字符串。

---

## 8. 常见坑

- **量化训练**：4/8bit 通常无法 resize embedding（会报错）；请先用 BF16/FP16 跑通。
- **assistant 输出污染**：assistant 一旦输出了多余字符（比如 `"<EOU>\n"` 或 `" <EOU>"`），就不再是严格单 token 分类，评估会变得不稳定。
- **评估指标**：本 recipe 用 `compute_accuracy`（token-level），前提是 assistant 段尽量只包含标签 token（推荐 `max_new_tokens=1` 推理时再做严格分类评估）。
