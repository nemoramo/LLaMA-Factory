# Speech Endpointing（三分类标签）

本目录提供多套 **Speech Endpointing / Turn-taking（三分类标签）** 的训练与导出配置（YAML），已按 **模型** 分组索引（见下）。

这个 recipe 用 **LLaMA-Factory + LoRA(SFT)** 把"端点检测/轮次接续（turn-taking）"做成 **下一 token 三分类**：
模型对输入对话只输出 **一个标签 token**（`<EOU>` / `<CONT_USER>` / `<UNADDRESSED>`）。

本方案使用 **新增 special tokens + resize vocab**（不复用 unused token），适合需要模型直接输出上述 3 个字符串标签的场景。

---

## 0. 配置索引（按模型）

### Qwen3 / Qwen3.5

- Qwen3-0.6B / Qwen3-1.7B / Qwen3.5-0.8B-Base
  - 通用训练配置（neat packing + LoRA + FlashAttention-2）：`examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml`
  - 通用导出配置（merge LoRA）：`examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_export.yaml`
  - 使用 Qwen3.5-0.8B-Base 时，额外覆盖：`model_name_or_path=Qwen/Qwen3.5-0.8B-Base template=qwen3_5_nothink`

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
  --output /path/to/speech_endpointing_valid.jsonl \
  --print-dataset-info --dataset-key speech_endpointing_valid
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
  "speech_endpointing_valid": {
    "file_name": "speech_endpointing_valid.jsonl",
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
- `${DATASET_DIR}/speech_endpointing_valid.jsonl`

---

## 5. 训练（LoRA + resize vocab）

编辑配置：`examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml`

推荐数据文件命名：
- **训练集**: `${DATASET_DIR}/speech_endpointing_train.jsonl`
- **验证集**: `${DATASET_DIR}/speech_endpointing_valid.jsonl`

如需使用其他数据，修改：
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

如果你要复现当前的 **Qwen3.5-0.8B-Base + no thinking + 0202 数据** 单卡实验，可以直接覆盖以下字段：

```bash
PROJECT_ROOT=/path/to/LLaMA-Factory \
DATASET_DIR=/path/to/your/dataset_dir \
OUTPUT_DIR=/path/to/output/qwen3_5_0_8b_base_lora_neatpacking \
PYTHON_BIN=/path/to/your/python \
HOME=/path/to/runtime_home \
TMPDIR=/path/to/runtime_tmp \
XDG_CACHE_HOME=/path/to/xdg_cache \
HF_HOME=/path/to/hf_cache \
HF_HUB_CACHE=/path/to/hf_cache/hub \
TORCH_HOME=/path/to/torch_cache \
TORCH_EXTENSIONS_DIR=/path/to/torch_extensions \
CUDA_VISIBLE_DEVICES=5 \
DISABLE_VERSION_CHECK=1 \
ALLOW_TORCH_2_9_CONV3D=1 \
PYTHONPATH=${PROJECT_ROOT}/src \
${PYTHON_BIN} -m llamafactory.cli train \
  ${PROJECT_ROOT}/examples/speech_endpointing/qwen3/generic/qwen3_speech_endpointing_lora_neat_packing_fa2.yaml \
  model_name_or_path=Qwen/Qwen3.5-0.8B-Base \
  template=qwen3_5_nothink \
  dataset_dir=${DATASET_DIR} \
  dataset=speech_endpointing_train \
  eval_dataset=speech_endpointing_valid \
  output_dir=${OUTPUT_DIR} \
  overwrite_output_dir=false \
  report_to=none \
  plot_loss=false
```

说明：
- `report_to=none`：当前环境若未安装 `tensorboard`，需要关闭默认 TensorBoard callback。
- `plot_loss=false`：当前环境若未安装 `matplotlib`，需要关闭 loss 曲线绘图。
- `HOME/TMPDIR/HF_HOME/...`：把 cache、临时文件和 torch extensions 放到 `/data*`，避免实验产物写进 `home` 或 `/root`。

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

### 训练过程中的指标查看

不要只依赖 `tail -f` 训练日志。每次保存出来的 checkpoint 目录里都会有一个 `trainer_state.json`，里面会保留结构化的 eval 结果和 best-checkpoint 状态，例如：

```text
${OUTPUT_DIR}/checkpoint-200/trainer_state.json
${OUTPUT_DIR}/checkpoint-400/trainer_state.json
...
```

对于 speech endpointing，建议重点看：
- `eval_label_acc`：3 类原始标签准确率，对应 `treat_unaddressed_as_eou=false`
- `eval_merged_label_acc`：把 `<UNADDRESSED>` 合并到 `<EOU>` 后的准确率，对应 `treat_unaddressed_as_eou=true`
- `eval_label_macro_f1`
- `eval_label_far_unad`
- `eval_label_interrupt`
- `eval_label_delay`
- `eval_label_missed`
- `best_metric` / `best_model_checkpoint`

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
        print(item["step"], item["eval_label_acc"], item["eval_merged_label_acc"], item["eval_label_far_unad"])
PY
```

如果想持续看结构化训练日志，也可以直接看 `${OUTPUT_DIR}/trainer_log.jsonl`。

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

## 8. 评估口径

speech endpointing 目前有 3 套常用评估口径，含义不同：

- **Trainer 内评估（推荐）**：开启 `compute_endpointing_metrics: true`
  - 3-way 原始标签指标：`eval_label_acc`、`eval_label_macro_f1`、`eval_label_far_unad`、`eval_label_interrupt`、`eval_label_delay`、`eval_label_missed`
  - 2-way merge 指标：`eval_merged_label_acc`、`eval_merged_label_macro_f1`、`eval_merged_label_interrupt`、`eval_merged_label_delay`
  - 其中 `merged_label_*` 等价于 `treat_unaddressed_as_eou=true`
  - 当前 generic Qwen3 recipe 默认用 `metric_for_best_model: eval_label_acc`

- **`eval_hf_endpointing.py`**：直接对本地 HF / merged 模型做离线评估
  - 会同时输出 `tag_eval`（3-way）和 `tag_eval_merge_unad_as_eou`（2-way merge）
  - 还会输出 `export_prompt_probe`
  - 这个 probe 会拿固定 endpointing prompt 做 next-token 检查：如果 `<EOU>` / `<CONT_USER>` / `<UNADDRESSED>` 没有占据 full-vocab top3，默认先排查 export
  - 优先检查导出后 `config.json` 里的 `tie_word_embeddings`，以及 `embed_tokens` / `lm_head` 在 merge 后是否仍然保持正确的 tied-embedding 关系，尤其是 Qwen3 / Qwen3.5
  - 然后再检查 `llamafactory-cli export` 的 `template` 是否正确、是否保留了 `add_special_tokens` / `resize_vocab`、当前评估目录是否真的是 merged 导出目录，以及评估 prompt 是否和训练 / 部署一致
  - 适合快速检查合并后模型的离线精度和导出健康状态

- **`eval_sglang_endpointing.py`**：对已部署的 OpenAI-compatible 服务做评估
  - 同时输出 `tag_eval`（等价于 `treat_unaddressed_as_eou=false`）
  - 以及 `tag_eval_merge_unad_as_eou`（等价于 `treat_unaddressed_as_eou=true`）
  - 适合对齐线上服务逻辑与真实部署 KPI

---

## 9. 常见坑

- **量化训练**：4/8bit 通常无法 resize embedding（会报错）；请先用 BF16/FP16 跑通。
- **assistant 输出污染**：assistant 一旦输出了多余字符（比如 `"<EOU>\n"` 或 `" <EOU>"`），就不再是严格单 token 分类，评估会变得不稳定。
- **评估指标**：speech endpointing 推荐开启 `compute_endpointing_metrics`。它会以 assistant 监督段的第一个标签 token 作为分类目标，输出 `label_acc`、`label_macro_f1`、`label_far_unad`、`label_interrupt`、`label_delay`、`label_missed`，并额外输出 `merged_label_*`（等价于 `treat_unaddressed_as_eou=true`）。legacy `accuracy` 仍会保留为 token-level 对照指标。
