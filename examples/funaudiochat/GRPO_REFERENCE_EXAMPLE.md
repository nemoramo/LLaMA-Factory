# FunAudioChat GRPO Reference Example

This branch contains a validated FunAudioChat GRPO reference run for Hausa ASR.

## Branch

- Branch: `feature/funaudiochat-grpo-asr`
- Worktree: `worktrees/funaudiochat-grpo-asr`

## Reference training config

- In-repo example: `examples/funaudiochat/funaudiochat_hausa_grpo_asr_full_llm_4gpu_tp2_stable.yaml`
- Successful run config: `/data2/mayufeng/saves/funaudiochat/grpo_asr_hausa_full_llm_4gpu_tp2_formal10k_20260311/train_config.yaml`
- Training command: `/data2/mayufeng/saves/funaudiochat/grpo_asr_hausa_full_llm_4gpu_tp2_formal10k_20260311/training_command.txt`

## Reference setup

- Base model: `/data2/mayufeng/saves/funaudiochat/hausa_fullsft_llm_plus_projector_freeze_audio_from_h100_ckpt211000_gpus1to4_20260227/checkpoint-60000`
- Stage: `grpo`
- Finetuning type: `full`
- GPU topology: 4 GPUs for training, colocated vLLM rollout with `tensor_parallel_size=2`
- Precision: `bf16` + `flash_attn: fa2`
- Trainable scope: LLM only
  - `funaudiochat_freeze_audio_tower: true`
  - `freeze_multi_modal_projector: true`
- Optimizer schedule:
  - `learning_rate: 1e-6`
  - `warmup_ratio: 0.02`
  - `lr_scheduler_type: cosine`
- Effective training batch:
  - `per_device_train_batch_size: 1`
  - `gradient_accumulation_steps: 8`
- GRPO:
  - `grpo_num_generations: 8`
  - `grpo_generation_batch_size: 8`
  - `grpo_loss_type: dapo`
  - `grpo_scale_rewards: group`
  - `grpo_beta: 0.0`
  - `grpo_temperature: 1.0`
  - `grpo_top_p: 0.95`

## Why this example matters

This is the first full FunAudioChat + GRPO configuration on this branch that completed a long full-parameter run without the earlier TP rollout hangs.

The stabilizing points for this example are:

- Canonical TP request construction for multimodal rollout
- FunAudioChat audio batching safeguards in the local vLLM path
- `TP=2` instead of the earlier unstable `TP=4`
- HF inference dtype fix for local API deployment (`dtype` + `torch_dtype`)

## Reference output

- Final GRPO model dir: `/data2/mayufeng/saves/funaudiochat/grpo_asr_hausa_full_llm_4gpu_tp2_formal10k_20260311`

## Reference eval (local port + `subprompt1`)

These numbers are from local OpenAI-compatible inference on port `30005`, using the `subprompt1` prompt format aligned with training.

- Eval output dir: `/data2/mayufeng/llamafactory_eval/funaudiochat_grpo_hausa_20260312/results_local_30005_subprompt1`

| Dataset | WER | CER | WERE |
| --- | ---: | ---: | ---: |
| youtube | 22.2054% | 8.1501% | 15.4933% |
| fleurs | 22.4374% | 6.2420% | 18.3940% |
| haiwa | 18.4142% | 5.9531% | 14.2410% |
| return_data | 29.2270% | 10.0369% | 20.6015% |
| weighted avg | 25.7980% | 8.4573% | 19.0125% |

## Matching eval script

- Inference client: `/home/mayufeng/projects/speech_related_tools/scripts/funaudiochat_vllm_infer_manifest.py`
- Eval script: `/home/mayufeng/projects/speech_related_tools/evaluate/eval_asr_wer_cer.py`
- Local runner used for the four Hausa sets:
  - `/data2/mayufeng/llamafactory_eval/funaudiochat_grpo_hausa_20260312/run_four_hausa_eval_local.sh`
