# FunAudioChat GRPO Expert Handoff

## Scope of this branch

This branch adds a first working `stage: grpo` path for `FunAudioChat` inside `LLaMA-Factory`.

Main areas:

- `src/llamafactory/train/grpo/`
  - `workflow.py`: GRPO entrypoint and model/trainer wiring.
  - `trainer.py`: FunAudioChat-specific GRPO trainer built around TRL GRPO, custom multimodal rollout, debug instrumentation.
  - `reward.py`: rule-based ASR reward using normalized text + WER/CER.
- `src/llamafactory/data/processor/grpo.py`
  - prompt-only dataset view for GRPO while preserving `reference_text`.
- `src/llamafactory/data/loader.py`
  - raw dataset path for `stage == "grpo"`.
- `src/llamafactory/hparams/finetuning_args.py`
  - GRPO config surface.
- `src/llamafactory/hparams/parser.py`
  - validation and guardrails for FunAudioChat GRPO.
- `examples/funaudiochat/*.yaml`
  - LoRA and full-LLM GRPO examples used during bring-up.
- `scripts/repro_funaudiochat_vllm_generate.py`
  - non-GRPO reproduction path for isolating `vLLM.generate()` with FunAudioChat audio requests.

## What works

- `stage: grpo` is integrated into `llamafactory-cli train`.
- FunAudioChat prompt-only GRPO data flow works.
- Rule-based ASR reward is wired into rollout.
- vLLM colocate rollout works for smoke tests.
- Full-LLM + colocate + TP>1 can start and train for a while with the current branch plus the local vLLM modifications described below.
- The parser now fail-fast blocks the known-unsafe `full + colocate + TP>1` FunAudioChat path unless the user sets
  `grpo_allow_experimental_funaudiochat_colocate_tp: true`.

## Local vLLM dependency outside this branch

The latest stability experiments depended on local changes in `~/projects/vllm` and are not contained in this commit.

Relevant local vLLM files:

- `~/projects/vllm/vllm/envs.py`
- `~/projects/vllm/vllm/v1/worker/gpu_model_runner.py`
- `~/projects/vllm/vllm/entrypoints/llm.py`
- `~/projects/vllm/vllm/model_executor/models/funaudiochat.py`

Most important external change:

- Added `VLLM_FUNAUDIOCHAT_AUDIO_BATCH_MODE=auto|microbatch|batch`
- `auto` currently forces FunAudioChat audio MM encoding to `microbatch` when `TP > 1`

Without those local vLLM changes, the latest full-LLM TP>1 debugging state is not reproducible.

## Main failure modes observed

### 1. Audio MM encoder hang in vLLM when audio batch > 1

Observed behavior:

- Training hangs with no traceback.
- GPUs remain busy.
- Logs stop inside the rollout path.

Most likely path:

- `vLLM.generate()`
- `embed_multimodal -> audio_tower`
- FunAudioChat audio batch size `> 1`

Evidence:

- A debug run previously stopped after `embed_multimodal start modality=audio num_items=2`
- Forcing audio MM encoding to per-item microbatching allowed training to pass earlier hang points

Current workaround:

- Use the local vLLM `audio batch mode = auto`, which becomes `microbatch` for FunAudioChat audio when `TP > 1`

### 2. Additional full-LLM colocate hang after earlier fixes

Even after the audio microbatch workaround, full-LLM colocate training still hangs later in rollout.

Latest focused reproduction:

- output dir:
  `/data2/mayufeng/saves/funaudiochat/grpo_asr_hausa_full_llm_4gpu_tp2_step107_debug_20260310_150427`

Latest observed stuck point:

- last completed trainer phase:
  - `world sync before rollout`
  - `sync-to-vllm weights done`
  - `world sync after weight sync`
  - `reset-prefix-cache done`
  - `prompt gather done`
  - `audio gather done`
  - `vllm input build done`
- final visible phase:
  - `GRPO rollout start step=79`

Interpretation:

- the current unresolved hang is now narrowed to `vLLM.generate()` itself or the immediately adjacent path after request construction
- it is not currently pointing at:
  - reward calculation
  - prompt/audio gather
  - cache reset
  - weight sync barriers

## Stabilization attempts already made

### In this branch

- Added explicit `stage: grpo` routing.
- Added prompt-only GRPO dataset processor.
- Added shared ASR text normalization utilities.
- Added FunAudioChat-specific GRPO trainer.
- Added rollout phase logging around:
  - weight sync
  - world barriers
  - cache reset
  - prompt gather
  - audio gather
  - vLLM input build
  - rollout start/done
- Split vLLM weight sync and cache reset into separate phases for diagnosis.
- Added optional `SIGUSR1` faulthandler registration in launcher workers.
- Disabled vLLM custom collective op usage via `use_custom_op_call = False`.

### Outside this branch in local vLLM

- Added audio MM encoder batch mode selection.
- Added debug logs in FunAudioChat MM encoding path.
- Added microbatch workaround for audio MM encoding with `TP > 1`.

## What still looks weak

- Full-LLM + colocate + TP>1 is still not stable enough for a full run.
- The unresolved hang is still inside the rollout engine path.
- This means the GRPO branch itself is usable for smoke tests and debugging, but not yet production-stable for long full-LLM colocate runs.

## Recommended next steps for expert review

1. Review the local vLLM FunAudioChat multimodal execution path under `TP > 1`, especially around `generate()` after request construction.
2. Decide whether colocate weight sync + live rollout is the right architecture for full-LLM FunAudioChat GRPO, or whether rollout should move to a decoupled engine.
3. Audit whether the FunAudioChat MM encoder path in vLLM is safe under variable audio lengths and `TP > 1`.
4. If staying with colocate, add a hard timeout and per-batch repro dump directly around the final `generate()` call.

## Suggested review entry points

- `src/llamafactory/train/grpo/trainer.py`
- `src/llamafactory/train/grpo/workflow.py`
- `src/llamafactory/data/processor/grpo.py`
- `src/llamafactory/hparams/finetuning_args.py`
- `~/projects/vllm/vllm/v1/worker/gpu_model_runner.py`
- `~/projects/vllm/vllm/model_executor/models/funaudiochat.py`
