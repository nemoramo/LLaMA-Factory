#!/usr/bin/env python3
"""
Standalone Unsloth Gemma-3n Audio SFT with dynamic prompt_pool sampling.

This script is intentionally *not* wired into LLaMA-Factory's Trainer / dataset pipeline.
It consumes the ShareGPT-Audio jsonl produced by LLaMA-Factory converters:
  - messages: [{"role": "user"/"assistant", "content": "... <audio> ..."}, ...]
  - audios: ["/abs/path.wav"]
  - prompt_pool (optional): [{"text": "...", "completion": "...", "weight": 0.2}, ...]

Dynamic prompt behavior:
  - one prompt_pool entry is sampled per example at collate time
  - entry.text is appended to the last user message
  - entry.completion overwrites the assistant target text

Limitations:
  - only 1 audio per example is supported (len(audios) == 1).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence


def _sanitize_weight(w: Any) -> float:
    try:
        fw = float(w)
    except Exception:
        fw = 1.0
    if not math.isfinite(fw) or fw < 0:
        return 0.0
    return fw


def _stable_hash64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _choose_from_pool(pool: Sequence[Any], rng: random.Random) -> Any:
    if len(pool) == 0:
        raise ValueError("prompt_pool is empty.")

    values: list[Any] = []
    weights: list[float] = []
    for item in pool:
        w = 1.0
        if isinstance(item, dict) and "weight" in item:
            w = _sanitize_weight(item.get("weight"))
        values.append(item)
        weights.append(w)

    if any(w > 0 for w in weights):
        return rng.choices(values, weights=weights, k=1)[0]
    return rng.choice(values)


def _get_sample_id(example: dict[str, Any], id_key: Optional[str]) -> str:
    if isinstance(id_key, str) and id_key and id_key in example and example[id_key] is not None:
        return str(example[id_key])
    audios = example.get("audios")
    if isinstance(audios, list) and len(audios) > 0 and audios[0] is not None:
        return str(audios[0])
    messages = example.get("messages")
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
    return str(messages) if messages is not None else ""


def _choose_from_pool_deterministic(pool: Sequence[Any], example: dict[str, Any], seed: int, id_key: Optional[str]) -> Any:
    if len(pool) == 0:
        raise ValueError("prompt_pool is empty.")

    sample_id = _get_sample_id(example, id_key=id_key)
    h = _stable_hash64(f"{int(seed)}|{sample_id}")

    values: list[Any] = []
    weights: list[float] = []
    for item in pool:
        w = 1.0
        if isinstance(item, dict) and "weight" in item:
            w = _sanitize_weight(item.get("weight"))
        values.append(item)
        weights.append(w)

    total = sum(weights)
    if total > 0:
        r = (h / float(2**64)) * total
        acc = 0.0
        for v, w in zip(values, weights):
            if w <= 0:
                continue
            acc += w
            if r < acc:
                return v
        return values[-1]

    return values[int(h % len(values))]


def _append_suffix_to_last_user(messages: list[dict[str, Any]], suffix: str) -> None:
    if not suffix:
        return
    for m in reversed(messages):
        if m.get("role") == "user":
            base = m.get("content", "")
            if not isinstance(base, str):
                base = str(base)
            sep = ""
            if base and not base.endswith(("\n", " ")) and not suffix.startswith(("\n", " ")):
                sep = "\n"
            m["content"] = f"{base}{sep}{suffix}" if base else suffix
            return
    messages.append({"role": "user", "content": suffix})


def _override_last_assistant(messages: list[dict[str, Any]], completion: Optional[str]) -> None:
    if completion is None:
        return
    completion_str = str(completion)
    for m in reversed(messages):
        if m.get("role") == "assistant":
            m["content"] = completion_str
            return
    messages.append({"role": "assistant", "content": completion_str})


def _load_audio_array(path: str, target_sr: int) -> "list[float]":
    """Load mono audio and resample to target_sr if needed. Returns float32 list."""
    import numpy as np

    if not path:
        raise ValueError("Empty audio path.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    audio: Any = None
    sr: Optional[int] = None

    # Try soundfile first (fast, common).
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(path, always_2d=False)
    except Exception:
        audio = None
        sr = None

    # Fallback: librosa (handles many formats + resampling).
    if audio is None or sr is None:
        try:
            import librosa  # type: ignore

            audio, sr = librosa.load(path, sr=None, mono=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio: {path}. Install `soundfile` or `librosa`. Original error: {e}"
            ) from e

    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert to float32 in [-1, 1] best-effort.
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(1.0, float(info.max))
    else:
        audio = audio.astype(np.float32)

    if sr is None:
        raise RuntimeError(f"Audio sample rate unknown: {path}")

    if int(sr) != int(target_sr):
        # Prefer librosa.resample if available, otherwise try torchaudio.
        try:
            import librosa  # type: ignore

            audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
            sr = int(target_sr)
        except Exception:
            try:
                import torch
                import torchaudio

                wav = torch.from_numpy(audio).float().unsqueeze(0)
                wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))
                audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
                sr = int(target_sr)
            except Exception as e:
                raise RuntimeError(
                    f"Need `librosa` or `torchaudio` to resample {path} from {sr} -> {target_sr}. Error: {e}"
                ) from e

    if audio.size == 0:
        raise RuntimeError(f"Empty audio after decoding: {path}")

    return audio.tolist()


def _extract_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages up to (but excluding) the first assistant message."""
    prompt_messages: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            break
        if isinstance(m, dict):
            prompt_messages.append(m)
    return prompt_messages


def _extract_reference_text(messages: list[dict[str, Any]]) -> Optional[str]:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            return None if content is None else str(content)
    return None


def _build_generation_prompt_text(
    processor: Any,
    example: dict[str, Any],
    *,
    dataset_audio_placeholder: str,
    dynamic_prompt: bool,
    dynamic_prompt_deterministic: bool,
    dynamic_prompt_id_key: Optional[str],
    seed: int,
    rng: random.Random,
) -> str:
    messages = example.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError("Each example must have `messages` as list[dict].")
    messages = copy.deepcopy(messages)

    prompt_messages = _extract_prompt_messages(messages)

    if dynamic_prompt:
        pool = example.get("prompt_pool")
        if isinstance(pool, list) and len(pool) > 0:
            if dynamic_prompt_deterministic:
                chosen = _choose_from_pool_deterministic(pool, example, seed=seed, id_key=dynamic_prompt_id_key)
            else:
                chosen = _choose_from_pool(pool, rng)
            if isinstance(chosen, dict):
                suffix = str(chosen.get("text") or chosen.get("suffix") or "")
            else:
                suffix = "" if chosen is None else str(chosen)
            _append_suffix_to_last_user(prompt_messages, suffix=suffix)

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True).strip()

    audio_token = getattr(processor, "audio_token", None) or getattr(processor.tokenizer, "audio_token", None)
    if not isinstance(audio_token, str) or not audio_token:
        raise ValueError("Cannot determine Gemma3n audio_token from processor/tokenizer.")

    return prompt_text.replace(dataset_audio_placeholder, audio_token)


def _generate_and_dump_samples(
    *,
    model: Any,
    processor: Any,
    dataset: Any,
    output_path: str,
    num_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    dataset_audio_placeholder: str,
    dynamic_prompt: bool,
    dynamic_prompt_deterministic: bool,
    dynamic_prompt_id_key: Optional[str],
    print_first: int = 3,
) -> None:
    if dataset is None or num_samples <= 0:
        return

    import json

    import torch

    # Best-effort device resolution.
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_sr = (
        getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        or getattr(processor, "audio_sampling_rate", None)
        or 16000
    )
    target_sr = int(target_sr)

    n = len(dataset)
    k = min(int(num_samples), int(n))
    rng = random.Random(int(seed))
    indices = list(range(n))
    indices = rng.sample(indices, k) if n > k else indices

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model_was_training = getattr(model, "training", False)
    model.eval()

    def _short(x: Optional[str], limit: int = 200) -> str:
        if x is None:
            return ""
        s = str(x).replace("\n", "\\n")
        return s if len(s) <= limit else s[: limit - 3] + "..."

    printed = 0
    with open(output_path, "w", encoding="utf-8") as f, torch.no_grad():
        for idx in indices:
            ex = dataset[int(idx)]

            prompt_text = _build_generation_prompt_text(
                processor,
                ex,
                dataset_audio_placeholder=dataset_audio_placeholder,
                dynamic_prompt=dynamic_prompt,
                dynamic_prompt_deterministic=dynamic_prompt_deterministic,
                dynamic_prompt_id_key=dynamic_prompt_id_key,
                seed=seed,
                rng=rng,
            )

            audio_paths = ex.get("audios")
            if not (isinstance(audio_paths, list) and len(audio_paths) == 1 and isinstance(audio_paths[0], str)):
                raise ValueError("Generation currently supports exactly 1 audio per example (audios=[path]).")

            audio_arr = _load_audio_array(audio_paths[0], target_sr=target_sr)
            inputs = processor(text=[prompt_text], audio=[audio_arr], return_tensors="pt", padding=True)
            for k_in, v_in in list(inputs.items()):
                if torch.is_tensor(v_in):
                    inputs[k_in] = v_in.to(device)

            prompt_len = int(inputs["input_ids"].shape[1])
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
            completion_ids = gen_ids[0, prompt_len:]
            pred = processor.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            ref = _extract_reference_text(ex.get("messages") or [])
            row = {
                "idx": int(idx),
                "audio": audio_paths[0],
                "prompt": prompt_text,
                "reference": ref,
                "prediction": pred,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if printed < int(print_first):
                print(f"[gen] idx={int(idx)} pred={_short(pred)} | ref={_short(ref)}")
                printed += 1

    if model_was_training:
        model.train()


@dataclass
class DynamicPromptAudioCollator:
    processor: Any
    seed: int
    dynamic_prompt: bool = True
    dynamic_prompt_deterministic: bool = False
    dynamic_prompt_id_key: Optional[str] = None
    dataset_audio_placeholder: str = "<audio>"

    _rng: Optional[random.Random] = None
    _rng_seeded: bool = False

    def _get_rng(self) -> random.Random:
        if self._rng is None:
            self._rng = random.Random()
        if not self._rng_seeded:
            # Seed per worker/rank to avoid identical sampling streams.
            worker_id = 0
            rank = 0
            worker_seed = 0
            try:
                import torch
                from torch.utils.data import get_worker_info

                wi = get_worker_info()
                worker_id = wi.id if wi else 0
                worker_seed = int(torch.initial_seed())
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    try:
                        rank = int(torch.distributed.get_rank())
                    except Exception:
                        rank = 0
            except Exception:
                pass

            mixed = (int(self.seed) + worker_seed + rank * 1000 + worker_id) % (2**32)
            self._rng.seed(mixed)
            self._rng_seeded = True
        return self._rng

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        texts: list[str] = []
        prompt_texts: list[str] = []
        audios: list[list[float]] = []

        audio_placeholder = self.dataset_audio_placeholder
        audio_token = getattr(self.processor, "audio_token", None) or getattr(self.processor.tokenizer, "audio_token", None)
        if not isinstance(audio_token, str) or not audio_token:
            raise ValueError("Cannot determine Gemma3n audio_token from processor/tokenizer.")

        full_audio_sequence = getattr(self.processor, "full_audio_sequence", None)
        if not isinstance(full_audio_sequence, str) or not full_audio_sequence:
            raise ValueError("Cannot determine Gemma3n full_audio_sequence from processor.")

        target_sr = (
            getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            or getattr(self.processor, "audio_sampling_rate", None)
            or 16000
        )
        target_sr = int(target_sr)

        rng = self._get_rng()

        for ex in examples:
            messages = ex.get("messages")
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                raise ValueError("Each example must have `messages` as list[dict].")
            messages = copy.deepcopy(messages)

            if self.dynamic_prompt:
                pool = ex.get("prompt_pool")
                if isinstance(pool, list) and len(pool) > 0:
                    if self.dynamic_prompt_deterministic:
                        chosen = _choose_from_pool_deterministic(pool, ex, seed=self.seed, id_key=self.dynamic_prompt_id_key)
                    else:
                        chosen = _choose_from_pool(pool, rng)
                    if isinstance(chosen, dict):
                        suffix = str(chosen.get("text") or chosen.get("suffix") or "")
                        completion = chosen.get("completion")
                    else:
                        suffix = str(chosen) if chosen is not None else ""
                        completion = None
                    _append_suffix_to_last_user(messages, suffix=suffix)
                    _override_last_assistant(messages, completion=completion)

            # Build full text.
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ).strip()
            full_text = full_text.replace(audio_placeholder, audio_token)
            texts.append(full_text)

            # Build prompt-only text (for masking prompt tokens in loss).
            prompt_messages: list[dict[str, Any]] = []
            for m in messages:
                if m.get("role") == "assistant":
                    break
                prompt_messages.append(m)
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            ).strip()
            prompt_text = prompt_text.replace(audio_placeholder, audio_token)
            prompt_texts.append(prompt_text)

            audio_paths = ex.get("audios")
            if not (isinstance(audio_paths, list) and len(audio_paths) == 1 and isinstance(audio_paths[0], str)):
                raise ValueError("This script currently supports exactly 1 audio per example (audios=[path]).")
            audios.append(_load_audio_array(audio_paths[0], target_sr=target_sr))

        # Tokenize full inputs (includes audio feature extraction + token_type_ids).
        batch = self.processor(text=texts, audio=audios, return_tensors="pt", padding=True)

        # Compute prompt lengths WITHOUT re-running the audio feature extractor.
        # We replicate the processor's text-side expansion for the audio placeholder.
        prompt_expanded = [t.replace(audio_token, full_audio_sequence) for t in prompt_texts]
        prompt_enc = self.processor.tokenizer(prompt_expanded, return_tensors="pt", padding=True)
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1).tolist()

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Mask prompt tokens (train on assistant only).
        for i, pl in enumerate(prompt_lens):
            labels[i, : int(pl)] = -100

        # Mask padding + special multimodal tokens.
        tok = self.processor.tokenizer
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == int(pad_id)] = -100
        for attr in ("image_token_id", "audio_token_id", "boi_token_id", "eoi_token_id", "boa_token_id", "eoa_token_id"):
            if hasattr(tok, attr):
                labels[labels == int(getattr(tok, attr))] = -100

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--model_name", type=str, default="unsloth/gemma-3n-E4B-it")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--hf_token", type=str, default=None)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,post,linear_start,linear_end,embedding_projection",
        help="Comma-separated target module names for Unsloth LoRA injection.",
    )
    p.add_argument("--finetune_vision_layers", action="store_true", help="Enable vision/audio branch finetuning in Unsloth.")

    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--optim", type=str, default="adamw_8bit")
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)

    # Eval loop (Transformers v4.56 uses `eval_strategy` instead of `evaluation_strategy`)
    p.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=500)

    # Dynamic prompt
    p.add_argument("--dynamic_prompt", action="store_true", help="Enable prompt_pool sampling (dynamic prompt).")
    p.add_argument(
        "--dynamic_prompt_deterministic",
        action="store_true",
        help="Deterministic per-sample prompt_pool choice (stable across epochs).",
    )
    p.add_argument("--dynamic_prompt_id_key", type=str, default=None)
    p.add_argument("--dataset_audio_placeholder", type=str, default="<audio>")

    # Generative samples during eval / after training
    p.add_argument("--eval_generate_samples", type=int, default=0, help="Dump N generations from eval set (0 disables).")
    p.add_argument("--eval_generate_max_new_tokens", type=int, default=128)
    p.add_argument("--eval_generate_do_sample", action="store_true")
    p.add_argument("--eval_generate_temperature", type=float, default=1.0)
    p.add_argument("--eval_generate_top_p", type=float, default=0.95)
    p.add_argument("--eval_generate_top_k", type=int, default=64)
    p.add_argument(
        "--eval_generate_dynamic_prompt",
        action="store_true",
        help="Sample prompt_pool for generation prompts (default: off).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Import heavy deps lazily (keeps this file importable without torch).
    import numpy as np
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from transformers import TrainerCallback

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model, processor = FastModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,  # auto
        max_seq_length=int(args.max_seq_length),
        load_in_4bit=bool(args.load_in_4bit),
        full_finetuning=False,
        token=args.hf_token,
    )

    target_modules = [m.strip() for m in str(args.target_modules).split(",") if m.strip()]
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=bool(args.finetune_vision_layers),
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        random_state=int(args.seed),
        use_rslora=False,
        loftq_config=None,
        target_modules=target_modules,
    )

    train_dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_jsonl, split="train") if args.eval_jsonl else None

    if str(args.eval_strategy) != "no" and eval_dataset is None:
        raise ValueError("`--eval_strategy` requires `--eval_jsonl`.")

    collator = DynamicPromptAudioCollator(
        processor=processor,
        seed=int(args.seed),
        dynamic_prompt=bool(args.dynamic_prompt),
        dynamic_prompt_deterministic=bool(args.dynamic_prompt_deterministic),
        dynamic_prompt_id_key=args.dynamic_prompt_id_key,
        dataset_audio_placeholder=args.dataset_audio_placeholder,
    )

    class _GenerateSamplesCallback(TrainerCallback):
        def on_evaluate(self, args_tr, state, control, **kwargs):  # noqa: ANN001
            if eval_dataset is None:
                return control
            if int(args.eval_generate_samples) <= 0:
                return control
            out_path = os.path.join(args.output_dir, f"eval_generations_step{int(state.global_step)}.jsonl")
            _generate_and_dump_samples(
                model=kwargs.get("model", model),
                processor=processor,
                dataset=eval_dataset,
                output_path=out_path,
                num_samples=int(args.eval_generate_samples),
                max_new_tokens=int(args.eval_generate_max_new_tokens),
                do_sample=bool(args.eval_generate_do_sample),
                temperature=float(args.eval_generate_temperature),
                top_p=float(args.eval_generate_top_p),
                top_k=int(args.eval_generate_top_k),
                seed=int(args.seed) + int(state.global_step),
                dataset_audio_placeholder=args.dataset_audio_placeholder,
                dynamic_prompt=bool(args.eval_generate_dynamic_prompt),
                dynamic_prompt_deterministic=bool(args.dynamic_prompt_deterministic),
                dynamic_prompt_id_key=args.dynamic_prompt_id_key,
            )
            print(f"[eval-generate] wrote: {out_path}")
            return control

    callbacks = [_GenerateSamplesCallback()] if eval_dataset is not None and int(args.eval_generate_samples) > 0 else None

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,  # keep TRL in text-mode; collator handles audio features
        data_collator=collator,
        callbacks=callbacks,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            per_device_eval_batch_size=int(args.per_device_eval_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            num_train_epochs=float(args.num_train_epochs),
            learning_rate=float(args.learning_rate),
            warmup_ratio=float(args.warmup_ratio),
            weight_decay=float(args.weight_decay),
            lr_scheduler_type=str(args.lr_scheduler_type),
            logging_steps=int(args.logging_steps),
            eval_strategy=str(args.eval_strategy),
            eval_steps=int(args.eval_steps),
            save_strategy="steps",
            save_steps=int(args.save_steps),
            optim=str(args.optim),
            seed=int(args.seed),
            report_to=str(args.report_to),
            dataloader_num_workers=int(args.dataloader_num_workers),
            remove_unused_columns=False,
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )

    trainer.train()
    # If no periodic eval is configured, still allow dumping a few generations after training.
    if eval_dataset is not None and int(args.eval_generate_samples) > 0 and str(args.eval_strategy) == "no":
        out_path = os.path.join(args.output_dir, "eval_generations_final.jsonl")
        _generate_and_dump_samples(
            model=model,
            processor=processor,
            dataset=eval_dataset,
            output_path=out_path,
            num_samples=int(args.eval_generate_samples),
            max_new_tokens=int(args.eval_generate_max_new_tokens),
            do_sample=bool(args.eval_generate_do_sample),
            temperature=float(args.eval_generate_temperature),
            top_p=float(args.eval_generate_top_p),
            top_k=int(args.eval_generate_top_k),
            seed=int(args.seed),
            dataset_audio_placeholder=args.dataset_audio_placeholder,
            dynamic_prompt=bool(args.eval_generate_dynamic_prompt),
            dynamic_prompt_deterministic=bool(args.dynamic_prompt_deterministic),
            dynamic_prompt_id_key=args.dynamic_prompt_id_key,
        )
        print(f"[eval-generate] wrote: {out_path}")

    trainer.save_model(args.output_dir)
    try:
        processor.save_pretrained(args.output_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
