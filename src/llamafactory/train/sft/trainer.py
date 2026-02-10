# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import MutableMapping
import json
import os
import time
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    _AUDIO_PROGRESS_FILENAME = "audio_progress.json"

    @override
    def _load_rng_state(self, checkpoint: Optional[str]) -> None:
        """Load RNG states from checkpoint in a torch>=2.6-compatible way."""
        env_key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        previous = os.environ.get(env_key)
        os.environ[env_key] = "1"
        try:
            super()._load_rng_state(checkpoint)
        finally:
            if previous is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous

    @contextmanager
    def _temporary_funaudiochat_eval_audio_attn(self, model: torch.nn.Module, implementation: str = "sdpa") -> "Any":
        r"""Temporarily switch FunAudioChat audio encoder attention for eval/predict.

        FunAudioChat's audio encoder may run under fp32 during `generate()` (evaluation), which can
        crash FlashAttention-2. We keep training-time attention unchanged, and only switch the audio
        encoder to SDPA/eager during generation in eval/predict.
        """
        unwrapped_model = model.module if hasattr(model, "module") else model

        def _iter_funaudiochat_audio_encoder_configs(m: torch.nn.Module):
            visited: set[int] = set()
            stack: list[Any] = [m]
            while stack:
                cur = stack.pop()
                if cur is None:
                    continue
                cur_id = id(cur)
                if cur_id in visited:
                    continue
                visited.add(cur_id)

                cfg = getattr(cur, "config", None)
                if cfg is not None:
                    model_type = getattr(cfg, "model_type", None)
                    if model_type == "funaudiochat" and getattr(cfg, "audio_config", None) is not None:
                        yield getattr(cfg, "audio_config")
                    elif model_type == "funaudiochat_audio_encoder":
                        yield cfg

                # Common wrappers (DDP/Deepspeed/PEFT) and known FunAudioChat submodules.
                for attr in (
                    "module",
                    "base_model",
                    "model",
                    "continuous_audio_tower",
                    "audio_tower",
                    "audio_invert_tower",
                ):
                    nxt = getattr(cur, attr, None)
                    if nxt is not None:
                        stack.append(nxt)

        patched: list[tuple[Any, Optional[str]]] = []
        seen_cfg: set[int] = set()
        for audio_cfg in _iter_funaudiochat_audio_encoder_configs(unwrapped_model):
            if audio_cfg is None:
                continue
            cfg_id = id(audio_cfg)
            if cfg_id in seen_cfg:
                continue
            seen_cfg.add(cfg_id)

            original_impl = getattr(audio_cfg, "_attn_implementation", None)
            if original_impl not in ("flash_attention_2", "flash_attention_3"):
                continue

            setattr(audio_cfg, "_attn_implementation", implementation)
            patched.append((audio_cfg, original_impl))

        if not patched:
            yield
            return

        try:
            yield
        finally:
            for audio_cfg, original_impl in patched:
                setattr(audio_cfg, "_attn_implementation", original_impl)

    @contextmanager
    def _temporary_generate_autocast(self) -> "Any":
        r"""Enable CUDA autocast during `generate()` in eval/predict.

        Transformers' `Seq2SeqTrainer.prediction_step()` calls `model.generate()` without autocast. For mixed-dtype
        models (e.g., LoRA + fp32 trainable modules_to_save), this can crash with dtype mismatches in linear layers.
        """
        device_type = getattr(getattr(self, "args", None), "device", None)
        device_type = getattr(device_type, "type", None) or "cuda"
        use_fp16 = bool(getattr(self.args, "fp16", False))
        use_bf16 = bool(getattr(self.args, "bf16", False))
        if device_type == "cuda" and (use_fp16 or use_bf16):
            dtype = torch.float16 if use_fp16 else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=dtype):
                yield
        else:
            yield

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        eval_data_collator: Optional[Any] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        audio_total_duration_sec: float | None = None,
        audio_progress_enabled: bool = False,
        audio_total_duration_ready: bool | None = None,
        audio_duration_cache_path: str | None = None,
        audio_duration_expected_files: dict[str, tuple[int, int]] | None = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if bool(getattr(training_args, "fp8", False)):
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        self.eval_data_collator = eval_data_collator
        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self._default_gen_kwargs: dict[str, Any] = gen_kwargs.copy() if gen_kwargs is not None else {}
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        self._audio_total_duration_sec = float(audio_total_duration_sec or 0.0)
        self._audio_consumed_duration_sec = 0.0
        self._audio_duration_cache_path = str(audio_duration_cache_path) if audio_duration_cache_path else None
        self._audio_duration_expected_files = audio_duration_expected_files or {}
        self._audio_total_duration_ready = (
            bool(audio_total_duration_ready)
            if audio_total_duration_ready is not None
            else bool(self._audio_total_duration_sec > 0)
        )
        self._audio_progress_enabled = bool(audio_progress_enabled) or bool(self._audio_total_duration_sec > 0)
        self._audio_cache_last_mtime = 0.0
        self._audio_cache_last_check_time = 0.0
        if self._audio_progress_enabled:
            self._audio_consumed_duration_sec = float(self._try_load_audio_progress_sec() or 0.0)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        use_dft_loss = bool(getattr(finetuning_args, "use_dft_loss", False))
        use_chunked_ce_loss = bool(getattr(finetuning_args, "use_chunked_ce_loss", False))
        use_eaft_loss = bool(getattr(finetuning_args, "use_eaft_loss", False))

        if use_dft_loss and use_chunked_ce_loss:
            raise ValueError("`use_dft_loss` and `use_chunked_ce_loss` are mutually exclusive.")
        if use_dft_loss and use_eaft_loss:
            raise ValueError("`use_dft_loss` and `use_eaft_loss` are mutually exclusive.")
        if use_chunked_ce_loss and use_eaft_loss:
            raise ValueError("`use_chunked_ce_loss` and `use_eaft_loss` are mutually exclusive.")

        if use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func
        elif use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, float(getattr(finetuning_args, "eaft_alpha", 1.0))
            )
        elif use_chunked_ce_loss:
            from ..trainer_utils import chunked_ce_loss_func

            if self.label_smoother is not None:
                logger.warning_rank0("Label smoothing will be ignored when `use_chunked_ce_loss=True`.")

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            shift_labels = not bool(getattr(getattr(unwrapped_model, "config", None), "is_encoder_decoder", False))
            self.compute_loss_func = partial(
                chunked_ce_loss_func,
                num_output_chunks=int(getattr(finetuning_args, "chunked_ce_num_chunks", 8)),
                upcast_logits=bool(getattr(finetuning_args, "chunked_ce_upcast_logits", True)),
                shift_labels=shift_labels,
            )

        if bool(getattr(training_args, "fp8", False)) and hasattr(self, "accelerator"):  # verify FP8 status
            verify_fp8_status(self.accelerator, training_args)

    def _get_resume_checkpoint_path(self) -> str | None:
        resume = getattr(self.args, "resume_from_checkpoint", None)
        if isinstance(resume, str) and resume and os.path.isdir(resume):
            return resume
        try:
            last = get_last_checkpoint(self.args.output_dir)
        except Exception:
            last = None
        if isinstance(last, str) and last and os.path.isdir(last):
            return last
        return None

    def _try_load_audio_progress_sec(self) -> float | None:
        ckpt = self._get_resume_checkpoint_path()
        if ckpt is None:
            return None
        path = os.path.join(ckpt, self._AUDIO_PROGRESS_FILENAME)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            return None
        v = obj.get("consumed_audio_duration_sec")
        try:
            return float(v)
        except Exception:
            return None

    def _pop_audio_duration_sec_from_inputs(self, inputs: Any) -> Any:
        """Pop audio_duration_sec from inputs (top-level or nested `data` dict)."""
        if not isinstance(inputs, MutableMapping):
            return None
        if "audio_duration_sec" in inputs:
            return inputs.pop("audio_duration_sec", None)
        data = inputs.get("data")
        if isinstance(data, MutableMapping) and "audio_duration_sec" in data:
            return data.pop("audio_duration_sec", None)
        return None

    @override
    def training_step(
        self, model: torch.nn.Module, inputs: dict[str, Any], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        audio_dur = self._pop_audio_duration_sec_from_inputs(inputs)
        if self._audio_progress_enabled and audio_dur is not None:
            t = None
            try:
                if torch.is_tensor(audio_dur):
                    ad = audio_dur.detach()
                    if ad.device.type == "cpu":
                        sec_local = float(ad.sum().item())
                        t = torch.tensor(sec_local, device=self.args.device, dtype=torch.float32)
                    else:
                        t = ad.to(dtype=torch.float32).sum()
                        if t.device != self.args.device:
                            t = t.to(device=self.args.device)
                elif isinstance(audio_dur, (list, tuple)):
                    s = 0.0
                    for x in audio_dur:
                        if x is None:
                            continue
                        try:
                            s += float(x)
                        except Exception:
                            continue
                    t = torch.tensor(float(s), device=self.args.device, dtype=torch.float32)
                else:
                    t = torch.tensor(float(audio_dur), device=self.args.device, dtype=torch.float32)
            except Exception:
                t = None

            if t is not None:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                sec = float(t.item())
                if sec > 0:
                    self._audio_consumed_duration_sec += sec

        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def _maybe_refresh_audio_total_duration_sec(self) -> None:
        if not self._audio_progress_enabled or self._audio_total_duration_ready:
            return
        if not self._audio_duration_cache_path or not self._audio_duration_expected_files:
            return
        now = time.time()
        if now - float(self._audio_cache_last_check_time or 0.0) < 60.0:
            return
        self._audio_cache_last_check_time = now

        try:
            st = os.stat(self._audio_duration_cache_path)
            cache_mtime = float(st.st_mtime)
        except OSError:
            return
        if cache_mtime <= float(self._audio_cache_last_mtime or 0.0):
            return

        try:
            with open(self._audio_duration_cache_path, encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:  # noqa: BLE001
            return
        if not isinstance(obj, dict):
            return

        total = obj.get("total_duration_sec")
        try:
            total_sec = float(total) if total is not None else 0.0
        except Exception:
            total_sec = 0.0
        if not (total_sec > 0):
            return

        files = obj.get("files")
        if not isinstance(files, dict):
            return
        for path, (size, mtime) in self._audio_duration_expected_files.items():
            entry = files.get(path)
            if not isinstance(entry, dict):
                return
            try:
                if int(entry.get("size", -1)) != int(size) or int(entry.get("mtime", -1)) != int(mtime):
                    return
            except Exception:
                return
            try:
                dur = float(entry.get("duration_sec") or 0.0)
                has_audio = entry.get("has_audio")
                has_audio = bool(has_audio) if has_audio is not None else True
                if int(size) > 0 and has_audio and dur <= 0:
                    return
            except Exception:
                return

        self._audio_total_duration_sec = float(total_sec)
        self._audio_total_duration_ready = True
        self._audio_cache_last_mtime = cache_mtime
        logger.info_rank0("Audio duration cache ready: total_hours=%.2f", float(total_sec) / 3600.0)

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if self._audio_progress_enabled:
            self._maybe_refresh_audio_total_duration_sec()
            logs = dict(logs)
            logs["audio_hours"] = float(self._audio_consumed_duration_sec / 3600.0)
            if self._audio_total_duration_sec > 0:
                logs["audio_total_hours"] = float(self._audio_total_duration_sec / 3600.0)
            logs["audio_total_ready"] = bool(self._audio_total_duration_ready)
            if self._audio_total_duration_ready and self._audio_total_duration_sec > 0:
                logs["audio_epoch"] = float(self._audio_consumed_duration_sec / self._audio_total_duration_sec)
        return super().log(logs, start_time=start_time)

    def _write_audio_progress(self, checkpoint_dir: str | None = None) -> None:
        if not self._audio_progress_enabled:
            return

        payload = {
            "global_step": int(getattr(self.state, "global_step", 0) or 0),
            "consumed_audio_duration_sec": float(self._audio_consumed_duration_sec),
            "total_audio_duration_sec": float(self._audio_total_duration_sec),
            "audio_total_ready": bool(self._audio_total_duration_ready),
            "audio_epoch": float(self._audio_consumed_duration_sec / self._audio_total_duration_sec)
            if self._audio_total_duration_sec > 0
            else 0.0,
            "audio_hours": float(self._audio_consumed_duration_sec / 3600.0),
            "audio_total_hours": float(self._audio_total_duration_sec / 3600.0),
        }

        def _dump(path: str) -> None:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
                    f.write("\n")
            except Exception as err:  # noqa: BLE001
                logger.warning_rank0("Failed to write audio progress to %s: %s", path, err)

        if checkpoint_dir is not None:
            _dump(os.path.join(checkpoint_dir, self._AUDIO_PROGRESS_FILENAME))
        _dump(os.path.join(self.args.output_dir, "audio_progress_latest.json"))

    @override
    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        try:
            if metrics is None:
                super()._save_checkpoint(model, trial)
            else:
                super()._save_checkpoint(model, trial, metrics)
        except TypeError:
            super()._save_checkpoint(model, trial)
        if self.args.should_save:
            checkpoint_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self._write_audio_progress(checkpoint_dir=checkpoint_dir)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        effective_optimizer = optimizer or self.optimizer
        custom_scheduler = create_custom_scheduler(self.args, num_training_steps, effective_optimizer)
        if custom_scheduler is not None:
            self.lr_scheduler = custom_scheduler
            return custom_scheduler
        return super().create_scheduler(num_training_steps, effective_optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_eval_dataloader(self, eval_dataset: Optional["Dataset"] = None):
        if self.eval_data_collator is None or self.eval_data_collator is self.data_collator:
            return super().get_eval_dataloader(eval_dataset)

        original_data_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_data_collator

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        self._update_effective_tokens_seen(model, inputs)

        # Remove auxiliary metadata keys that are not accepted by `model.forward()`.
        # For FunAudioChat, `audio_duration_sec` is used for progress logging but should never be passed to the model.
        if isinstance(inputs, MutableMapping):
            copied = False
            if "audio_duration_sec" in inputs:
                inputs = dict(inputs)
                inputs.pop("audio_duration_sec", None)
                copied = True
            data = inputs.get("data")
            if isinstance(data, MutableMapping) and "audio_duration_sec" in data:
                if not copied:
                    inputs = dict(inputs)
                data = dict(data)
                data.pop("audio_duration_sec", None)
                inputs["data"] = data

        return_outputs = kwargs.get("return_outputs", False)
        num_items_in_batch = kwargs.get("num_items_in_batch", None)
        if "return_outputs" not in kwargs and len(args) > 0:
            return_outputs = args[0]
        if "num_items_in_batch" not in kwargs and len(args) > 1:
            num_items_in_batch = args[1]

        # HF Trainer pops labels when label smoothing is enabled, then calls `LabelSmoother(outputs, labels)`.
        # This breaks models that need labels inside `forward()` to build auxiliary targets (e.g. FunAudioChat
        # needs labels to build speech labels / speech_loss), and it also breaks any ModelOutput that doesn't
        # expose a dict key named `logits` (FunAudioChat uses `text_logits`).
        #
        # We keep labels in the forward pass and compute label smoothing on the returned logits explicitly.
        if self.label_smoother is None and self.compute_loss_func is None:
            return super().compute_loss(model, inputs, *args, **kwargs)

        labels = inputs.get("labels") if "labels" in inputs else None
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            inputs = {**inputs, "num_items_in_batch": num_items_in_batch}

        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        else:
            # Label smoothing for decoder-only models: shift labels by 1 token.
            unwrapped_model = self.accelerator.unwrap_model(model)
            shift_labels = not bool(getattr(getattr(unwrapped_model, "config", None), "is_encoder_decoder", False))

            # Locate logits for LabelSmoother. Prefer standard `logits`, fall back to `text_logits` (FunAudioChat),
            # then to attribute `.logits`, and finally to the first tuple element for legacy tuple outputs.
            logits = None
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
                if logits is None:
                    logits = outputs.get("text_logits")
            if logits is None and hasattr(outputs, "logits"):
                logits = getattr(outputs, "logits")
            if logits is None and hasattr(outputs, "text_logits"):
                logits = getattr(outputs, "text_logits")
            if logits is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                logits = outputs[0]

            if logits is None:
                raise ValueError(
                    "Cannot locate logits for label smoothing. Expected `logits`/`text_logits` in model outputs."
                )

            loss = self.label_smoother({"logits": logits}, labels, shift_labels=shift_labels)

            # Add any auxiliary losses returned by the model outputs, excluding the main `loss`.
            aux_loss = None
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    if k == "loss":
                        continue
                    if not k.endswith("_loss"):
                        continue
                    if v is None or not torch.is_tensor(v):
                        continue

                    v_t = v.mean() if v.dim() > 0 else v
                    v_t = v_t.to(loss.device)
                    aux_loss = v_t if aux_loss is None else (aux_loss + v_t)
            else:
                # Best-effort fallback for non-dict outputs
                v = getattr(outputs, "speech_loss", None)
                if v is not None and torch.is_tensor(v):
                    aux_loss = v.mean() if v.dim() > 0 else v

            if aux_loss is not None:
                loss = loss + aux_loss

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def _update_effective_tokens_seen(self, model, inputs) -> None:
        if not getattr(model, "training", False):
            return

        labels = inputs.get("labels")
        if not torch.is_tensor(labels):
            return

        ignore_index = getattr(getattr(model, "config", None), "ignore_index", IGNORE_INDEX)
        effective_mask = labels.ne(int(ignore_index))

        audio_token_index = getattr(getattr(model, "config", None), "audio_token_index", None)
        if audio_token_index is not None:
            try:
                effective_mask = effective_mask & labels.ne(int(audio_token_index))
            except Exception:
                pass

        effective_tokens = effective_mask.sum()
        effective_tokens = effective_tokens.to(device=self.args.device, dtype=torch.int64).detach()
        effective_tokens = self.accelerator.gather(effective_tokens).sum()

        prev = int(getattr(self.state, "num_effective_tokens_seen", 0) or 0)
        setattr(self.state, "num_effective_tokens_seen", prev + int(effective_tokens.item()))

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        # NOTE:
        # - When `predict_with_generate=True`, we want to compute generative metrics (WER/CER),
        #   but still keep `eval_loss` for best-checkpoint selection.
        # - Passing `labels` into `model.generate()` is wasteful and can break some models.
        #   Therefore we compute loss in a separate loss-only forward pass.

        # Strip auxiliary metadata that should not be fed into forward/generate.
        if isinstance(inputs, MutableMapping):
            copied = False
            if "audio_duration_sec" in inputs:
                inputs = dict(inputs)
                inputs.pop("audio_duration_sec", None)
                copied = True
            data = inputs.get("data")
            if isinstance(data, MutableMapping) and "audio_duration_sec" in data:
                if not copied:
                    inputs = dict(inputs)
                data = dict(data)
                data.pop("audio_duration_sec", None)
                inputs["data"] = data

        labels = inputs.get("labels")

        def _prepare_loss_inputs(
            batch_inputs: dict[str, Union["torch.Tensor", Any]],
        ) -> dict[str, Union["torch.Tensor", Any]]:
            r"""Convert prompt-only + target-only batches to CLM-style batches for loss computation.

            In generation-eval mode, LLaMA-Factory preprocesses eval data as:
              - input_ids: prompt (plus modality tokens)
              - labels: reference text only
            For decoder-only models (e.g., Gemma3n), loss computation expects labels aligned with input_ids.
            """
            if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
                return batch_inputs  # keep seq2seq behavior

            if "input_ids" not in batch_inputs or "labels" not in batch_inputs:
                return batch_inputs

            input_ids = batch_inputs.get("input_ids")
            label_ids = batch_inputs.get("labels")
            attention_mask = batch_inputs.get("attention_mask")
            if not (
                torch.is_tensor(input_ids)
                and torch.is_tensor(label_ids)
                and torch.is_tensor(attention_mask)
                and input_ids.dim() == 2
                and label_ids.dim() == 2
                and attention_mask.dim() == 2
            ):
                return batch_inputs

            if input_ids.size(1) == label_ids.size(1):
                return batch_inputs

            # Only handle the common prompt-only+target-only layout:
            # attention_mask must match input_ids length.
            if attention_mask.size(1) != input_ids.size(1):
                return batch_inputs

            pad_token_id = getattr(self.processing_class, "pad_token_id", None)
            if pad_token_id is None:
                return batch_inputs

            # Treat both IGNORE_INDEX and pad_token_id as padding on the target side.
            tgt_is_valid = label_ids.ne(IGNORE_INDEX) & label_ids.ne(pad_token_id)
            tgt_input_ids = torch.where(tgt_is_valid, label_ids, label_ids.new_full(label_ids.shape, pad_token_id))
            tgt_attention_mask = tgt_is_valid.to(dtype=attention_mask.dtype)
            tgt_labels = torch.where(tgt_is_valid, label_ids, label_ids.new_full(label_ids.shape, IGNORE_INDEX))

            prompt_ignore = label_ids.new_full(input_ids.shape, IGNORE_INDEX)
            merged = dict(batch_inputs)
            merged["input_ids"] = torch.cat([input_ids, tgt_input_ids], dim=1)
            merged["attention_mask"] = torch.cat([attention_mask, tgt_attention_mask], dim=1)
            merged["labels"] = torch.cat([prompt_ignore, tgt_labels], dim=1)

            # If a standard 2D position_ids is present, rebuild it for the merged sequence.
            if (
                "position_ids" in merged
                and torch.is_tensor(merged["position_ids"])
                and merged["position_ids"].dim() == 2
            ):
                pos = torch.cumsum(merged["attention_mask"].long(), dim=1) - 1
                merged["position_ids"] = pos.masked_fill(merged["attention_mask"] == 0, 0)

            # Extend token_type_ids if present.
            if (
                "token_type_ids" in merged
                and torch.is_tensor(merged["token_type_ids"])
                and merged["token_type_ids"].dim() == 2
                and merged["token_type_ids"].size(1) == input_ids.size(1)
            ):
                zeros = merged["token_type_ids"].new_zeros((merged["token_type_ids"].size(0), tgt_input_ids.size(1)))
                merged["token_type_ids"] = torch.cat([merged["token_type_ids"], zeros], dim=1)

            return merged

        if self.args.predict_with_generate and not prediction_loss_only:
            if getattr(self, "_skip_generate_loss", False):
                # Generation-only pass (skip loss) for two-stage eval (Phase 2).
                gen_inputs = dict(inputs)
                gen_inputs.pop("labels", None)
                loss = None
                with self._temporary_generate_autocast(), self.compute_loss_context_manager():
                    _, generated_tokens, _ = super().prediction_step(
                        model,
                        gen_inputs,
                        prediction_loss_only=False,
                        ignore_keys=ignore_keys,
                        **gen_kwargs,
                    )
            else:
                # 1) Loss-only pass (keeps labels) to preserve `{metric_key_prefix}_loss`.
                loss_inputs = _prepare_loss_inputs(dict(inputs))
                loss, _, _ = super().prediction_step(
                    model,
                    loss_inputs,
                    prediction_loss_only=True,
                    ignore_keys=ignore_keys,
                )

                # 2) Generation pass (remove labels to avoid loss computation during generation).
                gen_inputs = dict(inputs)
                gen_inputs.pop("labels", None)
                with self._temporary_funaudiochat_eval_audio_attn(model):
                    with self._temporary_generate_autocast(), self.compute_loss_context_manager():
                        _, generated_tokens, _ = super().prediction_step(
                            model,
                            gen_inputs,
                            prediction_loss_only=False,
                            ignore_keys=ignore_keys,
                            **gen_kwargs,
                        )
        else:
            # Default behavior (no generation, or loss-only evaluation).
            # Keep labels when `prediction_loss_only=True` so eval_loss is available.
            step_inputs = dict(inputs)
            if prediction_loss_only:
                step_inputs = _prepare_loss_inputs(step_inputs)
            if self.args.predict_with_generate and not prediction_loss_only:
                step_inputs.pop("labels", None)
            if self.args.predict_with_generate and not prediction_loss_only:
                with self._temporary_funaudiochat_eval_audio_attn(model):
                    with self._temporary_generate_autocast(), self.compute_loss_context_manager():
                        loss, generated_tokens, _ = super().prediction_step(
                            model,
                            step_inputs,
                            prediction_loss_only=prediction_loss_only,
                            ignore_keys=ignore_keys,
                            **gen_kwargs,
                        )
            else:
                loss, generated_tokens, _ = super().prediction_step(
                    model,
                    step_inputs,
                    prediction_loss_only=prediction_loss_only,
                    ignore_keys=ignore_keys,
                    **gen_kwargs,
                )

        if generated_tokens is not None and self.args.predict_with_generate:
            # Remove prompt part in the generated tokens.
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    @override
    def predict(
        self,
        test_dataset: "Dataset",
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> "PredictionOutput":
        # Ensure generation kwargs provided at init are used by `Seq2SeqTrainer.predict()`.
        # Otherwise Transformers will overwrite `self._gen_kwargs` with an empty dict.
        if self._default_gen_kwargs:
            merged = dict(self._default_gen_kwargs)
            merged.update(gen_kwargs)
            gen_kwargs = merged
        has_processing_class = hasattr(self, "processing_class")
        original_padding_side = self.processing_class.padding_side if has_processing_class else None

        def _set_left_padding() -> None:
            if has_processing_class and self.args.predict_with_generate:
                self.processing_class.padding_side = "left"

        def _restore_padding() -> None:
            if has_processing_class and original_padding_side is not None:
                self.processing_class.padding_side = original_padding_side

        try:
            _set_left_padding()
            return super().predict(
                test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **gen_kwargs
            )
        finally:
            _restore_padding()

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    @override
    def evaluate(
        self,
        eval_dataset: Optional["Dataset"] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> dict[str, float]:
        r"""Overridden to support eval sampling when predict_with_generate and restore left padding for eval.

        When `predict_with_generate=True` and `eval_num_samples` is set, this method can optionally compute:
        - `{metric_key_prefix}_loss` on the full eval dataset, and
        - generative metrics (WER/CER/ROUGE/BLEU) on a sampled subset.
        """
        # Ensure generation kwargs provided at init are used by `Seq2SeqTrainer.evaluate()`.
        # Otherwise Transformers will overwrite `self._gen_kwargs` with an empty dict and fall back to
        # `model.generation_config` (e.g. max_new_tokens=2048), making CLI overrides ineffective.
        if self._default_gen_kwargs:
            merged = dict(self._default_gen_kwargs)
            merged.update(gen_kwargs)
            gen_kwargs = merged

        has_processing_class = hasattr(self, "processing_class")
        original_padding_side = self.processing_class.padding_side if has_processing_class else None

        def _set_left_padding() -> None:
            if has_processing_class:
                self.processing_class.padding_side = "left"

        def _restore_padding() -> None:
            if has_processing_class and original_padding_side is not None:
                self.processing_class.padding_side = original_padding_side

        try:
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if eval_dataset is None:
                return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

            if (
                self.args.predict_with_generate
                and self.finetuning_args.eval_num_samples is not None
                and getattr(self.finetuning_args, "eval_loss_on_full_dataset", True)
                and self.finetuning_args.eval_num_samples < len(eval_dataset)
            ):
                # Phase 1: compute full-dataset eval loss only.
                # Disable generation + compute_metrics to avoid duplicating the expensive generation pass.
                original_compute_metrics = self.compute_metrics
                original_preprocess_logits_for_metrics = getattr(self, "preprocess_logits_for_metrics", None)
                original_predict_with_generate = self.args.predict_with_generate
                self.compute_metrics = None
                if hasattr(self, "preprocess_logits_for_metrics"):
                    self.preprocess_logits_for_metrics = None
                try:
                    self.args.predict_with_generate = False
                    loss_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
                finally:
                    self.args.predict_with_generate = original_predict_with_generate
                    self.compute_metrics = original_compute_metrics
                    if hasattr(self, "preprocess_logits_for_metrics"):
                        self.preprocess_logits_for_metrics = original_preprocess_logits_for_metrics

                # Phase 2: compute generative metrics on a sampled subset (with generation).
                rng = np.random.default_rng(self.args.seed)
                sampled_dataset = eval_dataset.select(
                    rng.choice(len(eval_dataset), self.finetuning_args.eval_num_samples, replace=False)
                )
                logger.info_rank0(
                    f"Evaluation: `{metric_key_prefix}_loss` computed on full dataset (n={len(eval_dataset)}); "
                    f"generative metrics computed on subset (n={len(sampled_dataset)})."
                )
                had_skip_generate_loss = hasattr(self, "_skip_generate_loss")
                original_skip_generate_loss = getattr(self, "_skip_generate_loss", False)
                self._skip_generate_loss = True
                _set_left_padding()
                try:
                    gen_metrics = super().evaluate(sampled_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
                finally:
                    _restore_padding()
                    if had_skip_generate_loss:
                        self._skip_generate_loss = original_skip_generate_loss
                    else:
                        delattr(self, "_skip_generate_loss")

                # Merge: keep full loss, keep generative metrics from subset.
                merged = dict(gen_metrics)
                loss_key = f"{metric_key_prefix}_loss"
                if loss_key in loss_metrics:
                    merged[loss_key] = loss_metrics[loss_key]

                # Expose full-eval runtime stats to avoid confusion (Phase 2 runs on subset).
                for suffix in ("samples", "steps", "runtime", "samples_per_second", "steps_per_second"):
                    k = f"{metric_key_prefix}_{suffix}"
                    if k in loss_metrics:
                        merged[f"{k}_full"] = loss_metrics[k]
                return merged

            # Default behavior: either evaluate on full set, or sample everything together.
            if (
                self.args.predict_with_generate
                and self.finetuning_args.eval_num_samples is not None
                and eval_dataset is not None
                and self.finetuning_args.eval_num_samples < len(eval_dataset)
            ):
                rng = np.random.default_rng(self.args.seed)
                eval_dataset = eval_dataset.select(
                    rng.choice(len(eval_dataset), self.finetuning_args.eval_num_samples, replace=False)
                )

            if self.args.predict_with_generate:
                _set_left_padding()
                try:
                    return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
                finally:
                    _restore_padding()

            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        finally:
            _restore_padding()
