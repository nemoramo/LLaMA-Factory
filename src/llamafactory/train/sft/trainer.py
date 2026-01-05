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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        eval_data_collator: Optional[Any] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

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

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

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
        return super().compute_loss(model, inputs, *args, **kwargs)

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
            if "position_ids" in merged and torch.is_tensor(merged["position_ids"]) and merged["position_ids"].dim() == 2:
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
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **gen_kwargs)

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
