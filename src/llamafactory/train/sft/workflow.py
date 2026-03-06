# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

import os
from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...data.audio_progress import (
    get_audio_duration_file_fingerprint,
    get_audio_duration_files,
    get_cached_total_audio_duration_sec,
    is_audio_duration_cache_complete,
    maybe_launch_audio_duration_scan,
)
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .metric import ComputeAccuracy, ComputeEndpointingMetrics, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    ref_model = None
    if finetuning_args.use_asft_loss:
        ref_model = create_ref_model(model_args, finetuning_args)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    collator_kwargs = dict(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    train_data_collator = SFTDataCollatorWith4DAttentionMask(
        pad_to_multiple_of=8,
        audio_specaugment=model_args.audio_specaugment,
        audio_specaugment_mask_param=model_args.audio_specaugment_mask_param,
        audio_specaugment_num_masks=model_args.audio_specaugment_num_masks,
        audio_specaugment_fill_value=model_args.audio_specaugment_fill_value,
        **collator_kwargs,
    )  # for shift short attention
    eval_data_collator = SFTDataCollatorWith4DAttentionMask(
        pad_to_multiple_of=None,
        audio_specaugment=False,
        **collator_kwargs,
    )

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_endpointing_metrics:
            raise NotImplementedError("`compute_endpointing_metrics` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(
            tokenizer=tokenizer, compute_wer_cer=finetuning_args.compute_wer_cer
        )
    elif finetuning_args.compute_endpointing_metrics:
        metric_module["compute_metrics"] = ComputeEndpointingMetrics(tokenizer=tokenizer)
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    eos_token_ids: list[int] = []
    eos_attr = getattr(tokenizer, "eos_token_id", None)
    if eos_attr is not None:
        if isinstance(eos_attr, (list, tuple)):
            eos_token_ids.extend(int(stop_id) for stop_id in eos_attr if stop_id is not None)
        else:
            eos_token_ids.append(int(eos_attr))

    # Compatible with Transformers v4 and Transformers v5: include additional special tokens as EOS stops.
    extra_ids: list[int] = []
    if is_transformers_version_greater_than("4.58.0"):
        raw_extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if isinstance(raw_extra_ids, (list, tuple)):
            extra_ids = [int(i) for i in raw_extra_ids if i is not None]
        else:
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            converted = tokenizer.convert_tokens_to_ids(string_tokens)
            if isinstance(converted, (list, tuple)):
                extra_ids = [int(i) for i in converted if i is not None]
    else:
        raw_extra_ids = getattr(tokenizer, "additional_special_tokens_ids", [])
        if isinstance(raw_extra_ids, (list, tuple)):
            extra_ids = [int(i) for i in raw_extra_ids if i is not None]

    eos_token_ids.extend(i for i in extra_ids if i != -1)
    eos_token_ids = list(dict.fromkeys(eos_token_ids))
    if not eos_token_ids:
        raise ValueError("Cannot determine `eos_token_id` from tokenizer.")
    gen_kwargs["eos_token_id"] = eos_token_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs.setdefault("min_new_tokens", 1)
    gen_kwargs.setdefault("max_new_tokens", 256)
    gen_kwargs.setdefault("do_sample", False)

    # Initialize our Trainer
    if model_args.use_kt:
        from ktransformers.sft.lora import KTrainer  # type: ignore
        from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore

        GLOBAL_CONFIG._config["mod"] = "sft"

        data_collator = train_data_collator if training_args.do_train else eval_data_collator
        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **metric_module,
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False

    else:
        audio_total_duration_sec: float | None = None
        audio_total_duration_ready = False
        audio_duration_cache_path = None
        audio_duration_expected_files: dict[str, tuple[int, int]] | None = None
        if getattr(data_args, "log_audio_epochs", False) and training_args.do_train:
            dataset_names = data_args.dataset
            if isinstance(dataset_names, str):
                dataset_names = [s.strip() for s in dataset_names.split(",") if s.strip()]
            elif not isinstance(dataset_names, list):
                dataset_names = []

            audio_duration_cache_path = os.path.join(training_args.output_dir, "audio_duration_cache.json")
            dataset_dir = str(data_args.dataset_dir)
            dataset_names = [str(x) for x in dataset_names]
            expected_paths = get_audio_duration_files(dataset_dir=dataset_dir, dataset_names=dataset_names)
            audio_duration_expected_files = get_audio_duration_file_fingerprint(expected_paths)

            cached_total = (
                get_cached_total_audio_duration_sec(audio_duration_cache_path) if audio_duration_cache_path else None
            )
            if cached_total is not None:
                audio_total_duration_sec = float(cached_total)

            cache_complete = (
                is_audio_duration_cache_complete(
                    cache_path=audio_duration_cache_path,
                    dataset_dir=dataset_dir,
                    dataset_names=dataset_names,
                    expected_files=audio_duration_expected_files,
                )
                if audio_duration_cache_path and audio_duration_expected_files
                else False
            )
            audio_total_duration_ready = bool(cache_complete and (audio_total_duration_sec or 0.0) > 0)

            is_rank0 = bool(getattr(training_args, "process_index", 0) == 0)
            if is_rank0:
                if not audio_total_duration_ready and audio_duration_cache_path and audio_duration_expected_files:
                    max_mb_per_sec = None
                    try:
                        v = os.getenv("AUDIO_DURATION_SCAN_MAX_MBPS", "").strip()
                        if v:
                            max_mb_per_sec = float(v)
                    except Exception:
                        max_mb_per_sec = None

                    maybe_launch_audio_duration_scan(
                        dataset_dir=dataset_dir,
                        dataset_names=dataset_names,
                        cache_path=audio_duration_cache_path,
                        max_mb_per_sec=max_mb_per_sec,
                    )

        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=train_data_collator if training_args.do_train else eval_data_collator,
            eval_data_collator=eval_data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            audio_total_duration_sec=audio_total_duration_sec,
            audio_progress_enabled=bool(getattr(data_args, "log_audio_epochs", False) and training_args.do_train),
            audio_total_duration_ready=audio_total_duration_ready,
            audio_duration_cache_path=audio_duration_cache_path,
            audio_duration_expected_files=audio_duration_expected_files,
            ref_model=ref_model,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
