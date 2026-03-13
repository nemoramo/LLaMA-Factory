from __future__ import annotations

import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from accelerate.utils import gather_object
from transformers import GenerationConfig, Trainer
from typing_extensions import override

from ...extras.constants import AUDIO_PLACEHOLDER
from ...extras.packages import is_vllm_available
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Sampler
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin, TrainerCallback

    from ...data import Template
    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def _ensure_local_vllm_dist_info(version: str) -> None:
    try:
        import importlib.metadata as metadata

        metadata.version("vllm")
        return
    except Exception:
        pass

    root = Path(tempfile.gettempdir()) / "llamafactory_vllm_distinfo"
    dist_info = root / f"vllm-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n", encoding="utf-8"
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: llamafactory-grpo\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("vllm\n", encoding="utf-8")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if str(root) not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(root), *pythonpath_entries])


def _prepare_trl_vllm_imports() -> tuple[Any, Any, Any, Any]:
    vllm_root = os.path.expanduser("~/projects/vllm")
    if os.path.isdir(vllm_root) and vllm_root not in sys.path:
        sys.path.insert(0, vllm_root)

    import trl.import_utils as trl_import_utils

    import vllm
    from vllm import LLM, SamplingParams
    from vllm import sampling_params as vllm_sampling_params
    from vllm.distributed import parallel_state
    from vllm import platforms as vllm_platforms
    from vllm.config.device import DeviceConfig
    from vllm.platforms.cuda import CudaPlatform
    from vllm.sampling_params import StructuredOutputsParams

    if not hasattr(vllm_sampling_params, "GuidedDecodingParams"):
        vllm_sampling_params.GuidedDecodingParams = StructuredOutputsParams

    resolved_platform = vllm_platforms.current_platform
    if not getattr(resolved_platform, "device_type", None) and torch.cuda.is_available():
        original_post_init = DeviceConfig.__post_init__

        def _cuda_fallback_post_init(self):
            try:
                original_post_init(self)
            except RuntimeError as exc:
                if self.device == "auto" and "Failed to infer device type" in str(exc) and torch.cuda.is_available():
                    self.device_type = "cuda"
                    self.device = torch.device("cuda")
                    return
                raise

        if not getattr(DeviceConfig, "_llamafactory_cuda_fallback", False):
            DeviceConfig.__post_init__ = _cuda_fallback_post_init
            DeviceConfig._llamafactory_cuda_fallback = True

        resolved_platform = CudaPlatform()
        vllm_platforms._current_platform = resolved_platform
        for module_name, module in list(sys.modules.items()):
            if module_name.startswith("vllm") and hasattr(module, "current_platform"):
                setattr(module, "current_platform", resolved_platform)

    GroupCoordinator = parallel_state.GroupCoordinator
    if not getattr(GroupCoordinator, "_llamafactory_custom_all_reduce_patch", False):
        original_init = GroupCoordinator.__init__

        def _patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if getattr(self, "world_size", 1) > 1:
                self.use_custom_op_call = False

        GroupCoordinator.__init__ = _patched_init
        GroupCoordinator._llamafactory_custom_all_reduce_patch = True

    _ensure_local_vllm_dist_info(getattr(vllm, "__version__", "0.10.2"))
    trl_import_utils._vllm_available = True
    trl_import_utils._vllm_version = getattr(vllm, "__version__", "0.10.2")

    import trl.trainer.grpo_trainer as trl_grpo_trainer

    trl_grpo_trainer.LLM = LLM
    trl_grpo_trainer.SamplingParams = SamplingParams
    trl_grpo_trainer._VLLMStructuredOutputsParams = StructuredOutputsParams
    return vllm, LLM, SamplingParams, StructuredOutputsParams


if not is_vllm_available():
    try:
        _VLLM_MODULE, _VLLM_LLM, _SamplingParams, _StructuredOutputsParams = _prepare_trl_vllm_imports()
    except Exception:
        _VLLM_MODULE = None
        _VLLM_LLM = None
        _SamplingParams = None
        _StructuredOutputsParams = None
else:
    _VLLM_MODULE, _VLLM_LLM, _SamplingParams, _StructuredOutputsParams = _prepare_trl_vllm_imports()


import trl.trainer.grpo_trainer as trl_grpo_trainer


GRPOTrainer = trl_grpo_trainer.GRPOTrainer
FSDP = trl_grpo_trainer.FSDP
logger = trl_grpo_trainer.logger
nanmax = trl_grpo_trainer.nanmax
nanmin = trl_grpo_trainer.nanmin
nanstd = trl_grpo_trainer.nanstd
nullcontext = trl_grpo_trainer.nullcontext
pad = trl_grpo_trainer.pad
profiling_context = trl_grpo_trainer.profiling_context
unwrap_model_for_generation = trl_grpo_trainer.unwrap_model_for_generation


_AUDIO_INPUT_KEYS = (
    "speech_ids",
    "speech_attention_mask",
    "input_features",
    "feature_attention_mask",
    "feature_exist_mask",
)


def _collect_forbidden_completion_strings(tokenizer: Any, processor: Any, template: Any) -> list[str]:
    forbidden: list[str] = []
    for source in (processor, tokenizer, getattr(template, "mm_plugin", None)):
        if source is None:
            continue
        for attr in ("audio_token", "audio_bos_token", "audio_eos_token", "audio_pad_token"):
            value = getattr(source, attr, None)
            if isinstance(value, str) and value and value not in forbidden:
                forbidden.append(value)
    return forbidden


def _collect_forbidden_completion_token_ids(tokenizer: Any, forbidden_strings: list[str]) -> list[list[int]]:
    token_ids: list[list[int]] = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in forbidden_strings:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            token_ids.append([int(token_id)])
    return token_ids


def _remove_forbidden_token_sequences(
    token_ids: list[int],
    forbidden_sequences: list[list[int]],
    logprobs: Optional[list[float]] = None,
    fallback_token_id: Optional[int] = None,
) -> tuple[list[int], Optional[list[float]], int]:
    if not token_ids or not forbidden_sequences:
        return token_ids, logprobs, 0

    normalized_sequences = sorted({tuple(seq) for seq in forbidden_sequences if seq}, key=len, reverse=True)
    if not normalized_sequences:
        return token_ids, logprobs, 0

    cleaned_ids: list[int] = []
    cleaned_logprobs: Optional[list[float]] = [] if logprobs is not None else None
    removed = 0
    index = 0
    while index < len(token_ids):
        matched_len: Optional[int] = None
        for seq in normalized_sequences:
            seq_len = len(seq)
            if tuple(token_ids[index : index + seq_len]) == seq:
                matched_len = seq_len
                break

        if matched_len is not None:
            removed += matched_len
            index += matched_len
            continue

        cleaned_ids.append(token_ids[index])
        if cleaned_logprobs is not None and logprobs is not None and index < len(logprobs):
            cleaned_logprobs.append(logprobs[index])
        index += 1

    if not cleaned_ids and fallback_token_id is not None:
        cleaned_ids = [int(fallback_token_id)]
        if cleaned_logprobs is not None:
            cleaned_logprobs = [0.0]

    return cleaned_ids, cleaned_logprobs, removed


class _MaskTokenIdsLogitsProcessor:
    def __init__(self, blocked_token_ids: list[int]) -> None:
        self.blocked_token_ids = tuple(int(token_id) for token_id in blocked_token_ids)

    def __call__(self, past_tokens_ids: list[int], logits: "torch.Tensor") -> "torch.Tensor":
        if self.blocked_token_ids:
            logits[list(self.blocked_token_ids)] = float("-inf")
        return logits


class CustomFunAudioChatGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: "PreTrainedModel",
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        template: "Template",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        callbacks: Optional[list["TrainerCallback"]] = None,
        **kwargs,
    ) -> None:
        if processor is None:
            raise ValueError("FunAudioChat GRPO requires a processor.")

        args = kwargs["args"]
        if args.use_vllm and args.vllm_mode != "colocate":
            raise NotImplementedError(
                "FunAudioChat GRPO only supports `grpo_vllm_mode=colocate` for now. TRL's server-side vLLM client is still image-only."
            )

        if args.use_vllm:
            os.environ.pop("PYTORCH_ALLOC_CONF", None)
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        if args.use_vllm:
            if _VLLM_MODULE is None or _VLLM_LLM is None:
                raise ImportError("FunAudioChat GRPO requested vLLM rollout but `import vllm` failed.")

            original_llm = trl_grpo_trainer.LLM

            def _memory_safe_llm(*llm_args, **llm_kwargs):
                llm_kwargs.setdefault("disable_custom_all_reduce", True)
                llm_kwargs.setdefault("enforce_eager", True)
                try:
                    return original_llm(*llm_args, **llm_kwargs)
                except Exception as exc:
                    error_text = str(exc)
                    if llm_kwargs.get("enable_sleep_mode") and "Sleep mode is not supported on current platform" in error_text:
                        llm_kwargs = dict(llm_kwargs)
                        llm_kwargs["enable_sleep_mode"] = False
                        if hasattr(logger, "warning_rank0"):
                            logger.warning_rank0(
                                "vLLM sleep mode is unsupported on this platform; retrying with sleep mode disabled."
                            )
                        else:
                            logger.warning(
                                "vLLM sleep mode is unsupported on this platform; retrying with sleep mode disabled."
                            )
                        return original_llm(*llm_args, **llm_kwargs)
                    raise

            trl_grpo_trainer.LLM = _memory_safe_llm
        else:
            original_llm = None

        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.template = template
        self.text_tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.processor = processor
        self.processing_class = processor

        try:
            super().__init__(model=model, processing_class=processor, callbacks=callbacks, **kwargs)
        finally:
            if original_llm is not None:
                trl_grpo_trainer.LLM = original_llm

        self.text_tokenizer.padding_side = "left"
        if getattr(self.text_tokenizer, "truncation_side", None) is not None:
            self.text_tokenizer.truncation_side = "left"

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if self.args.use_liger_loss:
            raise NotImplementedError("FunAudioChat GRPO currently does not support `use_liger_loss`.")

        self._forbidden_completion_strings = _collect_forbidden_completion_strings(
            self.text_tokenizer, self.processor, self.template
        )
        self._forbidden_completion_token_ids = _collect_forbidden_completion_token_ids(
            self.text_tokenizer, self._forbidden_completion_strings
        )
        self._forbidden_completion_single_token_ids = sorted(
            {token_ids[0] for token_ids in self._forbidden_completion_token_ids if len(token_ids) == 1}
        )

        generation_kwargs = dict(getattr(self.args, "generation_kwargs", None) or {})
        existing_bad_words = [item for item in generation_kwargs.get("bad_words", []) if isinstance(item, str)]
        for token in self._forbidden_completion_strings:
            if token not in existing_bad_words:
                existing_bad_words.append(token)
        if existing_bad_words:
            generation_kwargs["bad_words"] = existing_bad_words
        existing_logit_bias = generation_kwargs.get("logit_bias")
        normalized_logit_bias: dict[int, float] = {}
        if isinstance(existing_logit_bias, dict):
            for token_id, bias in existing_logit_bias.items():
                try:
                    normalized_logit_bias[int(token_id)] = float(bias)
                except (TypeError, ValueError):
                    continue
        for token_id in self._forbidden_completion_single_token_ids:
            normalized_logit_bias[token_id] = min(normalized_logit_bias.get(token_id, 0.0), -100.0)
        if normalized_logit_bias:
            generation_kwargs["logit_bias"] = normalized_logit_bias
        logits_processors = generation_kwargs.get("logits_processors")
        if logits_processors is None:
            logits_processors = []
        elif not isinstance(logits_processors, list):
            logits_processors = list(logits_processors)
        if self._forbidden_completion_single_token_ids and not self.use_vllm:
            logits_processors.append(_MaskTokenIdsLogitsProcessor(self._forbidden_completion_single_token_ids))
        if logits_processors:
            generation_kwargs["logits_processors"] = logits_processors
        else:
            generation_kwargs.pop("logits_processors", None)
        if self._forbidden_completion_token_ids:
            generation_kwargs["bad_words_ids"] = self._forbidden_completion_token_ids
        self.args.generation_kwargs = generation_kwargs

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=bool(self.temperature and self.temperature > 0),
            temperature=self.temperature if self.temperature else None,
            top_p=self.top_p if self.top_p is not None else 1.0,
            top_k=-1 if self.top_k is None else self.top_k,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.pad_token_id,
            bos_token_id=getattr(self.text_tokenizer, "bos_token_id", None),
            eos_token_id=self.template.get_stop_token_ids(self.text_tokenizer),
            bad_words_ids=self._forbidden_completion_token_ids or None,
        )

        if self._forbidden_completion_strings:
            if hasattr(logger, "info_rank0"):
                logger.info_rank0(
                    "Applied FunAudioChat GRPO completion token guard: "
                    f"bad_words={self._forbidden_completion_strings}."
                )
            else:
                logger.info(
                    "Applied FunAudioChat GRPO completion token guard: "
                    f"bad_words={self._forbidden_completion_strings}."
                )

    def _should_emit_rollout_debug(self) -> bool:
        debug_every_step = str(os.getenv("LLAMAFACTORY_GRPO_DEBUG_EVERY_STEP", "0")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if debug_every_step:
            return True
        step = int(getattr(self.state, "global_step", 0))
        logging_steps = max(1, int(getattr(self.args, "logging_steps", 1)))
        return step < 50 or ((step + 1) % logging_steps == 0)

    def _log_rollout_debug(self, message: str, *args: Any) -> None:
        if not self._should_emit_rollout_debug():
            return
        debug_all_ranks = str(os.getenv("LLAMAFACTORY_GRPO_DEBUG_ALL_RANKS", "0")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        log_fn = logger.info if debug_all_ranks else getattr(logger, "info_rank0", logger.info)
        log_fn(f"[rank={getattr(self.accelerator, 'process_index', -1)}] {message}", *args)

    def _pin_current_cuda_device(self) -> None:
        if not torch.cuda.is_available():
            return
        local_rank = int(getattr(self.accelerator, "local_process_index", 0))
        if torch.cuda.current_device() != local_rank:
            torch.cuda.set_device(local_rank)

    def _get_audio_sampling_rate(self) -> int:
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        return int(
            getattr(self.model_args, "audio_sampling_rate", None)
            or getattr(feature_extractor, "sampling_rate", None)
            or getattr(self.processor, "audio_sampling_rate", None)
            or getattr(self.processor, "sampling_rate", None)
            or 16000
        )

    def _normalize_vllm_audio_payload(self, sample_audios: list[Any]) -> list[tuple[np.ndarray, int]]:
        if not sample_audios:
            return []

        audio_data = self.template.mm_plugin._regularize_audios(
            sample_audios, sampling_rate=self._get_audio_sampling_rate()
        )
        normalized_audio_pairs: list[tuple[np.ndarray, int]] = []
        for wav, sampling_rate in zip(audio_data["audios"], audio_data["sampling_rates"]):
            if isinstance(wav, torch.Tensor):
                wav_array = wav.detach().cpu().to(torch.float32).contiguous().numpy()
            else:
                wav_array = np.asarray(wav, dtype=np.float32)
                if not wav_array.flags["C_CONTIGUOUS"]:
                    wav_array = np.ascontiguousarray(wav_array)
            normalized_audio_pairs.append((wav_array, int(sampling_rate)))

        return normalized_audio_pairs

    def _build_vllm_request_summary(
        self, prompt_ids: list[int], normalized_audio_pairs: list[tuple[np.ndarray, int]]
    ) -> dict[str, Any]:
        audio_token_id = self.text_tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER)
        return {
            "prompt_len": int(len(prompt_ids)),
            "num_audios": int(len(normalized_audio_pairs)),
            "audio_num_samples": [int(wav.shape[0]) for wav, _ in normalized_audio_pairs],
            "audio_sampling_rates": [int(sampling_rate) for _, sampling_rate in normalized_audio_pairs],
            "num_audio_placeholders": int(sum(1 for token_id in prompt_ids if token_id == audio_token_id)),
            "has_multi_modal_data": bool(normalized_audio_pairs),
        }

    def _assert_tp_request_summaries_match(self, request_summaries: list[dict[str, Any]]) -> None:
        if self.vllm_tensor_parallel_size <= 1 or not torch.distributed.is_initialized():
            return

        self._pin_current_cuda_device()
        gathered_summaries = [None for _ in range(self.vllm_tensor_parallel_size)]
        torch.distributed.all_gather_object(gathered_summaries, request_summaries, group=self.tp_group)
        baseline = gathered_summaries[0]
        mismatched_ranks = [rank for rank, summaries in enumerate(gathered_summaries[1:], start=1) if summaries != baseline]
        if mismatched_ranks:
            dump_dir = os.getenv("LLAMAFACTORY_GRPO_VLLM_DUMP_DIR")
            dump_path = None
            if dump_dir:
                step_dir = Path(dump_dir) / f"step_{int(self.state.global_step):06d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                dump_path = step_dir / (
                    f"tp_request_summary_mismatch_proc_{int(getattr(self.accelerator, 'process_index', -1)):02d}"
                    f"_tp_{int(torch.distributed.get_rank(group=self.tp_group)):02d}.json"
                )
                dump_payload = {
                    "global_step": int(self.state.global_step),
                    "process_index": int(getattr(self.accelerator, "process_index", -1)),
                    "tp_rank": int(torch.distributed.get_rank(group=self.tp_group)),
                    "mismatched_ranks": mismatched_ranks,
                    "gathered_summaries": gathered_summaries,
                }
                dump_path.write_text(json.dumps(dump_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(
                "TP ranks built different FunAudioChat vLLM requests before generate: "
                f"mismatched_ranks={mismatched_ranks}, local_rank_in_group={torch.distributed.get_rank(group=self.tp_group)}"
                + (f", dump_path={dump_path}." if dump_path is not None else ".")
            )

    def _dump_vllm_request(
        self, request_idx: int, prompt_ids: list[int], normalized_audio_pairs: list[tuple[np.ndarray, int]]
    ) -> None:
        dump_dir = os.getenv("LLAMAFACTORY_GRPO_VLLM_DUMP_DIR")
        if not dump_dir:
            return

        step_dir = Path(dump_dir) / f"step_{int(self.state.global_step):06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"req_{request_idx:04d}"
        metadata = {
            "global_step": int(self.state.global_step),
            "request_idx": int(request_idx),
            "prompt_len": int(len(prompt_ids)),
            "prompt_token_ids": prompt_ids,
            "num_audios": int(len(normalized_audio_pairs)),
            "audio_num_samples": [int(wav.shape[0]) for wav, _ in normalized_audio_pairs],
            "audio_sampling_rates": [int(sampling_rate) for _, sampling_rate in normalized_audio_pairs],
        }
        (step_dir / f"{base_name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        for audio_idx, (wav, sampling_rate) in enumerate(normalized_audio_pairs):
            np.save(step_dir / f"{base_name}_audio_{audio_idx:02d}_{int(sampling_rate)}.npy", wav)

    def _build_vllm_inputs(
        self, all_prompt_ids: list[list[int]], all_audios: list[list[Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._log_rollout_debug(
            "GRPO vllm input build start step=%s prompt_lists=%s",
            self.state.global_step,
            len(all_prompt_ids),
        )
        vllm_inputs: list[dict[str, Any]] = []
        request_summaries: list[dict[str, Any]] = []
        for request_idx, (prompt_ids, sample_audios) in enumerate(zip(all_prompt_ids, all_audios)):
            multi_modal_data = None
            normalized_audio_pairs: list[tuple[np.ndarray, int]] = []
            if sample_audios:
                normalized_audio_pairs = self._normalize_vllm_audio_payload(sample_audios)
                multi_modal_data = {"audio": normalized_audio_pairs}
            vllm_inputs.append({"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data})
            request_summaries.append(self._build_vllm_request_summary(prompt_ids, normalized_audio_pairs))
            if os.getenv("LLAMAFACTORY_GRPO_VLLM_DUMP_DIR"):
                self._dump_vllm_request(request_idx, prompt_ids, normalized_audio_pairs)
        self._log_rollout_debug(
            "GRPO vllm input build done step=%s requests=%s",
            self.state.global_step,
            len(vllm_inputs),
        )
        return vllm_inputs, request_summaries

    def _should_use_safe_serial_multimodal_rollout(self, all_audios: list[list[Any]]) -> bool:
        force_batch = str(os.getenv("LLAMAFACTORY_GRPO_FORCE_BATCH_VLLM", "0")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if force_batch:
            return False

        return (
            self.use_vllm
            and self.vllm_tensor_parallel_size > 1
            and any(len(sample_audios) > 0 for sample_audios in all_audios)
        )

    def _get_colocate_vllm_model(self):
        return self.llm.llm_engine.model_executor.driver_worker.model_runner.model

    def _load_weights_into_colocate_vllm(self, weights) -> None:
        llm_model = self._get_colocate_vllm_model()
        llm_model.load_weights(weights)

    def _sync_fsdp1_params_to_vllm(self, module: "nn.Module", prefix: str = "", visited=None):
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if not isinstance(module, FSDP):
            return

        with FSDP.summon_full_params(module, recurse=False, writeback=False):
            def iter_module_weights():
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])
                    if full_name in visited:
                        continue
                    visited.add(full_name)
                    yield full_name, param.data

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                for full_name, param in iter_module_weights():
                    self.vllm_client.update_named_param(full_name, param)
            elif self.vllm_mode == "colocate":
                self._load_weights_into_colocate_vllm(iter_module_weights())

    def _sync_fsdp2_params_to_vllm(self, module: "nn.Module"):
        def iter_state_dict_weights():
            for name, param in module.state_dict().items():
                if param.is_cpu:
                    param = param.to(torch.device("cuda"))
                yield name, param.full_tensor()

        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            for name, param in iter_state_dict_weights():
                self.vllm_client.update_named_param(name, param)
        elif self.vllm_mode == "colocate":
            self._load_weights_into_colocate_vllm(iter_state_dict_weights())

    def _move_model_to_vllm(self):
        self._pin_current_cuda_device()
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if trl_grpo_trainer.is_peft_model(self.model):
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                if self.is_fsdp_enabled:
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(self.model)
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    def iter_merged_adapter_weights():
                        for name, param in self.model.named_parameters():
                            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                            if self.model.prefix in name or "original_module" in name:
                                continue
                            yield self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."]), param.data

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        for name, param in iter_merged_adapter_weights():
                            self.vllm_client.update_named_param(name, param)
                    elif self.vllm_mode == "colocate":
                        self._load_weights_into_colocate_vllm(iter_merged_adapter_weights())

                self.model.unmerge_adapter()
        else:
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                def iter_model_weights():
                    for name, param in self.model.named_parameters():
                        fixed_name = self._fix_param_name_to_vllm(name)
                        with gather_if_zero3([param]):
                            yield fixed_name, param.data

                if self.vllm_mode == "server" and self.accelerator.is_main_process:
                    for name, param in iter_model_weights():
                        self.vllm_client.update_named_param(name, param)
                elif self.vllm_mode == "colocate":
                    self._load_weights_into_colocate_vllm(iter_model_weights())

    def _reset_vllm_prefix_cache(self):
        self._pin_current_cuda_device()
        self._log_rollout_debug("GRPO vllm reset-prefix-cache start step=%s", self.state.global_step)
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
            if hasattr(self.vllm_client, "reset_encoder_cache"):
                self.vllm_client.reset_encoder_cache()
            elif hasattr(self.vllm_client, "reset_mm_cache"):
                self.vllm_client.reset_mm_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
            llm_engine = getattr(self.llm, "llm_engine", None)
            if hasattr(self.llm, "reset_encoder_cache"):
                self.llm.reset_encoder_cache()
            elif llm_engine is not None and hasattr(llm_engine, "reset_encoder_cache"):
                llm_engine.reset_encoder_cache()
            if hasattr(self.llm, "reset_mm_cache"):
                self.llm.reset_mm_cache()
            elif llm_engine is not None and hasattr(llm_engine, "reset_mm_cache"):
                llm_engine.reset_mm_cache()
        self._log_rollout_debug("GRPO vllm reset-prefix-cache done step=%s", self.state.global_step)

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
    def _get_train_sampler(self, *args, **kwargs) -> Optional["Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    @override
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "audio", "audios", "reference_text", "sample_id"]

    def _ensure_audio_placeholders(self, messages: list[dict[str, str]], num_audios: int) -> list[dict[str, str]]:
        if num_audios <= 0:
            return messages
        if any(AUDIO_PLACEHOLDER in str(message.get("content", "")) for message in messages):
            return messages
        patched = deepcopy(messages)
        if not patched:
            patched = [{"role": "user", "content": AUDIO_PLACEHOLDER * num_audios}]
        else:
            patched[0]["content"] = AUDIO_PLACEHOLDER * num_audios + str(patched[0].get("content", ""))
        return patched

    def _extract_system_message(self, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], Optional[str]]:
        if messages and messages[0].get("role") == "system":
            return deepcopy(messages[1:]), str(messages[0].get("content", ""))
        return deepcopy(messages), None

    def _strip_upstream_audio_placeholders(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        cleaned = deepcopy(messages)
        for message in cleaned:
            content = message.get("content")
            if content is None:
                continue
            message["content"] = str(content).replace("<audio>", "")
        return cleaned

    def _build_prompt_messages(self, prompt: Any, audios: list[Any]) -> tuple[list[dict[str, str]], Optional[str]]:
        if isinstance(prompt, list):
            messages, system = self._extract_system_message(prompt)
        else:
            messages = [{"role": "user", "content": str(prompt)}]
            system = None
        messages = self._strip_upstream_audio_placeholders(messages)
        messages = self._ensure_audio_placeholders(messages, len(audios))
        return messages, system

    def _normalize_sample_audios(self, audios: Any) -> list[Any]:
        if audios is None:
            return []
        if isinstance(audios, tuple):
            audios = list(audios)
        elif not isinstance(audios, list):
            audios = [audios]

        if len(audios) == 1 and isinstance(audios[0], (list, tuple)):
            audios = list(audios[0])

        return audios

    def _dedupe_audio_placeholders(self, messages: list[dict[str, str]], num_audios: int) -> list[dict[str, str]]:
        if num_audios <= 0:
            return messages

        bos_token = getattr(self.processor, "audio_bos_token", "<|audio_bos|>")
        eos_token = getattr(self.processor, "audio_eos_token", "<|audio_eos|>")
        full_audio_placeholder = f"{bos_token}<|AUDIO|>{eos_token}"

        normalized = deepcopy(messages)
        for message in normalized:
            content = str(message.get("content", ""))
            generic_count = content.count(AUDIO_PLACEHOLDER)
            full_count = content.count(full_audio_placeholder)
            excess = generic_count + full_count - num_audios
            while excess > 0 and AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, "", 1)
                excess -= 1
            message["content"] = content

        return normalized

    def _build_prompt_ids(self, prompt: Any, audios: list[Any]) -> list[int]:
        audios = self._normalize_sample_audios(audios)
        messages, system = self._build_prompt_messages(prompt, audios)
        messages = self._dedupe_audio_placeholders(messages, len(audios))
        try:
            processed_messages = self.template.mm_plugin.process_messages(messages, [], [], audios, self.processor)
        except Exception as exc:
            prompt_preview = messages[0]["content"] if messages else ""
            raise ValueError(
                "Failed to build FunAudioChat GRPO prompt ids: "
                f"num_audios={len(audios)}, audio_type={type(audios).__name__}, "
                f"first_audio_type={type(audios[0]).__name__ if audios else 'None'}, "
                f"prompt_preview={prompt_preview[:200]!r}"
            ) from exc
        paired_messages = processed_messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = self.template.encode_oneturn(self.text_tokenizer, paired_messages, system, None)
        return prompt_ids

    def _split_audio_tensors_by_sample(
        self, mm_inputs: dict[str, torch.Tensor], audios_per_prompt: list[list[Any]]
    ) -> dict[str, list[Optional[torch.Tensor]]]:
        split_inputs: dict[str, list[Optional[torch.Tensor]]] = {key: [] for key in _AUDIO_INPUT_KEYS}
        if not audios_per_prompt:
            return split_inputs

        audio_counts = [len(audios) for audios in audios_per_prompt]
        audio_offsets = [0]
        for count in audio_counts:
            audio_offsets.append(audio_offsets[-1] + count)

        feature_exist_mask = mm_inputs.get("feature_exist_mask")
        feature_counts = []
        if torch.is_tensor(feature_exist_mask):
            for idx in range(len(audio_counts)):
                start, end = audio_offsets[idx], audio_offsets[idx + 1]
                feature_counts.append(int(feature_exist_mask[start:end].sum().item()))
        else:
            feature_counts = [count for count in audio_counts]

        feature_offsets = [0]
        for count in feature_counts:
            feature_offsets.append(feature_offsets[-1] + count)

        for idx in range(len(audio_counts)):
            audio_start, audio_end = audio_offsets[idx], audio_offsets[idx + 1]
            feat_start, feat_end = feature_offsets[idx], feature_offsets[idx + 1]

            for key in ("speech_ids", "speech_attention_mask", "feature_exist_mask"):
                value = mm_inputs.get(key)
                if torch.is_tensor(value):
                    split_inputs[key].append(value[audio_start:audio_end])
                else:
                    split_inputs[key].append(None)

            for key in ("input_features", "feature_attention_mask"):
                value = mm_inputs.get(key)
                if torch.is_tensor(value):
                    split_inputs[key].append(value[feat_start:feat_end])
                else:
                    split_inputs[key].append(None)

        return split_inputs

    def _prepare_prompt_batch(
        self, prompts: list[Any], audios_per_prompt: list[list[Any]]
    ) -> tuple[list[list[int]], dict[str, torch.Tensor], dict[str, list[Optional[torch.Tensor]]]]:
        audios_per_prompt = [self._normalize_sample_audios(audios) for audios in audios_per_prompt]
        prompt_ids_list = [self._build_prompt_ids(prompt, audios) for prompt, audios in zip(prompts, audios_per_prompt)]
        prompt_id_tensors = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids_list]
        prompt_mask_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_id_tensors]
        prompt_inputs = {
            "input_ids": pad(prompt_id_tensors, padding_value=self.pad_token_id, padding_side="left"),
            "attention_mask": pad(prompt_mask_tensors, padding_value=0, padding_side="left"),
        }

        flat_audios = [audio for audios in audios_per_prompt for audio in audios]
        if not flat_audios:
            return prompt_ids_list, prompt_inputs, {key: [None] * len(prompts) for key in _AUDIO_INPUT_KEYS}

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            images=[],
            videos=[],
            audios=flat_audios,
            imglens=[0] * len(prompts),
            vidlens=[0] * len(prompts),
            audlens=[len(audios) for audios in audios_per_prompt],
            batch_ids=prompt_ids_list,
            processor=self.processor,
        )
        mm_inputs.pop("feature_load_fail_mask", None)
        mm_inputs.pop("audio_duration_sec_by_audio", None)
        return prompt_ids_list, prompt_inputs, self._split_audio_tensors_by_sample(mm_inputs, audios_per_prompt)

    def _sanitize_completion_outputs(
        self,
        completion_ids: list[list[int]],
        logprobs: Optional[list[list[float]]] = None,
    ) -> tuple[list[list[int]], Optional[list[list[float]]]]:
        if not self._forbidden_completion_token_ids:
            return completion_ids, logprobs

        cleaned_completion_ids: list[list[int]] = []
        cleaned_logprobs: Optional[list[list[float]]] = [] if logprobs is not None else None
        removed_tokens = 0
        sanitized_completions = 0
        for index, ids in enumerate(completion_ids):
            completion_logprobs = None if logprobs is None else logprobs[index]
            clean_ids, clean_logprobs, removed = _remove_forbidden_token_sequences(
                token_ids=ids,
                forbidden_sequences=self._forbidden_completion_token_ids,
                logprobs=completion_logprobs,
                fallback_token_id=self.eos_token_id,
            )
            cleaned_completion_ids.append(clean_ids)
            if cleaned_logprobs is not None:
                cleaned_logprobs.append(clean_logprobs or [0.0] * len(clean_ids))
            if removed:
                sanitized_completions += 1
                removed_tokens += removed

        if sanitized_completions:
            log_fn = getattr(logger, "warning_rank0", logger.warning)
            log_fn(
                "Sanitized %s GRPO completions by removing %s forbidden audio tokens after rollout generation.",
                sanitized_completions,
                removed_tokens,
            )

        return cleaned_completion_ids, cleaned_logprobs

    def _pack_audio_kwargs(
        self, start: int, batch_size: int, split_audio_kwargs: dict[str, list[Optional[torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        model_inputs: dict[str, torch.Tensor] = {}
        for key in _AUDIO_INPUT_KEYS:
            pieces = split_audio_kwargs.get(key)
            if pieces is None:
                continue
            selected = [piece for piece in pieces[start : start + batch_size] if piece is not None]
            if selected:
                model_inputs[key] = torch.cat(selected, dim=0)
        return model_inputs

    @override
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        speech_ids=None,
        speech_attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        feature_exist_mask=None,
    ):
        batch_size = batch_size or input_ids.size(0)
        split_audio_kwargs = {
            "speech_ids": speech_ids if isinstance(speech_ids, list) else None,
            "speech_attention_mask": speech_attention_mask if isinstance(speech_attention_mask, list) else None,
            "input_features": input_features if isinstance(input_features, list) else None,
            "feature_attention_mask": feature_attention_mask if isinstance(feature_attention_mask, list) else None,
            "feature_exist_mask": feature_exist_mask if isinstance(feature_exist_mask, list) else None,
        }

        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            self._log_rollout_debug(
                "GRPO token-logps chunk start step=%s start=%s chunk_batch=%s seq_len=%s",
                self.state.global_step,
                start,
                input_ids_batch.size(0),
                input_ids_batch.size(1),
            )
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            model_inputs.update(self._pack_audio_kwargs(start, batch_size, split_audio_kwargs))
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            self._log_rollout_debug(
                "GRPO token-logps chunk done step=%s start=%s logits_shape=%s",
                self.state.global_step,
                start,
                tuple(logits.shape),
            )
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = trl_grpo_trainer.selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = trl_grpo_trainer.entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _generate_single_turn(self, prompts: list[Any], audios: list[list[Any]]):
        device = self.accelerator.device
        if not audios:
            audios = [[] for _ in prompts]

        prompt_ids_list, prompt_inputs, split_audio_kwargs = self._prepare_prompt_batch(prompts, audios)
        if self.use_vllm:
            if self.vllm_tensor_parallel_size < self.accelerator.num_processes:
                self._log_rollout_debug("GRPO world sync before rollout step=%s", self.state.global_step)
                self.accelerator.wait_for_everyone()
                self._log_rollout_debug("GRPO world sync before rollout done step=%s", self.state.global_step)
            if self.args.vllm_enable_sleep_mode:
                self._log_rollout_debug("GRPO wake-up start step=%s", self.state.global_step)
                torch.cuda.empty_cache()
                self.llm.wake_up()
                self._log_rollout_debug("GRPO wake-up done step=%s", self.state.global_step)
            if self.state.global_step != self._last_loaded_step:
                self._log_rollout_debug(
                    "GRPO sync-to-vllm start step=%s last_loaded_step=%s",
                    self.state.global_step,
                    self._last_loaded_step,
                )
                self._move_model_to_vllm()
                self._log_rollout_debug("GRPO sync-to-vllm weights done step=%s", self.state.global_step)
                if self.vllm_tensor_parallel_size < self.accelerator.num_processes:
                    self._log_rollout_debug("GRPO world sync after weight sync step=%s", self.state.global_step)
                    self.accelerator.wait_for_everyone()
                    self._log_rollout_debug("GRPO world sync after weight sync done step=%s", self.state.global_step)
                self._reset_vllm_prefix_cache()
                if self.vllm_tensor_parallel_size < self.accelerator.num_processes:
                    self._log_rollout_debug("GRPO world sync after cache reset step=%s", self.state.global_step)
                    self.accelerator.wait_for_everyone()
                    self._log_rollout_debug("GRPO world sync after cache reset done step=%s", self.state.global_step)
                self._last_loaded_step = self.state.global_step
                self._log_rollout_debug("GRPO sync-to-vllm done step=%s", self.state.global_step)

            structured_outputs = (
                _StructuredOutputsParams(regex=self.guided_decoding_regex)
                if self.guided_decoding_regex and _StructuredOutputsParams is not None
                else None
            )
            generation_kwargs = {
                "n": 1,
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": -1 if self.top_k is None else self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": self.max_completion_length,
                "truncate_prompt_tokens": self.max_prompt_length,
                "structured_outputs": structured_outputs,
                "logprobs": 0,
            }
            if self.args.generation_kwargs is not None:
                generation_kwargs.update(self.args.generation_kwargs)
            generation_kwargs.pop("bad_words_ids", None)
            sampling_params = _SamplingParams(**generation_kwargs)

            if self.vllm_tensor_parallel_size > 1:
                original_size = len(prompt_ids_list)
                self._pin_current_cuda_device()
                self._log_rollout_debug(
                    "GRPO prompt gather start step=%s local_prompts=%s tp=%s",
                    self.state.global_step,
                    original_size,
                    self.vllm_tensor_parallel_size,
                )
                gathered_prompt_ids = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompt_ids, prompt_ids_list, group=self.tp_group)
                all_prompt_ids = [ids for sublist in gathered_prompt_ids for ids in sublist]
                self._log_rollout_debug(
                    "GRPO prompt gather done step=%s gathered_prompts=%s",
                    self.state.global_step,
                    len(all_prompt_ids),
                )

                self._log_rollout_debug(
                    "GRPO audio gather start step=%s local_audio_lists=%s",
                    self.state.global_step,
                    len(audios),
                )
                gathered_audios = [None for _ in range(self.vllm_tensor_parallel_size)]
                self._pin_current_cuda_device()
                torch.distributed.all_gather_object(gathered_audios, audios, group=self.tp_group)
                all_audios = [audio_list for sublist in gathered_audios for audio_list in sublist]
                self._log_rollout_debug(
                    "GRPO audio gather done step=%s gathered_audio_lists=%s",
                    self.state.global_step,
                    len(all_audios),
                )
            else:
                original_size = len(prompt_ids_list)
                all_prompt_ids = prompt_ids_list
                all_audios = audios

            total_audio_items = sum(len(sample_audios) for sample_audios in all_audios)
            if self.vllm_tensor_parallel_size > 1 and total_audio_items > 0:
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_payload = None
                if local_rank_in_group == 0:
                    vllm_inputs, request_summaries = self._build_vllm_inputs(all_prompt_ids, all_audios)
                    tp_payload = {"vllm_inputs": vllm_inputs, "request_summaries": request_summaries}
                self._pin_current_cuda_device()
                gathered_payloads = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_payloads, tp_payload, group=self.tp_group)
                canonical_payload = gathered_payloads[0]
                if canonical_payload is None:
                    raise RuntimeError(
                        "Failed to gather canonical FunAudioChat vLLM inputs within TP group: root payload is None."
                    )
                vllm_inputs = canonical_payload["vllm_inputs"]
                request_summaries = canonical_payload["request_summaries"]
                self._log_rollout_debug(
                    "GRPO canonical vllm input broadcast done step=%s requests=%s tp=%s",
                    self.state.global_step,
                    len(vllm_inputs),
                    self.vllm_tensor_parallel_size,
                )
            else:
                vllm_inputs, request_summaries = self._build_vllm_inputs(all_prompt_ids, all_audios)

            self._assert_tp_request_summaries_match(request_summaries)

            self._log_rollout_debug(
                "GRPO rollout start step=%s requests=%s total_audio_items=%s tp=%s",
                self.state.global_step,
                len(vllm_inputs),
                total_audio_items,
                self.vllm_tensor_parallel_size,
            )
            generate_start = perf_counter()
            self._pin_current_cuda_device()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self._should_use_safe_serial_multimodal_rollout(all_audios):
                self._log_rollout_debug(
                    "GRPO rollout safe-serial mode enabled step=%s requests=%s",
                    self.state.global_step,
                    len(vllm_inputs),
                )
                all_outputs = []
                for request_idx, vllm_input in enumerate(vllm_inputs):
                    normalized_audio_pairs = (
                        [] if vllm_input["multi_modal_data"] is None else vllm_input["multi_modal_data"]["audio"]
                    )
                    self._dump_vllm_request(request_idx, vllm_input["prompt_token_ids"], normalized_audio_pairs)
                    self._log_rollout_debug(
                        "GRPO rollout item start step=%s request=%s prompt_len=%s audio_items=%s audio_num_samples=%s",
                        self.state.global_step,
                        request_idx,
                        len(vllm_input["prompt_token_ids"]),
                        len(normalized_audio_pairs),
                        [int(wav.shape[0]) for wav, _ in normalized_audio_pairs],
                    )
                    item_start = perf_counter()
                    with profiling_context(self, f"vLLM.generate.item.{request_idx}"):
                        outputs = self.llm.generate([vllm_input], sampling_params=sampling_params, use_tqdm=False)
                    if len(outputs) != 1:
                        raise RuntimeError(
                            f"Expected exactly one vLLM output in safe-serial mode, got {len(outputs)}."
                        )
                    all_outputs.extend(outputs)
                    self._log_rollout_debug(
                        "GRPO rollout item done step=%s request=%s duration=%.2fs",
                        self.state.global_step,
                        request_idx,
                        perf_counter() - item_start,
                    )
            else:
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._log_rollout_debug(
                "GRPO rollout done step=%s requests=%s completions=%s duration=%.2fs",
                self.state.global_step,
                len(all_outputs),
                sum(len(outputs.outputs) for outputs in all_outputs),
                perf_counter() - generate_start,
            )

            all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
            all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
            all_logprobs = [
                [next(iter(logprob.values())).logprob for logprob in output.logprobs] if output.logprobs else []
                for outputs in all_outputs
                for output in outputs.outputs
            ]

            if self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(local_rank_in_group * original_size, (local_rank_in_group + 1) * original_size)
                prompt_ids = all_prompt_ids[tp_slice]
                completion_ids = all_completion_ids[tp_slice]
                logprobs = all_logprobs[tp_slice]
            else:
                prompt_ids = all_prompt_ids
                completion_ids = all_completion_ids
                logprobs = all_logprobs

            completion_ids, logprobs = self._sanitize_completion_outputs(completion_ids, logprobs)

            if self.args.vllm_enable_sleep_mode:
                self.llm.sleep(level=1)
        else:
            generate_inputs = {
                "input_ids": prompt_inputs["input_ids"].to(device),
                "attention_mask": prompt_inputs["attention_mask"].to(device),
            }
            generate_inputs.update(
                {
                    key: value.to(device)
                    for key, value in self._pack_audio_kwargs(0, len(prompts), split_audio_kwargs).items()
                }
            )
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            prompt_ids_tensor = generate_inputs["input_ids"]
            prompt_mask_tensor = generate_inputs["attention_mask"]
            prompt_length = prompt_ids_tensor.size(1)
            completion_tensor = prompt_completion_ids[:, prompt_length:]
            is_eos = completion_tensor == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids_tensor, prompt_mask_tensor.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_tensor, completion_mask.bool())]
            logprobs = None

        return prompt_ids, completion_ids, logprobs, split_audio_kwargs

    def _generate(self, prompts: list[Any], audios: list[list[Any]]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(prompts, audios)
        if self.use_vllm and self.vllm_tensor_parallel_size < self.accelerator.num_processes:
            self._log_rollout_debug("GRPO world sync after rollout step=%s", self.state.global_step)
            self.accelerator.wait_for_everyone()
            self._log_rollout_debug("GRPO world sync after rollout done step=%s", self.state.global_step)
        self._log_rollout_debug(
            "GRPO local decode ready step=%s prompts=%s completions=%s",
            self.state.global_step,
            len(prompt_ids),
            len(completion_ids),
        )

        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        self._log_rollout_debug(
            "GRPO metric gather start step=%s prompt_len_shape=%s completion_len_shape=%s",
            self.state.global_step,
            tuple(prompt_lengths.shape),
            tuple(completion_lengths.shape),
        )
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._log_rollout_debug(
            "GRPO metric gather done step=%s agg_prompt_shape=%s agg_completion_shape=%s",
            self.state.global_step,
            tuple(agg_prompt_lengths.shape),
            tuple(agg_completion_lengths.shape),
        )
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()

        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())
        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [example["prompt"] for example in inputs]
        audios = []
        for example in inputs:
            if example.get("audios") is not None:
                audios.append(example.get("audios") or [])
            elif example.get("audio") is not None:
                audios.append([example.get("audio")])
            else:
                audios.append([])

        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, forward_kwargs = self._generate(
            prompts, audios
        )

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        self._log_rollout_debug(
            "GRPO score start step=%s local_batch=%s seq_len=%s logits_to_keep=%s",
            self.state.global_step,
            prompt_completion_ids.size(0),
            prompt_completion_ids.size(1),
            logits_to_keep,
        )

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                self._log_rollout_debug("GRPO old-logps start step=%s", self.state.global_step)
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    speech_ids=forward_kwargs.get("speech_ids"),
                    speech_attention_mask=forward_kwargs.get("speech_attention_mask"),
                    input_features=forward_kwargs.get("input_features"),
                    feature_attention_mask=forward_kwargs.get("feature_attention_mask"),
                    feature_exist_mask=forward_kwargs.get("feature_exist_mask"),
                )
                self._log_rollout_debug("GRPO old-logps done step=%s", self.state.global_step)
            else:
                old_per_token_logps = None

            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            if self.beta != 0.0:
                if self.ref_model is not None:
                    self._log_rollout_debug("GRPO ref-logps start step=%s ref_model=yes", self.state.global_step)
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        speech_ids=forward_kwargs.get("speech_ids"),
                        speech_attention_mask=forward_kwargs.get("speech_attention_mask"),
                        input_features=forward_kwargs.get("input_features"),
                        feature_attention_mask=forward_kwargs.get("feature_attention_mask"),
                        feature_exist_mask=forward_kwargs.get("feature_exist_mask"),
                    )
                    self._log_rollout_debug("GRPO ref-logps done step=%s ref_model=yes", self.state.global_step)
                else:
                    self._log_rollout_debug("GRPO ref-logps start step=%s ref_model=adapter-disabled", self.state.global_step)
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            speech_ids=forward_kwargs.get("speech_ids"),
                            speech_attention_mask=forward_kwargs.get("speech_attention_mask"),
                            input_features=forward_kwargs.get("input_features"),
                            feature_attention_mask=forward_kwargs.get("feature_attention_mask"),
                            feature_exist_mask=forward_kwargs.get("feature_exist_mask"),
                        )
                    self._log_rollout_debug(
                        "GRPO ref-logps done step=%s ref_model=adapter-disabled", self.state.global_step
                    )
            else:
                ref_per_token_logps = None

        prompts_text = self.text_tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.text_tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        if trl_grpo_trainer.is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt and prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        reward_start = perf_counter()
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._log_rollout_debug(
            "GRPO reward done step=%s batch=%s duration=%.2fs",
            self.state.global_step,
            len(completions_text),
            perf_counter() - reward_start,
        )
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(f"Invalid value for scale_rewards: {self.scale_rewards}.")

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(torch.nanmean(rewards_per_func[:, i]).item())
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(nanstd(rewards_per_func[:, i]).item())
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        should_collect_completion_logs = self.log_completions and (
            self.state.global_step < 10
            or ((int(self.state.global_step) + 1) % max(1, int(getattr(self.args, "logging_steps", 1))) == 0)
        )
        if should_collect_completion_logs:
            gather_start = perf_counter()
            self._log_rollout_debug("GRPO log-gather start step=%s", self.state.global_step)
            self._pin_current_cuda_device()
            self._logs["prompt"].extend(gather_object(prompts_text))
            self._pin_current_cuda_device()
            self._logs["completion"].extend(gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._logs["advantages"].extend(all_process_advantages.tolist())
            self._log_rollout_debug(
                "GRPO log-gather done step=%s duration=%.2fs",
                self.state.global_step,
                perf_counter() - gather_start,
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        for key in _AUDIO_INPUT_KEYS:
            output[key] = forward_kwargs.get(key)
        return output

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        self._log_rollout_debug(
            "GRPO compute-loss start step=%s local_batch=%s seq_len=%s logits_to_keep=%s",
            self.state.global_step,
            input_ids.size(0),
            input_ids.size(1),
            logits_to_keep,
        )

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            speech_ids=inputs.get("speech_ids"),
            speech_attention_mask=inputs.get("speech_attention_mask"),
            input_features=inputs.get("input_features"),
            feature_attention_mask=inputs.get("feature_attention_mask"),
            feature_exist_mask=inputs.get("feature_exist_mask"),
        )
        self._log_rollout_debug(
            "GRPO compute-loss token-logps done step=%s logps_shape=%s entropy_shape=%s",
            self.state.global_step,
            tuple(per_token_logps.shape),
            tuple(entropies.shape) if entropies is not None else None,
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}.")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        self._log_rollout_debug(
            "GRPO compute-loss scalar ready step=%s loss=%.6f",
            self.state.global_step,
            loss.detach().float().item(),
        )

        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._log_rollout_debug("GRPO compute-loss kl-gather start step=%s", self.state.global_step)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())
            self._log_rollout_debug("GRPO compute-loss kl-gather done step=%s", self.state.global_step)

        mean_entropy = masked_batch_mean(entropies)
        self._log_rollout_debug("GRPO compute-loss entropy-gather start step=%s", self.state.global_step)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        self._log_rollout_debug("GRPO compute-loss entropy-gather done step=%s", self.state.global_step)

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        clip_ratio = (is_low_clipped | is_high_clipped).float()
        if entropy_mask is not None:
            clip_ratio = clip_ratio * entropy_mask
        clip_ratio = masked_batch_mean(clip_ratio)
        self._log_rollout_debug("GRPO compute-loss clip-gather start step=%s", self.state.global_step)
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).nanmean().item())
        self._log_rollout_debug("GRPO compute-loss clip-gather done step=%s", self.state.global_step)

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - inputs["old_per_token_logps"])
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=loss.device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=loss.device)
            self._log_rollout_debug("GRPO compute-loss sampling-gathers start step=%s", self.state.global_step)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            flat_ratio = inputs["importance_sampling_ratio"][completion_mask.bool()]
            min_ratio = torch.min(flat_ratio) if flat_ratio.numel() > 0 else torch.tensor(0.0, device=loss.device)
            mean_ratio = torch.mean(flat_ratio) if flat_ratio.numel() > 0 else torch.tensor(0.0, device=loss.device)
            max_ratio = torch.max(flat_ratio) if flat_ratio.numel() > 0 else torch.tensor(0.0, device=loss.device)
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_ratio)).item()
            )
            self._log_rollout_debug("GRPO compute-loss sampling-gathers done step=%s", self.state.global_step)

        self._log_rollout_debug("GRPO compute-loss done step=%s", self.state.global_step)
        return loss
