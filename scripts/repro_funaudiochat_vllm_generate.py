#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import AUDIO_PLACEHOLDER
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

AUDIO_TOKEN = "<|AUDIO|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal FunAudioChat vLLM reproduction script without GRPO. "
            "Use it to compare TP=1/2 and text-only/audio generate paths."
        )
    )
    parser.add_argument("--model", required=True, help="FunAudioChat checkpoint or model id.")
    parser.add_argument("--dataset", required=True, help="Dataset name from dataset_info.json.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory used by LLaMA-Factory.")
    parser.add_argument("--template", default="funaudiochat", help="Template name. Keep `funaudiochat`.")
    parser.add_argument("--cutoff-len", type=int, default=1024, help="Prompt cutoff length.")
    parser.add_argument("--max-new-tokens", type=int, default=192, help="Generation length.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--input-mode",
        choices=("audio", "text-only"),
        default="audio",
        help="Whether to send audio multimodal payloads or text-only prompts into vLLM.",
    )
    parser.add_argument(
        "--batch-mode",
        choices=("auto", "microbatch", "batch"),
        default="auto",
        help="Value forwarded to `VLLM_FUNAUDIOCHAT_AUDIO_BATCH_MODE`.",
    )
    parser.add_argument("--sample-offset", type=int, default=0, help="Dataset start offset.")
    parser.add_argument("--max-samples", type=int, default=2, help="Number of samples to send in one generate call.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--iterations", type=int, default=1, help="How many times to call generate() with the same batch.")
    parser.add_argument(
        "--request-mode",
        choices=("batch", "serial"),
        default="batch",
        help="Whether to send all requests in a single batched generate() call or one request at a time.",
    )
    parser.add_argument(
        "--reset-cache-each-iter",
        action="store_true",
        help="Reset vLLM prefix/mm cache before each iteration.",
    )
    parser.add_argument(
        "--self-reload-weights-each-iter",
        action="store_true",
        help="Exercise the colocate vLLM load_weights path by reloading the current model weights before each iteration.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Forward trust_remote_code to vLLM.")
    parser.add_argument("--debug", action="store_true", help="Enable extra debug env for local vLLM.")
    return parser.parse_args()


def normalize_sample_audios(audios: Any) -> list[Any]:
    if audios is None:
        return []
    if isinstance(audios, tuple):
        audios = list(audios)
    elif not isinstance(audios, list):
        audios = [audios]
    if len(audios) == 1 and isinstance(audios[0], (list, tuple)):
        audios = list(audios[0])
    return audios


def extract_system_message(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str | None]:
    if messages and messages[0].get("role") == "system":
        return deepcopy(messages[1:]), str(messages[0].get("content", ""))
    return deepcopy(messages), None


def strip_upstream_audio_placeholders(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned = deepcopy(messages)
    for message in cleaned:
        content = message.get("content")
        if content is not None:
            message["content"] = str(content).replace("<audio>", "")
    return cleaned


def ensure_audio_placeholders(messages: list[dict[str, str]], num_audios: int) -> list[dict[str, str]]:
    patched = deepcopy(messages)
    if num_audios <= 0:
        return patched

    placeholder_count = sum(str(message.get("content", "")).count(AUDIO_PLACEHOLDER) for message in patched)
    if placeholder_count >= num_audios:
        return patched

    missing = num_audios - placeholder_count
    if not patched:
        patched = [{"role": "user", "content": AUDIO_PLACEHOLDER * missing}]
    elif patched[0].get("role") == "user":
        patched[0]["content"] = AUDIO_PLACEHOLDER * missing + str(patched[0].get("content", ""))
    else:
        patched.insert(0, {"role": "user", "content": AUDIO_PLACEHOLDER * missing})
    return patched


def dedupe_audio_placeholders(processor: Any, messages: list[dict[str, str]], num_audios: int) -> list[dict[str, str]]:
    if num_audios <= 0:
        return deepcopy(messages)

    bos_token = getattr(processor, "audio_bos_token", "<|audio_bos|>")
    eos_token = getattr(processor, "audio_eos_token", "<|audio_eos|>")
    full_audio_placeholder = f"{bos_token}{AUDIO_PLACEHOLDER}{eos_token}"

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


def build_prompt_messages(prompt: Any, audios: list[Any]) -> tuple[list[dict[str, str]], str | None]:
    if isinstance(prompt, list):
        messages, system = extract_system_message(prompt)
    else:
        messages = [{"role": "user", "content": str(prompt)}]
        system = None
    messages = strip_upstream_audio_placeholders(messages)
    messages = ensure_audio_placeholders(messages, len(audios))
    return messages, system


def build_prompt_ids(template: Any, tokenizer: Any, processor: Any, prompt: Any, audios: list[Any]) -> list[int]:
    messages, system = build_prompt_messages(prompt, audios)
    messages = dedupe_audio_placeholders(processor, messages, len(audios))
    if audios:
        processed_messages = template.mm_plugin.process_messages(messages, [], [], audios, processor)
    else:
        processed_messages = messages
    paired_messages = processed_messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system, None)
    return prompt_ids


def summarize_request(prompt_ids: list[int], audios: list[Any], audio_data: dict[str, Any] | None) -> dict[str, Any]:
    sampling_rates = []
    audio_lengths = []
    if audio_data is not None:
        sampling_rates = [int(sr) for sr in audio_data.get("sampling_rates", [])]
        audio_lengths = [int(len(audio)) for audio in audio_data.get("audios", [])]
    return {
        "prompt_len": len(prompt_ids),
        "num_audios": len(audios),
        "audio_lengths": audio_lengths,
        "sampling_rates": sampling_rates,
        "has_multi_modal_data": bool(audio_data),
        "num_audio_placeholders": 0,
    }


def ensure_local_vllm_dist_info(version: str) -> None:
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
        "Wheel-Version: 1.0\nGenerator: llamafactory-grpo-repro\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("vllm\n", encoding="utf-8")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if str(root) not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(root), *pythonpath_entries])


def prepare_local_vllm_imports() -> tuple[Any, Any]:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config.device import DeviceConfig
    from vllm.platforms.cuda import CudaPlatform
    from vllm import platforms as vllm_platforms

    ensure_local_vllm_dist_info(getattr(vllm, "__version__", "0.10.2"))

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

    return LLM, SamplingParams


def main() -> None:
    args = parse_args()
    if not is_vllm_available():
        raise SystemExit("vLLM is required for this script.")

    LLM, SamplingParams = prepare_local_vllm_imports()

    os.environ["VLLM_FUNAUDIOCHAT_AUDIO_BATCH_MODE"] = args.batch_mode
    if args.debug:
        os.environ["VLLM_FUNAUDIOCHAT_DEBUG"] = "1"

    model_args, data_args, _, generating_args = get_infer_args(
        {
            "model_name_or_path": args.model,
            "dataset": args.dataset,
            "dataset_dir": args.dataset_dir,
            "template": args.template,
            "cutoff_len": args.cutoff_len,
            "max_samples": args.sample_offset + args.max_samples,
            "preprocessing_num_workers": 1,
            "overwrite_cache": False,
            "trust_remote_code": args.trust_remote_code,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        }
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_repro")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    template.mm_plugin.expand_mm_tokens = False

    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="grpo", **tokenizer_module)
    dataset = dataset_module["train_dataset"]
    end = min(len(dataset), args.sample_offset + args.max_samples)
    samples = [dataset[index] for index in range(args.sample_offset, end)]
    if not samples:
        raise SystemExit("No samples selected.")

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.cutoff_len + args.max_new_tokens,
        "disable_log_stats": True,
        "limit_mm_per_prompt": {"audio": 8},
    }
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,
        stop_token_ids=template.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        seed=args.seed,
    )

    vllm_inputs = []
    request_summaries = []
    audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
    feature_extractor = getattr(processor, "feature_extractor", None)
    audio_sampling_rate = getattr(feature_extractor, "sampling_rate", None) or getattr(processor, "sampling_rate", None) or 16000
    for sample in samples:
        audios = normalize_sample_audios(sample.get("audios"))
        if args.input_mode == "text-only":
            audios = []

        prompt_ids = build_prompt_ids(template, tokenizer, processor, sample["prompt"], audios)
        audio_data = None
        multi_modal_data = None
        if audios:
            audio_data = template.mm_plugin._regularize_audios(audios, sampling_rate=audio_sampling_rate)
            multi_modal_data = {"audio": list(zip(audio_data["audios"], audio_data["sampling_rates"]))}

        summary = summarize_request(prompt_ids, audios, audio_data)
        summary["num_audio_placeholders"] = sum(1 for token_id in prompt_ids if token_id == audio_token_id)
        request_summaries.append(summary)
        vllm_inputs.append({"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data})

    print(json.dumps({"engine_args": engine_args, "input_mode": args.input_mode, "batch_mode": args.batch_mode}, indent=2))
    for index, summary in enumerate(request_summaries):
        print(json.dumps({"request_index": index, **summary}, ensure_ascii=False))

    def _reset_caches() -> None:
        llm.reset_prefix_cache()
        if hasattr(llm, "reset_encoder_cache"):
            llm.reset_encoder_cache()
        elif hasattr(llm, "reset_mm_cache"):
            llm.reset_mm_cache()
        elif hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "reset_mm_cache"):
            llm.llm_engine.reset_mm_cache()

    self_reload_fn = None
    if args.self_reload_weights_each_iter:
        llm_engine = getattr(llm, "llm_engine", None)
        model_executor = getattr(llm_engine, "model_executor", None)
        if model_executor is not None and hasattr(model_executor, "collective_rpc"):
            self_reload_fn = lambda: model_executor.collective_rpc("reload_weights")
        elif model_executor is not None and hasattr(model_executor, "reload_weights"):
            self_reload_fn = model_executor.reload_weights
        else:
            raise RuntimeError("Could not locate colocate vLLM executor for self weight reload.")

    def _self_reload_weights() -> None:
        if self_reload_fn is None:
            return
        self_reload_fn()

    for iteration in range(args.iterations):
        if args.self_reload_weights_each_iter:
            print(json.dumps({"iteration": iteration, "event": "self_reload_start"}))
            _self_reload_weights()
            print(json.dumps({"iteration": iteration, "event": "self_reload_done"}))
        if args.reset_cache_each_iter:
            _reset_caches()

        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "request_mode": args.request_mode,
                    "input_mode": args.input_mode,
                    "num_requests": len(vllm_inputs),
                },
                ensure_ascii=False,
            )
        )

        if args.request_mode == "serial":
            outputs = []
            for request_index, vllm_input in enumerate(vllm_inputs):
                print(json.dumps({"iteration": iteration, "request_index": request_index, "event": "generate_start"}))
                result = llm.generate([vllm_input], sampling_params)
                print(json.dumps({"iteration": iteration, "request_index": request_index, "event": "generate_done"}))
                outputs.extend(result)
        else:
            print(json.dumps({"iteration": iteration, "event": "generate_start", "batched_requests": len(vllm_inputs)}))
            outputs = llm.generate(vllm_inputs, sampling_params)
            print(json.dumps({"iteration": iteration, "event": "generate_done", "batched_requests": len(vllm_inputs)}))

        for index, output in enumerate(outputs):
            text = output.outputs[0].text if output.outputs else ""
            print(json.dumps({"iteration": iteration, "request_index": index, "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
