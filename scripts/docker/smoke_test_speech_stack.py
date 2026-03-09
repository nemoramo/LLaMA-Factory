#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import PeftConfig
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data.template import TEMPLATES
from llamafactory.model.funaudiochat.register import register_funaudiochat
from llamafactory.model.qwen3_asr.register import register_qwen3_asr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENDPOINTING_LABELS = {"<EOU>", "<CONT_USER>", "<UNADDRESSED>"}


def _run(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def _check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required binary: {name}")
    version_line = _run([name, "-version"]).splitlines()[0]
    print(f"[binary] {version_line}")


def _check_python_module(name: str) -> None:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"[python] {name}={version}")


def _check_audio_stack() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        _check_binary(binary)

    for module_name in ("audioread", "librosa", "pydub", "soundfile", "soxr"):
        _check_python_module(module_name)


def _check_gpu(require_cuda: bool, require_fa2: bool) -> None:
    cuda_available = torch.cuda.is_available()
    print(f"[torch] cuda_available={cuda_available}")
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for this smoke test, but torch.cuda.is_available() is False.")

    if not cuda_available:
        return

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"[torch] device0={device_name} capability={capability}")

    if require_fa2 and not is_flash_attn_2_available():
        raise RuntimeError("FlashAttention-2 is required for this smoke test, but transformers cannot find it.")
    print(f"[torch] flash_attn_2_available={is_flash_attn_2_available()}")

    x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    y = x @ x.transpose(0, 1)
    torch.cuda.synchronize()
    print(f"[torch] bf16_matmul_mean={y.float().mean().item():.6f}")


def _check_templates_and_plugins() -> None:
    register_qwen3_asr()
    register_funaudiochat()

    required_templates = ("qwen3_asr", "funaudiochat", "qwen3_5_nothink")
    for name in required_templates:
        if name not in TEMPLATES:
            raise RuntimeError(f"Missing template registration: {name}")
        print(f"[template] {name}=ok")

    qwen3_asr_plugin = get_mm_plugin(name="qwen3_asr", audio_token="<|audio_pad|>")
    funaudiochat_plugin = get_mm_plugin(name="funaudiochat", audio_token="<|AUDIO|>")
    print(f"[plugin] qwen3_asr.audio_token={qwen3_asr_plugin.audio_token}")
    print(f"[plugin] funaudiochat.audio_token={funaudiochat_plugin.audio_token}")


def _check_example_configs() -> None:
    configs = [
        (
            PROJECT_ROOT / "examples/qwen3_asr/qwen3_asr_sft_lora.yaml",
            {"template": "qwen3_asr", "flash_attn": "fa2", "stage": "sft"},
        ),
        (
            PROJECT_ROOT / "examples/funaudiochat/funaudiochat_s2t_sft_full.yaml",
            {"template": "funaudiochat", "flash_attn": "fa2", "stage": "sft"},
        ),
        (
            PROJECT_ROOT
            / "examples/speech_endpointing/qwen3/0_8b_base/qwen3_5_0_8b_base_speech_endpointing_lora_neat_packing_fa2.yaml",
            {
                "template": "qwen3_5_nothink",
                "flash_attn": "fa2",
                "compute_endpointing_metrics": True,
                "resize_vocab": True,
            },
        ),
    ]

    for path, expected in configs:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise RuntimeError(f"Unexpected value in {path}: {key}={cfg.get(key)!r}, expected {value!r}")
        print(f"[config] {path.relative_to(PROJECT_ROOT)}=ok")


def _check_qwen3_asr_model(path: Path) -> None:
    register_qwen3_asr()
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if getattr(cfg, "model_type", None) != "qwen3_asr":
        raise RuntimeError(f"Unexpected qwen3_asr model_type: {getattr(cfg, 'model_type', None)!r}")
    print(f"[model] qwen3_asr config={cfg.model_type} processor={processor.__class__.__name__}")


def _check_funaudiochat_model(path: Path) -> None:
    register_funaudiochat()
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if getattr(cfg, "model_type", None) != "funaudiochat":
        raise RuntimeError(f"Unexpected funaudiochat model_type: {getattr(cfg, 'model_type', None)!r}")
    print(f"[model] funaudiochat config={cfg.model_type} processor={processor.__class__.__name__}")


def _check_qwen3_5_endpointing_adapter(path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    peft_config = PeftConfig.from_pretrained(path)
    label_tokens = ENDPOINTING_LABELS.intersection(set(tokenizer.get_vocab().keys()))
    if label_tokens != ENDPOINTING_LABELS:
        raise RuntimeError(f"Missing endpointing label tokens in tokenizer: {sorted(ENDPOINTING_LABELS - label_tokens)}")
    print(f"[adapter] qwen3.5 base_model={peft_config.base_model_name_or_path}")
    print(f"[adapter] endpointing_labels={sorted(label_tokens)}")


def _existing_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the speech training Docker image.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available.")
    parser.add_argument("--require-fa2", action="store_true", help="Fail if FlashAttention-2 is not available.")
    parser.add_argument("--qwen3-asr-model", type=str, help="Optional local Qwen3-ASR model path.")
    parser.add_argument("--funaudiochat-model", type=str, help="Optional local FunAudioChat model path.")
    parser.add_argument(
        "--qwen3-5-endpointing-adapter",
        type=str,
        help="Optional local Qwen3.5 speech-endpointing adapter path.",
    )
    args = parser.parse_args()

    _check_audio_stack()
    _check_gpu(require_cuda=args.require_cuda, require_fa2=args.require_fa2)
    _check_templates_and_plugins()
    _check_example_configs()

    qwen3_asr_model = _existing_path(args.qwen3_asr_model)
    funaudiochat_model = _existing_path(args.funaudiochat_model)
    qwen3_5_adapter = _existing_path(args.qwen3_5_endpointing_adapter)

    if qwen3_asr_model is not None:
        _check_qwen3_asr_model(qwen3_asr_model)
    if funaudiochat_model is not None:
        _check_funaudiochat_model(funaudiochat_model)
    if qwen3_5_adapter is not None:
        _check_qwen3_5_endpointing_adapter(qwen3_5_adapter)

    print("[result] speech stack smoke test passed")


if __name__ == "__main__":
    main()
