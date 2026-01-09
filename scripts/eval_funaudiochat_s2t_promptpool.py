#!/usr/bin/env python3
"""Batch ASR evaluation for FunAudioChat (S2T) using prompt_pool (language hint + normalized prompt).

What it does:
1) Runs LLaMA-Factory `do_predict=true predict_with_generate=true` on a fixed suite of Hausa/Swahili testsets.
2) Computes *normalized* WER/WERE using `~/projects/speech_related_tools/evaluate/eval_asr_wer_cer.py`.

Notes:
- `dynamic_prompt_sampling=true` is required so prompt_pool top1 entry is appended to the system prompt
  during evaluation tokenization (aligned with training promptpool behavior).
- This script is meant to be run inside the `llamafactory` conda env.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalJob:
    key: str
    lang: str
    dataset: str
    pretty: str


DEFAULT_JOBS: dict[str, EvalJob] = {
    # Hausa
    "hausa_youtube": EvalJob(
        key="hausa_youtube",
        lang="hausa",
        dataset="funaudiochat_asr_hausa_youtube_test_norm_text_promptpool",
        pretty="hausa_youtube_test",
    ),
    "hausa_fleurs": EvalJob(
        key="hausa_fleurs",
        lang="hausa",
        dataset="funaudiochat_asr_hausa_fleurs_test_norm_text_promptpool",
        pretty="hausa_fleurs_test",
    ),
    "hausa_haiwa": EvalJob(
        key="hausa_haiwa",
        lang="hausa",
        dataset="funaudiochat_asr_hausa_haiwa_test_norm_text_promptpool",
        pretty="hausa_haiwa_test",
    ),
    "hausa_return_data": EvalJob(
        key="hausa_return_data",
        lang="hausa",
        dataset="funaudiochat_asr_hausa_return_data_test_norm_text_promptpool",
        pretty="hausa_return_data_test",
    ),
    # Swahili
    "swahili_fleurs": EvalJob(
        key="swahili_fleurs",
        lang="swahili",
        dataset="funaudiochat_asr_swahili_fleurs_test_norm_text_promptpool",
        pretty="swahili_fleurs_test",
    ),
    "swahili_kaldi": EvalJob(
        key="swahili_kaldi",
        lang="swahili",
        dataset="funaudiochat_asr_swahili_kaldi_test_norm_text_promptpool",
        pretty="swahili_kaldi_test",
    ),
    "swahili_king_asr": EvalJob(
        key="swahili_king_asr",
        lang="swahili",
        dataset="funaudiochat_asr_swahili_king_asr_test_norm_text_promptpool",
        pretty="swahili_king_asr_test",
    ),
    "swahili_return_data": EvalJob(
        key="swahili_return_data",
        lang="swahili",
        dataset="funaudiochat_asr_swahili_return_data_test_norm_text_promptpool",
        pretty="swahili_return_data_test",
    ),
}


def _csv(value: str) -> list[str]:
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def _sanitize_tag(text: str) -> str:
    text = text.strip().replace("/", "--")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:120] if len(text) > 120 else text


def _infer_model_tag(path_or_id: str) -> str:
    base = path_or_id.rstrip("/").split("/")[-1] or path_or_id
    m = re.fullmatch(r"checkpoint-(\d+)", base)
    if m:
        return f"ckpt-{m.group(1)}"
    return _sanitize_tag(base)


def _looks_like_lora_adapter(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_config.json").exists()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_env(gpus: str, nproc: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["FORCE_TORCHRUN"] = "1"
    env["NPROC_PER_NODE"] = str(nproc)
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("DISABLE_VERSION_CHECK", "1")
    env.setdefault("PYTHONNOUSERSITE", "1")
    src_dir = _repo_root() / "src"
    env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return env


def _run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join([cmd[0], *[shlex(c) for c in cmd[1:]]])
    print(f"\n[cmd] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def shlex(s: str) -> str:
    return subprocess.list2cmdline([s])


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _concat_jsonl(src_files: Iterable[Path], dst_file: Path) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with dst_file.open("w", encoding="utf-8") as out:
        for p in src_files:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else (line + "\n"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval_funaudiochat_s2t_promptpool.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model path/id. If it's a LoRA checkpoint dir (adapter_model.safetensors), it will be treated as adapter.",
    )
    p.add_argument(
        "--as-adapter",
        action="store_true",
        help="Force treat --model as LoRA adapter and load it on top of --base-model.",
    )
    p.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL", "FunAudioLLM/Fun-Audio-Chat-8B"),
        help="Base model for LoRA adapter evaluation.",
    )
    p.add_argument(
        "--config",
        default=str(_repo_root() / "examples/funaudiochat/funaudiochat_s2t_sft_full.yaml"),
        help="Base YAML config to load; will be overridden to predict-only.",
    )
    p.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR", "/data2/mayufeng/manifests/llama_data"),
        help="Directory containing dataset_info.json for the eval datasets.",
    )
    p.add_argument(
        "--out-root",
        default=os.environ.get("OUT_ROOT", "/data2/mayufeng/llamafactory_eval/funaudiochat"),
        help="Where to write evaluation outputs.",
    )
    p.add_argument("--gpus", default=os.environ.get("GPUS", "6,7"), help="CUDA_VISIBLE_DEVICES list, e.g. '6,7'.")
    p.add_argument(
        "--nproc",
        type=int,
        default=0,
        help="NPROC_PER_NODE; default is len(--gpus). Keep <= visible GPU count.",
    )
    p.add_argument("--jobs", default="", help=f"Comma-separated job keys. Available: {','.join(DEFAULT_JOBS.keys())}")
    p.add_argument("--langs", default="hausa,swahili", help="Comma-separated langs to run when --jobs not set.")
    p.add_argument("--run-id", default="", help="Run id suffix (default: current timestamp).")

    # Inference / data knobs
    p.add_argument("--cutoff-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--audio-padding", default="longest", choices=["longest", "max_length"])
    p.add_argument("--bf16", default="true", choices=["true", "false"])
    p.add_argument("--flash-attn", default="fa2", choices=["auto", "disabled", "sdpa", "fa2"])
    p.add_argument("--overwrite-cache", default="false", choices=["true", "false"])
    p.add_argument("--preprocessing-num-workers", type=int, default=8)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0, help="Debug: truncate dataset to N samples.")

    # Normalized WER/WERE
    p.add_argument(
        "--evaluator",
        default=os.environ.get(
            "ASR_EVAL_PY", os.path.expanduser("~/projects/speech_related_tools/evaluate/eval_asr_wer_cer.py")
        ),
        help="Path to eval_asr_wer_cer.py",
    )
    p.add_argument("--nj", type=int, default=8, help="Parallelism for evaluator.")
    p.add_argument("--skip-normalized-metrics", action="store_true")
    p.add_argument("--no-combine", action="store_true", help="Skip per-language combined WER/WERE.")

    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    model_arg = args.model
    model_path = Path(model_arg).expanduser()
    treat_as_adapter = args.as_adapter or (model_path.exists() and _looks_like_lora_adapter(model_path))

    if not Path(args.config).expanduser().exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    if not Path(args.dataset_dir).expanduser().exists():
        raise FileNotFoundError(f"dataset_dir not found: {args.dataset_dir}")

    run_id = args.run_id.strip() or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _infer_model_tag(str(model_path if model_path.exists() else model_arg))
    batch_dir = Path(args.out_root).expanduser() / f"batch_{model_tag}_promptpool_{run_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    gpu_list = _csv(args.gpus)
    nproc = args.nproc if args.nproc and args.nproc > 0 else max(1, len(gpu_list))
    env = _build_env(args.gpus, nproc=nproc)

    if args.jobs.strip():
        job_keys = _csv(args.jobs)
    else:
        allowed_langs = set(_csv(args.langs))
        job_keys = [k for k, job in DEFAULT_JOBS.items() if job.lang in allowed_langs]

    unknown = [k for k in job_keys if k not in DEFAULT_JOBS]
    if unknown:
        raise ValueError(f"Unknown job keys: {unknown}. Available: {sorted(DEFAULT_JOBS.keys())}")

    predictor = [
        sys.executable,
        "-m",
        "llamafactory.cli",
        "train",
        str(Path(args.config).expanduser()),
        # predict-only
        "stage=sft",
        "template=funaudiochat",
        "do_train=false",
        "do_eval=false",
        "do_predict=true",
        "predict_with_generate=true",
        "compute_wer_cer=true",
        "report_to=none",
        # IMPORTANT: don't load train dataset
        "dataset=null",
        # promptpool alignment
        "dynamic_prompt_sampling=true",
        # data/infer knobs
        f"dataset_dir={str(Path(args.dataset_dir).expanduser())}",
        f"cutoff_len={args.cutoff_len}",
        f"audio_padding={args.audio_padding}",
        "audio_sampling_rate=16000",
        f"per_device_eval_batch_size={args.per_device_eval_batch_size}",
        "do_sample=false",
        "temperature=0.0",
        "top_p=1.0",
        "num_beams=1",
        f"max_new_tokens={args.max_new_tokens}",
        f"overwrite_cache={args.overwrite_cache}",
        f"preprocessing_num_workers={args.preprocessing_num_workers}",
        f"dataloader_num_workers={args.dataloader_num_workers}",
        f"bf16={args.bf16}",
        f"flash_attn={args.flash_attn}",
        # for LoRA checkpoints
        "finetuning_type=lora",
    ]

    if args.max_samples and args.max_samples > 0:
        predictor.append(f"max_samples={args.max_samples}")

    if treat_as_adapter:
        predictor.append(f"model_name_or_path={args.base_model}")
        predictor.append(f"adapter_name_or_path={str(model_path)}")
    else:
        predictor.append(f"model_name_or_path={model_arg}")

    evaluator_path = Path(args.evaluator).expanduser()
    if not args.skip_normalized_metrics and not evaluator_path.exists():
        raise FileNotFoundError(f"Evaluator not found: {evaluator_path}")

    results: dict[str, dict] = {
        "run_id": run_id,
        "batch_dir": str(batch_dir),
        "model_tag": model_tag,
        "treat_as_adapter": treat_as_adapter,
        "base_model": args.base_model if treat_as_adapter else None,
        "model": model_arg,
        "gpus": args.gpus,
        "nproc": nproc,
        "jobs": {},
        "combined": {},
    }

    pred_files_by_lang: dict[str, list[Path]] = {}

    for key in job_keys:
        job = DEFAULT_JOBS[key]
        out_dir = batch_dir / job.pretty
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = predictor + [f"eval_dataset={job.dataset}", f"output_dir={str(out_dir)}", "overwrite_output_dir=true"]
        print(f"\n=== [{job.lang}] {job.key}: {job.dataset} ===", flush=True)
        _run(cmd, env=env, dry_run=args.dry_run)

        pred_file = out_dir / "generated_predictions.jsonl"
        if not args.dry_run and not pred_file.exists():
            raise FileNotFoundError(f"Missing predictions: {pred_file}")

        pred_files_by_lang.setdefault(job.lang, []).append(pred_file)

        report_path = out_dir / "normalized_wer_were_eval.json"
        if not args.skip_normalized_metrics:
            eval_cmd = [
                sys.executable,
                str(evaluator_path),
                "--pairs",
                str(pred_file),
                "--ref-field",
                "label",
                "--hyp-field",
                "predict",
                "--lang-normalize",
                "--lang-fixed",
                job.lang,
                "--nj",
                str(args.nj),
                "--report",
                str(report_path),
            ]
            _run(eval_cmd, env=env, dry_run=args.dry_run)

        job_result = {
            "lang": job.lang,
            "dataset": job.dataset,
            "out_dir": str(out_dir),
        }
        if not args.dry_run and report_path.exists():
            metrics = _load_json(report_path)
            job_result.update(
                {
                    "wer": metrics.get("wer"),
                    "were": metrics.get("were"),
                    "cer": metrics.get("cer"),
                    "ref_words": metrics.get("ref_words"),
                }
            )
        results["jobs"][job.key] = job_result

    if not args.no_combine and not args.dry_run and not args.skip_normalized_metrics:
        for lang, files in pred_files_by_lang.items():
            if not files:
                continue
            combined_dir = batch_dir / f"{lang}_all"
            combined_dir.mkdir(parents=True, exist_ok=True)
            combined_preds = combined_dir / "generated_predictions.jsonl"
            _concat_jsonl(files, combined_preds)
            combined_report = combined_dir / "normalized_wer_were_eval.json"
            eval_cmd = [
                sys.executable,
                str(evaluator_path),
                "--pairs",
                str(combined_preds),
                "--ref-field",
                "label",
                "--hyp-field",
                "predict",
                "--lang-normalize",
                "--lang-fixed",
                lang,
                "--nj",
                str(args.nj),
                "--report",
                str(combined_report),
            ]
            print(f"\n=== [combined] {lang} ===", flush=True)
            _run(eval_cmd, env=env, dry_run=args.dry_run)
            metrics = _load_json(combined_report)
            results["combined"][lang] = {
                "out_dir": str(combined_dir),
                "wer": metrics.get("wer"),
                "were": metrics.get("were"),
                "cer": metrics.get("cer"),
                "ref_words": metrics.get("ref_words"),
            }

    _write_json(batch_dir / "summary.json", results)
    print(f"\n[done] Wrote: {batch_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
