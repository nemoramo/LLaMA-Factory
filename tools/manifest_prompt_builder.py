"""Convert ASR/text manifests into prompt/completion JSONL.

Each input JSONL line may contain:
- original_text: spoken / raw text (optional)
- text: normalized text (optional; falls back to original_text)
- prompt: pre-built prompt (optional; if present we bypass template sampling)
- completion: pre-built completion (optional; falls back to `text`/`original_text`)
- lang / language: language tag (optional, default "unknown")
- has_digits: optional bool; if missing, auto-detected via Unicode digits

Output JSONL lines contain:
- mode=sample (default):
    - prompt: formatted prompt using a randomly sampled template
    - completion: normalized text (`text` fallback to `original_text`)
    - metadata: includes lang, has_digits, original_text, text, prompt_id
- mode=pool:
    - prompt: empty base prompt
    - completion: default normalized completion
    - prompt_pool: list of candidate prompts (strings) with optional weights and completion overrides
      so that LLaMA-Factory can dynamically sample at training time.

Usage example:
python tools/manifest_prompt_builder.py \
  --manifest data.jsonl \
  --output out.jsonl \
  --template "语言={lang} 数字={has_digits} 文本：{input_text}" \
  --template "Normalize (lang={lang}, digits={has_digits}): {input_text}" \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES: list[str] = [
    "Normalize the following transcript into clean written form. Language={lang}. Text: {input_text}",
    "Convert spoken-form text to a normalized written transcript. (lang={lang}, digits={has_digits}) Input: {input_text}",
    "Please normalize this ASR transcript while preserving meaning. Use standard Arabic numerals if applicable: {input_text}",
    "ITN task: produce a normalized written version of the raw transcript. Language={lang}. Input: {input_text}",
]

DEFAULT_ORIGINAL_TEMPLATES: list[str] = [
    "Return the raw transcript exactly as spoken, preserving casing and punctuation. Language={lang}. Text: {input_text}",
    "Verbatim transcription requested (keep original formatting). Language={lang}. Input: {input_text}",
]


@dataclass
class Sample:
    prompt: str
    completion: str
    metadata: Mapping[str, Any]
    prompt_id: int


def _iter_jsonl(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skip %s:%s (malformed JSON)", path, lineno)


def _has_digits(text: str | None) -> bool:
    if not text:
        return False
    return any(ch.isdigit() for ch in text)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _normalize_record(raw: Mapping[str, Any], *, default_lang: str = "unknown") -> dict[str, Any] | None:
    original_text = raw.get("original_text") or raw.get("text")
    normalized_text = raw.get("text") or original_text

    if not original_text and not normalized_text and not raw.get("completion"):
        return None

    prompt_text = raw.get("prompt") or None
    completion_text = raw.get("completion") or normalized_text or original_text
    lang = raw.get("lang") or raw.get("language") or default_lang
    has_digits = raw.get("has_digits")
    if has_digits is None:
        has_digits = _has_digits(original_text) or _has_digits(normalized_text) or _has_digits(completion_text)
    else:
        has_digits = _coerce_bool(has_digits)

    input_text = original_text or normalized_text or completion_text or ""
    normalized = {
        "original_text": original_text or "",
        "text": normalized_text or "",
        "prompt": prompt_text,
        "completion": completion_text or "",
        "input_text": input_text,
        "lang": lang,
        "has_digits": bool(has_digits),
    }
    return normalized


def _choose_template(templates: Sequence[str], rng: random.Random) -> tuple[int, str]:
    if not templates:
        raise ValueError("No prompt templates provided.")
    idx = rng.randrange(len(templates))
    return idx, templates[idx]


def build_sample(
    record: Mapping[str, Any],
    templates: Sequence[str],
    rng: random.Random | None = None,
    *,
    default_lang: str = "unknown",
) -> Sample | None:
    rng = rng or random
    normalized = _normalize_record(record, default_lang=default_lang)
    if normalized is None:
        return None

    prompt_source = "template"
    prompt_id = -1
    if normalized.get("prompt"):
        prompt = normalized["prompt"]
        prompt_source = "manifest"
    else:
        if not templates:
            logger.warning("No templates provided and record has no prompt; record skipped.")
            return None
        prompt_id, template = _choose_template(templates, rng)
        try:
            prompt = template.format(**normalized)
        except KeyError as exc:
            logger.warning("Template missing key %s; record skipped", exc)
            return None

    completion = normalized["completion"] or normalized["text"]
    metadata = {
        "lang": normalized["lang"],
        "has_digits": normalized["has_digits"],
        "original_text": normalized["original_text"],
        "text": normalized["text"],
        "prompt_source": prompt_source,
        "prompt_id": prompt_id,
    }
    return Sample(prompt=prompt, completion=completion, metadata=metadata, prompt_id=prompt_id)


def build_dataset(
    records: Iterable[Mapping[str, Any]],
    templates: Sequence[str],
    *,
    seed: int | None = None,
    default_lang: str = "unknown",
    max_samples: int | None = None,
) -> Iterator[Sample]:
    rng = random.Random(seed)
    for record in records:
        sample = build_sample(record, templates, rng, default_lang=default_lang)
        if sample is None:
            continue
        yield sample
        if max_samples is not None:
            max_samples -= 1
            if max_samples <= 0:
                break


def _load_templates(args_templates: list[str] | None, templates_file: str | None) -> list[str]:
    templates: list[str] = []
    if templates_file:
        for line in Path(templates_file).read_text(encoding="utf-8").splitlines():
            template = line.strip()
            if template:
                templates.append(template)
    if args_templates:
        templates.extend(args_templates)
    if not templates:
        templates = DEFAULT_TEMPLATES.copy()
    return templates


def _write_jsonl(samples: Iterable[Sample], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            record = {
                "prompt": sample.prompt,
                "completion": sample.completion,
                "metadata": sample.metadata,
                "prompt_id": sample.prompt_id,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def _write_jsonl_pool(records: Iterable[Mapping[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def _load_original_templates(
    args_templates: list[str] | None, templates_file: str | None
) -> list[str]:
    templates: list[str] = []
    if templates_file:
        for line in Path(templates_file).read_text(encoding="utf-8").splitlines():
            template = line.strip()
            if template:
                templates.append(template)
    if args_templates:
        templates.extend(args_templates)
    if not templates:
        templates = DEFAULT_ORIGINAL_TEMPLATES.copy()
    return templates


def build_pool_record(
    record: Mapping[str, Any],
    templates: Sequence[str],
    original_templates: Sequence[str],
    rng: random.Random,
    *,
    default_lang: str = "unknown",
    original_prob: float = 0.2,
) -> dict[str, Any] | None:
    normalized = _normalize_record(record, default_lang=default_lang)
    if normalized is None:
        return None

    norm_completion = normalized["completion"] or normalized["text"] or normalized["original_text"]
    orig_completion = normalized["original_text"] or norm_completion

    candidates_norm: list[str] = []
    candidates_orig: list[str] = []

    # If manifest already provides a prompt, include it as one candidate.
    if normalized.get("prompt"):
        candidates_norm.append(str(normalized["prompt"]))

    def try_format(template: str) -> str | None:
        try:
            return template.format(**normalized)
        except KeyError as exc:
            logger.warning("Template missing key %s; skipped.", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Template format error: %s; skipped.", exc)
        return None

    for t in templates:
        formatted = try_format(t)
        if formatted:
            candidates_norm.append(formatted)

    for t in original_templates:
        formatted = try_format(t)
        if formatted:
            candidates_orig.append(formatted)

    # De-dup while keeping order.
    candidates_norm = list(dict.fromkeys(candidates_norm))
    candidates_orig = list(dict.fromkeys(candidates_orig))

    if not candidates_norm and not candidates_orig:
        return None

    p_orig = min(max(float(original_prob), 0.0), 1.0)
    if not candidates_orig or orig_completion == norm_completion or p_orig <= 0:
        orig_total = 0.0
        norm_total = 1.0
    else:
        orig_total = p_orig
        norm_total = 1.0 - p_orig

    prompt_pool: list[dict[str, Any]] = []

    if candidates_norm and norm_total > 0:
        w_each = norm_total / len(candidates_norm)
        for c in candidates_norm:
            prompt_pool.append({"text": c, "completion": norm_completion, "weight": w_each})

    if candidates_orig and orig_total > 0:
        w_each = orig_total / len(candidates_orig)
        for c in candidates_orig:
            prompt_pool.append({"text": c, "completion": orig_completion, "weight": w_each})

    metadata = {
        "lang": normalized["lang"],
        "has_digits": normalized["has_digits"],
        "original_text": normalized["original_text"],
        "text": normalized["text"],
        "prompt_source": "pool",
        "num_prompts": len(prompt_pool),
        "original_prob": orig_total,
    }

    return {
        "prompt": "",
        "completion": norm_completion,
        "prompt_pool": prompt_pool,
        "metadata": metadata,
        "prompt_id": -1,
    }


def build_pool_dataset(
    records: Iterable[Mapping[str, Any]],
    templates: Sequence[str],
    original_templates: Sequence[str],
    *,
    seed: int | None = None,
    default_lang: str = "unknown",
    max_samples: int | None = None,
    original_prob: float = 0.2,
) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    for record in records:
        sample = build_pool_record(
            record,
            templates,
            original_templates,
            rng,
            default_lang=default_lang,
            original_prob=original_prob,
        )
        if sample is None:
            continue
        yield sample
        if max_samples is not None:
            max_samples -= 1
            if max_samples <= 0:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt/completion JSONL from ASR/text manifests.")
    parser.add_argument("--manifest", required=True, nargs="+", help="Input manifest jsonl files.")
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    parser.add_argument(
        "--mode",
        choices=["sample", "pool"],
        default="sample",
        help="sample: emit single prompt/completion; pool: emit prompt_pool for dynamic sampling.",
    )
    parser.add_argument("--template", action="append", dest="templates", help="Extra prompt template (can repeat).")
    parser.add_argument("--templates-file", help="File with one prompt template per line.")
    parser.add_argument(
        "--original-template",
        action="append",
        dest="original_templates",
        help="Extra prompt template for original_text target (can repeat).",
    )
    parser.add_argument("--original-templates-file", help="File with one original prompt template per line.")
    parser.add_argument(
        "--original-prob",
        type=float,
        default=0.2,
        help="In pool mode, probability mass assigned to original_text target prompts.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for template sampling.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on emitted samples.")
    parser.add_argument("--default-lang", default="unknown", help="Fallback language tag when missing.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    manifest_paths = [Path(p) for p in args.manifest]
    templates = _load_templates(args.templates, args.templates_file)
    original_templates = _load_original_templates(args.original_templates, args.original_templates_file)

    logger.info(
        "Mode=%s, using %d norm templates, %d original templates; output -> %s",
        args.mode,
        len(templates),
        len(original_templates),
        args.output,
    )

    records = _iter_jsonl(manifest_paths)

    if args.mode == "pool":
        pool_records = build_pool_dataset(
            records,
            templates,
            original_templates,
            seed=args.seed,
            default_lang=args.default_lang,
            max_samples=args.max_samples,
            original_prob=args.original_prob,
        )
        written = _write_jsonl_pool(pool_records, Path(args.output))
    else:
        samples = build_dataset(
            records,
            templates,
            seed=args.seed,
            default_lang=args.default_lang,
            max_samples=args.max_samples,
        )
        written = _write_jsonl(samples, Path(args.output))

    logger.info("Done. Wrote %d samples.", written)


if __name__ == "__main__":
    main()
