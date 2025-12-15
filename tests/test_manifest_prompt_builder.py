# Copyright 2025 the LlamaFactory team.
# Additional author: ramos.ma (GitHub: nemoramo).
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

from tools.manifest_prompt_builder import (
    _has_digits,
    _normalize_record,
    build_dataset,
    build_pool_dataset,
)


def test_normalize_record_handles_missing_keys_and_strings():
    record = {"original_text": "hello", "has_digits": "false"}
    normalized = _normalize_record(record, default_lang="unk")
    assert normalized["text"] == "hello"
    assert normalized["lang"] == "unk"
    assert normalized["has_digits"] is False
    assert normalized["input_text"] == "hello"


def test_digit_detection_supports_unicode_digits():
    assert _has_digits("call me at ١٢٣") is True  # Eastern Arabic digits
    assert _has_digits("no digits here") is False


def test_build_dataset_skips_bad_records_and_formats_prompts():
    templates = ["lang={lang} digits={has_digits} src={input_text}"]
    records = [
        {"original_text": "call me at ١٢٣", "text": "Call me at 123", "lang": "en"},
        {"text": "Bonjour"},  # missing lang/has_digits/original_text
        {"original_text": "only original"},  # missing text; should fall back
        {"foo": "bar"},  # missing required fields; should be skipped
    ]

    samples = list(build_dataset(records, templates, seed=0, default_lang="unk"))

    assert len(samples) == 3

    # sample 0 keeps has_digits=True and lang
    assert samples[0].metadata["has_digits"] is True
    assert "lang=en" in samples[0].prompt

    # sample 1 uses defaults and falls back to provided text
    assert samples[1].metadata["lang"] == "unk"
    assert samples[1].metadata["has_digits"] is False
    assert "Bonjour" in samples[1].prompt

    # sample 2 falls back to original_text for completion
    assert samples[2].completion == "only original"
    assert "only original" in samples[2].prompt


def test_manifest_prompt_and_completion_bypass_template():
    templates: list[str] = []  # no templates needed when prompt exists
    records = [
        {"prompt": "Q: hi?", "completion": "A: hello", "lang": "en", "has_digits": False},
    ]

    samples = list(build_dataset(records, templates, seed=0, default_lang="unk"))
    assert len(samples) == 1
    assert samples[0].prompt == "Q: hi?"
    assert samples[0].completion == "A: hello"
    assert samples[0].metadata["prompt_source"] == "manifest"


def test_pool_mode_emits_prompt_pool_and_original_mass():
    templates = ["NORM lang={lang} text={input_text}"]
    original_templates = ["ORIG lang={lang} text={input_text}"]

    records = [
        {"text": "Normalized", "original_text": "Original", "lang": "en"},
    ]

    out = list(
        build_pool_dataset(
            records,
            templates,
            original_templates,
            seed=0,
            default_lang="unk",
            max_samples=None,
            original_prob=0.3,
        )
    )
    assert len(out) == 1
    rec = out[0]
    assert "prompt_pool" in rec
    pool = rec["prompt_pool"]
    assert len(pool) >= 2

    completions = {p["completion"] for p in pool}
    assert "Normalized" in completions
    assert "Original" in completions

    total_w = sum(float(p.get("weight", 0.0)) for p in pool)
    assert abs(total_w - 1.0) < 1e-6
