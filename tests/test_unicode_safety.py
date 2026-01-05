# Copyright 2025 the LlamaFactory team.
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

from __future__ import annotations

import pathlib
import unicodedata

import pytest


def _is_suspicious(ch: str) -> bool:
    """Return True for Unicode chars that are typically invisible/dangerous in source code."""
    cat = unicodedata.category(ch)
    if cat == "Cf":
        return True

    if ord(ch) > 0x7F and ch.isspace():
        return True

    # Line/paragraph separators (not ASCII, can appear invisible in some editors).
    if ch in ("\u2028", "\u2029"):
        return True

    return False


@pytest.mark.runs_on(["cpu"])
def test_no_suspicious_unicode_in_source():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "llamafactory"
    assert src_root.exists()

    findings: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="strict")
        line = 1
        col = 0
        for ch in text:
            if ch == "\n":
                line += 1
                col = 0
                continue
            col += 1
            if _is_suspicious(ch):
                code = f"U+{ord(ch):04X}"
                name = unicodedata.name(ch, "UNKNOWN")
                rel = path.relative_to(repo_root)
                findings.append(f"{rel}:{line}:{col}: {code} {name}")

    assert not findings, "Found suspicious unicode characters:\n" + "\n".join(findings[:200])
