# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
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

import re


def compute_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    r"""Compute normalized edit distance for sequences (WER/CER helper).

    When reference is empty, return 0.0 if hypothesis is also empty, otherwise 1.0.
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 1.0

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return float(dp[ref_len][hyp_len]) / float(ref_len)


def has_cjk(text: str) -> bool:
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def normalize_text(text: str) -> str:
    regex = r"(?<!\d)[.,;:'\"?!](?!\d)"
    text = re.sub(regex, "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
