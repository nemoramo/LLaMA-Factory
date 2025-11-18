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

"""Tests for WER/CER metric calculations."""

import pytest

from llamafactory.train.sft.metric import _compute_error_rate


class TestComputeErrorRate:
    """Test the _compute_error_rate function used for WER/CER."""

    def test_identical_sequences(self):
        """Test with identical reference and hypothesis."""
        reference = ["hello", "world"]
        hypothesis = ["hello", "world"]
        error_rate = _compute_error_rate(reference, hypothesis)
        assert error_rate == 0.0, "Identical sequences should have 0% error rate"

    def test_completely_different_sequences(self):
        """Test with completely different sequences."""
        reference = ["hello", "world"]
        hypothesis = ["foo", "bar"]
        error_rate = _compute_error_rate(reference, hypothesis)
        assert error_rate == 1.0, "Completely different sequences should have 100% error rate"

    def test_empty_hypothesis(self):
        """Test with empty hypothesis (prediction)."""
        reference = ["hello", "world"]
        hypothesis = []
        error_rate = _compute_error_rate(reference, hypothesis)
        # Empty hypothesis means all words in reference are errors
        # Error rate = len(reference) / len(reference) = 100%
        assert error_rate == 1.0, "Empty hypothesis should give 100% error rate"

    def test_empty_reference(self):
        """Test with empty reference (ground truth)."""
        reference = []
        hypothesis = ["hello", "world"]
        error_rate = _compute_error_rate(reference, hypothesis)
        # Special case: empty reference with non-empty hypothesis returns 1.0
        assert error_rate == 1.0, "Empty reference with non-empty hypothesis should return 100%"

    def test_both_empty(self):
        """Test with both reference and hypothesis empty."""
        reference = []
        hypothesis = []
        error_rate = _compute_error_rate(reference, hypothesis)
        # Special case: both empty returns 0.0
        assert error_rate == 0.0, "Both empty should return 0%"

    def test_partial_match(self):
        """Test with partial match."""
        reference = ["hello", "world"]
        hypothesis = ["hello", "universe"]
        error_rate = _compute_error_rate(reference, hypothesis)
        # One substitution out of 2 words = 50% error rate
        assert error_rate == 0.5, f"Expected 50% error rate, got {error_rate * 100}%"

    def test_insertion(self):
        """Test with insertions."""
        reference = ["hello"]
        hypothesis = ["hello", "world"]
        error_rate = _compute_error_rate(reference, hypothesis)
        # One insertion: edit distance = 1, reference length = 1
        # Error rate = 1/1 = 100%
        assert error_rate == 1.0, f"Expected 100% error rate for insertion, got {error_rate * 100}%"

    def test_deletion(self):
        """Test with deletions."""
        reference = ["hello", "world"]
        hypothesis = ["hello"]
        error_rate = _compute_error_rate(reference, hypothesis)
        # One deletion: edit distance = 1, reference length = 2
        # Error rate = 1/2 = 50%
        assert error_rate == 0.5, f"Expected 50% error rate for deletion, got {error_rate * 100}%"

    def test_character_level(self):
        """Test with character-level sequences (for CER)."""
        reference = list("hello")
        hypothesis = list("hallo")
        error_rate = _compute_error_rate(reference, hypothesis)
        # One substitution (e->a) out of 5 characters = 20% error rate
        assert error_rate == 0.2, f"Expected 20% error rate, got {error_rate * 100}%"

    def test_parameter_order_matters(self):
        """Test that parameter order affects the result when lengths differ."""
        short = ["hi"]
        long = ["hello", "world"]
        
        # Correct order: reference (ground truth) first, hypothesis (prediction) second
        # If reference is long and hypothesis is short: error_rate = edit_distance / len(reference)
        error_rate_1 = _compute_error_rate(long, short)
        
        # Swapped order: hypothesis first, reference second
        # If reference is short and hypothesis is long: error_rate = edit_distance / len(reference)
        error_rate_2 = _compute_error_rate(short, long)
        
        # These should be different!
        assert error_rate_1 != error_rate_2, \
            f"Parameter order should matter! Got {error_rate_1} vs {error_rate_2}"
        
        # error_rate_1 should be higher (dividing by smaller number)
        assert error_rate_2 > error_rate_1, \
            f"Swapped parameters should give higher error rate when ref is shorter"
