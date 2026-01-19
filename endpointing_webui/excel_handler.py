#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel import/export handler for Endpointing WebUI.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def parse_simple_history_format(text: str) -> Optional[str]:
    """
    Parse simple text format history to JSON.

    Supported formats:
    - "A: Hello | U: Hi there" (pipe separated)
    - "A: Hello\\nU: Hi there" (newline separated)
    - "assistant: Hello | user: Hi" (full role names)

    Returns JSON string or None if parsing fails.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    if text.startswith("["):
        return text

    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
    else:
        parts = [p.strip() for p in text.split("\n") if p.strip()]

    if not parts:
        return None

    result = []
    role_pattern = re.compile(r"^(A|U|assistant|user)\s*:\s*(.+)$", re.IGNORECASE)

    for part in parts:
        match = role_pattern.match(part.strip())
        if match:
            role_char = match.group(1).lower()
            content = match.group(2).strip()

            if role_char in ("a", "assistant"):
                role = "assistant"
            else:
                role = "user"

            result.append({"role": role, "text": content})

    return json.dumps(result) if result else None


class ExcelHandler:
    """Handler for Excel file import and export."""

    # Required and optional columns
    REQUIRED_COLUMNS = ["asr_text"]
    OPTIONAL_COLUMNS = ["history", "lang"]

    # Output columns
    OUTPUT_COLUMNS = [
        "label",
        "confidence",
        "p_eou",
        "p_cont_user",
        "p_unaddressed",
        "latency_ms",
        "model",
        "error",
    ]

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.results: List[Dict[str, Any]] = []
        self.column_mapping: Dict[str, str] = {}

    def load_excel(self, file_path: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Load Excel file and validate structure.

        Args:
            file_path: Path to the Excel file.

        Returns:
            Tuple of (success, message, dataframe).
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False, f"File not found: {file_path}", None

            # Read Excel file
            if path.suffix.lower() in [".xlsx", ".xls"]:
                self.df = pd.read_excel(file_path)
            elif path.suffix.lower() == ".csv":
                self.df = pd.read_csv(file_path)
            else:
                return False, f"Unsupported file format: {path.suffix}", None

            if self.df.empty:
                return False, "File is empty", None

            # Clean column names
            self.df.columns = [str(c).strip() for c in self.df.columns]

            return True, f"Loaded {len(self.df)} rows, columns: {list(self.df.columns)}", self.df

        except Exception as e:
            return False, f"Failed to load file: {str(e)}", None

    def get_columns(self) -> List[str]:
        """Get list of column names from loaded DataFrame."""
        if self.df is None:
            return []
        return list(self.df.columns)

    def set_column_mapping(
        self,
        asr_text_col: str,
        history_col: Optional[str] = None,
        lang_col: Optional[str] = None,
    ):
        """
        Set column mapping for data extraction.

        Args:
            asr_text_col: Column name for ASR text (required).
            history_col: Column name for history JSON (optional).
            lang_col: Column name for language code (optional).
        """
        self.column_mapping = {
            "asr_text": asr_text_col,
            "history": history_col,
            "lang": lang_col,
        }

    def validate_mapping(self) -> Tuple[bool, str]:
        """
        Validate the column mapping.

        Returns:
            Tuple of (success, message).
        """
        if self.df is None:
            return False, "No data loaded"

        asr_col = self.column_mapping.get("asr_text")
        if not asr_col:
            return False, "ASR text column is required"

        if asr_col not in self.df.columns:
            return False, f"ASR text column '{asr_col}' not found in data"

        # Check optional columns
        history_col = self.column_mapping.get("history")
        if history_col and history_col not in self.df.columns:
            return False, f"History column '{history_col}' not found in data"

        lang_col = self.column_mapping.get("lang")
        if lang_col and lang_col not in self.df.columns:
            return False, f"Language column '{lang_col}' not found in data"

        return True, "Column mapping is valid"

    def get_row_data(self, index: int) -> Dict[str, Any]:
        """
        Get data for a specific row using the column mapping.

        Args:
            index: Row index.

        Returns:
            Dictionary with asr_text, history, and lang.
        """
        if self.df is None or index >= len(self.df):
            return {"asr_text": "", "history": None, "lang": "en-US"}

        row = self.df.iloc[index]

        # Get ASR text
        asr_col = self.column_mapping.get("asr_text")
        asr_text = str(row[asr_col]) if asr_col and pd.notna(row.get(asr_col)) else ""

        # Get history (supports both JSON and simple text format)
        history_col = self.column_mapping.get("history")
        history = None
        if history_col and pd.notna(row.get(history_col)):
            raw_history = str(row[history_col]).strip()
            if raw_history:
                history = parse_simple_history_format(raw_history)

        # Get language
        lang_col = self.column_mapping.get("lang")
        lang = "en-US"
        if lang_col and pd.notna(row.get(lang_col)):
            lang = str(row[lang_col])

        return {
            "asr_text": asr_text,
            "history": history,
            "lang": lang,
        }

    def get_total_rows(self) -> int:
        """Get total number of rows."""
        return len(self.df) if self.df is not None else 0

    def set_result(self, index: int, result: Dict[str, Any]):
        """
        Set prediction result for a specific row.

        Args:
            index: Row index.
            result: Result dictionary from EndpointingResult.to_dict().
        """
        # Ensure results list is large enough
        while len(self.results) <= index:
            self.results.append({})

        self.results[index] = result

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get DataFrame with original data and results.

        Returns:
            Combined DataFrame.
        """
        if self.df is None:
            return pd.DataFrame()

        # Create a copy of original data
        result_df = self.df.copy()

        # Add result columns
        for col in self.OUTPUT_COLUMNS:
            result_df[col] = None

        # Fill in results
        for i, result in enumerate(self.results):
            if i < len(result_df) and result:
                for col in self.OUTPUT_COLUMNS:
                    if col in result:
                        result_df.at[i, col] = result[col]

        return result_df

    def export_excel(self, output_path: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Export results to Excel file.

        Args:
            output_path: Output file path. If None, generates a timestamped name.

        Returns:
            Tuple of (success, message, file_path).
        """
        try:
            result_df = self.get_results_dataframe()
            if result_df.empty:
                return False, "No data to export", None

            # Generate output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"endpointing_results_{timestamp}.xlsx"

            # Export to Excel
            result_df.to_excel(output_path, index=False, engine="openpyxl")

            return True, f"Exported {len(result_df)} rows to {output_path}", output_path

        except Exception as e:
            return False, f"Failed to export: {str(e)}", None

    def get_preview(self, num_rows: int = 5) -> pd.DataFrame:
        """
        Get preview of loaded data.

        Args:
            num_rows: Number of rows to preview.

        Returns:
            Preview DataFrame.
        """
        if self.df is None:
            return pd.DataFrame()

        return self.df.head(num_rows)

    def clear(self):
        """Clear loaded data and results."""
        self.df = None
        self.results = []
        self.column_mapping = {}


def create_sample_excel(output_path: str = "sample_input.xlsx"):
    """
    Create a sample Excel file for testing.

    Args:
        output_path: Output file path.
    """
    sample_data = [
        {
            "asr_text": "hello how are you",
            "history": json.dumps([]),
            "lang": "en-US",
        },
        {
            "asr_text": "I need help with",
            "history": json.dumps([{"role": "assistant", "text": "Hello! How can I help you today?"}]),
            "lang": "en-US",
        },
        {
            "asr_text": "thank you that's all",
            "history": json.dumps(
                [
                    {"role": "user", "text": "What's the weather like?"},
                    {"role": "assistant", "text": "It's sunny and warm today."},
                ]
            ),
            "lang": "en-US",
        },
        {
            "asr_text": "你好",
            "history": json.dumps([]),
            "lang": "zh-CN",
        },
        {
            "asr_text": "",
            "history": json.dumps([]),
            "lang": "en-US",
        },
    ]

    df = pd.DataFrame(sample_data)
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"Created sample file: {output_path}")


if __name__ == "__main__":
    create_sample_excel()
