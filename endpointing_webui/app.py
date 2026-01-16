#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Endpointing WebUI - Gradio-based annotation interface for speech endpointing.

Features:
- Batch import Excel files with ASR text and conversation history
- Call gRPC service to get predictions with probabilities
- Export results to Excel
"""

import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from excel_handler import ExcelHandler
from grpc_client import EndpointingClient


# Global state
excel_handler = ExcelHandler()
grpc_client: Optional[EndpointingClient] = None


def connect_grpc(host: str, port: int) -> str:
    """Connect to gRPC server."""
    global grpc_client

    try:
        port = int(port)
        if grpc_client:
            grpc_client.close()

        grpc_client = EndpointingClient(host=host, port=port)
        success, message = grpc_client.test_connection()

        if success:
            return f"✅ {message}"
        else:
            return f"❌ {message}"
    except Exception as e:
        return f"❌ Connection failed: {str(e)}"


def load_excel(file) -> tuple:
    """Load Excel file and return preview."""
    global excel_handler

    if file is None:
        return "❌ No file uploaded", None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])

    excel_handler = ExcelHandler()
    success, message, df = excel_handler.load_excel(file.name)

    if not success:
        return f"❌ {message}", None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])

    columns = excel_handler.get_columns()
    preview = excel_handler.get_preview(10)

    # Auto-detect columns
    asr_default = None
    history_default = None
    lang_default = None

    for col in columns:
        col_lower = col.lower()
        if "asr" in col_lower or "text" in col_lower:
            asr_default = col
        elif "history" in col_lower or "hist" in col_lower:
            history_default = col
        elif "lang" in col_lower or "language" in col_lower:
            lang_default = col

    return (
        f"✅ {message}",
        preview,
        gr.update(choices=columns, value=asr_default),
        gr.update(choices=["(None)"] + columns, value=history_default or "(None)"),
        gr.update(choices=["(None)"] + columns, value=lang_default or "(None)"),
    )


def run_inference(
    asr_col: str,
    history_col: str,
    lang_col: str,
    eou_threshold: float,
    treat_unaddressed: bool,
    concurrency: int,
    progress=gr.Progress(),
) -> tuple:
    """Run batch inference with progress tracking."""
    global excel_handler, grpc_client

    if grpc_client is None:
        return "❌ Not connected to gRPC server", None

    if excel_handler.df is None:
        return "❌ No data loaded", None

    # Set column mapping
    history_col_actual = None if history_col == "(None)" else history_col
    lang_col_actual = None if lang_col == "(None)" else lang_col

    excel_handler.set_column_mapping(
        asr_text_col=asr_col,
        history_col=history_col_actual,
        lang_col=lang_col_actual,
    )

    valid, msg = excel_handler.validate_mapping()
    if not valid:
        return f"❌ {msg}", None

    total = excel_handler.get_total_rows()
    completed = 0
    errors = 0

    # Clear previous results
    excel_handler.results = []

    def process_row(index: int):
        row_data = excel_handler.get_row_data(index)
        result = grpc_client.predict(
            asr_text=row_data["asr_text"],
            history_json=row_data["history"],
            lang=row_data["lang"],
            eou_threshold=eou_threshold,
            treat_unaddressed_as_eou=treat_unaddressed,
            request_id=f"webui-{uuid.uuid4().hex[:8]}",
        )
        return index, result

    # Run with thread pool
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_row, i): i for i in range(total)}

        for future in as_completed(futures):
            try:
                index, result = future.result()
                excel_handler.set_result(index, result.to_dict())
                if result.error:
                    errors += 1
            except Exception as e:
                index = futures[future]
                excel_handler.set_result(index, {"error": str(e)})
                errors += 1

            completed += 1
            progress(completed / total, desc=f"Processing {completed}/{total}")

    # Get results
    result_df = excel_handler.get_results_dataframe()

    status = f"✅ Completed: {total} rows, {errors} errors"
    return status, result_df


def export_results() -> tuple:
    """Export results to Excel file."""
    global excel_handler

    result_df = excel_handler.get_results_dataframe()
    if result_df.empty:
        return "❌ No results to export", None

    # Create temp file
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"endpointing_results_{uuid.uuid4().hex[:8]}.xlsx")

    success, message, file_path = excel_handler.export_excel(output_path)

    if success:
        return f"✅ {message}", file_path
    else:
        return f"❌ {message}", None


def create_ui():
    """Create Gradio UI."""

    with gr.Blocks(title="Endpointing WebUI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ Speech Endpointing Annotation Tool")
        gr.Markdown("Batch process ASR transcripts to predict end-of-utterance using gRPC service.")

        with gr.Row():
            # Left column - Configuration
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ gRPC Connection")

                with gr.Row():
                    host_input = gr.Textbox(
                        label="Host",
                        value="127.0.0.1",
                        scale=2,
                    )
                    port_input = gr.Number(
                        label="Port",
                        value=50051,
                        precision=0,
                        scale=1,
                    )

                connect_btn = gr.Button("🔗 Connect", variant="primary")
                connection_status = gr.Textbox(
                    label="Status",
                    value="Not connected",
                    interactive=False,
                )

                gr.Markdown("### 📁 Data Import")
                file_input = gr.File(
                    label="Upload Excel/CSV",
                    file_types=[".xlsx", ".xls", ".csv"],
                )
                load_status = gr.Textbox(
                    label="Load Status",
                    value="No file loaded",
                    interactive=False,
                )

                gr.Markdown("### 📋 Column Mapping")
                asr_col = gr.Dropdown(
                    label="ASR Text Column *",
                    choices=[],
                    interactive=True,
                )
                history_col = gr.Dropdown(
                    label="History Column (JSON)",
                    choices=["(None)"],
                    value="(None)",
                    interactive=True,
                )
                lang_col = gr.Dropdown(
                    label="Language Column",
                    choices=["(None)"],
                    value="(None)",
                    interactive=True,
                )

                gr.Markdown("### 🎛️ Inference Settings")
                eou_threshold = gr.Slider(
                    label="EOU Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                )
                treat_unaddressed = gr.Checkbox(
                    label="Treat UNADDRESSED as EOU",
                    value=True,
                )
                concurrency = gr.Slider(
                    label="Concurrency",
                    minimum=1,
                    maximum=32,
                    value=8,
                    step=1,
                )

            # Right column - Data & Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Data Preview")
                preview_table = gr.Dataframe(
                    label="Preview (first 10 rows)",
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    run_btn = gr.Button("🚀 Run Inference", variant="primary", scale=2)
                    export_btn = gr.Button("📥 Export Results", scale=1)

                inference_status = gr.Textbox(
                    label="Inference Status",
                    value="Ready",
                    interactive=False,
                )

                gr.Markdown("### 📈 Results")
                results_table = gr.Dataframe(
                    label="Results",
                    interactive=False,
                    wrap=True,
                )

                export_status = gr.Textbox(
                    label="Export Status",
                    value="",
                    interactive=False,
                )
                export_file = gr.File(
                    label="Download",
                    visible=True,
                )

        # Event handlers
        connect_btn.click(
            fn=connect_grpc,
            inputs=[host_input, port_input],
            outputs=[connection_status],
        )

        file_input.change(
            fn=load_excel,
            inputs=[file_input],
            outputs=[load_status, preview_table, asr_col, history_col, lang_col],
        )

        run_btn.click(
            fn=run_inference,
            inputs=[asr_col, history_col, lang_col, eou_threshold, treat_unaddressed, concurrency],
            outputs=[inference_status, results_table],
        )

        export_btn.click(
            fn=export_results,
            inputs=[],
            outputs=[export_status, export_file],
        )

    return app


def main():
    """Main entry point."""
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
