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


import json


def history_list_to_json(history_data) -> str:
    """Convert Gradio Dataframe data (dict/list/DataFrame) to JSON string for gRPC."""
    if history_data is None:
        return None

    # Handle pandas DataFrame
    if isinstance(history_data, pd.DataFrame):
        if history_data.empty:
            return None
        history_data = history_data.values.tolist()

    # Handle dict format from Gradio ({"headers": [...], "data": [...]})
    if isinstance(history_data, dict) and "data" in history_data:
        history_data = history_data.get("data", [])

    # Handle list format
    if not isinstance(history_data, list) or len(history_data) == 0:
        return None

    result = []
    for row in history_data:
        if row and len(row) >= 2 and row[1]:
            role = "assistant" if row[0] == "Assistant" else "user"
            result.append({"role": role, "text": row[1]})
    return json.dumps(result) if result else None


def single_predict(
    asr_text: str,
    history_data: list,
    lang: str,
    eou_threshold: float,
    treat_unaddressed: bool,
) -> tuple:
    global grpc_client

    if grpc_client is None:
        return "❌ Not connected to gRPC server", None

    if not asr_text or not asr_text.strip():
        return "❌ Please enter ASR text", None

    history_json = history_list_to_json(history_data)

    result = grpc_client.predict(
        asr_text=asr_text.strip(),
        history_json=history_json,
        lang=lang or "en-US",
        eou_threshold=eou_threshold,
        treat_unaddressed_as_eou=treat_unaddressed,
        request_id=f"webui-single-{uuid.uuid4().hex[:8]}",
    )

    if result.error:
        return f"❌ Error: {result.error}", None

    result_data = {
        "Field": ["Label", "Confidence", "P(EOU)", "P(CONT_USER)", "P(UNADDRESSED)", "Latency (ms)", "Model"],
        "Value": [
            result.label,
            f"{result.confidence:.4f}",
            f"{result.p_eou:.4f}",
            f"{result.p_cont_user:.4f}",
            f"{result.p_unaddressed:.4f}",
            str(result.latency_ms),
            result.model,
        ],
    }
    result_df = pd.DataFrame(result_data)

    status = f"✅ Prediction: **{result.label}** (confidence: {result.confidence:.2%}, latency: {result.latency_ms}ms)"
    return status, result_df


def add_history_row(history_data):
    """Add a new row to the history table. Gradio Dataframe uses list of lists."""
    if history_data is None or len(history_data) == 0:
        return [["Assistant", ""]]
    # history_data is a list of lists, e.g., [["Assistant", "Hello"], ["User", "Hi"]]
    if isinstance(history_data, list):
        return history_data + [["Assistant", ""]]
    # Fallback for DataFrame (shouldn't happen with Gradio Dataframe)
    if isinstance(history_data, pd.DataFrame):
        return history_data.values.tolist() + [["Assistant", ""]]
    return [["Assistant", ""]]


def clear_history():
    return []


single_query_results: list = []


def add_single_result_to_history(
    asr_text: str,
    history_data: list,
    lang: str,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    global single_query_results

    if result_df is None or result_df.empty:
        return get_single_results_dataframe()

    result_dict = {}
    for _, row in result_df.iterrows():
        result_dict[row["Field"].lower().replace(" ", "_").replace("(", "").replace(")", "")] = row["Value"]

    history_json = history_list_to_json(history_data)

    record = {
        "selected": True,
        "asr_text": asr_text,
        "history": history_json if history_json else "[]",
        "lang": lang,
        "label": result_dict.get("label", ""),
        "confidence": result_dict.get("confidence", ""),
        "p_eou": result_dict.get("peou", ""),
        "p_cont_user": result_dict.get("pcont_user", ""),
        "p_unaddressed": result_dict.get("punaddressed", ""),
        "latency_ms": result_dict.get("latency_ms", ""),
        "model": result_dict.get("model", ""),
    }

    single_query_results.append(record)
    return get_single_results_dataframe()


def get_single_results_dataframe() -> pd.DataFrame:
    global single_query_results
    if not single_query_results:
        return pd.DataFrame(
            columns=[
                "selected",
                "asr_text",
                "history",
                "lang",
                "label",
                "confidence",
                "p_eou",
                "p_cont_user",
                "p_unaddressed",
                "latency_ms",
                "model",
            ]
        )
    return pd.DataFrame(single_query_results)


def clear_single_results():
    global single_query_results
    single_query_results = []
    return pd.DataFrame(
        columns=[
            "selected",
            "asr_text",
            "history",
            "lang",
            "label",
            "confidence",
            "p_eou",
            "p_cont_user",
            "p_unaddressed",
            "latency_ms",
            "model",
        ]
    )


def select_all_results():
    global single_query_results
    for record in single_query_results:
        record["selected"] = True
    return get_single_results_dataframe()


def deselect_all_results():
    global single_query_results
    for record in single_query_results:
        record["selected"] = False
    return get_single_results_dataframe()


def update_selection_from_table(table_data) -> None:
    global single_query_results

    if table_data is None:
        return

    if isinstance(table_data, pd.DataFrame):
        if table_data.empty:
            return
        rows = table_data.values.tolist()
        columns = table_data.columns.tolist()
    elif isinstance(table_data, dict) and "data" in table_data:
        rows = table_data.get("data", [])
        columns = table_data.get("headers", [])
    elif isinstance(table_data, list):
        rows = table_data
        columns = None
    else:
        return

    selected_idx = 0
    if columns and "selected" in columns:
        selected_idx = columns.index("selected")

    for i, row in enumerate(rows):
        if i < len(single_query_results) and len(row) > selected_idx:
            val = row[selected_idx]
            if isinstance(val, bool):
                single_query_results[i]["selected"] = val
            elif isinstance(val, str):
                single_query_results[i]["selected"] = val.lower() in ("true", "1", "yes", "✓", "☑")
            else:
                single_query_results[i]["selected"] = bool(val)


def export_single_results(table_data) -> tuple:
    global single_query_results

    if not single_query_results:
        return "❌ No results to export. Run some predictions first.", None

    update_selection_from_table(table_data)

    selected_records = [r for r in single_query_results if r.get("selected", True)]

    if not selected_records:
        return "❌ No results selected. Please select at least one row to export.", None

    export_data = [{k: v for k, v in r.items() if k != "selected"} for r in selected_records]
    result_df = pd.DataFrame(export_data)

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"single_query_results_{uuid.uuid4().hex[:8]}.xlsx")

    try:
        result_df.to_excel(output_path, index=False, engine="openpyxl")
        return f"✅ Exported {len(result_df)} of {len(single_query_results)} results to Excel", output_path
    except Exception as e:
        return f"❌ Export failed: {str(e)}", None


def create_ui():
    """Create Gradio UI."""

    with gr.Blocks(title="Endpointing WebUI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ Speech Endpointing Annotation Tool")
        gr.Markdown("Predict end-of-utterance using gRPC service. Supports single query and batch processing.")

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
            connect_btn = gr.Button("🔗 Connect", variant="primary", scale=1)
            connection_status = gr.Textbox(
                label="Status",
                value="Not connected",
                interactive=False,
                scale=2,
            )

        with gr.Tabs():
            # Tab 1: Single Query
            with gr.TabItem("🔍 Single Query"):
                gr.Markdown("Test a single ASR utterance. Add history turns or leave empty for first turn.")
                with gr.Row():
                    with gr.Column(scale=1):
                        single_asr_input = gr.Textbox(
                            label="ASR Text *",
                            placeholder="Enter user utterance, e.g., 'hello how are you'",
                            lines=2,
                        )
                        gr.Markdown("**History** (optional - add conversation turns)")
                        single_history_input = gr.Dataframe(
                            headers=["Role", "Content"],
                            datatype=["str", "str"],
                            col_count=(2, "fixed"),
                            row_count=(0, "dynamic"),
                            value=[],
                            interactive=True,
                            label="Conversation History",
                        )
                        with gr.Row():
                            add_history_btn = gr.Button("➕ Add Turn", size="sm")
                            clear_history_btn = gr.Button("🗑️ Clear", size="sm")
                        gr.Markdown(
                            "*Role: `Assistant` or `User`. Leave empty for first turn.*", elem_classes="text-sm"
                        )

                        with gr.Row():
                            single_lang_input = gr.Dropdown(
                                label="Language",
                                choices=["en-US", "zh-CN", "es-ES", "fr-FR", "de-DE", "ja-JP", "ko-KR"],
                                value="en-US",
                                allow_custom_value=True,
                            )
                            single_threshold = gr.Slider(
                                label="EOU Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.6,
                                step=0.05,
                            )
                        single_treat_unaddressed = gr.Checkbox(
                            label="Treat UNADDRESSED as EOU",
                            value=True,
                        )
                        single_predict_btn = gr.Button("🚀 Predict", variant="primary")

                    with gr.Column(scale=1):
                        single_status = gr.Markdown("Ready for prediction")
                        single_result_table = gr.Dataframe(
                            label="Prediction Result",
                            interactive=False,
                            wrap=True,
                        )

                        gr.Markdown("#### 📋 Query History")
                        single_history_results = gr.Dataframe(
                            label="Accumulated Results (select rows to export)",
                            interactive=True,
                            wrap=True,
                        )
                        with gr.Row():
                            single_select_all_btn = gr.Button("☑️ Select All", size="sm", scale=1)
                            single_deselect_all_btn = gr.Button("☐ Deselect All", size="sm", scale=1)
                        with gr.Row():
                            single_export_btn = gr.Button("📥 Export Selected", scale=2)
                            single_clear_results_btn = gr.Button("🗑️ Clear History", size="sm", scale=1)
                        single_export_status = gr.Textbox(
                            label="Export Status",
                            value="",
                            interactive=False,
                        )
                        single_export_file = gr.File(
                            label="Download",
                            visible=True,
                        )

            # Tab 2: Batch Processing
            with gr.TabItem("📊 Batch Processing"):
                gr.Markdown("Upload Excel/CSV for batch inference.")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 Data Import")
                        file_input = gr.File(
                            label="Upload Excel/CSV",
                            file_types=[".xlsx", ".xls", ".csv"],
                        )
                        load_status = gr.Textbox(
                            label="Load Status",
                            value="No file loaded",
                            interactive=False,
                        )

                        gr.Markdown("#### 📋 Column Mapping")
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

                        gr.Markdown("#### 🎛️ Inference Settings")
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

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📊 Data Preview")
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

                        gr.Markdown("#### 📈 Results")
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

        add_history_btn.click(
            fn=add_history_row,
            inputs=[single_history_input],
            outputs=[single_history_input],
        )

        clear_history_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[single_history_input],
        )

        single_predict_btn.click(
            fn=single_predict,
            inputs=[
                single_asr_input,
                single_history_input,
                single_lang_input,
                single_threshold,
                single_treat_unaddressed,
            ],
            outputs=[single_status, single_result_table],
        ).then(
            fn=add_single_result_to_history,
            inputs=[
                single_asr_input,
                single_history_input,
                single_lang_input,
                single_result_table,
            ],
            outputs=[single_history_results],
        )

        single_select_all_btn.click(
            fn=select_all_results,
            inputs=[],
            outputs=[single_history_results],
        )

        single_deselect_all_btn.click(
            fn=deselect_all_results,
            inputs=[],
            outputs=[single_history_results],
        )

        single_export_btn.click(
            fn=export_single_results,
            inputs=[single_history_results],
            outputs=[single_export_status, single_export_file],
        )

        single_clear_results_btn.click(
            fn=clear_single_results,
            inputs=[],
            outputs=[single_history_results],
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
