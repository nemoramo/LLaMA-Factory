# Endpointing WebUI

A Gradio-based annotation interface for batch processing ASR transcripts to predict end-of-utterance using a gRPC endpointing service.

## Features

- **Excel/CSV Import**: Upload data files with ASR transcripts, conversation history, and language codes
- **Column Auto-Detection**: Automatically maps common column names (`asr_text`, `history`, `lang`)
- **Concurrent Inference**: Process multiple rows in parallel (configurable concurrency)
- **gRPC Integration**: Connects to `endpointing.v1.EndpointingService` for predictions
- **Detailed Results**: Shows all probabilities (`p_eou`, `p_cont_user`, `p_unaddressed`) and latency
- **Export to Excel**: Download results with all inference outputs

## Requirements

- Python 3.10+
- gRPC server running `endpointing.v1.EndpointingService/Predict`

## Installation

### Quick Start (Linux/Mac)

```bash
cd endpointing_webui
pip install -r requirements.txt
python app.py
```

### Windows

```batch
cd endpointing_webui
install.bat
run.bat
```

## Usage

1. **Start the WebUI**:
   ```bash
   python app.py
   ```
   The interface will be available at `http://127.0.0.1:7860`

2. **Configure gRPC Connection**:
   - Set Host (default: `127.0.0.1`) and Port (default: `50051`)
   - Click "Connect" to test the connection

3. **Upload Data**:
   - Upload an Excel (`.xlsx`, `.xls`) or CSV file
   - Columns are auto-detected or can be manually mapped:
     - **ASR Text Column** (required): The transcript text to evaluate
     - **History Column** (optional): JSON array of conversation history
     - **Language Column** (optional): Language code (e.g., `en-US`, `zh-CN`)

4. **Configure Inference Settings**:
   - **EOU Threshold**: Minimum probability for `<EOU>` decision (default: 0.6)
   - **Treat UNADDRESSED as EOU**: Merge `<UNADDRESSED>` probability into `<EOU>`
   - **Concurrency**: Number of parallel requests (default: 8)

5. **Run Inference**: Click "Run Inference" to process all rows

6. **Export Results**: Click "Export Results" to download the annotated data

## Input Data Format

### Required Column
- `asr_text` (or similar): The ASR transcript text to evaluate

### Optional Columns
- `history`: JSON string of conversation history
  ```json
  [
    {"role": "user", "text": "Hello"},
    {"role": "assistant", "text": "Hi, how can I help?"}
  ]
  ```
- `lang`: Language code (e.g., `en-US`, `zh-CN`). Defaults to `en-US`

### Example Input (Excel/CSV)

| asr_text | history | lang |
|----------|---------|------|
| hello how are you | [] | en-US |
| I need help with | [{"role": "assistant", "text": "Hello! How can I help?"}] | en-US |
| thank you that's all | [{"role": "user", "text": "What's the weather?"}, {"role": "assistant", "text": "It's sunny."}] | en-US |

## Output Columns

| Column | Description |
|--------|-------------|
| `label` | Predicted label: `<EOU>`, `<CONT_USER>`, or `<UNADDRESSED>` |
| `confidence` | Confidence score for the predicted label |
| `p_eou` | Probability of End-of-Utterance |
| `p_cont_user` | Probability of User Continuing |
| `p_unaddressed` | Probability of Unaddressed Speech |
| `latency_ms` | Inference latency in milliseconds |
| `model` | Model name used for prediction |
| `error` | Error message if inference failed |

## Label Meanings

- **`<EOU>`** (End of Utterance): User has finished speaking, system should respond
- **`<CONT_USER>`** (Continue User): User is still speaking, wait for more input
- **`<UNADDRESSED>`** (Unaddressed): Speech is not directed at the system

## gRPC Protocol

This WebUI connects to a gRPC service implementing `endpointing.v1.EndpointingService/Predict`.

**Proto Definition** (`endpointing.proto`):

```protobuf
package endpointing.v1;

service EndpointingService {
  rpc Predict(EndpointingRequest) returns (EndpointingResponse) {}
}

message EndpointingRequest {
  string request_id = 1;
  string session_id = 2;
  string lang = 3;
  Asr asr = 4;
  repeated HistoryTurn history = 5;
  Options options = 6;
}

message EndpointingResponse {
  string request_id = 1;
  string label = 2;
  double confidence = 3;
  string model = 4;
  int32 latency_ms = 5;
  double p_eou = 6;
  double p_cont_user = 7;
  double p_unaddressed = 8;
}
```

## File Structure

```
endpointing_webui/
├── app.py              # Main Gradio application
├── grpc_client.py      # gRPC client with retry logic
├── excel_handler.py    # Excel/CSV import/export
├── pb/                 # Protocol buffer stubs (endpointing.v1)
│   ├── __init__.py
│   ├── endpointing_pb2.py
│   └── endpointing_pb2_grpc.py
├── examples/
│   └── sample_input.xlsx
├── requirements.txt
├── install.bat         # Windows installation script
├── run.bat            # Windows run script
└── README.md
```

## Troubleshooting

### "Not connected to gRPC server"
- Ensure the gRPC server is running on the configured host:port
- Check firewall settings
- Verify the server implements `endpointing.v1.EndpointingService`

### "Failed to load file"
- Ensure the file is a valid Excel (.xlsx, .xls) or CSV format
- Check that the file is not corrupted or empty

### Empty inference results
- Verify the ASR text column is correctly mapped
- Check that input text is not empty

## Related Projects

- [vLLM Endpointing gRPC Service](../../deploy/vllm_endpointing_grpc/): The gRPC server this WebUI connects to
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory): The parent project for model training

## License

Apache-2.0
