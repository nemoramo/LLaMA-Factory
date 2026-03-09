# Docker Setup for NVIDIA GPUs

This directory contains Docker configuration files for running LLaMA Factory with NVIDIA GPU support.

## Prerequisites

### Linux-specific Requirements

Before running the Docker container with GPU support, you need to install the following packages:

1. **Docker**: The container runtime
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io

   # Or install Docker Engine from the official repository:
   # https://docs.docker.com/engine/install/
   ```

2. **Docker Compose** (if using the docker-compose method):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install docker-compose

   # Or install the latest version:
   # https://docs.docker.com/compose/install/
   ```

3. **NVIDIA Container Toolkit** (required for GPU support):
   ```bash
   # Add the NVIDIA GPG key and repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Restart Docker to apply changes
   sudo systemctl restart docker
   ```

   **Note**: Without `nvidia-container-toolkit`, the Docker container will not be able to access your NVIDIA GPU.

### Verify GPU Access

After installation, verify that Docker can access your GPU:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If successful, you should see your GPU information displayed.

## Usage

### Using Docker Compose (Recommended)

```bash
cd docker/docker-cuda/
docker compose up -d
docker compose exec llamafactory bash
```

### Using Docker Run

```bash
# Build the image
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --build-arg EXTRAS=metrics \
    -t llamafactory:latest .

# Run the container
docker run -dit --ipc=host --gpus=all \
    -p 7860:7860 \
    -p 8000:8000 \
    --name llamafactory \
    llamafactory:latest

# Enter the container
docker exec -it llamafactory bash
```

### Speech Workloads: Qwen3-ASR, Qwen3.5 Endpointing, FunAudioChat

For speech training stacks that need `ffmpeg` / `ffprobe`, `libsndfile`, and the Python audio helpers used by
`qwen3_asr` and `funaudiochat`, build the speech-focused image instead:

```bash
docker build -f ./docker/docker-cuda/Dockerfile.speech \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    -t llamafactory:speech .
```

Smoke test it on a single GPU:

```bash
export GPU_ID=0
export HF_CACHE=/path/to/huggingface_cache
export LOCAL_MODELS=/path/to/local_models
export LOCAL_ADAPTERS=/path/to/local_adapters

docker run --rm --ipc=host --gpus "\"device=${GPU_ID}\"" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${LOCAL_MODELS}:/models" \
    -v "${LOCAL_ADAPTERS}:/adapters" \
    llamafactory:speech \
    python scripts/docker/smoke_test_speech_stack.py \
      --require-cuda \
      --require-fa2 \
      --qwen3-asr-model /models/qwen3-asr/checkpoint \
      --funaudiochat-model /models/funaudiochat \
      --qwen3-5-endpointing-adapter /adapters/qwen3_5_endpointing
```

Notes:

1. `--gpus "\"device=${GPU_ID}\""` exposes only the selected physical GPU to the container, and it will appear as `cuda:0` inside.
2. The smoke test checks CUDA visibility, FlashAttention-2 availability, audio dependencies, `qwen3_asr` /
   `funaudiochat` registration, and the shipped Qwen3.5 speech-endpointing config.
3. Replace `HF_CACHE`, `LOCAL_MODELS`, and `LOCAL_ADAPTERS` with paths that match your environment.
4. `Dockerfile.speech` keeps the image focused on single-node speech workflows and does not preinstall `deepspeed`.
   If you need it, install `requirements/deepspeed.txt` inside the container after build.

## Troubleshooting

### GPU Not Detected

If your GPU is not detected inside the container:

1. Ensure `nvidia-container-toolkit` is installed
2. Check that the Docker daemon has been restarted after installation
3. Verify your NVIDIA drivers are properly installed: `nvidia-smi`
4. Check Docker GPU support: `docker run --rm --gpus all ubuntu nvidia-smi`

### Permission Denied

If you get permission errors, ensure your user is in the docker group:

```bash
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

## Additional Notes

- The default image is built on Ubuntu 22.04 (x86_64), CUDA 12.4, Python 3.11, PyTorch 2.6.0, and Flash-attn 2.7.4
- For different CUDA versions, you may need to adjust the base image in the Dockerfile
- Make sure your NVIDIA driver version is compatible with the CUDA version used in the Docker image
