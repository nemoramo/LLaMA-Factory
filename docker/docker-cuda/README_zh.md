# NVIDIA GPU Docker 使用说明

本目录提供了在 NVIDIA GPU 环境下运行 LLaMA Factory 的 Docker 配置。

English version: [README.md](./README.md)

## 前置依赖

### Linux 环境要求

在启用 GPU 的 Docker 容器前，请先安装以下组件。

1. `Docker`

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install docker.io

# 或使用 Docker 官方安装方式：
# https://docs.docker.com/engine/install/
```

2. `Docker Compose`

```bash
# Ubuntu / Debian
sudo apt-get install docker-compose

# 或使用最新版安装方式：
# https://docs.docker.com/compose/install/
```

3. `NVIDIA Container Toolkit`

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker
```

说明：如果没有安装 `nvidia-container-toolkit`，容器内无法访问 NVIDIA GPU。

### 验证 GPU 可见性

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

如果命令成功，说明 Docker 已经可以访问 GPU。

## 基础用法

### 使用 Docker Compose

```bash
cd docker/docker-cuda/
docker compose up -d
docker compose exec llamafactory bash
```

### 使用 Docker Run

```bash
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --build-arg EXTRAS=metrics \
    -t llamafactory:latest .

docker run -dit --ipc=host --gpus=all \
    -p 7860:7860 \
    -p 8000:8000 \
    --name llamafactory \
    llamafactory:latest

docker exec -it llamafactory bash
```

## 语音任务镜像

如果你要跑 `Qwen3-ASR`、`Qwen3.5 speech endpointing`、`FunAudioChat` 等语音相关任务，建议使用语音专用镜像。该镜像额外包含：

- `ffmpeg` / `ffprobe`
- `libsndfile`
- `requirements/speech.txt` 中的音频依赖

### 基础构建

```bash
docker build -f ./docker/docker-cuda/Dockerfile.speech \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --build-arg TRANSFORMERS_VERSION=5.2.0 \
    -t llamafactory:speech .
```

### 强制重装 FlashAttention-2

```bash
docker build -f ./docker/docker-cuda/Dockerfile.speech \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --build-arg TRANSFORMERS_VERSION=5.2.0 \
    --build-arg INSTALL_FLASHATTN=true \
    -t llamafactory:speech-fa2 .
```

### 单卡 smoke test

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

说明：

1. `--gpus "\"device=${GPU_ID}\""` 只会把指定物理卡暴露给容器，容器内会显示为 `cuda:0`。
2. `Dockerfile.speech` 现在默认通过 `TRANSFORMERS_VERSION=5.2.0` 显式固定 `transformers==5.2.0`。这是当前已经验证过的语音栈版本，覆盖 `Qwen3-ASR`、`Qwen3` / `Qwen3.5` speech endpointing 和 `FunAudioChat`。
3. 如果后续为了支持更多新模型需要升级 `transformers`，请先调整 `TRANSFORMERS_VERSION`，重建镜像，再重新跑 smoke test 和 `eval_hf_endpointing.py`，不要直接把升级结果当作安全。
4. smoke test 会检查 CUDA、FlashAttention-2、音频依赖、`qwen3_asr` / `funaudiochat` 注册，以及 Qwen3.5 endpointing 配置。
5. 请把 `HF_CACHE`、`LOCAL_MODELS`、`LOCAL_ADAPTERS` 替换成你自己的路径。
6. `Dockerfile.speech` 默认不预装 `deepspeed`。如果你需要它，可以在镜像内再安装 `requirements/deepspeed.txt`。
7. 如果你的 Docker 需要提权，请在上面的 `docker build` 和 `docker run` 前面加 `sudo`。

### 在 Docker 里跑 HF endpointing eval

`examples/speech_endpointing/eval_hf_endpointing.py` 已经在 `llamafactory:speech-fa2-20260309` 里做过实测，
使用本地 merged Qwen3 endpointing 导出模型时，`export_prompt_probe` 通过，结果是：

- top1 `<EOU>`
- top2 `<CONT_USER>`
- top3 `<UNADDRESSED>`

推荐直接评估 merged 导出模型：

```bash
docker run --rm --ipc=host --gpus "\"device=${GPU_ID}\"" \
    -v /path/to/exported_model:/smoke/model:ro \
    -v /path/to/eval_jsonl_dir:/smoke/data:ro \
    -v /path/to/output:/smoke/output \
    llamafactory:speech-fa2 \
    python /app/examples/speech_endpointing/eval_hf_endpointing.py \
      --base-model /smoke/model \
      --dataset /smoke/data/eval.jsonl \
      --out-dir /smoke/output \
      --export-prompt-probe
```

如果三个特殊标签没有进入 full-vocab top3，默认先排查 export，尤其先看 `tie_word_embeddings`，再判断是不是模型本身的问题。

### Qwen3-ASR 一键 20-step smoke

如果你想直接在 Docker 里做一个可复现的 `Qwen3-ASR` 20 step 功能验证，可以用这个脚本：

```bash
IMAGE=llamafactory:speech-fa2 \
GPU_ID=0 \
HF_CACHE=/path/to/huggingface_cache \
MODEL_PATH=/path/to/Qwen3-ASR-0.6B \
DATASET_DIR=/path/to/qwen3_asr_dataset_dir \
DATASET_NAME=qwen3_asr_hausa100_train \
EVAL_DATASET=qwen3_asr_hausa100_valid \
OUTPUT_DIR=/path/to/qwen3_asr_smoke20 \
MOUNT_MNT=1 \
scripts/docker/run_qwen3_asr_smoke.sh
```

说明：

1. 这个脚本会把当前仓库挂载到容器里的 `/app`，因此实际跑的是你本地当前代码，而不是镜像构建时打进去的旧代码。
2. 它会在输出目录下生成 `qwen3_asr_smoke_20step.yaml`，并把完整 Docker 命令写入 `training_command.txt`。
3. 如果 `MODEL_PATH` 和 `HF_CACHE` 都是本地现成目录，可以加 `OFFLINE=1` 禁止联网下载。
4. 如果你只想看生成的命令而不真正启动训练，可以加 `DRY_RUN=1`。

## 常见问题

### 容器里看不到 GPU

排查顺序：

1. 确认已经安装 `nvidia-container-toolkit`
2. 确认 Docker daemon 已重启
3. 先在宿主机执行 `nvidia-smi`
4. 再执行 `docker run --rm --gpus all ubuntu nvidia-smi`

### Docker 权限不足

```bash
sudo usermod -aG docker $USER
# 重新登录使权限生效
```

## 其他说明

- 默认镜像基于 Ubuntu 22.04、CUDA 12.4、Python 3.11、PyTorch 2.6.0、FlashAttention 2.7.4
- 如果你切换 CUDA 版本，通常也需要同步调整 Dockerfile 里的基础镜像
- 请确认宿主机 NVIDIA Driver 与镜像内 CUDA 版本兼容
