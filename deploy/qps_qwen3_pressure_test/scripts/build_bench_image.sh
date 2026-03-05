#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-vllm-bench:latest}"

echo "Building ${IMAGE_NAME} from ${ROOT_DIR}/Dockerfile.bench ..."
docker build --network sagemaker -t "${IMAGE_NAME}" -f "${ROOT_DIR}/Dockerfile.bench" "${ROOT_DIR}"
echo "Done."
