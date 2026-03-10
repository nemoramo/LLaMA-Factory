#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-vllm-bench:latest}"
RUN_ENV="${RUN_ENV:-local}"
BUILD_NETWORK="${BUILD_NETWORK:-}"

case "${RUN_ENV}" in
  local)
    ;;
  sagemaker)
    BUILD_NETWORK="${BUILD_NETWORK:-sagemaker}"
    ;;
  *)
    echo "Unsupported RUN_ENV: ${RUN_ENV} (expected local or sagemaker)"
    exit 1
    ;;
esac

echo "Building ${IMAGE_NAME} from ${ROOT_DIR}/Dockerfile.bench ..."
build_cmd=(docker build -t "${IMAGE_NAME}" -f "${ROOT_DIR}/Dockerfile.bench")
if [ -n "${BUILD_NETWORK}" ]; then
  build_cmd+=(--network "${BUILD_NETWORK}")
fi
build_cmd+=("${ROOT_DIR}")
"${build_cmd[@]}"
echo "Done."
