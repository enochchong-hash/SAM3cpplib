#!/usr/bin/env bash
# One-command setup for a fresh clone.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND=auto
MODEL=q8_0
MODEL_ARGS=()
DOWNLOAD=true
INIT=true
TRT=false
BUILD_ARGS=()
usage() {
    cat <<'EOF'
Usage: scripts/setup.sh [options]
  --backend auto|cpu|cuda  Backend (default: auto)
  --model VARIANT          Checkpoint (default: q8_0)
  --source-dir PATH        Import checkpoint from an offline directory
  --no-model               Skip checkpoint acquisition
  --no-submodules          Skip git submodule initialization
  --trt                    Install the pinned TensorRT SDK and enable it
  --cuda-root PATH         Explicit CUDA toolkit root
  --cuda-arch LIST         CMake GPU architectures
  --help                   Show this help
EOF
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND=${2:-}; shift 2 ;;
        --model) MODEL=${2:-}; shift 2 ;;
        --source-dir) MODEL_ARGS+=(--source-dir "${2:-}"); shift 2 ;;
        --no-model) DOWNLOAD=false; shift ;;
        --no-submodules) INIT=false; shift ;;
        --trt) TRT=true; shift ;;
        --cuda-root|--cuda-arch) BUILD_ARGS+=("$1" "${2:-}"); shift 2 ;;
        --help) usage; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$BACKEND" == auto || "$BACKEND" == cpu || "$BACKEND" == cuda ]] || { usage >&2; exit 2; }
if [[ "$INIT" == true ]]; then git -C "$ROOT_DIR" submodule update --init --recursive; fi
if [[ "$DOWNLOAD" == true ]]; then "$SCRIPT_DIR/download_models.sh" "${MODEL_ARGS[@]}" "$MODEL"; fi
VERIFY_ARGS=(--backend "$BACKEND")
if [[ "$TRT" == true ]]; then
    "$SCRIPT_DIR/setup/setup_tensorrt.sh"
    BUILD_ARGS+=(--trt)
    VERIFY_ARGS+=(--trt)
else
    BUILD_ARGS+=(--no-trt)
fi
"$SCRIPT_DIR/verify_setup.sh" "${VERIFY_ARGS[@]}"
"$SCRIPT_DIR/build.sh" "${BUILD_ARGS[@]}" "$BACKEND"
echo "Setup complete. Checkpoint: $ROOT_DIR/resources/models/sam3-$MODEL.ggml"
