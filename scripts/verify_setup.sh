#!/usr/bin/env bash
# Read-only prerequisites and asset check for deployment hosts.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND=auto
CUDA_ROOT=""
CHECK_TRT=false
errors=0
warnings=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND=${2:-}; shift 2 ;;
        --cuda-root) CUDA_ROOT=${2:-}; shift 2 ;;
        --trt) CHECK_TRT=true; shift ;;
        --help) echo "Usage: scripts/verify_setup.sh [--backend auto|cpu|cuda] [--cuda-root PATH] [--trt]"; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$BACKEND" == auto || "$BACKEND" == cpu || "$BACKEND" == cuda ]] || exit 2
ok() { echo "ok   $1"; }
miss() { echo "MISS $1" >&2; errors=$((errors + 1)); }
warn() { echo "WARN $1" >&2; warnings=$((warnings + 1)); }
for command_name in bash cmake c++ git; do command -v "$command_name" >/dev/null 2>&1 && ok "$command_name" || miss "command: $command_name"; done
[[ -f "$ROOT_DIR/3rdparty/ggml/CMakeLists.txt" ]] && ok "ggml submodule" || miss "ggml submodule (git submodule update --init --recursive)"

CUDA_ARGS=(--quiet); [[ -n "$CUDA_ROOT" ]] && CUDA_ARGS+=(--cuda-root "$CUDA_ROOT")
cuda_ok=false
source "$SCRIPT_DIR/setup/cuda_env.sh" "${CUDA_ARGS[@]}" && cuda_ok=true
if [[ "$BACKEND" == cuda && "$cuda_ok" == false ]]; then miss "CUDA toolkit";
elif [[ "$BACKEND" == auto && "$cuda_ok" == false ]]; then warn "CUDA toolkit not found; auto build will use CPU";
elif [[ "$cuda_ok" == true ]]; then ok "CUDA toolkit $CUDA_HOME"; fi

model_count=$(find "$ROOT_DIR/resources/models" -maxdepth 1 -type f -name 'sam3-*.ggml' 2>/dev/null | wc -l)
(( model_count > 0 )) && ok "$model_count local model(s)" || warn "no checkpoint; run scripts/download_models.sh q8_0"
if [[ "$CHECK_TRT" == true ]]; then
    [[ -f "$ROOT_DIR/3rdparty/tensorrt-include/NvInfer.h" ]] && ok "TensorRT headers" || miss "TensorRT headers"
    [[ -e "$ROOT_DIR/3rdparty/tensorrt-libs/libnvinfer.so" ]] && ok "TensorRT runtime" || miss "TensorRT runtime"
    onnx_count=$(find "$ROOT_DIR/resources/onnx" -maxdepth 1 -type f -name '*.onnx' 2>/dev/null | wc -l)
    (( onnx_count >= 3 )) && ok "$onnx_count ONNX graphs" || warn "ONNX graphs absent; GGML inference remains usable"
fi
echo "Setup verification: $errors error(s), $warnings warning(s)"
(( errors == 0 ))
