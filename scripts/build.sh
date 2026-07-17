#!/usr/bin/env bash
# Configure and build sam3cpplib on CPU, CUDA, or CUDA + TensorRT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND=auto
TRT_MODE=auto
BUILD_DIR="$ROOT_DIR/build"
BUILD_TYPE=Release
CUDA_ROOT=""
CUDA_ARCH="${SAM3_CUDA_ARCH:-}"
JOBS="${SAM3_BUILD_JOBS:-$(nproc)}"
CLEAN=false

usage() {
    cat <<'EOF'
Usage: scripts/build.sh [options] [auto|cpu|cuda]
  --trt / --no-trt   Force TensorRT support on/off (default: auto-detect SDK)
  --cuda-root PATH   Explicit CUDA toolkit root
  --cuda-arch LIST   CMake GPU architectures, e.g. 86 or '86;89;120'
  --build-dir PATH   Build directory (default: build)
  --debug            Debug build
  --clean            Remove the selected build directory first
  --jobs N           Parallel build jobs
EOF
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        auto|cpu|cuda) BACKEND=$1; shift ;;
        --trt) TRT_MODE=on; shift ;;
        --no-trt) TRT_MODE=off; shift ;;
        --cuda-root) [[ $# -ge 2 ]] || exit 2; CUDA_ROOT=$2; shift 2 ;;
        --cuda-arch) [[ $# -ge 2 ]] || exit 2; CUDA_ARCH=$2; shift 2 ;;
        --build-dir) [[ $# -ge 2 ]] || exit 2; BUILD_DIR=$2; shift 2 ;;
        --jobs) [[ $# -ge 2 ]] || exit 2; JOBS=$2; shift 2 ;;
        --debug) BUILD_TYPE=Debug; shift ;;
        --clean) CLEAN=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

CUDA_ARGS=(--quiet)
[[ -n "$CUDA_ROOT" ]] && CUDA_ARGS+=(--cuda-root "$CUDA_ROOT")
if [[ "$BACKEND" == auto ]]; then
    if source "$SCRIPT_DIR/setup/cuda_env.sh" "${CUDA_ARGS[@]}"; then BACKEND=cuda; else BACKEND=cpu; fi
elif [[ "$BACKEND" == cuda ]]; then
    source "$SCRIPT_DIR/setup/cuda_env.sh" "${CUDA_ARGS[@]}"
fi

if [[ "$TRT_MODE" == auto ]]; then
    if [[ "$BACKEND" == cuda && -f "$ROOT_DIR/3rdparty/tensorrt-include/NvInfer.h" && -e "$ROOT_DIR/3rdparty/tensorrt-libs/libnvinfer.so" ]]; then
        TRT_MODE=on
    else
        TRT_MODE=off
    fi
fi
[[ "$TRT_MODE" != on || "$BACKEND" == cuda ]] || { echo "Error: TensorRT requires the CUDA backend" >&2; exit 1; }
if [[ "$TRT_MODE" == on && ! -f "$ROOT_DIR/3rdparty/tensorrt-include/NvInfer.h" ]]; then
    echo "Error: TensorRT SDK assets missing; run scripts/setup/setup_tensorrt.sh first" >&2
    exit 1
fi

[[ "$CLEAN" == false ]] || rm -rf -- "$BUILD_DIR"
CMAKE_ARGS=(-S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE")
if [[ "$BACKEND" == cuda ]]; then
    # CUDACXX is exported by cuda_env.sh for a fresh configure. Do not also
    # force CMAKE_CUDA_COMPILER: changing only a symlink-resolved spelling
    # (for example /usr/local/cuda vs /usr/local/cuda-12.8) makes CMake wipe a
    # valid cache and can discard other requested options during reconfigure.
    CMAKE_ARGS+=(-DSAM3CPP_CUDA=ON -DCUDAToolkit_ROOT="$CUDA_HOME")
    if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        detected_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '. ') || detected_arch=""
        [[ "$detected_arch" =~ ^[0-9]+$ ]] && CUDA_ARCH=$detected_arch
    fi
    if [[ -n "$CUDA_ARCH" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
    else
        echo "Warning: GPU architecture unavailable; pass --cuda-arch for a headless/cross build" >&2
    fi
else
    CMAKE_ARGS+=(-DSAM3CPP_CUDA=OFF)
fi
[[ "$TRT_MODE" == on ]] && CMAKE_ARGS+=(-DSAM3CPP_TENSORRT=ON) || CMAKE_ARGS+=(-DSAM3CPP_TENSORRT=OFF)

echo "Configuring sam3cpplib: backend=$BACKEND TensorRT=$TRT_MODE build=$BUILD_TYPE"
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
echo "Build complete: $BUILD_DIR"
