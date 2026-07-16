#!/bin/bash
# Build + run the standalone consumer project in a scratch build tree.
# Usage: tests/consume_test/run.sh  (from anywhere; paths are script-relative)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"

# CUDA compiler for the ggml CUDA backend (only /usr/local/cuda* may be absolute).
if [[ -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD" --target consume_main -j"$(nproc)"
"$BUILD/consume_main"
