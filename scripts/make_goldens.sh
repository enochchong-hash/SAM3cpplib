#!/bin/bash
# THE golden-sample process: run the ggml **CPU** backend (the numerical
# reference) over the standard prompt cases on tests/data/cat.jpg and store
# the detection outputs under tests/goldens/. CUDA and TensorRT parity is
# then judged against these files by tests/parity_test.sh.
#
# CPU encoding is slow (~35 s per encode on this class of machine); the full
# run takes a few minutes. Regenerate only when the model file or the
# reference CPU implementation intentionally changes.
#
# Usage: scripts/make_goldens.sh --model resources/models/sam3-q8_0.ggml [--build build]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL=""
BUILD_DIR="$ROOT_DIR/build"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --build) BUILD_DIR="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$MODEL" && -f "$MODEL" ]] || { echo "usage: $0 --model sam3.ggml [--build dir]" >&2; exit 1; }

E2E="$BUILD_DIR/examples/sam3cpp_e2e_check"
[[ -x "$E2E" ]] || { echo "$E2E missing -- build with SAM3CPP_BUILD_EXAMPLES=ON first" >&2; exit 1; }
IMG="$ROOT_DIR/tests/data/cat.jpg"
[[ -f "$IMG" ]] || { echo "$IMG missing" >&2; exit 1; }

GOLD="$ROOT_DIR/tests/goldens"
mkdir -p "$GOLD"

# Case table: name | extra e2e_check args. Must stay in sync with
# tests/parity_test.sh. Covers every prompt shape the library supports:
# text, text+exemplar boxes, single point, multi+negative points, box.
run_case() {
    local name="$1"; shift
    echo "==> golden: $name"
    "$E2E" --no-gpu --runs 1 --model "$MODEL" --image "$IMG" \
           --out "$GOLD/$name" "$@" 2>"$GOLD/$name.log"
    rm -f "$GOLD/$name.log"
}

run_case text_point      --text cat --point-x 600 --point-y 600
run_case exemplar_multi  --text cat --pos-exemplars "300,150,900,1100" \
                         --points "600,600;300,300" --neg-points "1100,150"
run_case box_prompt      --text cat --box "150,80,1000,1150"

echo "==> Goldens written to $GOLD"
