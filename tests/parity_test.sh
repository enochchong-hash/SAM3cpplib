#!/bin/bash
# Backend parity gate: run the standard prompt cases (same table as
# scripts/development/make_goldens.sh) on the CUDA backend -- and on TensorRT when
# SAM3_TRT_* / --trt args are configured -- and compare against the CPU
# goldens in tests/goldens/.
#
# Tolerances: detection count must match exactly; score/iou within
# SCORE_TOL (default 0.02, the value production validation used); mask
# pixel count within MASK_TOL fraction (default 5% -- binary masks are
# thresholded at 0.5, so cross-backend numeric jitter moves boundary
# pixels; observed CPU-vs-CUDA/TRT drift is 2-3% on small masks while
# scores/boxes stay well inside their gates); box corners within
# BOX_TOL px (default 8).
#
# Usage:
#   tests/parity_test.sh --model resources/models/sam3-q8_0.ggml [--build build]
#       [--trt-onnx-dir DIR --trt-cache-dir DIR]   # additionally gate TensorRT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL=""
BUILD_DIR="$ROOT_DIR/build"
TRT_ARGS=()
SCORE_TOL="${SCORE_TOL:-0.02}"
MASK_TOL="${MASK_TOL:-0.05}"
BOX_TOL="${BOX_TOL:-8}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --build) BUILD_DIR="$2"; shift 2 ;;
        --trt-onnx-dir)  TRT_ARGS+=(--trt-onnx-dir "$2"); shift 2 ;;
        --trt-cache-dir) TRT_ARGS+=(--trt-cache-dir "$2"); shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$MODEL" && -f "$MODEL" ]] || { echo "usage: $0 --model sam3.ggml [--build dir] [--trt-onnx-dir D --trt-cache-dir D]" >&2; exit 1; }

E2E="$BUILD_DIR/examples/sam3cpp_e2e_check"
IMG="$ROOT_DIR/tests/data/cat.jpg"
GOLD="$ROOT_DIR/tests/goldens"
[[ -x "$E2E" && -f "$IMG" && -d "$GOLD" ]] || { echo "need built e2e_check, tests/data/cat.jpg and tests/goldens (scripts/development/make_goldens.sh)" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILURES=0

compare() {  # golden_dir actual_dir label
    python3 - "$1" "$2" "$3" "$SCORE_TOL" "$MASK_TOL" "$BOX_TOL" <<'PYEOF'
import re, sys, os
gold, act, label, score_tol, mask_tol, box_tol = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
def parse(path):
    dets = []
    for line in open(path):
        nums = {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.]+)", line)}
        box = [float(x) for x in re.search(r"box=([-\d.,]+)", line).group(1).split(",")]
        dets.append((nums, box))
    return dets
ok = True
for f in ("pcs_detections.txt", "pvs_detections.txt"):
    g, a = parse(os.path.join(gold, f)), parse(os.path.join(act, f))
    if len(g) != len(a):
        print(f"  FAIL [{label}/{f}] detection count {len(a)} != golden {len(g)}"); ok = False; continue
    for i, ((gn, gb), (an, ab)) in enumerate(zip(g, a)):
        for k in ("score", "iou"):
            if k in gn and abs(gn[k] - an[k]) > score_tol:
                print(f"  FAIL [{label}/{f}#{i}] {k} {an[k]:.4f} vs golden {gn[k]:.4f} (tol {score_tol})"); ok = False
        if gn.get("mask_px", 0) > 0 and abs(gn["mask_px"] - an["mask_px"]) / gn["mask_px"] > mask_tol:
            print(f"  FAIL [{label}/{f}#{i}] mask_px {an['mask_px']:.0f} vs golden {gn['mask_px']:.0f} (tol {mask_tol*100:.0f}%)"); ok = False
        if max(abs(x - y) for x, y in zip(gb, ab)) > box_tol:
            print(f"  FAIL [{label}/{f}#{i}] box {ab} vs golden {gb} (tol {box_tol}px)"); ok = False
print(f"  {'PASS' if ok else 'FAIL'} [{label}]")
sys.exit(0 if ok else 1)
PYEOF
}

declare -A CASES=(
    [text_point]="--text cat --point-x 600 --point-y 600"
    [exemplar_multi]="--text cat --pos-exemplars 300,150,900,1100 --points 600,600;300,300 --neg-points 1100,150"
    [box_prompt]="--text cat --box 150,80,1000,1150"
)

for name in text_point exemplar_multi box_prompt; do
    args=(${CASES[$name]})
    echo "==> case $name: CUDA"
    "$E2E" --runs 1 --model "$MODEL" --image "$IMG" --out "$WORK/cuda_$name" ${args[@]} >/dev/null 2>&1
    compare "$GOLD/$name" "$WORK/cuda_$name" "cuda/$name" || FAILURES=$((FAILURES+1))

    if [[ ${#TRT_ARGS[@]} -gt 0 ]]; then
        echo "==> case $name: TensorRT"
        "$E2E" --runs 1 --model "$MODEL" --image "$IMG" --out "$WORK/trt_$name" "${TRT_ARGS[@]}" ${args[@]} >/dev/null 2>&1
        compare "$GOLD/$name" "$WORK/trt_$name" "trt/$name" || FAILURES=$((FAILURES+1))
    fi
done

if [[ $FAILURES -gt 0 ]]; then
    echo "parity_test: $FAILURES case(s) FAILED"
    exit 1
fi
echo "parity_test: all cases passed"
