#!/bin/bash
# Regenerate all TensorRT ONNX graphs from a .ggml model in one go:
#   sam3_encoder.onnx  (ViT + SimpleFPN neck, static 1008x1008)
#   sam3_pcs.onnx      (text/exemplar head, dynamic geometry tokens)
#   sam3_pvs.onnx      (point/box head, dynamic sparse tokens)
#   sam3_encoder_fp8.onnx  (only with --fp8-amax; E4M3 Q/DQ injection)
#
# Offline tooling only -- never used at runtime. Needs python3 with numpy+onnx
# and the dump tools from a SAM3CPP_BUILD_EXAMPLES=ON build.
#
# Usage:
#   scripts/development/export_onnx.sh --model resources/models/sam3-q8_0.ggml \
#       [--out resources/onnx] [--build build] [--fp8-amax amax.json] [--check]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL=""
OUT_DIR="$ROOT_DIR/resources/onnx"
BUILD_DIR="$ROOT_DIR/build"
FP8_AMAX=""
PCS_FP8_AMAX=""
CHECK_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --out)          OUT_DIR="$2"; shift 2 ;;
        --build)        BUILD_DIR="$2"; shift 2 ;;
        --fp8-amax)     FP8_AMAX="$2"; shift 2 ;;
        --pcs-fp8-amax) PCS_FP8_AMAX="$2"; shift 2 ;;
        --check)        CHECK_FLAG="--check"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$MODEL" && -f "$MODEL" ]] || { echo "usage: $0 --model sam3.ggml [--out dir] [--build dir] [--fp8-amax amax.json] [--check]" >&2; exit 1; }

DUMP_ENC="$BUILD_DIR/examples/sam3cpp_dump_encoder_weights"
DUMP_HEADS="$BUILD_DIR/examples/sam3cpp_dump_pcs_pvs_weights"
[[ -x "$DUMP_ENC" && -x "$DUMP_HEADS" ]] || {
    echo "dump tools missing -- build first: cmake --build $BUILD_DIR -j (SAM3CPP_BUILD_EXAMPLES=ON)" >&2; exit 1; }
python3 -c 'import numpy, onnx' 2>/dev/null || { echo "python3 with numpy+onnx required" >&2; exit 1; }

WORK="$ROOT_DIR/var/onnx_export"
rm -rf "$WORK"
mkdir -p "$WORK/encoder" "$WORK/heads" "$OUT_DIR"

echo "==> Dumping encoder weights (ViT + necks)"
"$DUMP_ENC" --model "$MODEL" --out "$WORK/encoder"
echo "==> Dumping PCS/PVS head weights"
"$DUMP_HEADS" --model "$MODEL" --out "$WORK/heads"

CV="$SCRIPT_DIR/../convert"
echo "==> Authoring sam3_encoder.onnx"
python3 "$CV/convert_sam3_encoder_to_onnx.py" --export-dir "$WORK/encoder" --out "$OUT_DIR/sam3_encoder.onnx" $CHECK_FLAG
echo "==> Authoring sam3_pcs.onnx"
python3 "$CV/convert_sam3_pcs_to_onnx.py" --export-dir "$WORK/heads" --out "$OUT_DIR/sam3_pcs.onnx" $CHECK_FLAG
echo "==> Authoring sam3_pvs.onnx"
python3 "$CV/convert_sam3_pvs_to_onnx.py" --export-dir "$WORK/heads" --out "$OUT_DIR/sam3_pvs.onnx" $CHECK_FLAG

if [[ -n "$FP8_AMAX" ]]; then
    echo "==> Injecting FP8 (E4M3) Q/DQ -> sam3_encoder_fp8.onnx"
    python3 "$CV/fp8_inject_qdq.py" "$OUT_DIR/sam3_encoder.onnx" "$FP8_AMAX" "$OUT_DIR/sam3_encoder_fp8.onnx"
else
    echo "==> Skipping encoder FP8 variant (pass --fp8-amax amax.json; generate one"
    echo "    with scripts/convert/fp8_amax_calib.py -- see docs/tensorrt.md)"
fi

if [[ -n "$PCS_FP8_AMAX" ]]; then
    echo "==> Injecting FP8 (E4M3) Q/DQ -> sam3_pcs_fp8.onnx (fenc/ddec linear GEMMs)"
    python3 "$CV/fp8_pcs_inject_qdq.py" "$OUT_DIR/sam3_pcs.onnx" "$PCS_FP8_AMAX" "$OUT_DIR/sam3_pcs_fp8.onnx"
else
    echo "==> Skipping PCS FP8 variant (pass --pcs-fp8-amax amax.json; generate one"
    echo "    with examples/dump_pcs_calib_inputs + scripts/convert/fp8_pcs_amax_calib.py)"
fi

rm -rf "$WORK"
echo "==> Done. ONNX graphs in $OUT_DIR"
