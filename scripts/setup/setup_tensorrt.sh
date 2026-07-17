#!/bin/bash
# One-time dev-machine setup for the optional TensorRT image-encoder path
# (SAM3CPP_TENSORRT=ON). Never invoked by the deployed server -- fetches
# just the public C++ headers (from the TensorRT OSS repo, headers only, no
# build) and the closed-source runtime .so libraries (via a throwaway pip
# venv) needed to compile and link sam3cpplib with -DSAM3CPP_TENSORRT=ON.
#
# Pinned to TensorRT 10.13.3 (not 11.x): 11.x requires CUDA 13.x per its own
# docs, this station is on CUDA 12.8 / driver 570.211 and does not want a
# driver upgrade + reboot. 10.13.3 has confirmed SM_120 (Blackwell/RTX 50)
# support (added in 10.8) and a matching tensorrt-cu12 pip wheel.
set -euo pipefail

TRT_TAG="v10.13.3"
TRT_PIP_VERSION="10.13.3.9"
ONNX_TRT_COMMIT="9a9f7883dd7b8cb0a718395bac2075fab6f97da8"  # parsers/onnx pin at $TRT_TAG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INCLUDE_DIR="$ROOT_DIR/3rdparty/tensorrt-include"
LIB_DIR="$ROOT_DIR/3rdparty/tensorrt-libs"

usage() {
    cat <<'EOF'
Usage: scripts/setup/setup_tensorrt.sh [--copy-from PATH]

With no arguments, download the pinned TensorRT headers and CUDA 12 runtime
libraries. --copy-from imports an existing vendored SDK from a directory that
contains tensorrt-include/ and tensorrt-libs/.
EOF
}

case "${1:-}" in
    "") ;;
    --help) usage; exit 0 ;;
    --copy-from)
        [[ $# -eq 2 && -n "${2:-}" ]] || { usage >&2; exit 2; }
        ;;
    *) echo "Error: unknown option: $1" >&2; usage >&2; exit 2 ;;
esac

# Fast path: --copy-from <dir> hardlink-copies an existing vendored SDK
# (e.g. another checkout's 3rdparty/) instead of downloading ~2.7GB again.
if [[ "${1:-}" == "--copy-from" ]]; then
    SRC="$(cd "$2" && pwd)"
    [[ -d "$SRC/tensorrt-include" && -d "$SRC/tensorrt-libs" ]] || {
        echo "Error: $SRC must contain tensorrt-include/ and tensorrt-libs/" >&2
        exit 1
    }
    echo "==> Copying vendored TensorRT SDK from $SRC (hardlinks where possible)"
    rm -rf "$INCLUDE_DIR" "$LIB_DIR"
    cp -al "$SRC/tensorrt-include" "$INCLUDE_DIR" 2>/dev/null || cp -a "$SRC/tensorrt-include" "$INCLUDE_DIR"
    cp -al "$SRC/tensorrt-libs" "$LIB_DIR" 2>/dev/null || cp -a "$SRC/tensorrt-libs" "$LIB_DIR"
    echo "==> Done. Build with: ./scripts/build.sh cuda --trt"
    exit 0
fi

echo "==> TensorRT headers -> $INCLUDE_DIR"
rm -rf "$INCLUDE_DIR"
mkdir -p "$INCLUDE_DIR"
TMP_TRT="$(mktemp -d)"
trap 'rm -rf "$TMP_TRT"' EXIT
git clone --depth 1 --branch "$TRT_TAG" https://github.com/NVIDIA/TensorRT.git "$TMP_TRT/TensorRT"
cp "$TMP_TRT/TensorRT"/include/*.h "$INCLUDE_DIR/"

echo "==> onnx-tensorrt parser header -> $INCLUDE_DIR"
TMP_ONNXTRT="$(mktemp -d)"
git clone https://github.com/onnx/onnx-tensorrt.git "$TMP_ONNXTRT/onnx-tensorrt"
git -C "$TMP_ONNXTRT/onnx-tensorrt" checkout "$ONNX_TRT_COMMIT"
cp "$TMP_ONNXTRT/onnx-tensorrt/NvOnnxParser.h" "$INCLUDE_DIR/"
rm -rf "$TMP_ONNXTRT"

echo "==> TensorRT runtime libs (tensorrt-cu12==$TRT_PIP_VERSION) -> $LIB_DIR"
rm -rf "$LIB_DIR"
mkdir -p "$LIB_DIR"
TMP_VENV="$(mktemp -d)"
if command -v uv >/dev/null 2>&1; then
    uv venv --quiet "$TMP_VENV/venv"
    uv pip install --quiet --python "$TMP_VENV/venv/bin/python" "tensorrt-cu12==$TRT_PIP_VERSION"
else
    echo "    (uv not found on PATH, falling back to python3 -m venv + pip)"
    python3 -m venv "$TMP_VENV/venv"
    "$TMP_VENV/venv/bin/pip" install --quiet --upgrade pip
    "$TMP_VENV/venv/bin/pip" install --quiet "tensorrt-cu12==$TRT_PIP_VERSION"
fi
SITE_PKGS="$("$TMP_VENV/venv/bin/python" -c 'import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))')"
cp -P "$SITE_PKGS"/libnvinfer.so* "$LIB_DIR/"
cp -P "$SITE_PKGS"/libnvinfer_plugin.so* "$LIB_DIR/"
cp -P "$SITE_PKGS"/libnvonnxparser.so* "$LIB_DIR/"
# Needed by createInferBuilder (engine *building*, not pure inference) --
# large (~2GB), Linux variant only (skip libnvinfer_builder_resource_win.so*).
cp "$SITE_PKGS"/libnvinfer_builder_resource.so.* "$LIB_DIR/"
rm -rf "$TMP_VENV"

# pip's wheel only ships the fully-versioned .so.N files (no unversioned dev
# symlink) -- CMake's find_library() looks for the plain libFOO.so name, so
# add that symlink ourselves (same role a -dev system package would play).
for lib in nvinfer nvinfer_plugin nvonnxparser; do
    versioned="$(ls "$LIB_DIR"/lib${lib}.so.* 2>/dev/null | head -1)"
    if [ -n "$versioned" ]; then
        ln -sf "$(basename "$versioned")" "$LIB_DIR/lib${lib}.so"
    fi
done

echo "==> Done."
echo "    Headers: $INCLUDE_DIR"
echo "    Libs:    $LIB_DIR"
echo "    Build with: ./scripts/build.sh cuda --trt"
