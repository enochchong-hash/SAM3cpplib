# sam3cpplib

First-party C++ library for **SAM3 single-image promptable segmentation** — text,
exemplar-box, point, and box prompts — with three backends:

- **TensorRT** (FP16 default / FP8 opt-in): primary NVIDIA fast path
- **ggml CUDA**: portable GPU path
- **ggml CPU**: worst-case fallback and the golden-sample reference

Raw RGB in, raw 0/255 masks out — no image codecs, no HTTP, no Python at runtime.
Designed to be embedded in a larger application as a git submodule.

This library supersedes the `release/sam3` submodule-plus-patches arrangement; the code
is owned here (originally derived from MIT-licensed `PABannier/sam3.cpp`, see LICENSE).

**Status: P2 (TensorRT) complete** — all three backends build and pass their
parity gates: ggml CPU/CUDA are bitwise-identical to the production monolith
on cat.jpg (text / exemplar / point / negative-point / box prompts), and the
TensorRT FP16 path is bitwise-identical too (FP8 within ±0.002) at production
timings (~124/49/8 ms encoder/PCS/PVS warm). TensorRT can be configured
programmatically via `sam3_params::trt` or the SAM3_TRT_* env vars. See
[docs/PLAN.md](docs/PLAN.md) for the structure, port mapping, and phase gates.

## Build

```bash
git submodule update --init 3rdparty/ggml
export PATH=/usr/local/cuda/bin:$PATH   # for the CUDA backend
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# smoke checks
./build/tests/sam3cpp_mask_utils_test
./tests/consume_test/run.sh             # standalone submodule-consumer build

# end-to-end (needs a sam3 .ggml model, e.g. release/sam3's sam3-q8_0.ggml)
./build/examples/sam3cpp_segment_image --model sam3-q8_0.ggml --image cat.jpg --text cat
```

For TensorRT: `./scripts/setup_tensorrt.sh` once (or `--copy-from <dir>` to
reuse an already-vendored SDK), then configure with `-DSAM3CPP_TENSORRT=ON`.

Key CMake options: `SAM3CPP_CUDA` (ON), `SAM3CPP_TENSORRT` (OFF),
`SAM3CPP_IMAGE_IO` (ON; OFF = codec-free core, raw RGB in / raw masks out).
