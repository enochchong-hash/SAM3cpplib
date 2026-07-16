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

**Status: P1 (ggml core) complete** — CPU + CUDA backends build and pass the
parity gates (bitwise-identical outputs vs the production monolith on cat.jpg
across text / exemplar / point / negative-point / box prompts). TensorRT
backend sources are in the tree but unvalidated until P2. See
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

Key CMake options: `SAM3CPP_CUDA` (ON), `SAM3CPP_TENSORRT` (OFF until P2),
`SAM3CPP_IMAGE_IO` (ON; OFF = codec-free core, raw RGB in / raw masks out).
