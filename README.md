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

**Status: phases P0–P4 complete** (see [docs/PLAN.md](docs/PLAN.md)) — all
three backends pass their parity gates: ggml CPU/CUDA bitwise-identical to
the production monolith; TensorRT FP16 bitwise-identical at production
timings (~120/36/7 ms encoder/PCS/PVS warm), FP8 opt-ins for the encoder and
the PCS head validated within ±0.003. ONNX regeneration from the .ggml is
byte-identical to production; the golden process and parity tests are green.

## System overview

Encode an image once, then prompt it many times — two prompt styles share
the cached features:

```
 RGB ─► preprocess ─► [1] IMAGE ENCODER (ViT-1024×32 + SimpleFPN) ─► features in sam3_state
                                                                        │
   PCS "segment every ___"  (sam3_segment_pcs, ~36 ms)                  │   PVS "segment the object at ___"
   [2] text encoder  [3] geometry/exemplar encoder                      │   (sam3_segment_pvs, ~7 ms)
            └──► [4] fusion encoder ─► [5] DETR decoder + scoring       │   [7] prompt encoder
                                     ─► [6] seg head ─► detections      └─► [8] SAM mask decoder ─► mask
```

Full walkthrough of the pipeline, each subsystem's shape and role, and where
it lives in the source: **[docs/architecture.md](docs/architecture.md)**.

## Subsystem options at a glance

| Subsystem | Options | Switch |
|---|---|---|
| Whole pipeline | ggml CPU / ggml CUDA / TensorRT | `use_gpu`, `SAM3CPP_TENSORRT` build + `sam3_params::trt` or `SAM3_TRT_*` env |
| [1] Image encoder | FP16 / **FP8** | `sam3_set_encoder_fp8(state, bool)` per state |
| [2] Text encoder | FP32 only (precision floor) | — |
| [3]–[6] PCS heads | FP32 / FP16 / `mixed:text_` (default) | `pcs_precision` / `SAM3_TRT_PCS_PRECISION` |
| [4]+[5] fenc/DETR GEMMs | additionally **FP8** | `sam3_set_pcs_fp8(state, bool)` per state |
| [7]+[8] PVS | FP32 only (precision floor) | — |
| Weights (.ggml, all backends) | f32 / f16 / q4_0 / q4_1 / **q8_0** (production) | `examples/quantize.cpp` |

The full options matrix with defaults, env-var↔config mapping, and the
measured accuracy/speed trade-off of every choice:
[docs/architecture.md §3](docs/architecture.md) and
[docs/tensorrt.md](docs/tensorrt.md).

## Build

```bash
git submodule update --init 3rdparty/ggml
export PATH=/usr/local/cuda/bin:$PATH   # for the CUDA backend
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# smoke checks
./build/tests/sam3cpp_mask_utils_test
./tests/consume_test/run.sh             # standalone submodule-consumer build

# backend parity vs the committed CPU goldens (see docs/goldens.md)
tests/parity_test.sh --model resources/models/sam3-q8_0.ggml \
    [--trt-onnx-dir resources/onnx --trt-cache-dir var/trt_cache]

# end-to-end (needs a sam3 .ggml model, e.g. release/sam3's sam3-q8_0.ggml)
./build/examples/sam3cpp_segment_image --model sam3-q8_0.ggml --image cat.jpg --text cat

# guided tour of every feature: examples/features/01..10 (see examples/README.md)
./build/examples/sam3cpp_ex_01_load_and_encode sam3-q8_0.ggml tests/data/cat.jpg
```

For TensorRT: `./scripts/setup_tensorrt.sh` once (or `--copy-from <dir>` to
reuse an already-vendored SDK), then configure with `-DSAM3CPP_TENSORRT=ON`.

Key CMake options: `SAM3CPP_CUDA` (ON), `SAM3CPP_TENSORRT` (OFF),
`SAM3CPP_IMAGE_IO` (ON; OFF = codec-free core, raw RGB in / raw masks out).

## Documentation

| Doc | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | **system overview**: pipeline, all 8 subsystems, the per-subsystem options matrix, source map |
| [docs/api.md](docs/api.md) | public API reference (lifecycle, partial inference, accessors, TRT config) |
| [docs/tensorrt.md](docs/tensorrt.md) | precision map + rationale, the two FP8 opt-ins, generating FP8 graphs, engine caching |
| [docs/goldens.md](docs/goldens.md) | the golden-sample process and parity tolerances |
| [examples/README.md](examples/README.md) | runnable per-feature tour (01–10) |
| [docs/PLAN.md](docs/PLAN.md) | the extraction plan and phase-gate history (P0–P5) |
