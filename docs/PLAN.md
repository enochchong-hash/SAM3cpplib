# sam3cpplib — clean-room extraction plan

## Why this library exists

The current deployment (`release/sam3`) carries upstream `PABannier/sam3.cpp` as a
submodule plus a stack of 13 `git am` patches. That worked for incremental porting, but
the patch stack is now the primary maintenance cost: every library change means
prototype-branch commit → format-patch → re-apply → rebuild, and the upstream tree drags
in ~40% of code we never ship (SAM2, EdgeTAM, video tracking).

**sam3cpplib** is our own first-party library. It contains only the SAM3 single-image
promptable-segmentation pipeline, has **no dependency on the upstream sam3.cpp repo**
(code is ported in and owned here; upstream is credited in LICENSE), and keeps three
inference backends:

| Backend | Role |
|---|---|
| **TensorRT** (FP16 default, FP8 opt-in) | primary NVIDIA fast path (~206 ms warm full request) |
| **ggml CUDA** | cross-platform GPU path; also the only path on non-TensorRT NVIDIA setups |
| **ggml CPU** | worst-case fallback; primary role is the **golden-sample process** (reference outputs that CUDA/TRT parity tests compare against) |

Scope guarantees carried over from the validated deployment:
- Raw RGB in (`sam3_image`), raw 0/255 masks out — **no image codecs in the library**.
- Public API stays C-style `sam3_*` and **source-compatible with today's `sam3.h`**, so
  `release/sam3/src/ui/server.cpp` (which includes only `sam3.h`) migrates by switching
  its include path + link target, nothing else.
- Relocatable: works standalone and as a git submodule of a larger app.
  `CMAKE_CURRENT_SOURCE_DIR`-relative paths only; no absolute paths except
  `/usr/local/cuda*`.

Out of scope (deliberately dropped, not ported): SAM2 (Hiera), EdgeTAM (RepViT +
Perceiver), video tracking (memory encoder / memory attention / tracker state), Metal
build glue. The HTTP server, UI, and CLIs also stay out — they remain in `release/sam3`,
which will later consume this library.

## Folder and file structure

```
sam3cpplib/
├── CMakeLists.txt              # target sam3cpp + alias sam3cpp::sam3cpp; options below
├── LICENSE                     # MIT; retains PABannier/sam3.cpp copyright + our additions
├── README.md                   # quickstart: build, embed as submodule, run e2e check
├── .gitignore                  # build/, var/, tensorrt-{include,libs}/, models, onnx, engines
├── .gitmodules                 # exactly one submodule: 3rdparty/ggml
│
├── include/
│   └── sam3cpp/
│       └── sam3.h              # THE public header (ported from tools/sam3.cpp/sam3.h,
│                               #   minus tracker/SAM2/EdgeTAM API; plus sam3_trt_config)
│
├── src/
│   ├── sam3_internal.h         # shared internal decls: hparams, weight structs,
│   │                           #   sam3_model/sam3_state definitions, helper protos
│   ├── model_load.cpp          # .ggml container parser (magic/hparams/tensor records),
│   │                           #   q8_0 dequant via ggml type traits, weight-skip mode
│   ├── tokenizer.cpp           # CLIP-style BPE (embedded in the .ggml file)
│   ├── preprocess.cpp          # resize→1008², normalize (CPU)
│   ├── state.cpp               # inference-state alloc, PE caches (neck + prompt encoder)
│   ├── mask_utils.cpp          # NMS, mask upscale/binarize, convenience accessors
│   │                           #   (area/centroid/bbox/at/coords)
│   ├── api.cpp                 # public entry points + backend dispatch
│   │                           #   (encode_image / segment_pcs / segment_pvs / set_encoder_fp8
│   │                           #    / pcs_compute_exemplar_embedding)
│   │
│   ├── ggml/                   # ggml backend (CUDA + CPU golden path)
│   │   ├── backend.cpp         # backend init, GPU detection, graph-lowering support
│   │   ├── backbone.cpp        # ViT (RoPE, windowing) + SimpleFPN neck graphs
│   │   │                       #   incl. pixel-shuffle-GEMM deconv (CUDA) / native (CPU)
│   │   ├── pcs.cpp             # text encoder, geometry/exemplar encoder (ROI-align),
│   │   │                       #   fusion encoder, DETR decoder, scoring, seg head
│   │   └── pvs.cpp             # SAM prompt encoder + SAM mask decoder
│   │
│   └── trt/                    # TensorRT backend (SAM3CPP_TENSORRT=ON)
│       ├── trt_engine.{h,cpp}  # build-or-load cached engine (ONNX→engine, cache key =
│       │                       #   onnx bytes + GPU + TRT version + precision config,
│       │                       #   dynamic-shape optimization profiles 1..16)
│       ├── trt_runtime.{h,cpp} # IExecutionContext wrapper, lazy staging buffers,
│       │                       #   device_ptr chaining, dynamic setInputShape
│       ├── trt_encoder.cpp     # image-encoder path, FP16/FP8 engine selection
│       ├── trt_pcs.cpp         # PCS path (mixed:text_ precision) + CPU post
│       └── trt_pvs.cpp         # PVS path (FP32, dynamic prompt tokens) + CPU post
│
├── 3rdparty/
│   ├── ggml/                   # submodule: https://github.com/PABannier/ggml.git
│   │                           #   pinned at 331b9cb (the pin validated in production)
│   ├── tensorrt-include/       # gitignored; vendored by scripts/setup_tensorrt.sh
│   └── tensorrt-libs/          # gitignored; from pip wheel tensorrt-cu12==10.13.3.9
│
├── scripts/
│   ├── setup_tensorrt.sh       # one-time dev setup: vendor TRT 10.13.3 headers + libs
│   ├── export_onnx.sh          # .ggml → all three ONNX graphs (encoder/PCS/PVS) in one go
│   ├── make_goldens.sh         # THE golden-sample process: run the CPU backend on
│   │                           #   tests/data images, write tests/goldens/
│   └── convert/                # offline Python tooling (never used at runtime)
│       ├── convert_sam3_to_ggml.py         # HF checkpoint → .ggml container
│       ├── convert_sam3_encoder_to_onnx.py # weights → encoder ONNX (opset 13)
│       ├── convert_sam3_pcs_to_onnx.py     # weights → PCS ONNX (dynamic geom tokens)
│       ├── convert_sam3_pvs_to_onnx.py     # weights → PVS ONNX (dynamic sparse tokens)
│       ├── sam3_onnx_common.py             # shared onnx.helper utilities
│       ├── fp8_amax_calib.py               # split-graph amax calibration (8GB-safe)
│       └── fp8_inject_qdq.py               # opset19 + E4M3 Q/DQ injection (FP8 encoder)
│
├── examples/
│   ├── e2e_check.cpp           # port of pcs_pvs_e2e_check: full harness (points, neg
│   │                           #   points, box, exemplars, embeddings, cold/warm timing)
│   ├── segment_image.cpp       # minimal "load → encode → prompt → save mask" consumer demo
│   └── quantize.cpp            # f32 .ggml → q8_0 .ggml
│
├── tests/
│   ├── data/                   # small test images (cat.jpg + a couple of prompt cases)
│   ├── goldens/                # CPU-reference detections (score/box/mask hashes)
│   ├── parity_test.sh          # CPU-golden vs CUDA vs TRT, SCORE_TOL=0.02 gate
│   ├── mask_utils_test.cpp     # unit tests for the convenience accessors
│   └── consume_test/           # standalone CMake project that add_subdirectory()'s the
│                               #   lib — proves the submodule-consumer story every CI run
│
├── resources/
│   ├── models/                 # .ggml weights (gitignored; README explains download)
│   └── onnx/                   # exported ONNX graphs (gitignored; scripts/export_onnx.sh)
│
├── docs/
│   ├── PLAN.md                 # this file
│   ├── api.md                  # public API reference (partial inference, convenience API)
│   ├── tensorrt.md             # precision map, FP8, engine cache, env vars + trt_config
│   └── goldens.md              # the golden-sample process and tolerances
│
└── var/                        # runtime artifacts, gitignored (TRT engine caches)
```

## What gets ported, from where

Source of truth for the port is `tools/sam3.cpp` at branch `wip/trt-phase1`
(commit `049884b`, which is patches 0001–0013 in prototype form — i.e. the exact code
running in production today). Nothing is taken from pristine upstream, so **the patch
stack dies with this port**: the patched state becomes plain first-party code.

| New file | Ported from (tools/sam3.cpp) | Notes |
|---|---|---|
| `include/sam3cpp/sam3.h` | `sam3.h` | drop `sam3_tracker*`, `sam3_video_*`, SAM2/EdgeTAM enums; keep everything else verbatim; add `sam3_trt_config` |
| `src/model_load.cpp` | loader sections of `sam3.cpp` | SAM3 container only; keep `SAM3_TRT_SKIP_GGML_WEIGHTS` keep-list (sam_pe.* + geom input embeds) |
| `src/tokenizer.cpp` | BPE tokenizer section | unchanged |
| `src/preprocess.cpp` | preprocess section | unchanged (GPU preprocess is a future optimization) |
| `src/state.cpp` | inference-state + PE-cache sections | unchanged |
| `src/mask_utils.cpp` | NMS + mask post + patch-0013 accessors | unchanged |
| `src/ggml/backbone.cpp` | ViT/RoPE/window/neck graph builders | incl. patches 0001–0003 (graph lowering, pixel-shuffle GEMM, PE caches) |
| `src/ggml/pcs.cpp` | text/geometry/fusion/DETR/scoring/seg-head builders + `sam3_segment_pcs` ggml path | incl. exemplar label fix + embedding splice (patch 0010) |
| `src/ggml/pvs.cpp` | SAM prompt-encoder + mask-decoder builders + `sam3_segment_pvs` ggml path | incl. multi/neg-point + box (patch 0009) |
| `src/trt/trt_engine.{h,cpp}` | `sam3_trt_engine.{h,cpp}` | near-verbatim (already standalone) |
| `src/trt/trt_runtime.{h,cpp}` | `sam3_trt_runtime.{h,cpp}` | near-verbatim (already standalone) |
| `src/trt/trt_{encoder,pcs,pvs}.cpp` | TRT sections of `sam3.cpp` (patches 0004–0012) | FP16/FP8 selection, mixed:text_, dynamic tokens, device-ptr chaining |
| `src/api.cpp` | public functions of `sam3.cpp` | dispatch: TRT if configured → ggml otherwise |
| `scripts/convert/*` | same-named `.py` files | drop `convert_sam2_to_ggml.py`, `convert_edgetam_to_ggml.py` |
| `examples/e2e_check.cpp` | `examples/pcs_pvs_e2e_check.cpp` | unchanged |
| `examples/quantize.cpp` | `examples/quantize.cpp` | unchanged |
| `tests/goldens/`, `tests/data/` | `release/sam3/tests/goldens/` + `tests/cat.jpg` | regenerated via `scripts/make_goldens.sh` after port |

Size estimate: ~9–10k of the monolith's 15.6k lines survive (SAM3-only), plus ~620
lines of TRT engine/runtime, split into the modules above. The split is mechanical —
the monolith already has clear `/**** section ****/` banners; `src/sam3_internal.h`
absorbs the currently-file-local structs/helpers that more than one module needs.

## Public API (unchanged surface + one addition)

Kept verbatim: `sam3_load_model`, `sam3_create_state`, `sam3_encode_image`,
`sam3_segment_pcs`, `sam3_segment_pvs`, `sam3_set_encoder_fp8`,
`sam3_pcs_compute_exemplar_embedding` (documented-experimental), the
`sam3_mask_*` convenience accessors, and all parameter/result structs
(`sam3_pcs_params` incl. exemplars + embeddings, `sam3_pvs_params`, `sam3_detection`).

Added: **`sam3_trt_config`** — a programmatic alternative to the env vars for embedded
consumers (a host app should not have to mutate its own environment):

```cpp
struct sam3_trt_config {
    std::string encoder_onnx, encoder_onnx_fp8;   // empty = disabled
    std::string pcs_onnx, pvs_onnx;
    std::string cache_dir;                        // engine cache root
    std::string pcs_precision = "mixed:text_";
    bool        skip_ggml_weights = true;         // TRT-only deployments
};
// passed via sam3_params; env vars (SAM3_TRT_*) remain honored as fallback
// so release/sam3's start scripts keep working untouched.
```

## Build options

| Option | Default | Effect |
|---|---|---|
| `SAM3CPP_CUDA` | ON | ggml CUDA backend (`GGML_CUDA`) |
| `SAM3CPP_TENSORRT` | OFF | compile `src/trt/`; needs `scripts/setup_tensorrt.sh` run once |
| `SAM3CPP_BUILD_EXAMPLES` | ON if top-level | e2e_check, segment_image, quantize |
| `SAM3CPP_BUILD_TESTS` | ON if top-level | mask_utils_test, consume_test hook |

CPU-only build = both backends' options off is NOT a thing: ggml is always in (it is
the CPU fallback and the weight container's dequant layer); `SAM3CPP_CUDA=OFF
SAM3CPP_TENSORRT=OFF` yields the pure-CPU golden-sample build. Include-guard idiom from
`release/sam3` (cache-variable `SAM3CPP_INCLUDED`) carries over so two consumers can
`add_subdirectory()` the lib in a diamond without target collisions.

## Porting phases (each gated before the next)

1. **P0 — scaffold** *(this commit)*: folders, PLAN.md, README, LICENSE, .gitignore.
2. **P1 — ggml core**: ggml submodule pin, public header, `src/` minus `trt/`,
   CMake, `examples/`. **Gate**: CPU + CUDA builds; `e2e_check` on cat.jpg matches
   today's outputs exactly (same weights, same code → same numbers, tolerance 0 for
   CPU, 0.02 score for CUDA); `consume_test` builds standalone.
3. **P2 — TensorRT**: `src/trt/`, `setup_tensorrt.sh`, config struct. **Gate**: the
   production goldens hold (PCS 0.963, PVS iou 0.9859, ±0.02) on FP16 and FP8; warm
   timings within noise of today (~120/35/7 ms encoder/PCS/PVS).
4. **P3 — tooling**: `scripts/convert/`, `export_onnx.sh`, FP8 calib/inject.
   **Gate**: regenerate all ONNX from the .ggml, rebuild engines from scratch,
   goldens still hold.
5. **P4 — golden process + tests**: `make_goldens.sh`, `parity_test.sh`,
   `mask_utils_test`. **Gate**: one command regenerates goldens on CPU; parity gates
   green for CUDA + TRT.
6. **P5 — adopt in release/sam3** *(separate decision)*: server switches to
   sam3cpplib (submodule), upstream sam3.cpp submodule + 13-patch stack retired.

## Licensing

Upstream `PABannier/sam3.cpp` is MIT. `LICENSE` retains the upstream copyright notice
(Pierre-Antoine Bannier) alongside ours; ported files keep no per-file headers beyond
that. The ggml submodule keeps its own license. TensorRT headers/libs are never
committed (gitignored, fetched by script) — same handling as today.
