# sam3cpplib public API

One header: `#include "sam3cpp/sam3.h"` (or plain `"sam3.h"` — both include
paths are exported, so code written against upstream sam3.cpp compiles
unchanged). All functions are C++-style free functions with the `sam3_`
prefix; no classes, no exceptions — failures return `false`, `nullptr`, or an
empty container with a message on stderr.

> Runnable, per-feature walkthroughs of everything below live in
> [`examples/features/`](../examples/README.md) (01–10, in reading order).

## Lifecycle

```cpp
sam3_params params;
params.model_path = "sam3-q8_0.ggml";
params.use_gpu    = true;            // false = CPU reference backend
auto model = sam3_load_model(params);        // shared_ptr, nullptr on failure
auto state = sam3_create_state(*model, params);  // unique_ptr, one per stream of work
```

One `sam3_model` can serve multiple sequential `sam3_state`s; one GPU model
per process. SAM2/EdgeTAM containers are rejected (video tracking is not part
of this library).

## Partial inference — encode once, prompt many times

The expensive stage (image encoding, ~120 ms TensorRT / ~870 ms CUDA / ~35 s
CPU) is decoupled from prompting (~50 ms PCS, ~7 ms PVS on TensorRT):

```cpp
sam3_image img;                       // raw RGB8, width*height*3
// ... fill img.width/height/data ...
sam3_encode_image(*state, *model, img);      // once per image

sam3_pcs_params pcs; pcs.text_prompt = "cat";
sam3_result r1 = sam3_segment_pcs(*state, *model, pcs);   // text / exemplars

sam3_pvs_params pvs; pvs.pos_points.push_back({600, 600});
sam3_result r2 = sam3_segment_pvs(*state, *model, pvs);   // points / box
```

PCS prompt shapes: text, positive/negative exemplar boxes (normalized [0,1]
XYXY, same image), and precomputed `exemplar_embeddings` rows captured via
`sam3_pcs_compute_exemplar_embedding` (experimental cross-image use — see the
header comment). PVS prompt shapes: any mix of positive/negative points and
one box; `multimask` returns the 3-way ambiguity masks.

## Results and mask convenience accessors

`sam3_result::detections[i]` carries the model-predicted `box`, `score` /
`iou_score`, and a full-resolution binary `mask` (0/255 bytes). Derived
geometry, so consumers don't scan mask bytes themselves:

```cpp
size_t                  a = sam3_mask_area(d.mask);       // foreground px count
sam3_point              c = sam3_mask_centroid(d.mask);   // {-1,-1} if empty
sam3_box                b = sam3_mask_bbox(d.mask);       // tight box, x1/y1 exclusive
bool                    h = sam3_mask_at(d.mask, x, y);   // range-checked test
std::vector<sam3_point> p = sam3_mask_coords(d.mask);     // row-major coords
```

## TensorRT configuration (SAM3CPP_TENSORRT builds)

Two equivalent ways; the struct wins per field when `enabled`:

```cpp
params.trt.enabled          = true;
params.trt.encoder_onnx     = ".../sam3_encoder.onnx";
params.trt.encoder_onnx_fp8 = ".../sam3_encoder_fp8.onnx";  // optional
params.trt.pcs_onnx         = ".../sam3_pcs.onnx";
params.trt.pvs_onnx         = ".../sam3_pvs.onnx";
params.trt.cache_dir        = ".../var/trt_cache";  // engines land in /encoder,/pcs,/pvs
// params.trt.pcs_precision = "mixed:text_";        // validated default
// params.trt.skip_ggml_weights = true;             // default: TRT-only, no ggml fallback
```

or the `SAM3_TRT_*` environment variables (see `docs/tensorrt.md` for the
full per-subsystem precision map and the FP8 pipeline). Runtime precision
switches, per state, both engines resident, free to flip:

```cpp
sam3_set_encoder_fp8(*state, true);  // image encoder FP16 -> FP8
sam3_set_pcs_fp8(*state, true);      // PCS head: FP8 fenc/ddec GEMMs
                                     // (text stays FP32, attention FP16-fused)
```

PVS has no reduced-precision variant by design — it must run FP32 (FP16
already diverges on negative-point prompts) and its engine is only ~32 MB.

With `skip_ggml_weights` (the deployed default) the ~1.1 GB of ggml weights
are never loaded and there is **no ggml fallback** — out-of-scope requests
fail loudly instead of degrading. Set it false to keep the CUDA fallback at
the cost of that VRAM.

## Image I/O helpers (optional)

`sam3_load_image` / `sam3_save_mask` are stb-backed conveniences compiled
under `SAM3CPP_IMAGE_IO=ON` (default). The core library consumes raw RGB and
produces raw masks; build with `OFF` for a codec-free dependency surface.

## Test/debug surface

`sam3_test_*` and `sam3_dump_*` back the golden-sample process and parity
tooling (`docs/goldens.md`); they are compiled in but not part of the stable
consumer contract.
