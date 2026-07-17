# System overview and subsystem options

This is the orientation document: what the SAM3 pipeline looks like, what
each neural-net subsystem does, and — in one place — every execution option
each subsystem offers and how to select it.

Companion docs: [api.md](api.md) (function-level reference),
[tensorrt.md](tensorrt.md) (precision rationale, FP8 pipeline, engine
caching), [goldens.md](goldens.md) (how correctness is validated),
[../examples/README.md](../examples/README.md) (runnable tour, 01–10).

## 1. The system in one picture

SAM3 is *promptable segmentation*: encode an image once, then ask questions
about it. Two question styles share the cached image features:

- **PCS** ("concept"): *"segment every ___"* — prompted by text, exemplar
  boxes, or stored concept embeddings; returns ALL matching instances.
- **PVS** ("visual"): *"segment the object at ___"* — prompted by
  points/boxes; returns ONE mask (or 3 ambiguity candidates).

```
                 sam3_encode_image (once per image)
                 ┌──────────────────────────────────┐
 RGB image ──►   │ preprocess (CPU resize→1008², norm)│
                 │        ▼                          │
                 │ [1] IMAGE ENCODER  ViT-1024×32    │
                 │        ▼           + SimpleFPN    │
                 │ multi-scale features (288²/144²/72²/36², d=256)
                 └───────────────┬──────────────────┘
                                 ▼  cached in sam3_state
      ┌──────────────────────────┴───────────────────────────┐
      ▼  sam3_segment_pcs (per prompt)                       ▼  sam3_segment_pvs (per prompt)
 [2] TEXT ENCODER (text → 32×256 tokens)               [7] PROMPT ENCODER
 [3] GEOMETRY ENCODER (exemplar boxes/embeddings           (points/box → sparse tokens)
      → ≤16×256 tokens, ROI-pooled)                          ▼
      ▼                                                [8] SAM MASK DECODER
 [4] FUSION ENCODER (6 layers: image features               (two-way transformer
      cross-attend prompt tokens, 5184 tokens)               → mask + iou_score
      ▼                                                      [+3 multimask cands])
 [5] DETR DECODER (6 layers, 200 queries →
      boxes + presence) + SCORING (dot-product
      vs pooled prompt)
      ▼
 [6] SEG HEAD (MaskFormer-style: query embeddings
      × pixel features → per-instance masks)
      ▼
 detections: {box, score, full-res mask}
```

Everything upstream of the fork runs once (~120 ms on TensorRT); each PCS
prompt costs ~36 ms and each PVS prompt ~7 ms — that asymmetry is the
"encode once, prompt many" serving pattern (example 07).

## 2. The subsystems

| # | Subsystem | Shape | Role |
|---|---|---|---|
| 1 | Image encoder | ViT: 32 blocks, d=1024, 16 heads, window 24 (4 global); SimpleFPN neck → 4 scales at d=256 | turns 1008² RGB into multi-scale features; the dominant cost |
| 2 | PCS text encoder | 24-layer CLIP-style transformer, d=1024→256, ctx 32, BPE vocab 49 408 (embedded in the .ggml) | text prompt → 32 prompt tokens |
| 3 | PCS geometry encoder | 3 layers, d=256 | exemplar boxes → prompt tokens by ROI-pooling image features (also the source of persistable concept embeddings) |
| 4 | PCS fusion encoder | 6 layers, 8 heads, FFN 2048 | conditions 5184 image tokens on the prompt tokens |
| 5 | PCS DETR decoder + scoring | 6 layers, 200 queries + presence token; dot-product scoring MLPs | proposes instances, scores them against the prompt |
| 6 | PCS seg head | MaskFormer-style, upscales via FPN levels | per-instance full-resolution masks |
| 7 | PVS prompt encoder | Gaussian positional encoding + learned point/box/type embeddings | points/box → sparse tokens (1–16, dynamic) |
| 8 | PVS SAM mask decoder | 2 two-way blocks + hypernetwork MLPs, 288² mask logits, 3-way multimask + IoU head | interactive single-object mask |

Non-neural stages (CPU, all backends): preprocessing (bilinear resize +
normalize), tokenization (BPE), NMS + score thresholding, mask upscale to
original resolution, mask geometry accessors.

## 3. Available options per subsystem

Three execution backends exist for the pipeline as a whole:

| Backend | Build | Select at runtime | Role |
|---|---|---|---|
| ggml **CPU** | always in | `sam3_params::use_gpu = false` | numerical reference; generates `tests/goldens/` (~35 s/encode) |
| ggml **CUDA** | `SAM3CPP_CUDA=ON` (default) | `use_gpu = true` (default) | portable GPU path (~0.9 s encode / 220 ms PCS / 65 ms PVS) |
| **TensorRT** | `SAM3CPP_TENSORRT=ON` + `scripts/setup_tensorrt.sh` | `sam3_params::trt.enabled` or `SAM3_TRT_ENCODER=1` (per-engine ONNX paths, see below) | fast path (~120 / 36 / 7 ms); per-subsystem precision options |

TensorRT granularity is **three engines** — encoder, PCS head, PVS head —
each independently configured; a stage without an engine falls back to ggml
CUDA unless `skip_ggml_weights` is set (deployed mode: no fallback, fail
loudly).

Per-subsystem precision options on the TensorRT path (the full rationale
table is in [tensorrt.md](tensorrt.md)):

Subsystems [2]–[6] compile into ONE TensorRT engine; its base mode comes
from `pcs_precision` (`SAM3_TRT_PCS_PRECISION` = `fp32` | `fp16` |
`mixed:<name-substrings>`, default `mixed:text_` = text FP32, rest FP16).
The FP8 entries below are per-state overlays on that base mode.

| Subsystem | Options | Default | How to switch |
|---|---|---|---|
| [1] Image encoder | FP16 / FP8 | FP16 | `sam3_set_encoder_fp8(state, true)` per state; engines from `encoder_onnx` / `encoder_onnx_fp8` (`SAM3_TRT_ONNX_PATH[_FP8]`) |
| [2] PCS text encoder | FP32 only | FP32 | fixed (precision floor) — kept FP32 by the default `mixed:text_` |
| [3] PCS geometry encoder | FP16 / FP32 | FP16 | `pcs_precision` (base mode of the shared PCS engine) |
| [4] PCS fusion encoder | FP16 + fused MHA / FP32; weight matmuls (Q/K/V/out + FFN) additionally FP8 | FP16, FP8 off | base: `pcs_precision`; FP8 overlay: `sam3_set_pcs_fp8(state, true)` per state, engine from `pcs_onnx_fp8` (`SAM3_TRT_PCS_ONNX_PATH_FP8`) |
| [5] PCS DETR decoder + scoring | FP16 / FP32; attention/FFN weight matmuls additionally FP8 (bbox/RPB/scoring MLPs stay FP16) | FP16, FP8 off | same two knobs as [4] (one engine, one overlay) |
| [6] PCS seg head | FP16 / FP32 | FP16 | `pcs_precision` |
| [7] PVS prompt encoder | FP32 only | FP32 | fixed (precision floor: FP16 diverges on negative points) |
| [8] PVS SAM mask decoder | FP32 only | FP32 | fixed (same engine and floor as [7]) |

Terminology for that FP8 row: a transformer layer has two kinds of matmul.
**Linear GEMMs** multiply activations by a *fixed weight matrix* (the
Q/K/V/output projections and the two FFN layers) -- those are what the PCS
FP8 option quantizes (132 of them across fenc+ddec). **Attention BMMs**
multiply activations by activations (Q.K^T and softmax.V) -- those stay FP16
so TensorRT's fused-MHA kernel keeps matching (the source of the 97->35 ms
PCS win). So with `mixed:text_` + PCS FP8, one request runs text FP32,
attention FP16-fused, small MLPs FP16, and the heavy weight multiplications
FP8.

Both FP8 switches share one contract: per-state, both engines stay resident
once used, flipping is free, and an unavailable FP8 engine falls back with a
stderr log instead of failing the request.

Weight-side option (all backends): the `.ggml` container ships as f32/f16 or
quantized `q4_0/q4_1/q8_0` (produced by `examples/quantize.cpp`); the
production configuration is `sam3-q8_0.ggml`. This is independent of the
TensorRT activation precisions above — the ONNX export dequantizes to FP32
and TensorRT re-quantizes per the table.

## 4. Where each subsystem lives in the source

| Subsystem | ggml graph builder | TensorRT path |
|---|---|---|
| Image encoder | `src/ggml/backbone.cpp` | `src/trt/trt_encoder.cpp` (+ `convert_sam3_encoder_to_onnx.py`) |
| Text/geometry/fusion/DETR/scoring/seg | `src/ggml/pcs.cpp` | `src/trt/trt_pcs.cpp` (+ `convert_sam3_pcs_to_onnx.py`) |
| PVS prompt encoder + SAM decoder | `src/ggml/pvs.cpp` | `src/trt/trt_pvs.cpp` (+ `convert_sam3_pvs_to_onnx.py`) |
| Orchestration (public API, sub-graph sequencing, backend dispatch) | `src/api.cpp` | same file — TRT is tried first, per stage |
