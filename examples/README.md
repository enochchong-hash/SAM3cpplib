# sam3cpplib examples

Two kinds of programs live here:

- **`features/` — the guided tour.** One small, heavily-commented program per
  library feature, numbered in reading order. Start here.
- **Tooling/validation programs** (`e2e_check`, `segment_image`, `quantize`,
  `dump_*_weights`) — used by the parity gates and the ONNX export pipeline
  (`docs/goldens.md`, `scripts/export_onnx.sh`).

## Build & run

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release     # add -DSAM3CPP_TENSORRT=ON for 09
cmake --build build -j$(nproc)

MODEL=path/to/sam3-q8_0.ggml     # see resources/models/README.md
IMG=tests/data/cat.jpg           # bundled test image

./build/examples/sam3cpp_ex_01_load_and_encode $MODEL $IMG
```

Every feature example takes positional arguments — `model.ggml` first,
`image` second — and prints what it demonstrates. All of them run on the
CUDA backend by default and fall back to CPU automatically when no GPU is
available (pass `cpu` where noted to force the reference backend).

## The tour

| # | Example | Feature demonstrated |
|---|---------|----------------------|
| 01 | `01_load_and_encode.cpp` | Model lifecycle: `sam3_load_model` → `sam3_create_state` → `sam3_encode_image`; GPU vs CPU backend selection; where the time goes |
| 02 | `02_text_prompt.cpp` | PCS text prompting (`sam3_segment_pcs`); `score_threshold` / `nms_threshold` tuning |
| 03 | `03_point_prompts.cpp` | PVS point prompting (`sam3_segment_pvs`): single positive point, then multiple positives + a negative point to carve away a region |
| 04 | `04_box_prompt.cpp` | PVS box prompting; `multimask` mode returning the 3-way ambiguity candidates |
| 05 | `05_exemplar_boxes.cpp` | PCS exemplar boxes (same-image, normalized coords): positive exemplars select by example, negative exemplars suppress |
| 06 | `06_concept_embeddings.cpp` | `sam3_pcs_compute_exemplar_embedding`: capture a reusable 256-float concept row from a reference image, persist it (1 KB), inject it on another image — no reference re-encode (EXPERIMENTAL for cross-image use) |
| 07 | `07_encode_once_prompt_many.cpp` | The partial-inference pattern: pay for `sam3_encode_image` once, then run many cheap PCS/PVS prompts against the cached features |
| 08 | `08_mask_geometry.cpp` | Mask convenience accessors: `sam3_mask_area/centroid/bbox/at/coords` — ready-made geometry instead of scanning mask bytes |
| 09 | `09_tensorrt_config.cpp` | Programmatic TensorRT setup via `sam3_params::trt` (no env vars) + the runtime FP16↔FP8 encoder switch (`sam3_set_encoder_fp8`) |

Reference docs: [docs/api.md](../docs/api.md) (API reference, including the
TensorRT configuration surface) and [docs/goldens.md](../docs/goldens.md)
(how outputs are validated). The full precision map and FP8 background live
in the deployment's `release/sam3/docs/tensorrt.md` until P5 consolidates
docs here.

## What is deliberately *not* here

Video tracking, SAM2, and EdgeTAM are not part of sam3cpplib (see
`docs/PLAN.md`); the upstream sam3.cpp library in `release/sam3` still
provides those.
