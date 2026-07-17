# TensorRT backend: precision map, FP8, engines

Build once: `scripts/setup/setup_tensorrt.sh` (vendors TensorRT 10.13.3 headers +
runtime libs), then `cmake -B build -DSAM3CPP_TENSORRT=ON`. Engines are built
from the ONNX graphs on first use and cached (key = ONNX bytes + GPU + TRT
version + precision config); export the graphs with `scripts/development/export_onnx.sh`.

## Per-subsystem precision map

Every precision below was chosen by measurement (17-variant PCS sweep, PVS
prompt-shape matrix, FP8 validation runs — see the release/sam3 project docs
for the full history). "FP8" always means explicit E4M3 Q/DQ in the ONNX
graph; TensorRT then picks FP8 kernels for the quantized regions.

| Subsystem | Precision | FP8 option? | Why |
|---|---|---|---|
| Image encoder (ViT-1024×32 + SimpleFPN) | FP16 default | **yes — opt-in** (`sam3_set_encoder_fp8`) | FP8 on GEMMs + attention BMMs: −450 MB VRAM, ~5% faster, score Δ≤0.001 on goldens |
| PCS text encoder (24-layer CLIP) | **FP32** | **no, by design** | the one precision-sensitive component: FP16 already drops confidence 0.963→0.820; FP8 would be strictly worse |
| PCS geometry/exemplar encoder | FP16 | no | ≤16 tokens/request — nothing to win, not worth the accuracy risk |
| PCS fusion encoder (6 × 5184 tokens) | FP16 + fused MHA | **yes — opt-in** (`sam3_set_pcs_fp8`) | linear GEMMs (qkv/out/FFN) carry the FLOPs; FP8 on them keeps the fused FP16 MHA intact. Measured: score Δ0.002, box ≤1.6 px, mask boundary drift ≤5.4% on small masks, warm 34 vs 36 ms |
| PCS DETR decoder (6 × 200 queries) | FP16 | **yes — same opt-in** | attention + FFN GEMMs quantized; the bbox/RPB/qpos MLPs stay FP16 (known sign-flip-near-zero sensitivity in the RPB log-distance transform) |
| PCS scoring + seg head | FP16 | no | tiny MLPs / directly mask-quality-critical |
| PVS prompt encoder + SAM decoder | **FP32** | **no, by design** | FP16 already diverges 27% on negative-point prompts; the whole engine is ~32 MB — there is no FP8 story |
| ggml CUDA / CPU backends | q8_0/F16 weights, F32 activations | no | ggml has no FP8 kernels; FP8 is a TensorRT-only capability of this library |

## The two FP8 opt-ins

Both follow the same contract: **per-state runtime switch, both engines stay
resident once used, switching is free, graceful fallback with a stderr log
if the FP8 engine is unavailable.**

```cpp
sam3_set_encoder_fp8(*state, true);   // image encoder  FP16 -> FP8
sam3_set_pcs_fp8(*state, true);       // PCS head       FP16 -> FP8 GEMMs
```

Configuration (env var or `sam3_params::trt` field — struct wins when
`.enabled`):

| Engine | env var | `sam3_trt_config` field |
|---|---|---|
| encoder FP16 | `SAM3_TRT_ONNX_PATH` | `encoder_onnx` |
| encoder FP8 | `SAM3_TRT_ONNX_PATH_FP8` | `encoder_onnx_fp8` |
| PCS FP16 | `SAM3_TRT_PCS_ONNX_PATH` | `pcs_onnx` |
| PCS FP8 | `SAM3_TRT_PCS_ONNX_PATH_FP8` | `pcs_onnx_fp8` |
| PVS | `SAM3_TRT_PVS_ONNX_PATH` | `pvs_onnx` |
| engine caches | `SAM3_TRT_*_CACHE_DIR` | `cache_dir` (+ `/encoder` `/pcs` `/pvs`) |
| PCS mixed precision | `SAM3_TRT_PCS_PRECISION` | `pcs_precision` (default `mixed:text_`) |

`mixed:text_` (pin every layer whose name starts with `text_` to FP32) stays
in force for the FP8 PCS graph too — Q/DQ quantizes the fenc/ddec GEMMs while
the precision constraint keeps the text encoder FP32.

## Generating the FP8 graphs

```bash
# 1. Encoder: calibrate once (split-graph, fits 8–16 GB RAM), inject Q/DQ
python3 scripts/convert/fp8_amax_calib.py sam3_encoder.onnx <calib_imgs> amax.json
# a validated amax for sam3-q8_0 ships at resources/fp8_amax_sam3-q8_0.json

# 2. PCS: capture REAL graph inputs, calibrate, inject
./build/examples/sam3cpp_dump_pcs_calib_inputs model.ggml calib/ img1.jpg img2.jpg ...
python3 scripts/convert/fp8_pcs_amax_calib.py sam3_pcs.onnx calib/ pcs_amax.json
# a validated PCS amax ships at resources/fp8_pcs_amax_sam3-q8_0.json

# 3. Or all at once through the exporter:
scripts/development/export_onnx.sh --model model.ggml \
    --fp8-amax resources/fp8_amax_sam3-q8_0.json \
    --pcs-fp8-amax resources/fp8_pcs_amax_sam3-q8_0.json
```

Operational note: TensorRT engine **builds** want the GPU largely free — stop
the serving process first (`release/sam3/scripts/start.sh --stop`) or the
builder skips tactics for lack of memory and produces a slower engine.

## Runnable example

`examples/features/09_tensorrt_config.cpp` (programmatic config + encoder
FP8 switch) and `examples/features/10_pcs_fp8.cpp` (PCS FP8 switch, accuracy
and timing comparison).

Measured PCS FP8 trade-off (RTX 5060, sam3-q8_0, cat.jpg, vs the CPU
goldens): text score 0.9605 vs FP16's 0.9627 (golden 0.9613), exemplar case
0.9479 vs golden 0.9498, boxes within 1.6 px; mask pixel counts drift up to
~5.4% on SMALL masks (boundary jitter -- large masks stay within 0.1%);
timing parity to ~6% faster warm; engine ~1% smaller (PCS weights are
FP32-in-graph, only the 132 quantized GEMMs shrink). Treat it like the
encoder's FP8: an opt-in precision/VRAM experiment knob, not a default.
