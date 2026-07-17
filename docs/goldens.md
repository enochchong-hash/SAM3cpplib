# The golden-sample process

The **ggml CPU backend is the numerical reference** for this library. It is
deliberately kept even though it is ~25-30x slower than CUDA: its outputs on
the standard prompt cases are the "goldens" that every faster backend (ggml
CUDA, TensorRT FP16, TensorRT FP8) is judged against.

## Files

| Piece | Role |
|---|---|
| `tests/data/cat.jpg` | the standard test image (1200x1198) |
| `scripts/development/make_goldens.sh` | runs the CPU backend over the case table, writes `tests/goldens/<case>/{pcs,pvs}_detections.txt` |
| `tests/goldens/` | committed reference outputs (regenerate only on intentional model/reference changes) |
| `tests/parity_test.sh` | runs the same cases on CUDA (+ TensorRT with `--trt-onnx-dir/--trt-cache-dir`) and compares against the goldens |

## Case table (kept in sync between the two scripts)

| Case | Prompts | Exercises |
|---|---|---|
| `text_point` | text "cat" + point (600,600) | PCS text path, PVS single positive point |
| `exemplar_multi` | text + positive exemplar box + 2 points + 1 negative point | geometry encoder (ROI-align), PVS multi/negative tokens |
| `box_prompt` | text + box prompt | PVS box corner tokens |

## Tolerances (tests/parity_test.sh)

- detection **count**: exact
- `score` / `iou`: ±0.02 (`SCORE_TOL`) — the same gate production validation
  used for the ggml→CUDA and CUDA→TensorRT migrations
- `mask_px`: ±2% relative (`MASK_TOL`)
- box corners: ±8 px (`BOX_TOL`)

CPU-vs-CPU comparisons are expected to be **bitwise identical** (same code,
same weights); the tolerances exist for the cross-backend comparisons.

## Typical flow

```bash
# once per model / intentional reference change (slow: CPU encodes)
scripts/development/make_goldens.sh --model resources/models/sam3-q8_0.ggml

# every change (CUDA only)
tests/parity_test.sh --model resources/models/sam3-q8_0.ggml

# with TensorRT engines too
tests/parity_test.sh --model resources/models/sam3-q8_0.ggml \
    --trt-onnx-dir resources/onnx --trt-cache-dir var/trt_cache
```

Reference numbers on the RTX 5060 dev box (sam3-q8_0, cat.jpg, `text_point`):
CPU PCS score 0.961301 / PVS iou 0.985699; CUDA 0.961892 / 0.985920;
TensorRT FP16 0.962549 / 0.985965; TensorRT FP8 0.963656 / 0.985543.
