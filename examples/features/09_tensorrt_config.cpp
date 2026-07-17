// 09 — TensorRT: programmatic configuration + FP16/FP8 runtime switch
//
// Requires a library built with -DSAM3CPP_TENSORRT=ON (after
// scripts/setup/setup_tensorrt.sh) and the exported ONNX graphs
// (scripts/development/export_onnx.sh). Without that build this example still runs --
// on the ggml backend, with a note.
//
// An embedded host app configures TensorRT through sam3_params::trt instead
// of mutating its own environment (the SAM3_TRT_* env vars keep working and
// win when params.trt.enabled is false):
//
//   trt.enabled          master switch
//   trt.runtime_data     thin hparams/tokenizer/helper sidecar; setting this
//                       means the full model checkpoint is never opened
//   trt.encoder_onnx     image encoder (FP16 engine built on first use,
//   trt.encoder_onnx_fp8 optional FP8-quantized variant       then cached)
//   trt.pcs_onnx/pvs_onnx  text head / point head
//   trt.cache_dir        engine cache root (subdirs /encoder /pcs /pvs)
//   trt.pcs_precision    "mixed:text_" (validated default) | fp32 | fp16
//   trt.skip_ggml_weights true = don't load the ~1.1GB ggml weights at all
//                         (TRT-only serving, no fallback -- the deployed mode)
//
// sam3_set_encoder_fp8(state, true) switches the ENCODER to the FP8 engine
// per state at runtime (-450MB VRAM, slightly faster, slightly lower
// accuracy); both engines stay resident once used, switching is free.
//
// VRAM note: holding BOTH encoder engines + PCS + PVS needs ~4 GB free.
// If the FP8 engine cannot be loaded (e.g. another process holds the VRAM),
// the library logs "FP8 encoder requested but unavailable ... using FP16"
// on stderr and continues on FP16 -- watch for identical FP16/FP8 scores
// below as the tell-tale.
//
// Usage: sam3cpp_ex_09_tensorrt_config <image> <onnx-dir> <cache-dir>
#include "sam3cpp/sam3.h"

#include <chrono>
#include <cstdio>
#include <string>

static double encode_ms(sam3_state& st, const sam3_model& m, const sam3_image& img) {
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!sam3_encode_image(st, m, img)) return -1;
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <image> <onnx-dir> <engine-cache-dir>\n", argv[0]);
        return 1;
    }
    const std::string onnx = argv[2];

    sam3_params params;

    // ── Programmatic TensorRT config -- no environment variables ────────
    params.trt.enabled          = true;
    params.trt.runtime_data     = onnx + "/sam3_runtime.sam3rt";
    params.trt.encoder_onnx     = onnx + "/sam3_encoder.onnx";
    params.trt.encoder_onnx_fp8 = onnx + "/sam3_encoder_fp8.onnx";  // optional
    params.trt.pcs_onnx         = onnx + "/sam3_pcs.onnx";
    params.trt.pvs_onnx         = onnx + "/sam3_pvs.onnx";
    params.trt.cache_dir        = argv[3];
    // params.trt.pcs_precision    = "mixed:text_";  // default (validated)
    // params.trt.skip_ggml_weights = true;          // default: TRT-only, no fallback

    auto model = sam3_load_model(params);
    auto state = model ? sam3_create_state(*model, params) : nullptr;
    sam3_image image = sam3_load_image(argv[1]);
    if (!state || image.data.empty()) return 1;

    // ── FP16 encoder (default) ───────────────────────────────────────────
    double cold = encode_ms(*state, *model, image);          // may build/load engines
    double warm = encode_ms(*state, *model, image);
    printf("FP16 encoder: cold %.0f ms (engine build/load), warm %.1f ms\n", cold, warm);

    sam3_pcs_params pcs; pcs.text_prompt = "cat";
    sam3_result r16 = sam3_segment_pcs(*state, *model, pcs);

    // ── Switch this state to the FP8 encoder and compare ────────────────
    // In a non-TensorRT build this logs "needs a SAM3_TRT_ENCODER build"
    // and encoding continues on ggml unchanged.
    sam3_set_encoder_fp8(*state, true);
    cold = encode_ms(*state, *model, image);
    warm = encode_ms(*state, *model, image);
    printf("FP8  encoder: cold %.0f ms, warm %.1f ms\n", cold, warm);
    sam3_result r8 = sam3_segment_pcs(*state, *model, pcs);

    if (!r16.detections.empty() && !r8.detections.empty())
        printf("'cat' score: FP16 %.4f vs FP8 %.4f (validated |delta| <= 0.02)\n",
               r16.detections[0].score, r8.detections[0].score);

    sam3_set_encoder_fp8(*state, false);   // and back -- switching is free
    printf("switched back to FP16 -- both engines stay resident\n");
    return 0;
}
