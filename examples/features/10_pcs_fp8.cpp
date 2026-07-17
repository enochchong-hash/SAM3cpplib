// 10 — TensorRT: the PCS FP8 switch (companion to 09's encoder FP8)
//
// The PCS (text/exemplar) head gains an FP8 option: the fusion-encoder and
// DETR-decoder linear GEMMs run as E4M3 FP8, while
//   * the text encoder stays FP32   (precision-sensitive),
//   * attention stays FP16          (TensorRT fused MHA -- the 97->35 ms win),
//   * geometry/scoring/seg-head/bbox+RPB MLPs stay FP16 (tiny or delicate).
// See docs/tensorrt.md for the full per-subsystem precision map, including
// why PVS deliberately has NO reduced-precision variant (FP32 floor).
//
// Same contract as the encoder switch: per-state, both engines resident
// once used, free to flip, graceful fallback with a stderr log.
//
// Needs a SAM3CPP_TENSORRT build + sam3_pcs_fp8.onnx (scripts/export_onnx.sh
// --pcs-fp8-amax resources/fp8_pcs_amax_sam3-q8_0.json).
//
// Usage: sam3cpp_ex_10_pcs_fp8 <model.ggml> <image> <onnx-dir> <cache-dir>
#include "sam3cpp/sam3.h"

#include <chrono>
#include <cstdio>
#include <string>

static double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.ggml> <image> <onnx-dir> <engine-cache-dir>\n", argv[0]);
        return 1;
    }
    const std::string onnx = argv[3];

    sam3_params params;
    params.model_path = argv[1];
    params.trt.enabled      = true;
    params.trt.encoder_onnx = onnx + "/sam3_encoder.onnx";
    params.trt.pcs_onnx     = onnx + "/sam3_pcs.onnx";
    params.trt.pcs_onnx_fp8 = onnx + "/sam3_pcs_fp8.onnx";   // the new option
    params.trt.pvs_onnx     = onnx + "/sam3_pvs.onnx";
    params.trt.cache_dir    = argv[4];

    auto model = sam3_load_model(params);
    auto state = model ? sam3_create_state(*model, params) : nullptr;
    sam3_image image = sam3_load_image(argv[2]);
    if (!state || image.data.empty()) return 1;
    if (!sam3_encode_image(*state, *model, image)) return 1;   // FP16 encoder

    sam3_pcs_params pcs;
    pcs.text_prompt = "cat";

    // ── FP16 PCS (default) ──────────────────────────────────────────────
    sam3_segment_pcs(*state, *model, pcs);                    // warmup/engine load
    auto t0 = std::chrono::high_resolution_clock::now();
    sam3_result r16 = sam3_segment_pcs(*state, *model, pcs);
    double t_fp16 = ms_since(t0);

    // ── Switch this state's PCS head to FP8 ─────────────────────────────
    sam3_set_pcs_fp8(*state, true);
    sam3_segment_pcs(*state, *model, pcs);                    // engine build/load
    t0 = std::chrono::high_resolution_clock::now();
    sam3_result r8 = sam3_segment_pcs(*state, *model, pcs);
    double t_fp8 = ms_since(t0);

    if (!r16.detections.empty() && !r8.detections.empty()) {
        printf("PCS 'cat'   FP16: score=%.4f in %.1f ms\n", r16.detections[0].score, t_fp16);
        printf("PCS 'cat'   FP8 : score=%.4f in %.1f ms\n", r8.detections[0].score, t_fp8);
        printf("delta %.4f (gate: <=0.02); identical scores mean the FP8 engine\n"
               "was unavailable and the library fell back -- check stderr\n",
               r16.detections[0].score - r8.detections[0].score);
    }

    sam3_set_pcs_fp8(*state, false);   // back to FP16; both engines resident
    printf("switched back -- flipping per request is free\n");
    return 0;
}
