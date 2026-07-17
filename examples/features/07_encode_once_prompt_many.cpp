// 07 — Partial inference: encode once, prompt many times
//
// The core serving pattern this library is built around. Image encoding
// dominates the cost; every prompt afterwards only runs its (much smaller)
// head against the features cached in sam3_state:
//
//              TensorRT   ggml-CUDA    (RTX 5060, sam3-q8_0)
//   encode      ~120 ms     ~870 ms     once per image
//   PCS text     ~36 ms     ~220 ms     per prompt
//   PVS point     ~7 ms      ~65 ms     per prompt
//
// So an interactive UI (or a multi-query pipeline) should hold one state
// per image and fire prompts at it -- exactly what this example measures.
//
// Usage: sam3cpp_ex_07_encode_once_prompt_many <model.ggml> <image>
#include "sam3cpp/sam3.h"

#include <chrono>
#include <cstdio>

static double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.ggml> <image>\n", argv[0]); return 1; }
    sam3_params params;
    params.model_path = argv[1];
    auto model = sam3_load_model(params);
    auto state = model ? sam3_create_state(*model, params) : nullptr;
    sam3_image image = sam3_load_image(argv[2]);
    if (!state || image.data.empty()) return 1;

    // ── Pay the encode cost once ─────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!sam3_encode_image(*state, *model, image)) return 1;
    printf("%-34s %8.1f ms\n", "encode (once)", ms_since(t0));

    // ── Then every prompt is cheap ───────────────────────────────────────
    struct { const char* label; sam3_pvs_params pvs; } clicks[] = {
        {"PVS click #1 (600,600)",   {}},
        {"PVS click #2 (300,300)",   {}},
        {"PVS click #3 (900,900)",   {}},
    };
    clicks[0].pvs.pos_points.push_back({600, 600});
    clicks[1].pvs.pos_points.push_back({300, 300});
    clicks[2].pvs.pos_points.push_back({900, 900});

    for (auto& c : clicks) {
        t0 = std::chrono::high_resolution_clock::now();
        sam3_result r = sam3_segment_pvs(*state, *model, c.pvs);
        printf("%-34s %8.1f ms   (iou=%.3f)\n", c.label, ms_since(t0),
               r.detections.empty() ? 0.f : r.detections[0].iou_score);
    }

    const char* queries[] = {"cat", "ear", "grass"};
    for (const char* q : queries) {
        sam3_pcs_params pcs;
        pcs.text_prompt = q;
        t0 = std::chrono::high_resolution_clock::now();
        sam3_result r = sam3_segment_pcs(*state, *model, pcs);
        printf("PCS text '%-8s'                  %8.1f ms   (%zu det)\n",
               q, ms_since(t0), r.detections.size());
    }

    printf("\nnote: first PCS/PVS call after load includes one-time warmup;\n"
           "      steady-state per-prompt cost is what calls 2+ show.\n");
    return 0;
}
