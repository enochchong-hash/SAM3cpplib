// 03 — PVS: segment the object under a point (and refine with more points)
//
// PVS ("promptable visual segmentation") segments ONE object indicated by
// geometric prompts -- the classic interactive SAM workflow:
//   * positive points ("this pixel is on the object")
//   * negative points ("this pixel is NOT on the object")
//   * optionally a box (example 04)
// Coordinates are in ORIGINAL image pixels. Returns one detection whose
// `iou_score` is the model's own quality estimate for the mask.
//
// This example runs twice on the same encoded image:
//   run A: single positive point
//   run B: two positives + one negative -- watch the mask change
//
// Usage: sam3cpp_ex_03_point_prompts <model.ggml> <image> [x y]
#include "sam3cpp/sam3.h"

#include <cstdio>
#include <cstdlib>

static void report(const char* label, const sam3_result& res) {
    printf("%s: %zu detection(s)\n", label, res.detections.size());
    for (const auto& d : res.detections) {
        printf("  iou=%.4f obj_score=%.2f box=(%.0f,%.0f)-(%.0f,%.0f) area=%zu px\n",
               d.iou_score, d.mask.obj_score,
               d.box.x0, d.box.y0, d.box.x1, d.box.y1, sam3_mask_area(d.mask));
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.ggml> <image> [x=600 y=600]\n", argv[0]);
        return 1;
    }
    const float px = argc > 3 ? strtof(argv[3], nullptr) : 600;
    const float py = argc > 4 ? strtof(argv[4], nullptr) : 600;

    sam3_params params;
    params.model_path = argv[1];
    auto model = sam3_load_model(params);
    auto state = model ? sam3_create_state(*model, params) : nullptr;
    sam3_image image = sam3_load_image(argv[2]);
    if (!state || image.data.empty()) return 1;
    if (!sam3_encode_image(*state, *model, image)) return 1;   // once!

    // ── A: one positive point ────────────────────────────────────────────
    sam3_pvs_params one;
    one.pos_points.push_back({px, py});
    report("A: single point", sam3_segment_pvs(*state, *model, one));

    // ── B: refine -- add a second positive and a negative point ─────────
    // The negative point pushes the mask away from that region; this is the
    // interactive click-to-fix loop, and each call costs only the PVS head
    // (~7 ms on TensorRT), not a re-encode.
    sam3_pvs_params multi;
    multi.pos_points.push_back({px, py});
    multi.pos_points.push_back({px * 0.5f, py * 0.5f});
    multi.neg_points.push_back({image.width * 0.92f, image.height * 0.12f});
    report("B: 2 pos + 1 neg", sam3_segment_pvs(*state, *model, multi));

    return 0;
}
