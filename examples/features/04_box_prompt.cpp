// 04 — PVS: box prompt and multimask (ambiguity) mode
//
// A box prompt segments the dominant object inside the box (pixel coords,
// set use_box=true). Boxes are usually less ambiguous than a single point.
//
// `multimask = true` asks the SAM decoder for its 3 candidate
// interpretations (whole object / part / subpart) instead of the single
// best mask -- each with its own iou_score, so the caller can pick.
//
// Usage: sam3cpp_ex_04_box_prompt <model.ggml> <image> [x0 y0 x1 y1]
#include "sam3cpp/sam3.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.ggml> <image> [x0 y0 x1 y1]\n", argv[0]);
        return 1;
    }
    sam3_params params;
    params.model_path = argv[1];
    auto model = sam3_load_model(params);
    auto state = model ? sam3_create_state(*model, params) : nullptr;
    sam3_image image = sam3_load_image(argv[2]);
    if (!state || image.data.empty()) return 1;
    if (!sam3_encode_image(*state, *model, image)) return 1;

    sam3_box box;
    if (argc > 6) {
        box = {strtof(argv[3],0), strtof(argv[4],0), strtof(argv[5],0), strtof(argv[6],0)};
    } else {
        // default: a box that tightly frames the subject on the bundled cat
        // image (the same case tests/goldens/box_prompt validates). A sloppy,
        // much-too-large box gives the decoder little to anchor on and the
        // iou_score drops accordingly -- try it: pass 120 120 1080 1078.
        box = {150, 80, 1000, 1150};
    }

    // ── A: box, single best mask ─────────────────────────────────────────
    sam3_pvs_params pvs;
    pvs.box     = box;
    pvs.use_box = true;
    sam3_result best = sam3_segment_pvs(*state, *model, pvs);
    printf("A: box (%.0f,%.0f)-(%.0f,%.0f), best mask:\n", box.x0, box.y0, box.x1, box.y1);
    for (const auto& d : best.detections)
        printf("  iou=%.4f area=%zu px\n", d.iou_score, sam3_mask_area(d.mask));

    // ── B: same box, all 3 ambiguity candidates ─────────────────────────
    // On an ambiguous box (like this default) the single-mask path can pick
    // a low-confidence interpretation; the multimask candidates expose the
    // alternatives so the caller chooses by iou_score (or shows all three).
    pvs.multimask = true;
    sam3_result all = sam3_segment_pvs(*state, *model, pvs);
    size_t best_i = 0;
    for (size_t i = 1; i < all.detections.size(); ++i)
        if (all.detections[i].iou_score > all.detections[best_i].iou_score) best_i = i;
    printf("B: multimask candidates (%zu):\n", all.detections.size());
    for (size_t i = 0; i < all.detections.size(); ++i) {
        const auto& d = all.detections[i];
        printf("  [%zu] iou=%.4f area=%zu px%s\n", i, d.iou_score,
               sam3_mask_area(d.mask), i == best_i ? "  <- highest iou" : "");
    }
    // Points and a box can also be combined in one sam3_pvs_params -- e.g.
    // a box plus a negative point to exclude a region inside it.
    return 0;
}
