// 02 — PCS: segment everything matching a text prompt
//
// PCS ("promptable concept segmentation") finds ALL instances of a concept.
// The text prompt runs through the model's own CLIP-style tokenizer + text
// encoder (embedded in the .ggml file -- nothing to configure), is fused
// with the image features, and a DETR-style decoder proposes up to 200
// candidates which are score-thresholded and NMS-deduplicated.
//
// Two knobs on sam3_pcs_params:
//   score_threshold (default 0.5) -- drop candidates below this confidence
//   nms_threshold   (default 0.1) -- IoU above which overlapping detections
//                                    are considered duplicates
//
// Usage: sam3cpp_ex_02_text_prompt <model.ggml> <image> [text] [score_thr]
#include "sam3cpp/sam3.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.ggml> <image> [text=cat] [score_thr=0.5]\n", argv[0]);
        return 1;
    }
    sam3_params params;
    params.model_path = argv[1];

    auto model = sam3_load_model(params);
    auto state = model ? sam3_create_state(*model, params) : nullptr;
    sam3_image image = sam3_load_image(argv[2]);
    if (!state || image.data.empty()) return 1;
    if (!sam3_encode_image(*state, *model, image)) return 1;

    sam3_pcs_params pcs;
    pcs.text_prompt     = argc > 3 ? argv[3] : "cat";
    pcs.score_threshold = argc > 4 ? strtof(argv[4], nullptr) : 0.5f;
    // pcs.nms_threshold = 0.1f;  // default; raise to keep tighter overlaps

    sam3_result res = sam3_segment_pcs(*state, *model, pcs);

    printf("'%s' (score>=%.2f): %zu detection(s)\n",
           pcs.text_prompt.c_str(), pcs.score_threshold, res.detections.size());
    for (size_t i = 0; i < res.detections.size(); ++i) {
        const auto& d = res.detections[i];
        // d.box   -- model-predicted box in ORIGINAL image pixels
        // d.score -- concept confidence in [0,1]
        // d.mask  -- full-resolution binary mask (width x height, 0/255)
        printf("  [%zu] score=%.3f box=(%.0f,%.0f)-(%.0f,%.0f) mask=%dx%d\n",
               i, d.score, d.box.x0, d.box.y0, d.box.x1, d.box.y1,
               d.mask.width, d.mask.height);
    }
    return 0;
}
