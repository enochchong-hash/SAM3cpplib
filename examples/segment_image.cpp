// segment_image -- minimal sam3cpplib consumer demo:
//   load model -> encode image once -> prompt (text or point) -> save masks,
// printing the convenience geometry (area / centroid / bbox) per detection.
//
// Usage:
//   segment_image --model m.ggml --image cat.jpg --text "cat"
//   segment_image --model m.ggml --image cat.jpg --point 600,600
#include "sam3cpp/sam3.h"

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    sam3_params params;
    std::string image_path, text;
    float px = -1, py = -1;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model") && i + 1 < argc) { params.model_path = argv[++i]; }
        else if (!strcmp(argv[i], "--image") && i + 1 < argc) { image_path = argv[++i]; }
        else if (!strcmp(argv[i], "--text")  && i + 1 < argc) { text = argv[++i]; }
        else if (!strcmp(argv[i], "--point") && i + 1 < argc) { sscanf(argv[++i], "%f,%f", &px, &py); }
        else if (!strcmp(argv[i], "--no-gpu"))                { params.use_gpu = false; }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (params.model_path.empty() || image_path.empty() || (text.empty() && px < 0)) {
        fprintf(stderr, "usage: %s --model m.ggml --image i.jpg (--text t | --point x,y) [--no-gpu]\n",
                argv[0]);
        return 1;
    }

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    auto state = sam3_create_state(*model, params);
    sam3_image image = sam3_load_image(image_path);
    if (!state || image.data.empty()) { fprintf(stderr, "state/image load failed\n"); return 1; }

    // Encode once; prompt as many times as you like afterwards.
    if (!sam3_encode_image(*state, *model, image)) { fprintf(stderr, "encode failed\n"); return 1; }

    sam3_result result;
    if (!text.empty()) {
        sam3_pcs_params pcs;
        pcs.text_prompt = text;
        result = sam3_segment_pcs(*state, *model, pcs);
    } else {
        sam3_pvs_params pvs;
        pvs.pos_points.push_back({px, py});
        result = sam3_segment_pvs(*state, *model, pvs);
    }

    printf("%zu detection(s)\n", result.detections.size());
    int idx = 0;
    for (const auto& d : result.detections) {
        const auto area = sam3_mask_area(d.mask);
        const auto c    = sam3_mask_centroid(d.mask);
        const auto bb   = sam3_mask_bbox(d.mask);
        printf("  [%d] score=%.4f box=(%.1f,%.1f,%.1f,%.1f) mask: area=%zu px, "
               "centroid=(%.1f,%.1f), tight bbox=(%.0f,%.0f,%.0f,%.0f)\n",
               idx, d.score > 0 ? d.score : d.iou_score,
               d.box.x0, d.box.y0, d.box.x1, d.box.y1,
               area, c.x, c.y, bb.x0, bb.y0, bb.x1, bb.y1);
        char path[256];
        snprintf(path, sizeof(path), "mask_%d.png", idx);
        if (sam3_save_mask(d.mask, path)) printf("      wrote %s\n", path);
        ++idx;
    }
    return 0;
}
