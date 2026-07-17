// 08 — Mask convenience accessors: ready-made geometry from a mask
//
// A sam3_mask is width x height bytes, 0 or 255 (foreground = byte > 127),
// at the ORIGINAL image resolution. Instead of scanning bytes yourself:
//
//   sam3_mask_area(m)      foreground pixel count
//   sam3_mask_centroid(m)  center of mass, {-1,-1} if empty
//   sam3_mask_bbox(m)      tight box; x1/y1 EXCLUSIVE (width = x1-x0);
//                          may be tighter than detection.box (the model's
//                          predicted box)
//   sam3_mask_at(m,x,y)    range-checked foreground test
//   sam3_mask_coords(m)    all foreground pixels, row-major
//
// All pure CPU functions -- usable on any sam3_mask, no model needed.
//
// Usage: sam3cpp_ex_08_mask_geometry <model.ggml> <image> [x y]
#include "sam3cpp/sam3.h"

#include <cstdio>
#include <cstdlib>

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
    if (!sam3_encode_image(*state, *model, image)) return 1;

    sam3_pvs_params pvs;
    pvs.pos_points.push_back({px, py});
    sam3_result res = sam3_segment_pvs(*state, *model, pvs);
    if (res.detections.empty()) { printf("no detection\n"); return 0; }
    const sam3_mask& m = res.detections[0].mask;

    // ── Geometry without touching mask bytes ────────────────────────────
    const size_t     area = sam3_mask_area(m);
    const sam3_point c    = sam3_mask_centroid(m);
    const sam3_box   bb   = sam3_mask_bbox(m);

    printf("mask %dx%d\n", m.width, m.height);
    printf("  area      : %zu px (%.1f%% of image)\n",
           area, 100.0 * area / ((double)m.width * m.height));
    printf("  centroid  : (%.1f, %.1f)\n", c.x, c.y);
    printf("  tight bbox: (%.0f,%.0f)-(%.0f,%.0f)  [%gx%g]\n",
           bb.x0, bb.y0, bb.x1, bb.y1, bb.x1 - bb.x0, bb.y1 - bb.y0);
    printf("  model box : (%.0f,%.0f)-(%.0f,%.0f)  (predicted, may be looser)\n",
           res.detections[0].box.x0, res.detections[0].box.y0,
           res.detections[0].box.x1, res.detections[0].box.y1);

    // Point membership -- e.g. "did the user click inside the mask?"
    printf("  at(%.0f,%.0f) : %s   at(0,0): %s\n", px, py,
           sam3_mask_at(m, (int)px, (int)py) ? "foreground" : "background",
           sam3_mask_at(m, 0, 0)             ? "foreground" : "background");

    // Full coordinate list -- for downstream geometry (hulls, sampling, ...).
    auto coords = sam3_mask_coords(m);
    printf("  coords    : %zu points, first=(%.0f,%.0f) last=(%.0f,%.0f)\n",
           coords.size(), coords.front().x, coords.front().y,
           coords.back().x, coords.back().y);

    // The centroid is a natural anchor for labels/UI markers:
    printf("  centroid is %s the mask\n",
           sam3_mask_at(m, (int)c.x, (int)c.y) ? "inside" : "outside (concave shape)");
    return 0;
}
