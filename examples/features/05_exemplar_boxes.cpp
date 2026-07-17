// 05 — PCS: exemplar boxes ("find things that look like THIS")
//
// Instead of (or in addition to) text, PCS accepts exemplar boxes drawn on
// the SAME image: the geometry encoder ROI-pools backbone features from each
// box and turns them into prompt tokens.
//   pos_exemplars -- "instances look like this"
//   neg_exemplars -- "...but not like this" (suppresses lookalikes)
//
// NOTE the coordinate convention difference:
//   PVS points/boxes      -> PIXELS
//   PCS exemplar boxes    -> NORMALIZED [0,1] XYXY (divide by width/height)
//
// Exemplars compose with a text prompt (both prompt the same detector) or
// can be used with text="" for a pure visual-example query.
//
// Usage: sam3cpp_ex_05_exemplar_boxes <model.ggml> <image> [x0 y0 x1 y1 in px]
#include "sam3cpp/sam3.h"

#include <cstdio>
#include <cstdlib>

static void report(const char* label, const sam3_result& res) {
    printf("%s: %zu detection(s)\n", label, res.detections.size());
    for (const auto& d : res.detections)
        printf("  score=%.3f box=(%.0f,%.0f)-(%.0f,%.0f) area=%zu px\n",
               d.score, d.box.x0, d.box.y0, d.box.x1, d.box.y1,
               sam3_mask_area(d.mask));
}

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

    // Exemplar box in pixels (default: a region on the bundled cat image),
    // then normalized to [0,1] as the API requires.
    float ex[4] = {300, 150, 900, 1100};
    if (argc > 6) for (int i = 0; i < 4; ++i) ex[i] = strtof(argv[3 + i], nullptr);
    const sam3_box ex_norm = {ex[0] / image.width, ex[1] / image.height,
                              ex[2] / image.width, ex[3] / image.height};

    // ── A: text + positive exemplar (exemplar sharpens the text query) ──
    sam3_pcs_params a;
    a.text_prompt = "cat";
    a.pos_exemplars.push_back(ex_norm);
    report("A: text 'cat' + positive exemplar", sam3_segment_pcs(*state, *model, a));

    // ── B: exemplar only (no text) -- pure query-by-example ─────────────
    sam3_pcs_params b;
    b.pos_exemplars.push_back(ex_norm);
    report("B: positive exemplar only", sam3_segment_pcs(*state, *model, b));

    // ── C: negative exemplar -- suppress a region's lookalikes ──────────
    // Marking the same region NEGATIVE tells the detector instances like it
    // are NOT the concept: with 'cat' + a negative exemplar over the cat,
    // detections should disappear (or lose confidence).
    sam3_pcs_params c;
    c.text_prompt = "cat";
    c.neg_exemplars.push_back(ex_norm);
    report("C: text 'cat' + NEGATIVE exemplar over it", sam3_segment_pcs(*state, *model, c));

    return 0;
}
