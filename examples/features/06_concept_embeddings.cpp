// 06 — PCS: persistable concept embeddings (reference-image workflow)
//
// sam3_pcs_compute_exemplar_embedding turns one exemplar box on the
// CURRENTLY ENCODED image into a 256-float row -- the geometry encoder's
// output token for that box. That row can be
//   * persisted (1 KB per concept -- a "concept library" on disk), and
//   * injected later via sam3_pcs_params::exemplar_embeddings on ANY image,
//     without re-encoding the reference image.
//
// EXPERIMENTAL: applying an embedding captured on image A to image B is
// outside the model's training distribution (validated project finding:
// same-image works well; cross-image transfer is weak) -- validate per use
// case. This example captures on <ref-image> and applies on <image>; pass
// the same file for both to see the well-supported same-image behavior.
//
// Usage: sam3cpp_ex_06_concept_embeddings <model.ggml> <image> [ref-image]
#include "sam3cpp/sam3.h"

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.ggml> <image> [ref-image=<image>]\n", argv[0]);
        return 1;
    }
    const char* target_path = argv[2];
    const char* ref_path    = argc > 3 ? argv[3] : argv[2];

    sam3_params params;
    params.model_path = argv[1];
    auto model = sam3_load_model(params);
    auto state = model ? sam3_create_state(*model, params) : nullptr;
    if (!state) return 1;

    // ── 1. Capture: encode the REFERENCE image, compute the embedding ───
    sam3_image ref = sam3_load_image(ref_path);
    if (ref.data.empty()) return 1;
    if (!sam3_encode_image(*state, *model, ref)) return 1;

    const sam3_box box_norm = {300.f / ref.width, 150.f / ref.height,
                               900.f / ref.width, 1100.f / ref.height};
    std::vector<float> concept =
        sam3_pcs_compute_exemplar_embedding(*state, *model, box_norm, /*positive=*/true);
    if (concept.empty()) return 1;
    printf("captured concept embedding: %zu floats (%zu bytes -- persist this)\n",
           concept.size(), concept.size() * sizeof(float));

    // ── 2. Persist / reload -- exactly what a concept library would do ──
    {
        FILE* f = fopen("concept.bin", "wb");
        fwrite(concept.data(), sizeof(float), concept.size(), f);
        fclose(f);
        concept.assign(256, 0.f);
        f = fopen("concept.bin", "rb");
        size_t got = fread(concept.data(), sizeof(float), 256, f);
        fclose(f);
        printf("round-tripped concept.bin (%zu floats)\n", got);
    }

    // ── 3. Apply: encode the TARGET image, prompt with the stored row ───
    // No reference image needed anymore -- the whole point of the feature.
    sam3_image target = sam3_load_image(target_path);
    if (target.data.empty()) return 1;
    if (!sam3_encode_image(*state, *model, target)) return 1;

    sam3_pcs_params pcs;
    pcs.exemplar_embeddings.push_back(concept);   // one row per stored concept
    // pcs.text_prompt = "cat";                   // optionally combine with text
    sam3_result res = sam3_segment_pcs(*state, *model, pcs);

    printf("stored-concept query on %s: %zu detection(s)\n",
           target_path, res.detections.size());
    for (const auto& d : res.detections)
        printf("  score=%.3f box=(%.0f,%.0f)-(%.0f,%.0f)\n",
               d.score, d.box.x0, d.box.y0, d.box.x1, d.box.y1);
    return 0;
}
