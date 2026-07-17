// Offline tool: capture REAL PCS-graph input tensors for FP8 amax
// calibration (scripts/convert/fp8_pcs_amax_calib.py). For each image the
// tool encodes it and dumps the three backbone feature inputs; it also
// tokenizes a small representative prompt set and computes geometry rows
// (CLS-only and with exemplar boxes) exactly the way the runtime feeds the
// TensorRT PCS engine.
//
// This is dev tooling for the quantization pipeline, not consumer API -- it
// deliberately reaches into sam3_internal.h for sam3_precompute_geom_input
// and the embedded tokenizer.
//
// Usage: sam3cpp_dump_pcs_calib_inputs <model.ggml> <out_dir> <image> [image...]
#include "sam3_internal.h"

#include <cstdio>
#include <string>
#include <vector>

static bool write_bin(const std::string& path, const void* p, size_t bytes) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return false; }
    fwrite(p, 1, bytes, f);
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.ggml> <out_dir> <image> [image...]\n", argv[0]);
        return 1;
    }
    const std::string out = argv[2];
    mkdir(out.c_str(), 0755);

    sam3_params params;
    params.model_path = argv[1];
    auto model = sam3_load_model(params);
    auto state = model ? sam3_create_state(*model, params) : nullptr;
    if (!state) return 1;

    // Representative text prompts (diverse token counts / content).
    const char* prompts[] = {"cat", "person", "bus", "a small object on the grass"};
    for (size_t p = 0; p < sizeof(prompts) / sizeof(*prompts); ++p) {
        auto ids = sam3_tokenize(const_cast<sam3_model&>(*model).tokenizer,
                                 prompts[p], model->hparams.text_ctx_len);
        write_bin(out + "/tokens_" + std::to_string(p) + ".bin",
                  ids.data(), ids.size() * sizeof(int32_t));
    }

    const int H = model->hparams.n_img_embd();   // 72
    const int D = model->hparams.neck_dim;       // 256

    for (int i = 3; i < argc; ++i) {
        const int k = i - 3;
        sam3_image image = sam3_load_image(argv[i]);
        if (image.data.empty()) { fprintf(stderr, "skip %s\n", argv[i]); continue; }
        if (!sam3_encode_image(*state, *model, image)) return 1;
        const std::string pre = out + "/img" + std::to_string(k) + "_";

        // The three feature inputs, exactly as trt_pcs.cpp binds them:
        // img_feats = neck_det[2] (72x72), fpn0 = neck_det[0] (288x288),
        // fpn1 = neck_det[1] (144x144); ggml [D,W,H] bytes == ONNX [1,H,W,D].
        sam3_dump_state_tensor(*state, "neck_det_2", pre + "img_feats.bin");
        sam3_dump_state_tensor(*state, "neck_det_0", pre + "fpn0.bin");
        sam3_dump_state_tensor(*state, "neck_det_1", pre + "fpn1.bin");

        // Geometry rows from these features (host copy first).
        std::vector<float> feats((size_t)D * H * H);
        ggml_backend_tensor_get(state->neck_det[2], feats.data(), 0,
                                feats.size() * sizeof(float));

        sam3_pcs_params none;                     // CLS token only (n_geo=1)
        auto g1 = sam3_precompute_geom_input(*model, none, feats.data(), H, H);
        write_bin(pre + "geom1.bin", g1.data(), g1.size() * sizeof(float));

        sam3_pcs_params boxed;                    // 1 pos + 1 neg (n_geo=3)
        boxed.pos_exemplars.push_back({0.25f, 0.12f, 0.75f, 0.9f});
        boxed.neg_exemplars.push_back({0.0f, 0.0f, 0.2f, 0.2f});
        auto g3 = sam3_precompute_geom_input(*model, boxed, feats.data(), H, H);
        write_bin(pre + "geom3.bin", g3.data(), g3.size() * sizeof(float));

        printf("captured %s -> %simg_feats/fpn0/fpn1/geom{1,3}\n", argv[i], pre.c_str());
    }
    printf("done: calibration inputs in %s\n", out.c_str());
    return 0;
}
