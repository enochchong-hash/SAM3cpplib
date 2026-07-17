// 01 — Model lifecycle and image encoding
//
// The three objects every sam3cpplib program touches:
//
//   sam3_model  (shared_ptr)  weights + backend; load once per process
//   sam3_state  (unique_ptr)  per-image caches (backbone features, PE caches)
//   sam3_image                raw RGB8 pixels in, no codecs required
//
// Encoding is the expensive stage (~120 ms TensorRT / ~0.9 s ggml-CUDA /
// ~35 s CPU on the RTX 5060 dev box); everything that follows (text / point /
// box prompting, examples 02-07) reuses the encoded features.
//
// Usage: sam3cpp_ex_01_load_and_encode <model.ggml> <image> [cpu]
//        "cpu" forces the ggml CPU reference backend (slow, bit-exact
//        golden-sample generator -- see docs/goldens.md).
#include "sam3cpp/sam3.h"

#include <chrono>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.ggml> <image> [cpu]\n", argv[0]);
        return 1;
    }

    // ── 1. Load the model ───────────────────────────────────────────────
    sam3_params params;
    params.model_path = argv[1];
    params.use_gpu    = !(argc > 3 && strcmp(argv[3], "cpu") == 0);
    params.n_threads  = 4;  // used by the CPU backend and CPU post-processing

    auto model = sam3_load_model(params);   // nullptr on failure (bad path,
    if (!model) return 1;                   // wrong magic, SAM2 file, ...)

    printf("model type   : %d (SAM3_MODEL_SAM3=0, .._VISUAL=1)\n",
           (int)sam3_get_model_type(*model));
    printf("visual-only  : %s (visual-only models have no text/PCS path)\n",
           sam3_is_visual_only(*model) ? "yes" : "no");

    // ── 2. Create the inference state ───────────────────────────────────
    // One state per stream of work; sequential states may share the model.
    auto state = sam3_create_state(*model, params);
    if (!state) return 1;

    // ── 3. Load an image ────────────────────────────────────────────────
    // sam3_load_image is an optional stb-backed helper (SAM3CPP_IMAGE_IO=ON).
    // A host app can instead fill sam3_image{width, height, 3, rgb_bytes}
    // from its own pipeline -- the library core never decodes files.
    sam3_image image = sam3_load_image(argv[2]);
    if (image.data.empty()) { fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
    printf("image        : %dx%d\n", image.width, image.height);

    // ── 4. Encode ───────────────────────────────────────────────────────
    // Internally: resize to the model's native 1008x1008, run the ViT
    // backbone + SimpleFPN neck, cache the multi-scale features in `state`.
    // Call this once per image, then prompt as often as you like (ex. 07).
    for (int i = 0; i < 2; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!sam3_encode_image(*state, *model, image)) return 1;
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        printf("encode #%d    : %.1f ms%s\n", i, ms,
               i == 0 ? " (first call includes one-time warmup/engine load)" : "");
    }

    printf("OK -- features are cached in the state; see 02..07 for prompting\n");
    return 0;  // state/model free themselves (RAII)
}
