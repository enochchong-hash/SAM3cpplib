// Minimal consumer: exercises the public header (types, accessors, and API
// symbol linkage) without needing model weights. A successful build + run
// proves the add_subdirectory() consumption story.
#include "sam3cpp/sam3.h"

#include <cstdio>

int main() {
    // Link check: touch the model-lifecycle API with an invalid path.
    sam3_params params;
    params.model_path = "/nonexistent.ggml";
    auto model = sam3_load_model(params);
    if (model) {
        fprintf(stderr, "unexpectedly loaded a nonexistent model\n");
        return 1;
    }

    // Header-only functionality: convenience accessors on a synthetic mask.
    sam3_mask m;
    m.width = 4;
    m.height = 4;
    m.data.assign(16, 0);
    m.data[5] = 255;
    if (sam3_mask_area(m) != 1 || !sam3_mask_at(m, 1, 1)) {
        fprintf(stderr, "mask accessor mismatch\n");
        return 1;
    }

    printf("consume_test: OK (sam3cpplib linked and callable, version %s)\n", SAM3_VERSION);
    return 0;
}
