// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*****************************************************************************
** Utility — image I/O
*****************************************************************************/

sam3_image sam3_load_image(const std::string& path) {
    sam3_image img;
    int w, h, c;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, path.c_str());
        return img;
    }
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(data, data + w * h * 3);
    stbi_image_free(data);
    return img;
}

bool sam3_save_mask(const sam3_mask& mask, const std::string& path) {
    if (mask.data.empty()) return false;
    return stbi_write_png(path.c_str(), mask.width, mask.height, 1,
                          mask.data.data(), mask.width) != 0;
}

