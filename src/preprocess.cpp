// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** Image preprocessing
*****************************************************************************/

// Bilinear resize of a [H, W, 3] uint8 image to [dst_h, dst_w, 3].
static void sam3_resize_bilinear(const uint8_t* src, int src_w, int src_h,
                                 uint8_t* dst, int dst_w, int dst_h) {
    // Bilinear resize matching torch.nn.functional.interpolate(bilinear, align_corners=False).
    // ALL arithmetic is in double to get an exact result for uint8 inputs (0-255),
    // ensuring the bilinear result is independent of FMA/SIMD/compiler behavior.
    // The exact double result is then rounded to uint8, matching torch's round().
    const double sx = (double)src_w / dst_w;
    const double sy = (double)src_h / dst_h;
    for (int y = 0; y < dst_h; ++y) {
        double fy = (y + 0.5) * sy - 0.5;
        if (fy < 0.0) fy = 0.0;
        const int y0 = (int)fy;
        const int y1 = (y0 < src_h - 1) ? y0 + 1 : y0;
        const double wy = fy - y0;
        const double wy0 = 1.0 - wy;
        for (int x = 0; x < dst_w; ++x) {
            double fx = (x + 0.5) * sx - 0.5;
            if (fx < 0.0) fx = 0.0;
            const int x0 = (int)fx;
            const int x1 = (x0 < src_w - 1) ? x0 + 1 : x0;
            const double wx = fx - x0;
            const double wx0 = 1.0 - wx;
            for (int c = 0; c < 3; ++c) {
                const double p00 = src[(y0 * src_w + x0) * 3 + c];
                const double p01 = src[(y0 * src_w + x1) * 3 + c];
                const double p10 = src[(y1 * src_w + x0) * 3 + c];
                const double p11 = src[(y1 * src_w + x1) * 3 + c];
                double v = wy0 * (wx0 * p00 + wx * p01) +
                           wy * (wx0 * p10 + wx * p11);
                // Round to nearest integer, matching torch's round().to(uint8).
                // For exact half-values (v = N.5), torch uses banker's rounding
                // (round-half-to-even), but bilinear with double precision on
                // uint8 inputs virtually never hits exact halves.
                int iv = (int)(v + 0.5);
                if (iv < 0) iv = 0;
                if (iv > 255) iv = 255;
                dst[(y * dst_w + x) * 3 + c] = (uint8_t)iv;
            }
        }
    }
}

// Preprocess an image: resize to img_size × img_size, convert to float, normalize.
// Returns a float tensor in [C, H, W] layout (channel-first), range normalized with
// mean=0.5, std=0.5 → pixel values in [-1, 1].
std::vector<float> sam3_preprocess_image(const sam3_image& image, int img_size) {
    const int C = 3;
    std::vector<float> result(C * img_size * img_size);

    // Resize to img_size × img_size via uint8 bilinear (matching torch pipeline)
    std::vector<uint8_t> resized;
    const uint8_t* pixels = image.data.data();
    int w = image.width, h = image.height;

    if (w != img_size || h != img_size) {
        resized.resize(img_size * img_size * 3);
        sam3_resize_bilinear(pixels, w, h, resized.data(), img_size, img_size);
        pixels = resized.data();
        w = img_size;
        h = img_size;
    }

    // Convert to float [C, H, W] with normalization: (pixel / 255.0 - 0.5) / 0.5
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < img_size; ++y) {
            for (int x = 0; x < img_size; ++x) {
                float v = pixels[(y * img_size + x) * 3 + c] / 255.0f;
                result[c * img_size * img_size + y * img_size + x] = (v - 0.5f) / 0.5f;
            }
        }
    }

    return result;
}

// SAM2 preprocessing: resize + ImageNet normalization.
// Returns [C, H, W] float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
