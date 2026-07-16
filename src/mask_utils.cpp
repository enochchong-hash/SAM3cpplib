// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** Post-processing: NMS, bilinear interpolation, mask binarization
*****************************************************************************/

// Compute IoU between two boxes [x0, y0, x1, y1].
static float sam3_box_iou(const sam3_box& a, const sam3_box& b) {
    float x0 = std::max(a.x0, b.x0);
    float y0 = std::max(a.y0, b.y0);
    float x1 = std::min(a.x1, b.x1);
    float y1 = std::min(a.y1, b.y1);
    float inter = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
    float area_a = (a.x1 - a.x0) * (a.y1 - a.y0);
    float area_b = (b.x1 - b.x0) * (b.y1 - b.y0);
    float uni = area_a + area_b - inter;
    return (uni > 0.0f) ? inter / uni : 0.0f;
}

// Non-maximum suppression on detections, sorted by score descending.
// Returns indices of kept detections.
/*****************************************************************************
** Mask convenience accessors (public API -- see sam3.h)
*****************************************************************************/

size_t sam3_mask_area(const sam3_mask& mask) {
    size_t n = 0;
    for (uint8_t v : mask.data) n += (v > 127) ? 1 : 0;
    return n;
}

sam3_point sam3_mask_centroid(const sam3_mask& mask) {
    double sx = 0, sy = 0;
    size_t n = 0;
    for (int y = 0; y < mask.height; ++y) {
        const uint8_t* row = mask.data.data() + (size_t)y * mask.width;
        for (int x = 0; x < mask.width; ++x) {
            if (row[x] > 127) { sx += x; sy += y; ++n; }
        }
    }
    if (n == 0) return {-1.0f, -1.0f};
    return {(float)(sx / n), (float)(sy / n)};
}

sam3_box sam3_mask_bbox(const sam3_mask& mask) {
    int min_x = mask.width, min_y = mask.height, max_x = -1, max_y = -1;
    for (int y = 0; y < mask.height; ++y) {
        const uint8_t* row = mask.data.data() + (size_t)y * mask.width;
        for (int x = 0; x < mask.width; ++x) {
            if (row[x] > 127) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }
    if (max_x < 0) return {0.0f, 0.0f, 0.0f, 0.0f};
    // x1/y1 exclusive: width = x1-x0.
    return {(float)min_x, (float)min_y, (float)(max_x + 1), (float)(max_y + 1)};
}

bool sam3_mask_at(const sam3_mask& mask, int x, int y) {
    if (x < 0 || y < 0 || x >= mask.width || y >= mask.height) return false;
    return mask.data[(size_t)y * mask.width + x] > 127;
}

std::vector<sam3_point> sam3_mask_coords(const sam3_mask& mask) {
    std::vector<sam3_point> out;
    out.reserve(sam3_mask_area(mask));
    for (int y = 0; y < mask.height; ++y) {
        const uint8_t* row = mask.data.data() + (size_t)y * mask.width;
        for (int x = 0; x < mask.width; ++x) {
            if (row[x] > 127) out.push_back({(float)x, (float)y});
        }
    }
    return out;
}

std::vector<int> sam3_nms(const std::vector<sam3_detection>& dets, float iou_thresh) {
    // Sort indices by score descending
    std::vector<int> indices(dets.size());
    for (int i = 0; i < (int)dets.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return dets[a].score > dets[b].score;
    });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int> keep;

    for (int idx : indices) {
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (int j : indices) {
            if (suppressed[j] || j == idx) continue;
            if (sam3_box_iou(dets[idx].box, dets[j].box) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

// Bilinear interpolation of a flat mask [H_in * W_in] to [H_out * W_out].
// Uses double for coordinate math to match PyTorch F.interpolate precision.
std::vector<float> sam3_bilinear_interpolate(const float* src, int src_w, int src_h,
                                                    int dst_w, int dst_h) {
    std::vector<float> dst(dst_w * dst_h);
    const double sx = (double)src_w / dst_w;
    const double sy = (double)src_h / dst_h;

    for (int y = 0; y < dst_h; ++y) {
        double fy = (y + 0.5) * sy - 0.5;
        fy = std::max(0.0, std::min(fy, (double)(src_h - 1)));
        const int y0 = std::min((int)fy, src_h - 2);
        const int y1 = y0 + 1;
        const float wy = (float)(fy - y0);

        for (int x = 0; x < dst_w; ++x) {
            double fx = (x + 0.5) * sx - 0.5;
            fx = std::max(0.0, std::min(fx, (double)(src_w - 1)));
            const int x0 = std::min((int)fx, src_w - 2);
            const int x1 = x0 + 1;
            const float wx = (float)(fx - x0);

            float v = (1 - wy) * ((1 - wx) * src[y0 * src_w + x0] + wx * src[y0 * src_w + x1]) + wy * ((1 - wx) * src[y1 * src_w + x0] + wx * src[y1 * src_w + x1]);
            dst[y * dst_w + x] = v;
        }
    }
    return dst;
}

// Convert (cx, cy, w, h) in [0,1] to (x0, y0, x1, y1) in pixel coordinates.
sam3_box sam3_cxcywh_to_xyxy(float cx, float cy, float w, float h,
                                    int img_w, int img_h) {
    sam3_box box;
    box.x0 = (cx - w * 0.5f) * img_w;
    box.y0 = (cy - h * 0.5f) * img_h;
    box.x1 = (cx + w * 0.5f) * img_w;
    box.y1 = (cy + h * 0.5f) * img_h;
    // Clamp to image bounds
    box.x0 = std::max(0.0f, std::min(box.x0, (float)img_w));
    box.y0 = std::max(0.0f, std::min(box.y0, (float)img_h));
    box.x1 = std::max(0.0f, std::min(box.x1, (float)img_w));
    box.y1 = std::max(0.0f, std::min(box.y1, (float)img_h));
    return box;
}

