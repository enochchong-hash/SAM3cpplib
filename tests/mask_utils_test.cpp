// Unit tests for the sam3_mask_* convenience accessors (pure CPU geometry;
// no model needed). Returns 0 on success, non-zero with a message on failure.
#include "sam3cpp/sam3.h"

#include <cmath>
#include <cstdio>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static bool feq(float a, float b) { return std::fabs(a - b) < 1e-4f; }

int main() {
    // 10x10 mask with a 3x2 foreground block at x=[4,7), y=[5,7)
    sam3_mask m;
    m.width = 10;
    m.height = 10;
    m.data.assign(100, 0);
    for (int y = 5; y < 7; ++y)
        for (int x = 4; x < 7; ++x)
            m.data[y * 10 + x] = 255;

    CHECK(sam3_mask_area(m) == 6);

    const auto c = sam3_mask_centroid(m);
    CHECK(feq(c.x, 5.0f) && feq(c.y, 5.5f));

    const auto bb = sam3_mask_bbox(m);
    CHECK(feq(bb.x0, 4) && feq(bb.y0, 5) && feq(bb.x1, 7) && feq(bb.y1, 7));

    CHECK(sam3_mask_at(m, 4, 5));
    CHECK(sam3_mask_at(m, 6, 6));
    CHECK(!sam3_mask_at(m, 7, 5));   // x1 is exclusive
    CHECK(!sam3_mask_at(m, -1, 5));  // out of range
    CHECK(!sam3_mask_at(m, 4, 10));

    const auto coords = sam3_mask_coords(m);
    CHECK(coords.size() == 6);
    CHECK(feq(coords[0].x, 4) && feq(coords[0].y, 5));  // row-major order
    CHECK(feq(coords[5].x, 6) && feq(coords[5].y, 6));

    // Foreground threshold: byte > 127
    m.data[5 * 10 + 4] = 127;
    CHECK(sam3_mask_area(m) == 5);
    m.data[5 * 10 + 4] = 128;
    CHECK(sam3_mask_area(m) == 6);

    // Empty mask
    sam3_mask e;
    CHECK(sam3_mask_area(e) == 0);
    const auto ec = sam3_mask_centroid(e);
    CHECK(feq(ec.x, -1) && feq(ec.y, -1));
    const auto eb = sam3_mask_bbox(e);
    CHECK(feq(eb.x0, 0) && feq(eb.y0, 0) && feq(eb.x1, 0) && feq(eb.y1, 0));
    CHECK(!sam3_mask_at(e, 0, 0));
    CHECK(sam3_mask_coords(e).empty());

    printf("mask_utils_test: all checks passed\n");
    return 0;
}
