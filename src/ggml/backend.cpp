// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** Internal Helper Implementations
*****************************************************************************/

/*
** ── GPU backend support (gemma4 local extension — docs/sam3/PLAN.md)
**
** Upstream is Metal/CPU-only. We run the same single-backend design on CUDA:
** one gallocr per stage on the GPU backend, exactly like Metal. The only ops
** the CUDA backend lacks are lowered at graph-build time instead:
**   - flash_attn_ext head dims 16/32/56  -> naive attention (sam3_flash_attn_ext)
**   - win_part / win_unpart              -> pad+reshape+permute (sam3_win_part/_unpart)
** g_gpu_backend flags that a GPU (non-Metal) model is active so the wrappers
** know to lower. One GPU model per process; CPU models are unaffected.
*/
ggml_backend_t g_gpu_backend = nullptr;  // registered at GPU model load

bool sam3_graph_compute(ggml_backend_t backend, struct ggml_cgraph* graph, int n_threads) {
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    // DIAGNOSTIC ONLY: SAM3_STAGE_TIMING=1 prints every sub-graph compute call
    // site-by-site (node count + first node's name as a label, since callers
    // don't pass one) -- lets every PCS/PVS sub-graph (and the video/memory
    // paths) be timed without touching each of their call sites individually.
    static const bool stage_timing = getenv("SAM3_STAGE_TIMING") != nullptr;
    std::chrono::high_resolution_clock::time_point t0;
    if (stage_timing) t0 = std::chrono::high_resolution_clock::now();
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed: %s\n",
                __func__, ggml_status_to_string(status));
        return false;
    }
    if (stage_timing) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        int n = ggml_graph_n_nodes(graph);
        const char* last_name = n > 0 ? ggml_get_name(ggml_graph_node(graph, n - 1)) : "?";
        fprintf(stderr, "[stage_timing] %d nodes, last=%s: %.2f ms\n", n, last_name, ms);
    }
    return true;
}

// Drop-in for ggml_win_part: on the GPU path, lower to pad+reshape+permute
// (all CUDA-supported); otherwise use the native op (CPU/Metal kernels).
// Semantics mirror GGML_OP_WIN_PART: [C,W,H,1] -> [C,w,w,npx*npy], zero-padded
// at the right/bottom, window index i3 = py*npx + px.
struct ggml_tensor* sam3_win_part(struct ggml_context* ctx,
                                         struct ggml_tensor* a, int w) {
    if (!g_gpu_backend) {
        return ggml_win_part(ctx, a, w);
    }
    GGML_ASSERT(a->ne[3] == 1 && a->type == GGML_TYPE_F32);
    const int64_t C = a->ne[0], W = a->ne[1], H = a->ne[2];
    const int px = (int)((w - W % w) % w);
    const int py = (int)((w - H % w) % w);
    const int64_t Wp = W + px, Hp = H + py;
    const int64_t npx = Wp / w, npy = Hp / w;

    struct ggml_tensor* x = a;
    if (px || py) {
        x = ggml_pad(ctx, x, 0, px, py, 0);              // [C, Wp, Hp, 1]
    }
    x = ggml_reshape_4d(ctx, x, C * w, npx, w, npy);     // split W=(wx,px), H=(wy,py)
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));// -> [C*w, w, npx, npy]
    return ggml_reshape_4d(ctx, x, C, w, w, npx * npy);
}

// Drop-in for ggml_win_unpart (inverse of the above, crops the padding).
struct ggml_tensor* sam3_win_unpart(struct ggml_context* ctx,
                                           struct ggml_tensor* a,
                                           int w0, int h0, int w) {
    if (!g_gpu_backend) {
        return ggml_win_unpart(ctx, a, w0, h0, w);
    }
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    const int64_t C = a->ne[0];
    const int64_t npx = (w0 + (w - w0 % w) % w) / w;
    const int64_t npy = (h0 + (w - h0 % w) % w) / w;
    const int64_t Wp = npx * w, Hp = npy * w;

    struct ggml_tensor* x = ggml_reshape_4d(ctx, a, C * w, w, npx, npy);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));   // [C*w, npx, w, npy]
    x = ggml_reshape_4d(ctx, x, C, Wp, Hp, 1);
    if (Wp != w0 || Hp != h0) {
        x = ggml_cont(ctx, ggml_view_3d(ctx, x, C, w0, h0, x->nb[1], x->nb[2], 0));
    }
    return ggml_reshape_3d(ctx, x, C, w0, h0);
}

// Drop-in for ggml_conv_transpose_2d_p0, stride==kernel_size==2 case only (the
// only case this model uses — SimpleFPN neck deconvs, mask-decoder upscale).
// ggml-cuda's native kernel (conv2d-transpose.cu) is one thread per output
// element, each independently re-reading all C_in input values from global
// memory — for a C_in=1024 deconv with C_out=512, that's up to 512x redundant
// re-reads of the same input element per spatial location (measured: this is
// ~1750ms of a ~2600ms encode, docs/sam3/PLAN.md Phase 3 step 1). Since
// stride==kernel, this reduces exactly to 4 ordinary GEMMs (one per 2x2
// "phase") plus a pixel-shuffle interleave — all CUDA-native mul_mat/
// permute/concat/reshape ops, no redundant reads.
// weight: [KW=2,KH=2,C_out,C_in]; input: [W,H,C_in,B]; returns [2W,2H,C_out,B]
// — same contract as the op it replaces. Asserts on anything this model
// doesn't actually exercise (B>1, kernel!=2) rather than silently computing
// the wrong thing.
struct ggml_tensor* sam3_conv_transpose_2d_p0(struct ggml_context* ctx,
                                                     struct ggml_tensor* weight,
                                                     struct ggml_tensor* input,
                                                     int stride) {
    if (!g_gpu_backend) {
        return ggml_conv_transpose_2d_p0(ctx, weight, input, stride);
    }
    const int64_t C_out = weight->ne[2], C_in = weight->ne[3];
    const int64_t W = input->ne[0], H = input->ne[1], B = input->ne[3];
    GGML_ASSERT(weight->ne[0] == 2 && weight->ne[1] == 2 && stride == 2);
    GGML_ASSERT(input->ne[2] == C_in);
    GGML_ASSERT(B == 1);  // batching not exercised anywhere in this model

    // [W,H,C_in,B] -> [C_in,W,H,B] -> [C_in, W*H*B] for the GEMM.
    auto* inp = ggml_cont(ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
    inp = ggml_reshape_2d(ctx, inp, C_in, W * H * B);

    struct ggml_tensor* phase[4];  // index = py*2+px, each ends as [W,H,C_out,B]
    for (int py = 0; py < 2; ++py) {
        for (int px = 0; px < 2; ++px) {
            // weight[px,py,:,:] as a contiguous [C_in,C_out] matrix — swap the
            // co/ci axis order via the view's strides so no extra transpose
            // is needed before the mul_mat.
            auto* wv = ggml_view_4d(ctx, weight, 1, 1, C_in, C_out,
                                     weight->nb[1], weight->nb[3], weight->nb[2],
                                     (size_t)px * weight->nb[0] + (size_t)py * weight->nb[1]);
            auto* w2 = ggml_reshape_2d(ctx, ggml_cont(ctx, wv), C_in, C_out);
            auto* p = ggml_mul_mat(ctx, w2, inp);                          // [C_out, W*H*B]
            p = ggml_reshape_4d(ctx, p, C_out, W, H, B);
            phase[py * 2 + px] = ggml_cont(ctx, ggml_permute(ctx, p, 2, 0, 1, 3)); // [W,H,C_out,B]
        }
    }

    // Interleave px into W per py-row (stacking the px pair into the unused
    // batch slot — safe since B==1 is asserted above — then permuting it to
    // be adjacent-and-faster than W so the merge-reshape lands as 2x+px).
    struct ggml_tensor* row[2];
    for (int py = 0; py < 2; ++py) {
        auto* stacked = ggml_concat(ctx, phase[py * 2 + 0], phase[py * 2 + 1], 3); // [W,H,C_out,2]
        stacked = ggml_cont(ctx, ggml_permute(ctx, stacked, 1, 2, 3, 0));          // [2,W,H,C_out]
        row[py] = ggml_reshape_4d(ctx, stacked, 2 * W, H, C_out, 1);              // [2W,H,C_out,1]
    }
    // Interleave py into H the same way.
    auto* stacked2 = ggml_concat(ctx, row[0], row[1], 3);                 // [2W,H,C_out,2]
    stacked2 = ggml_cont(ctx, ggml_permute(ctx, stacked2, 0, 2, 3, 1));   // [2W,2,H,C_out]
    return ggml_reshape_4d(ctx, stacked2, 2 * W, 2 * H, C_out, 1);       // [2W,2H,C_out,1]
}

// Head dims with CUDA flash-attention kernels (ggml-cuda/fattn.cu dispatch).
static bool sam3_cuda_fa_head_dim(int64_t hd) {
    switch (hd) {
        case 40: case 64: case 72: case 80:
        case 96: case 112: case 128: case 256:
            return true;
        default:
            return false;
    }
}

// Drop-in for ggml_flash_attn_ext. On the GPU-sched path, head dims without
// CUDA FA kernels (16/32/56 in this model family) are lowered to naive
// attention built from ops that all have CUDA kernels — otherwise those FA
// nodes fall back to the CPU backend mid-pipeline (slow, and the fork's CPU
// tiled-FA kernel misbehaves on sched-copied inputs: garbage/segfault).
// Output layout and mask semantics mirror ggml_flash_attn_ext exactly:
// result [DV, H, n_q, B] contiguous; mask F16 [n_kv, >=n_q, mh, mb] with
// broadcast over dims 2/3 (same contract as ggml_soft_max_ext).
struct ggml_tensor* sam3_flash_attn_ext(struct ggml_context* ctx,
                                               struct ggml_tensor* q,
                                               struct ggml_tensor* k,
                                               struct ggml_tensor* v,
                                               struct ggml_tensor* mask,
                                               float scale,
                                               float max_bias,
                                               float logit_softcap) {
    // SAM3_ATTN=fa|naive forces one implementation on every call site
    // (debug/parity tool); default: FA except GPU-unsupported head dims.
    static const char* attn_env = getenv("SAM3_ATTN");
    bool use_fa = true;
    if (attn_env && strcmp(attn_env, "naive") == 0) {
        use_fa = false;
    } else if (attn_env && strcmp(attn_env, "fa") == 0) {
        use_fa = true;
    } else if (g_gpu_backend) {
        use_fa = sam3_cuda_fa_head_dim(q->ne[0]) && sam3_cuda_fa_head_dim(v->ne[0]);
    }
    if (use_fa) {
        return ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap);
    }
    GGML_ASSERT(max_bias == 0.0f && logit_softcap == 0.0f);

    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);              // [n_kv, n_q, H, B]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);            // fused scale+mask+softmax
    struct ggml_tensor* vt  = ggml_cont(ctx, ggml_transpose(ctx, v));  // [n_kv, DV, H, B]
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);           // [DV, n_q, H, B]
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));      // [DV, H, n_q, B] like FA
    return kqv;
}

// gemma4 debug: SAM3_DUMP_STAGES=<dir> writes each PCS stage handoff (already
// host-side vectors) as raw f32 <dir>/<name>.bin for CPU-vs-GPU parity diffs.
void sam3_debug_dump_vec(const char* name, const float* p, size_t n) {
    const char* dir = getenv("SAM3_DUMP_STAGES");
    if (!dir) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, sizeof(float), n, f);
    fclose(f);
}

void sam3_name_tensorf(struct ggml_tensor* t, const char* fmt, int index) {
    if (!t) {
        return;
    }
    char name[64];
    snprintf(name, sizeof(name), fmt, index);
    ggml_set_name(t, name);
}

struct ggml_tensor* sam3_layer_norm(struct ggml_context* ctx,
                                           struct ggml_tensor* x,
                                           struct ggml_tensor* w,
                                           struct ggml_tensor* b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul_inplace(ctx, x, w);
    if (b) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

struct ggml_tensor* sam3_layer_norm_2d(struct ggml_context* ctx,
                                              struct ggml_tensor* x,
                                              struct ggml_tensor* w,
                                              struct ggml_tensor* b) {
    // x is [C, H, W, B] in ggml layout — norm over C dimension (dim 0)
    x = ggml_norm(ctx, x, 1e-6f);
    // w, b are [C, 1, 1] — broadcast multiply/add
    x = ggml_mul_inplace(ctx, x, w);
    if (b) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// Read tensor data from backend into a float buffer, handling F32, F16, and
// quantized types.  n = number of float elements to produce.
void sam3_read_f32(struct ggml_tensor* t, float* dst, int64_t n) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst, 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), dst, (int)n);
    } else if (ggml_is_quantized(t->type)) {
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const auto * traits = ggml_get_type_traits(t->type);
        traits->to_float(raw.data(), dst, n);
    } else {
        fprintf(stderr, "%s: unsupported tensor type %d\n", __func__, (int)t->type);
    }
}

struct ggml_tensor* sam3_conv_transpose_weight(struct ggml_context* ctx,
                                                      struct ggml_tensor* w) {
    return w->type == GGML_TYPE_F16 ? w : ggml_cast(ctx, w, GGML_TYPE_F16);
}

