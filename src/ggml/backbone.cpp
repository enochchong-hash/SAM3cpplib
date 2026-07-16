// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** RoPE — 2D axial rotary positional embeddings
*****************************************************************************/

// Precompute RoPE frequencies as [N, head_dim/2, 2] (cos, sin pairs).
// This matches compute_axial_cis() from vitdet.py stored as real (cos, sin)
// instead of complex numbers.
// The conversion script already stores freqs_cis per block, so this function
// is only needed if we want to recompute them from scratch.
void sam3_compute_axial_cis(float* out,
                                   int dim, int end_x, int end_y,
                                   float theta, float scale_pos) {
    const int half_dim = dim / 4;  // 16 for dim=64

    // Compute frequency bases: 1.0 / (theta ^ (arange(0,dim,4)[:dim//4] / dim))
    std::vector<float> freqs(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        freqs[i] = 1.0f / powf(theta, (float)(i * 4) / (float)dim);
    }

    // For each spatial position, compute axial frequencies
    const int N = end_x * end_y;
    for (int idx = 0; idx < N; ++idx) {
        float t_x = (float)(idx % end_x) * scale_pos;
        int row = idx / end_x;  // intentional integer floor division (row index)
        float t_y = (float)row * scale_pos;

        // X frequencies → first 16 complex values (stored as cos, sin)
        for (int i = 0; i < half_dim; ++i) {
            float angle_x = t_x * freqs[i];
            out[idx * dim + i * 2 + 0] = cosf(angle_x);
            out[idx * dim + i * 2 + 1] = sinf(angle_x);
        }
        // Y frequencies → next 16 complex values
        for (int i = 0; i < half_dim; ++i) {
            float angle_y = t_y * freqs[i];
            out[idx * dim + half_dim * 2 + i * 2 + 0] = cosf(angle_y);
            out[idx * dim + half_dim * 2 + i * 2 + 1] = sinf(angle_y);
        }
    }
}

/*****************************************************************************
** Sinusoidal 2D positional encoding (for FPN neck outputs)
*****************************************************************************/

// Generates sinusoidal PE matching PositionEmbeddingSine from Python.
// num_pos_feats = d_model / 2 = 128, temperature = 10000, normalize = true, scale = 2pi.
// Returns data in ggml column-major layout for a tensor with ne = {d_model, W, H, 1},
// i.e. element (c, w, h) at flat index c + w*d_model + h*d_model*W.
// First half channels (0..half-1) encode y, second half (half..d_model-1) encode x.
std::vector<float> sam3_sinusoidal_pe_2d(int H, int W, int d_model) {
    const int half = d_model / 2;  // 128
    const float scale = 2.0f * (float)M_PI;
    const float temperature = 10000.0f;

    std::vector<float> pe(d_model * H * W);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // Normalized positions: (pos+1) / (max_pos+1) * scale
            float pos_y = ((float)(y + 1) / (float)(H)) * scale;
            float pos_x = ((float)(x + 1) / (float)(W)) * scale;

            for (int i = 0; i < half; ++i) {
                int paired = i & ~1;  // 0,0,2,2,4,4,… (pairs sin/cos channels, matches Python // 2)
                float dim_t = powf(temperature, (float)paired / (float)half);

                float val_x, val_y;
                if (i % 2 == 0) {
                    val_x = sinf(pos_x / dim_t);
                    val_y = sinf(pos_y / dim_t);
                } else {
                    val_x = cosf(pos_x / dim_t);
                    val_y = cosf(pos_y / dim_t);
                }

                // ggml layout: ne = {d_model, W, H, 1}
                // element (c, x, y, 0) at flat index: c + x*d_model + y*d_model*W
                // First half channels are y, second half are x.
                pe[(i) + x * d_model + y * d_model * W] = val_y;
                pe[(i + half) + x * d_model + y * d_model * W] = val_x;
            }
        }
    }

    return pe;
}

// 1-D sinusoidal PE matching Python get_1d_sine_pe(pos_inds, dim, temperature).
// Output: [dim] floats — first dim/2 are sin, second dim/2 are cos.
/*****************************************************************************
** ViT forward pass — graph building
*****************************************************************************/

// All ViT graph functions use the sam.cpp convention:
//   ne[0] = embed_dim (E=1024), ne[1] = spatial W, ne[2] = spatial H, ne[3] = batch

// Apply RoPE to Q and K tensors using complex multiplication.
// x shape: [head_dim, N, num_heads*B] in ggml layout
// freqs_cis shape: [2, 32, N] in ggml layout — stored as (cos,sin) interleaved pairs
//
// Python's apply_rotary_enc does:
//   xq_ = view_as_complex(xq.reshape(..., -1, 2))  # pairs consecutive dims
//   xq_out = view_as_real(xq_ * freqs_cis).flatten(3)
//
// In real arithmetic: for each pair (x[2i], x[2i+1]) and freq (cos, sin):
//   out[2i]   = x[2i]*cos - x[2i+1]*sin
//   out[2i+1] = x[2i]*sin + x[2i+1]*cos
struct ggml_tensor* sam3_apply_rope(struct ggml_context* ctx,
                                           struct ggml_tensor* x,
                                           struct ggml_tensor* freqs_cis) {
    // freqs_cis: [2, 32, N] — dim0=2 (cos,sin), dim1=32 (half_head=head_dim/2), dim2=N
    // x: [head_dim, N, num_heads*B] — dim0=64, dim1=N, dim2=batch*heads

    const int64_t head_dim = x->ne[0];  // 64
    const int64_t N = x->ne[1];         // number of tokens
    const int64_t nheads_B = x->ne[2];  // num_heads * batch
    const int64_t half = head_dim / 2;  // 32

    // Reshape x to [2, half, N, nheads_B] to expose (real, imag) pairs
    auto* x_pairs = ggml_reshape_4d(ctx, x, 2, half, N, nheads_B);

    // freqs_cis: [2, 32, N] → [2, half, N, 1] for broadcast
    auto* fc = ggml_reshape_4d(ctx, freqs_cis, 2, half, N, 1);

    // Extract cos (offset 0) and sin (offset 1) from dim0.
    // fc is [2, half, N, 1] — to slice dim0 we keep strides of dims 1,2,3
    // as nb1,nb2,nb3 of the view, so the view walks over (half, N, 1) correctly.
    auto* cos_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                               fc->nb[1], fc->nb[2], fc->nb[3], 0);
    auto* sin_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                               fc->nb[1], fc->nb[2], fc->nb[3], fc->nb[0]);

    // Extract x_re (offset 0) and x_im (offset 1) from dim0.
    // x_pairs is [2, half, N, nheads_B] — same slicing logic.
    auto* x_re = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B,
                              x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], 0);
    auto* x_im = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B,
                              x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], x_pairs->nb[0]);

    // Complex multiply: (x_re + j*x_im) * (cos + j*sin)
    auto* out_re = ggml_sub(ctx, ggml_mul(ctx, x_re, cos_f), ggml_mul(ctx, x_im, sin_f));
    auto* out_im = ggml_add(ctx, ggml_mul(ctx, x_re, sin_f), ggml_mul(ctx, x_im, cos_f));

    // Interleave back: [2, half, N, nheads_B]
    auto* out = ggml_concat(ctx, out_re, out_im, 0);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, out), head_dim, N, nheads_B);
}

// Single ViT block forward: pre-norm → attn (window or global, with RoPE) → residual → pre-norm → MLP → residual
// x: [E, W, H, B] in ggml layout (following sam.cpp convention)
struct ggml_tensor* sam3_vit_block_forward(struct ggml_context* ctx,
                                                  struct ggml_tensor* x,
                                                  const sam3_vit_block& blk,
                                                  const sam3_hparams& hp,
                                                  int block_idx) {
    const int E = hp.vit_embed_dim;     // 1024
    const int NH = hp.vit_num_heads;    // 16
    const int HD = hp.vit_head_dim();   // 64
    const int WS = hp.vit_window_size;  // 24
    const bool is_global = hp.is_global_attn(block_idx);

    auto* shortcut = x;

    x = sam3_layer_norm(ctx, x, blk.norm1_w, blk.norm1_b);

    const int64_t w0 = x->ne[1];
    const int64_t h0 = x->ne[2];

    if (!is_global) {
        // Window partition: [E, W, H, B] → [E, WS, WS, B*num_windows]
        x = sam3_win_part(ctx, x, WS);
    }

    const int64_t W_cur = x->ne[1];
    const int64_t H_cur = x->ne[2];
    const int64_t B_cur = x->ne[3];

    {
        auto* cur = ggml_mul_mat(ctx, blk.qkv_w, x);
        cur = ggml_add(ctx, cur, blk.qkv_b);
        // cur: [3*E, W_cur, H_cur, B_cur]

        // [3*E, W*H, B_cur] → [E, 3, W*H, B_cur] → permute → [E, W*H, B_cur, 3]
        cur = ggml_reshape_4d(ctx, cur, E, 3, W_cur * H_cur, B_cur);
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 3, 1, 2));
        // cur: [E, W*H, B_cur, 3]  (ne[3]=3 separates Q/K/V)

        auto* Q = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                               cur->nb[1], cur->nb[2], 0);
        auto* K = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                               cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
        auto* V = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                               cur->nb[1], cur->nb[2], 2 * cur->nb[3]);

        Q = ggml_reshape_4d(ctx, Q, HD, NH, W_cur * H_cur, B_cur);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        Q = ggml_reshape_3d(ctx, Q, HD, W_cur * H_cur, NH * B_cur);

        K = ggml_reshape_4d(ctx, K, HD, NH, W_cur * H_cur, B_cur);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        K = ggml_reshape_3d(ctx, K, HD, W_cur * H_cur, NH * B_cur);

        V = ggml_reshape_4d(ctx, V, HD, NH, W_cur * H_cur, B_cur);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);  // [HD, N, NH, B_cur] non-contiguous view; flash_attn uses strides

        if (blk.freqs_cis) {
            Q = sam3_apply_rope(ctx, Q, blk.freqs_cis);
            K = sam3_apply_rope(ctx, K, blk.freqs_cis);
        }

        Q = ggml_reshape_4d(ctx, Q, HD, W_cur * H_cur, NH, B_cur);
        K = ggml_reshape_4d(ctx, K, HD, W_cur * H_cur, NH, B_cur);

        float scale = 1.0f / sqrtf((float)HD);
        auto* attn_out = sam3_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        // flash_attn_ext returns [HD, NH, N, B_cur] — HD and NH adjacent,
        // so reshaping directly to [E, W, H, B] is correct.
        x = ggml_reshape_4d(ctx, attn_out, E, W_cur, H_cur, B_cur);

        x = ggml_mul_mat(ctx, blk.proj_w, x);
        x = ggml_add(ctx, x, blk.proj_b);
    }

    if (!is_global) {
        x = sam3_win_unpart(ctx, x, w0, h0, WS);
    }

    x = ggml_add(ctx, shortcut, x);

    shortcut = x;
    x = sam3_layer_norm(ctx, x, blk.norm2_w, blk.norm2_b);

    x = ggml_mul_mat(ctx, blk.mlp_fc1_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc1_b);
    x = ggml_gelu_erf(ctx, x);
    x = ggml_mul_mat(ctx, blk.mlp_fc2_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc2_b);

    x = ggml_add(ctx, shortcut, x);

    return x;
}

// Build the full ViT graph.
// Input: [img_size, img_size, 3, 1] (ggml convention: [W, H, C, B])
// Output: [E, W, H, 1] where E=1024, W=H=72
struct ggml_tensor* sam3_build_vit_prefix_graph(struct ggml_context* ctx,
                                                       struct ggml_tensor* input,
                                                       const sam3_model& model) {
    const auto& hp = model.hparams;
    const int E = hp.vit_embed_dim;  // 1024
    const int H = hp.n_img_embd();   // 72
    const int W = hp.n_img_embd();   // 72

    // Patch embedding: ggml conv outputs [W, H, E, 1], permute to [E, W, H, B]
    auto* x = ggml_conv_2d_sk_p0(ctx, model.vit.patch_embed_w, input);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));

    // pos_embed [E, 24, 24] is Hiera pretrained resolution — tile 3x3 to [E, 72, 72]
    auto* pos_2d = model.vit.pos_embed;
    auto* pos_target = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, E, W, H, 1);
    auto* pos_tiled = ggml_repeat(ctx, pos_2d, pos_target);

    x = ggml_add(ctx, x, pos_tiled);

    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul_inplace(ctx, x, model.vit.ln_pre_w);
    x = ggml_add_inplace(ctx, x, model.vit.ln_pre_b);

    return x;
}

// Build the full ViT graph.
// Input: [img_size, img_size, 3, 1] (ggml convention: [W, H, C, B])
// Output: [E, W, H, 1] where E=1024, W=H=72
struct ggml_tensor* sam3_build_vit_graph(struct ggml_context* ctx,
                                                struct ggml_tensor* input,
                                                const sam3_model& model) {
    const auto& hp = model.hparams;

    struct ggml_tensor * x = sam3_build_vit_prefix_graph(ctx, input, model);

    // ── 32 transformer blocks ─────────────────────────────────────────────
    for (int i = 0; i < hp.vit_depth; ++i) {
        x = sam3_vit_block_forward(ctx, x, model.vit.blocks[i], hp, i);
    }

    // Output: [E, W, H, 1] = [1024, 72, 72, 1]
    return x;
}

/*****************************************************************************
** Neck (SimpleFPN) — graph building
*****************************************************************************/

// Build the SimpleFPN neck graph for one path (detector or tracker).
// Input: ViT output [E, W, H, B] with E=1024, W=H=72
// But the conv ops expect [W, H, C, B], so we must permute before convolutions.
// Output: 4 feature maps at different scales in [C, W, H, B] layout.
//   out[0]: [256, 288, 288, B]  (4× upsample)
//   out[1]: [256, 144, 144, B]  (2× upsample)
//   out[2]: [256,  72,  72, B]  (1×)
//   out[3]: [256,  36,  36, B]  (0.5× downsample)
void sam3_build_neck_graph(struct ggml_context* ctx,
                                  struct ggml_tensor* vit_out,
                                  const sam3_neck& neck,
                                  struct ggml_tensor* out[4]) {
    // Permute from [E, W, H, B] to [W, H, E, B] for conv operations
    auto* x = ggml_cont(ctx, ggml_permute(ctx, vit_out, 2, 0, 1, 3));

    // Helper: add bias to conv output.
    // Conv output is [W, H, C, B]. Bias is [C] (1D).
    // Reshape bias to [1, 1, C, 1] so ggml_repeat can broadcast.
    auto add_bias = [&](struct ggml_tensor* conv_out, struct ggml_tensor* bias) -> struct ggml_tensor* {
        auto* b3d = ggml_reshape_3d(ctx, bias, 1, 1, bias->ne[0]);
        return ggml_add(ctx, conv_out, ggml_repeat(ctx, b3d, conv_out));
    };

    // Scale 0 (4× upsample)
    {
        auto* s0 = sam3_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, neck.scales[0].deconv1_w), x, 2);
        s0 = add_bias(s0, neck.scales[0].deconv1_b);
        s0 = ggml_gelu_erf(ctx, s0);
        s0 = sam3_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, neck.scales[0].deconv2_w), s0, 2);
        s0 = add_bias(s0, neck.scales[0].deconv2_b);
        s0 = ggml_conv_2d_sk_p0(ctx, neck.scales[0].conv1x1_w, s0);
        s0 = add_bias(s0, neck.scales[0].conv1x1_b);
        s0 = ggml_conv_2d_s1_ph(ctx, neck.scales[0].conv3x3_w, s0);
        s0 = add_bias(s0, neck.scales[0].conv3x3_b);
        out[0] = ggml_cont(ctx, ggml_permute(ctx, s0, 1, 2, 0, 3));
    }

    // Scale 1 (2× upsample)
    {
        auto* s1 = sam3_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, neck.scales[1].deconv1_w), x, 2);
        s1 = add_bias(s1, neck.scales[1].deconv1_b);
        s1 = ggml_conv_2d_sk_p0(ctx, neck.scales[1].conv1x1_w, s1);
        s1 = add_bias(s1, neck.scales[1].conv1x1_b);
        s1 = ggml_conv_2d_s1_ph(ctx, neck.scales[1].conv3x3_w, s1);
        s1 = add_bias(s1, neck.scales[1].conv3x3_b);
        out[1] = ggml_cont(ctx, ggml_permute(ctx, s1, 1, 2, 0, 3));
    }

    // Scale 2 (1×)
    {
        auto* s2 = ggml_conv_2d_sk_p0(ctx, neck.scales[2].conv1x1_w, x);
        s2 = add_bias(s2, neck.scales[2].conv1x1_b);
        s2 = ggml_conv_2d_s1_ph(ctx, neck.scales[2].conv3x3_w, s2);
        s2 = add_bias(s2, neck.scales[2].conv3x3_b);
        out[2] = ggml_cont(ctx, ggml_permute(ctx, s2, 1, 2, 0, 3));
    }

    // Scale 3 (0.5× downsample)
    {
        auto* s3 = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        s3 = ggml_conv_2d_sk_p0(ctx, neck.scales[3].conv1x1_w, s3);
        s3 = add_bias(s3, neck.scales[3].conv1x1_b);
        s3 = ggml_conv_2d_s1_ph(ctx, neck.scales[3].conv3x3_w, s3);
        s3 = add_bias(s3, neck.scales[3].conv3x3_b);
        out[3] = ggml_cont(ctx, ggml_permute(ctx, s3, 1, 2, 0, 3));
    }
}

