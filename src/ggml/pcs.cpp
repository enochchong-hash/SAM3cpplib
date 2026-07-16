// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** Text Encoder — graph building (Phase 4)
*****************************************************************************/

// Build a causal (lower-triangular) attention mask for the text encoder.
// Returns: [L, L] F16 tensor. mask[kv][q] = 0 if kv <= q, -inf otherwise.
// Marked as input — caller must upload data via ggml_backend_tensor_set after alloc.
static struct ggml_tensor* sam3_build_causal_mask(struct ggml_context* ctx, int L) {
    auto* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, L);
    ggml_set_name(mask, "causal_mask");
    ggml_set_input(mask);
    return mask;
}

// Fill a pre-allocated causal mask buffer (host-side, F16).
// mask_data must hold L*L ggml_fp16_t values.
void sam3_fill_causal_mask(ggml_fp16_t* mask_data, int L) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neginf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < L; ++q) {
        for (int kv = 0; kv < L; ++kv) {
            mask_data[kv + q * L] = (kv <= q) ? zero : neginf;
        }
    }
}

// Single text encoder block forward pass.
// Input x: [E, L] where E=text_width=1024, L=seq_len (typically 32).
// causal_mask: [L, L] F16 additive mask for ggml_flash_attn_ext.
// Returns: [E, L]
static struct ggml_tensor* sam3_text_block_forward(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   const sam3_text_block& blk,
                                                   const sam3_hparams& hp,
                                                   struct ggml_tensor* causal_mask,
                                                   int block_idx) {
    const int E = hp.text_width;   // 1024
    const int NH = hp.text_heads;  // 16
    const int HD = E / NH;         // 64
    const int64_t L = x->ne[1];    // sequence length

    auto* shortcut = x;
    x = sam3_layer_norm(ctx, x, blk.ln1_w, blk.ln1_b);
    sam3_name_tensorf(x, "text_block_%02d_after_ln1", block_idx);

    auto* qkv = ggml_mul_mat(ctx, blk.attn_in_proj_w, x);
    qkv = ggml_add(ctx, qkv, blk.attn_in_proj_b);
    sam3_name_tensorf(qkv, "text_block_%02d_qkv", block_idx);

    // [3*E, L] → [E, 3, L] → permute → [E, L, 3]
    qkv = ggml_reshape_3d(ctx, qkv, E, 3, L);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 2, 1, 3));
    // qkv: [E, L, 3]

    auto* Q = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 0);
    auto* K = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 1 * qkv->nb[2]);
    auto* V = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 2 * qkv->nb[2]);

    // [E, L] → [HD, NH, L] → permute → [HD, L, NH, 1]
    Q = ggml_reshape_3d(ctx, Q, HD, NH, L);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    Q = ggml_reshape_4d(ctx, Q, HD, L, NH, 1);

    K = ggml_reshape_3d(ctx, K, HD, NH, L);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    K = ggml_reshape_4d(ctx, K, HD, L, NH, 1);

    V = ggml_reshape_3d(ctx, V, HD, NH, L);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);  // non-contiguous; flash_attn uses strides

    float scale = 1.0f / sqrtf((float)HD);
    auto* attn_out = sam3_flash_attn_ext(ctx, Q, K, V, causal_mask, scale, 0.0f, 0.0f);
    x = ggml_reshape_2d(ctx, attn_out, E, L);

    x = ggml_mul_mat(ctx, blk.attn_out_proj_w, x);
    x = ggml_add(ctx, x, blk.attn_out_proj_b);

    if (blk.ls1) {
        x = ggml_mul(ctx, x, blk.ls1);
    }
    sam3_name_tensorf(x, "text_block_%02d_attn_out", block_idx);

    x = ggml_add(ctx, shortcut, x);
    sam3_name_tensorf(x, "text_block_%02d_after_attn_residual", block_idx);

    shortcut = x;
    x = sam3_layer_norm(ctx, x, blk.ln2_w, blk.ln2_b);
    sam3_name_tensorf(x, "text_block_%02d_after_ln2", block_idx);

    x = ggml_mul_mat(ctx, blk.mlp_fc1_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc1_b);
    sam3_name_tensorf(x, "text_block_%02d_mlp_fc1", block_idx);
    x = ggml_gelu_erf(ctx, x);
    sam3_name_tensorf(x, "text_block_%02d_mlp_gelu", block_idx);
    x = ggml_mul_mat(ctx, blk.mlp_fc2_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc2_b);

    if (blk.ls2) {
        x = ggml_mul(ctx, x, blk.ls2);
    }
    sam3_name_tensorf(x, "text_block_%02d_mlp_out", block_idx);

    x = ggml_add(ctx, shortcut, x);
    sam3_name_tensorf(x, "text_block_%02d_out", block_idx);

    return x;
}

// Build the full text encoder computation graph.
// token_ids: [L] int32 tensor (BPE token IDs, padded to ctx_len with 0s).
//            Must be marked as input by caller; data uploaded after alloc.
// Returns: text_features tensor [text_out_dim, L] = [256, L].
// Also creates the causal mask internally (marked as input).
struct ggml_tensor* sam3_build_text_encoder_graph(struct ggml_context* ctx,
                                                         struct ggml_tensor* token_ids,
                                                         const sam3_model& model) {
    const auto& hp = model.hparams;
    const auto& enc = model.text_enc;
    const int L = hp.text_ctx_len;  // 32

    auto* x = ggml_get_rows(ctx, enc.token_embed_w, token_ids);
    ggml_set_name(x, "text_token_embed");

    x = ggml_add(ctx, x, enc.pos_embed);
    ggml_set_name(x, "text_after_pos_embed");

    auto* causal_mask = sam3_build_causal_mask(ctx, L);

    for (int i = 0; i < hp.text_layers; ++i) {
        x = sam3_text_block_forward(ctx, x, enc.blocks[i], hp, causal_mask, i);
    }

    x = sam3_layer_norm(ctx, x, enc.ln_final_w, enc.ln_final_b);
    ggml_set_name(x, "text_final_ln");

    // Resizer: project 1024 → 256
    x = ggml_mul_mat(ctx, enc.resizer_w, x);
    x = ggml_add(ctx, x, enc.resizer_b);
    ggml_set_name(x, "text_features_2d");

    return x;
}

/*****************************************************************************
** Multi-head attention helper (used by fusion encoder, DETR decoder, seg head)
*****************************************************************************/

// Standard multi-head attention with fused in_proj.
// q_in, k_in, v_in: [D, N, B]  (if fused_qkv, only q_in is used and contains QKV stacked)
// in_proj_w: [D, 3*D] (fused Q/K/V projection)
// in_proj_b: [3*D]
// out_proj_w: [D, D], out_proj_b: [D]
// n_heads: number of attention heads
// Returns: [D, N_q, B]
//
// If separate_kv is true, q_in/k_in/v_in are already separate (no fused proj needed).
// The in_proj is applied to form Q from q_in, and K/V from the concatenated k/v source.
static struct ggml_tensor* sam3_multihead_attn_fused(
    struct ggml_context* ctx,
    struct ggml_tensor* q_in,        // [D, N_q, B]
    struct ggml_tensor* kv_in,       // [D, N_kv, B] (can be same as q_in for self-attn)
    struct ggml_tensor* in_proj_w,   // [D, 3*D] — fused QKV weights
    struct ggml_tensor* in_proj_b,   // [3*D]
    struct ggml_tensor* out_proj_w,  // [D, D]
    struct ggml_tensor* out_proj_b,  // [D]
    int n_heads,
    struct ggml_tensor* attn_mask = nullptr)  // [N_kv, N_q] or nullptr
{
    const int64_t D = q_in->ne[0];  // 256
    const int64_t N_q = q_in->ne[1];
    const int64_t B = q_in->ne[2];
    const int64_t N_kv = kv_in->ne[1];
    const int64_t HD = D / n_heads;

    auto* q_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], 0);
    auto* k_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], D * in_proj_w->nb[1]);
    auto* v_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], 2 * D * in_proj_w->nb[1]);

    auto* q_b = ggml_view_1d(ctx, in_proj_b, D, 0);
    auto* k_b = ggml_view_1d(ctx, in_proj_b, D, D * sizeof(float));
    auto* v_b = ggml_view_1d(ctx, in_proj_b, D, 2 * D * sizeof(float));

    auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
    auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, kv_in), k_b);
    auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, kv_in), v_b);

    Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [HD, N_q, NH, B]

    K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);  // [HD, N_kv, NH, B] non-contiguous; flash_attn uses strides

    float scale = 1.0f / sqrtf((float)HD);
    auto* attn_out = sam3_flash_attn_ext(ctx, Q, K, V, attn_mask, scale, 0.0f, 0.0f);

    auto* merged = ggml_reshape_3d(ctx, attn_out, D, N_q, B);
    merged = ggml_mul_mat(ctx, out_proj_w, merged);
    merged = ggml_add(ctx, merged, out_proj_b);

    return merged;
}

static struct ggml_tensor* sam3_expand_token_attn_bias(
    struct ggml_context* ctx,
    struct ggml_tensor* token_bias,  // [T, 1, B] or nullptr
    int64_t n_q,
    int n_heads,
    int64_t batch) {
    if (!token_bias) {
        return nullptr;
    }

    const int64_t n_kv = token_bias->ne[0];
    auto* bias_4d = ggml_reshape_4d(ctx, token_bias, n_kv, 1, 1, batch);
    auto* full_bias = ggml_repeat(
        ctx,
        bias_4d,
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_q, n_heads, batch));

    // ggml_flash_attn_ext expects mask storage in fp16.
    return ggml_cont(ctx, ggml_cast(ctx, full_bias, GGML_TYPE_F16));
}

/*****************************************************************************
** Geometry / exemplar encoder — graph building
*****************************************************************************/

// Sinusoidal positional encoding for box coordinates.
// Matches Python PositionEmbeddingSine.encode_boxes(cx, cy, w, h).
// Output: [258] = [pos_y(128), pos_x(128), h, w]
static void sam3_sine_encode_box(float* out, float cx, float cy, float w, float h,
                                 int num_pos_feats, int temperature) {
    const float scale = 2.0f * (float)M_PI;
    float x_embed = cx * scale;
    float y_embed = cy * scale;

    // dim_t[i] = temperature^(2*(i//2)/num_pos_feats)
    for (int i = 0; i < num_pos_feats; ++i) {
        int div_idx = 2 * (i / 2);
        float dim_t = powf((float)temperature, (float)div_idx / (float)num_pos_feats);
        float px = x_embed / dim_t;
        float py = y_embed / dim_t;
        // Interleaved sin/cos: even indices get sin, odd get cos
        if (i % 2 == 0) {
            out[i] = sinf(py);                  // pos_y first
            out[num_pos_feats + i] = sinf(px);  // pos_x second
        } else {
            out[i] = cosf(py);
            out[num_pos_feats + i] = cosf(px);
        }
    }
    out[2 * num_pos_feats] = h;      // h
    out[2 * num_pos_feats + 1] = w;  // w
}

// CPU-side ROI Align matching torchvision.ops.roi_align behavior.
// Features in ggml [C, W, H] layout. Box in XYXY format scaled to feature grid coords.
// Uses sub-sampling matching torchvision's sampling_ratio=0 (auto).
// Output: [C * roi_size * roi_size] in ggml layout.
static void sam3_roi_align_single(
    const float* feats,  // [C, W, H] ggml layout (C innermost, then W, then H)
    int C, int W_feat, int H_feat,
    float x0, float y0, float x1, float y1,
    int roi_size,
    float* out)  // [C, roi_size, roi_size]
{
    float roi_w = std::max(x1 - x0, 1e-6f);
    float roi_h = std::max(y1 - y0, 1e-6f);
    float bin_w = roi_w / (float)roi_size;
    float bin_h = roi_h / (float)roi_size;

    // Match torchvision sampling_ratio=0: use ceil(bin_size) sub-samples
    int sample_y_count = std::max(1, (int)ceilf(bin_h));
    int sample_x_count = std::max(1, (int)ceilf(bin_w));
    float inv_count = 1.0f / (float)(sample_y_count * sample_x_count);

    auto bilinear_sample = [&](float sx, float sy, int c) -> float {
        // Clamp to [-0.5, W-0.5] then adjust — matching torchvision
        if (sx < -1.0f || sx > (float)W_feat || sy < -1.0f || sy > (float)H_feat)
            return 0.0f;

        sy = std::max(0.0f, sy);
        sx = std::max(0.0f, sx);

        int y_lo = (int)sy;
        int x_lo = (int)sx;
        int y_hi = std::min(y_lo + 1, H_feat - 1);
        int x_hi = std::min(x_lo + 1, W_feat - 1);
        y_lo = std::min(y_lo, H_feat - 1);
        x_lo = std::min(x_lo, W_feat - 1);

        float ly = sy - (float)y_lo;
        float lx = sx - (float)x_lo;

        // ggml layout: feats[c + x * C + y * C * W_feat]
        float v00 = feats[c + x_lo * C + y_lo * C * W_feat];
        float v10 = feats[c + x_hi * C + y_lo * C * W_feat];
        float v01 = feats[c + x_lo * C + y_hi * C * W_feat];
        float v11 = feats[c + x_hi * C + y_hi * C * W_feat];

        return v00 * (1 - ly) * (1 - lx) + v10 * (1 - ly) * lx + v01 * ly * (1 - lx) + v11 * ly * lx;
    };

    for (int ph = 0; ph < roi_size; ++ph) {
        for (int pw = 0; pw < roi_size; ++pw) {
            for (int c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (int iy = 0; iy < sample_y_count; ++iy) {
                    float sy = y0 + bin_h * ((float)ph + ((float)iy + 0.5f) / (float)sample_y_count);
                    for (int ix = 0; ix < sample_x_count; ++ix) {
                        float sx = x0 + bin_w * ((float)pw + ((float)ix + 0.5f) / (float)sample_x_count);
                        sum += bilinear_sample(sx, sy, c);
                    }
                }
                out[c + pw * C + ph * C * roi_size] = sum * inv_count;
            }
        }
    }
}

// Build geometry encoder graph and pre-compute box embeddings on CPU.
// Returns the geometry features as a pre-computed input tensor [D, N_geo, 1]
// where N_geo = n_exemplar_boxes + 1 (CLS).
// For dummy prompts (no boxes), N_geo = 1 (just CLS token).

sam3_geom_result sam3_build_geom_enc_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    const sam3_pcs_params& params,
    struct ggml_tensor* img_feats,  // [D, N_img, 1] where N_img = H*H
    struct ggml_tensor* img_pe)     // [D, N_img, 1] sinusoidal PE
{
    const auto& ge = model.geom_enc;
    const int D = model.hparams.neck_dim;  // 256
    const int n_heads = 8;
    const int n_boxes = (int)(params.pos_exemplars.size() + params.neg_exemplars.size());
    const int N_geo = n_boxes + (int)params.exemplar_embeddings.size() + 1;  // +1 for CLS

    // Input tensor: pre-computed geometry embeddings after final_proj + norm [D, N_geo, 1]
    // The caller pre-computes: final_proj(box_embeds + CLS) → LayerNorm
    auto* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_geo, 1);
    ggml_set_name(x, "geom_post_final_proj");  // also used as upload target
    ggml_set_input(x);

    // Transformer layers (3 layers: self-attn + cross-attn + FFN, pre-norm)
    for (int i = 0; i < (int)ge.layers.size(); ++i) {
        const auto& ly = ge.layers[i];

        // 1. Self-attention (pre-norm, pos_enc_at_attn=False)
        {
            auto* shortcut = x;
            auto* xn = sam3_layer_norm(ctx, x, ly.norm1_w, ly.norm1_b);

            // Q = K = V = norm(x) — no positional encoding at self-attention
            auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0), xn),
                               ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0));
            auto* K = ggml_add(ctx, ggml_mul_mat(ctx, ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]), xn),
                               ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float)));
            auto* V = ggml_add(ctx, ggml_mul_mat(ctx, ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]), xn),
                               ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float)));

            const int64_t S = N_geo;
            const int64_t HD = D / n_heads;

            Q = ggml_reshape_4d(ctx, Q, HD, n_heads, S, 1);
            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            K = ggml_reshape_4d(ctx, K, HD, n_heads, S, 1);
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            V = ggml_reshape_4d(ctx, V, HD, n_heads, S, 1);
            V = ggml_permute(ctx, V, 0, 2, 1, 3);

            float scale = 1.0f / sqrtf((float)HD);
            auto* sa_out = sam3_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
            sa_out = ggml_reshape_3d(ctx, sa_out, D, S, 1);

            sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
            sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

            x = ggml_add(ctx, shortcut, sa_out);
        }

        // 2. Cross-attention (pre-norm, Q from x, K from img+PE, V from img)
        {
            auto* shortcut = x;
            auto* xn = sam3_layer_norm(ctx, x, ly.norm2_w, ly.norm2_b);

            // Q from normalized geometry tokens (no pos)
            // K from image features + PE
            // V from image features
            auto* q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
            auto* k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
            auto* v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);
            auto* q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
            auto* k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
            auto* v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

            auto* k_input = ggml_add(ctx, img_feats, img_pe);  // pos_enc_at_cross_attn_keys

            auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, xn), q_b);
            auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_input), k_b);
            auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, img_feats), v_b);

            const int64_t S_q = N_geo;
            const int64_t S_kv = img_feats->ne[1];
            const int64_t HD = D / n_heads;

            Q = ggml_reshape_4d(ctx, Q, HD, n_heads, S_q, 1);
            Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            K = ggml_reshape_4d(ctx, K, HD, n_heads, S_kv, 1);
            K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            V = ggml_reshape_4d(ctx, V, HD, n_heads, S_kv, 1);
            V = ggml_permute(ctx, V, 0, 2, 1, 3);

            float scale = 1.0f / sqrtf((float)HD);
            auto* ca_out = sam3_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
            ca_out = ggml_reshape_3d(ctx, ca_out, D, S_q, 1);

            ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
            ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

            x = ggml_add(ctx, shortcut, ca_out);
        }

        // 3. FFN (pre-norm, ReLU)
        {
            auto* shortcut = x;
            auto* xn = sam3_layer_norm(ctx, x, ly.norm3_w, ly.norm3_b);
            auto* ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, xn);
            ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
            ffn = ggml_relu(ctx, ffn);
            ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
            ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);
            x = ggml_add(ctx, shortcut, ffn);
        }
        sam3_name_tensorf(x, "geom_layer%d_out", i);
    }

    // Final encode norm
    x = sam3_layer_norm(ctx, x, ge.encode_norm_w, ge.encode_norm_b);
    ggml_set_name(x, "geom_output");
    ggml_set_output(x);

    sam3_geom_result result;
    result.geo_feats = x;
    result.n_tokens = N_geo;
    return result;
}

// Pre-compute geometry encoder input on CPU: box embeddings + CLS token.
// Reads model weights from GPU, computes embeddings, returns float vector.
// Layout: [D, N_geo] row-major where N_geo = n_boxes + 1 (CLS last).
std::vector<float> sam3_precompute_geom_input(
    const sam3_model& model,
    const sam3_pcs_params& params,
    const float* img_feats_data,  // [D, W, H] ggml-layout backbone features (nullable if no boxes)
    int W_feat, int H_feat) {
    const auto& ge = model.geom_enc;
    const int D = model.hparams.neck_dim;  // 256
    const int roi_size = 7;
    const int num_pos_feats = D / 2;  // 128

    // Collect all exemplar boxes
    struct box_info {
        float cx, cy, w, h;
        int label;
    };
    std::vector<box_info> boxes;
    // Label convention: index 1 = POSITIVE, index 0 = NEGATIVE (same as
    // SAM's point labels). This was originally coded inverted (pos->0), and
    // since nothing had ever exercised exemplars, it went unnoticed until
    // real-image validation: with pos->0, a positive cat exemplar KILLED the
    // cat detection (0 results) while "pos(cat)+neg(ear)" detected both
    // ears -- exactly the behavior of swapped labels. With pos->1 the same
    // matrix behaves correctly (see docs/sam3/PLAN.md exemplar validation).
    // NOTE: boxes arrive NORMALIZED [0,1] XYXY (see sam3_pcs_params) -- the
    // cx/cy/w/h below stay in [0,1], which the ROI-align grid conversion
    // (fx0 = (cx - w/2) * W_feat) depends on.
    for (const auto& b : params.pos_exemplars) {
        float cx = (b.x0 + b.x1) * 0.5f;
        float cy = (b.y0 + b.y1) * 0.5f;
        float bw = b.x1 - b.x0;
        float bh = b.y1 - b.y0;
        boxes.push_back({cx, cy, bw, bh, 1});  // label 1 = positive
    }
    for (const auto& b : params.neg_exemplars) {
        float cx = (b.x0 + b.x1) * 0.5f;
        float cy = (b.y0 + b.y1) * 0.5f;
        float bw = b.x1 - b.x0;
        float bh = b.y1 - b.y0;
        boxes.push_back({cx, cy, bw, bh, 0});  // label 0 = negative
    }

    const int n_boxes = (int)boxes.size();
    const int n_emb = (int)params.exemplar_embeddings.size();
    // Rows processed through final_proj+norm here: boxes + CLS. Precomputed
    // embedding rows are already post-proj/post-norm (they were captured
    // from a previous precompute's output) and are spliced in verbatim
    // afterwards -- final layout: [boxes..., embeddings..., CLS].
    const int N_geo = n_boxes + 1;

    // Read needed weights from GPU
    std::vector<float> box_proj_w_data(4 * D), box_proj_b_data(D);
    std::vector<float> type_embed_data(D * 2);
    std::vector<float> cls_data(D);
    std::vector<float> box_pos_proj_w_data(258 * D), box_pos_proj_b_data(D);
    std::vector<float> box_pool_proj_w_data(7 * 7 * D * D), box_pool_proj_b_data(D);
    std::vector<float> img_pre_norm_w_data(D), img_pre_norm_b_data(D);

    auto read_f32 = [](struct ggml_tensor* t, float* dst, size_t n) {
        sam3_read_f32(t, dst, n);
    };

    read_f32(ge.box_proj_w, box_proj_w_data.data(), 4 * D);
    read_f32(ge.box_proj_b, box_proj_b_data.data(), D);
    read_f32(ge.type_embed, type_embed_data.data(), D * 2);
    read_f32(ge.cls_token, cls_data.data(), D);
    read_f32(ge.box_pos_proj_w, box_pos_proj_w_data.data(), 258 * D);
    read_f32(ge.box_pos_proj_b, box_pos_proj_b_data.data(), D);

    if (n_boxes > 0) {
        read_f32(ge.box_pool_proj_w, box_pool_proj_w_data.data(), 7 * 7 * D * D);
        read_f32(ge.box_pool_proj_b, box_pool_proj_b_data.data(), D);
        read_f32(ge.img_pre_norm_w, img_pre_norm_w_data.data(), D);
        read_f32(ge.img_pre_norm_b, img_pre_norm_b_data.data(), D);
    }

    // Output: [D * N_geo] in row-major [D, N_geo] ggml order
    std::vector<float> out(D * N_geo, 0.0f);

    // Encode each box
    for (int bi = 0; bi < n_boxes; ++bi) {
        const auto& box = boxes[bi];
        float embed[256] = {};

        // 1. Direct projection: Linear(4, D)
        // box_proj_w: ggml [4, D] = PyTorch [D, 4]
        // out = x @ W^T + b where x=[cx,cy,w,h]
        {
            float coords[4] = {box.cx, box.cy, box.w, box.h};
            for (int d = 0; d < D; ++d) {
                float sum = box_proj_b_data[d];
                for (int j = 0; j < 4; ++j)
                    sum += coords[j] * box_proj_w_data[j + d * 4];  // ggml: [4, D] stride
                embed[d] = sum;
            }
        }

        // 2. ROI Align + Conv2d(D, D, 7) pool projection
        if (img_feats_data) {
            // Apply img_pre_norm (LayerNorm) to image features before pooling
            // For simplicity, we compute LayerNorm per spatial position
            // Actually, LayerNorm is applied to the [D] dimension at each spatial location
            // But we need the full feature map, so let's normalize in-place copy
            const int N_spatial = W_feat * H_feat;
            std::vector<float> normed_feats(D * N_spatial);

            for (int s = 0; s < N_spatial; ++s) {
                // Compute mean and variance over D dimension
                float mean = 0.0f;
                for (int d = 0; d < D; ++d)
                    mean += img_feats_data[d + s * D];
                mean /= D;

                float var = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float diff = img_feats_data[d + s * D] - mean;
                    var += diff * diff;
                }
                var /= D;

                float inv_std = 1.0f / sqrtf(var + 1e-5f);
                for (int d = 0; d < D; ++d) {
                    normed_feats[d + s * D] =
                        (img_feats_data[d + s * D] - mean) * inv_std * img_pre_norm_w_data[d] + img_pre_norm_b_data[d];
                }
            }

            // Convert CxCyWH [0,1] → XYXY in feature grid coordinates
            float fx0 = (box.cx - box.w * 0.5f) * (float)W_feat;
            float fy0 = (box.cy - box.h * 0.5f) * (float)H_feat;
            float fx1 = (box.cx + box.w * 0.5f) * (float)W_feat;
            float fy1 = (box.cy + box.h * 0.5f) * (float)H_feat;

            // ROI Align
            std::vector<float> roi_data(D * roi_size * roi_size);
            sam3_roi_align_single(normed_feats.data(), D, W_feat, H_feat,
                                  fx0, fy0, fx1, fy1, roi_size, roi_data.data());

            // Conv2d(D, D, 7): kernel [7, 7, D, D] in ggml = [D_out, D_in, kH, kW] in PyTorch
            // Since roi_size = 7 = kernel_size, output is [D, 1, 1]
            // This is effectively a matrix multiply: out[d_out] = sum over (d_in, kh, kw)
            for (int d_out = 0; d_out < D; ++d_out) {
                float sum = box_pool_proj_b_data[d_out];
                for (int kh = 0; kh < roi_size; ++kh) {
                    for (int kw = 0; kw < roi_size; ++kw) {
                        for (int d_in = 0; d_in < D; ++d_in) {
                            // ggml weight layout: [kW=7, kH=7, D_in=256, D_out=256]
                            int w_idx = kw + kh * 7 + d_in * 7 * 7 + d_out * 7 * 7 * D;
                            // roi_data layout: [C, W, H] = [D_in, kW, kH]
                            int r_idx = d_in + kw * D + kh * D * roi_size;
                            sum += box_pool_proj_w_data[w_idx] * roi_data[r_idx];
                        }
                    }
                }
                embed[d_out] += sum;
            }
        }

        // 3. Sinusoidal positional encoding + Linear(258, D)
        {
            float pos_enc[258];
            sam3_sine_encode_box(pos_enc, box.cx, box.cy, box.w, box.h,
                                 num_pos_feats, 10000);

            for (int d = 0; d < D; ++d) {
                float sum = box_pos_proj_b_data[d];
                for (int j = 0; j < 258; ++j)
                    sum += pos_enc[j] * box_pos_proj_w_data[j + d * 258];
                embed[d] += sum;
            }
        }

        // 4. Label embedding
        {
            int label = box.label;
            for (int d = 0; d < D; ++d)
                embed[d] += type_embed_data[d + label * D];
        }

        // Store in output: ggml layout [D, N_geo] — embed goes at column bi
        for (int d = 0; d < D; ++d)
            out[d + bi * D] = embed[d];
    }

    // CLS token goes at position n_boxes (last)
    for (int d = 0; d < D; ++d)
        out[d + n_boxes * D] = cls_data[d];

    // Apply final_proj (Linear(D,D)) + LayerNorm on CPU
    std::vector<float> proj_w(D * D), proj_b(D);
    std::vector<float> ln_w(D), ln_b(D);
    read_f32(ge.post_proj_w, proj_w.data(), D * D);
    read_f32(ge.post_proj_b, proj_b.data(), D);
    read_f32(ge.norm_w, ln_w.data(), D);
    read_f32(ge.norm_b, ln_b.data(), D);

    std::vector<float> projected(D * N_geo);
    for (int t = 0; t < N_geo; ++t) {
        // Linear: y = W @ x + b
        // W is stored in PyTorch row-major [D_out, D_in] → ggml [D_in, D_out]
        // proj_w[d_in + d_out * D] = W_py[d_out, d_in]
        for (int d_out = 0; d_out < D; ++d_out) {
            float sum = proj_b[d_out];
            for (int d_in = 0; d_in < D; ++d_in)
                sum += out[d_in + t * D] * proj_w[d_in + d_out * D];
            projected[d_out + t * D] = sum;
        }
    }
    if (n_boxes > 0) {
        fprintf(stderr, "%s: %d exemplar boxes encoded (%d pos, %d neg)\n",
                __func__, n_boxes,
                (int)params.pos_exemplars.size(),
                (int)params.neg_exemplars.size());
    }

    // LayerNorm over D dimension for each token
    for (int t = 0; t < N_geo; ++t) {
        float mean = 0.0f;
        for (int d = 0; d < D; ++d)
            mean += projected[d + t * D];
        mean /= D;

        float var = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = projected[d + t * D] - mean;
            var += diff * diff;
        }
        var /= D;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int d = 0; d < D; ++d) {
            float normalized = (projected[d + t * D] - mean) * inv_std;
            out[d + t * D] = normalized * ln_w[d] + ln_b[d];
        }
    }

    if (n_emb > 0) {
        // Splice precomputed concept-embedding rows in before the CLS row.
        // They are captured post-proj/post-norm (see
        // sam3_pcs_compute_exemplar_embedding), so no further processing.
        std::vector<float> final_out(D * (n_boxes + n_emb + 1));
        std::copy(out.begin(), out.begin() + (size_t)n_boxes * D, final_out.begin());
        for (int e = 0; e < n_emb; ++e) {
            const auto& row = params.exemplar_embeddings[e];
            if ((int)row.size() != D) {
                fprintf(stderr, "%s: exemplar_embeddings[%d] has %zu floats, expected %d -- zero-filling\n",
                        __func__, e, row.size(), D);
                std::fill_n(final_out.begin() + (size_t)(n_boxes + e) * D, D, 0.0f);
                continue;
            }
            std::copy(row.begin(), row.end(), final_out.begin() + (size_t)(n_boxes + e) * D);
        }
        std::copy(out.begin() + (size_t)n_boxes * D, out.begin() + (size_t)(n_boxes + 1) * D,
                  final_out.begin() + (size_t)(n_boxes + n_emb) * D);  // CLS stays last
        return final_out;
    }
    return out;
}

/*****************************************************************************
** Fusion encoder — graph building (6 layers)
*****************************************************************************/

// Single fusion encoder layer.
// x: [D, N, B] image features (N=5184), prompt: [D, T, B] text/exemplar tokens, pos: [D, N, B]
// Returns: updated x [D, N, B]
static struct ggml_tensor* sam3_fenc_layer_forward(
    struct ggml_context* ctx,
    const sam3_fenc_layer& ly,
    struct ggml_tensor* x,
    struct ggml_tensor* prompt,
    struct ggml_tensor* pos,
    struct ggml_tensor* prompt_attn_bias,
    int n_heads) {
    // Self-attention: Q/K get positional encoding, V does not
    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm1_w, ly.norm1_b);
        auto* q_in = ggml_add(ctx, x_norm, pos);
        auto* k_in = ggml_add(ctx, x_norm, pos);

        const int64_t D = x->ne[0];

        auto* q_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]);

        auto* q_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float));

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, x_norm), v_b);

        const int64_t N = x->ne[1];
        const int64_t B = x->ne[2];
        const int64_t HD = D / n_heads;

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        float scale = 1.0f / sqrtf((float)HD);
        auto* sa_out = sam3_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        sa_out = ggml_reshape_3d(ctx, sa_out, D, N, B);

        sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
        sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

        x = ggml_add(ctx, shortcut, sa_out);
    }

    // Cross-attention: Q from image features, K/V from prompt tokens.
    // ca_q_w stores fused [D, 3*D] weights split as Q-proj, K-proj, V-proj.
    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm2_w, ly.norm2_b);
        const int64_t D = x->ne[0];

        auto* q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);

        auto* q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, x_norm), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, prompt), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, prompt), v_b);

        const int64_t N_q = x->ne[1];
        const int64_t N_kv = prompt->ne[1];
        const int64_t B = x->ne[2];
        const int64_t HD = D / n_heads;

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        auto* ca_mask = sam3_expand_token_attn_bias(ctx, prompt_attn_bias, N_q, n_heads, B);
        float scale = 1.0f / sqrtf((float)HD);
        auto* ca_out = sam3_flash_attn_ext(ctx, Q, K, V, ca_mask, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);

        ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

        x = ggml_add(ctx, shortcut, ca_out);
    }

    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm3_w, ly.norm3_b);

        auto* ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, x_norm);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
        ffn = ggml_relu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);

        x = ggml_add(ctx, shortcut, ffn);
    }

    return x;
}

// Build full fusion encoder graph (6 layers).
// image_feats: [D, N, B] where N=5184 (72*72), D=256
// prompt_tokens: [D, T, B] text/exemplar features
// pos_enc: [D, N, B] sinusoidal positional encoding for image features
// Returns: conditioned_features [D, N, B]
struct ggml_tensor* sam3_build_fenc_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* image_feats,
    struct ggml_tensor* prompt_tokens,
    struct ggml_tensor* pos_enc,
    struct ggml_tensor* prompt_attn_bias) {
    const auto& hp = model.hparams;
    auto* x = image_feats;

    for (int i = 0; i < hp.fenc_layers; ++i) {
        x = sam3_fenc_layer_forward(ctx, model.fenc.layers[i], x, prompt_tokens,
                                    pos_enc, prompt_attn_bias, hp.fenc_heads);
        sam3_name_tensorf(x, "fenc_layer%d_out", i);
    }

    return x;
}

/*****************************************************************************
** DETR decoder — graph building (6 layers)
*****************************************************************************/

// inverse_sigmoid: log(x / (1 - x)), clamped to avoid inf
// Python reference uses eps=1e-3: x1 = x.clamp(min=eps), x2 = (1-x).clamp(min=eps)
static struct ggml_tensor* sam3_inverse_sigmoid(struct ggml_context* ctx, struct ggml_tensor* x) {
    // clamp x to [1e-3, 1-1e-3] to match Python eps=1e-3
    x = ggml_clamp(ctx, x, 1e-3f, 1.0f - 1e-3f);
    // log(x / (1 - x)) = log(x) - log(1 - x)
    auto* log_x = ggml_log(ctx, x);
    // Compute (1 - x) as (-1)*x + 1.  We use ggml_scale_bias which takes float
    // scalars (no tensor allocation needed, safe in no_alloc contexts).
    auto* one_minus = ggml_scale_bias(ctx, x, -1.0f, 1.0f);
    auto* log_1mx = ggml_log(ctx, one_minus);
    return ggml_sub(ctx, log_x, log_1mx);
}

// Box refinement MLP (3 layers: D→D→D→4 with ReLU)
static struct ggml_tensor* sam3_bbox_mlp(struct ggml_context* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* w[3],
                                         struct ggml_tensor* b[3]) {
    for (int j = 0; j < 3; ++j) {
        x = ggml_mul_mat(ctx, w[j], x);
        x = ggml_add(ctx, x, b[j]);
        if (j < 2) x = ggml_relu(ctx, x);
    }
    return x;
}

// Build sinusoidal positional embedding for 4D reference points in the ggml graph.
// ref_boxes: [4, NQ, B] — (cx, cy, w, h) after sigmoid, B=1
// sine_dim_t: [1, 64] — pre-computed angle multipliers (2π / 10000^(2i/128))
// Returns: [512, NQ, B] sinusoidal embedding matching Python gen_sineembed_for_position
static struct ggml_tensor* sam3_build_sine_pos_embed_4d(
    struct ggml_context* ctx,
    struct ggml_tensor* ref_boxes,     // [4, NQ, B]
    struct ggml_tensor* sine_dim_t) {  // [1, 64]
    const int64_t NQ = ref_boxes->ne[1];

    // Python output order: [cy, cx, w, h] → coord indices from boxes [cx(0),cy(1),w(2),h(3)]
    const int coord_order[4] = {1, 0, 2, 3};

    struct ggml_tensor* coord_embeds[4];

    for (int c = 0; c < 4; ++c) {
        int ci = coord_order[c];
        // Extract one coordinate: view into ref_boxes [4, NQ, 1] at element ci
        auto* coord = ggml_view_2d(ctx, ref_boxes, 1, NQ,
                                   ref_boxes->nb[1], ci * sizeof(float));  // [1, NQ]

        // Outer product: angles[i, q] = dim_t[i] * coord[q]
        // ggml_mul_mat(A=[1,64], B=[1,NQ]) = A^T @ B = [64,1]@[1,NQ] = [64, NQ]
        auto* angles = ggml_mul_mat(ctx, sine_dim_t, coord);  // [64, NQ]

        auto* sin_vals = ggml_sin(ctx, angles);  // [64, NQ]
        auto* cos_vals = ggml_cos(ctx, angles);  // [64, NQ]

        // Interleave: [sin_0, cos_0, sin_1, cos_1, ...]
        auto* sin_r = ggml_reshape_3d(ctx, sin_vals, 1, 64, NQ);
        auto* cos_r = ggml_reshape_3d(ctx, cos_vals, 1, 64, NQ);
        auto* interleaved = ggml_concat(ctx, sin_r, cos_r, 0);  // [2, 64, NQ]
        coord_embeds[c] = ggml_reshape_2d(ctx, interleaved, 128, NQ);
    }

    // Concatenate all 4 coordinates → [512, NQ]
    auto* embed = ggml_concat(ctx, coord_embeds[0], coord_embeds[1], 0);  // [256, NQ]
    embed = ggml_concat(ctx, embed, coord_embeds[2], 0);                  // [384, NQ]
    embed = ggml_concat(ctx, embed, coord_embeds[3], 0);                  // [512, NQ]

    return embed;
}

// Build query positional encoding from reference boxes via sine embed + ref_point_head MLP.
// ref_boxes: [4, NQ, 1] — after sigmoid
// sine_dim_t: [1, 64]
// Returns: [D, NQ+1, 1] (zeros for presence token at index 0, MLP output for object queries)
static struct ggml_tensor* sam3_build_query_pos(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* ref_boxes,   // [4, NQ, 1]
    struct ggml_tensor* sine_dim_t,  // [1, 64]
    int layer_idx = -1) {
    const auto& tensors = model.tensors;
    const int64_t NQ = ref_boxes->ne[1];
    const int D = model.hparams.neck_dim;  // 256

    // 1. Sine positional embedding: [512, NQ]
    auto* sine_embed = sam3_build_sine_pos_embed_4d(ctx, ref_boxes, sine_dim_t);
    if (layer_idx == 0) {
        ggml_set_name(sine_embed, "ddec_query_sine_0");
    }

    // 2. ref_point_head MLP: 512 → 256 → 256
    // Layer 0: relu(W0 @ sine_embed + b0)
    auto* h = ggml_mul_mat(ctx, tensors.at("ddec.ref_point_head.layers.0.weight"), sine_embed);
    h = ggml_add(ctx, h, tensors.at("ddec.ref_point_head.layers.0.bias"));
    h = ggml_relu(ctx, h);
    // Layer 1: W1 @ h + b1 (no activation)
    auto* qpos_obj = ggml_mul_mat(ctx, tensors.at("ddec.ref_point_head.layers.1.weight"), h);
    qpos_obj = ggml_add(ctx, qpos_obj, tensors.at("ddec.ref_point_head.layers.1.bias"));
    // qpos_obj: [D, NQ]
    if (layer_idx == 0) {
        ggml_set_name(qpos_obj, "ddec_query_pos_0");
    }

    // 3. Reshape to 3D and prepend zeros for presence token
    qpos_obj = ggml_reshape_3d(ctx, qpos_obj, D, NQ, 1);
    auto* qpos_pres = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    ggml_set_name(qpos_pres, "ddec_query_pos_pres");
    ggml_set_input(qpos_pres);  // zeros — set by caller

    return ggml_concat(ctx, qpos_pres, qpos_obj, 1);  // [D, NQ+1, 1]
}

// Build box-relative positional bias for DETR cross-attention.
// ref_boxes: [4, N_q, B] — (cx, cy, w, h) in [0,1]
// rpb_coords: [feat_hw] — normalized coords [0/H, 1/H, ..., (H-1)/H] (input tensor)
// Returns: bias tensor [N_kv, N_q+1, n_heads, B] for ggml_flash_attn_ext mask
//
// Python _get_rpb_matrix with boxRPB="log":
//   1. boxes → xyxy
//   2. deltas_x[q,w,:2] = [coord_w - x0, coord_w - x1]
//   3. deltas_y[q,h,:2] = [coord_h - y0, coord_h - y1]
//   4. log transform: sign(d*8) * log2(|d*8|+1) / log2(8)
//   5. MLP: [2] → [256] → [n_heads]
//   6. outer sum: B[h,w] = delta_y[h] + delta_x[w]
static struct ggml_tensor* sam3_compute_box_rpb(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* ref_boxes,   // [4, N_q, B]
    struct ggml_tensor* rpb_coords,  // [feat_hw] — pre-filled grid coordinates
    int feat_hw,
    int layer_idx = -1) {
    const int64_t NQ = ref_boxes->ne[1];
    const int NH = model.hparams.ddec_heads;  // 8
    const int W = feat_hw;
    const int H = feat_hw;
    const auto& tensors = model.tensors;

    // ── 1. Convert cxcywh → xyxy ─────────────────────────────────────────
    // ggml_view_2d on strided data is non-contiguous — ggml_scale requires contiguous.
    // Use ggml_cont to make each coordinate slice contiguous.
    auto* cx = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 0));
    auto* cy = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 1 * sizeof(float)));
    auto* bw = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 2 * sizeof(float)));
    auto* bh = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 3 * sizeof(float)));
    // x0 = cx - w/2, x1 = cx + w/2
    auto* half_w = ggml_scale(ctx, bw, 0.5f);
    auto* half_h = ggml_scale(ctx, bh, 0.5f);
    auto* x0 = ggml_sub(ctx, cx, half_w);  // [1, NQ]
    auto* x1 = ggml_add(ctx, cx, half_w);
    auto* y0 = ggml_sub(ctx, cy, half_h);
    auto* y1 = ggml_add(ctx, cy, half_h);

    // ── 2. Compute deltas via outer subtract ──────────────────────────────
    // coords: [W] → reshape to [W, 1] for outer subtract
    auto* cw = ggml_reshape_2d(ctx, rpb_coords, W, 1);  // [W, 1]

    // Outer subtract: delta[w, q] = coord[w] - edge[q]
    // Use ggml_mul_mat trick: not applicable for subtraction.
    // Instead: repeat coords to [W, NQ], repeat edge to [W, NQ], subtract.
    auto* shape_wn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, NQ);

    auto* cw_rep = ggml_repeat(ctx, cw, shape_wn);         // [W, NQ] (each column = coords)
    auto* x0_t = ggml_cont(ctx, ggml_transpose(ctx, x0));  // [NQ, 1]
    auto* x0_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, x0_t, 1, NQ), shape_wn);
    auto* x1_t = ggml_cont(ctx, ggml_transpose(ctx, x1));
    auto* x1_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, x1_t, 1, NQ), shape_wn);
    auto* y0_t = ggml_cont(ctx, ggml_transpose(ctx, y0));
    auto* y0_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, y0_t, 1, NQ), shape_wn);
    auto* y1_t = ggml_cont(ctx, ggml_transpose(ctx, y1));
    auto* y1_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, y1_t, 1, NQ), shape_wn);

    auto* dx0 = ggml_sub(ctx, cw_rep, x0_rep);  // [W, NQ]
    auto* dx1 = ggml_sub(ctx, cw_rep, x1_rep);
    auto* dy0 = ggml_sub(ctx, cw_rep, y0_rep);  // reusing coords for H (H==W)
    auto* dy1 = ggml_sub(ctx, cw_rep, y1_rep);

    // Stack into [2, W, NQ]: reshape each to [1, W, NQ], concat dim 0
    auto* dx0_r = ggml_reshape_3d(ctx, dx0, 1, W, NQ);
    auto* dx1_r = ggml_reshape_3d(ctx, dx1, 1, W, NQ);
    auto* deltas_x = ggml_concat(ctx, dx0_r, dx1_r, 0);  // [2, W, NQ]
    auto* dy0_r = ggml_reshape_3d(ctx, dy0, 1, H, NQ);
    auto* dy1_r = ggml_reshape_3d(ctx, dy1, 1, H, NQ);
    auto* deltas_y = ggml_concat(ctx, dy0_r, dy1_r, 0);  // [2, H, NQ]

    // ── 3. Log transform: sign(d*8) * log2(|d*8|+1) / log2(8) ────────────
    const float scale8 = 8.0f;
    const float inv_log2_8 = 1.0f / log2f(8.0f);  // = 1/3

    auto rpb_log = [&](struct ggml_tensor* d) -> struct ggml_tensor* {
        auto* d8 = ggml_scale(ctx, d, scale8);
        auto* sign_d = ggml_sgn(ctx, d8);
        auto* abs_d = ggml_abs(ctx, d8);
        auto* log_val = ggml_log(ctx, ggml_scale_bias(ctx, abs_d, 1.0f, 1.0f));
        // log2(x) = ln(x) / ln(2)
        log_val = ggml_scale(ctx, log_val, 1.0f / logf(2.0f));
        return ggml_mul(ctx, sign_d, ggml_scale(ctx, log_val, inv_log2_8));
    };

    deltas_x = rpb_log(deltas_x);  // [2, W, NQ]
    deltas_y = rpb_log(deltas_y);  // [2, H, NQ]

    // ── 4. MLP: [2, W*NQ] → [NH, W*NQ] ───────────────────────────────────
    // boxRPB_embed_x: MLP(2, 256, 8, 2) = Linear(2→256)+ReLU, Linear(256→8)
    // Reshape to [2, W*NQ] so matmul treats each (w, q) pair as a sample
    auto rpb_mlp = [&](struct ggml_tensor* d, const char* axis) -> struct ggml_tensor* {
        int64_t spatial = d->ne[1];
        int64_t nq = d->ne[2];
        auto* flat = ggml_reshape_2d(ctx, d, 2, spatial * nq);  // [2, W*NQ]
        auto wn0 = std::string("ddec.boxRPB_embed_") + axis + ".layers.0.weight";
        auto bn0 = std::string("ddec.boxRPB_embed_") + axis + ".layers.0.bias";
        auto wn1 = std::string("ddec.boxRPB_embed_") + axis + ".layers.1.weight";
        auto bn1 = std::string("ddec.boxRPB_embed_") + axis + ".layers.1.bias";
        flat = ggml_mul_mat(ctx, tensors.at(wn0), flat);
        flat = ggml_add(ctx, flat, tensors.at(bn0));
        flat = ggml_relu(ctx, flat);
        flat = ggml_mul_mat(ctx, tensors.at(wn1), flat);
        flat = ggml_add(ctx, flat, tensors.at(bn1));
        // flat: [NH, W*NQ] → reshape to [NH, spatial, NQ]
        return ggml_reshape_3d(ctx, flat, NH, spatial, nq);
    };

    auto* rpb_x = rpb_mlp(deltas_x, "x");  // [NH, W, NQ]
    auto* rpb_y = rpb_mlp(deltas_y, "y");  // [NH, H, NQ]

    // ── 5. Outer sum: B[nh, w, h, q] = rpb_y[nh, h, q] + rpb_x[nh, w, q] ─
    // Reshape for broadcasting:
    //   rpb_y → [NH, 1, H, NQ]
    //   rpb_x → [NH, W, 1, NQ]
    //
    // Keep W in ne[1] so reshaping [NH, W, H, NQ] → [NH, H*W, NQ, 1]
    // preserves Python's flatten(H, W) order where W is the fast spatial axis.
    auto* rpb_y_4d = ggml_reshape_4d(ctx, rpb_y, NH, 1, H, NQ);
    auto* rpb_x_4d = ggml_reshape_4d(ctx, rpb_x, NH, W, 1, NQ);

    // ggml_add broadcasts: where one dim is 1, the other is used
    auto* rpb_hw = ggml_repeat(ctx, rpb_y_4d,
                               ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NH, W, H, NQ));
    auto* rpb_hw_x = ggml_repeat(ctx, rpb_x_4d,
                                 ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NH, W, H, NQ));
    auto* rpb = ggml_add(ctx, rpb_hw, rpb_hw_x);  // [NH, W, H, NQ]

    // ── 6. Reshape to [H*W, NQ, NH, 1] for flash_attn_ext mask ───────────
    // Current: [NH, W, H, NQ]. Need: [N_kv=H*W, NQ, NH, B=1]
    rpb = ggml_reshape_4d(ctx, rpb, NH, H * W, NQ, 1);
    rpb = ggml_cont(ctx, ggml_permute(ctx, rpb, 2, 0, 1, 3));  // [H*W, NQ, NH, 1]

    // Prepend zeros for presence token: mask for presence token has no box-relative bias
    auto* pres_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H * W, 1, NH, 1);
    ggml_set_name(pres_mask, "rpb_pres_zeros");
    ggml_set_input(pres_mask);  // zeros — set by caller

    // [H*W, NQ+1, NH, 1]
    auto* full_rpb = ggml_concat(ctx, pres_mask, rpb, 1);
    full_rpb = ggml_cont(ctx, full_rpb);
    if (layer_idx == 0) {
        ggml_set_name(full_rpb, "ddec_rpb_mask_0");
    }

    return full_rpb;
}

// Single DETR decoder layer.
// queries: [D, N_q, B] where N_q = 201 (200 object queries + 1 presence token)
// query_pos: [D, N_q, B] positional encoding for queries
// enc_feats: [D, N_kv, B] conditioned image features from fusion encoder
// enc_pos: [D, N_kv, B] positional encoding for image features
// text_feats: [D, T, B] text features
// rpb_mask: [N_kv, N_q, n_heads, B] box-relative positional bias (or nullptr)
// Returns: updated queries [D, N_q, B]
static struct ggml_tensor* sam3_ddec_layer_forward(
    struct ggml_context* ctx,
    const sam3_ddec_layer& ly,
    struct ggml_tensor* queries,
    struct ggml_tensor* query_pos,
    struct ggml_tensor* enc_feats,
    struct ggml_tensor* enc_pos,
    struct ggml_tensor* text_feats,
    int n_heads,
    struct ggml_tensor* text_attn_bias = nullptr,
    struct ggml_tensor* rpb_mask = nullptr,
    int layer_idx = -1) {
    const int64_t D = queries->ne[0];

    // Python decoder layer order (all post-norm):
    //   1. Self-attention → norm2 (post-norm)
    //   2. Text cross-attention (ca_text) → catext_norm (post-norm)
    //   3. Image cross-attention (cross_attn) → norm1 (post-norm)
    //   4. FFN → norm3 (post-norm)
    //
    // Norm weight mapping:
    //   ly.norm2_w  = ".norm2.weight"        = Python norm2 (post-SA)
    //   ly.norm3_w  = ".norm_ca_text.weight"  = Python catext_norm (post-text-CA)
    //   ly.norm1_w  = ".norm1.weight"         = Python norm1 (post-image-CA)
    //   ly.norm4_w  = ".norm3.weight"         = Python norm3 (post-FFN)

    // 1. Self-attention among queries (post-norm)
    {
        // Q = K = queries + query_pos, V = queries (no pos)
        auto* q_in = ggml_add(ctx, queries, query_pos);

        auto* q_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float));

        const int64_t N = queries->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, q_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, queries), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        float scale = 1.0f / sqrtf((float)HD);
        auto* sa_out = sam3_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        sa_out = ggml_reshape_3d(ctx, sa_out, D, N, B);
        sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
        sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

        queries = ggml_add(ctx, queries, sa_out);
        // Post-norm: norm2 (Python's post-SA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm2_w, ly.norm2_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_sa");
        }
    }

    // 2. Cross-attention to text tokens (post-norm)
    {
        // Q = queries + query_pos, K = V = text_feats
        auto* q_in = ggml_add(ctx, queries, query_pos);

        auto* q_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], D * ly.ca_text_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], 2 * D * ly.ca_text_q_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, 2 * D * sizeof(float));

        const int64_t N_q = queries->ne[1];
        const int64_t N_kv = text_feats->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, text_feats), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, text_feats), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        auto* text_mask = sam3_expand_token_attn_bias(ctx, text_attn_bias, N_q, n_heads, B);
        float scale = 1.0f / sqrtf((float)HD);
        auto* ca_out = sam3_flash_attn_ext(ctx, Q, K, V, text_mask, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);
        ca_out = ggml_mul_mat(ctx, ly.ca_text_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_text_out_b);

        queries = ggml_add(ctx, queries, ca_out);
        // Post-norm: catext_norm (Python's post-text-CA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm3_w, ly.norm3_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_text_ca");
        }
    }

    // 3. Cross-attention to conditioned image features (post-norm)
    {
        // Q = queries + query_pos, K = enc_feats + enc_pos, V = enc_feats
        auto* q_in = ggml_add(ctx, queries, query_pos);
        auto* k_in = ggml_add(ctx, enc_feats, enc_pos);

        auto* q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

        const int64_t N_q = queries->ne[1];
        const int64_t N_kv = enc_feats->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, enc_feats), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        float scale = 1.0f / sqrtf((float)HD);
        struct ggml_tensor* ca_out = nullptr;

        if (rpb_mask) {
            // Keep the box-relative positional bias in fp32. The CPU flash-attn
            // kernel reads mask storage as fp16, which is good enough for token
            // padding masks but introduces avoidable drift here.
            auto* kq = ggml_mul_mat(ctx, K, Q);                      // [N_kv, N_q, NH, B]
            kq = ggml_soft_max_ext(ctx, kq, rpb_mask, scale, 0.0f);  // [N_kv, N_q, NH, B]

            auto* v_t = ggml_cont(ctx, ggml_transpose(ctx, V));  // [N_kv, HD, NH, B]
            ca_out = ggml_mul_mat(ctx, v_t, kq);                 // [HD, N_q, NH, B]
            ca_out = ggml_cont(ctx, ggml_permute(ctx, ca_out, 0, 2, 1, 3));
        } else {
            ca_out = sam3_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        }

        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);
        ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

        queries = ggml_add(ctx, queries, ca_out);
        // Post-norm: norm1 (Python's post-image-CA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm1_w, ly.norm1_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_img_ca");
        }
    }

    // 4. FFN (post-norm, ReLU)
    {
        auto* ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, queries);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
        ffn = ggml_relu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);

        queries = ggml_add(ctx, queries, ffn);
        // Post-norm: norm3 (Python's post-FFN norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm4_w, ly.norm4_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_full_out");
        }
    }

    return queries;
}

// DotProductScoring: classify queries against text features via dot product.
//
// Python reference (DotProductScoring.forward):
//   1. prompt_mlp(prompt) → residual MLP + LN on text features
//   2. mean_pool_text(result, prompt_mask) → pooled [BS, D] (only valid tokens)
//   3. prompt_proj(pooled) → [BS, D]
//   4. hs_proj(hs) → [num_layer, BS, N_q, D]
//   5. matmul(proj_hs, proj_pooled.unsqueeze(-1)) → dot product → [num_layer, BS, N_q, 1]
//   6. scale by 1/sqrt(D)
//   7. clamp to [-12, 12]
//
// query_outputs: [D, N_q, B] — the 200 object query outputs
// text_features: [D, T, B] — text encoder output (already through resizer)
// text_valid_mask: [T, 1, B] — 1.0 for valid tokens, 0.0 for padding (or nullptr for all-valid)
// Returns: class_scores [N_q, B] (one score per query per batch)
static struct ggml_tensor* sam3_dot_product_scoring(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* query_outputs,    // [D, N_q, B]
    struct ggml_tensor* text_features,    // [D, T, B]
    struct ggml_tensor* text_valid_mask)  // [T, 1, B] or nullptr
{
    const auto& tensors = model.tensors;
    const int64_t D = query_outputs->ne[0];  // 256
    const int64_t T = text_features->ne[1];
    const int64_t B = text_features->ne[2];

    // Step 1: Apply prompt_mlp on text features (residual MLP + LayerNorm)
    auto* text_mlp = text_features;  // [D, T, B]
    auto* orig_text = text_features;
    text_mlp = ggml_mul_mat(ctx, tensors.at("scoring.prompt_mlp.layers.0.weight"), text_mlp);
    text_mlp = ggml_add(ctx, text_mlp, tensors.at("scoring.prompt_mlp.layers.0.bias"));
    text_mlp = ggml_relu(ctx, text_mlp);
    text_mlp = ggml_mul_mat(ctx, tensors.at("scoring.prompt_mlp.layers.1.weight"), text_mlp);
    text_mlp = ggml_add(ctx, text_mlp, tensors.at("scoring.prompt_mlp.layers.1.bias"));
    text_mlp = ggml_add(ctx, text_mlp, orig_text);
    text_mlp = sam3_layer_norm(ctx, text_mlp,
                               tensors.at("scoring.prompt_mlp.out_norm.weight"),
                               tensors.at("scoring.prompt_mlp.out_norm.bias"));
    // text_mlp: [D, T, B]
    ggml_set_name(text_mlp, "scoring_prompt_mlp_out");

    // Step 2: Mean-pool over valid text tokens → [D, 1, B]
    // Python: pooled = (prompt * is_valid).sum(0) / num_valid
    struct ggml_tensor* text_pooled;
    if (text_valid_mask) {
        // Permute to [T, D, B] for masked pooling
        auto* tp = ggml_cont(ctx, ggml_permute(ctx, text_mlp, 1, 0, 2, 3));  // [T, D, B]
        // Mask: text_valid_mask is [T, 1, B] — broadcast multiply zeros out padding
        tp = ggml_mul(ctx, tp, text_valid_mask);  // [T, D, B] with padding zeroed
        // Sum over T dimension: pool_1d with SUM kernel=T
        // ggml_pool_1d AVG divides by T; we want SUM then divide by n_valid.
        // Use AVG and then scale by T/n_valid? Or use a manual approach.
        // Simpler: sum via pool_1d with AVG, then scale by T/n_valid.
        // But n_valid is dynamic. Instead: sum = mean * T, then divide by n_valid.
        // We pass n_valid as part of the mask: text_valid_mask sums to n_valid.
        // pool_1d(masked, AVG, T, T, 0) = sum(masked) / T. Multiply by T → sum(masked).
        // Then divide by n_valid. But n_valid is a scalar we know CPU-side.
        // For simplicity: compute AVG over ALL T positions (with padding zeroed out).
        // This gives sum(valid) / T. To get sum(valid) / n_valid, scale by T / n_valid.
        // We embed the scale factor into the mask: mask = (T / n_valid) for valid, 0 for pad.
        // Then AVG(mask * features) = sum(valid * T/n_valid) / T = sum(valid) / n_valid. ✓
        // Caller should set mask values to T/n_valid for valid tokens, 0 for padding.
        // full-width AVG pool == row mean; CUDA has no POOL_1D kernel, MEAN it is
        auto* pooled_t = g_gpu_backend ? ggml_mean(ctx, tp)
                                       : ggml_pool_1d(ctx, tp, GGML_OP_POOL_AVG, (int)T, (int)T, 0);
        text_pooled = ggml_cont(ctx, ggml_permute(ctx, pooled_t, 1, 0, 2, 3));  // [D, 1, B]
    } else {
        // All tokens valid — simple mean
        auto* tp = ggml_cont(ctx, ggml_permute(ctx, text_mlp, 1, 0, 2, 3));
        // full-width AVG pool == row mean; CUDA has no POOL_1D kernel, MEAN it is
        auto* pooled_t = g_gpu_backend ? ggml_mean(ctx, tp)
                                       : ggml_pool_1d(ctx, tp, GGML_OP_POOL_AVG, (int)T, (int)T, 0);
        text_pooled = ggml_cont(ctx, ggml_permute(ctx, pooled_t, 1, 0, 2, 3));
    }
    ggml_set_name(text_pooled, "scoring_pooled");

    // Step 3: Project pooled prompt through prompt_proj: D→D
    auto* proj_pooled = ggml_mul_mat(ctx, tensors.at("scoring.prompt_proj.weight"), text_pooled);
    proj_pooled = ggml_add(ctx, proj_pooled, tensors.at("scoring.prompt_proj.bias"));
    // proj_pooled: [D, 1, B]
    ggml_set_name(proj_pooled, "scoring_proj_pooled");

    // Step 4: Project queries through hs_proj: D→D
    auto* proj_hs = ggml_mul_mat(ctx, tensors.at("scoring.hs_proj.weight"), query_outputs);
    proj_hs = ggml_add(ctx, proj_hs, tensors.at("scoring.hs_proj.bias"));
    // proj_hs: [D, N_q, B]
    ggml_set_name(proj_hs, "scoring_proj_hs");

    // Step 5: Dot product — for each query, dot with pooled prompt
    // matmul(proj_hs, proj_pooled.unsqueeze(-1)) in Python = batched vector-matrix multiply
    // ggml_mul_mat(A, B) = A^T @ B
    // With A = proj_pooled [D, 1, B], B = proj_hs [D, N_q, B]:
    // result = [1, N_q, B] — each element is dot product of query with pooled prompt
    auto* scores = ggml_mul_mat(ctx, proj_pooled, proj_hs);  // [1, N_q, B]

    // Step 6: Scale by 1/sqrt(D)
    float scale = 1.0f / sqrtf((float)D);
    scores = ggml_scale(ctx, scores, scale);

    // Step 7: Clamp to [-12, 12]
    scores = ggml_clamp(ctx, scores, -12.0f, 12.0f);

    // Reshape to [N_q, B]
    const int64_t N_q = query_outputs->ne[1];
    scores = ggml_reshape_2d(ctx, scores, N_q, B);
    ggml_set_name(scores, "scoring_class_scores");

    return scores;
}

// Build full DETR decoder graph.
// enc_feats: [D, N_kv, B] conditioned features from fusion encoder (N_kv=5184)
// enc_pos: [D, N_kv, B] positional encoding
// text_feats: [D, T, B] text features
// Returns struct with:
//   queries: [D, 201, B] (all query outputs including presence token)
//   pred_boxes: [4, 200, B] (cx, cy, w, h in [0,1])
//   class_scores: [200, B]
//   presence_score: [1, B]

sam3_ddec_output sam3_build_ddec_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* enc_feats,                  // [D, N_kv, B]
    struct ggml_tensor* enc_pos,                    // [D, N_kv, B]
    struct ggml_tensor* text_feats,                 // [D, T, B]
    struct ggml_tensor* sine_dim_t,                 // [1, 64] — pre-computed angle multipliers
    struct ggml_tensor* rpb_coords,                 // [feat_hw] — normalized grid coords (or nullptr)
    struct ggml_tensor* text_attn_bias,
    struct ggml_tensor* text_valid_mask)  // [T, 1, B] for scoring (or nullptr)
{
    const auto& hp = model.hparams;
    const auto& tensors = model.tensors;
    const int D = hp.neck_dim;            // 256
    const int NQ = hp.ddec_num_queries;   // 200
    const int B = (int)enc_feats->ne[2];  // batch (1)
    const int feat_hw = hp.n_img_embd();  // 72

    // ── Initialize queries from query_embed ──────────────────────────────
    auto* content = ggml_reshape_3d(ctx, model.ddec.query_embed, D, NQ, 1);
    auto* pres_tok = ggml_reshape_3d(ctx, model.ddec.presence_token, D, 1, 1);
    auto* queries = ggml_concat(ctx, pres_tok, content, 1);  // [D, NQ+1, B=1]

    // Reference points: sigmoid → initial anchor boxes
    auto* ref_pts_raw = ggml_cont(ctx, tensors.at("ddec.reference_points.weight"));  // [4, NQ]
    auto* ref_boxes = ggml_sigmoid(ctx, ref_pts_raw);                                // [4, NQ]
    ref_boxes = ggml_reshape_3d(ctx, ref_boxes, 4, NQ, 1);                           // [4, NQ, 1]
#ifndef NDEBUG
    auto* ref_boxes_dbg = ggml_cont(ctx, ref_boxes);
    ggml_set_name(ref_boxes_dbg, "ddec_ref_boxes_init");
#endif

    // ── Run decoder layers ───────────────────────────────────────────────
    // Per-layer: recompute query_pos from updated ref_boxes (matching Python exactly)
    struct ggml_tensor* last_presence = pres_tok;
    for (int i = 0; i < hp.ddec_layers; ++i) {
        // Recompute query_pos from current ref_boxes via sine embed + ref_point_head MLP
        auto* query_pos = sam3_build_query_pos(ctx, model, ref_boxes, sine_dim_t, i);

        // Compute box-relative positional bias for image cross-attention
        struct ggml_tensor* rpb_mask = nullptr;
        if (rpb_coords) {
            rpb_mask = sam3_compute_box_rpb(ctx, model, ref_boxes, rpb_coords, feat_hw, i);
        }

        queries = sam3_ddec_layer_forward(ctx, model.ddec.layers[i],
                                          queries, query_pos,
                                          enc_feats, enc_pos,
                                          text_feats, hp.ddec_heads,
                                          text_attn_bias,
                                          rpb_mask,
                                          i);

        // Box refinement after each layer (on object queries only, not presence token)
        auto* obj_q = ggml_view_3d(ctx, queries, D, NQ, 1,
                                   queries->nb[1], queries->nb[2], 1 * queries->nb[1]);
        obj_q = ggml_cont(ctx, obj_q);
        sam3_name_tensorf(obj_q, "ddec_layer%d_out", i);

        auto* pres_q = ggml_view_3d(ctx, queries, D, 1, 1,
                                    queries->nb[1], queries->nb[2], 0);
        pres_q = ggml_cont(ctx, pres_q);
        if (i == 0) {
            ggml_set_name(pres_q, "ddec_layer0_presence");
        }
        last_presence = pres_q;

        // Apply the final decoder norm before box refinement (use_normed_output_consistently)
        auto* obj_q_normed = sam3_layer_norm(ctx, obj_q,
                                             tensors.at("ddec.norm.weight"),
                                             tensors.at("ddec.norm.bias"));

        // Shared bbox_embed MLP
        auto* bd = obj_q_normed;
        for (int j = 0; j < 3; ++j) {
            auto wn = "ddec.bbox_embed.layers." + std::to_string(j) + ".weight";
            auto bn = "ddec.bbox_embed.layers." + std::to_string(j) + ".bias";
            bd = ggml_mul_mat(ctx, tensors.at(wn), bd);
            bd = ggml_add(ctx, bd, tensors.at(bn));
            if (j < 2) bd = ggml_relu(ctx, bd);
        }
        // bd: [4, NQ, 1]

        // ref_boxes = sigmoid(inverse_sigmoid(ref_boxes) + box_delta)
        auto* ref_inv_cur = sam3_inverse_sigmoid(ctx, ref_boxes);
        ref_boxes = ggml_sigmoid(ctx, ggml_add(ctx, ref_inv_cur, bd));
        sam3_name_tensorf(ref_boxes, "ddec_layer%d_refboxes", i);
    }

    // ── Final normalization ──────────────────────────────────────────────
    // Match Python: decoder.norm is applied to object queries only.
    auto* obj_queries = ggml_view_3d(ctx, queries, D, NQ, 1,
                                     queries->nb[1], queries->nb[2], 1 * queries->nb[1]);
    obj_queries = ggml_cont(ctx, obj_queries);
    obj_queries = sam3_layer_norm(ctx, obj_queries,
                                  tensors.at("ddec.norm.weight"),
                                  tensors.at("ddec.norm.bias"));
    ggml_set_name(obj_queries, "ddec_normed_output");

    auto* queries_for_seg = ggml_concat(ctx, last_presence, obj_queries, 1);

    auto* class_scores = sam3_dot_product_scoring(ctx, model, obj_queries, text_feats, text_valid_mask);
    // class_scores: [NQ, B]

    // ── Presence score ───────────────────────────────────────────────────
    // Presence token head: LN + 3-layer MLP (D→D→D→1)
    auto* pres_out = sam3_layer_norm(ctx, last_presence,
                                     tensors.at("ddec.presence_token_out_norm.weight"),
                                     tensors.at("ddec.presence_token_out_norm.bias"));

    for (int j = 0; j < 3; ++j) {
        auto wn = "ddec.presence_token_head.layers." + std::to_string(j) + ".weight";
        auto bn = "ddec.presence_token_head.layers." + std::to_string(j) + ".bias";
        pres_out = ggml_mul_mat(ctx, tensors.at(wn), pres_out);
        pres_out = ggml_add(ctx, pres_out, tensors.at(bn));
        if (j < 2) pres_out = ggml_relu(ctx, pres_out);
    }
    // Keep presence as raw logit (no sigmoid yet — applied during post-processing)
    auto* presence_score = ggml_reshape_2d(ctx, pres_out, 1, 1);
    // presence_score: [1, B] — raw logit

    sam3_ddec_output out;
    out.queries = queries_for_seg;        // [D, NQ+1, B]
    out.presence_feats = last_presence;   // [D, 1, B]
    out.pred_boxes = ref_boxes;           // [4, NQ, B]
    out.class_scores = class_scores;      // [NQ, B]
    out.presence_score = presence_score;  // [1, B]

    return out;
}

/*****************************************************************************
** Segmentation head (MaskFormer) — graph building
*****************************************************************************/

// Build pixel decoder: progressively upsample FPN features.
// fpn_feats[0]: [D, 288, 288, B] (highest res)
// fpn_feats[1]: [D, 144, 144, B]
// fpn_feats[2]: [D,  72,  72, B] (lowest res)
// Returns: [D, 288, 288, B] pixel features
//
// Python PixelDecoder.forward:
//   prev_fpn = backbone_feats[-1]  (lowest res)
//   for bb_feat in backbone_feats[:-1][::-1]:  (iterate from second-lowest to highest)
//       prev_fpn = bb_feat + F.interpolate(prev_fpn, size=bb_feat.shape[-2:], mode="nearest")
//       prev_fpn = conv_layers[i](prev_fpn)    # conv on the MERGED result
//       prev_fpn = F.relu(norms[i](prev_fpn))  # GroupNorm then ReLU
//
// Python uses GroupNorm(8, 256) — we use ggml_group_norm which normalizes ne[2]
// (the channel dim) in groups.  The conv output is [W, H, D, B] with D in ne[2].
static struct ggml_tensor* sam3_pixel_decoder(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* fpn_feats[3])  // [D, W, H, B] at 3 scales
{
    const auto& seg = model.seg_head;

    // Start from lowest resolution
    auto* feat = fpn_feats[2];  // [D, 72, 72, B]

    // Iteration 0: merge with FPN[1] (144x144)
    // prev_fpn = FPN[1] + upsample(prev_fpn)
    // Permute to [W, H, D, B] for conv operations
    auto* prev = ggml_cont(ctx, ggml_permute(ctx, feat, 2, 0, 1, 3));          // [72, 72, D, B]
    prev = ggml_upscale(ctx, prev, 2, GGML_SCALE_MODE_NEAREST);                // [144, 144, D, B]
    auto* fpn1 = ggml_cont(ctx, ggml_permute(ctx, fpn_feats[1], 2, 0, 1, 3));  // [144, 144, D, B]
    prev = ggml_add(ctx, fpn1, prev);                                          // merged
    // Conv 3x3 on the MERGED result (not individual FPN feat)
    prev = ggml_conv_2d_s1_ph(ctx, seg.up_conv_w[0], prev);
    {
        auto* b3d = ggml_reshape_3d(ctx, seg.up_conv_b[0], 1, 1, seg.up_conv_b[0]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, b3d, prev));
    }
    // GroupNorm(8, 256) then ReLU — prev is [W, H, D, B] with D in ne[2]
    prev = ggml_group_norm(ctx, prev, 8, 1e-5f);
    {
        auto* w3d = ggml_reshape_3d(ctx, seg.up_norm_w[0], 1, 1, seg.up_norm_w[0]->ne[0]);
        prev = ggml_mul(ctx, prev, ggml_repeat(ctx, w3d, prev));
        auto* bn3d = ggml_reshape_3d(ctx, seg.up_norm_b[0], 1, 1, seg.up_norm_b[0]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, bn3d, prev));
    }
    prev = ggml_relu(ctx, prev);
    ggml_set_name(prev, "seg_pixel_dec_stage0");

    // Iteration 1: merge with FPN[0] (288x288)
    prev = ggml_upscale(ctx, prev, 2, GGML_SCALE_MODE_NEAREST);                // [288, 288, D, B]
    auto* fpn0 = ggml_cont(ctx, ggml_permute(ctx, fpn_feats[0], 2, 0, 1, 3));  // [288, 288, D, B]
    prev = ggml_add(ctx, fpn0, prev);                                          // merged
    // Conv 3x3 on the MERGED result
    prev = ggml_conv_2d_s1_ph(ctx, seg.up_conv_w[1], prev);
    {
        auto* b3d = ggml_reshape_3d(ctx, seg.up_conv_b[1], 1, 1, seg.up_conv_b[1]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, b3d, prev));
    }
    // GroupNorm(8, 256) then ReLU
    prev = ggml_group_norm(ctx, prev, 8, 1e-5f);
    {
        auto* w3d = ggml_reshape_3d(ctx, seg.up_norm_w[1], 1, 1, seg.up_norm_w[1]->ne[0]);
        prev = ggml_mul(ctx, prev, ggml_repeat(ctx, w3d, prev));
        auto* bn3d = ggml_reshape_3d(ctx, seg.up_norm_b[1], 1, 1, seg.up_norm_b[1]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, bn3d, prev));
    }
    prev = ggml_relu(ctx, prev);
    ggml_set_name(prev, "seg_pixel_dec_stage1");

    // Python PixelDecoder allocates 3 conv layers but only uses 2 (one per
    // upsample step). The 3rd conv (up_conv_w[2]) is unused.

    auto* out = ggml_cont(ctx, ggml_permute(ctx, prev, 1, 2, 0, 3));  // [D, 288, 288, B]
    return out;
}

// Build the full segmentation head graph.
//
// Python UniversalSegmentationHead.forward:
//   1. Cross-attend encoder_hidden_states to prompt → updated encoder
//   2. _embed_pixels: replace lowest-res FPN feat with spatial portion of encoder output
//   3. Run pixel decoder on modified FPN feats
//   4. instance_seg_head (Conv1x1)
//   5. mask_predictor: einsum(mask_embed(queries), instance_embeds)
//
// enc_hidden: [D, N_spatial, B] — fusion encoder output (cross-attended in step 1)
// fpn_feats[3]: the 3 FPN features at different resolutions
// query_outputs: [D, N, B] selected object query outputs
// text_features: [D, T, B] for cross-attention (prompt)
// Returns: mask_logits [W*H, N, B] (raw logits, not sigmoid)
struct ggml_tensor* sam3_build_seg_head_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* enc_hidden,     // [D, N_spatial, B] fusion encoder output
    struct ggml_tensor* fpn_feats[3],   // FPN features at 3 scales
    struct ggml_tensor* query_outputs,  // [D, N, B]
    struct ggml_tensor* text_features,  // [D, T, B] (for cross-attn, can be nullptr)
    struct ggml_tensor* text_attn_bias) {
    const auto& seg = model.seg_head;
    const auto& tensors = model.tensors;
    const int64_t D = enc_hidden->ne[0];     // 256
    const int64_t B = enc_hidden->ne[2];     // 1
    const int64_t N = query_outputs->ne[1];  // number of selected queries

    auto* enc = enc_hidden;
    if (text_features) {
        auto* ca_norm = sam3_layer_norm(ctx, enc,
                                        tensors.at("seg.cross_attn_norm.weight"),
                                        tensors.at("seg.cross_attn_norm.bias"));

        auto* ca_mask = sam3_expand_token_attn_bias(ctx, text_attn_bias, enc->ne[1], 8, B);
        auto* ca_out = sam3_multihead_attn_fused(ctx, ca_norm, text_features,
                                                 seg.ca_prompt_q_w, seg.ca_prompt_q_b,
                                                 seg.ca_prompt_out_w, seg.ca_prompt_out_b,
                                                 8, ca_mask);
        enc = ggml_add(ctx, enc, ca_out);
    }
    // enc: [D, N_spatial, B]
    ggml_set_name(enc, "seg_enc_after_ca");

    // Replace lowest-res FPN feat with spatial portion of encoder output
    const int64_t feat_hw = model.hparams.n_img_embd();  // 72
    auto* enc_spatial = ggml_reshape_4d(ctx, enc, D, feat_hw, feat_hw, B);
#ifndef NDEBUG
    auto* enc_spatial_dbg = ggml_cont(ctx, ggml_permute(ctx, enc_spatial, 2, 0, 1, 3));
    ggml_set_name(enc_spatial_dbg, "seg_enc_visual");
#endif

    struct ggml_tensor* modified_fpn[3] = {
        fpn_feats[0],
        fpn_feats[1],
        enc_spatial,  // replaces original lowest-res FPN
    };

    auto* pixel_feats = sam3_pixel_decoder(ctx, model, modified_fpn);
#ifndef NDEBUG
    auto* pixel_feats_dbg = ggml_cont(ctx, ggml_permute(ctx, pixel_feats, 2, 0, 1, 3));
    ggml_set_name(pixel_feats_dbg, "seg_pixel_decoder_out");
#endif

    const int64_t W = pixel_feats->ne[1];  // 288
    const int64_t H = pixel_feats->ne[2];  // 288

    // Instance segmentation head (Conv1x1)
    auto* pf_conv = ggml_cont(ctx, ggml_permute(ctx, pixel_feats, 2, 0, 1, 3));
    pf_conv = ggml_conv_2d_sk_p0(ctx, tensors.at("seg.instance_seg_head.weight"), pf_conv);
    {
        auto* b3d = ggml_reshape_3d(ctx, tensors.at("seg.instance_seg_head.bias"),
                                    1, 1, tensors.at("seg.instance_seg_head.bias")->ne[0]);
        pf_conv = ggml_add(ctx, pf_conv, ggml_repeat(ctx, b3d, pf_conv));
    }
    auto* pixel_embed = ggml_cont(ctx, ggml_permute(ctx, pf_conv, 1, 2, 0, 3));  // [D, W, H, B]
#ifndef NDEBUG
    auto* pixel_embed_dbg = ggml_cont(ctx, ggml_permute(ctx, pixel_embed, 2, 0, 1, 3));
    ggml_set_name(pixel_embed_dbg, "seg_instance_embed");
#endif

    // Mask embedding MLP
    auto* mask_embed = query_outputs;
    for (int j = 0; j < 3; ++j) {
        auto wn = "seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".weight";
        auto bn = "seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".bias";
        mask_embed = ggml_mul_mat(ctx, tensors.at(wn), mask_embed);
        mask_embed = ggml_add(ctx, mask_embed, tensors.at(bn));
        if (j < 2) mask_embed = ggml_relu(ctx, mask_embed);
    }
    ggml_set_name(mask_embed, "seg_mask_embed");

    // Mask prediction: einsum('bqc,bchw->bqhw')
    auto* pe_flat = ggml_reshape_3d(ctx, pixel_embed, D, W * H, B);
    auto* masks = ggml_mul_mat(ctx, pe_flat, mask_embed);
    ggml_set_name(masks, "seg_mask_logits");

    return masks;
}

