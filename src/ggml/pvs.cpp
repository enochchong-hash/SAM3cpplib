// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** SAM attention helper (separate Q, K, V weight/bias)
*****************************************************************************/

static struct ggml_tensor* sam3_sam_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q_in,  // [D, N_q, B]
    struct ggml_tensor* k_in,  // [D, N_kv, B]
    struct ggml_tensor* v_in,  // [D, N_kv, B]
    const sam3_sam_attn& attn,
    int n_heads) {
    const int64_t N_q = q_in->ne[1];
    const int64_t B = q_in->ne[2];
    const int64_t N_kv = k_in->ne[1];

    // Project
    auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, attn.q_w, q_in), attn.q_b);
    auto* K = ggml_add(ctx, ggml_mul_mat(ctx, attn.k_w, k_in), attn.k_b);
    auto* V = ggml_add(ctx, ggml_mul_mat(ctx, attn.v_w, v_in), attn.v_b);

    // Debug: mark projections for the first SA call (N_q=8 tokens, block 0)
    static int _sa_call_count = 0;
    if (_sa_call_count == 0 && N_q <= 16) {
        ggml_set_name(Q, "dbg_sa0_Q_proj");
        ggml_set_output(Q);
        ggml_set_name(V, "dbg_sa0_V_proj");
        ggml_set_output(V);
    }
    _sa_call_count++;

    // internal_dim = out_proj cols = attn.q_w->ne[1]
    const int64_t ID = attn.q_w->ne[1];
    const int64_t HD = ID / n_heads;

    // Reshape to multi-head: [HD, N, NH, B]
    Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [HD, N_q, NH, B]

    K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));  // [HD, N_kv, NH, B] contiguous

    // Attention
    float scale = 1.0f / sqrtf((float)HD);
    auto* out = sam3_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
    // out: [HD, NH, N_q, B] (flash_attn_ext swaps dims 1,2 vs input)

#if 0  // Manual SDPA (for debugging only)
    auto* Q3 = ggml_reshape_3d(ctx, Q, HD, N_q, n_heads * B);
    auto* K3 = ggml_reshape_3d(ctx, K, HD, N_kv, n_heads * B);
    auto* V3 = ggml_reshape_3d(ctx, V, HD, N_kv, n_heads * B);
    // QK^T: ggml_mul_mat(K, Q) → K^T @ Q → [N_kv, N_q, NH*B]
    auto* attn_scores = ggml_mul_mat(ctx, K3, Q3);
    attn_scores = ggml_scale(ctx, attn_scores, scale);
    attn_scores = ggml_soft_max(ctx, attn_scores);

    // attn @ V: need attn^T [N_q, N_kv] and V^T [HD, N_kv]
    // ggml_mul_mat(attn^T, V) = (attn^T)^T @ V = attn @ V = [N_q, HD]... no.
    // ggml_mul_mat(A, B) = A^T @ B where A=[K, M], B=[K, N] → [M, N]
    // Want: output[q, d] = sum_k attn[q, k] * V[k, d]
    // = (V^T @ attn^T)^T... let me think differently.
    // attn_scores is [N_kv, N_q, NH*B]. For each head:
    //   attn[k, q] = attn_scores[k, q]  (col q has the weights for query q)
    // V3 is [HD, N_kv, NH*B].
    // Want: out[d, q] = sum_k V[d, k] * attn[k, q] = V @ attn
    // = ggml_mul_mat? mul_mat(A, B) = A^T B with A=[K, M], B=[K, N] → [M, N]
    // V has ne=[HD, N_kv, ...]. attn has ne=[N_kv, N_q, ...].
    // If A=V3 (ne0=HD, ne1=N_kv) and B=attn_scores (ne0=N_kv, ne1=N_q):
    // Shared dim ne0: V3 ne0=HD ≠ attn ne0=N_kv. Mismatch!
    //
    // Need to transpose V: V^T is [N_kv, HD]. Then A=V^T, B=attn_scores.
    // A ne0=N_kv, B ne0=N_kv → shared. A^T B = V @ attn → [HD, N_q]. ✓
    auto* VT = ggml_permute(ctx, V3, 1, 0, 2, 3);  // [N_kv, HD, NH*B]
    VT = ggml_cont(ctx, VT);
    auto* out3 = ggml_mul_mat(ctx, VT, attn_scores);  // [HD, N_q, NH*B]

    // Reshape back to 4D: [HD, N_q, NH, B]
    auto* out = ggml_reshape_4d(ctx, out3, HD, N_q, n_heads, B);
    // Permute to [HD, NH, N_q, B] to match flash_attn_ext output convention
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
#endif

    // Merge heads: [ID=HD*NH, N_q, B]
    auto* merged = ggml_reshape_3d(ctx, out, ID, N_q, B);

    // Debug: mark merged attention output for first SA call
    static int _sa_merge_count = 0;
    if (_sa_merge_count == 0 && N_q <= 16) {
        ggml_set_name(merged, "dbg_sa0_merged");
        ggml_set_output(merged);
    }
    _sa_merge_count++;

    // Output projection
    out = ggml_mul_mat(ctx, attn.out_w, merged);
    out = ggml_add(ctx, out, attn.out_b);

    return out;
}

/*****************************************************************************
** SAM prompt encoder — graph building (Phase 6, Step 6.1)
*****************************************************************************/

// Random Fourier positional encoding for a single (x, y) coordinate
// coords_norm: normalized to [0, 1], pe_gaussian: PyTorch row-major [2, 128]
// Output: [256] = [sin(128); cos(128)]
void sam3_pe_encode_coord(float* out, float x_norm, float y_norm,
                                 const float* pe_gauss, int num_pos_feats) {
    // Map [0,1] → [-1,1]
    float coords[2] = {2.0f * x_norm - 1.0f, 2.0f * y_norm - 1.0f};

    // coords @ pe_gaussian → [128]
    for (int i = 0; i < num_pos_feats; ++i) {
        float dot = coords[0] * pe_gauss[i] +
                    coords[1] * pe_gauss[num_pos_feats + i];
        dot *= 2.0f * (float)M_PI;
        out[i] = sinf(dot);
        out[i + num_pos_feats] = cosf(dot);
    }
}

// Read SAM prompt encoder weights from GPU and cache them in state.
// Also pre-computes the dense PE grid and no-mask tiled embedding.
// These never change between PVS calls for the same model.
void sam3_populate_pe_cache(sam3_state& state, const sam3_model& model) {
    if (state.pe_cache_valid) return;

    const int D = model.hparams.sam_embed_dim;  // 256
    const int H = sam3_eff_feat_size(state, model.hparams);
    const int num_pos_feats = D / 2;            // 128
    const int pe_nel = 2 * num_pos_feats;       // 256
    const auto& pe = model.sam_pe;

    state.pe_gauss_cache.resize(pe_nel);
    if (pe.pe_gaussian->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(pe_nel);
        ggml_backend_tensor_get(pe.pe_gaussian, tmp.data(), 0, pe_nel * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), state.pe_gauss_cache.data(), pe_nel);
    } else {
        ggml_backend_tensor_get(pe.pe_gaussian, state.pe_gauss_cache.data(), 0, pe_nel * sizeof(float));
    }

    for (int i = 0; i < 4; ++i) {
        if (pe.point_embed[i]->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(D);
            ggml_backend_tensor_get(pe.point_embed[i], tmp.data(), 0, D * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(tmp.data(), state.point_emb_cache[i], D);
        } else {
            ggml_backend_tensor_get(pe.point_embed[i], state.point_emb_cache[i], 0, D * sizeof(float));
        }
    }

    if (pe.not_a_point_embed->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(D);
        ggml_backend_tensor_get(pe.not_a_point_embed, tmp.data(), 0, D * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), state.not_a_point_cache, D);
    } else {
        ggml_backend_tensor_get(pe.not_a_point_embed, state.not_a_point_cache, 0, D * sizeof(float));
    }

    if (pe.no_mask_embed->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(D);
        ggml_backend_tensor_get(pe.no_mask_embed, tmp.data(), 0, D * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), state.no_mask_emb_cache, D);
    } else {
        ggml_backend_tensor_get(pe.no_mask_embed, state.no_mask_emb_cache, 0, D * sizeof(float));
    }

    state.dense_pe_cache.resize(D * H * H);
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < H; ++col) {
            float x_norm = ((float)col + 0.5f) / (float)H;
            float y_norm = ((float)row + 0.5f) / (float)H;
            float pe_vec[256];
            sam3_pe_encode_coord(pe_vec, x_norm, y_norm,
                                 state.pe_gauss_cache.data(), num_pos_feats);
            for (int d = 0; d < D; ++d)
                state.dense_pe_cache[d + col * D + row * D * H] = pe_vec[d];
        }
    }

    state.dense_nomask_cache.resize(D * H * H);
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < H; ++col) {
            for (int d = 0; d < D; ++d)
                state.dense_nomask_cache[d + col * D + row * D * H] = state.no_mask_emb_cache[d];
        }
    }

    state.pe_cache_valid = true;
    state.pe_cache_feat_size = H;
    SAM3_LOG(2, "%s: PE cache populated (%d embeddings, %.1f KB dense grids)\n",
             __func__, pe_nel, 2.0f * D * H * H * sizeof(float) / 1024.0f);
}

// Match the official image predictor prompt ordering:
//   box corners (if any) → user points → trailing padding point.
// Even when a box is present, SAM keeps the final padding token because boxes
// are merged into the point stream before calling PromptEncoder(points=..., boxes=None).
void sam3_collect_pvs_prompt_tokens(const sam3_pvs_params& params,
                                           std::vector<float>& all_coords,
                                           std::vector<int>& all_labels) {
    all_coords.clear();
    all_labels.clear();

    if (params.use_box) {
        all_coords.push_back(params.box.x0);
        all_coords.push_back(params.box.y0);
        all_labels.push_back(2);

        all_coords.push_back(params.box.x1);
        all_coords.push_back(params.box.y1);
        all_labels.push_back(3);
    }

    for (const auto& pt : params.pos_points) {
        all_coords.push_back(pt.x);
        all_coords.push_back(pt.y);
        all_labels.push_back(1);
    }
    for (const auto& pt : params.neg_points) {
        all_coords.push_back(pt.x);
        all_coords.push_back(pt.y);
        all_labels.push_back(0);
    }

    all_coords.push_back(0.0f);
    all_coords.push_back(0.0f);
    all_labels.push_back(-1);
}

// Build sparse and dense embeddings from point/box prompts
// sparse_out: [D, N_pts, 1] where N_pts = n_pos + n_neg + (use_box ? 2 : 0) + pad
// dense_out:  [D, H, H, 1] (no-mask default or mask downsample)

sam3_pe_result sam3_build_sam_pe(
    struct ggml_context* ctx,
    const sam3_pvs_params& params,
    int embed_dim, int feat_size) {
    const int D = embed_dim;  // 256
    const int H = feat_size;  // 72

    int N_pts = (int)(params.pos_points.size() + params.neg_points.size()) + 1;  // pad
    if (params.use_box) N_pts += 2;

    auto* sparse = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_pts, 1);
    ggml_set_name(sparse, "sam_pe_sparse");
    ggml_set_input(sparse);

    auto* image_pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, H, 1);
    ggml_set_name(image_pe, "sam_pe_image_pe");
    ggml_set_input(image_pe);

    auto* dense = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, H, 1);
    ggml_set_name(dense, "sam_pe_dense");
    ggml_set_input(dense);

    sam3_pe_result result;
    result.sparse = sparse;
    result.dense = dense;
    result.image_pe = image_pe;
    result.n_tokens = N_pts;
    return result;
}

/*****************************************************************************
** SAM mask decoder — graph building (Phase 6, Step 6.2)
*****************************************************************************/

// TwoWayAttentionBlock forward
static void sam3_twoway_block_forward(
    struct ggml_context* ctx,
    struct ggml_tensor*& queries,  // [D, N_q, B] — modified in place
    struct ggml_tensor*& keys,     // [D, N_kv, B] — modified in place
    struct ggml_tensor* query_pe,  // [D, N_q, B]
    struct ggml_tensor* key_pe,    // [D, N_kv, B]
    const sam3_twoway_block& blk,
    int n_heads,
    bool skip_first_layer_pe) {
    // 1. Self-attention on queries
    if (skip_first_layer_pe) {
        // Python: queries = self.self_attn(q=queries, k=queries, v=queries)
        // No residual connection when skipping first layer PE
        queries = sam3_sam_attention(ctx, queries, queries, queries, blk.self_attn, n_heads);
    } else {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, q, queries, blk.self_attn, n_heads);
        queries = ggml_add(ctx, queries, attn_out);
    }
    queries = sam3_layer_norm(ctx, queries, blk.norm1_w, blk.norm1_b);
    if (skip_first_layer_pe) {
        ggml_set_name(queries, "dbg_twoway_skip_sa_norm");
        ggml_set_output(queries);
    }

    // 2. Cross-attention: tokens attending to image
    {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* k = ggml_add(ctx, keys, key_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, k, keys, blk.ca_tok2img, n_heads);
        queries = ggml_add(ctx, queries, attn_out);
        queries = sam3_layer_norm(ctx, queries, blk.norm2_w, blk.norm2_b);
    }
    if (skip_first_layer_pe) {
        ggml_set_name(queries, "dbg_twoway_skip_ca_tok2img");
        ggml_set_output(queries);
    }

    // 3. MLP on queries (ReLU activation)
    {
        auto* mlp = ggml_mul_mat(ctx, blk.mlp_fc1_w, queries);
        mlp = ggml_add(ctx, mlp, blk.mlp_fc1_b);
        mlp = ggml_relu(ctx, mlp);
        mlp = ggml_mul_mat(ctx, blk.mlp_fc2_w, mlp);
        mlp = ggml_add(ctx, mlp, blk.mlp_fc2_b);
        queries = ggml_add(ctx, queries, mlp);
        queries = sam3_layer_norm(ctx, queries, blk.norm3_w, blk.norm3_b);
    }
    if (skip_first_layer_pe) {
        ggml_set_name(queries, "dbg_twoway_skip_mlp");
        ggml_set_output(queries);
    }

    // 4. Cross-attention: image attending to tokens
    {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* k = ggml_add(ctx, keys, key_pe);
        // Note: q and k are swapped — image (k) attends to tokens (q)
        auto* attn_out = sam3_sam_attention(ctx, k, q, queries, blk.ca_img2tok, n_heads);
        keys = ggml_add(ctx, keys, attn_out);
        keys = sam3_layer_norm(ctx, keys, blk.norm4_w, blk.norm4_b);
    }
    if (skip_first_layer_pe) {
        ggml_set_name(keys, "dbg_twoway_skip_img2tok");
        ggml_set_output(keys);
    }
}

// MLP forward: N layers with ReLU (except last), optional sigmoid on last
static struct ggml_tensor* sam3_mlp_forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* const* weights,
    struct ggml_tensor* const* biases,
    int n_layers,
    bool sigmoid_output = false) {
    for (int i = 0; i < n_layers; ++i) {
        x = ggml_mul_mat(ctx, weights[i], x);
        x = ggml_add(ctx, x, biases[i]);
        if (i < n_layers - 1) {
            x = ggml_relu(ctx, x);
        }
    }
    if (sigmoid_output) {
        x = ggml_sigmoid(ctx, x);
    }
    return x;
}

// Full SAM mask decoder graph
// Inputs:
//   image_feats:  [D, H, H, 1] — tracker neck features (scale 2 = 72×72)
//   image_pe:     [D, H, H, 1] — dense positional encoding
//   sparse_emb:   [D, N_pts, 1] — sparse prompt embeddings
//   dense_emb:    [D, H, H, 1] — dense prompt embeddings (no_mask default)
//   feat_s0:      [D, H0, H0, 1] — high-res features (scale 0 = 288×288)
//   feat_s1:      [D, H1, H1, 1] — mid-res features (scale 1 = 144×144)
// Outputs: sam3_dec_result with masks, iou_pred, obj_score, sam_token_out

sam3_dec_result sam3_build_sam_dec_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* image_feats,  // [D, H, H, 1]
    struct ggml_tensor* image_pe,     // [D, H, H, 1]
    struct ggml_tensor* sparse_emb,   // [D, N_pts, 1]
    struct ggml_tensor* dense_emb,    // [D, H, H, 1]
    struct ggml_tensor* feat_s0,      // [D, H*4, H*4, 1] high-res
    struct ggml_tensor* feat_s1,     // [D, H*2, H*2, 1] mid-res
    int eff_feat_size)
{
    const auto& dec = model.sam_dec;
    const auto& hp = model.hparams;
    const int D = hp.sam_embed_dim;  // 256
    const int H = (eff_feat_size > 0) ? eff_feat_size : hp.feat_size();
    const int N_pts = (int)sparse_emb->ne[1];
    const int n_heads = 8;                               // SAM uses 8 heads
    const int num_mask_tokens = hp.sam_n_multimask + 1;  // 4

    // ── Concatenate output tokens ────────────────────────────────────────
    // When pred_obj_scores=True:  [obj_score(1,D), iou(1,D), masks(4,D)] = 6 tokens
    // When pred_obj_scores=False: [iou(1,D), masks(4,D)] = 5 tokens (older SAM2)
    const bool has_obj_score = (dec.obj_score_token != nullptr);
    const int n_special = (has_obj_score ? 6 : 5);

    struct ggml_tensor* output_tokens;
    if (has_obj_score) {
        output_tokens = ggml_concat(ctx, dec.obj_score_token, dec.iou_token, 1);
    } else {
        output_tokens = ggml_reshape_2d(ctx, dec.iou_token, D, 1);
    }
    output_tokens = ggml_concat(ctx, output_tokens, dec.mask_tokens, 1);
    output_tokens = ggml_reshape_3d(ctx, output_tokens, D, n_special, 1);
    auto* tokens = ggml_concat(ctx, output_tokens, sparse_emb, 1);
    ggml_set_name(tokens, "sam_dec_tokens_initial");
    ggml_set_output(tokens);

    const int N_tok = 6 + N_pts;

    auto* src = ggml_add(ctx, image_feats, dense_emb);
    src = ggml_reshape_3d(ctx, src, D, H * H, 1);
    auto* pos_src = ggml_reshape_3d(ctx, image_pe, D, H * H, 1);

    auto* queries = tokens;
    auto* keys = src;
    auto* query_pe = tokens;  // query PE = initial point embedding
    auto* key_pe = pos_src;

    for (int i = 0; i < hp.sam_dec_depth; ++i) {
        sam3_twoway_block_forward(ctx, queries, keys, query_pe, key_pe,
                                  dec.twoway_blocks[i], n_heads,
                                  /*skip_first_layer_pe=*/(i == 0));
        sam3_name_tensorf(queries, "sam_dec_block%d_queries", i);
        ggml_set_output(queries);
        sam3_name_tensorf(keys, "sam_dec_block%d_keys", i);
        ggml_set_output(keys);
    }

    // Final attention: tokens → image
    {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* k = ggml_add(ctx, keys, key_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, k, keys, dec.final_attn, n_heads);
        queries = ggml_add(ctx, queries, attn_out);
        queries = sam3_layer_norm(ctx, queries, dec.final_norm_w, dec.final_norm_b);
        ggml_set_name(queries, "sam_dec_final_queries");
    }

    // Debug: mark transformer outputs
    ggml_set_name(queries, "dbg_dec_queries_out");
    ggml_set_output(queries);
    ggml_set_name(keys, "dbg_dec_keys_out");
    ggml_set_output(keys);

    // ── Extract output tokens ────────────────────────────────────────────
    // With pred_obj_scores=True (6 tokens):  obj(0), iou(1), masks(2..5)
    // With pred_obj_scores=False (5 tokens): iou(0), masks(1..4)
    const int s = has_obj_score ? 1 : 0;
    auto* iou_token_out = ggml_view_3d(ctx, queries, D, 1, 1,
                                       queries->nb[1], queries->nb[2],
                                       s * queries->nb[1]);
    iou_token_out = ggml_cont(ctx, iou_token_out);  // [D, 1, 1]

    auto* mask_tokens_out = ggml_view_3d(ctx, queries, D, num_mask_tokens, 1,
                                         queries->nb[1], queries->nb[2],
                                         (s + 1) * queries->nb[1]);
    mask_tokens_out = ggml_cont(ctx, mask_tokens_out);  // [D, 4, 1]
    ggml_set_name(mask_tokens_out, "sam_dec_mask_tokens");

    struct ggml_tensor* obj_in = nullptr;
    if (has_obj_score) {
        obj_in = ggml_view_3d(ctx, queries, D, 1, 1,
                              queries->nb[1], queries->nb[2], 0);
        obj_in = ggml_cont(ctx, obj_in);  // [D, 1, 1]
    }

    // SAM output token = first mask token, used for object pointer
    auto* sam_token = ggml_view_2d(ctx, queries, D, 1,
                                   queries->nb[1], (s + 1) * queries->nb[1]);
    sam_token = ggml_cont(ctx, sam_token);  // [D, 1]
    ggml_set_name(sam_token, "sam_dec_sam_token");

    // Upscale: [D, H*H, 1] → ConvTranspose → high-res masks
    auto* src_img = ggml_reshape_4d(ctx, keys, D, H, H, 1);
    src_img = ggml_cont(ctx, ggml_permute(ctx, src_img, 2, 0, 1, 3));

    auto* up1 = sam3_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, dec.up1_w), src_img, 2);
    up1 = ggml_add(ctx, up1, ggml_reshape_4d(ctx, dec.up1_b, 1, 1, ggml_nelements(dec.up1_b), 1));

    auto* fs1 = ggml_cont(ctx, ggml_permute(ctx, feat_s1, 2, 0, 1, 3));
    auto* hs1 = ggml_conv_2d_sk_p0(ctx, dec.conv_s1_w, fs1);             // [W, H, 64, B]
    hs1 = ggml_add(ctx, hs1, ggml_reshape_4d(ctx, dec.conv_s1_b, 1, 1, 64, 1));
    ggml_set_name(hs1, "sam_dec_feat_s1_proj");

    // Python: act1(ln1(dc1(src) + feat_s1)) — add before LayerNorm
    up1 = ggml_add(ctx, up1, hs1);
    up1 = ggml_cont(ctx, ggml_permute(ctx, up1, 1, 2, 0, 3));
    up1 = sam3_layer_norm_2d(ctx, up1, dec.up1_norm_w, dec.up1_norm_b);

    up1 = ggml_gelu_erf(ctx, up1);

    // Permute back to [W, H, C, B] for next deconv
    up1 = ggml_cont(ctx, ggml_permute(ctx, up1, 2, 0, 1, 3));  // [144, 144, 64, 1]

    // dc2: ConvTranspose2d(64, 32, k=2, s=2) → [288, 288, 32, 1]
    auto* up2 = sam3_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, dec.up2_w), up1, 2);
    up2 = ggml_add(ctx, up2, ggml_reshape_4d(ctx, dec.up2_b, 1, 1, ggml_nelements(dec.up2_b), 1));

    // conv_s0: 1x1 conv on feat_s0 (256→32). feat_s0 is [C, W, H, B] — permute for conv.
    auto* fs0 = ggml_cont(ctx, ggml_permute(ctx, feat_s0, 2, 0, 1, 3));  // [W, H, C, B]
    auto* hs0 = ggml_conv_2d_sk_p0(ctx, dec.conv_s0_w, fs0);             // [W, H, 32, B]
    hs0 = ggml_add(ctx, hs0, ggml_reshape_4d(ctx, dec.conv_s0_b, 1, 1, 32, 1));
    ggml_set_name(hs0, "sam_dec_feat_s0_proj");

    // Python: act2(dc2(upscaled_embedding) + feat_s0) — no LayerNorm here
    up2 = ggml_add(ctx, up2, hs0);  // both [W, H, 32, B]

    // Permute to [C, W, H, B] for subsequent operations
    up2 = ggml_cont(ctx, ggml_permute(ctx, up2, 1, 2, 0, 3));  // [32, 288, 288, 1]

    // GELU activation (exact, matching Python nn.GELU)
    up2 = ggml_gelu_erf(ctx, up2);

    // up2: [32, 288, 288, 1] — this is our upscaled_embedding
    ggml_set_name(up2, "sam_dec_upscaled");

    // ── Hypernetwork: predict masks ──────────────────────────────────────
    // For each mask token i, pass through 3-layer MLP to get [32] vector
    // Then dot product with upscaled_embedding [32, (H*4)^2] to get mask
    const int H4 = H * 4;
    auto* up_flat = ggml_reshape_3d(ctx, up2, 32, H4 * H4, 1);

    // Process each mask token through its hypernetwork MLP
    // mask_tokens_out: [D, 4, 1]
    struct ggml_tensor* mask_list[4];
    for (int m = 0; m < num_mask_tokens; ++m) {
        // Extract token m: [D, 1, 1]
        auto* tok = ggml_view_3d(ctx, mask_tokens_out, D, 1, 1,
                                 mask_tokens_out->nb[1], mask_tokens_out->nb[2],
                                 m * mask_tokens_out->nb[1]);
        tok = ggml_cont(ctx, tok);  // [D, 1, 1]

        // MLP: 3 layers, 256→256→256→32, ReLU on first two
        auto* hyper = sam3_mlp_forward(ctx, tok,
                                       dec.hyper_w[m], dec.hyper_b[m], 3);
        // hyper: [32, 1, 1]

        // Dot product: hyper^T @ up_flat → [1, 288*288, 1]
        // Use mul_mat: up_flat^T [288*288, 32] @ hyper [32, 1] → [288*288, 1, 1]
        auto* mask = ggml_mul_mat(ctx, up_flat, hyper);  // [288*288, 1, 1]
        mask_list[m] = mask;
    }

    // Stack masks: [288*288, 4, 1]
    auto* masks = mask_list[0];
    for (int m = 1; m < num_mask_tokens; ++m) {
        masks = ggml_concat(ctx, masks, mask_list[m], 1);
    }
    ggml_set_name(masks, "sam_dec_masks");

    // ── IoU prediction ───────────────────────────────────────────────────
    // iou_token_out: [D, 1, 1]
    auto* iou_pred = sam3_mlp_forward(ctx, iou_token_out,
                                      dec.iou_head_w, dec.iou_head_b, 3,
                                      /*sigmoid_output=*/true);
    // iou_pred: [4, 1, 1] → reshape to [4, 1]
    iou_pred = ggml_reshape_2d(ctx, iou_pred, num_mask_tokens, 1);
    ggml_set_name(iou_pred, "sam_dec_iou");

    // ── Object score ─────────────────────────────────────────────────────
    struct ggml_tensor* obj_score;
    if (has_obj_score) {
        // obj_in: [D, 1, 1] → MLP → [1, 1, 1]
        obj_score = sam3_mlp_forward(ctx, obj_in,
                                     dec.obj_head_w, dec.obj_head_b, 3);
        obj_score = ggml_reshape_2d(ctx, obj_score, 1, 1);
    } else {
        // No obj_score prediction — return raw logit 10.0 (sigmoid ≈ 1.0, object always present)
        obj_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
        ggml_set_name(obj_score, "sam_dec_obj_score");
        ggml_set_input(obj_score);
        // Mark that callers must set this to 10.0f before compute
    }

    sam3_dec_result res;
    res.masks = masks;
    res.iou_pred = iou_pred;
    res.obj_score = obj_score;
    res.sam_token = sam_token;
    res.mask_tokens = mask_tokens_out;
    return res;
}

