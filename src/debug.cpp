// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

bool sam3_encode_vit_from_preprocessed_selective(sam3_state                    & state,
                                                 const sam3_model              & model,
                                                 const float                   * chw_data,
                                                 int                             img_size,
                                                 const std::vector<std::string> & output_tensors) {
    const auto & hp = model.hparams;

    if (img_size != hp.img_size) {
        fprintf(stderr, "%s: img_size mismatch: got %d, expected %d\n",
                __func__, img_size, hp.img_size);
        return false;
    }

    state.orig_width = img_size;
    state.orig_height = img_size;

    const size_t buf_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    auto * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img_size, img_size, 3, 1);
    ggml_set_name(inp, "input_image");
    ggml_set_input(inp);

    auto * vit_out = sam3_build_vit_graph(ctx0, inp, model);
    ggml_set_name(vit_out, "vit_output");
    ggml_set_output(vit_out);

    if (!sam3_mark_named_outputs(ctx0, output_tensors)) {
        ggml_free(ctx0);
        return false;
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(graph, vit_out);

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph)) {
        fprintf(stderr, "%s: failed to reserve graph memory\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    const size_t data_bytes = (size_t) 3 * img_size * img_size * sizeof(float);
    ggml_backend_tensor_set(inp, chw_data, 0, data_bytes);
    if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    if (state.galloc) {
        ggml_gallocr_free(state.galloc);
    }
    if (state.ctx) {
        ggml_free(state.ctx);
    }
    if (state.pe_buf) {
        ggml_backend_buffer_free(state.pe_buf);
        state.pe_buf = nullptr;
    }
    if (state.pe_ctx) {
        ggml_free(state.pe_ctx);
        state.pe_ctx = nullptr;
    }
    // This path (SAM2/EdgeTAM dispatch) doesn't populate the neck PE cache at
    // all, but frees it if a prior sam3_encode_image call had built one --
    // invalidate defensively so a later sam3_encode_image call never trusts a
    // stale neck_pe_valid flag pointing at now-freed pe_ctx/pe_buf.
    state.neck_pe_valid = false;

    state.ctx = ctx0;
    state.galloc = galloc;
    state.backend = model.backend;
    sam3_clear_encoder_state(state);
    state.vit_output = vit_out;
    if (sam3_eff_feat_size(state, hp) != state.pe_cache_feat_size) {
        state.pe_cache_valid = false;
    }

    return true;
}

static void sam3_normalize_ne4(const int64_t input_ne[4], int64_t ne[4]);
static struct ggml_tensor * sam3_new_f32_tensor_4d_from_ne(struct ggml_context * ctx,
                                                           const int64_t ne[4]);
bool sam3_copy_tensor_to_f32(struct ggml_tensor * t, std::vector<float> & output);
static struct ggml_tensor * sam3_build_vit_prefix_stage_from_input(struct ggml_context     * ctx,
                                                                   struct ggml_tensor      * input,
                                                                   const sam3_model        & model,
                                                                   sam3_vit_prefix_stage     stage);

bool sam3_test_run_vit_block0_input(const sam3_model   & model,
                                    const float        * chw_data,
                                    int                  img_size,
                                    std::vector<float> & output_data,
                                    int64_t              output_ne[4],
                                    int                  n_threads) {
    if (!chw_data) {
        fprintf(stderr, "%s: chw_data is null\n", __func__);
        return false;
    }
    if (img_size != model.hparams.img_size) {
        fprintf(stderr, "%s: img_size mismatch: got %d, expected %d\n",
                __func__, img_size, model.hparams.img_size);
        return false;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to create ggml context\n", __func__);
        return false;
    }

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, img_size, img_size, 3, 1);
    ggml_set_name(input, "vit_prefix_input");
    ggml_set_input(input);

    struct ggml_tensor * output = sam3_build_vit_prefix_graph(ctx, input, model);
    ggml_set_name(output, "vit_block0_input");
    ggml_set_output(output);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, output);

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    const size_t input_bytes = (size_t) 3 * img_size * img_size * sizeof(float);
    ggml_backend_tensor_set(input, chw_data, 0, input_bytes);
    sam3_graph_compute(model.backend, graph, n_threads);

    for (int i = 0; i < 4; ++i) {
        output_ne[i] = output->ne[i];
    }

    const bool ok = sam3_copy_tensor_to_f32(output, output_data);
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return ok;
}

bool sam3_test_run_vit_prefix_stage(const sam3_model         & model,
                                    sam3_vit_prefix_stage      stage,
                                    const float              * input_data,
                                    const int64_t              input_ne[4],
                                    std::vector<float>       & output_data,
                                    int64_t                    output_ne[4],
                                    int                        n_threads) {
    if (!input_data) {
        fprintf(stderr, "%s: input_data is null\n", __func__);
        return false;
    }

    int64_t ne[4];
    sam3_normalize_ne4(input_ne, ne);

    switch (stage) {
        case SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL:
        case SAM3_VIT_PREFIX_STAGE_PATCH_EMBED:
            if (ne[0] != model.hparams.img_size || ne[1] != model.hparams.img_size || ne[2] != 3) {
                fprintf(stderr, "%s: patch_embed input shape mismatch [%lld,%lld,%lld,%lld]\n",
                        __func__,
                        (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
                return false;
            }
            break;

        case SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW:
        case SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT:
            if (ne[0] != model.vit.patch_embed_w->ne[0] * model.vit.patch_embed_w->ne[1] * model.vit.patch_embed_w->ne[2] ||
                ne[1] != model.hparams.n_img_embd() ||
                ne[2] != model.hparams.n_img_embd()) {
                fprintf(stderr, "%s: patch_mulmat input shape mismatch [%lld,%lld,%lld,%lld]\n",
                        __func__,
                        (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
                return false;
            }
            break;

        default:
            if (ne[0] != model.hparams.vit_embed_dim ||
                ne[1] != model.hparams.n_img_embd() ||
                ne[2] != model.hparams.n_img_embd()) {
                fprintf(stderr, "%s: prefix feature input shape mismatch [%lld,%lld,%lld,%lld]\n",
                        __func__,
                        (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
                return false;
            }
            break;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to create ggml context\n", __func__);
        return false;
    }

    struct ggml_tensor * input = nullptr;
    if (stage == SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW || stage == SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT) {
        input = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ne[0], ne[1], ne[2], ne[3]);
    } else {
        input = sam3_new_f32_tensor_4d_from_ne(ctx, ne);
    }
    ggml_set_name(input, "vit_prefix_stage_input");
    ggml_set_input(input);

    struct ggml_tensor * output = sam3_build_vit_prefix_stage_from_input(ctx, input, model, stage);
    if (!output) {
        fprintf(stderr, "%s: failed to build prefix stage %d\n", __func__, (int) stage);
        ggml_free(ctx);
        return false;
    }
    ggml_set_name(output, "vit_prefix_stage_output");
    ggml_set_output(output);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, output);

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    if (input->type == GGML_TYPE_F16) {
        const int64_t numel = ggml_nelements(input);
        std::vector<ggml_fp16_t> tmp((size_t) numel);
        ggml_fp32_to_fp16_row(input_data, tmp.data(), numel);
        ggml_backend_tensor_set(input, tmp.data(), 0, (size_t) numel * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(input, input_data, 0, (size_t) ggml_nbytes(input));
    }
    sam3_graph_compute(model.backend, graph, n_threads);

    for (int i = 0; i < 4; ++i) {
        output_ne[i] = output->ne[i];
    }

    const bool ok = sam3_copy_tensor_to_f32(output, output_data);
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return ok;
}

bool sam3_test_run_patch_mulmat_host_ref(const sam3_model         & model,
                                         const float              * input_data,
                                         const int64_t              input_ne[4],
                                         bool                       use_double_accum,
                                         std::vector<float>       & output_data,
                                         int64_t                    output_ne[4]) {
    if (!input_data) {
        return false;
    }

    int64_t ne[4];
    sam3_normalize_ne4(input_ne, ne);

    const int64_t patch_k = model.vit.patch_embed_w->ne[0] * model.vit.patch_embed_w->ne[1] * model.vit.patch_embed_w->ne[2];
    const int64_t n_img = model.hparams.n_img_embd();

    if (ne[0] != patch_k || ne[1] != n_img || ne[2] != n_img) {
        fprintf(stderr, "%s: patch_mulmat input shape mismatch [%lld,%lld,%lld,%lld]\n",
                __func__,
                (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
        return false;
    }

    const int64_t n_patch = ne[1] * ne[2] * ne[3];
    const int64_t n_out = model.vit.patch_embed_w->ne[3];

    std::vector<ggml_fp16_t> input_f16((size_t) (patch_k * n_patch));
    ggml_fp32_to_fp16_row(input_data, input_f16.data(), patch_k * n_patch);

    std::vector<ggml_fp16_t> weight_f16((size_t) (patch_k * n_out));
    ggml_backend_tensor_get(model.vit.patch_embed_w, weight_f16.data(), 0, weight_f16.size() * sizeof(ggml_fp16_t));

    output_data.resize((size_t) (n_patch * n_out));
    output_ne[0] = n_patch;
    output_ne[1] = n_out;
    output_ne[2] = 1;
    output_ne[3] = 1;

    for (int64_t oc = 0; oc < n_out; ++oc) {
        const ggml_fp16_t * w_row = weight_f16.data() + oc * patch_k;
        float * dst_row = output_data.data() + oc * n_patch;

        for (int64_t p = 0; p < n_patch; ++p) {
            const ggml_fp16_t * x_row = input_f16.data() + p * patch_k;

            if (use_double_accum) {
                double acc = 0.0;
                for (int64_t k = 0; k < patch_k; ++k) {
                    acc += (double) ggml_fp16_to_fp32(x_row[k]) * (double) ggml_fp16_to_fp32(w_row[k]);
                }
                dst_row[p] = (float) acc;
            } else {
                float acc = 0.0f;
                for (int64_t k = 0; k < patch_k; ++k) {
                    acc += ggml_fp16_to_fp32(x_row[k]) * ggml_fp16_to_fp32(w_row[k]);
                }
                dst_row[p] = acc;
            }
        }
    }

    return true;
}

bool sam3_test_run_vit_block_linear_host_ref(const sam3_model         & model,
                                             int                        block_idx,
                                             sam3_vit_block_stage       stage,
                                             const float              * input_data,
                                             const int64_t              input_ne[4],
                                             bool                       use_double_accum,
                                             std::vector<float>       & output_data,
                                             int64_t                    output_ne[4]) {
    if (!input_data || block_idx < 0 || block_idx >= (int) model.vit.blocks.size()) {
        return false;
    }

    const sam3_vit_block & blk = model.vit.blocks[(size_t) block_idx];

    const ggml_tensor * w = nullptr;
    const ggml_tensor * b = nullptr;

    switch (stage) {
        case SAM3_VIT_BLOCK_STAGE_QKV_PROJ:
            w = blk.qkv_w;
            b = blk.qkv_b;
            break;
        case SAM3_VIT_BLOCK_STAGE_ATTN_PROJ:
            w = blk.proj_w;
            b = blk.proj_b;
            break;
        case SAM3_VIT_BLOCK_STAGE_MLP_FC1:
            w = blk.mlp_fc1_w;
            b = blk.mlp_fc1_b;
            break;
        case SAM3_VIT_BLOCK_STAGE_MLP_FC2:
            w = blk.mlp_fc2_w;
            b = blk.mlp_fc2_b;
            break;
        default:
            fprintf(stderr, "%s: unsupported stage %d\n", __func__, (int) stage);
            return false;
    }

    if (!w || !b || w->type != GGML_TYPE_F16 || b->type != GGML_TYPE_F32) {
        fprintf(stderr, "%s: unsupported tensor types for stage %d\n", __func__, (int) stage);
        return false;
    }

    int64_t ne[4];
    sam3_normalize_ne4(input_ne, ne);

    const int64_t k = w->ne[0];
    const int64_t out_dim = w->ne[1];
    const int64_t n_col = ne[1] * ne[2] * ne[3];

    if (ne[0] != k) {
        fprintf(stderr, "%s: linear input shape mismatch [%lld,%lld,%lld,%lld] for K=%lld\n",
                __func__,
                (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
                (long long) k);
        return false;
    }

    std::vector<ggml_fp16_t> input_f16((size_t) (k * n_col));
    ggml_fp32_to_fp16_row(input_data, input_f16.data(), k * n_col);

    std::vector<ggml_fp16_t> weight_f16((size_t) (k * out_dim));
    ggml_backend_tensor_get(w, weight_f16.data(), 0, weight_f16.size() * sizeof(ggml_fp16_t));

    std::vector<float> bias_f32((size_t) out_dim);
    ggml_backend_tensor_get(b, bias_f32.data(), 0, bias_f32.size() * sizeof(float));

    output_data.resize((size_t) (out_dim * n_col));
    output_ne[0] = out_dim;
    output_ne[1] = ne[1];
    output_ne[2] = ne[2];
    output_ne[3] = ne[3];

    for (int64_t col = 0; col < n_col; ++col) {
        const ggml_fp16_t * x_col = input_f16.data() + col * k;
        float * dst_col = output_data.data() + col * out_dim;

        for (int64_t oc = 0; oc < out_dim; ++oc) {
            const ggml_fp16_t * w_row = weight_f16.data() + oc * k;

            if (use_double_accum) {
                double acc = (double) bias_f32[oc];
                for (int64_t i = 0; i < k; ++i) {
                    acc += (double) ggml_fp16_to_fp32(w_row[i]) * (double) ggml_fp16_to_fp32(x_col[i]);
                }
                dst_col[oc] = (float) acc;
            } else {
                float acc = bias_f32[oc];
                for (int64_t i = 0; i < k; ++i) {
                    acc += ggml_fp16_to_fp32(w_row[i]) * ggml_fp16_to_fp32(x_col[i]);
                }
                dst_col[oc] = acc;
            }
        }
    }

    return true;
}

static void sam3_normalize_ne4(const int64_t input_ne[4], int64_t ne[4]) {
    for (int i = 0; i < 4; ++i) {
        ne[i] = (input_ne && input_ne[i] > 0) ? input_ne[i] : 1;
    }
}

static struct ggml_tensor * sam3_new_f32_tensor_4d_from_ne(struct ggml_context * ctx,
                                                           const int64_t ne[4]) {
    return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
}

bool sam3_copy_tensor_to_f32(struct ggml_tensor * t, std::vector<float> & output) {
    if (!t) {
        return false;
    }
    if (!ggml_is_contiguous(t)) {
        fprintf(stderr, "%s: tensor '%s' is not contiguous\n", __func__, ggml_get_name(t));
        return false;
    }

    const int64_t numel = ggml_nelements(t);
    output.resize((size_t) numel);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, output.data(), 0, (size_t) numel * sizeof(float));
        return true;
    }

    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t) numel);
        ggml_backend_tensor_get(t, tmp.data(), 0, (size_t) numel * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), output.data(), numel);
        return true;
    }

    fprintf(stderr, "%s: unsupported tensor type %d for '%s'\n",
            __func__, (int) t->type, ggml_get_name(t));
    return false;
}

static struct ggml_tensor * sam3_build_vit_prefix_stage_from_input(struct ggml_context     * ctx,
                                                                   struct ggml_tensor      * input,
                                                                   const sam3_model        & model,
                                                                   sam3_vit_prefix_stage     stage) {
    switch (stage) {
        case SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL:
            return ggml_im2col(ctx,
                               model.vit.patch_embed_w,
                               input,
                               model.vit.patch_embed_w->ne[0],
                               model.vit.patch_embed_w->ne[1],
                               0, 0, 1, 1, true,
                               model.vit.patch_embed_w->type);

        case SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW:
        case SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT: {
            struct ggml_tensor * result = ggml_mul_mat(
                    ctx,
                    ggml_reshape_2d(ctx, input, input->ne[0], input->ne[3] * input->ne[2] * input->ne[1]),
                    ggml_reshape_2d(ctx,
                                    model.vit.patch_embed_w,
                                    model.vit.patch_embed_w->ne[0] * model.vit.patch_embed_w->ne[1] * model.vit.patch_embed_w->ne[2],
                                    model.vit.patch_embed_w->ne[3]));

            if (stage == SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW) {
                return result;
            }

            result = ggml_reshape_4d(ctx, result, input->ne[1], input->ne[2], input->ne[3], model.vit.patch_embed_w->ne[3]);
            return ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
        }

        case SAM3_VIT_PREFIX_STAGE_PATCH_EMBED: {
            struct ggml_tensor * x = ggml_conv_2d_sk_p0(ctx, model.vit.patch_embed_w, input);
            return ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        }

        case SAM3_VIT_PREFIX_STAGE_POS_ADD: {
            struct ggml_tensor * pos_target = ggml_new_tensor_4d(
                    ctx, GGML_TYPE_F32,
                    model.hparams.vit_embed_dim,
                    model.hparams.n_img_embd(),
                    model.hparams.n_img_embd(),
                    1);
            struct ggml_tensor * pos_tiled = ggml_repeat(ctx, model.vit.pos_embed, pos_target);
            return ggml_add(ctx, input, pos_tiled);
        }

        case SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM:
            return ggml_norm(ctx, input, 1e-5f);

        case SAM3_VIT_PREFIX_STAGE_LN_PRE: {
            struct ggml_tensor * x = ggml_norm(ctx, input, 1e-5f);
            x = ggml_mul_inplace(ctx, x, model.vit.ln_pre_w);
            x = ggml_add_inplace(ctx, x, model.vit.ln_pre_b);
            return x;
        }
    }

    return nullptr;
}

static struct ggml_tensor * sam3_build_vit_attn_core_from_qkv(struct ggml_context * ctx,
                                                              struct ggml_tensor  * qkv,
                                                              const sam3_vit_block & blk,
                                                              const sam3_hparams   & hp) {
    const int E = hp.vit_embed_dim;
    const int NH = hp.vit_num_heads;
    const int HD = hp.vit_head_dim();

    const int64_t W_cur = qkv->ne[1];
    const int64_t H_cur = qkv->ne[2];
    const int64_t B_cur = qkv->ne[3];

    struct ggml_tensor * cur = ggml_reshape_4d(ctx, qkv, E, 3, W_cur * H_cur, B_cur);
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 3, 1, 2));

    struct ggml_tensor * Q = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                                          cur->nb[1], cur->nb[2], 0);
    struct ggml_tensor * K = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                                          cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
    struct ggml_tensor * V = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                                          cur->nb[1], cur->nb[2], 2 * cur->nb[3]);

    Q = ggml_reshape_4d(ctx, Q, HD, NH, W_cur * H_cur, B_cur);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(ctx, Q, HD, W_cur * H_cur, NH * B_cur);

    K = ggml_reshape_4d(ctx, K, HD, NH, W_cur * H_cur, B_cur);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(ctx, K, HD, W_cur * H_cur, NH * B_cur);

    V = ggml_reshape_4d(ctx, V, HD, NH, W_cur * H_cur, B_cur);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    if (blk.freqs_cis) {
        Q = sam3_apply_rope(ctx, Q, blk.freqs_cis);
        K = sam3_apply_rope(ctx, K, blk.freqs_cis);
    }

    Q = ggml_reshape_4d(ctx, Q, HD, W_cur * H_cur, NH, B_cur);
    K = ggml_reshape_4d(ctx, K, HD, W_cur * H_cur, NH, B_cur);

    const float scale = 1.0f / sqrtf((float) HD);
    struct ggml_tensor * attn_out = sam3_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);

    return ggml_cont(ctx, ggml_reshape_4d(ctx, attn_out, E, W_cur, H_cur, B_cur));
}

static struct ggml_tensor * sam3_build_vit_block_stage_from_input(struct ggml_context     * ctx,
                                                                  struct ggml_tensor      * input,
                                                                  const sam3_vit_block    & blk,
                                                                  const sam3_hparams      & hp,
                                                                  sam3_vit_block_stage      stage) {
    switch (stage) {
        case SAM3_VIT_BLOCK_STAGE_NORM1:
            return sam3_layer_norm(ctx, input, blk.norm1_w, blk.norm1_b);

        case SAM3_VIT_BLOCK_STAGE_WINDOW_PART:
            return sam3_win_part(ctx, input, hp.vit_window_size);

        case SAM3_VIT_BLOCK_STAGE_QKV_PROJ:
            return ggml_add(ctx, ggml_mul_mat(ctx, blk.qkv_w, input), blk.qkv_b);

        case SAM3_VIT_BLOCK_STAGE_ATTN_CORE:
            return sam3_build_vit_attn_core_from_qkv(ctx, input, blk, hp);

        case SAM3_VIT_BLOCK_STAGE_ATTN_PROJ:
            return ggml_add(ctx, ggml_mul_mat(ctx, blk.proj_w, input), blk.proj_b);

        case SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART:
            return sam3_win_unpart(ctx, input, hp.n_img_embd(), hp.n_img_embd(), hp.vit_window_size);

        case SAM3_VIT_BLOCK_STAGE_NORM2:
            return sam3_layer_norm(ctx, input, blk.norm2_w, blk.norm2_b);

        case SAM3_VIT_BLOCK_STAGE_MLP_FC1:
            return ggml_add(ctx, ggml_mul_mat(ctx, blk.mlp_fc1_w, input), blk.mlp_fc1_b);

        case SAM3_VIT_BLOCK_STAGE_MLP_GELU:
            return ggml_gelu_erf(ctx, input);

        case SAM3_VIT_BLOCK_STAGE_MLP_FC2:
            return ggml_add(ctx, ggml_mul_mat(ctx, blk.mlp_fc2_w, input), blk.mlp_fc2_b);

        case SAM3_VIT_BLOCK_STAGE_MLP: {
            struct ggml_tensor * x = ggml_mul_mat(ctx, blk.mlp_fc1_w, input);
            x = ggml_add(ctx, x, blk.mlp_fc1_b);
            x = ggml_gelu_erf(ctx, x);
            x = ggml_mul_mat(ctx, blk.mlp_fc2_w, x);
            x = ggml_add(ctx, x, blk.mlp_fc2_b);
            return x;
        }
    }

    return nullptr;
}

bool sam3_test_run_vit_block_stage(const sam3_model        & model,
                                   int                       block_idx,
                                   sam3_vit_block_stage      stage,
                                   const float             * input_data,
                                   const int64_t             input_ne[4],
                                   std::vector<float>      & output_data,
                                   int64_t                   output_ne[4],
                                   int                       n_threads) {
    if (!input_data) {
        fprintf(stderr, "%s: input_data is null\n", __func__);
        return false;
    }
    if (block_idx < 0 || block_idx >= (int) model.vit.blocks.size()) {
        fprintf(stderr, "%s: invalid block_idx=%d\n", __func__, block_idx);
        return false;
    }

    const auto & hp = model.hparams;
    const auto & blk = model.vit.blocks[block_idx];

    int64_t ne[4];
    sam3_normalize_ne4(input_ne, ne);

    switch (stage) {
        case SAM3_VIT_BLOCK_STAGE_WINDOW_PART:
        case SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART:
            if (ne[0] != hp.vit_embed_dim) {
                fprintf(stderr, "%s: window stage input ne0=%lld expected %d\n",
                        __func__, (long long) ne[0], hp.vit_embed_dim);
                return false;
            }
            break;
        case SAM3_VIT_BLOCK_STAGE_QKV_PROJ:
            if (ne[0] != hp.vit_embed_dim) {
                fprintf(stderr, "%s: qkv input ne0=%lld expected %d\n",
                        __func__, (long long) ne[0], hp.vit_embed_dim);
                return false;
            }
            break;
        case SAM3_VIT_BLOCK_STAGE_ATTN_CORE:
            if (ne[0] != 3 * hp.vit_embed_dim) {
                fprintf(stderr, "%s: attn input ne0=%lld expected %d\n",
                        __func__, (long long) ne[0], 3 * hp.vit_embed_dim);
                return false;
            }
            break;
        case SAM3_VIT_BLOCK_STAGE_MLP_FC1:
            if (ne[0] != hp.vit_embed_dim) {
                fprintf(stderr, "%s: mlp_fc1 input ne0=%lld expected %d\n",
                        __func__, (long long) ne[0], hp.vit_embed_dim);
                return false;
            }
            break;
        case SAM3_VIT_BLOCK_STAGE_MLP_GELU:
        case SAM3_VIT_BLOCK_STAGE_MLP_FC2:
            if (ne[0] != hp.vit_mlp_dim) {
                fprintf(stderr, "%s: mlp stage input ne0=%lld expected %d\n",
                        __func__, (long long) ne[0], hp.vit_mlp_dim);
                return false;
            }
            break;
        default:
            if (ne[0] != hp.vit_embed_dim) {
                fprintf(stderr, "%s: stage input ne0=%lld expected %d\n",
                        __func__, (long long) ne[0], hp.vit_embed_dim);
                return false;
            }
            break;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to create ggml context\n", __func__);
        return false;
    }

    struct ggml_tensor * input = sam3_new_f32_tensor_4d_from_ne(ctx, ne);
    ggml_set_name(input, "vit_block_stage_input");
    ggml_set_input(input);

    struct ggml_tensor * output = sam3_build_vit_block_stage_from_input(ctx, input, blk, hp, stage);
    if (!output) {
        fprintf(stderr, "%s: failed to build stage %d\n", __func__, (int) stage);
        ggml_free(ctx);
        return false;
    }
    ggml_set_name(output, "vit_block_stage_output");
    ggml_set_output(output);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, output);

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph for stage %d\n", __func__, (int) stage);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    const size_t input_bytes = (size_t) ne[0] * (size_t) ne[1] * (size_t) ne[2] * (size_t) ne[3] * sizeof(float);
    ggml_backend_tensor_set(input, input_data, 0, input_bytes);
    sam3_graph_compute(model.backend, graph, n_threads);

    for (int i = 0; i < 4; ++i) {
        output_ne[i] = output->ne[i];
    }

    const bool ok = sam3_copy_tensor_to_f32(output, output_data);
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return ok;
}

// Test-only: encode from pre-preprocessed float data (bypasses C++ resize/normalize).
// chw_data is in [C, H, W] layout, already normalized, with C=3, H=W=img_size.
bool sam3_encode_image_from_preprocessed(sam3_state& state,
                                         const sam3_model& model,
                                         const float* chw_data,
                                         int img_size) {
    auto t_start = std::chrono::high_resolution_clock::now();
    const auto& hp = model.hparams;

    if (img_size != hp.img_size) {
        fprintf(stderr, "%s: img_size mismatch: got %d, expected %d\n",
                __func__, img_size, hp.img_size);
        return false;
    }

    fprintf(stderr, "%s: encoding from preprocessed %dx%d\n", __func__, img_size, img_size);

#ifdef SAM3_TRT_ENCODER
    if (sam3_try_trt_encode_image(state, model, chw_data, img_size)) {
        return true;
    }
#endif
    if (model.trt_only_weights) {
        fprintf(stderr, "%s: TRT encode unavailable/out of scope but the model was loaded "
                        "with SAM3_TRT_SKIP_GGML_WEIGHTS=1 -- ggml weights absent, refusing "
                        "to run the fallback graph\n", __func__);
        return false;
    }

    state.orig_width = img_size;
    state.orig_height = img_size;

    // ── Build computation graph (SAM3 path) ──
    const size_t buf_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    auto* inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img_size, img_size, 3, 1);
    ggml_set_name(inp, "input_image");
    ggml_set_input(inp);

    auto* vit_out = sam3_build_vit_graph(ctx0, inp, model);
    ggml_set_name(vit_out, "vit_output");
    ggml_set_output(vit_out);

    // Mark debug intermediate tensors as outputs so graph allocator preserves them
    {
        const char* dbg_names[] = {
            "dbg_patch_embed",
            "dbg_after_pos_embed",
            "dbg_ln_pre_norm",
            "dbg_ln_pre_scale",
            "dbg_after_ln_pre",
            "dbg_block_15_norm1",
            "dbg_block_15_qkv_proj",
            "dbg_block_15_q_split",
            "dbg_block_15_k_split",
            "dbg_block_15_v_split",
            "dbg_block_15_q_heads_base",
            "dbg_block_15_k_heads_base",
            "dbg_block_15_v_heads_base",
            "dbg_block_15_q_heads",
            "dbg_block_15_k_heads",
            "dbg_block_15_v_flash",
            "dbg_block_15_q_rope",
            "dbg_block_15_k_rope",
            "dbg_block_15_q_flash",
            "dbg_block_15_k_flash",
            "dbg_block_15_attn_out",
            "dbg_block_15_attn_proj",
            "dbg_block_15_resid1",
            "dbg_block_15_norm2",
            "dbg_block_15_mlp",
        };
        for (const char* dn : dbg_names) {
            auto* dt = ggml_get_tensor(ctx0, dn);
            if (dt) ggml_set_output(dt);
        }
        for (int i = 0; i < (int)model.hparams.vit_depth; ++i) {
            char bn[64];
            snprintf(bn, sizeof(bn), "dbg_block_%d_out", i);
            auto* dt = ggml_get_tensor(ctx0, bn);
            if (dt) ggml_set_output(dt);
        }
    }

    struct ggml_tensor* neck_det_out[4] = {};
    struct ggml_tensor* neck_trk_out[4];
    if (!model.hparams.visual_only) {
        sam3_build_neck_graph(ctx0, vit_out, model.neck_det, neck_det_out);
    }
    sam3_build_neck_graph(ctx0, vit_out, model.neck_trk, neck_trk_out);

    for (int i = 0; i < 4; ++i) {
        char name[64];
        if (!model.hparams.visual_only) {
            snprintf(name, sizeof(name), "neck_det_%d", i);
            ggml_set_name(neck_det_out[i], name);
            ggml_set_output(neck_det_out[i]);
        }
        snprintf(name, sizeof(name), "neck_trk_%d", i);
        ggml_set_name(neck_trk_out[i], name);
        ggml_set_output(neck_trk_out[i]);
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 16384, false);
    for (int i = 0; i < 4; ++i) {
        if (!model.hparams.visual_only) {
            ggml_build_forward_expand(graph, neck_det_out[i]);
        }
        ggml_build_forward_expand(graph, neck_trk_out[i]);
    }

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    if (!ggml_gallocr_reserve(galloc, graph)) {
        fprintf(stderr, "%s: failed to reserve graph memory\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    // Upload preprocessed data directly (already in CHW = ggml [W, H, C, B] layout)
    const size_t data_bytes = (size_t)3 * img_size * img_size * sizeof(float);
    ggml_backend_tensor_set(inp, chw_data, 0, data_bytes);

    // Compute
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx0);
            return false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "%s: graph computed in %.1f ms (%d threads)\n",
                __func__, ms, state.n_threads);
    }

    // Cache results in state
    if (state.galloc) ggml_gallocr_free(state.galloc);
    if (state.ctx) ggml_free(state.ctx);

    state.ctx = ctx0;
    state.galloc = galloc;
    state.backend = model.backend;
    state.vit_output = vit_out;

    for (int i = 0; i < 4; ++i) {
        state.neck_det[i] = model.hparams.visual_only ? nullptr : neck_det_out[i];
        state.neck_trk[i] = neck_trk_out[i];
    }

    // Compute sinusoidal PEs
    {
        const int neck_dim = hp.neck_dim;
        const int scale_sizes[4] = {
            hp.n_img_embd() * 4,
            hp.n_img_embd() * 2,
            hp.n_img_embd(),
            hp.n_img_embd() / 2,
        };

        if (state.pe_buf) {
            ggml_backend_buffer_free(state.pe_buf);
            state.pe_buf = nullptr;
        }
        if (state.pe_ctx) {
            ggml_free(state.pe_ctx);
            state.pe_ctx = nullptr;
        }

        struct ggml_init_params pe_params = {
            /*.mem_size   =*/ggml_tensor_overhead() * 4 + 256,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        state.pe_ctx = ggml_init(pe_params);

        struct ggml_tensor* pe_tensors[4];
        for (int i = 0; i < 4; ++i) {
            const int S = scale_sizes[i];
            pe_tensors[i] = ggml_new_tensor_4d(state.pe_ctx, GGML_TYPE_F32, neck_dim, S, S, 1);
            char name[64];
            snprintf(name, sizeof(name), "pe_%d", i);
            ggml_set_name(pe_tensors[i], name);
        }

        state.pe_buf = ggml_backend_alloc_ctx_tensors(state.pe_ctx, model.backend);
        if (!state.pe_buf) {
            fprintf(stderr, "%s: failed to allocate PE buffer\n", __func__);
        } else {
            for (int i = 0; i < 4; ++i) {
                const int S = scale_sizes[i];
                auto pe_data = sam3_sinusoidal_pe_2d(S, S, neck_dim);
                ggml_backend_tensor_set(pe_tensors[i], pe_data.data(), 0, pe_data.size() * sizeof(float));
                state.neck_det_pe[i] = pe_tensors[i];
                state.neck_trk_pe[i] = pe_tensors[i];
            }
        }
    }

    if (sam3_eff_feat_size(state, hp) != state.pe_cache_feat_size) {
        state.pe_cache_valid = false;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    fprintf(stderr, "%s: encoded successfully in %.1f ms\n", __func__, total_ms);
    return true;
}

/*****************************************************************************
** Tokenizer — standalone test API (does not require model weights)
*****************************************************************************/

// Global tokenizer instance for the test API.
static sam3_bpe_tokenizer g_test_tokenizer;
static bool g_test_tokenizer_loaded = false;

bool sam3_test_load_tokenizer(const std::string& model_path) {
    std::ifstream fin(model_path, std::ios::binary);
    if (!fin) return false;

    // Read header
    uint32_t magic;
    int32_t version, ftype, n_tensors;
    fin.read(reinterpret_cast<char*>(&magic), 4);
    fin.read(reinterpret_cast<char*>(&version), 4);
    fin.read(reinterpret_cast<char*>(&ftype), 4);
    fin.read(reinterpret_cast<char*>(&n_tensors), 4);
    if (magic != SAM3_MAGIC || version != SAM3_FILE_VERSION) return false;

    // Skip hparams
    sam3_hparams hp;
    if (!sam3_load_hparams(fin, hp)) return false;
    if (hp.visual_only) return false;

    // Skip tensors
    for (int t = 0; t < n_tensors; ++t) {
        int32_t n_dims, name_len, dtype;
        fin.read(reinterpret_cast<char*>(&n_dims), 4);
        fin.read(reinterpret_cast<char*>(&name_len), 4);
        fin.read(reinterpret_cast<char*>(&dtype), 4);
        if (fin.fail()) return false;

        // Read shape to compute data size
        int64_t n_el = 1;
        std::vector<int64_t> shape(n_dims);
        for (int i = 0; i < n_dims; ++i) {
            int32_t d;
            fin.read(reinterpret_cast<char*>(&d), 4);
            shape[i] = d;
            n_el *= d;
        }

        // Skip name
        fin.seekg(name_len, std::ios::cur);

        // Skip padding to 32-byte alignment
        size_t pos = fin.tellg();
        size_t pad = (32 - pos % 32) % 32;
        if (pad > 0) fin.seekg(pad, std::ios::cur);

        // Compute data size and skip
        const ggml_type file_type = static_cast<ggml_type>(dtype);
        size_t bytes;
        if (ggml_is_quantized(file_type)) {
            const int64_t n_rows = n_el / shape[0];
            bytes = ggml_row_size(file_type, shape[0]) * n_rows;
        } else {
            const size_t elem_size = (file_type == GGML_TYPE_F16) ? 2 : 4;
            bytes = n_el * elem_size;
        }
        fin.seekg(bytes, std::ios::cur);
        if (fin.fail()) return false;
    }

    // Read embedded tokenizer
    if (!sam3_load_bpe_vocab_from_stream(fin, g_test_tokenizer)) return false;
    g_test_tokenizer_loaded = true;
    return true;
}

std::vector<int32_t> sam3_test_tokenize(const std::string& text) {
    if (!g_test_tokenizer_loaded) return {};
    return sam3_tokenize(g_test_tokenizer, text, 32);
}

static bool sam3_dump_tensor_to_path(struct ggml_tensor* t,
                                     const std::string& tensor_name,
                                     const std::string& output_path) {
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' is null\n", __func__, tensor_name.c_str());
        return false;
    }

    int64_t numel = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > 0) {
            numel *= t->ne[i];
        }
    }

    std::vector<float> data(numel);
    if (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32 && !ggml_is_quantized(t->type)) {
        fprintf(stderr, "%s: unsupported tensor type %d for '%s'\n",
                __func__, (int)t->type, tensor_name.c_str());
        return false;
    }

    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];

    if (ggml_is_quantized(t->type)) {
        // Weight tensors registered at a quantized type (e.g. Q8_0 qkv/proj/mlp
        // matrices in a q8_0 build, see sam3_register_tensors's T2/T3/T4) are
        // always freshly-allocated and contiguous -- no permuted/viewed
        // quantized tensor exists in this codebase, so only that case is
        // handled here.
        if (!ggml_is_contiguous(t)) {
            fprintf(stderr, "%s: non-contiguous quantized tensor '%s' not supported\n",
                    __func__, tensor_name.c_str());
            return false;
        }
        const auto * traits = ggml_get_type_traits(t->type);
        if (!traits->to_float) {
            fprintf(stderr, "%s: no dequantize function for '%s' (type=%s)\n",
                    __func__, tensor_name.c_str(), ggml_type_name(t->type));
            return false;
        }
        const size_t row_size = ggml_row_size(t->type, ne0);
        std::vector<char> raw(row_size * (numel / ne0));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        traits->to_float(raw.data(), data.data(), numel);
    } else if (ggml_is_contiguous(t)) {
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> f16_data(numel);
            ggml_backend_tensor_get(t, f16_data.data(), 0, numel * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(f16_data.data(), data.data(), numel);
        } else {
            ggml_backend_tensor_get(t, data.data(), 0, numel * sizeof(float));
        }
    } else if (t->nb[0] == ggml_type_size(t->type)) {
        // Serialize non-contiguous logical tensors in row-major ggml order.
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> row(ne0);
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        const size_t row_idx = ((size_t) i3 * ne2 * ne1 + (size_t) i2 * ne1 + (size_t) i1) * ne0;
                        const size_t offs = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1];
                        ggml_backend_tensor_get(t, row.data(), offs, ne0 * sizeof(ggml_fp16_t));
                        ggml_fp16_to_fp32_row(row.data(), data.data() + row_idx, ne0);
                    }
                }
            }
        } else {
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        const size_t row_idx = ((size_t) i3 * ne2 * ne1 + (size_t) i2 * ne1 + (size_t) i1) * ne0;
                        const size_t offs = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1];
                        ggml_backend_tensor_get(t, data.data() + row_idx, offs, ne0 * sizeof(float));
                    }
                }
            }
        }
    } else {
        fprintf(stderr, "%s: unsupported non-contiguous layout for '%s' (nb0=%llu)\n",
                __func__, tensor_name.c_str(), (unsigned long long) t->nb[0]);
        return false;
    }

    {
        std::ofstream f(output_path + ".bin", std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(data.data()), numel * sizeof(float));
    }

    {
        std::ofstream f(output_path + ".shape");
        if (!f) return false;
        int ndims = ggml_n_dims(t);
        for (int i = 0; i < ndims; ++i) {
            if (i > 0) f << ",";
            f << t->ne[i];
        }
        f << "\n";
    }

    fprintf(stderr, "%s: dumped '%s' [", __func__, tensor_name.c_str());
    for (int i = 0; i < ggml_n_dims(t); ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%lld", (long long)t->ne[i]);
    }
    fprintf(stderr, "] to %s\n", output_path.c_str());

    return true;
}

static bool sam3_dump_raw_f32_to_path(const float* data,
                                      const std::vector<int64_t>& shape,
                                      const std::string& output_path) {
    int64_t numel = 1;
    for (int64_t d : shape) {
        numel *= d;
    }

    {
        std::ofstream f(output_path + ".bin", std::ios::binary);
        if (!f) {
            return false;
        }
        f.write(reinterpret_cast<const char*>(data), numel * sizeof(float));
    }

    {
        std::ofstream f(output_path + ".shape");
        if (!f) {
            return false;
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                f << ",";
            }
            f << shape[i];
        }
        f << "\n";
    }

    return true;
}

static bool sam3_load_ref_f32_data(const std::string& path,
                                   std::vector<float>& data,
                                   int expected_numel = -1) {
    std::ifstream shape_f(path + ".shape");
    if (!shape_f) {
        fprintf(stderr, "%s: missing %s.shape\n", __func__, path.c_str());
        return false;
    }

    int64_t numel = 1;
    std::string line;
    std::getline(shape_f, line);
    size_t pos = 0;
    while (pos < line.size()) {
        size_t end = line.find(',', pos);
        if (end == std::string::npos) {
            end = line.size();
        }
        if (end > pos) {
            numel *= std::stoll(line.substr(pos, end - pos));
        }
        pos = end + 1;
    }

    if (expected_numel >= 0 && numel != expected_numel) {
        fprintf(stderr, "%s: %s expected %d elements, got %lld\n",
                __func__, path.c_str(), expected_numel, (long long)numel);
        return false;
    }

    std::ifstream data_f(path + ".bin", std::ios::binary);
    if (!data_f) {
        fprintf(stderr, "%s: missing %s.bin\n", __func__, path.c_str());
        return false;
    }

    data.resize((size_t)numel);
    data_f.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));
    return data_f.good() || data_f.eof();
}

static std::vector<float> sam3_reorder_nchw_to_ggml_dwh(const std::vector<float>& src,
                                                        int channels,
                                                        int height,
                                                        int width) {
    std::vector<float> dst((size_t)channels * height * width);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t src_idx = (size_t)c * height * width + (size_t)y * width + x;
                const size_t dst_idx = (size_t)c + (size_t)x * channels + (size_t)y * channels * width;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
    return dst;
}

static bool sam3_load_kv_text_file(const std::string& path,
                                   std::map<std::string, std::string>& kv) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "%s: missing %s\n", __func__, path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        kv[line.substr(0, eq)] = line.substr(eq + 1);
    }

    return true;
}

static int sam3_meta_get_int(const std::map<std::string, std::string>& kv,
                             const std::string& key,
                             int default_value = 0) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        return default_value;
    }
    return std::stoi(it->second);
}

bool sam3_test_dump_text_encoder(const sam3_model& model,
                                 const std::vector<int32_t>& token_ids,
                                 const std::string& output_dir,
                                 int n_threads) {
    const int L = model.hparams.text_ctx_len;
    if ((int)token_ids.size() != L) {
        fprintf(stderr, "%s: expected %d token IDs, got %zu\n",
                __func__, L, token_ids.size());
        return false;
    }

    const size_t buf_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    auto* inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(inp_tokens, "text_token_ids");
    ggml_set_input(inp_tokens);

    auto* text_features = sam3_build_text_encoder_graph(ctx0, inp_tokens, model);
    std::vector<std::string> tensor_names = {
        "causal_mask",
        "text_token_embed",
        "text_after_pos_embed",
        "text_final_ln",
        "text_features_2d",
    };
    for (int i = 0; i < model.hparams.text_layers; ++i) {
        char name[64];

        snprintf(name, sizeof(name), "text_block_%02d_after_ln1", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_qkv", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_attn_out", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_after_attn_residual", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_after_ln2", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_fc1", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_gelu", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_out", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_out", i);
        tensor_names.emplace_back(name);
    }

    ggml_set_output(text_features);
    for (const auto& name : tensor_names) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (t && t != inp_tokens) {
            ggml_set_output(t);
        }
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(graph, text_features);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate text encoder graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(inp_tokens, token_ids.data(), 0, L * sizeof(int32_t));

    auto* causal_mask = ggml_get_tensor(ctx0, "causal_mask");
    if (!causal_mask) {
        fprintf(stderr, "%s: causal_mask tensor not found\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    std::vector<ggml_fp16_t> mask_data(L * L);
    sam3_fill_causal_mask(mask_data.data(), L);
    ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, L * L * sizeof(ggml_fp16_t));

    sam3_graph_compute(model.backend, graph, n_threads);

    bool ok = true;
    for (const auto& name : tensor_names) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found in graph\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

bool sam3_test_dump_phase5(const sam3_model& model,
                           const sam3_state& state,
                           const std::vector<int32_t>& token_ids,
                           const std::string& output_dir,
                           int n_threads) {
    const auto& hp = model.hparams;
    const int D = hp.neck_dim;
    const int H = hp.n_img_embd();
    const int L = hp.text_ctx_len;
    const int NQ = hp.ddec_num_queries;

    if ((int)token_ids.size() != L) {
        fprintf(stderr, "%s: expected %d token IDs, got %zu\n",
                __func__, L, token_ids.size());
        return false;
    }
    if (!state.neck_det[0] || !state.neck_det_pe[2]) {
        fprintf(stderr, "%s: encoded detector features are missing\n", __func__);
        return false;
    }

    const size_t buf_size = ggml_tensor_overhead() * 65536 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    auto* inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(inp_tokens, "text_token_ids");
    ggml_set_input(inp_tokens);

    auto* text_features_2d = sam3_build_text_encoder_graph(ctx0, inp_tokens, model);
    auto* text_features = ggml_reshape_3d(ctx0, text_features_2d, D, L, 1);
    ggml_set_name(text_features, "text_features");

    // Make a snapshot copy of text_features for dumping — the graph allocator
    // may reuse the view's underlying buffer for later ops.
    auto* text_features_snap = ggml_cont(ctx0, ggml_reshape_2d(ctx0, text_features_2d, D, L));
    ggml_set_name(text_features_snap, "text_features_snap");

    auto* img_feats = ggml_reshape_3d(ctx0, state.neck_det[2], D, H * H, 1);
    auto* img_pe = ggml_reshape_3d(ctx0, state.neck_det_pe[2], D, H * H, 1);

    auto* sine_dim_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 64);
    ggml_set_name(sine_dim_t, "sine_dim_t");
    ggml_set_input(sine_dim_t);

    auto* rpb_coords = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, H);
    ggml_set_name(rpb_coords, "rpb_coords");
    ggml_set_input(rpb_coords);

    auto* text_valid_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_valid_mask, "text_valid_mask");
    ggml_set_input(text_valid_mask);

    auto* text_attn_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_attn_bias, "text_attn_bias");
    ggml_set_input(text_attn_bias);

    auto* conditioned = sam3_build_fenc_graph(ctx0, model, img_feats, text_features,
                                              img_pe, text_attn_bias);
    auto* fenc_layer5_out = ggml_cont(ctx0, conditioned);
    ggml_set_name(fenc_layer5_out, "fenc_layer5_out");
    ggml_set_name(conditioned, "fenc_output");

    auto ddec_out = sam3_build_ddec_graph(ctx0, model, conditioned, img_pe, text_features,
                                          sine_dim_t, rpb_coords, text_attn_bias, text_valid_mask);

    struct ggml_tensor* fpn_feats[3] = {
        state.neck_det[0],
        state.neck_det[1],
        state.neck_det[2],
    };

    auto* obj_queries = ggml_view_3d(ctx0, ddec_out.queries, D, NQ, 1,
                                     ddec_out.queries->nb[1], ddec_out.queries->nb[2],
                                     1 * ddec_out.queries->nb[1]);
    obj_queries = ggml_cont(ctx0, obj_queries);

    auto* mask_logits = sam3_build_seg_head_graph(ctx0, model, conditioned, fpn_feats,
                                                  obj_queries, text_features, text_attn_bias);
    ggml_set_name(mask_logits, "seg_mask_logits");

    struct named_tensor {
        const char* name;
        struct ggml_tensor* tensor;
    };

    const std::vector<named_tensor> direct_tensors = {
        {"text_features", text_features_snap},
        {"text_valid_mask", text_valid_mask},
        {"fenc_img_input", img_feats},
        {"fenc_pos_embed", img_pe},
        {"fenc_prompt", text_features_snap},
        {"img_pe_72", state.neck_det_pe[2]},
        {"ddec_query_embed", model.ddec.query_embed},
        {"ddec_ref_pts_raw", model.tensors.at("ddec.reference_points.weight")},
        {"ddec_presence_token", model.ddec.presence_token},
        {"ddec_pred_boxes", ddec_out.pred_boxes},
        {"ddec_presence_logit", ddec_out.presence_score},
    };

    std::vector<std::string> named_outputs = {
        "fenc_output",
        "ddec_ref_boxes_init",
        "ddec_query_sine_0",
        "ddec_query_pos_0",
        "ddec_rpb_mask_0",
        "ddec_layer0_after_sa",
        "ddec_layer0_after_text_ca",
        "ddec_layer0_after_img_ca",
        "ddec_layer0_full_out",
        "ddec_layer0_presence",
        "ddec_normed_output",
        "scoring_prompt_mlp_out",
        "scoring_pooled",
        "scoring_proj_pooled",
        "scoring_proj_hs",
        "scoring_class_scores",
        "seg_enc_after_ca",
        "seg_enc_visual",
        "seg_pixel_dec_stage0",
        "seg_pixel_dec_stage1",
        "seg_pixel_decoder_out",
        "seg_instance_embed",
        "seg_mask_embed",
        "seg_mask_logits",
    };

    for (int i = 0; i < hp.fenc_layers; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "fenc_layer%d_out", i);
        named_outputs.emplace_back(name);
    }
    for (int i = 0; i < hp.ddec_layers; ++i) {
        char out_name[64];
        char box_name[64];
        snprintf(out_name, sizeof(out_name), "ddec_layer%d_out", i);
        snprintf(box_name, sizeof(box_name), "ddec_layer%d_refboxes", i);
        named_outputs.emplace_back(out_name);
        named_outputs.emplace_back(box_name);
    }

    ggml_set_output(text_features_snap);
    ggml_set_output(text_valid_mask);
    ggml_set_output(img_feats);
    ggml_set_output(img_pe);
    for (const auto& name : named_outputs) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_set_output(t);
        }
    }
    ggml_set_output(mask_logits);
    ggml_set_output(ddec_out.class_scores);
    ggml_set_output(ddec_out.pred_boxes);
    ggml_set_output(ddec_out.presence_score);

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 65536, false);
    ggml_build_forward_expand(graph, ddec_out.class_scores);
    ggml_build_forward_expand(graph, ddec_out.pred_boxes);
    ggml_build_forward_expand(graph, ddec_out.presence_score);
    ggml_build_forward_expand(graph, mask_logits);
    ggml_build_forward_expand(graph, text_features_snap);
    ggml_build_forward_expand(graph, text_valid_mask);
    ggml_build_forward_expand(graph, img_feats);
    ggml_build_forward_expand(graph, img_pe);
    for (const auto& name : named_outputs) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_build_forward_expand(graph, t);
        }
    }

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate phase 5 graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(inp_tokens, token_ids.data(), 0, L * sizeof(int32_t));

    auto* causal_mask = ggml_get_tensor(ctx0, "causal_mask");
    if (causal_mask) {
        std::vector<ggml_fp16_t> mask_data(L * L);
        sam3_fill_causal_mask(mask_data.data(), L);
        ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, L * L * sizeof(ggml_fp16_t));
    }

    {
        std::vector<float> dim_t_data(64);
        for (int i = 0; i < 64; ++i) {
            dim_t_data[i] = 2.0f * (float)M_PI / powf(10000.0f, 2.0f * (float)i / 128.0f);
        }
        ggml_backend_tensor_set(sine_dim_t, dim_t_data.data(), 0, 64 * sizeof(float));
    }

    {
        std::vector<float> coords(H);
        for (int i = 0; i < H; ++i) {
            coords[i] = (float)i / (float)H;
        }
        ggml_backend_tensor_set(rpb_coords, coords.data(), 0, H * sizeof(float));
    }

    auto* qpos_pres = ggml_get_tensor(ctx0, "ddec_query_pos_pres");
    if (qpos_pres) {
        std::vector<float> zeros(D, 0.0f);
        ggml_backend_tensor_set(qpos_pres, zeros.data(), 0, D * sizeof(float));
    }

    auto* rpb_pz = ggml_get_tensor(ctx0, "rpb_pres_zeros");
    if (rpb_pz) {
        int n = (int)(rpb_pz->ne[0] * rpb_pz->ne[1] * rpb_pz->ne[2] * rpb_pz->ne[3]);
        std::vector<float> zeros(n, 0.0f);
        ggml_backend_tensor_set(rpb_pz, zeros.data(), 0, n * sizeof(float));
    }

    {
        int n_valid = 0;
        for (int i = 0; i < L; ++i) {
            if (token_ids[i] != 0) {
                ++n_valid;
            }
        }
        if (n_valid == 0) {
            n_valid = 1;
        }

        const float scale = (float)L / (float)n_valid;
        std::vector<float> valid_mask(L);
        std::vector<float> attn_bias(L);
        for (int i = 0; i < L; ++i) {
            const bool is_valid = token_ids[i] != 0;
            valid_mask[i] = is_valid ? scale : 0.0f;
            attn_bias[i] = is_valid ? 0.0f : -1.0e9f;
        }
        ggml_backend_tensor_set(text_valid_mask, valid_mask.data(), 0, L * sizeof(float));
        ggml_backend_tensor_set(text_attn_bias, attn_bias.data(), 0, L * sizeof(float));
    }

    sam3_graph_compute(model.backend, graph, n_threads);

    bool ok = true;
    for (const auto& item : direct_tensors) {
        if (!sam3_dump_tensor_to_path(item.tensor, item.name, output_dir + "/" + item.name)) {
            ok = false;
        }
    }
    for (const auto& name : named_outputs) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found in graph\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

bool sam3_test_dump_phase5_from_ref_inputs(const sam3_model& model,
                                           const std::vector<int32_t>& token_ids,
                                           const std::string& prephase_ref_dir,
                                           const std::string& phase5_ref_dir,
                                           const std::string& output_dir,
                                           int n_threads) {
    const auto& hp = model.hparams;
    const int D = hp.neck_dim;
    const int H = hp.n_img_embd();
    const int L = hp.text_ctx_len;
    const int NQ = hp.ddec_num_queries;
    const int H1 = H * 2;
    const int H0 = H * 4;

    if ((int)token_ids.size() != L) {
        fprintf(stderr, "%s: expected %d token IDs, got %zu\n",
                __func__, L, token_ids.size());
        return false;
    }

    std::vector<float> neck_det_0;
    std::vector<float> neck_det_1;
    std::vector<float> fenc_img_input_data;
    std::vector<float> fenc_pos_embed_data;
    std::vector<float> img_pe_72_data;
    std::vector<float> text_features_data;
    if (!sam3_load_ref_f32_data(prephase_ref_dir + "/neck_det_0", neck_det_0, D * H0 * H0) ||
        !sam3_load_ref_f32_data(prephase_ref_dir + "/neck_det_1", neck_det_1, D * H1 * H1) ||
        !sam3_load_ref_f32_data(phase5_ref_dir + "/fenc_img_input", fenc_img_input_data, D * H * H) ||
        !sam3_load_ref_f32_data(phase5_ref_dir + "/fenc_pos_embed", fenc_pos_embed_data, D * H * H) ||
        !sam3_load_ref_f32_data(phase5_ref_dir + "/img_pe_72", img_pe_72_data, D * H * H) ||
        !sam3_load_ref_f32_data(phase5_ref_dir + "/text_features", text_features_data, D * L)) {
        return false;
    }

    neck_det_0 = sam3_reorder_nchw_to_ggml_dwh(neck_det_0, D, H0, H0);
    neck_det_1 = sam3_reorder_nchw_to_ggml_dwh(neck_det_1, D, H1, H1);

    const size_t buf_size = ggml_tensor_overhead() * 65536 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    auto* text_features = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, L, 1);
    ggml_set_name(text_features, "text_features");
    ggml_set_input(text_features);

    auto* neck0 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H0, H0, 1);
    ggml_set_name(neck0, "ref_neck_det_0");
    ggml_set_input(neck0);
    auto* neck1 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H1, H1, 1);
    ggml_set_name(neck1, "ref_neck_det_1");
    ggml_set_input(neck1);
    auto* img_feats = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, H * H, 1);
    ggml_set_name(img_feats, "ref_fenc_img_input");
    ggml_set_input(img_feats);
    auto* img_pe = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, H * H, 1);
    ggml_set_name(img_pe, "ref_fenc_pos_embed");
    ggml_set_input(img_pe);
    auto* img_pe_72_cmp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, D * H * H);
    ggml_set_name(img_pe_72_cmp, "ref_img_pe_72_flat");
    ggml_set_input(img_pe_72_cmp);

    auto* sine_dim_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 64);
    ggml_set_name(sine_dim_t, "sine_dim_t");
    ggml_set_input(sine_dim_t);

    auto* rpb_coords = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, H);
    ggml_set_name(rpb_coords, "rpb_coords");
    ggml_set_input(rpb_coords);

    auto* text_valid_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_valid_mask, "text_valid_mask");
    ggml_set_input(text_valid_mask);

    auto* text_attn_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_attn_bias, "text_attn_bias");
    ggml_set_input(text_attn_bias);

    auto* conditioned = sam3_build_fenc_graph(ctx0, model, img_feats, text_features,
                                              img_pe, text_attn_bias);
    auto* fenc_layer5_out = ggml_cont(ctx0, conditioned);
    ggml_set_name(fenc_layer5_out, "fenc_layer5_out");
    ggml_set_name(conditioned, "fenc_output");

    auto ddec_out = sam3_build_ddec_graph(ctx0, model, conditioned, img_pe, text_features,
                                          sine_dim_t, rpb_coords, text_attn_bias, text_valid_mask);

    struct ggml_tensor* fpn_feats[3] = {neck0, neck1, nullptr};

    auto* obj_queries = ggml_view_3d(ctx0, ddec_out.queries, D, NQ, 1,
                                     ddec_out.queries->nb[1], ddec_out.queries->nb[2],
                                     1 * ddec_out.queries->nb[1]);
    obj_queries = ggml_cont(ctx0, obj_queries);

    auto* mask_logits = sam3_build_seg_head_graph(ctx0, model, conditioned, fpn_feats,
                                                  obj_queries, text_features, text_attn_bias);
    ggml_set_name(mask_logits, "seg_mask_logits");

    struct named_tensor {
        const char* name;
        struct ggml_tensor* tensor;
    };

    const std::vector<named_tensor> direct_tensors = {
        {"text_features", text_features},
        {"text_valid_mask", text_valid_mask},
        {"fenc_img_input", img_feats},
        {"fenc_pos_embed", img_pe},
        {"fenc_prompt", text_features},
        {"img_pe_72", img_pe_72_cmp},
        {"ddec_query_embed", model.ddec.query_embed},
        {"ddec_ref_pts_raw", model.tensors.at("ddec.reference_points.weight")},
        {"ddec_presence_token", model.ddec.presence_token},
        {"ddec_pred_boxes", ddec_out.pred_boxes},
        {"ddec_presence_logit", ddec_out.presence_score},
    };

    std::vector<std::string> named_outputs = {
        "fenc_output",
        "ddec_ref_boxes_init",
        "ddec_query_sine_0",
        "ddec_query_pos_0",
        "ddec_rpb_mask_0",
        "ddec_layer0_after_sa",
        "ddec_layer0_after_text_ca",
        "ddec_layer0_after_img_ca",
        "ddec_layer0_full_out",
        "ddec_layer0_presence",
        "ddec_normed_output",
        "scoring_prompt_mlp_out",
        "scoring_pooled",
        "scoring_proj_pooled",
        "scoring_proj_hs",
        "scoring_class_scores",
        "seg_enc_after_ca",
        "seg_enc_visual",
        "seg_pixel_dec_stage0",
        "seg_pixel_dec_stage1",
        "seg_pixel_decoder_out",
        "seg_instance_embed",
        "seg_mask_embed",
        "seg_mask_logits",
    };

    for (int i = 0; i < hp.fenc_layers; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "fenc_layer%d_out", i);
        named_outputs.emplace_back(name);
    }
    for (int i = 0; i < hp.ddec_layers; ++i) {
        char out_name[64];
        char box_name[64];
        snprintf(out_name, sizeof(out_name), "ddec_layer%d_out", i);
        snprintf(box_name, sizeof(box_name), "ddec_layer%d_refboxes", i);
        named_outputs.emplace_back(out_name);
        named_outputs.emplace_back(box_name);
    }

    ggml_set_output(text_features);
    ggml_set_output(text_valid_mask);
    ggml_set_output(img_feats);
    ggml_set_output(img_pe);
    ggml_set_output(img_pe_72_cmp);
    for (const auto& name : named_outputs) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_set_output(t);
        }
    }
    ggml_set_output(mask_logits);
    ggml_set_output(ddec_out.class_scores);
    ggml_set_output(ddec_out.pred_boxes);
    ggml_set_output(ddec_out.presence_score);

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 65536, false);
    ggml_build_forward_expand(graph, ddec_out.class_scores);
    ggml_build_forward_expand(graph, ddec_out.pred_boxes);
    ggml_build_forward_expand(graph, ddec_out.presence_score);
    ggml_build_forward_expand(graph, mask_logits);
    ggml_build_forward_expand(graph, text_features);
    ggml_build_forward_expand(graph, text_valid_mask);
    ggml_build_forward_expand(graph, img_feats);
    ggml_build_forward_expand(graph, img_pe);
    ggml_build_forward_expand(graph, img_pe_72_cmp);
    for (const auto& name : named_outputs) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_build_forward_expand(graph, t);
        }
    }

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate phase 5 graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(text_features, text_features_data.data(), 0, text_features_data.size() * sizeof(float));
    ggml_backend_tensor_set(neck0, neck_det_0.data(), 0, neck_det_0.size() * sizeof(float));
    ggml_backend_tensor_set(neck1, neck_det_1.data(), 0, neck_det_1.size() * sizeof(float));
    ggml_backend_tensor_set(img_feats, fenc_img_input_data.data(), 0, fenc_img_input_data.size() * sizeof(float));
    ggml_backend_tensor_set(img_pe, fenc_pos_embed_data.data(), 0, fenc_pos_embed_data.size() * sizeof(float));
    ggml_backend_tensor_set(img_pe_72_cmp, img_pe_72_data.data(), 0, img_pe_72_data.size() * sizeof(float));

    {
        std::vector<float> dim_t_data(64);
        for (int i = 0; i < 64; ++i) {
            dim_t_data[i] = 2.0f * (float)M_PI / powf(10000.0f, 2.0f * (float)i / 128.0f);
        }
        ggml_backend_tensor_set(sine_dim_t, dim_t_data.data(), 0, 64 * sizeof(float));
    }

    {
        std::vector<float> coords(H);
        for (int i = 0; i < H; ++i) {
            coords[i] = (float)i / (float)H;
        }
        ggml_backend_tensor_set(rpb_coords, coords.data(), 0, H * sizeof(float));
    }

    auto* qpos_pres = ggml_get_tensor(ctx0, "ddec_query_pos_pres");
    if (qpos_pres) {
        std::vector<float> zeros(D, 0.0f);
        ggml_backend_tensor_set(qpos_pres, zeros.data(), 0, D * sizeof(float));
    }

    auto* rpb_pz = ggml_get_tensor(ctx0, "rpb_pres_zeros");
    if (rpb_pz) {
        int n = (int)(rpb_pz->ne[0] * rpb_pz->ne[1] * rpb_pz->ne[2] * rpb_pz->ne[3]);
        std::vector<float> zeros(n, 0.0f);
        ggml_backend_tensor_set(rpb_pz, zeros.data(), 0, n * sizeof(float));
    }

    {
        int n_valid = 0;
        for (int i = 0; i < L; ++i) {
            if (token_ids[i] != 0) {
                ++n_valid;
            }
        }
        if (n_valid == 0) {
            n_valid = 1;
        }

        const float scale = (float)L / (float)n_valid;
        std::vector<float> valid_mask(L);
        std::vector<float> attn_bias(L);
        for (int i = 0; i < L; ++i) {
            const bool is_valid = token_ids[i] != 0;
            valid_mask[i] = is_valid ? scale : 0.0f;
            attn_bias[i] = is_valid ? 0.0f : -1.0e9f;
        }
        ggml_backend_tensor_set(text_valid_mask, valid_mask.data(), 0, L * sizeof(float));
        ggml_backend_tensor_set(text_attn_bias, attn_bias.data(), 0, L * sizeof(float));
    }

    sam3_graph_compute(model.backend, graph, n_threads);

    bool ok = true;
    for (const auto& item : direct_tensors) {
        if (!sam3_dump_tensor_to_path(item.tensor, item.name, output_dir + "/" + item.name)) {
            ok = false;
        }
    }
    for (const auto& name : named_outputs) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found in graph\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

bool sam3_test_fenc_only(const sam3_model& model,
                         const std::string& ref_dir,
                         const std::string& output_dir,
                         int n_threads) {
    const auto& hp = model.hparams;
    const int D = hp.neck_dim;      // 256
    const int H = hp.n_img_embd();  // 72
    const int N = H * H;            // 5184

    fprintf(stderr, "%s: D=%d H=%d N=%d fenc_layers=%d\n",
            __func__, D, H, N, hp.fenc_layers);

    // ── Load reference tensors from Python dump ──
    // The Python script saves batch-first [1, N, D] tensors.
    // Memory layout: N blocks of D floats = same as ggml [D, N, 1].
    std::vector<float> img_feat_data;
    std::vector<float> pos_data;
    std::vector<float> prompt_data;
    std::vector<float> attn_bias_data;

    if (!sam3_load_ref_f32_data(ref_dir + "/fenc_input_tgt", img_feat_data, D * N)) {
        fprintf(stderr, "%s: failed to load fenc_input_tgt\n", __func__);
        return false;
    }
    if (!sam3_load_ref_f32_data(ref_dir + "/fenc_input_pos", pos_data, D * N)) {
        fprintf(stderr, "%s: failed to load fenc_input_pos\n", __func__);
        return false;
    }
    if (!sam3_load_ref_f32_data(ref_dir + "/fenc_input_prompt", prompt_data)) {
        fprintf(stderr, "%s: failed to load fenc_input_prompt\n", __func__);
        return false;
    }

    // Determine prompt length from loaded data
    const int T = (int)prompt_data.size() / D;
    fprintf(stderr, "%s: prompt tokens T=%d\n", __func__, T);

    // Attn bias: [1, T] float (0.0 valid, -1e9 padding)
    // If not present, assume all valid (no bias)
    bool have_attn_bias = sam3_load_ref_f32_data(ref_dir + "/fenc_attn_bias", attn_bias_data, T);
    if (!have_attn_bias) {
        fprintf(stderr, "%s: no fenc_attn_bias found, assuming all tokens valid\n", __func__);
        attn_bias_data.assign(T, 0.0f);
    }

    // ── Build fenc-only graph ──
    const size_t buf_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    // Create input tensors (ggml layout: [D, N, 1])
    auto* img_feats = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, N, 1);
    ggml_set_name(img_feats, "fenc_img_input");
    ggml_set_input(img_feats);

    auto* pos_enc = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, N, 1);
    ggml_set_name(pos_enc, "fenc_pos_input");
    ggml_set_input(pos_enc);

    auto* prompt_tokens = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, T, 1);
    ggml_set_name(prompt_tokens, "fenc_prompt_input");
    ggml_set_input(prompt_tokens);

    auto* text_attn_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, T, 1, 1);
    ggml_set_name(text_attn_bias, "fenc_attn_bias");
    ggml_set_input(text_attn_bias);

    // Build fusion encoder graph
    auto* conditioned = sam3_build_fenc_graph(ctx0, model, img_feats, prompt_tokens,
                                              pos_enc, text_attn_bias);
    // sam3_build_fenc_graph names the last layer fenc_layer5_out.  ggml_set_name
    // will overwrite it, so create a cont copy so both names resolve.
    auto* fenc_last = ggml_cont(ctx0, conditioned);
    ggml_set_name(fenc_last, "fenc_layer5_out");
    ggml_set_name(conditioned, "fenc_output");

    // Mark all per-layer outputs as graph outputs for extraction
    std::vector<std::string> output_names = {"fenc_output"};
    for (int i = 0; i < hp.fenc_layers; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "fenc_layer%d_out", i);
        output_names.emplace_back(name);
    }

    ggml_set_output(img_feats);
    ggml_set_output(pos_enc);
    ggml_set_output(prompt_tokens);
    for (const auto& name : output_names) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_set_output(t);
        }
    }

    // Build and allocate graph
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(graph, conditioned);
    ggml_build_forward_expand(graph, fenc_last);
    for (const auto& name : output_names) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_build_forward_expand(graph, t);
        }
    }

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate fenc graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    // ── Set input data ──
    ggml_backend_tensor_set(img_feats, img_feat_data.data(), 0,
                            img_feat_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos_enc, pos_data.data(), 0,
                            pos_data.size() * sizeof(float));
    ggml_backend_tensor_set(prompt_tokens, prompt_data.data(), 0,
                            prompt_data.size() * sizeof(float));
    ggml_backend_tensor_set(text_attn_bias, attn_bias_data.data(), 0,
                            attn_bias_data.size() * sizeof(float));

    // ── Compute ──
    fprintf(stderr, "%s: computing fenc graph (%d nodes)...\n", __func__, ggml_graph_n_nodes(graph));
    sam3_graph_compute(model.backend, graph, n_threads);

    // ── Dump outputs ──
    bool ok = true;
    for (const auto& name : output_names) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    // Also dump inputs for verification
    sam3_dump_tensor_to_path(img_feats, "fenc_img_input", output_dir + "/fenc_img_input");
    sam3_dump_tensor_to_path(pos_enc, "fenc_pos_input", output_dir + "/fenc_pos_input");
    sam3_dump_tensor_to_path(prompt_tokens, "fenc_prompt_input", output_dir + "/fenc_prompt_input");

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

static bool sam3_test_dump_phase6_impl(const sam3_model& model,
                                       struct ggml_tensor* state_neck0,
                                       struct ggml_tensor* state_neck1,
                                       struct ggml_tensor* state_neck2,
                                       const std::vector<float>* neck0_data,
                                       const std::vector<float>* neck1_data,
                                       const std::vector<float>* neck2_data,
                                       const sam3_pvs_params& params,
                                       const std::string& output_dir,
                                       int n_threads) {
    const auto& hp = model.hparams;
    const int D = hp.sam_embed_dim;
    const int H = hp.n_img_embd();
    const int H1 = H * 2;
    const int H0 = H * 4;

    const size_t buf_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    struct ggml_tensor* neck0 = state_neck0;
    struct ggml_tensor* neck1 = state_neck1;
    struct ggml_tensor* neck2 = state_neck2;
    if (neck0_data && neck1_data && neck2_data) {
        neck0 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H0, H0, 1);
        ggml_set_name(neck0, "sam_dec_feat_s0_input");
        ggml_set_input(neck0);

        neck1 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H1, H1, 1);
        ggml_set_name(neck1, "sam_dec_feat_s1_input");
        ggml_set_input(neck1);

        neck2 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H, H, 1);
        ggml_set_name(neck2, "sam_dec_feat_s2_input");
        ggml_set_input(neck2);
    }

    if (!neck0 || !neck1 || !neck2) {
        fprintf(stderr, "%s: missing tracker feature inputs\n", __func__);
        ggml_free(ctx0);
        return false;
    }

    auto pe_out = sam3_build_sam_pe(ctx0, params, D, H);

    auto* no_mem = ggml_reshape_4d(ctx0, model.tensors.at("no_mem_embed"), D, 1, 1, 1);
    auto* image_feats = ggml_add(ctx0, neck2, no_mem);
    ggml_set_name(image_feats, "sam_dec_image_feats");

    auto dec_out = sam3_build_sam_dec_graph(ctx0, model, image_feats,
                                            pe_out.image_pe, pe_out.sparse, pe_out.dense,
                                            neck0, neck1);

    struct named_tensor {
        const char* name;
        struct ggml_tensor* tensor;
    };

    const std::vector<named_tensor> direct_tensors = {
        {"sam_pe_sparse", pe_out.sparse},
        {"sam_pe_dense", pe_out.dense},
        {"sam_pe_image_pe", pe_out.image_pe},
        {"sam_dec_image_feats", image_feats},
        {"sam_dec_sam_token", dec_out.sam_token},
    };

    std::vector<std::string> named_outputs = {
        "sam_dec_tokens_initial",
        "sam_dec_block0_queries",
        "sam_dec_block0_keys",
        "sam_dec_block1_queries",
        "sam_dec_block1_keys",
        "sam_dec_final_queries",
        "sam_dec_feat_s1_proj",
        "sam_dec_feat_s0_proj",
        "sam_dec_upscaled",
        "sam_dec_mask_tokens",
        "sam_dec_masks",
        "sam_dec_iou",
        "sam_dec_obj_score",
    };

    ggml_set_output(dec_out.masks);
    ggml_set_output(dec_out.iou_pred);
    ggml_set_output(dec_out.obj_score);
    ggml_set_output(dec_out.sam_token);
    ggml_set_output(dec_out.mask_tokens);
    for (const auto& item : direct_tensors) {
        ggml_set_output(item.tensor);
    }
    for (const auto& name : named_outputs) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found in graph\n", __func__, name.c_str());
            ggml_free(ctx0);
            return false;
        }
        ggml_set_output(t);
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 32768, false);
    ggml_build_forward_expand(graph, dec_out.masks);
    ggml_build_forward_expand(graph, dec_out.iou_pred);
    ggml_build_forward_expand(graph, dec_out.obj_score);
    ggml_build_forward_expand(graph, dec_out.sam_token);
    ggml_build_forward_expand(graph, dec_out.mask_tokens);
    for (const auto& item : direct_tensors) {
        ggml_build_forward_expand(graph, item.tensor);
    }
    for (const auto& name : named_outputs) {
        ggml_build_forward_expand(graph, ggml_get_tensor(ctx0, name.c_str()));
    }

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate phase 6 graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    if (neck0_data && neck1_data && neck2_data) {
        ggml_backend_tensor_set(neck0, neck0_data->data(), 0, neck0_data->size() * sizeof(float));
        ggml_backend_tensor_set(neck1, neck1_data->data(), 0, neck1_data->size() * sizeof(float));
        ggml_backend_tensor_set(neck2, neck2_data->data(), 0, neck2_data->size() * sizeof(float));
    }

    sam3_state pe_cache_state = {};
    sam3_populate_pe_cache(pe_cache_state, model);

    std::vector<float> all_coords;
    std::vector<int> all_labels;
    sam3_collect_pvs_prompt_tokens(params, all_coords, all_labels);
    if ((int)all_labels.size() != pe_out.n_tokens) {
        fprintf(stderr, "%s: prompt token count mismatch (%zu vs %d)\n",
                __func__, all_labels.size(), pe_out.n_tokens);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    {
        const int num_pos_feats = D / 2;
        std::vector<float> sparse_data((size_t)pe_out.n_tokens * D, 0.0f);
        for (int p = 0; p < pe_out.n_tokens; ++p) {
            float px = all_coords[p * 2 + 0] + 0.5f;
            float py = all_coords[p * 2 + 1] + 0.5f;
            float x_norm = px / (float)hp.img_size;
            float y_norm = py / (float)hp.img_size;
            float pe_vec[256];
            sam3_pe_encode_coord(pe_vec, x_norm, y_norm,
                                 pe_cache_state.pe_gauss_cache.data(), num_pos_feats);

            const int label = all_labels[p];
            if (label == -1) {
                for (int d = 0; d < D; ++d) {
                    sparse_data[(size_t)p * D + d] = pe_cache_state.not_a_point_cache[d];
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    sparse_data[(size_t)p * D + d] = pe_vec[d] + pe_cache_state.point_emb_cache[label][d];
                }
            }
        }

        ggml_backend_tensor_set(pe_out.sparse, sparse_data.data(), 0,
                                sparse_data.size() * sizeof(float));
        ggml_backend_tensor_set(pe_out.image_pe, pe_cache_state.dense_pe_cache.data(), 0,
                                pe_cache_state.dense_pe_cache.size() * sizeof(float));
        ggml_backend_tensor_set(pe_out.dense, pe_cache_state.dense_nomask_cache.data(), 0,
                                pe_cache_state.dense_nomask_cache.size() * sizeof(float));
    }

    sam3_graph_compute(model.backend, graph, n_threads);

    bool ok = true;
    for (const auto& item : direct_tensors) {
        if (!sam3_dump_tensor_to_path(item.tensor, item.name, output_dir + "/" + item.name)) {
            ok = false;
        }
    }
    for (const auto& name : named_outputs) {
        auto* t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found after compute\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

bool sam3_test_dump_phase6(const sam3_model& model,
                           const sam3_state& state,
                           const sam3_pvs_params& params,
                           const std::string& output_dir,
                           int n_threads) {
    return sam3_test_dump_phase6_impl(model,
                                      state.neck_trk[0], state.neck_trk[1], state.neck_trk[2],
                                      nullptr, nullptr, nullptr,
                                      params, output_dir, n_threads);
}

bool sam3_test_dump_phase6_from_ref_inputs(const sam3_model& model,
                                           const std::string& prephase_ref_dir,
                                           const sam3_pvs_params& params,
                                           const std::string& output_dir,
                                           int n_threads) {
    const auto& hp = model.hparams;
    const int D = hp.sam_embed_dim;
    const int H = hp.n_img_embd();
    const int H1 = H * 2;
    const int H0 = H * 4;

    std::vector<float> neck_trk_0;
    std::vector<float> neck_trk_1;
    std::vector<float> neck_trk_2;
    if (!sam3_load_ref_f32_data(prephase_ref_dir + "/neck_trk_0", neck_trk_0, D * H0 * H0) ||
        !sam3_load_ref_f32_data(prephase_ref_dir + "/neck_trk_1", neck_trk_1, D * H1 * H1) ||
        !sam3_load_ref_f32_data(prephase_ref_dir + "/neck_trk_2", neck_trk_2, D * H * H)) {
        return false;
    }

    neck_trk_0 = sam3_reorder_nchw_to_ggml_dwh(neck_trk_0, D, H0, H0);
    neck_trk_1 = sam3_reorder_nchw_to_ggml_dwh(neck_trk_1, D, H1, H1);
    neck_trk_2 = sam3_reorder_nchw_to_ggml_dwh(neck_trk_2, D, H, H);

    return sam3_test_dump_phase6_impl(model,
                                      nullptr, nullptr, nullptr,
                                      &neck_trk_0, &neck_trk_1, &neck_trk_2,
                                      params, output_dir, n_threads);
}

/*****************************************************************************
** Test: Geometry encoder dump
*****************************************************************************/

bool sam3_test_dump_geom_enc(const sam3_model& model,
                             const std::string& prephase_ref_dir,
                             const sam3_pcs_params& params,
                             const std::string& output_dir,
                             int n_threads) {
    const auto& hp = model.hparams;
    const int D = hp.neck_dim;      // 256
    const int H = hp.n_img_embd();  // 72

    // Load backbone features from Phase 3 reference (NCHW format)
    std::vector<float> neck_det_2;
    if (!sam3_load_ref_f32_data(prephase_ref_dir + "/neck_det_2", neck_det_2, D * H * H)) {
        fprintf(stderr, "%s: failed to load neck_det_2\n", __func__);
        return false;
    }
    // Reorder from NCHW to ggml [D, W, H] layout
    auto neck_det_2_ggml = sam3_reorder_nchw_to_ggml_dwh(neck_det_2, D, H, H);

    // Compute sinusoidal PE for image features (matching Python PositionEmbeddingSine)
    std::vector<float> img_pe_nchw(D * H * H);
    {
        const float scale = 2.0f * (float)M_PI;
        const float eps = 1e-6f;
        const int num_pos_feats = D / 2;  // 128

        for (int row = 0; row < H; ++row) {
            for (int col = 0; col < H; ++col) {
                float y_embed = ((float)(row + 1) - 0.5f) / ((float)H + eps) * scale;
                float x_embed = ((float)(col + 1) - 0.5f) / ((float)H + eps) * scale;

                for (int i = 0; i < num_pos_feats; ++i) {
                    int div_idx = 2 * (i / 2);
                    float dim_t = powf(10000.0f, (float)div_idx / (float)num_pos_feats);

                    float px = x_embed / dim_t;
                    float py = y_embed / dim_t;

                    float pe_y, pe_x;
                    if (i % 2 == 0) {
                        pe_y = sinf(py);
                        pe_x = sinf(px);
                    } else {
                        pe_y = cosf(py);
                        pe_x = cosf(px);
                    }
                    // Output: [1, D, H, W] NCHW — pos_y first half, pos_x second half
                    img_pe_nchw[i * H * H + row * H + col] = pe_y;
                    img_pe_nchw[(num_pos_feats + i) * H * H + row * H + col] = pe_x;
                }
            }
        }
    }
    auto img_pe_ggml = sam3_reorder_nchw_to_ggml_dwh(img_pe_nchw, D, H, H);

    // ── Build graph ─────────────────────────────────────────────────────
    const size_t buf_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init context\n", __func__);
        return false;
    }

    // Image features as input tensors (from reference data)
    auto* img_feats_t = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, H * H, 1);
    ggml_set_name(img_feats_t, "img_feats_input");
    ggml_set_input(img_feats_t);

    auto* img_pe_t = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, H * H, 1);
    ggml_set_name(img_pe_t, "img_pe_input");
    ggml_set_input(img_pe_t);

    // Build geometry encoder graph
    auto geom_out = sam3_build_geom_enc_graph(ctx0, model, params, img_feats_t, img_pe_t);

    // Build and allocate graph
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(graph, geom_out.geo_feats);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    // Upload image features
    ggml_backend_tensor_set(img_feats_t, neck_det_2_ggml.data(), 0, D * H * H * sizeof(float));
    ggml_backend_tensor_set(img_pe_t, img_pe_ggml.data(), 0, D * H * H * sizeof(float));

    // Pre-compute geometry input on CPU and upload
    auto geom_data = sam3_precompute_geom_input(model, params, neck_det_2_ggml.data(), H, H);
    auto* gi = ggml_get_tensor(ctx0, "geom_post_final_proj");
    if (gi) {
        ggml_backend_tensor_set(gi, geom_data.data(), 0, geom_data.size() * sizeof(float));
    }

    // Compute
    sam3_graph_compute(model.backend, graph, n_threads);

    // Dump outputs
    mkdir(output_dir.c_str(), 0755);

    // Dump the pre-computed input (after final_proj + norm, before transformer)
    {
        auto* pre_proj = ggml_get_tensor(ctx0, "geom_post_final_proj");
        if (pre_proj) sam3_dump_tensor_to_path(pre_proj, "post_final_proj", output_dir + "/post_final_proj");
    }

    // Dump final output
    {
        auto* out = ggml_get_tensor(ctx0, "geom_output");
        if (out) sam3_dump_tensor_to_path(out, "geom_output", output_dir + "/geom_output");
    }

    // Also dump the pre-computed input data (before ggml processing)
    {
        const int N_geo = geom_out.n_tokens;
        std::string path = output_dir + "/geom_input_precomputed";
        {
            std::ofstream f(path + ".bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(geom_data.data()), geom_data.size() * sizeof(float));
        }
        {
            std::ofstream f(path + ".shape");
            f << D << "," << N_geo;
        }
        fprintf(stderr, "%s: dumped geom_input_precomputed [%d, %d]\n", __func__, D, N_geo);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return true;
}

/*****************************************************************************
** Debug: dump state tensors
*****************************************************************************/

static struct ggml_tensor * sam3_find_state_tensor(const sam3_state & state,
                                                   const std::string & tensor_name) {
    struct ggml_tensor* t = nullptr;

    if (tensor_name == "vit_output") {
        t = state.vit_output;
    } else if (tensor_name == "neck_det_0") {
        t = state.neck_det[0];
    } else if (tensor_name == "neck_det_1") {
        t = state.neck_det[1];
    } else if (tensor_name == "neck_det_2") {
        t = state.neck_det[2];
    } else if (tensor_name == "neck_det_3") {
        t = state.neck_det[3];
    } else if (tensor_name == "neck_trk_0") {
        t = state.neck_trk[0];
    } else if (tensor_name == "neck_trk_1") {
        t = state.neck_trk[1];
    } else if (tensor_name == "neck_trk_2") {
        t = state.neck_trk[2];
    } else if (tensor_name == "neck_trk_3") {
        t = state.neck_trk[3];
    } else if (tensor_name == "neck_det_pe_0") {
        t = state.neck_det_pe[0];
    } else if (tensor_name == "neck_det_pe_1") {
        t = state.neck_det_pe[1];
    } else if (tensor_name == "neck_det_pe_2") {
        t = state.neck_det_pe[2];
    } else if (tensor_name == "neck_det_pe_3") {
        t = state.neck_det_pe[3];
    } else {
        // Search by ggml name in the context
        if (state.ctx) {
            t = ggml_get_tensor(state.ctx, tensor_name.c_str());
        }
        // Also search PE context
        if (!t && state.pe_ctx) {
            t = ggml_get_tensor(state.pe_ctx, tensor_name.c_str());
        }
    }

    return t;
}

static bool sam3_fill_tensor_info(struct ggml_tensor * t, sam3_tensor_info & info) {
    if (!t) {
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        info.ne[i] = t->ne[i];
        info.nb[i] = t->nb[i];
    }
    info.type = (int) t->type;
    info.op = (int) t->op;
    info.is_contiguous = ggml_is_contiguous(t);
    return true;
}

bool sam3_get_state_tensor_info(const sam3_state & state,
                                const std::string & tensor_name,
                                sam3_tensor_info & info) {
    struct ggml_tensor * t = sam3_find_state_tensor(state, tensor_name);
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found in state\n", __func__, tensor_name.c_str());
        return false;
    }
    return sam3_fill_tensor_info(t, info);
}

bool sam3_dump_state_tensor(const sam3_state& state,
                            const std::string& tensor_name,
                            const std::string& output_path) {
    struct ggml_tensor * t = sam3_find_state_tensor(state, tensor_name);
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found in state\n", __func__, tensor_name.c_str());
        return false;
    }
    return sam3_dump_tensor_to_path(t, tensor_name, output_path);
}

bool sam3_get_model_tensor_info(const sam3_model & model,
                                const std::string & tensor_name,
                                sam3_tensor_info & info) {
    auto it = model.tensors.find(tensor_name);
    if (it == model.tensors.end() || !it->second) {
        fprintf(stderr, "%s: tensor '%s' not found in model\n", __func__, tensor_name.c_str());
        return false;
    }
    return sam3_fill_tensor_info(it->second, info);
}

bool sam3_dump_model_tensor(const sam3_model & model,
                            const std::string & tensor_name,
                            const std::string & output_path) {
    auto it = model.tensors.find(tensor_name);
    if (it == model.tensors.end() || !it->second) {
        fprintf(stderr, "%s: tensor '%s' not found in model\n", __func__, tensor_name.c_str());
        return false;
    }
    return sam3_dump_tensor_to_path(it->second, tensor_name, output_path);
}
