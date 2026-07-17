// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** Image backbone — public API
*****************************************************************************/

void sam3_set_encoder_fp8(sam3_state& state, bool fp8) {
#ifdef SAM3_TRT_ENCODER
    state.trt_encoder_fp8 = fp8;
#else
    (void)state;
    if (fp8) fprintf(stderr, "%s: FP8 encoder needs a SAM3_TRT_ENCODER build -- ignored\n", __func__);
#endif
}

void sam3_set_pcs_fp8(sam3_state& state, bool fp8) {
#ifdef SAM3_TRT_ENCODER
    state.trt_pcs_fp8 = fp8;
#else
    (void)state;
    if (fp8) fprintf(stderr, "%s: FP8 PCS needs a SAM3_TRT_ENCODER build -- ignored\n", __func__);
#endif
}

bool sam3_encode_image(sam3_state& state,
                       const sam3_model& model,
                       const sam3_image& image) {
#if SAM3_LOG_LEVEL >= 1
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    const auto& hp = model.hparams;
    const int img_size = sam3_eff_img_size(state, hp);

    SAM3_LOG(2, "%s: encoding %dx%d image → %dx%d\n", __func__,
             image.width, image.height, img_size, img_size);

    state.orig_width = image.width;
    state.orig_height = image.height;

    // DIAGNOSTIC ONLY: SAM3_STAGE_TIMING=1 also times preprocessing and
    // graph-build+reserve+alloc separately from the compute call below.
    static const bool stage_timing_enc = getenv("SAM3_STAGE_TIMING") != nullptr;
    std::chrono::high_resolution_clock::time_point t_pre0;
    if (stage_timing_enc) t_pre0 = std::chrono::high_resolution_clock::now();

    auto img_data = sam3_preprocess_image(image, img_size);

    if (stage_timing_enc) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_pre0).count();
        fprintf(stderr, "[stage_timing] preprocess_image: %.2f ms\n", ms);
    }

#ifdef SAM3_TRT_ENCODER
    if (sam3_try_trt_encode_image(state, model, img_data.data(), img_size)) {
        return true;
    }
#endif
#if defined(SAM3_TRT_ENCODER) && defined(SAM3_TRT_ONLY)
    // This build only supports the TRT-covered encoder scope (native
    // 1008x1008 resolution -- see sam3_try_trt_encode_image) -- no ggml ViT
    // fallback graph is compiled in, so a resolution mismatch or a TRT
    // failure is a hard error here, not a silent (slow but correct)
    // degrade. Also out of scope in this mode: video tracking/propagate,
    // which independently needs this ggml path's neck-PE-cache population
    // and has no substitute here. Rebuild without SAM3_TRT_ONLY for that.
    fprintf(stderr, "%s: TRT encode unavailable/out of scope and this build is "
                    "TRT-only (SAM3_TRT_ONLY) -- no ggml fallback compiled in\n", __func__);
    return false;
#else
    if (model.trt_only_weights) {
        fprintf(stderr, "%s: TRT encode unavailable/out of scope but the model was loaded "
                        "with SAM3_TRT_SKIP_GGML_WEIGHTS=1 -- ggml weights absent, refusing "
                        "to run the fallback graph\n", __func__);
        return false;
    }

    std::chrono::high_resolution_clock::time_point t_build0;
    if (stage_timing_enc) t_build0 = std::chrono::high_resolution_clock::now();

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

    SAM3_LOG(2, "%s: graph allocated, %d nodes\n", __func__, ggml_graph_n_nodes(graph));
    if (stage_timing_enc) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_build0).count();
        fprintf(stderr, "[stage_timing] graph build+reserve+alloc: %.2f ms (%d nodes)\n",
                ms, ggml_graph_n_nodes(graph));
    }

    ggml_backend_tensor_set(inp, img_data.data(), 0, img_data.size() * sizeof(float));

    {
#if SAM3_LOG_LEVEL >= 1
        auto t0 = std::chrono::high_resolution_clock::now();
#endif
        if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx0);
            return false;
        }
#if SAM3_LOG_LEVEL >= 1
        auto t1 = std::chrono::high_resolution_clock::now();
        double compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        SAM3_LOG(1, "%s: graph computed in %.1f ms (%d threads)\n",
                 __func__, compute_ms, state.n_threads);
#endif
    }

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

    // PEs live in a separate buffer so they survive gallocr teardown.
    // sam3_sinusoidal_pe_2d is a pure function of (scale sizes, neck_dim) --
    // both derived from fixed hyperparameters -- so this whole block is
    // skipped once already built for the current effective feature size,
    // instead of recomputing ~28M sin/cos evaluations (measured ~179ms,
    // docs/sam3/PLAN.md Phase 3) on every single encode call.
    std::chrono::high_resolution_clock::time_point t_pe0;
    if (stage_timing_enc) t_pe0 = std::chrono::high_resolution_clock::now();
    if (!state.neck_pe_valid || state.neck_pe_feat_size != hp.n_img_embd()) {
        const int neck_dim = hp.neck_dim;  // 256
        const int scale_sizes[4] = {
            hp.n_img_embd() * 4,  // 288
            hp.n_img_embd() * 2,  // 144
            hp.n_img_embd(),      //  72
            hp.n_img_embd() / 2,  //  36
        };

        size_t pe_total = 0;
        for (int i = 0; i < 4; ++i) {
            pe_total += (size_t)neck_dim * scale_sizes[i] * scale_sizes[i] * sizeof(float);
        }

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
                // Tracker shares the same spatial dimensions → same PE
                state.neck_trk_pe[i] = pe_tensors[i];
            }
            state.neck_pe_valid = true;
            state.neck_pe_feat_size = hp.n_img_embd();
        }
    }

    if (stage_timing_enc) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_pe0).count();
        fprintf(stderr, "[stage_timing] neck PE rebuild (sam3_sinusoidal_pe_2d x4): %.2f ms\n", ms);
    }

    // Invalidate PE cache only if the effective feat_size actually changed --
    // sam3_populate_pe_cache is a pure function of it (plus fixed model
    // weights), so recomputing on every encode when it never changes wastes
    // ~28M sin/cos evals worth of work in the next PVS/PCS call for nothing.
    if (sam3_eff_feat_size(state, hp) != state.pe_cache_feat_size) {
        state.pe_cache_valid = false;
    }

#if SAM3_LOG_LEVEL >= 1
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    SAM3_LOG(1, "%s: image encoded successfully in %.1f ms\n", __func__, total_ms);
#endif
    return true;
#endif  // !(SAM3_TRT_ENCODER && SAM3_TRT_ONLY)
}

void sam3_clear_encoder_state(sam3_state & state) {
    state.vit_output = nullptr;
    for (int i = 0; i < 4; ++i) {
        state.neck_det[i] = nullptr;
        state.neck_trk[i] = nullptr;
        state.neck_det_pe[i] = nullptr;
        state.neck_trk_pe[i] = nullptr;
    }
}

bool sam3_mark_named_outputs(struct ggml_context * ctx,
                                    const std::vector<std::string> & output_tensors) {
    for (const auto & name : output_tensors) {
        struct ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: requested tensor '%s' was not found\n", __func__, name.c_str());
            return false;
        }
        ggml_set_output(t);
    }
    return true;
}

// See sam3.h. Runs sam3_precompute_geom_input for a single positive/negative
// exemplar box against the currently encoded image's backbone features and
// returns that box's post-proj/post-norm geometry row -- the persistable
// "concept embedding".
std::vector<float> sam3_pcs_compute_exemplar_embedding(sam3_state& state,
                                                       const sam3_model& model,
                                                       const sam3_box& box_normalized,
                                                       bool positive) {
    const auto& hp = model.hparams;
    const int D = hp.neck_dim;
    const int H = hp.n_img_embd();
    if (!state.neck_det[2]) {
        fprintf(stderr, "%s: image not encoded -- call sam3_encode_image on the reference image first\n",
                __func__);
        return {};
    }
    std::vector<float> feats(D * H * H);
    ggml_backend_tensor_get(state.neck_det[2], feats.data(), 0, feats.size() * sizeof(float));

    sam3_pcs_params p;
    if (positive) p.pos_exemplars.push_back(box_normalized);
    else          p.neg_exemplars.push_back(box_normalized);
    auto rows = sam3_precompute_geom_input(model, p, feats.data(), H, H);
    // rows = [box row, CLS row]; the box row is the embedding.
    return std::vector<float>(rows.begin(), rows.begin() + D);
}

/*****************************************************************************
** Image segmentation — PCS (text-prompted)
*****************************************************************************/

sam3_result sam3_segment_pcs(sam3_state& state,
                             const sam3_model& model,
                             const sam3_pcs_params& params) {
    if (model.hparams.visual_only) {
        fprintf(stderr, "%s: ERROR: PCS not available on visual-only model\n", __func__);
        return sam3_result{};
    }

#if SAM3_LOG_LEVEL >= 1
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    const auto& hp = model.hparams;
    const int D = hp.neck_dim;           // 256
    const int H = hp.n_img_embd();       // 72
    const int L = hp.text_ctx_len;       // 32
    const int NQ = hp.ddec_num_queries;  // 200
    const int N_spatial = H * H;         // 5184
    sam3_result result;

    // ── Check that image has been encoded ────────────────────────────────
    if (!state.neck_det[0]) {
        fprintf(stderr, "%s: image not encoded — call sam3_encode_image first\n", __func__);
        return result;
    }

    // ── Tokenize text prompt ─────────────────────────────────────────────
    auto token_ids = sam3_tokenize(const_cast<sam3_bpe_tokenizer&>(model.tokenizer),
                                   params.text_prompt, L);
    if (token_ids.empty()) {
        fprintf(stderr, "%s: failed to tokenize text prompt\n", __func__);
        return result;
    }

    SAM3_LOG(2, "%s: text='%s', %zu tokens\n", __func__,
             params.text_prompt.c_str(), token_ids.size());

#ifdef SAM3_TRT_ENCODER
    {
        sam3_result trt_result;
        if (sam3_try_trt_segment_pcs(state, model, params, token_ids, trt_result)) {
            return trt_result;
        }
    }
#endif
#if defined(SAM3_TRT_ENCODER) && defined(SAM3_TRT_ONLY)
    // This build only supports the TRT-covered PCS scope (see
    // sam3_try_trt_segment_pcs's doc: zero exemplar boxes) -- no ggml
    // fallback graph is compiled in, so an out-of-scope prompt or a TRT
    // failure is a hard error here, not a silent (slow but correct)
    // degrade. Rebuild without SAM3_TRT_ONLY for exemplar-box support.
    fprintf(stderr, "%s: TRT PCS unavailable/out of scope and this build is "
                    "TRT-only (SAM3_TRT_ONLY) -- no ggml fallback compiled in\n", __func__);
    return result;
#else
    if (model.trt_only_weights) {
        fprintf(stderr, "%s: TRT PCS unavailable/out of scope but the model was loaded "
                        "with SAM3_TRT_SKIP_GGML_WEIGHTS=1 -- ggml weights absent, refusing "
                        "to run the fallback graph\n", __func__);
        return result;
    }

    // ── Helper: run a sub-graph with its own context and allocator ──────
    // Each stage below follows this exact pattern:
    //   1. Create fresh ggml_context + graph + allocator
    //   2. Create INPUT tensors (ggml_set_input)
    //   3. Build the computation graph
    //   4. Mark outputs (ggml_set_output)
    //   5. Allocate → set input data → compute → read output data
    //   6. Free allocator + context
    // This ensures ZERO buffer sharing between stages.

    // ── Pre-compute shared CPU data used by multiple stages ─────────────
    const int n_boxes = (int)(params.pos_exemplars.size() + params.neg_exemplars.size());
    const int N_geo = n_boxes + (int)params.exemplar_embeddings.size() + 1;  // +1 for CLS
    const int T = L + N_geo;        // total prompt tokens (text + geometry)

    // Read image features and PE from state into CPU buffers (used by multiple stages)
    std::vector<float> img_feats_cpu(D * N_spatial);
    std::vector<float> img_pe_cpu(D * N_spatial);
    ggml_backend_tensor_get(state.neck_det[2], img_feats_cpu.data(), 0,
                            D * N_spatial * sizeof(float));
    ggml_backend_tensor_get(state.neck_det_pe[2], img_pe_cpu.data(), 0,
                            D * N_spatial * sizeof(float));

    // Pre-compute constant input vectors
    std::vector<float> sine_dim_t_cpu(64);
    for (int i = 0; i < 64; ++i)
        sine_dim_t_cpu[i] = 2.0f * (float)M_PI / powf(10000.0f, 2.0f * (float)i / 128.0f);

    std::vector<float> rpb_coords_cpu(H);
    for (int i = 0; i < H; ++i) rpb_coords_cpu[i] = (float)i / (float)H;

    // Build combined attention bias for the prompt: [T] = [L + N_geo]
    // 0.0 for valid tokens, -1e9 for padding text tokens, 0.0 for geo tokens
    std::vector<float> combined_bias_cpu(T);
    for (int i = 0; i < L; ++i)
        combined_bias_cpu[i] = (token_ids[i] != 0) ? 0.0f : -1.0e9f;
    for (int i = 0; i < N_geo; ++i)
        combined_bias_cpu[L + i] = 0.0f;

    // Build text validity mask for DotProductScoring: [T]
    int n_valid_tokens = 0;
    for (int i = 0; i < L; ++i)
        if (token_ids[i] != 0) ++n_valid_tokens;
    n_valid_tokens += N_geo;
    if (n_valid_tokens == 0) n_valid_tokens = 1;
    float tvm_scale = (float)T / (float)n_valid_tokens;
    std::vector<float> text_valid_mask_cpu(T);
    for (int i = 0; i < L; ++i)
        text_valid_mask_cpu[i] = (token_ids[i] != 0) ? tvm_scale : 0.0f;
    for (int i = 0; i < N_geo; ++i)
        text_valid_mask_cpu[L + i] = tvm_scale;

    /*
    ** ── SUB-GRAPH 1: Text Encoder ────────────────────────────────────
    */
    std::vector<float> text_feats_cpu(D * L);
    {
        const size_t sz = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() * 2;
        struct ggml_init_params gp = {sz, nullptr, true};
        auto* ctx = ggml_init(gp);

        auto* inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
        ggml_set_name(inp, "text_token_ids");
        ggml_set_input(inp);

        auto* out = sam3_build_text_encoder_graph(ctx, inp, model);
        ggml_set_output(out);

        auto* graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, out);

        auto* causal = ggml_get_tensor(ctx, "causal_mask");
        auto* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
            fprintf(stderr, "%s: text encoder alloc failed\n", __func__);
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }

        if (inp->buffer) {
            ggml_backend_tensor_set(inp, token_ids.data(), 0, L * sizeof(int32_t));
        } else {
            fprintf(stderr, "%s: ERROR: text encoder input tensor has no buffer!\n", __func__);
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }
        if (causal && causal->buffer) {
            std::vector<ggml_fp16_t> cm(L * L);
            sam3_fill_causal_mask(cm.data(), L);
            ggml_backend_tensor_set(causal, cm.data(), 0, L * L * sizeof(ggml_fp16_t));
        }

        if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }
        ggml_backend_tensor_get(out, text_feats_cpu.data(), 0, D * L * sizeof(float));
        sam3_debug_dump_vec("s1_text_feats", text_feats_cpu.data(), text_feats_cpu.size());

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    SAM3_LOG(2, "%s: text encoder done\n", __func__);

    /*
    ** ── SUB-GRAPH 2: Geometry Encoder ────────────────────────────────
    */
    std::vector<float> geo_feats_cpu(D * N_geo);
    {
        const size_t sz = ggml_tensor_overhead() * 4096 + ggml_graph_overhead() * 2;
        struct ggml_init_params gp = {sz, nullptr, true};
        auto* ctx = ggml_init(gp);

        auto* g_img = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_spatial, 1);
        ggml_set_name(g_img, "geo_img");
        ggml_set_input(g_img);
        auto* g_pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_spatial, 1);
        ggml_set_name(g_pe, "geo_pe");
        ggml_set_input(g_pe);

        auto gr = sam3_build_geom_enc_graph(ctx, model, params, g_img, g_pe);
        ggml_set_output(gr.geo_feats);

        auto* graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph, gr.geo_feats);

        auto* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
            fprintf(stderr, "%s: geometry encoder alloc failed\n", __func__);
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }

        ggml_backend_tensor_set(g_img, img_feats_cpu.data(), 0, D * N_spatial * sizeof(float));
        ggml_backend_tensor_set(g_pe, img_pe_cpu.data(), 0, D * N_spatial * sizeof(float));

        // Pre-transformer input (CLS + optional box embeddings)
        {
            const float* feats_ptr = n_boxes > 0 ? img_feats_cpu.data() : nullptr;
            auto geom_data = sam3_precompute_geom_input(model, params, feats_ptr, H, H);
            auto* gi = ggml_get_tensor(ctx, "geom_post_final_proj");
            if (gi) ggml_backend_tensor_set(gi, geom_data.data(), 0, geom_data.size() * sizeof(float));
        }

        if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }
        ggml_backend_tensor_get(gr.geo_feats, geo_feats_cpu.data(), 0,
                                D * N_geo * sizeof(float));
        sam3_debug_dump_vec("s2_geo_feats", geo_feats_cpu.data(), geo_feats_cpu.size());

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    SAM3_LOG(2, "%s: geometry encoder done\n", __func__);

    // Build combined prompt on CPU: [text_feats, geo_feats] → [D * T]
    std::vector<float> combined_prompt_cpu(D * T);
    memcpy(combined_prompt_cpu.data(), text_feats_cpu.data(), D * L * sizeof(float));
    memcpy(combined_prompt_cpu.data() + D * L, geo_feats_cpu.data(), D * N_geo * sizeof(float));

    SAM3_LOG(2, "%s: starting fusion encoder\n", __func__);
    /*
    ** ── SUB-GRAPH 3: Fusion Encoder ──────────────────────────────────
    */
    std::vector<float> fenc_output_cpu(D * N_spatial);
    {
        const size_t sz = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() * 2;
        struct ggml_init_params gp = {sz, nullptr, true};
        auto* ctx = ggml_init(gp);

        auto* img = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_spatial, 1);
        ggml_set_name(img, "fenc_img");
        ggml_set_input(img);
        auto* pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_spatial, 1);
        ggml_set_name(pe, "fenc_pe");
        ggml_set_input(pe);
        auto* prompt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, T, 1);
        ggml_set_name(prompt, "fenc_prompt");
        ggml_set_input(prompt);
        auto* bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, 1, 1);
        ggml_set_name(bias, "fenc_bias");
        ggml_set_input(bias);

        auto* out = sam3_build_fenc_graph(ctx, model, img, prompt, pe, bias);
        ggml_set_output(out);

        auto* graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, out);

        // gemma4 debug: make per-layer outs safely readable under the sched
        struct ggml_tensor* dbg_layers[16] = {};
        if (getenv("SAM3_DUMP_STAGES")) {
            for (int li = 0; li < model.hparams.fenc_layers && li < 16; ++li) {
                char nm[64];
                snprintf(nm, sizeof(nm), "fenc_layer%d_out", li);
                dbg_layers[li] = ggml_get_tensor(ctx, nm);
                if (dbg_layers[li]) ggml_set_output(dbg_layers[li]);
            }
        }

        auto* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
            fprintf(stderr, "%s: fusion encoder alloc failed\n", __func__);
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }

        ggml_backend_tensor_set(img, img_feats_cpu.data(), 0, D * N_spatial * sizeof(float));
        ggml_backend_tensor_set(pe, img_pe_cpu.data(), 0, D * N_spatial * sizeof(float));
        ggml_backend_tensor_set(prompt, combined_prompt_cpu.data(), 0, D * T * sizeof(float));
        ggml_backend_tensor_set(bias, combined_bias_cpu.data(), 0, T * sizeof(float));

        if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }
        ggml_backend_tensor_get(out, fenc_output_cpu.data(), 0, D * N_spatial * sizeof(float));
        sam3_debug_dump_vec("s3_fenc_output", fenc_output_cpu.data(), fenc_output_cpu.size());
        if (getenv("SAM3_DUMP_STAGES")) {
            for (int li = 0; li < 16; ++li) {
                if (!dbg_layers[li]) continue;
                std::vector<float> tmp(ggml_nelements(dbg_layers[li]));
                if (sam3_copy_tensor_to_f32(dbg_layers[li], tmp)) {
                    char nm[64];
                    snprintf(nm, sizeof(nm), "s3_fenc_layer%d", li);
                    sam3_debug_dump_vec(nm, tmp.data(), tmp.size());
                }
            }
        }

        SAM3_LOG(2, "%s: fenc_out[0..4] = [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                 __func__, fenc_output_cpu[0], fenc_output_cpu[1], fenc_output_cpu[2],
                 fenc_output_cpu[3], fenc_output_cpu[4]);

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    SAM3_LOG(2, "%s: fusion encoder done\n", __func__);
    /*
    ** ── SUB-GRAPH 4: DETR Decoder + Scoring ──────────────────────────
    */
    std::vector<float> scores_data(NQ);
    std::vector<float> boxes_data(4 * NQ);
    std::vector<float> queries_data(D * (NQ + 1));  // 201 = NQ + presence token
    float presence_logit = 0.0f;
    {
        const size_t sz = ggml_tensor_overhead() * 32768 + ggml_graph_overhead() * 2;
        struct ggml_init_params gp = {sz, nullptr, true};
        auto* ctx = ggml_init(gp);

        auto* enc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_spatial, 1);
        ggml_set_name(enc, "ddec_enc");
        ggml_set_input(enc);
        auto* pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_spatial, 1);
        ggml_set_name(pe, "ddec_pe");
        ggml_set_input(pe);
        auto* txt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, T, 1);
        ggml_set_name(txt, "ddec_text");
        ggml_set_input(txt);
        auto* sdt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 64);
        ggml_set_name(sdt, "sine_dim_t");
        ggml_set_input(sdt);
        auto* rpb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_set_name(rpb, "rpb_coords");
        ggml_set_input(rpb);
        auto* tab = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, 1, 1);
        ggml_set_name(tab, "text_attn_bias");
        ggml_set_input(tab);
        auto* tvm = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, 1, 1);
        ggml_set_name(tvm, "text_valid_mask");
        ggml_set_input(tvm);

        auto dout = sam3_build_ddec_graph(ctx, model, enc, pe, txt, sdt, rpb, tab, tvm);
        ggml_set_output(dout.class_scores);
        ggml_set_output(dout.pred_boxes);
        ggml_set_output(dout.presence_score);
        ggml_set_output(dout.queries);

        auto* graph = ggml_new_graph_custom(ctx, 65536, false);
        ggml_build_forward_expand(graph, dout.class_scores);
        ggml_build_forward_expand(graph, dout.pred_boxes);
        ggml_build_forward_expand(graph, dout.presence_score);
        ggml_build_forward_expand(graph, dout.queries);

        auto* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
            fprintf(stderr, "%s: DETR decoder alloc failed\n", __func__);
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }

        ggml_backend_tensor_set(enc, fenc_output_cpu.data(), 0, D * N_spatial * sizeof(float));
        ggml_backend_tensor_set(pe, img_pe_cpu.data(), 0, D * N_spatial * sizeof(float));
        ggml_backend_tensor_set(txt, combined_prompt_cpu.data(), 0, D * T * sizeof(float));
        ggml_backend_tensor_set(sdt, sine_dim_t_cpu.data(), 0, 64 * sizeof(float));
        ggml_backend_tensor_set(rpb, rpb_coords_cpu.data(), 0, H * sizeof(float));
        ggml_backend_tensor_set(tab, combined_bias_cpu.data(), 0, T * sizeof(float));
        ggml_backend_tensor_set(tvm, text_valid_mask_cpu.data(), 0, T * sizeof(float));

        // Presence token position encoding: zeros
        auto* qpos = ggml_get_tensor(ctx, "ddec_query_pos_pres");
        if (qpos) {
            std::vector<float> z(D, 0.0f);
            ggml_backend_tensor_set(qpos, z.data(), 0, D * sizeof(float));
        }
        auto* rpbz = ggml_get_tensor(ctx, "rpb_pres_zeros");
        if (rpbz) {
            int n = (int)(rpbz->ne[0] * rpbz->ne[1] * rpbz->ne[2] * rpbz->ne[3]);
            std::vector<float> z(n, 0.0f);
            ggml_backend_tensor_set(rpbz, z.data(), 0, n * sizeof(float));
        }

        if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }

        ggml_backend_tensor_get(dout.class_scores, scores_data.data(), 0, NQ * sizeof(float));
        ggml_backend_tensor_get(dout.pred_boxes, boxes_data.data(), 0, 4 * NQ * sizeof(float));
        ggml_backend_tensor_get(dout.presence_score, &presence_logit, 0, sizeof(float));
        sam3_debug_dump_vec("s4_scores", scores_data.data(), scores_data.size());
        sam3_debug_dump_vec("s4_boxes", boxes_data.data(), boxes_data.size());
        sam3_debug_dump_vec("s4_presence", &presence_logit, 1);
        ggml_backend_tensor_get(dout.queries, queries_data.data(), 0,
                                D * (NQ + 1) * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    float presence_prob = 1.0f / (1.0f + expf(-presence_logit));

    SAM3_LOG(2, "%s: DETR decoder done\n", __func__);
    /*
    ** ── SUB-GRAPH 5: Segmentation Head ───────────────────────────────
    */
    const int mask_hw = H * 4;  // 288 for SAM3
    std::vector<float> all_masks(NQ * mask_hw * mask_hw);
    {
        const size_t sz = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() * 2;
        struct ggml_init_params gp = {sz, nullptr, true};
        auto* ctx = ggml_init(gp);

        auto* enc_h = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_spatial, 1);
        ggml_set_name(enc_h, "seg_enc");
        ggml_set_input(enc_h);

        // FPN features at 3 scales — create fresh inputs
        const int H0 = H * 4, H1 = H * 2;
        auto* fpn0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H0, H0, 1);
        ggml_set_name(fpn0, "seg_fpn0");
        ggml_set_input(fpn0);
        auto* fpn1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H1, H1, 1);
        ggml_set_name(fpn1, "seg_fpn1");
        ggml_set_input(fpn1);
        auto* fpn2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, H, 1);
        ggml_set_name(fpn2, "seg_fpn2");
        ggml_set_input(fpn2);
        struct ggml_tensor* fpn_feats[3] = {fpn0, fpn1, fpn2};

        // Object queries (skip presence token at index 0 → start at index 1)
        auto* oq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NQ, 1);
        ggml_set_name(oq, "seg_queries");
        ggml_set_input(oq);

        auto* txt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, T, 1);
        ggml_set_name(txt, "seg_text");
        ggml_set_input(txt);
        auto* tab = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, 1, 1);
        ggml_set_name(tab, "seg_bias");
        ggml_set_input(tab);

        auto* out = sam3_build_seg_head_graph(ctx, model, enc_h, fpn_feats,
                                              oq, txt, tab);
        ggml_set_output(out);

        auto* graph = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(graph, out);

        SAM3_LOG(2, "%s: seg head graph: %d nodes\n", __func__, ggml_graph_n_nodes(graph));

        auto* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
            fprintf(stderr, "%s: segmentation head alloc failed\n", __func__);
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }

        ggml_backend_tensor_set(enc_h, fenc_output_cpu.data(), 0,
                                D * N_spatial * sizeof(float));

        // Copy FPN features from state
        {
            size_t s0 = (size_t)D * H0 * H0 * sizeof(float);
            size_t s1 = (size_t)D * H1 * H1 * sizeof(float);
            size_t s2 = (size_t)D * H * H * sizeof(float);
            std::vector<float> b0(D * H0 * H0), b1(D * H1 * H1), b2(D * H * H);
            ggml_backend_tensor_get(state.neck_det[0], b0.data(), 0, s0);
            if (fpn0->buffer) ggml_backend_tensor_set(fpn0, b0.data(), 0, s0);
            ggml_backend_tensor_get(state.neck_det[1], b1.data(), 0, s1);
            if (fpn1->buffer) ggml_backend_tensor_set(fpn1, b1.data(), 0, s1);
            ggml_backend_tensor_get(state.neck_det[2], b2.data(), 0, s2);
            if (fpn2->buffer) ggml_backend_tensor_set(fpn2, b2.data(), 0, s2);
        }

        // Object queries: extract from DETR queries (skip presence token at slot 0)
        // queries_data is flat [D * 201], presence token is at positions [0..D-1]
        // Object queries start at position [D..D*(NQ+1)-1]
        ggml_backend_tensor_set(oq, queries_data.data() + D, 0, D * NQ * sizeof(float));

        ggml_backend_tensor_set(txt, combined_prompt_cpu.data(), 0, D * T * sizeof(float));
        ggml_backend_tensor_set(tab, combined_bias_cpu.data(), 0, T * sizeof(float));

        if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return result;
        }
        ggml_backend_tensor_get(out, all_masks.data(), 0, all_masks.size() * sizeof(float));
        sam3_debug_dump_vec("s5_mask_logits", all_masks.data(), all_masks.size());

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    /*
    ** ── Post-processing: thresholding + NMS + mask resize ────────────
    */
    std::vector<sam3_detection> dets;
    for (int q = 0; q < NQ; ++q) {
        float class_prob = 1.0f / (1.0f + expf(-scores_data[q]));
        float score = class_prob * presence_prob;
        if (score < params.score_threshold) continue;

        sam3_detection det;
        float cx = boxes_data[0 + q * 4];
        float cy = boxes_data[1 + q * 4];
        float bw = boxes_data[2 + q * 4];
        float bh = boxes_data[3 + q * 4];

        det.box = sam3_cxcywh_to_xyxy(cx, cy, bw, bh, state.orig_width, state.orig_height);
        det.score = score;

        const float* mask_ptr = all_masks.data() + q * mask_hw * mask_hw;
        auto mask_resized = sam3_bilinear_interpolate(mask_ptr, mask_hw, mask_hw,
                                                      state.orig_width, state.orig_height);
        det.mask.width = state.orig_width;
        det.mask.height = state.orig_height;
        det.mask.data.resize(state.orig_width * state.orig_height);
        for (int i = 0; i < (int)mask_resized.size(); ++i)
            det.mask.data[i] = (mask_resized[i] > 0.0f) ? 255 : 0;
        det.mask.iou_score = score;

        dets.push_back(std::move(det));
    }

    SAM3_LOG(2, "%s: %zu detections above threshold %.2f (presence=%.3f, logit=%.3f)\n",
             __func__, dets.size(), params.score_threshold, presence_prob, presence_logit);

    auto keep = sam3_nms(dets, params.nms_threshold);
    for (int i = 0; i < (int)keep.size(); ++i) {
        dets[keep[i]].instance_id = i + 1;
        result.detections.push_back(std::move(dets[keep[i]]));
    }

    SAM3_LOG(2, "%s: %zu detections after NMS\n", __func__, result.detections.size());

#if SAM3_LOG_LEVEL >= 1
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    SAM3_LOG(1, "%s: completed in %.1f ms\n", __func__, total_ms);
#endif

    return result;
#endif  // !(SAM3_TRT_ENCODER && SAM3_TRT_ONLY)
}

/*****************************************************************************
** Image segmentation — PVS (visual-prompted) (Phase 6, Step 6.3)
*****************************************************************************/

sam3_result sam3_segment_pvs(sam3_state& state,
                             const sam3_model& model,
                             const sam3_pvs_params& params) {
#ifdef SAM3_TRT_ENCODER
    {
        sam3_result trt_result;
        if (sam3_try_trt_segment_pvs(state, model, params, trt_result)) {
            return trt_result;
        }
    }
#endif
#if defined(SAM3_TRT_ENCODER) && defined(SAM3_TRT_ONLY)
    // This build only supports the TRT-covered PVS scope (see
    // sam3_try_trt_segment_pvs's doc: any point/box mix up to 16 prompt
    // tokens) -- no ggml fallback graph is compiled in, so an out-of-scope
    // prompt or a TRT failure is a hard error here, not a silent (slow but
    // correct) degrade. Rebuild without SAM3_TRT_ONLY for unbounded prompt
    // support.
    fprintf(stderr, "%s: TRT PVS unavailable/out of scope and this build is "
                    "TRT-only (SAM3_TRT_ONLY) -- no ggml fallback compiled in\n", __func__);
    return sam3_result{};
#else
    if (model.trt_only_weights) {
        fprintf(stderr, "%s: TRT PVS unavailable/out of scope but the model was loaded "
                        "with SAM3_TRT_SKIP_GGML_WEIGHTS=1 -- ggml weights absent, refusing "
                        "to run the fallback graph\n", __func__);
        return sam3_result{};
    }
#if SAM3_LOG_LEVEL >= 1
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    const auto& hp = model.hparams;
    const int D = hp.sam_embed_dim;                      // 256
    const int H = sam3_eff_feat_size(state, hp);
    const int num_mask_tokens = hp.sam_n_multimask + 1;  // 4
    const int eff_img_size = sam3_eff_img_size(state, hp);
    sam3_result result;

    // ── Validate ─────────────────────────────────────────────────────────
    if (!state.neck_trk[0]) {
        fprintf(stderr, "%s: image not encoded — call sam3_encode_image first\n", __func__);
        return result;
    }
    if (params.pos_points.empty() && !params.use_box) {
        fprintf(stderr, "%s: no prompts provided (need at least one point or box)\n", __func__);
        return result;
    }

    SAM3_LOG(2, "%s: %zu pos points, %zu neg points, box=%s, multimask=%s\n",
             __func__, params.pos_points.size(), params.neg_points.size(),
             params.use_box ? "yes" : "no", params.multimask ? "yes" : "no");

    // ── Build computation graph ──────────────────────────────────────────
    const size_t buf_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return result;
    }

    // ── SAM prompt encoder (CPU pre-compute + input tensors) ─────────────
    auto pe_out = sam3_build_sam_pe(ctx0, params, D, H);

    // ── Create fresh input tensors for tracker features ──────────────────
    // CRITICAL: Do NOT use state.neck_trk[*] directly in graph operations
    // (like ggml_add). They are tensors from a previous graph, and
    // ggml_build_forward_expand traces all ancestors — which would pull in
    // the ENTIRE ViT + neck recomputation (2500+ nodes, ~37 seconds).
    // Instead: create fresh input tensors and copy data from state on CPU.
    const int H0 = H * 4;  // 288
    const int H1 = H * 2;  // 144

    auto* image_feats = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H, H, 1);
    ggml_set_name(image_feats, "sam_dec_image_feats");
    ggml_set_input(image_feats);

    auto* feat_s0 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H0, H0, 1);
    ggml_set_name(feat_s0, "pvs_feat_s0");
    ggml_set_input(feat_s0);

    auto* feat_s1 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H1, H1, 1);
    ggml_set_name(feat_s1, "pvs_feat_s1");
    ggml_set_input(feat_s1);

    // ── SAM mask decoder graph ───────────────────────────────────────────
    auto dec_out = sam3_build_sam_dec_graph(ctx0, model,
                                            image_feats,
                                            pe_out.image_pe,
                                            pe_out.sparse,
                                            pe_out.dense,
                                            feat_s0,
                                            feat_s1, H);

    // Mark outputs
    ggml_set_output(dec_out.masks);
    ggml_set_output(dec_out.iou_pred);
    ggml_set_output(dec_out.obj_score);
    ggml_set_output(dec_out.sam_token);

    // ── Build and allocate graph ─────────────────────────────────────────
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 32768, false);
    ggml_build_forward_expand(graph, dec_out.masks);
    ggml_build_forward_expand(graph, dec_out.iou_pred);
    ggml_build_forward_expand(graph, dec_out.obj_score);
    ggml_build_forward_expand(graph, dec_out.sam_token);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph)) {
        fprintf(stderr, "%s: failed to reserve graph memory\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return result;
    }
    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return result;
    }

    SAM3_LOG(2, "%s: graph allocated, %d nodes\n", __func__, ggml_graph_n_nodes(graph));

    // Set default obj_score when pred_obj_scores=False (older SAM2 models)
    if (!model.sam_dec.obj_score_token) {
        auto* t = ggml_graph_get_tensor(graph, "sam_dec_obj_score");
        if (t) { float v = 10.0f; ggml_backend_tensor_set(t, &v, 0, sizeof(float)); }
    }

    // ── Upload input data (using cached embeddings) ────────────────────
    // Populate PE cache on first call (reads model weights from GPU once)
    sam3_populate_pe_cache(state, model);

    {
        const int N_pts = pe_out.n_tokens;
        const int num_pos_feats = D / 2;

        std::vector<float> all_coords;
        std::vector<int> all_labels;
        sam3_collect_pvs_prompt_tokens(params, all_coords, all_labels);

        // Sparse embeddings — only this changes per call
        std::vector<float> sparse_data(N_pts * D, 0.0f);
        for (int p = 0; p < N_pts; ++p) {
            // Scale from original image space to model input space, then shift to pixel center
            float px = all_coords[p * 2 + 0] / (float)state.orig_width * (float)eff_img_size + 0.5f;
            float py = all_coords[p * 2 + 1] / (float)state.orig_height * (float)eff_img_size + 0.5f;
            float x_norm = px / (float)eff_img_size;
            float y_norm = py / (float)eff_img_size;
            float pe_vec[256];
            sam3_pe_encode_coord(pe_vec, x_norm, y_norm,
                                 state.pe_gauss_cache.data(), num_pos_feats);
            int label = all_labels[p];
            if (label == -1) {
                for (int d = 0; d < D; ++d)
                    sparse_data[p * D + d] = state.not_a_point_cache[d];
            } else {
                for (int d = 0; d < D; ++d)
                    sparse_data[p * D + d] = pe_vec[d] + state.point_emb_cache[label][d];
            }
        }
        ggml_backend_tensor_set(pe_out.sparse, sparse_data.data(), 0, N_pts * D * sizeof(float));

        // Dump sparse embeddings if requested
        {
            const char* dd = getenv("SAM2_DUMP_DIR");
            if (dd) {
                char p[512]; snprintf(p, sizeof(p), "%s/cpp_sparse_emb.bin", dd);
                FILE* f = fopen(p, "wb");
                if (f) { fwrite(sparse_data.data(), sizeof(float), N_pts * D, f); fclose(f); }
                fprintf(stderr, "  [DUMP] cpp_sparse_emb: %d tokens x %d dims\n", N_pts, D);
            }
        }

        // Dense PE grid and no-mask embedding — use pre-computed caches
        ggml_backend_tensor_set(pe_out.image_pe, state.dense_pe_cache.data(),
                                0, D * H * H * sizeof(float));
        ggml_backend_tensor_set(pe_out.dense, state.dense_nomask_cache.data(),
                                0, D * H * H * sizeof(float));
    }

    // ── Copy tracker features from state to fresh input tensors ─────────
    // image_feats = neck_trk[2] + no_mem_embed (computed on CPU)
    {
        const int n2 = D * H * H;
        std::vector<float> trk2(n2), no_mem_data(D);
        ggml_backend_tensor_get(state.neck_trk[2], trk2.data(), 0, n2 * sizeof(float));
        ggml_backend_tensor_get(model.tensors.at("no_mem_embed"), no_mem_data.data(), 0, D * sizeof(float));
        // Add no_mem_embed (broadcast [D] to [D, H, H])
        for (int s = 0; s < H * H; ++s)
            for (int d = 0; d < D; ++d)
                trk2[d + s * D] += no_mem_data[d];
        ggml_backend_tensor_set(image_feats, trk2.data(), 0, n2 * sizeof(float));

        // Dump image_feats (with no_mem_embed) if requested
        {
            const char* dd = getenv("SAM2_DUMP_DIR");
            if (dd) {
                char p[512]; snprintf(p, sizeof(p), "%s/cpp_image_feats.bin", dd);
                FILE* f = fopen(p, "wb");
                if (f) { fwrite(trk2.data(), sizeof(float), n2, f); fclose(f); }
                fprintf(stderr, "  [DUMP] cpp_image_feats: [%d, %d, %d]\n", D, H, H);
            }
        }

        // feat_s0 = neck_trk[0], feat_s1 = neck_trk[1]
        const int n0 = D * H0 * H0;
        const int n1 = D * H1 * H1;
        std::vector<float> trk0(n0), trk1(n1);
        ggml_backend_tensor_get(state.neck_trk[0], trk0.data(), 0, n0 * sizeof(float));
        ggml_backend_tensor_set(feat_s0, trk0.data(), 0, n0 * sizeof(float));
        ggml_backend_tensor_get(state.neck_trk[1], trk1.data(), 0, n1 * sizeof(float));
        ggml_backend_tensor_set(feat_s1, trk1.data(), 0, n1 * sizeof(float));
    }

    // ── Compute ──────────────────────────────────────────────────────────
    {
#if SAM3_LOG_LEVEL >= 1
        auto t0 = std::chrono::high_resolution_clock::now();
#endif
        if (!sam3_graph_compute(model.backend, graph, state.n_threads)) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx0);
            return result;
        }
#if SAM3_LOG_LEVEL >= 1
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        SAM3_LOG(1, "%s: graph computed in %.1f ms (%d threads)\n",
                 __func__, ms, state.n_threads);
#endif
    }

    // ── Dump decoder outputs if SAM2_DUMP_DIR set ──────────────────────
    {
        const char* dump_dir = getenv("SAM2_DUMP_DIR");
        if (dump_dir) {
            auto dump_t = [&](const char* name, struct ggml_tensor* t) {
                if (!t) return;
                int64_t nb = ggml_nbytes(t);
                std::vector<char> buf(nb);
                ggml_backend_tensor_get(t, buf.data(), 0, nb);
                char path[512];
                snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, name);
                FILE* f = fopen(path, "wb");
                if (f) { fwrite(buf.data(), 1, nb, f); fclose(f); }
                snprintf(path, sizeof(path), "%s/%s.shape", dump_dir, name);
                f = fopen(path, "w");
                if (f) {
                    fprintf(f, "%lld,%lld,%lld,%lld",
                            (long long)t->ne[0], (long long)t->ne[1],
                            (long long)t->ne[2], (long long)t->ne[3]);
                    fclose(f);
                }
                fprintf(stderr, "  [DUMP] %s: [%lld,%lld,%lld,%lld]\n", name,
                        (long long)t->ne[0], (long long)t->ne[1],
                        (long long)t->ne[2], (long long)t->ne[3]);
            };
            dump_t("cpp_pvs_masks", dec_out.masks);
            dump_t("cpp_pvs_iou", dec_out.iou_pred);
            dump_t("cpp_pvs_obj_score", dec_out.obj_score);
            // Decoder transformer intermediates
            dump_t("cpp_dec_queries", ggml_graph_get_tensor(graph, "dbg_dec_queries_out"));
            dump_t("cpp_dec_keys", ggml_graph_get_tensor(graph, "dbg_dec_keys_out"));
            // Block 0 internals
            const char* b0_names[] = {"sam_dec_tokens_initial",
                                       "dbg_twoway_skip_sa_norm", "dbg_twoway_skip_ca_tok2img",
                                       "dbg_twoway_skip_mlp", "dbg_twoway_skip_img2tok",
                                       "dbg_sa0_Q_proj", "dbg_sa0_merged"};
            for (auto* bn : b0_names) dump_t(bn, ggml_graph_get_tensor(graph, bn));
            // Per-block outputs
            for (int bi = 0; bi < 2; bi++) {
                char bn[64];
                snprintf(bn, sizeof(bn), "sam_dec_block%d_queries", bi);
                dump_t(bn, ggml_graph_get_tensor(graph, bn));
                snprintf(bn, sizeof(bn), "sam_dec_block%d_keys", bi);
                dump_t(bn, ggml_graph_get_tensor(graph, bn));
            }
        }
    }

    // ── Read outputs ─────────────────────────────────────────────────────
    // masks: [H*4×H*4, 4, 1]
    const int mask_hw = H * 4;
    std::vector<float> masks_data(mask_hw * mask_hw * num_mask_tokens);
    ggml_backend_tensor_get(dec_out.masks, masks_data.data(), 0, masks_data.size() * sizeof(float));

    // IoU predictions: [4, 1]
    std::vector<float> iou_data(num_mask_tokens);
    ggml_backend_tensor_get(dec_out.iou_pred, iou_data.data(), 0, num_mask_tokens * sizeof(float));

    // Object score: [1, 1]
    float obj_logit = 0.0f;
    ggml_backend_tensor_get(dec_out.obj_score, &obj_logit, 0, sizeof(float));
    float obj_score = 1.0f / (1.0f + expf(-obj_logit));

    // SAM output token: [D, 1] — needed for object pointer extraction in tracking
    std::vector<float> sam_token_data(D);
    ggml_backend_tensor_get(dec_out.sam_token, sam_token_data.data(), 0, D * sizeof(float));

    SAM3_LOG(2, "%s: obj_score=%.4f (logit=%.4f), iou=[%.3f, %.3f, %.3f, %.3f]\n",
             __func__, obj_score, obj_logit,
             iou_data[0], iou_data[1], iou_data[2], iou_data[3]);

    // ── Select masks based on multimask mode ─────────────────────────────
    // Python: if multimask_output → masks[:, 1:, :, :], iou_pred[:, 1:]
    //         else                → masks[:, 0:1, :, :], iou_pred[:, 0:1]
    int start_idx, end_idx;
    if (params.multimask) {
        start_idx = 1;
        end_idx = num_mask_tokens;
    } else {
        start_idx = 0;
        end_idx = 1;
    }

    for (int m = start_idx; m < end_idx; ++m) {
        sam3_detection det;
        det.sam_token = sam_token_data;

        // Resize mask from 288×288 to original image size
        const float* mask_ptr = masks_data.data() + m * mask_hw * mask_hw;
        auto mask_resized = sam3_bilinear_interpolate(mask_ptr, mask_hw, mask_hw,
                                                      state.orig_width, state.orig_height);

        // Binarize at threshold 0.0 (sigmoid(logit) > 0.5 ↔ logit > 0.0)
        det.mask.width = state.orig_width;
        det.mask.height = state.orig_height;
        det.mask.data.resize(state.orig_width * state.orig_height);
        for (int i = 0; i < (int)mask_resized.size(); ++i) {
            det.mask.data[i] = (mask_resized[i] > 0.0f) ? 255 : 0;
        }

        det.mask.iou_score = iou_data[m];
        det.mask.obj_score = obj_score;
        det.mask.instance_id = m;
        det.score = iou_data[m];
        det.iou_score = iou_data[m];
        det.instance_id = m;

        // Compute bounding box from mask
        int min_x = state.orig_width, min_y = state.orig_height;
        int max_x = 0, max_y = 0;
        for (int y = 0; y < state.orig_height; ++y) {
            for (int x = 0; x < state.orig_width; ++x) {
                if (det.mask.data[y * state.orig_width + x] > 0) {
                    min_x = std::min(min_x, x);
                    min_y = std::min(min_y, y);
                    max_x = std::max(max_x, x);
                    max_y = std::max(max_y, y);
                }
            }
        }
        det.box = {(float)min_x, (float)min_y, (float)max_x, (float)max_y};

        result.detections.push_back(std::move(det));
    }

    SAM3_LOG(2, "%s: %zu masks returned\n", __func__, result.detections.size());

    // ── Cleanup ──────────────────────────────────────────────────────────
    ggml_gallocr_free(galloc);
    ggml_free(ctx0);

#if SAM3_LOG_LEVEL >= 1
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    SAM3_LOG(1, "%s: completed in %.1f ms\n", __func__, total_ms);
#endif

    return result;
#endif  // !(SAM3_TRT_ENCODER && SAM3_TRT_ONLY)
}

