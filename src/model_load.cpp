// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** Model loading — internal helpers
*****************************************************************************/

bool sam3_load_hparams(std::ifstream& fin, sam3_hparams& hp) {
    auto rd = [&](int32_t& v) { fin.read(reinterpret_cast<char*>(&v), 4); };
    rd(hp.img_size);
    rd(hp.patch_size);
    rd(hp.vit_embed_dim);
    rd(hp.vit_depth);
    rd(hp.vit_num_heads);
    int32_t mlp_ratio_x1000;
    rd(mlp_ratio_x1000);
    hp.vit_mlp_dim = static_cast<int32_t>(hp.vit_embed_dim * (mlp_ratio_x1000 / 1000.0f));
    rd(hp.vit_window_size);
    rd(hp.n_global_attn);
    for (int i = 0; i < hp.n_global_attn && i < 4; ++i) {
        rd(hp.global_attn_idx[i]);
    }
    rd(hp.text_width);
    rd(hp.text_heads);
    rd(hp.text_layers);
    rd(hp.text_ctx_len);
    rd(hp.text_vocab_size);
    rd(hp.text_out_dim);
    rd(hp.neck_dim);
    rd(hp.fenc_layers);
    rd(hp.fenc_heads);
    rd(hp.fenc_ffn_dim);
    rd(hp.ddec_layers);
    rd(hp.ddec_heads);
    rd(hp.ddec_ffn_dim);
    rd(hp.ddec_num_queries);
    rd(hp.geom_layers);
    rd(hp.n_presence_tokens);
    rd(hp.n_geom_queries);
    rd(hp.sam_embed_dim);
    rd(hp.sam_dec_depth);
    rd(hp.sam_n_multimask);
    rd(hp.sam_iou_head_depth);
    rd(hp.mem_out_dim);
    rd(hp.mem_attn_layers);
    rd(hp.num_maskmem);
    rd(hp.max_obj_ptrs);
    rd(hp.n_amb_experts);
    rd(hp.visual_only);
    return !fin.fail();
}

static void sam3_print_hparams(const sam3_hparams& hp) {
    fprintf(stderr, "  img_size       = %d\n", hp.img_size);
    fprintf(stderr, "  patch_size     = %d\n", hp.patch_size);
    fprintf(stderr, "  vit_embed_dim  = %d\n", hp.vit_embed_dim);
    fprintf(stderr, "  vit_depth      = %d\n", hp.vit_depth);
    fprintf(stderr, "  vit_num_heads  = %d\n", hp.vit_num_heads);
    fprintf(stderr, "  vit_mlp_dim    = %d\n", hp.vit_mlp_dim);
    fprintf(stderr, "  vit_window     = %d\n", hp.vit_window_size);
    fprintf(stderr, "  text_width     = %d\n", hp.text_width);
    fprintf(stderr, "  text_layers    = %d\n", hp.text_layers);
    fprintf(stderr, "  neck_dim       = %d\n", hp.neck_dim);
    fprintf(stderr, "  fenc_layers    = %d\n", hp.fenc_layers);
    fprintf(stderr, "  ddec_layers    = %d\n", hp.ddec_layers);
    fprintf(stderr, "  ddec_queries   = %d\n", hp.ddec_num_queries);
    fprintf(stderr, "  sam_embed_dim  = %d\n", hp.sam_embed_dim);
    fprintf(stderr, "  mem_attn_lyrs  = %d\n", hp.mem_attn_layers);
    fprintf(stderr, "  num_maskmem    = %d\n", hp.num_maskmem);
    fprintf(stderr, "  visual_only    = %d\n", hp.visual_only);
}

static void sam3_register_tensors(sam3_model& model) {
    const auto& hp = model.hparams;
    auto& tensors = model.tensors;
    auto ctx = model.ctx;

    // SAM3_TRT_SKIP_GGML_WEIGHTS: register only sam_pe.* and the geometry-input
    // helpers used by TRT request preparation (see the flag's doc on
    // sam3_model). A skipped tensor is never created in the ctx, so backend
    // allocation stays around 13MB instead of ~1.1GB;
    // the corresponding model struct fields stay nullptr and the loader
    // seeks past their file data. Any registration this predicate rejects
    // MUST be matched by a trt_only_weights guard on every compute path
    // that would dereference it.
    auto keep = [&](const std::string& name) -> bool {
        if (!model.trt_only_weights) return true;
        if (name.compare(0, 7, "sam_pe.") == 0) return true;
        // The TRT PCS exemplar path builds its geometry input tokens on the
        // CPU (sam3_precompute_geom_input / the concept-embedding capture
        // API) from these ~13MB of weights -- keep them loadable.
        static const char* geom_input_weights[] = {
            "geom.boxes_direct_project.weight", "geom.boxes_direct_project.bias",
            "geom.label_embed.weight",          "geom.cls_embed.weight",
            "geom.boxes_pos_enc_project.weight", "geom.boxes_pos_enc_project.bias",
            "geom.boxes_pool_project.weight",   "geom.boxes_pool_project.bias",
            "geom.img_pre_norm.weight",         "geom.img_pre_norm.bias",
            "geom.final_proj.weight",           "geom.final_proj.bias",
            "geom.norm.weight",                 "geom.norm.bias",
        };
        for (const char* n : geom_input_weights) {
            if (name == n) return true;
        }
        return false;
    };

    auto T1 = [&](const std::string& name, int64_t d0) -> ggml_tensor* {
        if (!keep(name)) return nullptr;
        auto* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d0);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    const ggml_type WTYPE = model.weight_type;
    const int64_t   WBLK  = ggml_blck_size(WTYPE);  // 1 for F32/F16, 32 for Q4/Q8

    auto T2 = [&](const std::string& name, int64_t d0, int64_t d1) -> ggml_tensor* {
        if (!keep(name)) return nullptr;
        const ggml_type type = (d0 % WBLK == 0) ? WTYPE : GGML_TYPE_F32;
        auto* t = ggml_new_tensor_2d(ctx, type, d0, d1);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T3 = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2) -> ggml_tensor* {
        if (!keep(name)) return nullptr;
        const ggml_type type = (d0 % WBLK == 0) ? WTYPE : GGML_TYPE_F32;
        auto* t = ggml_new_tensor_3d(ctx, type, d0, d1, d2);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T4 = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor* {
        if (!keep(name)) return nullptr;
        const ggml_type type = (d0 % WBLK == 0) ? WTYPE : GGML_TYPE_F32;
        auto* t = ggml_new_tensor_4d(ctx, type, d0, d1, d2, d3);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T1f = T1;
    auto T2f = [&](const std::string& name, int64_t d0, int64_t d1) -> ggml_tensor* {
        if (!keep(name)) return nullptr;
        auto* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0, d1);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T3f = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2) -> ggml_tensor* {
        if (!keep(name)) return nullptr;
        auto* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d0, d1, d2);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T4f = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor* {
        if (!keep(name)) return nullptr;
        auto* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d0, d1, d2, d3);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };

    const int E = hp.vit_embed_dim;      // 1024
    const int D = hp.neck_dim;           // 256
    const int TW = hp.text_width;        // 1024
    const int MLP = hp.vit_mlp_dim;      // 4736
    const int FFN = hp.fenc_ffn_dim;     // 2048
    const int NQ = hp.ddec_num_queries;  // 200
    const int MD = hp.mem_out_dim;       // 64
    const int H = hp.n_img_embd();       // 72

    // ── ViT backbone ─────────────────────────────────────────────────────
    model.vit.blocks.resize(hp.vit_depth);

    model.vit.patch_embed_w = T4("vit.patch_embed.proj.weight", hp.patch_size, hp.patch_size, 3, E);
    // pos_embed: Hiera stores [1, 24, 24, 1024] at pretrained resolution (no cls token).
    // Conversion script writes reversed dims → ggml [E, 24, 24, 1].
    // Tiled 3x at runtime to [E, 72, 72, 1].
    {
        const int pretrained_grid = hp.img_size / hp.patch_size / 3;  // 1008/14/3 = 24
        model.vit.pos_embed = T4f("vit.pos_embed", E, pretrained_grid, pretrained_grid, 1);
    }
    model.vit.ln_pre_w = T1f("vit.ln_pre.weight", E);
    model.vit.ln_pre_b = T1f("vit.ln_pre.bias", E);

    for (int i = 0; i < hp.vit_depth; ++i) {
        auto& blk = model.vit.blocks[i];
        auto p = "vit.blocks." + std::to_string(i);
        blk.norm1_w = T1f(p + ".norm1.weight", E);
        blk.norm1_b = T1f(p + ".norm1.bias", E);
        blk.qkv_w = T2(p + ".attn.qkv.weight", E, 3 * E);
        blk.qkv_b = T1f(p + ".attn.qkv.bias", 3 * E);
        blk.proj_w = T2(p + ".attn.proj.weight", E, E);
        blk.proj_b = T1f(p + ".attn.proj.bias", E);
        blk.norm2_w = T1f(p + ".norm2.weight", E);
        blk.norm2_b = T1f(p + ".norm2.bias", E);
        blk.mlp_fc1_w = T2(p + ".mlp.lin1.weight", E, MLP);
        blk.mlp_fc1_b = T1f(p + ".mlp.lin1.bias", MLP);
        blk.mlp_fc2_w = T2(p + ".mlp.lin2.weight", MLP, E);
        blk.mlp_fc2_b = T1f(p + ".mlp.lin2.bias", E);

        // RoPE freqs_cis: [N, 32, 2] where N=5184 for global, 576 for window
        int64_t rope_n = hp.is_global_attn(i) ? hp.n_img_tokens() : (hp.vit_window_size * hp.vit_window_size);
        blk.freqs_cis = T3f(p + ".attn.freqs_cis", 2, 32, rope_n);
    }

    // ── Neck (detector + tracker) ────────────────────────────────────────
    // ggml weight layout: conv2d [kW, kH, Cin, Cout], conv_transpose [kW, kH, Cout, Cin]
    auto register_neck = [&](sam3_neck& neck, const std::string& prefix) {
        // scale 0 (4x): ConvTranspose(E→512, k=2, s=2), GELU, ConvTranspose(512→D, k=2, s=2), Conv1x1(D→D), Conv3x3(D→D)
        neck.scales[0].deconv1_w = T4(prefix + "0.dconv_2x2_0.weight", 2, 2, 512, E);  // [kW, kH, Cout=512, Cin=E]
        neck.scales[0].deconv1_b = T1f(prefix + "0.dconv_2x2_0.bias", 512);
        neck.scales[0].deconv2_w = T4(prefix + "0.dconv_2x2_1.weight", 2, 2, D, 512);  // [kW, kH, Cout=D, Cin=512]
        neck.scales[0].deconv2_b = T1f(prefix + "0.dconv_2x2_1.bias", D);
        neck.scales[0].conv1x1_w = T4(prefix + "0.conv_1x1.weight", 1, 1, D, D);  // Conv2d(D→D)
        neck.scales[0].conv1x1_b = T1f(prefix + "0.conv_1x1.bias", D);
        neck.scales[0].conv3x3_w = T4(prefix + "0.conv_3x3.weight", 3, 3, D, D);  // Conv2d(D→D)
        neck.scales[0].conv3x3_b = T1f(prefix + "0.conv_3x3.bias", D);

        // scale 1 (2x): ConvTranspose(E→512, k=2, s=2), Conv1x1(512→D), Conv3x3(D→D)
        neck.scales[1].deconv1_w = T4(prefix + "1.dconv_2x2.weight", 2, 2, 512, E);  // ConvTranspose
        neck.scales[1].deconv1_b = T1f(prefix + "1.dconv_2x2.bias", 512);
        neck.scales[1].conv1x1_w = T4(prefix + "1.conv_1x1.weight", 1, 1, 512, D);  // Conv2d(512→D): Cin=512, Cout=D
        neck.scales[1].conv1x1_b = T1f(prefix + "1.conv_1x1.bias", D);
        neck.scales[1].conv3x3_w = T4(prefix + "1.conv_3x3.weight", 3, 3, D, D);
        neck.scales[1].conv3x3_b = T1f(prefix + "1.conv_3x3.bias", D);

        // scale 2 (1x): Conv1x1(E→D), Conv3x3(D→D)
        neck.scales[2].conv1x1_w = T4(prefix + "2.conv_1x1.weight", 1, 1, E, D);  // Conv2d(E→D): Cin=E, Cout=D
        neck.scales[2].conv1x1_b = T1f(prefix + "2.conv_1x1.bias", D);
        neck.scales[2].conv3x3_w = T4(prefix + "2.conv_3x3.weight", 3, 3, D, D);
        neck.scales[2].conv3x3_b = T1f(prefix + "2.conv_3x3.bias", D);

        // scale 3 (0.5x): MaxPool(k=2, s=2), Conv1x1(E→D), Conv3x3(D→D)
        neck.scales[3].conv1x1_w = T4(prefix + "3.conv_1x1.weight", 1, 1, E, D);
        neck.scales[3].conv1x1_b = T1f(prefix + "3.conv_1x1.bias", D);
        neck.scales[3].conv3x3_w = T4(prefix + "3.conv_3x3.weight", 3, 3, D, D);
        neck.scales[3].conv3x3_b = T1f(prefix + "3.conv_3x3.bias", D);
    };
    if (!hp.visual_only) {
        register_neck(model.neck_det, "neck.det.");
    }
    register_neck(model.neck_trk, "neck.trk.");

    // Helper lambdas used by multiple sections (detector + tracker)
    auto reg = [&](const std::string& n, int64_t d0, int64_t d1, bool is_f32 = false) -> ggml_tensor* {
        if (!keep(n)) return nullptr;
        const ggml_type rtype = (is_f32 || d0 % WBLK != 0) ? GGML_TYPE_F32 : WTYPE;
        auto* t = ggml_new_tensor_2d(ctx, rtype, d0, d1);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };
    auto reg1 = [&](const std::string& n, int64_t d0) -> ggml_tensor* {
        if (!keep(n)) return nullptr;
        auto* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d0);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };
    auto reg4 = [&](const std::string& n, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor* {
        if (!keep(n)) return nullptr;
        const ggml_type rtype = (d0 % WBLK == 0) ? WTYPE : GGML_TYPE_F32;
        auto* t = ggml_new_tensor_4d(ctx, rtype, d0, d1, d2, d3);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };

    // ── Detector-only tensors (skipped for visual-only models) ──────────
    if (!hp.visual_only) {

    // ── Text encoder ─────────────────────────────────────────────────────
    model.text_enc.blocks.resize(hp.text_layers);
    model.text_enc.token_embed_w = T2f("text.token_embed.weight", TW, hp.text_vocab_size);
    model.text_enc.pos_embed = T2f("text.pos_embed", TW, hp.text_ctx_len);
    model.text_enc.ln_final_w = T1f("text.ln_final.weight", TW);
    model.text_enc.ln_final_b = T1f("text.ln_final.bias", TW);
    model.text_enc.resizer_w = T2("text.resizer.weight", TW, hp.text_out_dim);
    model.text_enc.resizer_b = T1f("text.resizer.bias", hp.text_out_dim);
    // text.text_projection is intentionally not registered — the conversion
    // script skips it and the loader rejects unknown tensors. See struct comment.

    for (int i = 0; i < hp.text_layers; ++i) {
        auto& blk = model.text_enc.blocks[i];
        auto p = "text.blocks." + std::to_string(i);
        blk.attn_in_proj_w = T2(p + ".attn.in_proj.weight", TW, 3 * TW);
        blk.attn_in_proj_b = T1f(p + ".attn.in_proj.bias", 3 * TW);
        blk.attn_out_proj_w = T2(p + ".attn.out_proj.weight", TW, TW);
        blk.attn_out_proj_b = T1f(p + ".attn.out_proj.bias", TW);
        blk.ln1_w = T1f(p + ".ln_1.weight", TW);
        blk.ln1_b = T1f(p + ".ln_1.bias", TW);
        blk.ln2_w = T1f(p + ".ln_2.weight", TW);
        blk.ln2_b = T1f(p + ".ln_2.bias", TW);
        blk.mlp_fc1_w = T2(p + ".mlp.fc1.weight", TW, TW * 4);
        blk.mlp_fc1_b = T1f(p + ".mlp.fc1.bias", TW * 4);
        blk.mlp_fc2_w = T2(p + ".mlp.fc2.weight", TW * 4, TW);
        blk.mlp_fc2_b = T1f(p + ".mlp.fc2.bias", TW);
    }

    // ── Fusion encoder ───────────────────────────────────────────────────
    model.fenc.layers.resize(hp.fenc_layers);
    for (int i = 0; i < hp.fenc_layers; ++i) {
        auto& ly = model.fenc.layers[i];
        auto p = "fenc.layers." + std::to_string(i);
        // self-attention
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        // cross-attention
        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_kv_w = nullptr;  // fused in_proj for MHA
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);
        // FFN
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w = T1f(p + ".norm3.weight", D);
        ly.norm3_b = T1f(p + ".norm3.bias", D);
    }

    // ── DETR decoder ─────────────────────────────────────────────────────
    model.ddec.layers.resize(hp.ddec_layers);
    model.ddec.query_embed = T2f("ddec.query_embed.weight", D, NQ);
    model.ddec.presence_token = T2f("ddec.presence_token.weight", D, 1);

    // Reference points, norms, bbox embed, ref_point_head, boxRPB, presence_head
    // These use the exact checkpoint names after renaming
    T2f("ddec.reference_points.weight", 4, NQ);
    T1f("ddec.norm.weight", D);
    T1f("ddec.norm.bias", D);

    // bbox_embed MLP (3 layers: 256→256→256→4)
    for (int j = 0; j < 3; ++j) {
        int out = (j == 2) ? 4 : D;
        auto bp = "ddec.bbox_embed.layers." + std::to_string(j);
        T2(bp + ".weight", D, out);
        T1f(bp + ".bias", out);
    }

    // ref_point_head MLP (2 layers: 512→256→256)
    T2("ddec.ref_point_head.layers.0.weight", 512, D);
    T1f("ddec.ref_point_head.layers.0.bias", D);
    T2("ddec.ref_point_head.layers.1.weight", D, D);
    T1f("ddec.ref_point_head.layers.1.bias", D);

    // boxRPB MLPs (x and y, each 2 layers)
    for (const auto& axis : {"x", "y"}) {
        auto bp = std::string("ddec.boxRPB_embed_") + axis;
        T2(bp + ".layers.0.weight", 2, D);
        T1f(bp + ".layers.0.bias", D);
        T2(bp + ".layers.1.weight", D, hp.ddec_heads);
        T1f(bp + ".layers.1.bias", hp.ddec_heads);
    }

    // presence_token_head MLP (3 layers: 256→256→256→1)
    for (int j = 0; j < 3; ++j) {
        int out = (j == 2) ? 1 : D;
        auto bp = "ddec.presence_token_head.layers." + std::to_string(j);
        T2(bp + ".weight", D, out);
        T1f(bp + ".bias", out);
    }
    T1f("ddec.presence_token_out_norm.weight", D);
    T1f("ddec.presence_token_out_norm.bias", D);

    // DETR decoder layers
    for (int i = 0; i < hp.ddec_layers; ++i) {
        auto& ly = model.ddec.layers[i];
        auto p = "ddec.layers." + std::to_string(i);
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);

        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);

        ly.ca_text_q_w = T2(p + ".ca_text.in_proj_weight", D, 3 * D);
        ly.ca_text_q_b = T1f(p + ".ca_text.in_proj_bias", 3 * D);
        ly.ca_text_out_w = T2(p + ".ca_text.out_proj.weight", D, D);
        ly.ca_text_out_b = T1f(p + ".ca_text.out_proj.bias", D);
        ly.norm3_w = T1f(p + ".norm_ca_text.weight", D);
        ly.norm3_b = T1f(p + ".norm_ca_text.bias", D);

        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm4_w = T1f(p + ".norm3.weight", D);
        ly.norm4_b = T1f(p + ".norm3.bias", D);
    }

    // ── DotProductScoring ────────────────────────────────────────────────
    reg("scoring.prompt_proj.weight", D, D);
    reg1("scoring.prompt_proj.bias", D);
    reg("scoring.hs_proj.weight", D, D);
    reg1("scoring.hs_proj.bias", D);
    reg("scoring.prompt_mlp.layers.0.weight", D, FFN);
    reg1("scoring.prompt_mlp.layers.0.bias", FFN);
    reg("scoring.prompt_mlp.layers.1.weight", FFN, D);
    reg1("scoring.prompt_mlp.layers.1.bias", D);
    reg1("scoring.prompt_mlp.out_norm.weight", D);
    reg1("scoring.prompt_mlp.out_norm.bias", D);

    // ── Geometry encoder ───────────────────────────────────────────────────
    model.geom_enc.layers.resize(hp.geom_layers);

    // Direct projections
    model.geom_enc.point_proj_w = T2("geom.points_direct_project.weight", 2, D);
    model.geom_enc.point_proj_b = T1f("geom.points_direct_project.bias", D);
    model.geom_enc.box_proj_w = T2("geom.boxes_direct_project.weight", 4, D);
    model.geom_enc.box_proj_b = T1f("geom.boxes_direct_project.bias", D);
    // Pooling projections
    model.geom_enc.point_pool_proj_w = T2("geom.points_pool_project.weight", D, D);
    model.geom_enc.point_pool_proj_b = T1f("geom.points_pool_project.bias", D);
    model.geom_enc.box_pool_proj_w = T4("geom.boxes_pool_project.weight", 7, 7, D, D);
    model.geom_enc.box_pool_proj_b = T1f("geom.boxes_pool_project.bias", D);
    // Positional encoding projections
    model.geom_enc.point_pos_proj_w = T2("geom.points_pos_enc_project.weight", D, D);
    model.geom_enc.point_pos_proj_b = T1f("geom.points_pos_enc_project.bias", D);
    model.geom_enc.box_pos_proj_w = T2("geom.boxes_pos_enc_project.weight", 258, D);
    model.geom_enc.box_pos_proj_b = T1f("geom.boxes_pos_enc_project.bias", D);
    // Label and CLS
    model.geom_enc.type_embed = T2f("geom.label_embed.weight", D, 2);
    model.geom_enc.cls_token = T2f("geom.cls_embed.weight", D, 1);
    // Final projection + norms
    model.geom_enc.post_proj_w = T2("geom.final_proj.weight", D, D);
    model.geom_enc.post_proj_b = T1f("geom.final_proj.bias", D);
    model.geom_enc.norm_w = T1f("geom.norm.weight", D);
    model.geom_enc.norm_b = T1f("geom.norm.bias", D);
    model.geom_enc.encode_norm_w = T1f("geom.encode_norm.weight", D);
    model.geom_enc.encode_norm_b = T1f("geom.encode_norm.bias", D);
    model.geom_enc.img_pre_norm_w = T1f("geom.img_pre_norm.weight", D);
    model.geom_enc.img_pre_norm_b = T1f("geom.img_pre_norm.bias", D);

    for (int i = 0; i < hp.geom_layers; ++i) {
        auto& ly = model.geom_enc.layers[i];
        auto p = "geom.layers." + std::to_string(i);
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w = T1f(p + ".norm3.weight", D);
        ly.norm3_b = T1f(p + ".norm3.bias", D);
    }

    // ── Segmentation head ────────────────────────────────────────────────
    // Pixel decoder (3 conv layers + norms)
    for (int i = 0; i < 3; ++i) {
        auto si = std::to_string(i);
        model.seg_head.up_conv_w[i] = T4("seg.pixel_decoder.conv_layers." + si + ".weight", 3, 3, D, D);
        model.seg_head.up_conv_b[i] = T1f("seg.pixel_decoder.conv_layers." + si + ".bias", D);
        model.seg_head.up_norm_w[i] = T1f("seg.pixel_decoder.norms." + si + ".weight", D);
        model.seg_head.up_norm_b[i] = T1f("seg.pixel_decoder.norms." + si + ".bias", D);
    }

    // Mask predictor (3-layer MLP: 256→256→256→256)
    for (int j = 0; j < 3; ++j) {
        auto bp = "seg.mask_predictor.mask_embed.layers." + std::to_string(j);
        model.seg_head.mask_embed_w = T2(bp + ".weight", D, D);  // overwritten but last one
        model.seg_head.mask_embed_b = T1f(bp + ".bias", D);
    }
    // Re-register properly: all 3 layers with unique names are already in tensors map
    // The struct only has one pointer — use the tensors map at runtime
    // For now, just ensure all 6 tensors are registered (they are via the loop above —
    // each T2/T1f call registers under unique names)

    // Cross-attention to prompt
    model.seg_head.ca_prompt_q_w = T2("seg.cross_attend_prompt.in_proj_weight", D, 3 * D);
    model.seg_head.ca_prompt_q_b = T1f("seg.cross_attend_prompt.in_proj_bias", 3 * D);
    model.seg_head.ca_prompt_out_w = T2("seg.cross_attend_prompt.out_proj.weight", D, D);
    model.seg_head.ca_prompt_out_b = T1f("seg.cross_attend_prompt.out_proj.bias", D);

    // Cross-attn norm
    reg1("seg.cross_attn_norm.weight", D);
    reg1("seg.cross_attn_norm.bias", D);

    // Instance and semantic seg heads (Conv 1x1)
    reg4("seg.instance_seg_head.weight", 1, 1, D, D);
    reg1("seg.instance_seg_head.bias", D);
    reg4("seg.semantic_seg_head.weight", 1, 1, D, 1);
    reg1("seg.semantic_seg_head.bias", 1);

    } // end if (!hp.visual_only) — detector-only tensors

    // ── SAM prompt encoder ───────────────────────────────────────────────
    model.sam_pe.pe_gaussian = T2f("sam_pe.pe_gaussian", 2, 128);
    for (int i = 0; i < 4; ++i)
        model.sam_pe.point_embed[i] = T2f("sam_pe.point_embeddings." + std::to_string(i) + ".weight", D, 1);
    model.sam_pe.not_a_point_embed = T2f("sam_pe.not_a_point_embed.weight", D, 1);
    model.sam_pe.no_mask_embed = T2f("sam_pe.no_mask_embed.weight", D, 1);

    // mask_downscaling: sequential with numeric indices
    model.sam_pe.mask_ds_conv_w[0] = T4("sam_pe.mask_ds.0.weight", 2, 2, 1, 4);
    model.sam_pe.mask_ds_conv_b[0] = T1f("sam_pe.mask_ds.0.bias", 4);
    model.sam_pe.mask_ds_norm_w[0] = T1f("sam_pe.mask_ds.1.weight", 4);
    model.sam_pe.mask_ds_norm_b[0] = T1f("sam_pe.mask_ds.1.bias", 4);
    model.sam_pe.mask_ds_conv_w[1] = T4("sam_pe.mask_ds.3.weight", 2, 2, 4, 16);
    model.sam_pe.mask_ds_conv_b[1] = T1f("sam_pe.mask_ds.3.bias", 16);
    model.sam_pe.mask_ds_norm_w[1] = T1f("sam_pe.mask_ds.4.weight", 16);
    model.sam_pe.mask_ds_norm_b[1] = T1f("sam_pe.mask_ds.4.bias", 16);
    model.sam_pe.mask_ds_conv_w[2] = T4("sam_pe.mask_ds.6.weight", 1, 1, 16, D);
    model.sam_pe.mask_ds_conv_b[2] = T1f("sam_pe.mask_ds.6.bias", D);

    // ── SAM mask decoder ─────────────────────────────────────────────────
    model.sam_dec.iou_token = T2f("sam_dec.iou_token.weight", D, 1);
    model.sam_dec.mask_tokens = T2f("sam_dec.mask_tokens.weight", D, 4);
    model.sam_dec.obj_score_token = T2f("sam_dec.obj_score_token.weight", D, 1);

    model.sam_dec.twoway_blocks.resize(hp.sam_dec_depth);
    for (int i = 0; i < hp.sam_dec_depth; ++i) {
        auto& blk = model.sam_dec.twoway_blocks[i];
        auto p = "sam_dec.twoway." + std::to_string(i);

        auto reg_attn = [&](sam3_sam_attn& a, const std::string& pfx, int in_dim, int out_dim) {
            a.q_w = T2(pfx + ".q_proj.weight", in_dim, out_dim);
            a.q_b = T1f(pfx + ".q_proj.bias", out_dim);
            a.k_w = T2(pfx + ".k_proj.weight", in_dim, out_dim);
            a.k_b = T1f(pfx + ".k_proj.bias", out_dim);
            a.v_w = T2(pfx + ".v_proj.weight", in_dim, out_dim);
            a.v_b = T1f(pfx + ".v_proj.bias", out_dim);
            a.out_w = T2(pfx + ".out_proj.weight", out_dim, in_dim);
            a.out_b = T1f(pfx + ".out_proj.bias", in_dim);
        };

        reg_attn(blk.self_attn, p + ".sa", D, D);
        reg_attn(blk.ca_tok2img, p + ".cross_attn_token_to_image", D, 128);
        reg_attn(blk.ca_img2tok, p + ".cross_attn_image_to_token", D, 128);

        blk.norm1_w = T1f(p + ".norm1.weight", D);
        blk.norm1_b = T1f(p + ".norm1.bias", D);
        blk.norm2_w = T1f(p + ".norm2.weight", D);
        blk.norm2_b = T1f(p + ".norm2.bias", D);
        blk.norm3_w = T1f(p + ".norm3.weight", D);
        blk.norm3_b = T1f(p + ".norm3.bias", D);
        blk.norm4_w = T1f(p + ".norm4.weight", D);
        blk.norm4_b = T1f(p + ".norm4.bias", D);

        blk.mlp_fc1_w = T2(p + ".mlp.lin1.weight", D, FFN);
        blk.mlp_fc1_b = T1f(p + ".mlp.lin1.bias", FFN);
        blk.mlp_fc2_w = T2(p + ".mlp.lin2.weight", FFN, D);
        blk.mlp_fc2_b = T1f(p + ".mlp.lin2.bias", D);
    }

    // final attention
    auto reg_sam_attn = [&](sam3_sam_attn& a, const std::string& pfx, int in_dim, int out_dim) {
        a.q_w = T2(pfx + ".q_proj.weight", in_dim, out_dim);
        a.q_b = T1f(pfx + ".q_proj.bias", out_dim);
        a.k_w = T2(pfx + ".k_proj.weight", in_dim, out_dim);
        a.k_b = T1f(pfx + ".k_proj.bias", out_dim);
        a.v_w = T2(pfx + ".v_proj.weight", in_dim, out_dim);
        a.v_b = T1f(pfx + ".v_proj.bias", out_dim);
        a.out_w = T2(pfx + ".out_proj.weight", out_dim, in_dim);
        a.out_b = T1f(pfx + ".out_proj.bias", in_dim);
    };
    reg_sam_attn(model.sam_dec.final_attn, "sam_dec.final_attn", D, 128);
    model.sam_dec.final_norm_w = T1f("sam_dec.final_norm.weight", D);
    model.sam_dec.final_norm_b = T1f("sam_dec.final_norm.bias", D);

    // upscaling
    model.sam_dec.up1_w = T4("sam_dec.upscale.0.weight", 2, 2, 64, D);
    model.sam_dec.up1_b = T1f("sam_dec.upscale.0.bias", 64);
    model.sam_dec.up1_norm_w = T1f("sam_dec.upscale.1.weight", 64);
    model.sam_dec.up1_norm_b = T1f("sam_dec.upscale.1.bias", 64);
    model.sam_dec.up2_w = T4("sam_dec.upscale.3.weight", 2, 2, 32, 64);
    model.sam_dec.up2_b = T1f("sam_dec.upscale.3.bias", 32);

    // high-res feature convolutions
    model.sam_dec.conv_s0_w = T4("sam_dec.conv_s0.weight", 1, 1, D, 32);
    model.sam_dec.conv_s0_b = T1f("sam_dec.conv_s0.bias", 32);
    model.sam_dec.conv_s1_w = T4("sam_dec.conv_s1.weight", 1, 1, D, 64);
    model.sam_dec.conv_s1_b = T1f("sam_dec.conv_s1.bias", 64);

    // hypernetwork MLPs (4 × 3 layers: 256→256→256→32)
    for (int m = 0; m < 4; ++m) {
        for (int j = 0; j < 3; ++j) {
            int in_d = D, out_d = (j == 2) ? 32 : D;
            auto bp = "sam_dec.hyper." + std::to_string(m) + ".layers." + std::to_string(j);
            model.sam_dec.hyper_w[m][j] = T2(bp + ".weight", in_d, out_d);
            model.sam_dec.hyper_b[m][j] = T1f(bp + ".bias", out_d);
        }
    }

    // IoU prediction head (3 layers: 256→256→256→4)
    for (int j = 0; j < 3; ++j) {
        int out_d = (j == 2) ? 4 : D;
        auto bp = "sam_dec.iou_prediction_head.layers." + std::to_string(j);
        model.sam_dec.iou_head_w[j] = T2(bp + ".weight", D, out_d);
        model.sam_dec.iou_head_b[j] = T1f(bp + ".bias", out_d);
    }

    // object score head (3 layers: 256→256→256→1)
    for (int j = 0; j < 3; ++j) {
        int out_d = (j == 2) ? 1 : D;
        auto bp = "sam_dec.pred_obj_score_head.layers." + std::to_string(j);
        model.sam_dec.obj_head_w[j] = T2(bp + ".weight", D, out_d);
        model.sam_dec.obj_head_b[j] = T1f(bp + ".bias", out_d);
    }

    // no_mem_embed is nominally a tracker tensor, but the single-image PVS
    // path also conditions image features with it (features + no_mem_embed
    // emulates "no memory frames" for the SAM decoder) -- so it stays.
    T3f("no_mem_embed", D, 1, 1);

    // The remaining tracker weights (mem_enc.*, mem_attn.*, obj_ptr*,
    // no_mem_pos_enc, no_obj_*, trk_mask_ds.*) exist in the .ggml container
    // but are NOT registered: sam3cpplib does not port video tracking.
    // sam3_load_tensors seeks past unregistered records.
}

// Load tensors from the binary file into the already-registered ggml tensors
static bool sam3_load_tensors(std::ifstream& fin, sam3_model& model, int n_tensors) {
    int n_loaded = 0;
    int n_skipped = 0;
    for (int t = 0; t < n_tensors; ++t) {
        int32_t n_dims, name_len, dtype;
        fin.read(reinterpret_cast<char*>(&n_dims), 4);
        fin.read(reinterpret_cast<char*>(&name_len), 4);
        fin.read(reinterpret_cast<char*>(&dtype), 4);
        if (fin.fail()) break;

        // Read shape (reversed in file)
        std::vector<int64_t> shape(n_dims);
        for (int i = 0; i < n_dims; ++i) {
            int32_t d;
            fin.read(reinterpret_cast<char*>(&d), 4);
            shape[i] = d;
        }

        // Read name
        std::string name(name_len, '\0');
        fin.read(&name[0], name_len);

        // Skip to 32-byte alignment
        size_t pos = fin.tellg();
        size_t pad = (32 - pos % 32) % 32;
        if (pad > 0) fin.seekg(pad, std::ios::cur);

        // Compute element count and data size from file dtype -- needed both
        // for reading a registered tensor and for seeking past a skipped one
        // (the embedded BPE vocab sits after the tensor records in the same
        // stream, so skipped bytes must be skipped exactly).
        int64_t n_el = 1;
        for (auto d : shape) n_el *= d;

        const ggml_type file_type = static_cast<ggml_type>(dtype);
        size_t bytes;
        if (ggml_is_quantized(file_type)) {
            const int64_t n_rows = n_el / shape[0];
            bytes = ggml_row_size(file_type, shape[0]) * n_rows;
        } else {
            const size_t file_elem_size = (file_type == GGML_TYPE_F16) ? 2 : 4;
            bytes = n_el * file_elem_size;
        }

        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            // Unregistered records are skipped: the tracker weights that SAM3
            // containers carry (mem_enc.*, mem_attn.*, obj_ptr*, ...) are
            // deliberately not registered, and SAM3_TRT_SKIP_GGML_WEIGHTS
            // additionally unregisters everything but TRT request helpers.
            fin.seekg(bytes, std::ios::cur);
            ++n_skipped;
            continue;
        }

        auto* tensor = it->second;

        // Read into a temporary CPU buffer, then copy to backend
        std::vector<char> buf(bytes);
        fin.read(buf.data(), bytes);
        if (fin.fail()) {
            fprintf(stderr, "%s: failed to read tensor '%s'\n", __func__, name.c_str());
            return false;
        }

        // If the file dtype matches the registered tensor type, copy directly.
        // Otherwise, convert as needed (f16<->f32, or dequantize->f32).
        if (file_type == tensor->type) {
            ggml_backend_tensor_set(tensor, buf.data(), 0, bytes);
        } else if (file_type == GGML_TYPE_F16 && tensor->type == GGML_TYPE_F32) {
            // Convert f16 → f32
            std::vector<float> f32_buf(n_el);
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(buf.data()),
                                  f32_buf.data(), n_el);
            ggml_backend_tensor_set(tensor, f32_buf.data(), 0, n_el * sizeof(float));
        } else if (file_type == GGML_TYPE_F32 && tensor->type == GGML_TYPE_F16) {
            // Convert f32 → f16
            std::vector<ggml_fp16_t> f16_buf(n_el);
            ggml_fp32_to_fp16_row(reinterpret_cast<const float*>(buf.data()),
                                  f16_buf.data(), n_el);
            ggml_backend_tensor_set(tensor, f16_buf.data(), 0, n_el * sizeof(ggml_fp16_t));
        } else if (ggml_is_quantized(file_type) && tensor->type == GGML_TYPE_F32) {
            // Dequantize → f32 (e.g., embedding stored quantized but registered as f32)
            const auto * traits = ggml_get_type_traits(file_type);
            if (!traits->to_float) {
                fprintf(stderr, "%s: no dequantize function for '%s' (type=%s)\n",
                        __func__, name.c_str(), ggml_type_name(file_type));
                return false;
            }
            std::vector<float> f32_buf(n_el);
            traits->to_float(buf.data(), f32_buf.data(), n_el);
            ggml_backend_tensor_set(tensor, f32_buf.data(), 0, n_el * sizeof(float));
        } else {
            fprintf(stderr, "%s: unsupported type conversion for '%s' (file=%s, tensor=%s)\n",
                    __func__, name.c_str(), ggml_type_name(file_type),
                    ggml_type_name(tensor->type));
            return false;
        }
        n_loaded++;
    }

    if (n_skipped > 0) {
        fprintf(stderr, "%s: loaded %d tensors, skipped %d unregistered (registered %zu)\n",
                __func__, n_loaded, n_skipped, model.tensors.size());
    } else {
        fprintf(stderr, "%s: loaded %d tensors (registered %zu)\n",
                __func__, n_loaded, model.tensors.size());
    }

    if (n_loaded != (int)model.tensors.size()) {
        fprintf(stderr, "%s: tensor count mismatch: file has %d, model registered %zu\n",
                __func__, n_loaded, model.tensors.size());
        return false;
    }
    return true;
}

/*****************************************************************************
** Model loading — public API
*****************************************************************************/

std::shared_ptr<sam3_model> sam3_load_model(const sam3_params& params) {
    std::string load_path = params.model_path;
    bool expect_trt_runtime_data = false;
#ifdef SAM3_TRT_ENCODER
    if (params.trt.enabled && !params.trt.runtime_data.empty()) {
        load_path = params.trt.runtime_data;
        expect_trt_runtime_data = true;
    } else if (const char* p = getenv("SAM3_TRT_RUNTIME_DATA_PATH")) {
        if (*p != '\0') {
            load_path = p;
            expect_trt_runtime_data = true;
        }
    }
#endif
    fprintf(stderr, "%s: loading %s from '%s'\n", __func__,
            expect_trt_runtime_data ? "TensorRT runtime data" : "model",
            load_path.c_str());

    std::ifstream fin(load_path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, load_path.c_str());
        return nullptr;
    }

    // ── Read + validate header ───────────────────────────────────────────
    uint32_t magic;
    int32_t version, ftype, n_tensors;
    fin.read(reinterpret_cast<char*>(&magic), 4);
    fin.read(reinterpret_cast<char*>(&version), 4);
    fin.read(reinterpret_cast<char*>(&ftype), 4);
    fin.read(reinterpret_cast<char*>(&n_tensors), 4);

    const bool is_trt_runtime_data = magic == SAM3_TRT_MAGIC;
    if (magic == SAM2_MAGIC) {
        fprintf(stderr, "%s: SAM2/EdgeTAM model files are not supported by sam3cpplib "
                        "(video tracking was deliberately not ported; use the upstream "
                        "sam3.cpp library for those)\n", __func__);
        return nullptr;
    }
    if (magic != SAM3_MAGIC && !is_trt_runtime_data) {
        fprintf(stderr, "%s: unknown magic: 0x%08x (expected sam3=0x%08x or s3rt=0x%08x)\n",
                __func__, magic, SAM3_MAGIC, SAM3_TRT_MAGIC);
        return nullptr;
    }
    if (expect_trt_runtime_data && !is_trt_runtime_data) {
        fprintf(stderr, "%s: '%s' is not a TensorRT runtime-data sidecar\n",
                __func__, load_path.c_str());
        return nullptr;
    }
    if (is_trt_runtime_data &&
        !(params.trt.enabled || getenv("SAM3_TRT_ENCODER") != nullptr)) {
        fprintf(stderr, "%s: TensorRT runtime data requires TensorRT to be enabled\n", __func__);
        return nullptr;
    }
    if (version != SAM3_FILE_VERSION) {
        fprintf(stderr, "%s: unsupported SAM3 version: %d (expected %d)\n",
                __func__, version, SAM3_FILE_VERSION);
        return nullptr;
    }
    fprintf(stderr, "%s: %s format v%d, ftype %d, %d tensors\n",
            __func__, is_trt_runtime_data ? "SAM3 TRT runtime-data" : "SAM3",
            version, ftype, n_tensors);

    auto model = std::make_shared<sam3_model>();
    {
        ggml_type wtype;
        switch (ftype) {
            case 0:  wtype = GGML_TYPE_F32;  break;
            case 1:  wtype = GGML_TYPE_F16;  break;
            case 2:  wtype = GGML_TYPE_Q4_0; break;
            case 3:  wtype = GGML_TYPE_Q4_1; break;
            case 8:  wtype = GGML_TYPE_Q8_0; break;
            default:
                fprintf(stderr, "%s: unsupported ftype: %d\n", __func__, ftype);
                return nullptr;
        }
        model->weight_type = wtype;
    }

    // ── Read hyperparameters ─────────────────────────────────────────────
    if (!sam3_load_hparams(fin, model->hparams)) {
        fprintf(stderr, "%s: failed to read SAM3 hyperparameters\n", __func__);
        return nullptr;
    }
    sam3_print_hparams(model->hparams);

    // ── Init backend ─────────────────────────────────────────────────────
#ifdef GGML_USE_METAL
    if (params.use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model->backend = ggml_backend_metal_init();
    }
#endif
    // Non-Metal GPU (CUDA on Linux) via the device registry (gemma4 local
    // ext). Ops the GPU backend lacks are lowered at graph-build time — see
    // sam3_flash_attn_ext / sam3_win_part.
    if (!model->backend && params.use_gpu) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev) {
            model->backend = ggml_backend_dev_init(dev, nullptr);
            if (model->backend) {
                fprintf(stderr, "%s: using GPU backend: %s (%s)\n", __func__,
                        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                if (g_gpu_backend) {
                    fprintf(stderr, "%s: warning: replacing active GPU registration "
                            "(one GPU model per process is supported)\n", __func__);
                }
                g_gpu_backend = model->backend;
            }
        }
    }
    if (!model->backend) {
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        model->backend = ggml_backend_cpu_init();
    }
    if (!model->backend) {
        fprintf(stderr, "%s: failed to init backend\n", __func__);
        return nullptr;
    }

    // ── Create ggml context (no_alloc — we use backend_alloc_ctx_tensors)
    // Estimate: ~3000 tensors, generous overhead
    size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    model->ctx = ggml_init(ctx_params);
    if (!model->ctx) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return nullptr;
    }

    // ── Register all tensor shapes ───────────────────────────────────────
    bool skip_ggml_weights = is_trt_runtime_data ||
                             getenv("SAM3_TRT_SKIP_GGML_WEIGHTS") != nullptr;
#ifdef SAM3_TRT_ENCODER
    // Programmatic TRT config: captured here (before any engine lookup);
    // env vars stay authoritative when params.trt.enabled is false.
    sam3_trt_set_config(params.trt);
    if (params.trt.enabled && params.trt.skip_ggml_weights) {
        skip_ggml_weights = true;
    }
#endif
    if (skip_ggml_weights) {
        model->trt_only_weights = true;
        fprintf(stderr, "%s: TRT-only weights requested -- loading only prompt/geometry helpers; "
                        "ggml inference for encode/PCS/PVS is DISABLED (TensorRT engines "
                        "required, no fallback)\n", __func__);
    }
    sam3_register_tensors(*model);
    fprintf(stderr, "%s: registered %zu tensors\n", __func__, model->tensors.size());

    // ── Allocate backend buffer for all tensors ──────────────────────────
    model->buffer = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);
    if (!model->buffer) {
        fprintf(stderr, "%s: failed to allocate tensor buffer\n", __func__);
        return nullptr;
    }
    fprintf(stderr, "%s: buffer size = %.2f MB\n", __func__,
            ggml_backend_buffer_get_size(model->buffer) / (1024.0 * 1024.0));

    // ── Load tensor data from file ───────────────────────────────────────
    if (!sam3_load_tensors(fin, *model, n_tensors)) {
        fprintf(stderr, "%s: failed to load tensors\n", __func__);
        return nullptr;
    }

    // ── Load embedded BPE tokenizer ──────────────────────────────────────
    if (!model->hparams.visual_only) {
        if (!sam3_load_bpe_vocab_from_stream(fin, model->tokenizer)) {
            fprintf(stderr, "%s: failed to load embedded tokenizer\n", __func__);
            return nullptr;
        }
    } else {
        fprintf(stderr, "%s: visual-only model — skipping tokenizer\n", __func__);
    }

    fprintf(stderr, "%s: model loaded successfully\n", __func__);
    return model;
}

void sam3_free_model(sam3_model& model) {
    if (g_gpu_backend && g_gpu_backend == model.backend) {
        g_gpu_backend = nullptr;
    }
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.backend) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
    }
}

bool sam3_is_visual_only(const sam3_model& model) {
    return model.hparams.visual_only != 0;
}

sam3_model_type sam3_get_model_type(const sam3_model& model) {
    return model.hparams.model_type;
}
