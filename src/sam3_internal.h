// sam3cpplib internal header -- shared across all src/ translation units.
// Ported from tools/sam3.cpp @ 049884b (wip/trt-phase1, upstream
// PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
//
// Everything here is implementation detail: consumers include only
// include/sam3cpp/sam3.h.
#pragma once

#define _USE_MATH_DEFINES

#include "sam3cpp/sam3.h"

/* ggml */
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

/* C++ standard library */
#include <algorithm>
#include <cassert>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

/* Logging: 0=silent, 1=summary timing, 2=verbose progress. Override with
   -DSAM3_LOG_LEVEL=0 at build time for zero-overhead silent builds. */
#ifndef SAM3_LOG_LEVEL
#define SAM3_LOG_LEVEL 1
#endif
#define SAM3_LOG(level, ...) \
    do { if ((level) <= SAM3_LOG_LEVEL) fprintf(stderr, __VA_ARGS__); } while (0)

/*****************************************************************************
** Constants
*****************************************************************************/

static constexpr uint32_t SAM3_MAGIC     = 0x73616D33;  // "sam3"
static constexpr uint32_t SAM2_MAGIC     = 0x73616D32;  // "sam2"
static constexpr uint32_t SAM3_TOK_MAGIC = 0x746F6B00;  // "tok\0"
static constexpr int      SAM3_FILE_VERSION = 3;
static constexpr int      SAM2_VERSION   = 1;



/*****************************************************************************
** Internal Data Types -- Hyperparameters
*****************************************************************************/

struct sam3_hparams {
    // ── Model type ───────────────────────────────────────────────────────
    sam3_model_type model_type = SAM3_MODEL_SAM3;

    // ── SAM3 fields (unchanged) ─────────────────────────────────────────
    int32_t img_size        = 1008;
    int32_t patch_size      = 14;
    int32_t vit_embed_dim   = 1024;
    int32_t vit_depth       = 32;
    int32_t vit_num_heads   = 16;
    int32_t vit_mlp_dim     = 4736;  // 1024 * 4.625
    int32_t vit_window_size = 24;
    int32_t n_global_attn   = 4;
    int32_t global_attn_idx[4] = {7, 15, 23, 31};

    int32_t text_width      = 1024;
    int32_t text_heads      = 16;
    int32_t text_layers     = 24;
    int32_t text_ctx_len    = 32;
    int32_t text_vocab_size = 49408;
    int32_t text_out_dim    = 256;

    int32_t neck_dim        = 256;

    int32_t fenc_layers     = 6;
    int32_t fenc_heads      = 8;
    int32_t fenc_ffn_dim    = 2048;

    int32_t ddec_layers       = 6;
    int32_t ddec_heads        = 8;
    int32_t ddec_ffn_dim      = 2048;
    int32_t ddec_num_queries  = 200;

    int32_t geom_layers        = 3;
    int32_t n_presence_tokens  = 1;
    int32_t n_geom_queries     = 4;

    int32_t sam_embed_dim     = 256;
    int32_t sam_dec_depth     = 2;
    int32_t sam_n_multimask   = 3;
    int32_t sam_iou_head_depth = 3;

    int32_t mem_out_dim     = 64;
    int32_t mem_attn_layers = 4;
    int32_t num_maskmem     = 7;
    int32_t max_obj_ptrs    = 16;

    int32_t n_amb_experts   = 2;

    int32_t visual_only     = 0;  // 1 = no text encoder / detector path

    // ── SAM3 derived helpers ────────────────────────────────────────────
    int32_t n_img_embd() const { return img_size / patch_size; }            // 72
    int32_t n_img_tokens() const { return n_img_embd() * n_img_embd(); }    // 5184
    int32_t vit_head_dim() const { return vit_embed_dim / vit_num_heads; }  // 64

    bool is_global_attn(int layer) const {
        for (int i = 0; i < n_global_attn; ++i) {
            if (global_attn_idx[i] == layer) return true;
        }
        return false;
    }

    // Feature size for the backbone (SAM3-only library: always the ViT grid).
    int32_t feat_size() const { return n_img_embd(); }
};

// Compute feat_size for an arbitrary img_size.
static inline int sam3_effective_feat_size(const sam3_hparams& hp, int img_size) {
    return img_size / hp.patch_size;
}


/*****************************************************************************
** Internal Data Types -- Layer Weight Structs
*****************************************************************************/

/*
** ── ViT Backbone ─────────────────────────────────────────────────────────
*/

struct sam3_vit_block {
    struct ggml_tensor* norm1_w   = nullptr;
    struct ggml_tensor* norm1_b   = nullptr;
    struct ggml_tensor* qkv_w     = nullptr;
    struct ggml_tensor* qkv_b     = nullptr;
    struct ggml_tensor* proj_w    = nullptr;
    struct ggml_tensor* proj_b    = nullptr;
    struct ggml_tensor* norm2_w   = nullptr;
    struct ggml_tensor* norm2_b   = nullptr;
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
    struct ggml_tensor* freqs_cis = nullptr;  // [N, 32, 2] RoPE
};

struct sam3_vit {
    struct ggml_tensor*          patch_embed_w = nullptr;  // [patch, patch, 3, embed]
    struct ggml_tensor*          pos_embed     = nullptr;  // [embed, 24, 24, 1]
    struct ggml_tensor*          ln_pre_w      = nullptr;
    struct ggml_tensor*          ln_pre_b      = nullptr;
    std::vector<sam3_vit_block>  blocks;
};

/*
** ── Neck (SimpleFPN) ─────────────────────────────────────────────────────
*/

struct sam3_neck_scale {
    struct ggml_tensor* deconv1_w  = nullptr;
    struct ggml_tensor* deconv1_b  = nullptr;
    struct ggml_tensor* deconv2_w  = nullptr;  // only for 4x scale
    struct ggml_tensor* deconv2_b  = nullptr;
    struct ggml_tensor* conv1x1_w  = nullptr;
    struct ggml_tensor* conv1x1_b  = nullptr;
    struct ggml_tensor* conv3x3_w  = nullptr;
    struct ggml_tensor* conv3x3_b  = nullptr;
};

struct sam3_neck {
    sam3_neck_scale scales[4];
    struct ggml_tensor* norms_w[4] = {};
    struct ggml_tensor* norms_b[4] = {};
};

/*
** ── Text Encoder ─────────────────────────────────────────────────────────
*/

struct sam3_text_block {
    struct ggml_tensor* attn_in_proj_w  = nullptr;
    struct ggml_tensor* attn_in_proj_b  = nullptr;
    struct ggml_tensor* attn_out_proj_w = nullptr;
    struct ggml_tensor* attn_out_proj_b = nullptr;
    struct ggml_tensor* ln1_w           = nullptr;
    struct ggml_tensor* ln1_b           = nullptr;
    struct ggml_tensor* ln2_w           = nullptr;
    struct ggml_tensor* ln2_b           = nullptr;
    struct ggml_tensor* mlp_fc1_w       = nullptr;
    struct ggml_tensor* mlp_fc1_b       = nullptr;
    struct ggml_tensor* mlp_fc2_w       = nullptr;
    struct ggml_tensor* mlp_fc2_b       = nullptr;
    struct ggml_tensor* ls1             = nullptr;  // LayerScale
    struct ggml_tensor* ls2             = nullptr;
};

struct sam3_text_encoder {
    struct ggml_tensor* token_embed_w = nullptr;  // [vocab, width]
    struct ggml_tensor* pos_embed     = nullptr;  // [ctx_len, width]
    struct ggml_tensor* ln_final_w    = nullptr;
    struct ggml_tensor* ln_final_b    = nullptr;
    struct ggml_tensor* resizer_w     = nullptr;  // [out_dim, width]
    struct ggml_tensor* resizer_b     = nullptr;
    // Note: text_projection ([width, proj_dim]) exists in the checkpoint but is
    // intentionally not loaded. In SAM3, VETextEncoder discards the pooled output
    // that text_projection operates on — only the full token sequence (through
    // resizer) is used for downstream fusion/decoding.
    std::vector<sam3_text_block> blocks;
};

/*
** ── Fusion Encoder ───────────────────────────────────────────────────────
*/

struct sam3_fenc_layer {
    // self-attention
    struct ggml_tensor* sa_in_proj_w  = nullptr;
    struct ggml_tensor* sa_in_proj_b  = nullptr;
    struct ggml_tensor* sa_out_proj_w = nullptr;
    struct ggml_tensor* sa_out_proj_b = nullptr;
    struct ggml_tensor* norm1_w       = nullptr;
    struct ggml_tensor* norm1_b       = nullptr;
    // cross-attention to prompt tokens
    struct ggml_tensor* ca_q_w        = nullptr;
    struct ggml_tensor* ca_q_b        = nullptr;
    struct ggml_tensor* ca_kv_w       = nullptr;
    struct ggml_tensor* ca_kv_b       = nullptr;
    struct ggml_tensor* ca_out_w      = nullptr;
    struct ggml_tensor* ca_out_b      = nullptr;
    struct ggml_tensor* norm2_w       = nullptr;
    struct ggml_tensor* norm2_b       = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w     = nullptr;
    struct ggml_tensor* ffn_fc1_b     = nullptr;
    struct ggml_tensor* ffn_fc2_w     = nullptr;
    struct ggml_tensor* ffn_fc2_b     = nullptr;
    struct ggml_tensor* norm3_w       = nullptr;
    struct ggml_tensor* norm3_b       = nullptr;
};

struct sam3_fusion_encoder {
    std::vector<sam3_fenc_layer> layers;
};

/*
** ── DETR Decoder ─────────────────────────────────────────────────────────
*/

struct sam3_ddec_layer {
    // self-attention
    struct ggml_tensor* sa_in_proj_w   = nullptr;
    struct ggml_tensor* sa_in_proj_b   = nullptr;
    struct ggml_tensor* sa_out_proj_w  = nullptr;
    struct ggml_tensor* sa_out_proj_b  = nullptr;
    struct ggml_tensor* norm1_w        = nullptr;
    struct ggml_tensor* norm1_b        = nullptr;
    // cross-attention to image
    struct ggml_tensor* ca_q_w         = nullptr;
    struct ggml_tensor* ca_q_b         = nullptr;
    struct ggml_tensor* ca_kv_w        = nullptr;
    struct ggml_tensor* ca_kv_b        = nullptr;
    struct ggml_tensor* ca_out_w       = nullptr;
    struct ggml_tensor* ca_out_b       = nullptr;
    struct ggml_tensor* norm2_w        = nullptr;
    struct ggml_tensor* norm2_b        = nullptr;
    // cross-attention to text
    struct ggml_tensor* ca_text_q_w    = nullptr;
    struct ggml_tensor* ca_text_q_b    = nullptr;
    struct ggml_tensor* ca_text_kv_w   = nullptr;
    struct ggml_tensor* ca_text_kv_b   = nullptr;
    struct ggml_tensor* ca_text_out_w  = nullptr;
    struct ggml_tensor* ca_text_out_b  = nullptr;
    struct ggml_tensor* norm3_w        = nullptr;
    struct ggml_tensor* norm3_b        = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w      = nullptr;
    struct ggml_tensor* ffn_fc1_b      = nullptr;
    struct ggml_tensor* ffn_fc2_w      = nullptr;
    struct ggml_tensor* ffn_fc2_b      = nullptr;
    struct ggml_tensor* norm4_w        = nullptr;
    struct ggml_tensor* norm4_b        = nullptr;
    // box refinement MLP (3 layers)
    struct ggml_tensor* bbox_w[3]      = {};
    struct ggml_tensor* bbox_b[3]      = {};
};

struct sam3_detr_decoder {
    struct ggml_tensor*          query_embed      = nullptr;  // [num_queries, 512]
    struct ggml_tensor*          presence_token   = nullptr;  // [1, 256]
    // DotProductScoring MLP
    struct ggml_tensor*          score_mlp_w[2]   = {};
    struct ggml_tensor*          score_mlp_b[2]   = {};
    struct ggml_tensor*          score_ln_w       = nullptr;
    struct ggml_tensor*          score_ln_b       = nullptr;
    // Presence head
    struct ggml_tensor*          presence_head_w[2] = {};
    struct ggml_tensor*          presence_head_b[2] = {};
    std::vector<sam3_ddec_layer> layers;
};

/*
** ── Geometry / Exemplar Encoder ──────────────────────────────────────────
*/

struct sam3_geom_layer {
    struct ggml_tensor* sa_in_proj_w  = nullptr;
    struct ggml_tensor* sa_in_proj_b  = nullptr;
    struct ggml_tensor* sa_out_proj_w = nullptr;
    struct ggml_tensor* sa_out_proj_b = nullptr;
    struct ggml_tensor* norm1_w       = nullptr;
    struct ggml_tensor* norm1_b       = nullptr;
    struct ggml_tensor* ca_q_w        = nullptr;
    struct ggml_tensor* ca_q_b        = nullptr;
    struct ggml_tensor* ca_kv_w       = nullptr;
    struct ggml_tensor* ca_kv_b       = nullptr;
    struct ggml_tensor* ca_out_w      = nullptr;
    struct ggml_tensor* ca_out_b      = nullptr;
    struct ggml_tensor* norm2_w       = nullptr;
    struct ggml_tensor* norm2_b       = nullptr;
    struct ggml_tensor* ffn_fc1_w     = nullptr;
    struct ggml_tensor* ffn_fc1_b     = nullptr;
    struct ggml_tensor* ffn_fc2_w     = nullptr;
    struct ggml_tensor* ffn_fc2_b     = nullptr;
    struct ggml_tensor* norm3_w       = nullptr;
    struct ggml_tensor* norm3_b       = nullptr;
};

struct sam3_geom_encoder {
    // Direct projections
    struct ggml_tensor* point_proj_w      = nullptr;  // Linear(2, D)
    struct ggml_tensor* point_proj_b      = nullptr;
    struct ggml_tensor* box_proj_w        = nullptr;  // Linear(4, D)
    struct ggml_tensor* box_proj_b        = nullptr;
    // Pooling projections
    struct ggml_tensor* point_pool_proj_w = nullptr;  // Linear(D, D)
    struct ggml_tensor* point_pool_proj_b = nullptr;
    struct ggml_tensor* box_pool_proj_w   = nullptr;  // Conv2d(D, D, 7)
    struct ggml_tensor* box_pool_proj_b   = nullptr;
    // Positional encoding projections
    struct ggml_tensor* point_pos_proj_w  = nullptr;  // Linear(D, D)
    struct ggml_tensor* point_pos_proj_b  = nullptr;
    struct ggml_tensor* box_pos_proj_w    = nullptr;  // Linear(258, 256)
    struct ggml_tensor* box_pos_proj_b    = nullptr;
    // Label and CLS embeddings
    struct ggml_tensor* type_embed        = nullptr;  // Embedding(2, D)
    struct ggml_tensor* cls_token         = nullptr;  // Embedding(1, D)
    // Final projection + norms
    struct ggml_tensor* post_proj_w       = nullptr;  // Linear(D, D)
    struct ggml_tensor* post_proj_b       = nullptr;
    struct ggml_tensor* norm_w            = nullptr;  // LayerNorm final_proj
    struct ggml_tensor* norm_b            = nullptr;
    struct ggml_tensor* encode_norm_w     = nullptr;  // LayerNorm after xfmr
    struct ggml_tensor* encode_norm_b     = nullptr;
    struct ggml_tensor* img_pre_norm_w    = nullptr;  // LayerNorm before pool
    struct ggml_tensor* img_pre_norm_b    = nullptr;
    std::vector<sam3_geom_layer> layers;
};

/*
** ── Segmentation Head (MaskFormer) ───────────────────────────────────────
*/

struct sam3_seg_head {
    struct ggml_tensor* up_conv_w[3]      = {};
    struct ggml_tensor* up_conv_b[3]      = {};
    struct ggml_tensor* up_norm_w[3]      = {};
    struct ggml_tensor* up_norm_b[3]      = {};
    struct ggml_tensor* ca_prompt_q_w     = nullptr;
    struct ggml_tensor* ca_prompt_q_b     = nullptr;
    struct ggml_tensor* ca_prompt_kv_w    = nullptr;
    struct ggml_tensor* ca_prompt_kv_b    = nullptr;
    struct ggml_tensor* ca_prompt_out_w   = nullptr;
    struct ggml_tensor* ca_prompt_out_b   = nullptr;
    struct ggml_tensor* mask_embed_w      = nullptr;
    struct ggml_tensor* mask_embed_b      = nullptr;
};

/*
** ── SAM Prompt Encoder (Tracker Path) ────────────────────────────────────
*/

struct sam3_sam_prompt_enc {
    struct ggml_tensor* pe_gaussian         = nullptr;  // [2, 128]
    struct ggml_tensor* point_embed[4]      = {};       // neg, pos, box_tl, box_br
    struct ggml_tensor* not_a_point_embed   = nullptr;  // [256]
    struct ggml_tensor* no_mask_embed       = nullptr;  // [256]
    struct ggml_tensor* mask_ds_conv_w[3]   = {};
    struct ggml_tensor* mask_ds_conv_b[3]   = {};
    struct ggml_tensor* mask_ds_norm_w[2]   = {};
    struct ggml_tensor* mask_ds_norm_b[2]   = {};
};

/*
** ── SAM Mask Decoder (Tracker Path) ──────────────────────────────────────
*/

struct sam3_sam_attn {
    struct ggml_tensor* q_w   = nullptr;
    struct ggml_tensor* q_b   = nullptr;
    struct ggml_tensor* k_w   = nullptr;
    struct ggml_tensor* k_b   = nullptr;
    struct ggml_tensor* v_w   = nullptr;
    struct ggml_tensor* v_b   = nullptr;
    struct ggml_tensor* out_w = nullptr;
    struct ggml_tensor* out_b = nullptr;
};

struct sam3_twoway_block {
    sam3_sam_attn       self_attn;
    sam3_sam_attn       ca_tok2img;
    sam3_sam_attn       ca_img2tok;
    struct ggml_tensor* norm1_w   = nullptr;
    struct ggml_tensor* norm1_b   = nullptr;
    struct ggml_tensor* norm2_w   = nullptr;
    struct ggml_tensor* norm2_b   = nullptr;
    struct ggml_tensor* norm3_w   = nullptr;
    struct ggml_tensor* norm3_b   = nullptr;
    struct ggml_tensor* norm4_w   = nullptr;
    struct ggml_tensor* norm4_b   = nullptr;
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
};

struct sam3_sam_mask_dec {
    struct ggml_tensor*           iou_token       = nullptr;  // [1, 256]
    struct ggml_tensor*           mask_tokens     = nullptr;  // [4, 256]
    struct ggml_tensor*           obj_score_token = nullptr;  // [1, 256]

    std::vector<sam3_twoway_block> twoway_blocks;             // [2]

    sam3_sam_attn                 final_attn;
    struct ggml_tensor*           final_norm_w    = nullptr;
    struct ggml_tensor*           final_norm_b    = nullptr;

    // upscaling
    struct ggml_tensor* up1_w        = nullptr;
    struct ggml_tensor* up1_b        = nullptr;
    struct ggml_tensor* up1_norm_w   = nullptr;
    struct ggml_tensor* up1_norm_b   = nullptr;
    struct ggml_tensor* up2_w        = nullptr;
    struct ggml_tensor* up2_b        = nullptr;

    // high-res feature convolutions
    struct ggml_tensor* conv_s0_w    = nullptr;
    struct ggml_tensor* conv_s0_b    = nullptr;
    struct ggml_tensor* conv_s1_w    = nullptr;
    struct ggml_tensor* conv_s1_b    = nullptr;

    // hypernetwork MLPs: 4 masks x 3 layers
    struct ggml_tensor* hyper_w[4][3]  = {};
    struct ggml_tensor* hyper_b[4][3]  = {};

    // IoU prediction head (3 layers)
    struct ggml_tensor* iou_head_w[3]  = {};
    struct ggml_tensor* iou_head_b[3]  = {};

    // object score head (3 layers)
    struct ggml_tensor* obj_head_w[3]  = {};
    struct ggml_tensor* obj_head_b[3]  = {};
};

/*
** ── Memory Encoder ───────────────────────────────────────────────────────
*/

struct sam3_mem_enc {
    // mask downsampler (4 conv stages + final 1x1)
    struct ggml_tensor* ds_conv_w[5]      = {};
    struct ggml_tensor* ds_conv_b[5]      = {};
    struct ggml_tensor* ds_norm_w[4]      = {};
    struct ggml_tensor* ds_norm_b[4]      = {};
    // pixel feature projection
    struct ggml_tensor* pix_proj_w        = nullptr;
    struct ggml_tensor* pix_proj_b        = nullptr;
    // fuser (2 CXBlock layers)
    struct ggml_tensor* fuser_dw_w[2]     = {};
    struct ggml_tensor* fuser_dw_b[2]     = {};
    struct ggml_tensor* fuser_norm_w[2]   = {};
    struct ggml_tensor* fuser_norm_b[2]   = {};
    struct ggml_tensor* fuser_fc1_w[2]    = {};
    struct ggml_tensor* fuser_fc1_b[2]    = {};
    struct ggml_tensor* fuser_fc2_w[2]    = {};
    struct ggml_tensor* fuser_fc2_b[2]    = {};
    struct ggml_tensor* fuser_gamma[2]    = {};
    // output projection
    struct ggml_tensor* out_proj_w        = nullptr;
    struct ggml_tensor* out_proj_b        = nullptr;
    // temporal pos encodings
    struct ggml_tensor* tpos[7]           = {};
};

/*
** ── Memory Attention (Tracker Transformer) ───────────────────────────────
*/

struct sam3_mem_attn_layer {
    // self-attention (RoPE, 1 head, 256-dim)
    struct ggml_tensor* sa_q_w    = nullptr;
    struct ggml_tensor* sa_q_b    = nullptr;
    struct ggml_tensor* sa_k_w    = nullptr;
    struct ggml_tensor* sa_k_b    = nullptr;
    struct ggml_tensor* sa_v_w    = nullptr;
    struct ggml_tensor* sa_v_b    = nullptr;
    struct ggml_tensor* sa_out_w  = nullptr;
    struct ggml_tensor* sa_out_b  = nullptr;
    struct ggml_tensor* norm1_w   = nullptr;
    struct ggml_tensor* norm1_b   = nullptr;
    // cross-attention (RoPE, kv_dim=64)
    struct ggml_tensor* ca_q_w    = nullptr;
    struct ggml_tensor* ca_q_b    = nullptr;
    struct ggml_tensor* ca_k_w    = nullptr;  // [256, 64]
    struct ggml_tensor* ca_k_b    = nullptr;
    struct ggml_tensor* ca_v_w    = nullptr;  // [256, 64]
    struct ggml_tensor* ca_v_b    = nullptr;
    struct ggml_tensor* ca_out_w  = nullptr;
    struct ggml_tensor* ca_out_b  = nullptr;
    struct ggml_tensor* norm2_w   = nullptr;
    struct ggml_tensor* norm2_b   = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w = nullptr;
    struct ggml_tensor* ffn_fc1_b = nullptr;
    struct ggml_tensor* ffn_fc2_w = nullptr;
    struct ggml_tensor* ffn_fc2_b = nullptr;
    struct ggml_tensor* norm3_w   = nullptr;
    struct ggml_tensor* norm3_b   = nullptr;
};

struct sam3_mem_attn {
    std::vector<sam3_mem_attn_layer> layers;
};

/*
** ── BPE Tokenizer ────────────────────────────────────────────────────────
*/

struct sam3_bpe_tokenizer {
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<int, std::string> decoder;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, int> merge_ranks;       // "a\x1fb" → rank
    std::unordered_map<uint8_t, std::string> byte_encoder;  // byte → unicode UTF-8
    std::unordered_map<std::string, std::string> cache;
    int sot_token = 49406;
    int eot_token = 49407;
};

/*****************************************************************************
** Top-Level Opaque Types (defined here, forward-declared in sam3.h)
*****************************************************************************/

struct sam3_model {
    sam3_hparams        hparams;
    ggml_type           weight_type = GGML_TYPE_F16;

    // SAM3_TRT_SKIP_GGML_WEIGHTS=1 (full-SAM3 models only): only the sam_pe.*
    // prompt-encoder tensors (~7KB -- the one weight set the TensorRT path
    // still reads on the CPU side, via sam3_populate_pe_cache) are registered
    // and loaded; the other ~1.1GB of weights are skipped entirely, never
    // allocated on the backend. Every ggml compute fallback for
    // encode/PCS/PVS checks this flag and hard-fails instead of running a
    // graph over absent weights. Only sane with SAM3_TRT_ENCODER built in
    // and all three engine env vars configured.
    bool trt_only_weights = false;

    // ── SAM3-specific (loaded only when model_type != SAM2) ──────────────
    sam3_vit            vit;
    sam3_neck           neck_det;
    sam3_neck           neck_trk;
    sam3_text_encoder   text_enc;
    sam3_fusion_encoder fenc;
    sam3_detr_decoder   ddec;
    sam3_geom_encoder   geom_enc;
    sam3_seg_head       seg_head;


    // ── SAM prompt encoder + mask decoder (PVS path) ─────────────────────
    sam3_sam_prompt_enc sam_pe;
    sam3_sam_mask_dec   sam_dec;



    // precomputed RoPE frequencies (SAM3 only)
    struct ggml_tensor* rope_freqs         = nullptr;  // [n_img_tokens, head_dim]

    // ggml backend
    struct ggml_context*    ctx     = nullptr;
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buffer  = nullptr;

    // tensor lookup
    std::map<std::string, struct ggml_tensor*> tensors;

    // tokenizer
    sam3_bpe_tokenizer tokenizer;
};

struct sam3_state {
    // cached backbone outputs
    struct ggml_tensor* vit_output       = nullptr;  // [1, embed, H, W]
    struct ggml_tensor* neck_det[4]      = {};       // FPN levels (det path)
    struct ggml_tensor* neck_trk[4]      = {};       // FPN levels (trk path)
    struct ggml_tensor* neck_det_pe[4]   = {};       // sinusoidal PE
    struct ggml_tensor* neck_trk_pe[4]   = {};

    int orig_width  = 0;
    int orig_height = 0;
    int n_threads   = 4;

    int encode_img_size  = 0;  // effective img_size for encoding (0 = hp.img_size)
    int encode_feat_size = 0;  // effective feat_size for the active backbone

    struct ggml_context*  ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
    struct ggml_gallocr*  galloc  = nullptr;

    // Holds vit_output/neck_det/neck_trk when populated by the TensorRT
    // encoder path (SAM3_TRT_ENCODER=1) instead of the ggml graph above --
    // a separate ggml_backend_alloc_ctx_tensors buffer (not gallocr-managed,
    // since there's no compute graph here, just persistent leaf tensors the
    // TRT runtime's output is copied into).
    struct ggml_context*  trt_out_ctx = nullptr;
    // Runtime encoder-precision selector (see sam3_set_encoder_fp8): when
    // true and SAM3_TRT_ONNX_PATH_FP8 is configured, image encodes use the
    // FP8 engine; both engines stay resident once used, so switching per
    // request costs nothing after first load.
    bool trt_encoder_fp8 = false;
    // Same selector for the PCS head (see sam3_set_pcs_fp8): when true and a
    // PCS FP8 engine is configured, sam3_segment_pcs uses it; falls back to
    // the standard PCS engine with a log line otherwise.
    bool trt_pcs_fp8 = false;
    ggml_backend_buffer_t trt_out_buf = nullptr;

    // PE buffer: holds sinusoidal PE tensors for neck outputs. sam3_sinusoidal_pe_2d
    // is a pure function of (scale sizes, neck_dim) alone, both derived from
    // fixed hyperparameters -- valid as long as the effective feature size the
    // cache was built for hasn't changed (never happens in a fixed-image-size
    // deployment, but tracked properly rather than assumed).
    struct ggml_context*  pe_ctx  = nullptr;
    ggml_backend_buffer_t pe_buf  = nullptr;
    bool neck_pe_valid = false;
    int  neck_pe_feat_size = 0;  // n_img_embd() the cached PE tensors were built for

    // Cached SAM prompt encoder embeddings (read from GPU once, reused).
    // sam3_populate_pe_cache is a pure function of (sam_embed_dim,
    // sam3_eff_feat_size) plus constant model weights -- pe_cache_feat_size
    // tracks what feat_size the cache was built for so callers only
    // invalidate it when that actually changes, not unconditionally.
    bool pe_cache_valid = false;
    int  pe_cache_feat_size = 0;
    std::vector<float> pe_gauss_cache;      // [2 * num_pos_feats]
    float point_emb_cache[4][256]   = {};
    float not_a_point_cache[256]    = {};
    float no_mask_emb_cache[256]    = {};
    std::vector<float> dense_pe_cache;      // [D * H * H] -- PE grid
    std::vector<float> dense_nomask_cache;  // [D * H * H] -- no-mask tiled
};


static inline int sam3_eff_img_size(const sam3_state& s, const sam3_hparams& hp) {
    return (s.encode_img_size > 0) ? s.encode_img_size : hp.img_size;
}
static inline int sam3_eff_feat_size(const sam3_state& s, const sam3_hparams& hp) {
    return (s.encode_feat_size > 0) ? s.encode_feat_size : hp.feat_size();
}



/*****************************************************************************
** Internal graph-result types (shared by graph builders and their callers)
*****************************************************************************/

// Geometry encoder output: geometry features as a pre-computed input tensor
// [D, N_geo, 1] where N_geo = n_exemplar_boxes + 1 (CLS).
struct sam3_geom_result {
    struct ggml_tensor* geo_feats;  // [D, N_geo, 1]
    int n_tokens;                   // N_geo
};

// DETR decoder outputs.
struct sam3_ddec_output {
    struct ggml_tensor* queries;         // [D, 201, B]
    struct ggml_tensor* presence_feats;  // [D, 1, B] pre-decoder-norm presence token
    struct ggml_tensor* pred_boxes;      // [4, 200, B]
    struct ggml_tensor* class_scores;    // [200, B]
    struct ggml_tensor* presence_score;  // [1, B]
};

// SAM prompt-encoder outputs.
struct sam3_pe_result {
    struct ggml_tensor* sparse;    // [D, N_pts, 1]
    struct ggml_tensor* dense;     // [D, H, H, 1]
    struct ggml_tensor* image_pe;  // [D, H, H, 1] -- dense positional encoding grid
    int n_tokens;
};

// SAM mask-decoder outputs.
struct sam3_dec_result {
    struct ggml_tensor* masks;        // [288*288, N_masks, 1]
    struct ggml_tensor* iou_pred;     // [N_masks, 1]
    struct ggml_tensor* obj_score;    // [1, 1]
    struct ggml_tensor* sam_token;    // [D, 1] -- for object pointer
    struct ggml_tensor* mask_tokens;  // [D, N_masks, 1] -- raw SAM mask tokens
};

/*****************************************************************************
** Cross-module internal declarations
*****************************************************************************/

// ── ggml/backend.cpp: backend init + CUDA graph-lowering helpers ──────────
// Set to the model backend when a GPU (non-Metal) model is active, so the
// graph-lowering wrappers below know to lower unsupported ops. One GPU
// model per process; CPU models are unaffected.
extern ggml_backend_t g_gpu_backend;

bool sam3_graph_compute(ggml_backend_t backend, struct ggml_cgraph* graph, int n_threads);
struct ggml_tensor* sam3_win_part(struct ggml_context* ctx, struct ggml_tensor* a, int w);
struct ggml_tensor* sam3_win_unpart(struct ggml_context* ctx, struct ggml_tensor* a,
                                    int w0, int h0, int w);
struct ggml_tensor* sam3_conv_transpose_2d_p0(struct ggml_context* ctx,
                                              struct ggml_tensor* weight,
                                              struct ggml_tensor* input,
                                              int stride);
struct ggml_tensor* sam3_flash_attn_ext(struct ggml_context* ctx,
                                        struct ggml_tensor* q,
                                        struct ggml_tensor* k,
                                        struct ggml_tensor* v,
                                        struct ggml_tensor* mask,
                                        float scale, float max_bias, float logit_softcap);
void sam3_debug_dump_vec(const char* name, const float* p, size_t n);
void sam3_name_tensorf(struct ggml_tensor* t, const char* fmt, int index);
struct ggml_tensor* sam3_layer_norm(struct ggml_context* ctx, struct ggml_tensor* x,
                                    struct ggml_tensor* w, struct ggml_tensor* b);
struct ggml_tensor* sam3_layer_norm_2d(struct ggml_context* ctx, struct ggml_tensor* x,
                                       struct ggml_tensor* w, struct ggml_tensor* b);
void sam3_read_f32(struct ggml_tensor* t, float* dst, int64_t n);
struct ggml_tensor* sam3_conv_transpose_weight(struct ggml_context* ctx, struct ggml_tensor* w);

// ── tokenizer.cpp ──────────────────────────────────────────────────────────
bool sam3_load_bpe_vocab_from_stream(std::ifstream& fin, sam3_bpe_tokenizer& tok);
std::vector<int32_t> sam3_tokenize(sam3_bpe_tokenizer& tok, const std::string& text, int ctx_len);

// ── model_load.cpp ─────────────────────────────────────────────────────────
bool sam3_load_hparams(std::ifstream& fin, sam3_hparams& hp);

// ── preprocess.cpp ─────────────────────────────────────────────────────────
std::vector<float> sam3_preprocess_image(const sam3_image& image, int img_size);

// ── ggml/backbone.cpp: ViT + SimpleFPN neck graph builders ────────────────
void sam3_compute_axial_cis(float* out, int dim, int end_x, int end_y,
                            float theta, float scale_pos);
std::vector<float> sam3_sinusoidal_pe_2d(int H, int W, int d_model);
struct ggml_tensor* sam3_apply_rope(struct ggml_context* ctx, struct ggml_tensor* x,
                                    struct ggml_tensor* freqs_cis);
struct ggml_tensor* sam3_vit_block_forward(struct ggml_context* ctx, struct ggml_tensor* x,
                                           const sam3_vit_block& blk, const sam3_hparams& hp,
                                           int block_idx);
struct ggml_tensor* sam3_build_vit_prefix_graph(struct ggml_context* ctx,
                                                struct ggml_tensor* input,
                                                const sam3_model& model);
struct ggml_tensor* sam3_build_vit_graph(struct ggml_context* ctx, struct ggml_tensor* input,
                                         const sam3_model& model);
void sam3_build_neck_graph(struct ggml_context* ctx, struct ggml_tensor* vit_out,
                           const sam3_neck& neck, struct ggml_tensor* out[4]);

// ── ggml/pcs.cpp: text/geometry/fusion/DETR/seg-head graph builders ───────
void sam3_fill_causal_mask(ggml_fp16_t* mask_data, int L);
struct ggml_tensor* sam3_build_text_encoder_graph(struct ggml_context* ctx,
                                                  struct ggml_tensor* token_ids,
                                                  const sam3_model& model);
sam3_geom_result sam3_build_geom_enc_graph(struct ggml_context* ctx,
                                           const sam3_model& model,
                                           const sam3_pcs_params& params,
                                           struct ggml_tensor* img_feats,
                                           struct ggml_tensor* img_pe);
std::vector<float> sam3_precompute_geom_input(const sam3_model& model,
                                              const sam3_pcs_params& params,
                                              const float* img_feats_data,
                                              int W_feat, int H_feat);
struct ggml_tensor* sam3_build_fenc_graph(struct ggml_context* ctx,
                                          const sam3_model& model,
                                          struct ggml_tensor* image_feats,
                                          struct ggml_tensor* prompt_tokens,
                                          struct ggml_tensor* pos_enc,
                                          struct ggml_tensor* prompt_attn_bias = nullptr);
sam3_ddec_output sam3_build_ddec_graph(struct ggml_context* ctx,
                                       const sam3_model& model,
                                       struct ggml_tensor* enc_feats,
                                       struct ggml_tensor* enc_pos,
                                       struct ggml_tensor* text_feats,
                                       struct ggml_tensor* sine_dim_t,
                                       struct ggml_tensor* rpb_coords,
                                       struct ggml_tensor* text_attn_bias = nullptr,
                                       struct ggml_tensor* text_valid_mask = nullptr);
struct ggml_tensor* sam3_build_seg_head_graph(struct ggml_context* ctx,
                                              const sam3_model& model,
                                              struct ggml_tensor* enc_hidden,
                                              struct ggml_tensor* fpn_feats[3],
                                              struct ggml_tensor* query_outputs,
                                              struct ggml_tensor* text_features,
                                              struct ggml_tensor* text_attn_bias = nullptr);

// ── ggml/pvs.cpp: SAM prompt encoder + mask decoder graph builders ────────
void sam3_pe_encode_coord(float* out, float x_norm, float y_norm,
                          const float* pe_gauss, int num_pos_feats);
void sam3_populate_pe_cache(sam3_state& state, const sam3_model& model);
void sam3_collect_pvs_prompt_tokens(const sam3_pvs_params& params,
                                    std::vector<float>& all_coords,
                                    std::vector<int>& all_labels);
sam3_pe_result sam3_build_sam_pe(struct ggml_context* ctx, const sam3_pvs_params& params,
                                 int embed_dim, int feat_size);
sam3_dec_result sam3_build_sam_dec_graph(struct ggml_context* ctx,
                                         const sam3_model& model,
                                         struct ggml_tensor* image_feats,
                                         struct ggml_tensor* image_pe,
                                         struct ggml_tensor* sparse_emb,
                                         struct ggml_tensor* dense_emb,
                                         struct ggml_tensor* feat_s0,
                                         struct ggml_tensor* feat_s1,
                                         int eff_feat_size = 0);

// ── mask_utils.cpp ─────────────────────────────────────────────────────────
std::vector<int> sam3_nms(const std::vector<sam3_detection>& dets, float iou_thresh);
std::vector<float> sam3_bilinear_interpolate(const float* src, int src_w, int src_h,
                                             int dst_w, int dst_h);
sam3_box sam3_cxcywh_to_xyxy(float cx, float cy, float w, float h, int img_w, int img_h);

// ── api.cpp ────────────────────────────────────────────────────────────────
void sam3_clear_encoder_state(sam3_state & state);
bool sam3_mark_named_outputs(struct ggml_context * ctx,
                             const std::vector<std::string> & output_tensors);

// ── debug.cpp ──────────────────────────────────────────────────────────────
bool sam3_copy_tensor_to_f32(struct ggml_tensor * t, std::vector<float> & output);

// ── trt/ (compiled only with SAM3_TRT_ENCODER; see src/trt/) ──────────────
#ifdef SAM3_TRT_ENCODER
struct sam3_trt_engine;
// Programmatic config (sam3_params::trt): captured at load time; fields are
// consulted only when cfg.enabled, env vars remain the fallback.
void sam3_trt_set_config(const sam3_trt_config& cfg);
bool sam3_trt_enabled();
std::string sam3_trt_cfg_value(const char* env_name);
sam3_trt_engine* sam3_get_trt_engine_cached(const char* onnx_env_name,
                                            const char* cache_env_name,
                                            bool allow_fp16 = true,
                                            const std::vector<std::string>& fp32_name_substrings = {});
bool sam3_try_trt_encode_image(sam3_state& state, const sam3_model& model,
                               const float* chw_data, int img_size);
bool sam3_try_trt_segment_pcs(sam3_state& state, const sam3_model& model,
                              const sam3_pcs_params& params,
                              const std::vector<int32_t>& token_ids,
                              sam3_result& out_result);
bool sam3_try_trt_segment_pvs(sam3_state& state, const sam3_model& model,
                              const sam3_pvs_params& params,
                              sam3_result& out_result);
#endif  // SAM3_TRT_ENCODER
