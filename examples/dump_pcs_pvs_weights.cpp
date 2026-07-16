// sam3_dump_pcs_pvs_weights — export the text/geometry/fusion encoders, DETR
// decoder+scoring, MaskFormer seg head (PCS), SAM prompt encoder, SAM
// two-way mask decoder (PVS), and the shared no_mem_embed tensor to raw f32
// blobs + a manifest.json, for the offline TensorRT ONNX-authoring step
// (convert_sam3_pcs_to_onnx.py / convert_sam3_pvs_to_onnx.py).
//
// Sibling to dump_encoder_weights.cpp (image encoder). One-time dev-machine
// tool, part of the TensorRT migration (see docs/sam3/PLAN.md) -- not part
// of the deployed server.
//
// Usage: sam3_dump_pcs_pvs_weights --model m.ggml --out pcs_pvs_export/
#include "sam3.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <sys/stat.h>

int main(int argc, char** argv) {
    sam3_params params;
    params.use_gpu = false;  // pure weight export -- no inference, CPU load is enough
    std::string out_dir = "pcs_pvs_export";

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--model") == 0 && i + 1 < argc) { params.model_path = argv[++i]; }
        else if (strcmp(argv[i], "--out")   == 0 && i + 1 < argc) { out_dir = argv[++i]; }
        else if (strcmp(argv[i], "--gpu")   == 0)                  { params.use_gpu = true; }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (params.model_path.empty()) {
        fprintf(stderr, "usage: %s --model m.ggml [--out pcs_pvs_export/] [--gpu]\n", argv[0]);
        return 1;
    }

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const sam3_model_type mtype = sam3_get_model_type(*model);
    if (mtype != SAM3_MODEL_SAM3) {
        fprintf(stderr, "error: --model is not a full SAM3 model (visual-only/SAM2/EdgeTAM lack the "
                         "text/detector path this tool exports)\n");
        return 1;
    }

    mkdir(out_dir.c_str(), 0755);
    std::ofstream manifest(out_dir + "/manifest.json");
    if (!manifest) { fprintf(stderr, "failed to open manifest.json for writing\n"); return 1; }
    manifest << "{\n\"tensors\": [\n";
    bool first = true;
    int n_dumped = 0, n_missing = 0;

    // Same manifest-entry format as dump_encoder_weights.cpp (ggml_ne always
    // the full, never-squeezed 4-tuple -- see that file's comment on why).
    auto dump_one = [&](const std::string& name, bool required) -> bool {
        sam3_tensor_info info;
        if (!sam3_get_model_tensor_info(*model, name, info)) {
            if (required) {
                fprintf(stderr, "error: required tensor '%s' missing from model\n", name.c_str());
                exit(1);
            }
            n_missing++;
            return false;
        }
        if (!sam3_dump_model_tensor(*model, name, out_dir + "/" + name)) {
            fprintf(stderr, "error: failed to dump '%s'\n", name.c_str());
            exit(1);
        }
        if (!first) manifest << ",\n";
        first = false;
        manifest << "  {\"name\": \"" << name << "\", \"file\": \"" << name << ".bin\""
                 << ", \"dtype\": \"f32\", \"ggml_ne\": ["
                 << info.ne[0] << ", " << info.ne[1] << ", " << info.ne[2] << ", " << info.ne[3]
                 << "]}";
        n_dumped++;
        return true;
    };
    auto req = [&](const std::string& name) { dump_one(name, true); };
    auto opt = [&](const std::string& name) { dump_one(name, false); };  // dead/unused tensors: dump if present, skip quietly if not

    // ── Text encoder ─────────────────────────────────────────────────────
    req("text.token_embed.weight");
    req("text.pos_embed");
    req("text.ln_final.weight");
    req("text.ln_final.bias");
    req("text.resizer.weight");
    req("text.resizer.bias");
    for (int i = 0; i < 24; ++i) {
        const std::string p = "text.blocks." + std::to_string(i) + ".";
        req(p + "attn.in_proj.weight");
        req(p + "attn.in_proj.bias");
        req(p + "attn.out_proj.weight");
        req(p + "attn.out_proj.bias");
        req(p + "ln_1.weight");
        req(p + "ln_1.bias");
        req(p + "ln_2.weight");
        req(p + "ln_2.bias");
        req(p + "mlp.fc1.weight");
        req(p + "mlp.fc1.bias");
        req(p + "mlp.fc2.weight");
        req(p + "mlp.fc2.bias");
    }

    // ── Geometry encoder (only the graph-side transformer; the CPU-side
    //    ROI-align/pooling precompute is untouched, not needed here except
    //    cls_embed/final_proj/norm which ARE consumed by that CPU code too,
    //    dumped for the Python-side "no exemplar boxes" constant-fold) ────
    req("geom.label_embed.weight");
    req("geom.cls_embed.weight");
    req("geom.final_proj.weight");
    req("geom.final_proj.bias");
    req("geom.norm.weight");
    req("geom.norm.bias");
    req("geom.encode_norm.weight");
    req("geom.encode_norm.bias");
    req("geom.img_pre_norm.weight");
    req("geom.img_pre_norm.bias");
    for (int i = 0; i < 3; ++i) {
        const std::string p = "geom.layers." + std::to_string(i) + ".";
        req(p + "sa.in_proj_weight");
        req(p + "sa.in_proj_bias");
        req(p + "sa.out_proj.weight");
        req(p + "sa.out_proj.bias");
        req(p + "norm1.weight");
        req(p + "norm1.bias");
        req(p + "ca.in_proj_weight");
        req(p + "ca.in_proj_bias");
        req(p + "ca.out_proj.weight");
        req(p + "ca.out_proj.bias");
        req(p + "norm2.weight");
        req(p + "norm2.bias");
        req(p + "linear1.weight");
        req(p + "linear1.bias");
        req(p + "linear2.weight");
        req(p + "linear2.bias");
        req(p + "norm3.weight");
        req(p + "norm3.bias");
    }
    // Dead-for-this-scope (no exemplar boxes) but dumped anyway, cheap:
    opt("geom.boxes_pool_project.weight");
    opt("geom.boxes_pool_project.bias");
    opt("geom.boxes_pos_enc_project.weight");
    opt("geom.boxes_pos_enc_project.bias");
    opt("geom.boxes_direct_project.weight");
    opt("geom.boxes_direct_project.bias");

    // ── Fusion encoder ───────────────────────────────────────────────────
    for (int i = 0; i < 6; ++i) {
        const std::string p = "fenc.layers." + std::to_string(i) + ".";
        req(p + "sa.in_proj_weight");
        req(p + "sa.in_proj_bias");
        req(p + "sa.out_proj.weight");
        req(p + "sa.out_proj.bias");
        req(p + "norm1.weight");
        req(p + "norm1.bias");
        req(p + "ca.in_proj_weight");
        req(p + "ca.in_proj_bias");
        req(p + "ca.out_proj.weight");
        req(p + "ca.out_proj.bias");
        req(p + "norm2.weight");
        req(p + "norm2.bias");
        req(p + "linear1.weight");
        req(p + "linear1.bias");
        req(p + "linear2.weight");
        req(p + "linear2.bias");
        req(p + "norm3.weight");
        req(p + "norm3.bias");
    }

    // ── DETR decoder + scoring ───────────────────────────────────────────
    req("ddec.query_embed.weight");
    req("ddec.presence_token.weight");
    req("ddec.reference_points.weight");
    req("ddec.norm.weight");
    req("ddec.norm.bias");
    for (int j = 0; j < 3; ++j) {
        req("ddec.bbox_embed.layers." + std::to_string(j) + ".weight");
        req("ddec.bbox_embed.layers." + std::to_string(j) + ".bias");
    }
    for (int j = 0; j < 2; ++j) {
        req("ddec.ref_point_head.layers." + std::to_string(j) + ".weight");
        req("ddec.ref_point_head.layers." + std::to_string(j) + ".bias");
    }
    for (const char* axis : {"x", "y"}) {
        for (int j = 0; j < 2; ++j) {
            req(std::string("ddec.boxRPB_embed_") + axis + ".layers." + std::to_string(j) + ".weight");
            req(std::string("ddec.boxRPB_embed_") + axis + ".layers." + std::to_string(j) + ".bias");
        }
    }
    for (int j = 0; j < 3; ++j) {
        req("ddec.presence_token_head.layers." + std::to_string(j) + ".weight");
        req("ddec.presence_token_head.layers." + std::to_string(j) + ".bias");
    }
    req("ddec.presence_token_out_norm.weight");
    req("ddec.presence_token_out_norm.bias");
    for (int i = 0; i < 6; ++i) {
        const std::string p = "ddec.layers." + std::to_string(i) + ".";
        req(p + "sa.in_proj_weight");
        req(p + "sa.in_proj_bias");
        req(p + "sa.out_proj.weight");
        req(p + "sa.out_proj.bias");
        req(p + "norm1.weight");   // post image-CA (step 3) -- see convert script comment
        req(p + "norm1.bias");
        req(p + "ca.in_proj_weight");
        req(p + "ca.in_proj_bias");
        req(p + "ca.out_proj.weight");
        req(p + "ca.out_proj.bias");
        req(p + "norm2.weight");   // post self-attn (step 1)
        req(p + "norm2.bias");
        req(p + "ca_text.in_proj_weight");
        req(p + "ca_text.in_proj_bias");
        req(p + "ca_text.out_proj.weight");
        req(p + "ca_text.out_proj.bias");
        req(p + "norm_ca_text.weight");  // post prompt-CA (step 2)
        req(p + "norm_ca_text.bias");
        req(p + "linear1.weight");
        req(p + "linear1.bias");
        req(p + "linear2.weight");
        req(p + "linear2.bias");
        req(p + "norm3.weight");   // post-FFN (step 4)
        req(p + "norm3.bias");
    }
    req("scoring.prompt_proj.weight");
    req("scoring.prompt_proj.bias");
    req("scoring.hs_proj.weight");
    req("scoring.hs_proj.bias");
    req("scoring.prompt_mlp.layers.0.weight");
    req("scoring.prompt_mlp.layers.0.bias");
    req("scoring.prompt_mlp.layers.1.weight");
    req("scoring.prompt_mlp.layers.1.bias");
    req("scoring.prompt_mlp.out_norm.weight");
    req("scoring.prompt_mlp.out_norm.bias");

    // ── MaskFormer segmentation head ─────────────────────────────────────
    for (int i = 0; i < 2; ++i) {  // conv_layers.2/norms.2 confirmed dead, skip
        req("seg.pixel_decoder.conv_layers." + std::to_string(i) + ".weight");
        req("seg.pixel_decoder.conv_layers." + std::to_string(i) + ".bias");
        req("seg.pixel_decoder.norms." + std::to_string(i) + ".weight");
        req("seg.pixel_decoder.norms." + std::to_string(i) + ".bias");
    }
    for (int j = 0; j < 3; ++j) {
        req("seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".weight");
        req("seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".bias");
    }
    req("seg.cross_attend_prompt.in_proj_weight");
    req("seg.cross_attend_prompt.in_proj_bias");
    req("seg.cross_attend_prompt.out_proj.weight");
    req("seg.cross_attend_prompt.out_proj.bias");
    req("seg.cross_attn_norm.weight");
    req("seg.cross_attn_norm.bias");
    req("seg.instance_seg_head.weight");
    req("seg.instance_seg_head.bias");

    // ── SAM prompt encoder (PVS) ─────────────────────────────────────────
    req("sam_pe.pe_gaussian");
    for (int i = 0; i < 4; ++i) {
        req("sam_pe.point_embeddings." + std::to_string(i) + ".weight");
    }
    req("sam_pe.not_a_point_embed.weight");
    req("sam_pe.no_mask_embed.weight");
    // sam_pe.mask_ds.* confirmed dead (no mask-prompt support) -- not dumped.

    // ── SAM two-way mask decoder (PVS) ───────────────────────────────────
    req("sam_dec.iou_token.weight");
    req("sam_dec.mask_tokens.weight");
    req("sam_dec.obj_score_token.weight");
    for (int i = 0; i < 2; ++i) {
        const std::string p = "sam_dec.twoway." + std::to_string(i) + ".";
        for (const char* qkv : {"q", "k", "v"}) {
            req(p + "sa." + qkv + "_proj.weight");
            req(p + "sa." + qkv + "_proj.bias");
        }
        req(p + "sa.out_proj.weight");
        req(p + "sa.out_proj.bias");
        for (const char* attn : {"cross_attn_token_to_image", "cross_attn_image_to_token"}) {
            for (const char* qkv : {"q", "k", "v"}) {
                req(p + attn + "." + qkv + "_proj.weight");
                req(p + attn + "." + qkv + "_proj.bias");
            }
            req(p + attn + ".out_proj.weight");
            req(p + attn + ".out_proj.bias");
        }
        for (int n = 1; n <= 4; ++n) {
            req(p + "norm" + std::to_string(n) + ".weight");
            req(p + "norm" + std::to_string(n) + ".bias");
        }
        req(p + "mlp.lin1.weight");
        req(p + "mlp.lin1.bias");
        req(p + "mlp.lin2.weight");
        req(p + "mlp.lin2.bias");
    }
    for (const char* qkv : {"q", "k", "v"}) {
        req(std::string("sam_dec.final_attn.") + qkv + "_proj.weight");
        req(std::string("sam_dec.final_attn.") + qkv + "_proj.bias");
    }
    req("sam_dec.final_attn.out_proj.weight");
    req("sam_dec.final_attn.out_proj.bias");
    req("sam_dec.final_norm.weight");
    req("sam_dec.final_norm.bias");
    req("sam_dec.upscale.0.weight");
    req("sam_dec.upscale.0.bias");
    req("sam_dec.upscale.1.weight");
    req("sam_dec.upscale.1.bias");
    req("sam_dec.upscale.3.weight");
    req("sam_dec.upscale.3.bias");
    req("sam_dec.conv_s0.weight");
    req("sam_dec.conv_s0.bias");
    req("sam_dec.conv_s1.weight");
    req("sam_dec.conv_s1.bias");
    for (int m = 0; m < 4; ++m) {
        for (int j = 0; j < 3; ++j) {
            req("sam_dec.hyper." + std::to_string(m) + ".layers." + std::to_string(j) + ".weight");
            req("sam_dec.hyper." + std::to_string(m) + ".layers." + std::to_string(j) + ".bias");
        }
    }
    for (int j = 0; j < 3; ++j) {
        req("sam_dec.iou_prediction_head.layers." + std::to_string(j) + ".weight");
        req("sam_dec.iou_prediction_head.layers." + std::to_string(j) + ".bias");
        req("sam_dec.pred_obj_score_head.layers." + std::to_string(j) + ".weight");
        req("sam_dec.pred_obj_score_head.layers." + std::to_string(j) + ".bias");
    }

    // ── Shared ───────────────────────────────────────────────────────────
    req("no_mem_embed");

    manifest << "\n]}\n";
    manifest.close();

    fprintf(stderr, "%s: dumped %d tensors (%d optional tensors absent) to '%s' (manifest.json)\n",
            __func__, n_dumped, n_missing, out_dir.c_str());
    return 0;
}
